"""How the CloudWatch sink behaves under partial IAM permissions. Stubbed client, no AWS.

A deployment role is often scoped to PutLogEvents on one existing group, without
CreateLogGroup or PutRetentionPolicy. Losing those optional calls must not stop the turn
from being recorded - that failure mode is invisible in production, because the sink is
deliberately silent.
"""
import pytest

import rag_logging


class ResourceAlreadyExists(Exception):
    pass


class AccessDenied(Exception):
    def __str__(self):
        return "User is not authorized to perform this operation"


class FakeExceptions:
    ResourceAlreadyExistsException = ResourceAlreadyExists


class FakeLogsClient:
    """Records calls; raises whatever the test asks it to for a given operation."""

    exceptions = FakeExceptions()

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = []
        self.put_events = []

    def _maybe_fail(self, op):
        self.calls.append(op)
        if op in self.fail_on:
            raise AccessDenied()

    def create_log_group(self, **kw):
        self._maybe_fail("create_log_group")

    def put_retention_policy(self, **kw):
        self._maybe_fail("put_retention_policy")

    def create_log_stream(self, **kw):
        self._maybe_fail("create_log_stream")

    def put_log_events(self, **kw):
        self._maybe_fail("put_log_events")
        self.put_events.append(kw)


@pytest.fixture
def sink_factory(monkeypatch):
    """Build a sink whose boto3 client is a FakeLogsClient, and deliver records synchronously."""
    monkeypatch.setattr(rag_logging, "LOG_GROUP", "/test/group")
    monkeypatch.setattr(rag_logging, "RETENTION_DAYS", 30)

    def build(fail_on=(), group_exists=False):
        client = FakeLogsClient(fail_on=set(fail_on) | ({"create_log_group"} if group_exists else set()))
        if group_exists:
            def create_log_group(**kw):
                client.calls.append("create_log_group")
                raise ResourceAlreadyExists()
            client.create_log_group = create_log_group

        monkeypatch.setattr(rag_logging.boto3, "Session",
                            lambda **kw: type("S", (), {"client": lambda self, name: client})())
        sink = rag_logging._CloudWatchSink()
        sink.submit = lambda message: sink._put(message)   # run inline, no worker thread
        return sink, client

    return build


def test_happy_path_creates_group_stream_and_retention(sink_factory):
    sink, client = sink_factory()
    sink.submit('{"event": "qa"}')

    assert client.calls == ["create_log_group", "put_retention_policy",
                            "create_log_stream", "put_log_events"]
    assert len(client.put_events) == 1
    assert sink._broken is False


def test_existing_group_skips_retention(sink_factory):
    """Retention is only applied at creation, so an existing group must not be re-stamped."""
    sink, client = sink_factory(group_exists=True)
    sink.submit('{"event": "qa"}')

    assert "put_retention_policy" not in client.calls
    assert len(client.put_events) == 1, "an existing group must still receive events"


def test_denied_create_log_group_still_logs(sink_factory, capsys):
    """The deployment case: no CreateLogGroup, but the group exists and writes are allowed."""
    sink, client = sink_factory(fail_on=["create_log_group"])
    sink.submit('{"event": "qa"}')

    assert len(client.put_events) == 1, "logging must survive a denied CreateLogGroup"
    assert sink._broken is False
    err = capsys.readouterr().err
    assert "create_log_group failed" in err and "continuing" in err


def test_denied_retention_still_logs(sink_factory):
    sink, client = sink_factory(fail_on=["put_retention_policy"])
    sink.submit('{"event": "qa"}')

    assert len(client.put_events) == 1, "logging must survive a denied PutRetentionPolicy"
    assert sink._broken is False


def test_denied_create_stream_disables_the_sink(sink_factory, capsys):
    """Without a stream there is nowhere to write, so this one is genuinely fatal."""
    sink, client = sink_factory(fail_on=["create_log_stream"])
    sink.submit('{"event": "qa"}')

    assert client.put_events == []
    assert sink._broken is True
    err = capsys.readouterr().err
    assert "[rag_logging] disabled:" in err
    assert rag_logging.AWS_REGION in err, "the warning must name the region for diagnosis"
    assert "/test/group" in err


def test_denied_put_events_warns_once(sink_factory, capsys):
    sink, client = sink_factory(fail_on=["put_log_events"])
    for _ in range(3):
        sink.submit('{"event": "qa"}')

    err = capsys.readouterr().err
    assert err.count("[rag_logging] disabled:") == 1, f"warning was repeated:\n{err}"
    assert sink._broken is True


def test_broken_sink_stops_calling_aws(sink_factory):
    sink, client = sink_factory(fail_on=["create_log_stream"])
    for _ in range(5):
        sink.submit('{"event": "qa"}')

    assert client.calls.count("create_log_stream") == 1, \
        "a broken sink must not retry on every turn"
