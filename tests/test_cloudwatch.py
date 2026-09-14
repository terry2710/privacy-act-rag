"""The CloudWatch sink itself: real group/stream creation, delivery, and graceful degradation."""
import json
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.aws


@pytest.fixture(scope="module")
def delivered(aws, cw_log_group):
    """Emit records through the real sink and read them back from CloudWatch."""
    from conftest import reload_logging

    records = [
        {"event": "qa", "schema": "qa/2", "request_id": f"pytest{i}", "session_id": "pytest-cw",
         "question": f"probe {i}", "chunks": [{"rank": 1, "score": 0.5, "chunk_id": "abc123"}]}
        for i in range(3)
    ]
    with reload_logging(PRV_LOG_GROUP=cw_log_group, PRV_LOG_RETENTION_DAYS="1",
                        PRV_QA_LOG_FILE=None) as mod:
        assert mod._sink is not None, "sink should be enabled when PRV_LOG_GROUP is set"
        for r in records:
            mod.emit(r)
        # The stream is created lazily on the worker thread, so it only exists once the
        # executor has drained - read it after shutdown(), not before.
        mod._sink.shutdown()
        stream = mod._sink._stream or ""
        broken = mod._sink._broken

    client = aws.client("logs")
    events = []
    for _ in range(15):
        time.sleep(2)
        events = client.get_log_events(logGroupName=cw_log_group, logStreamName=stream,
                                       startFromHead=True)["events"]
        if len(events) >= len(records):
            break
    return stream, broken, [json.loads(e["message"]) for e in events]


def test_sink_opened_a_stream(delivered):
    stream, broken, _ = delivered
    assert broken is False, "the sink reported itself broken"
    assert stream, "no log stream was created"


def test_stream_name_is_date_prefixed(delivered):
    stream, _, _ = delivered
    date, _, suffix = stream.rpartition("/")
    assert len(date.split("/")) == 3, f"expected YYYY/MM/DD prefix, got {stream}"
    assert len(suffix) == 12


def test_all_records_arrived_intact(delivered):
    _, _, events = delivered
    assert len(events) == 3
    assert [e["request_id"] for e in events] == ["pytest0", "pytest1", "pytest2"], \
        "single-worker executor should preserve ordering"
    assert all(e["session_id"] == "pytest-cw" for e in events)


def test_events_are_valid_json_with_a_timestamp(delivered):
    _, _, events = delivered
    for e in events:
        assert e["ts"].endswith("Z")
        assert e["chunks"][0]["chunk_id"] == "abc123"


def test_retention_policy_is_applied(aws, cw_log_group, delivered):
    groups = aws.client("logs").describe_log_groups(logGroupNamePrefix=cw_log_group)["logGroups"]
    assert groups[0].get("retentionInDays") == 1


def test_broken_credentials_degrade_silently(cw_log_group):
    """A missing IAM permission must warn once on stderr and never raise.

    Runs in a subprocess so the bogus credentials cannot leak into the rest of the session.
    """
    script = (
        "import sys, rag_logging\n"
        "for i in range(3): rag_logging.emit({'event': 'qa', 'request_id': str(i)})\n"
        "rag_logging._sink.shutdown()\n"
        "print('BROKEN', rag_logging._sink._broken)\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PRV_LOG_GROUP": cw_log_group,
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "AKIABOGUSBOGUSBOGUS",
        "AWS_SECRET_ACCESS_KEY": "bogusbogusbogusbogusbogusbogusbogusbogus",
    }
    from conftest import PROJECT_ROOT
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
                            capture_output=True, text=True, timeout=180)

    assert result.returncode == 0, f"emitting with bad credentials raised:\n{result.stderr}"
    assert "BROKEN True" in result.stdout, "the sink should mark itself broken"
    assert result.stderr.count("[rag_logging] disabled:") == 1, \
        f"expected exactly one warning, got:\n{result.stderr}"
