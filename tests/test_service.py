import pytest

from api.service import IndexNotReady, RagService
from core import rag_backend


def test_warm_up_loads_index_once():
    calls = []

    def loader():
        calls.append(1)
        return "INDEX"

    service = RagService(index_loader=loader)
    assert service.is_ready() is False
    assert service.warm_up() is True
    assert service.warm_up() is True
    assert service.is_ready() is True
    assert calls == [1]


def test_warm_up_without_cached_index_is_not_ready():
    service = RagService(index_loader=lambda: None)
    assert service.warm_up() is False
    assert service.is_ready() is False


def test_warm_up_failure_is_not_ready():
    def loader():
        raise RuntimeError("s3 unavailable")

    service = RagService(index_loader=loader)
    assert service.warm_up() is False
    assert service.is_ready() is False


def test_non_blocking_warm_up_gives_up_while_another_load_runs():
    service = RagService(index_loader=lambda: "INDEX")
    service._lock.acquire()
    try:
        assert service.warm_up(wait=False) is False
    finally:
        service._lock.release()


def test_answer_without_index_raises_and_never_calls_backend(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("prv_rag_response must not run without an index")

    monkeypatch.setattr(rag_backend, "prv_rag_response", unexpected)
    service = RagService(index_loader=lambda: None)
    with pytest.raises(IndexNotReady):
        service.answer("What is APP 1?")


def test_answer_retries_loading_on_demand(monkeypatch):
    loads = iter([None, "INDEX"])
    service = RagService(index_loader=lambda: next(loads))
    assert service.warm_up() is False

    def fake_response(index, question, session_id=None, return_details=False):
        assert index == "INDEX"
        return "answer", {"request_id": "req1", "chunks": [], "pipeline_version": "v1"}

    monkeypatch.setattr(rag_backend, "prv_rag_response", fake_response)
    result = service.answer("What is APP 1?", session_id="sess1")
    assert result["request_id"] == "req1"
    assert result["session_id"] == "sess1"
