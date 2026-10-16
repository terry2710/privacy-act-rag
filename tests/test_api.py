import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from api.db import Base, make_session_factory
from api.main import create_app
from api.repository import SqlRepository
from api.service import IndexNotReady


class FakeService:
    def __init__(self, ready=True, fail=False, index_missing=False):
        self.ready = ready
        self.fail = fail
        self.index_missing = index_missing
        self.feedback = []
        self.warmed_up = threading.Event()

    def warm_up(self, wait=True):
        self.warmed_up.set()
        return self.ready

    def is_ready(self):
        return self.ready

    def answer(self, question, session_id=None):
        if self.index_missing:
            raise IndexNotReady()
        if self.fail:
            raise RuntimeError("bedrock unavailable")
        return {
            "request_id": "req123",
            "session_id": session_id or "sess1",
            "answer": "An entity must handle personal information openly.",
            "chunks": [
                {
                    "rank": 1,
                    "chunk_id": "abc",
                    "page": 12,
                    "source": "privacy-act.pdf",
                    "score": 0.81,
                    "text": "APP 1 requires open and transparent management.",
                }
            ],
            "pipeline_version": "fixed-dense-v1",
        }

    def record_feedback(self, request_id, session_id, rating):
        self.feedback.append((request_id, session_id, rating))


class FakeRepository:
    def __init__(self, persist=True):
        self.persist = persist
        self.queries = []
        self.feedback = []

    def save_query(self, **fields):
        self.queries.append(fields)
        return self.persist

    def save_feedback(self, request_id, owner_sub, rating):
        self.feedback.append((request_id, owner_sub, rating))
        return self.persist


def make_client(service=None, repository=None):
    return TestClient(create_app(service or FakeService(), repository or FakeRepository()))


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def repository():
    return FakeRepository()


@pytest.fixture
def client(service, repository):
    return make_client(service, repository)


def test_live_returns_ok(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reflects_service_state():
    response = make_client(service=FakeService(ready=False)).get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_query_returns_answer_with_citations(client):
    response = client.post("/api/queries", json={"question": "What is APP 1?"})
    body = response.json()
    assert response.status_code == 200
    assert body["request_id"] == "req123"
    assert body["outcome"] == "answered"
    assert body["citations"][0]["chunk_id"] == "abc"
    assert body["citations"][0]["page"] == 12


def test_query_response_carries_request_id_header(client):
    response = client.post("/api/queries", json={"question": "What is APP 1?"})
    assert response.headers["x-request-id"]


def test_query_echoes_supplied_request_id_header(client):
    response = client.post(
        "/api/queries",
        json={"question": "What is APP 1?"},
        headers={"x-request-id": "trace-42"},
    )
    assert response.headers["x-request-id"] == "trace-42"


@pytest.mark.parametrize("question", ["", "   ", "x" * 2001])
def test_query_rejects_invalid_question(client, question):
    response = client.post("/api/queries", json={"question": question})
    assert response.status_code == 422


def test_query_rejects_missing_body(client):
    response = client.post("/api/queries")
    assert response.status_code == 422


def test_query_upstream_failure_returns_502_without_leaking_detail():
    response = make_client(service=FakeService(fail=True)).post(
        "/api/queries", json={"question": "What is APP 1?"}
    )
    assert response.status_code == 502
    assert response.json()["error"] == "upstream_error"
    assert "bedrock" not in response.text


def test_query_is_persisted(client, repository):
    client.post("/api/queries", json={"question": "  What is APP 1?  "})
    assert len(repository.queries) == 1
    saved = repository.queries[0]
    assert saved["request_id"] == "req123"
    assert saved["owner_sub"] == "anonymous"
    assert saved["question"] == "What is APP 1?"
    assert saved["outcome"] == "answered"


def test_failed_query_is_not_persisted(repository):
    make_client(FakeService(fail=True), repository).post(
        "/api/queries", json={"question": "What is APP 1?"}
    )
    assert repository.queries == []


def test_query_succeeds_when_persistence_fails():
    response = make_client(repository=FakeRepository(persist=False)).post(
        "/api/queries", json={"question": "What is APP 1?"}
    )
    assert response.status_code == 200


def test_feedback_is_recorded_and_persisted(client, service, repository):
    response = client.post(
        "/api/feedback",
        json={"request_id": "req123", "session_id": "sess1", "rating": "helpful"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "recorded", "persisted": True}
    assert service.feedback == [("req123", "sess1", "helpful")]
    assert repository.feedback == [("req123", "anonymous", "helpful")]


def test_feedback_reports_when_not_persisted():
    response = make_client(repository=FakeRepository(persist=False)).post(
        "/api/feedback", json={"request_id": "req123", "rating": "helpful"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "recorded", "persisted": False}


def test_feedback_rejects_unknown_rating(client):
    response = client.post(
        "/api/feedback",
        json={"request_id": "req123", "rating": "excellent"},
    )
    assert response.status_code == 422


def test_query_then_feedback_round_trip_with_sql_repository():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    client = make_client(repository=SqlRepository(make_session_factory(engine)))

    client.post("/api/queries", json={"question": "What is APP 1?"})
    response = client.post("/api/feedback", json={"request_id": "req123", "rating": "helpful"})

    assert response.json() == {"status": "recorded", "persisted": True}


def test_query_returns_503_while_index_is_not_ready(repository):
    response = make_client(FakeService(index_missing=True), repository).post(
        "/api/queries", json={"question": "What is APP 1?"}
    )
    assert response.status_code == 503
    assert response.json()["error"] == "index_not_ready"
    assert response.headers["retry-after"] == "30"
    assert repository.queries == []


def test_startup_warms_up_index_in_background(service, repository):
    with TestClient(create_app(service, repository)):
        assert service.warmed_up.wait(timeout=2)
