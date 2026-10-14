import pytest
from fastapi.testclient import TestClient

from api.main import create_app


class FakeService:
    def __init__(self, ready=True, fail=False):
        self.ready = ready
        self.fail = fail
        self.feedback = []

    def is_ready(self):
        return self.ready

    def answer(self, question, session_id=None):
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


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_live_returns_ok(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reflects_service_state():
    client = TestClient(create_app(FakeService(ready=False)))
    response = client.get("/health/ready")
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
    client = TestClient(create_app(FakeService(fail=True)))
    response = client.post("/api/queries", json={"question": "What is APP 1?"})
    assert response.status_code == 502
    assert response.json()["error"] == "upstream_error"
    assert "bedrock" not in response.text


def test_feedback_is_recorded(client, service):
    response = client.post(
        "/api/feedback",
        json={"request_id": "req123", "session_id": "sess1", "rating": "helpful"},
    )
    assert response.status_code == 200
    assert response.json() == {"status": "recorded"}
    assert service.feedback == [("req123", "sess1", "helpful")]


def test_feedback_rejects_unknown_rating(client):
    response = client.post(
        "/api/feedback",
        json={"request_id": "req123", "rating": "excellent"},
    )
    assert response.status_code == 422
