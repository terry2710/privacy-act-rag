"""The Q&A logging path with a stubbed index and LLM. No AWS, no network, deterministic."""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import rag_backend
import rag_logging

SHORT_A = "Principle 1: collection of personal information must be lawful."
SHORT_B = "Principle 2: anonymity and pseudonymity options must be offered."
LONG = "x" * 5000

DOCS = [
    (Document(page_content=SHORT_A, metadata={"page": 3, "source": "privacy_act.pdf"}), 0.1234567),
    (Document(page_content=SHORT_B, metadata={"page": 4, "source": "privacy_act.pdf"}), 0.5),
    (Document(page_content=LONG, metadata={"page": 9, "source": "privacy_act.pdf"}), 1.25),
]


class FakeIndex:
    def __init__(self, docs=DOCS, exc=None):
        self.docs, self.exc, self.calls = docs, exc, []

    def similarity_search_with_score(self, query, k):
        self.calls.append((query, k))
        if self.exc:
            raise self.exc
        return self.docs[:k]


def fake_llm(text="Stub answer.", exc=None):
    def _call(_prompt_value):
        if exc:
            raise exc
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": 123, "output_tokens": 45, "total_tokens": 168},
            response_metadata={"stopReason": "end_turn"},
        )
    return RunnableLambda(_call)


@pytest.fixture
def stub_llm(monkeypatch):
    def _install(**kwargs):
        monkeypatch.setattr(rag_backend, "prv_llm", lambda: fake_llm(**kwargs))
    return _install


@pytest.fixture
def turn(jsonl_logging, stub_llm):
    """One successful Q&A turn; yields (answer, logged_event, fake_index)."""
    stub_llm()
    index = FakeIndex()
    answer = rag_backend.prv_rag_response(index, "What is APP 1?", k=3, session_id="sess-abc")
    events = jsonl_logging.read_events()
    assert len(events) == 1, f"expected exactly one event, got {len(events)}"
    return answer, events[0], index


# ----------------------------------------------------------------- happy path

def test_answer_returned_verbatim(turn):
    answer, _, _ = turn
    assert answer == "Stub answer."


def test_retriever_called_with_k(turn):
    _, _, index = turn
    assert index.calls == [("What is APP 1?", 3)]


def test_event_identity_fields(turn):
    _, e, _ = turn
    assert e["event"] == "qa"
    assert e["schema"] == "qa/2"
    assert len(e["request_id"]) == 32
    assert e["session_id"] == "sess-abc"
    assert e["ts"].endswith("Z")


def test_question_and_model_metadata(turn):
    _, e, _ = turn
    assert e["question"] == "What is APP 1?"
    assert e["question_chars"] == len("What is APP 1?")
    assert e["k"] == 3
    assert e["model_id"] == rag_backend.LLM_MODEL_ID
    assert e["embedding_model_id"] == rag_backend.EMBEDDING_MODEL_ID


def test_timings_are_consistent(turn):
    _, e, _ = turn
    assert e["retrieval_ms"] >= 0 and e["llm_ms"] >= 0
    assert e["total_ms"] >= max(e["retrieval_ms"], e["llm_ms"])


def test_answer_and_usage_logged(turn):
    _, e, _ = turn
    assert e["answer"] == "Stub answer."
    assert e["answer_truncated"] is False
    assert e["answer_chars"] == len("Stub answer.")
    assert (e["input_tokens"], e["output_tokens"], e["total_tokens"]) == (123, 45, 168)
    assert e["stop_reason"] == "end_turn"


def test_context_is_the_joined_chunks(turn):
    _, e, _ = turn
    assert e["context_chars"] == len("\n\n".join([SHORT_A, SHORT_B, LONG]))
    assert e["chunk_count"] == 3


# --------------------------------------------------------------- chunk detail

def test_chunks_are_ranked(turn):
    _, e, _ = turn
    assert [c["rank"] for c in e["chunks"]] == [1, 2, 3]


def test_raw_l2_distance_is_preserved(turn):
    _, e, _ = turn
    assert [c["l2_sq"] for c in e["chunks"]] == [0.123457, 0.5, 1.25]


def test_score_is_cosine_not_raw_distance(turn):
    _, e, _ = turn
    # cosine = 1 - l2_sq/2
    assert [c["score"] for c in e["chunks"]] == [0.938272, 0.75, 0.375]


def test_cosine_decreases_as_distance_grows(turn):
    _, e, _ = turn
    scores = [c["score"] for c in e["chunks"]]
    assert scores == sorted(scores, reverse=True)


def test_cosine_identity_holds(turn):
    _, e, _ = turn
    for c in e["chunks"]:
        assert abs(c["l2_sq"] - (2 - 2 * c["score"])) < 1e-5


@pytest.mark.parametrize("l2_sq, expected", [
    (0.0, 1.0), (2.0, 0.0), (4.0, -1.0),
    (-1e-9, 1.0),        # float noise below zero must clamp
    (4.0000003, -1.0),   # and above the maximum distance
])
def test_cosine_conversion_is_clamped(l2_sq, expected):
    assert rag_logging.cosine_from_l2_sq(l2_sq) == expected


def test_chunk_metadata_and_id(turn):
    _, e, _ = turn
    first = e["chunks"][0]
    assert first["page"] == 3
    assert first["source"] == "privacy_act.pdf"
    assert first["chunk_id"] == rag_logging.chunk_id(SHORT_A)
    assert len(first["chunk_id"]) == 12


def test_long_chunk_text_is_truncated_but_length_is_kept(turn):
    _, e, _ = turn
    short, long_chunk = e["chunks"][0], e["chunks"][2]
    assert short["text_truncated"] is False
    assert long_chunk["text_truncated"] is True
    assert len(long_chunk["text"]) == 800          # PRV_LOG_CHUNK_CHARS default
    assert long_chunk["chars"] == 5000             # untruncated length still reported


# --------------------------------------------------------------- error paths

def test_retrieval_failure_is_logged_then_reraised(jsonl_logging, stub_llm):
    stub_llm()
    boom = RuntimeError("faiss exploded")
    with pytest.raises(RuntimeError) as excinfo:
        rag_backend.prv_rag_response(FakeIndex(exc=boom), "q", k=2, session_id="s")
    assert excinfo.value is boom, "the original exception must propagate unchanged"

    e = jsonl_logging.read_events()[0]
    assert e["event"] == "qa_error"
    assert e["stage"] == "retrieval"
    assert e["error_type"] == "RuntimeError"
    assert e["error"] == "faiss exploded"
    assert e["total_ms"] is not None
    assert e["question"] == "q" and e["session_id"] == "s"


def test_llm_failure_keeps_retrieval_detail(jsonl_logging, stub_llm):
    stub_llm(exc=ValueError("throttled"))
    with pytest.raises(ValueError):
        rag_backend.prv_rag_response(FakeIndex(), "q2", k=1)

    e = jsonl_logging.read_events()[0]
    assert e["event"] == "qa_error"
    assert e["stage"] == "llm"
    assert e["chunk_count"] == 1 and e["chunks"]
    assert e["retrieval_ms"] is not None
    assert e["llm_ms"] is not None and e["total_ms"] is not None
    assert e["session_id"] is None


# ------------------------------------------------------- CloudWatch size cap

def test_oversized_event_is_shrunk_below_the_cloudwatch_cap(tmp_path, stub_llm):
    from conftest import reload_logging

    stub_llm()
    log_file = tmp_path / "big.jsonl"
    with reload_logging(PRV_LOG_GROUP="", PRV_QA_LOG_FILE=str(log_file),
                        PRV_LOG_CHUNK_CHARS="full") as mod:
        huge = [(Document(page_content="y" * 120_000, metadata={"page": i}), 0.1 * i)
                for i in range(1, 4)]
        rag_backend.prv_rag_response(FakeIndex(huge), "big", k=3)

        import json
        e = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
        assert e["size_capped"] is True
        assert len(json.dumps(e, ensure_ascii=False).encode()) <= mod.MAX_EVENT_BYTES
        # metadata must survive even when the text is dropped
        assert all(c["chars"] == 120_000 and c["chunk_id"] for c in e["chunks"])


def test_normal_event_is_not_flagged_as_capped(turn):
    _, e, _ = turn
    assert e.get("size_capped") is None


def test_logging_failure_cannot_break_a_turn(jsonl_logging, stub_llm, monkeypatch, capsys):
    """The answer is already computed when we log, so a logging fault must never surface."""
    stub_llm()

    def explode(_record):
        raise RuntimeError("serialisation is broken")

    monkeypatch.setattr(rag_logging, "_serialize", explode)
    answer = rag_backend.prv_rag_response(FakeIndex(), "q", k=1)

    assert answer == "Stub answer.", "the user's answer must survive a logging failure"
    assert jsonl_logging.read_events() == [], "nothing should have been written"
    assert "could not serialise event" in capsys.readouterr().err
