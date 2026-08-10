"""End-to-end against the real FAISS index and Bedrock, logging to a local JSONL only."""
import pytest

import rag_backend

pytestmark = [pytest.mark.aws, pytest.mark.index]

QUESTION = "What are the Australian Privacy Principles about collection of personal information?"


@pytest.fixture(scope="module")
def index(aws, index_dir):
    return rag_backend.prv_index()


@pytest.fixture(scope="module")
def turn(index, tmp_path_factory):
    """One real Q&A turn. Module-scoped so the Bedrock calls happen once."""
    from conftest import reload_logging

    log_file = tmp_path_factory.mktemp("e2e") / "qa.jsonl"
    with reload_logging(PRV_LOG_GROUP="", PRV_QA_LOG_FILE=str(log_file)):
        answer = rag_backend.prv_rag_response(index, QUESTION, session_id="pytest-e2e")

    import json
    events = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(events) == 1
    return answer, events[0]


def test_index_loads_from_cache_without_rebuilding(index):
    assert index.index.ntotal > 0


def test_real_answer_is_produced(turn):
    answer, _ = turn
    assert answer.strip()
    assert len(answer) > 50


def test_real_event_is_complete(turn):
    _, e = turn
    assert e["event"] == "qa"
    assert e["schema"] == "qa/2"
    assert e["question"] == QUESTION
    assert e["session_id"] == "pytest-e2e"
    assert e["chunk_count"] == rag_backend.RETRIEVER_K


def test_bedrock_reports_token_usage(turn):
    """Nova Pro via ChatBedrock populates usage_metadata - regression guard if the model changes."""
    _, e = turn
    assert e["input_tokens"] > 0
    assert e["output_tokens"] > 0
    assert e["total_tokens"] == e["input_tokens"] + e["output_tokens"]
    assert e["stop_reason"]


def test_timings_are_recorded(turn):
    _, e = turn
    assert e["retrieval_ms"] > 0 and e["llm_ms"] > 0
    assert e["total_ms"] >= e["retrieval_ms"] + e["llm_ms"] - 1


def test_real_scores_are_cosine_and_ordered(turn):
    _, e = turn
    scores = [c["score"] for c in e["chunks"]]
    assert scores == sorted(scores, reverse=True), "cosine must decrease with rank"
    assert all(0.0 <= s <= 1.0 for s in scores), f"cosine out of range: {scores}"
    for c in e["chunks"]:
        assert abs(c["l2_sq"] - (2 - 2 * c["score"])) < 1e-4


def test_chunks_carry_source_metadata(turn):
    _, e = turn
    for c in e["chunks"]:
        assert c["page"] is not None
        assert "legislation.gov.au" in (c["source"] or "")
        assert c["chars"] > 0


def test_embeddings_are_unit_norm(index):
    """The cosine conversion is only valid while embeddings are normalised - see
    rag_logging.cosine_from_l2_sq. If this fails, the index needs MAX_INNER_PRODUCT instead."""
    import numpy as np
    vectors = index.index.reconstruct_n(0, min(index.index.ntotal, 200))
    norms = np.linalg.norm(vectors, axis=1)
    assert np.abs(norms - 1.0).max() < 1e-5, f"embeddings are not unit-norm: {norms.min()}..{norms.max()}"


def test_off_topic_question_scores_far_lower(index):
    """Sanity check that the cosine score actually discriminates."""
    on_topic = index.similarity_search_with_score("What is a notifiable data breach?", k=1)[0][1]
    off_topic = index.similarity_search_with_score("What is the recipe for chocolate cake?", k=1)[0][1]
    import rag_logging
    assert rag_logging.cosine_from_l2_sq(on_topic) > rag_logging.cosine_from_l2_sq(off_topic) + 0.2
