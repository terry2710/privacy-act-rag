"""Pure-logic tests for feedback_review.py - no AWS calls (no `aws` marker needed).

The AWS-touching parts (run_query/fetch_not_helpful/fetch_qa_turns) are exercised manually
against the real log group, since they need real CloudWatch data and cost a small amount of
Logs Insights scan time; what's tested here is the joining/merging logic, which is where a
silent bug (dropped item, clobbered human annotation) would actually hurt.
"""
from feedback_review import build_queue, merge_queue


def test_build_queue_joins_qa_fields_by_request_id():
    feedback_rows = [
        {"@timestamp": "2026-09-10 01:00:00.000", "request_id": "abc123", "session_id": "s1", "rating": "not_helpful"},
    ]
    qa_by_id = {
        "abc123": {
            "question": "What is a notifiable data breach?",
            "answer": "...",
            "chunk_count": "4",
            "top_chunk_id": "chunkX",
            "top_page": "12",
            "top_score": "0.41",
            "embedding_model_id": "amazon.titan-embed-text-v2:0",
            "pipeline_version": "fixed-dense-v1",
        }
    }

    items = build_queue(feedback_rows, qa_by_id)

    assert len(items) == 1
    item = items[0]
    assert item["request_id"] == "abc123"
    assert item["question"] == "What is a notifiable data breach?"
    assert item["qa_found"] is True
    assert item["review_status"] == "pending"
    # Human-filled annotations start empty, not fabricated from the qa record.
    assert item["category"] is None
    assert item["source_section"] is None
    assert item["expected_terms"] is None


def test_build_queue_marks_missing_qa_record():
    feedback_rows = [{"@timestamp": "t", "request_id": "orphan", "session_id": "s1", "rating": "not_helpful"}]

    items = build_queue(feedback_rows, qa_by_id={})

    assert items[0]["qa_found"] is False
    assert items[0]["question"] is None


def test_build_queue_skips_rows_without_request_id():
    # Defensive: a malformed/partial log row should never crash the join.
    items = build_queue([{"@timestamp": "t", "session_id": "s1"}], qa_by_id={})
    assert items == []


def test_merge_queue_preserves_human_annotations_on_rerun():
    existing = [{
        "request_id": "abc123",
        "feedback_ts": "2026-09-10 01:00:00.000",
        "review_status": "promoted",
        "category": "app-framework",
        "source_section": "APP 6",
        "expected_terms": ["some distinctive phrase"],
        "notes": "confirmed retrieval miss",
    }]
    # A second feedback_review.py run re-derives this item fresh from CloudWatch - it must not
    # clobber the human's review_status/category/source_section/expected_terms/notes.
    rederived = [{
        "request_id": "abc123",
        "feedback_ts": "2026-09-10 01:00:00.000",
        "review_status": "pending",
        "category": None,
        "source_section": None,
        "expected_terms": None,
        "notes": "",
    }]

    merged, added = merge_queue(existing, rederived)

    assert added == 0
    assert len(merged) == 1
    assert merged[0]["review_status"] == "promoted"
    assert merged[0]["source_section"] == "APP 6"


def test_merge_queue_adds_only_genuinely_new_items():
    existing = [{"request_id": "old1", "feedback_ts": "2026-09-01T00:00:00Z", "review_status": "dismissed"}]
    new = [
        {"request_id": "old1", "feedback_ts": "2026-09-01T00:00:00Z", "review_status": "pending"},
        {"request_id": "new1", "feedback_ts": "2026-09-12T00:00:00Z", "review_status": "pending"},
    ]

    merged, added = merge_queue(existing, new)

    assert added == 1
    ids = {item["request_id"] for item in merged}
    assert ids == {"old1", "new1"}
    # Newest feedback first.
    assert merged[0]["request_id"] == "new1"
