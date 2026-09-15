"""Tests for promote_feedback.py's pure logic: no file I/O, no AWS."""
import copy

import pytest

from promote_feedback import bump_version, promote, validate_dataset

BASE_DATASET = {
    "dataset_version": "1.0",
    "cases": [
        {"id": "app-breach", "category": "interpretation", "question": "Q1?",
         "source_section": "6A", "expected_terms": ["6a breach"]},
    ],
}


def make_queue(*items):
    return {"generated_at": "2026-09-15T00:00:00Z", "items": list(items)}


def promoted_item(request_id="abcdef1234", **overrides):
    item = {
        "request_id": request_id,
        "question": "What must an APP entity do with unsolicited information?",
        "review_status": "promoted",
        "category": "production-feedback",
        "source_section": "APP 4",
        "expected_terms": ["destroy or de-identify"],
    }
    item.update(overrides)
    return item


def test_bump_version_increments_minor():
    assert bump_version("1.0") == "1.1"
    assert bump_version("1.9") == "1.10"


def test_bump_version_falls_back_for_unparseable_version():
    assert bump_version("weird") == "weird-fb"


def test_validate_dataset_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing"):
        validate_dataset({"cases": [{"id": "x", "category": "c"}]})


def test_validate_dataset_rejects_duplicate_ids():
    dataset = {"cases": [
        {"id": "x", "category": "c", "question": "q", "source_section": "s", "expected_terms": ["t"]},
        {"id": "x", "category": "c", "question": "q2", "source_section": "s2", "expected_terms": ["t2"]},
    ]}
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(dataset)


def test_validate_dataset_rejects_empty_expected_terms():
    dataset = {"cases": [
        {"id": "x", "category": "c", "question": "q", "source_section": "s", "expected_terms": []},
    ]}
    with pytest.raises(ValueError, match="no expected terms"):
        validate_dataset(dataset)


def test_promote_appends_fully_annotated_item_and_bumps_version():
    dataset = copy.deepcopy(BASE_DATASET)
    queue = make_queue(promoted_item())

    new_dataset, new_queue, messages, promoted_count = promote(queue, dataset)

    assert promoted_count == 1
    assert new_dataset["dataset_version"] == "1.1"
    ids = [c["id"] for c in new_dataset["cases"]]
    assert "fb-abcdef12" in ids
    new_case = next(c for c in new_dataset["cases"] if c["id"] == "fb-abcdef12")
    assert new_case["origin"] == "production_feedback"
    assert new_case["origin_request_id"] == "abcdef1234"
    assert new_queue["items"][0]["review_status"] == "promoted_committed"
    assert any("promoted abcdef1234" in m for m in messages)


def test_promote_skips_pending_and_dismissed_items():
    dataset = copy.deepcopy(BASE_DATASET)
    queue = make_queue(
        promoted_item(request_id="pending001", review_status="pending"),
        promoted_item(request_id="dismiss001", review_status="dismissed"),
    )

    new_dataset, new_queue, messages, promoted_count = promote(queue, dataset)

    assert promoted_count == 0
    assert new_dataset["cases"] == BASE_DATASET["cases"]
    assert new_dataset.get("dataset_version", "1.0") == "1.0"


def test_promote_skips_and_reports_missing_annotations():
    dataset = copy.deepcopy(BASE_DATASET)
    queue = make_queue(promoted_item(request_id="incomplete1", source_section=None))

    new_dataset, new_queue, messages, promoted_count = promote(queue, dataset)

    assert promoted_count == 0
    assert any("missing" in m and "incomplete1" in m for m in messages)
    assert new_queue["items"][0]["review_status"] == "promoted"  # left alone, not "committed"


def test_promote_is_idempotent_on_rerun():
    dataset = copy.deepcopy(BASE_DATASET)
    queue = make_queue(promoted_item())

    dataset, queue, _, first_run_count = promote(queue, dataset)
    dataset, queue, messages, second_run_count = promote(queue, dataset)

    assert first_run_count == 1
    assert second_run_count == 0
    assert len([c for c in dataset["cases"] if c["id"] == "fb-abcdef12"]) == 1
    assert queue["items"][0]["review_status"] == "promoted_committed"
