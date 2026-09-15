"""Tests for check_eval_regression.py's pure comparison logic and baseline round-trip.

Deliberately does not build a real index or import rag_backend/rag_evaluation at module scope -
that needs sentence-transformers/torch and a real (or cached) BGE download, which is exercised
as its own CI step (`python check_eval_regression.py`), not through pytest.
"""
import json

from check_eval_regression import check_regression, load_baseline, save_baseline

BASELINE = {"provider": "bge", "model_id": "BAAI/bge-base-en-v1.5", "dataset_version": "1.0",
            "cases": 25, "hit_rate": 0.64, "mrr": 0.45}


def test_check_regression_passes_within_tolerance():
    summary = {"hit_rate": 0.63, "mrr": 0.44}  # tiny drop, inside the 0.02 tolerance
    assert check_regression(summary, BASELINE, fb_miss_ids=[]) == []


def test_check_regression_passes_on_improvement():
    summary = {"hit_rate": 0.72, "mrr": 0.50}
    assert check_regression(summary, BASELINE, fb_miss_ids=[]) == []


def test_check_regression_flags_hit_rate_drop():
    summary = {"hit_rate": 0.50, "mrr": 0.45}
    failures = check_regression(summary, BASELINE, fb_miss_ids=[])
    assert len(failures) == 1
    assert "hit_rate regressed" in failures[0]


def test_check_regression_flags_mrr_drop():
    summary = {"hit_rate": 0.64, "mrr": 0.30}
    failures = check_regression(summary, BASELINE, fb_miss_ids=[])
    assert len(failures) == 1
    assert "mrr regressed" in failures[0]


def test_check_regression_flags_both():
    summary = {"hit_rate": 0.40, "mrr": 0.20}
    failures = check_regression(summary, BASELINE, fb_miss_ids=[])
    assert len(failures) == 2


def test_check_regression_always_fails_on_any_feedback_case_miss_even_within_tolerance():
    # Aggregate metrics look fine, but a specific production-derived case regressed - that must
    # fail regardless of the aggregate tolerance, since that's the entire point of the case.
    summary = {"hit_rate": 0.64, "mrr": 0.45}
    failures = check_regression(summary, BASELINE, fb_miss_ids=["fb-abcdef12"])
    assert len(failures) == 1
    assert "production-feedback case(s) regressed" in failures[0]
    assert "fb-abcdef12" in failures[0]


def test_baseline_round_trip(tmp_path):
    path = tmp_path / "eval_baseline.json"
    assert load_baseline(path) is None

    save_baseline({"cases": 25, "hit_rate": 0.64, "mrr": 0.45}, "bge", "BAAI/bge-base-en-v1.5",
                   "1.0", path=path)

    loaded = load_baseline(path)
    assert loaded["hit_rate"] == 0.64
    assert loaded["mrr"] == 0.45
    assert loaded["dataset_version"] == "1.0"
    # Human-readable and diffable in a PR, like the rest of the repo's data/ files.
    assert json.loads(path.read_text())
