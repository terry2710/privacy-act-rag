"""Deterministic retrieval evaluation for the Privacy Act index."""

import json
from pathlib import Path

from rag_retrieval import HybridReranker


DATASET_PATH = Path(__file__).parent / "data" / "retrieval_eval.json"


def load_dataset(path=DATASET_PATH):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("retrieval evaluation dataset must contain a non-empty cases list")

    required = {"id", "category", "question", "source_section", "expected_terms"}
    for case in cases:
        missing = required.difference(case)
        if missing:
            case_id = case.get("id", "<unknown>")
            raise ValueError(f"evaluation case {case_id} is missing {sorted(missing)}")
        if not case["expected_terms"]:
            raise ValueError(f"evaluation case {case['id']} has no expected terms")
    return dataset


DEFAULT_CASES = tuple(load_dataset()["cases"])


def _cosine_from_l2_sq(score):
    return max(-1.0, min(1.0, 1.0 - float(score) / 2.0))


def _normalize_text(text):
    return " ".join(text.lower().split())


def evaluate_retrieval(index, cases=DEFAULT_CASES, k=4):
    """Run labeled queries and return an aggregate summary plus per-case rows."""
    rows = []

    for case in cases:
        scored_docs = index.similarity_search_with_score(case["question"], k=k)
        expected_terms = tuple(_normalize_text(term) for term in case["expected_terms"])
        first_rank = None

        for rank, (doc, _) in enumerate(scored_docs, start=1):
            text = _normalize_text(doc.page_content)
            if any(term in text for term in expected_terms):
                first_rank = rank
                break

        top_cosine = _cosine_from_l2_sq(scored_docs[0][1]) if scored_docs else None
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "source_section": case["source_section"],
                "result": "Pass" if first_rank is not None else "Miss",
                "first_rank": first_rank,
                "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                "top_cosine": top_cosine,
            }
        )

    total = len(rows)
    hits = sum(row["result"] == "Pass" for row in rows)
    summary = {
        "cases": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / total if total else 0.0,
    }
    return summary, rows


def compare_retrieval(index, cases=DEFAULT_CASES, k=4, candidate_k=50):
    """Evaluate dense retrieval and hybrid reranking over the same embedded queries."""
    reranker = HybridReranker(index, candidate_k=candidate_k)
    dense_rows = []
    hybrid_rows = []

    for case in cases:
        dense_results, hybrid_results = reranker.retrieve(case["question"], k=k)
        dense_rows.append(_evaluate_case(case, dense_results))
        hybrid_rows.append(_evaluate_case(case, hybrid_results))

    dense_summary = _summarize(dense_rows)
    hybrid_summary = _summarize(hybrid_rows)
    rows = []
    for dense, hybrid in zip(dense_rows, hybrid_rows):
        rows.append(
            {
                **dense,
                "dense_result": dense["result"],
                "dense_rank": dense["first_rank"],
                "hybrid_result": hybrid["result"],
                "hybrid_rank": hybrid["first_rank"],
            }
        )
    return dense_summary, hybrid_summary, rows


def _evaluate_case(case, scored_docs):
    expected_terms = tuple(_normalize_text(term) for term in case["expected_terms"])
    first_rank = None
    for rank, (doc, _) in enumerate(scored_docs, start=1):
        text = _normalize_text(doc.page_content)
        if any(term in text for term in expected_terms):
            first_rank = rank
            break

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "source_section": case["source_section"],
        "result": "Pass" if first_rank is not None else "Miss",
        "first_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "top_cosine": _cosine_from_l2_sq(scored_docs[0][1]) if scored_docs else None,
    }


def _summarize(rows):
    total = len(rows)
    hits = sum(row["result"] == "Pass" for row in rows)
    return {
        "cases": total,
        "hits": hits,
        "hit_rate": hits / total if total else 0.0,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / total if total else 0.0,
    }
