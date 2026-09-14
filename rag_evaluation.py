"""Small, deterministic retrieval benchmark for the Privacy Act index."""

DEFAULT_CASES = (
    {
        "id": "app-breach",
        "question": "When does an act or practice breach an Australian Privacy Principle?",
        "expected_terms": ("6a breach of an australian privacy principle",),
    },
    {
        "id": "eligible-breach",
        "question": "What is an eligible data breach?",
        "expected_terms": ("26we eligible data breach",),
    },
    {
        "id": "breach-notice",
        "question": "What information must an eligible data breach notification contain?",
        "expected_terms": ("identity and contact details of the entity",),
    },
    {
        "id": "app-11",
        "question": "What does Australian Privacy Principle 11 require?",
        "expected_terms": ("australian privacy principle 11", "app 11"),
    },
    {
        "id": "app-12",
        "question": "When must an APP entity give an individual access to personal information?",
        "expected_terms": ("australian privacy principle 12", "app 12"),
    },
)


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
                "question": case["question"],
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
