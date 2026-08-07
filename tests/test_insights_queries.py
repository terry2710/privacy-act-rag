"""Every Logs Insights query documented in the README must actually run and return data.

These queries are the reason the log exists, and a silently-wrong one (empty grouping keys,
unparsed field) looks identical to "no traffic yet" in the console. Pin them down here.
"""
import time

import pytest

pytestmark = pytest.mark.aws

# Mirrors what a real turn writes, so the queries exercise the same field layout.
SAMPLE = [
    ("What are the functions of the Information Commissioner?", [0.819188, 0.7956, 0.7853, 0.7135], 252),
    ("What is a notifiable data breach?", [0.634173, 0.6155, 0.6126, 0.6056], 226),
    ("What is the recipe for chocolate cake?", [0.126639, 0.1200, 0.1150, 0.1100], 210),
]


@pytest.fixture(scope="module")
def seeded(aws, cw_log_group):
    from conftest import reload_logging

    with reload_logging(PRV_LOG_GROUP=cw_log_group, PRV_QA_LOG_FILE=None) as mod:
        for question, scores, page in SAMPLE:
            mod.emit({
                "event": "qa", "schema": "qa/2", "session_id": "pytest-insights",
                "question": question, "total_ms": 3000.0, "llm_ms": 2000.0,
                "chunk_count": len(scores), "input_tokens": 1200, "output_tokens": 200,
                "chunks": [{"rank": i + 1, "score": s, "l2_sq": round(2 - 2 * s, 6),
                            "chunk_id": f"chunk{page}{i}", "page": page}
                           for i, s in enumerate(scores)],
            })
        mod._sink.shutdown()
    time.sleep(20)   # Logs Insights needs the events indexed before a query will see them
    return cw_log_group


def run_query(session, group, query, timeout=60):
    client = session.client("logs")
    qid = client.start_query(logGroupName=group, startTime=int(time.time()) - 3600,
                             endTime=int(time.time()) + 60, queryString=query)["queryId"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        result = client.get_query_results(queryId=qid)
        if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
    assert result["status"] == "Complete", f"query did not complete: {result['status']}\n{query}"
    return [{f["field"]: f["value"] for f in row if f["field"] != "@ptr"}
            for row in result["results"]]


def test_recent_turns(aws, seeded):
    rows = run_query(aws, seeded, """
        fields @timestamp, question, total_ms, chunk_count, input_tokens, output_tokens
        | filter event = "qa"
        | sort @timestamp desc""")
    assert len(rows) >= 3
    assert all(r.get("question") for r in rows), "question field was not parsed"
    assert all(r.get("total_ms") for r in rows)


def test_latency_trend(aws, seeded):
    rows = run_query(aws, seeded, """
        filter event = "qa"
        | stats avg(llm_ms) as avg_llm_ms, max(llm_ms) as max_llm_ms by bin(1h)""")
    assert rows and float(rows[0]["avg_llm_ms"]) > 0


def test_best_match_per_question(aws, seeded):
    rows = run_query(aws, seeded, """
        fields question, chunks.0.score as best_cosine, chunks.0.page as best_page
        | filter event = "qa" and schema = "qa/2"
        | sort best_cosine asc""")
    assert len(rows) >= 3
    assert all("best_cosine" in r and "best_page" in r for r in rows), \
        "nested chunks.0.* fields did not resolve"
    scores = [float(r["best_cosine"]) for r in rows]
    assert scores == sorted(scores), "sort by a nested field did not apply"
    assert scores[0] < 0.2, "the off-topic question should rank worst"


def test_weak_retrieval_filter(aws, seeded):
    rows = run_query(aws, seeded, """
        fields question, chunks.0.score as best_cosine
        | filter event = "qa" and schema = "qa/2" and chunks.0.score < 0.6
        | sort best_cosine asc""")
    questions = [r["question"] for r in rows]
    assert "What is the recipe for chocolate cake?" in questions
    assert "What are the functions of the Information Commissioner?" not in questions


def test_most_retrieved_chunks(aws, seeded):
    rows = run_query(aws, seeded, """
        filter event = "qa" and schema = "qa/2"
        | stats count(*) as hits, avg(chunks.0.score) as avg_cosine
            by chunks.0.chunk_id as chunk, chunks.0.page as page
        | sort hits desc""")
    assert rows
    # The unnest form silently produced empty grouping keys; that is what this guards against.
    assert all(r.get("chunk") and r.get("page") for r in rows), \
        f"grouping keys came back empty: {rows}"
    assert all(r.get("avg_cosine") for r in rows)


def test_confidence_trend(aws, seeded):
    rows = run_query(aws, seeded, """
        filter event = "qa" and schema = "qa/2"
        | stats avg(chunks.0.score) as avg_best, min(chunks.0.score) as worst_best by bin(1h)""")
    assert rows
    assert float(rows[0]["avg_best"]) > 0, "aggregate over a nested field was dropped"
    assert float(rows[0]["worst_best"]) <= float(rows[0]["avg_best"])


def test_score_decay_across_ranks(aws, seeded):
    rows = run_query(aws, seeded, """
        filter event = "qa" and schema = "qa/2"
        | stats avg(chunks.0.score) as r1, avg(chunks.1.score) as r2,
                avg(chunks.2.score) as r3, avg(chunks.3.score) as r4""")
    assert rows
    ranks = [float(rows[0][f"r{i}"]) for i in range(1, 5)]
    assert ranks == sorted(ranks, reverse=True), f"expected decaying relevance, got {ranks}"


def test_failures_by_stage(aws, seeded):
    # No errors seeded, so this must complete cleanly and return nothing rather than error.
    rows = run_query(aws, seeded, """
        filter event = "qa_error"
        | stats count(*) by stage, error_type""")
    assert rows == []
