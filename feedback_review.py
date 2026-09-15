"""Pull production "not helpful" feedback and join it to its Q&A turn for human review.

Roadmap step 1.2: "生产反馈回灌 + CI 回归门禁" (production feedback -> eval loop -> CI gate).

This is the first half of the loop: it only *reads* CloudWatch Logs (Logs Insights is billed
per GB scanned - for this project's traffic volume that's a fraction of a cent per run, see
README) and never touches data/retrieval_eval.json. A human reviews the resulting queue file
and annotates the cases worth turning into regression tests; promote_feedback.py then commits
those into the eval dataset.

Usage:
    python feedback_review.py                      # last 7 days -> data/feedback_queue.json
    python feedback_review.py --days 30
    python feedback_review.py --log-group /privacy-act-rag/qa --out data/feedback_queue.json

Re-running this script merges rather than replaces: it preserves review_status/notes/
expected_terms a human already filled in, and only appends items for request_ids not already
in the queue file.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import boto3

DEFAULT_LOG_GROUP = os.environ.get("PRV_LOG_GROUP") or "/privacy-act-rag/qa"
DEFAULT_OUT = Path(__file__).parent / "data" / "feedback_queue.json"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def run_query(client, group, query, start_time, end_time, timeout=90):
    qid = client.start_query(
        logGroupName=group, startTime=start_time, endTime=end_time, queryString=query
    )["queryId"]
    deadline = time.time() + timeout
    result = None
    while time.time() < deadline:
        time.sleep(2)
        result = client.get_query_results(queryId=qid)
        if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
    if result is None or result["status"] != "Complete":
        status = result["status"] if result else "no response"
        raise RuntimeError(f"query did not complete ({status}): {query}")
    return [{f["field"]: f["value"] for f in row if f["field"] != "@ptr"} for row in result["results"]]


def fetch_not_helpful(client, group, start_time, end_time):
    return run_query(client, group, """
        fields @timestamp, request_id, session_id, rating
        | filter event = "feedback" and schema = "feedback/1" and rating = "not_helpful"
        | sort @timestamp desc""", start_time, end_time)


def fetch_qa_turns(client, group, request_ids, start_time, end_time):
    """One query covering every request_id, via OR'd equality clauses.

    Deliberately not CloudWatch Logs Insights' `in [...]` operator: an OR chain of plain
    string-equality comparisons is the most conservatively-documented boolean form, and this
    only ever runs over a handful of ids (one per "not helpful" click), so there's no volume
    reason to reach for anything cleverer.
    """
    if not request_ids:
        return {}
    clause = " or ".join(f'request_id = "{rid}"' for rid in request_ids)
    rows = run_query(client, group, f"""
        fields @timestamp, request_id, question, answer, chunk_count, embedding_model_id,
               pipeline_version, chunks.0.chunk_id as top_chunk_id, chunks.0.page as top_page,
               chunks.0.score as top_score
        | filter event = "qa" and schema = "qa/2" and ({clause})""",
        start_time, end_time)
    return {row["request_id"]: row for row in rows}


def build_queue(feedback_rows, qa_by_id):
    items = []
    for row in feedback_rows:
        rid = row.get("request_id")
        if not rid:
            continue
        qa = qa_by_id.get(rid, {})
        items.append({
            "request_id": rid,
            "feedback_ts": row.get("@timestamp"),
            "session_id": row.get("session_id"),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "chunk_count": qa.get("chunk_count"),
            "top_chunk_id": qa.get("top_chunk_id"),
            "top_page": qa.get("top_page"),
            "top_score": qa.get("top_score"),
            "embedding_model_id": qa.get("embedding_model_id"),
            "pipeline_version": qa.get("pipeline_version"),
            "qa_found": rid in qa_by_id,
            # Filled in by a human during review; promote_feedback.py reads these back.
            "review_status": "pending",  # pending | promoted | dismissed | promoted_committed
            "category": None,
            "source_section": None,
            "expected_terms": None,
            "notes": "",
        })
    return items


def merge_queue(existing_items, new_items):
    """Keep every human annotation already on disk; only add genuinely new request_ids."""
    by_id = {item["request_id"]: item for item in existing_items}
    added = 0
    for item in new_items:
        if item["request_id"] not in by_id:
            by_id[item["request_id"]] = item
            added += 1
    merged = sorted(by_id.values(), key=lambda i: i.get("feedback_ts") or "", reverse=True)
    return merged, added


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7, help="lookback window in days (default 7)")
    parser.add_argument("--log-group", default=DEFAULT_LOG_GROUP)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    end_time = int(time.time())
    start_time = end_time - args.days * 86400

    client = boto3.Session(region_name=AWS_REGION).client("logs")

    print(f"Querying {args.log_group!r} for not_helpful feedback in the last {args.days}d...",
          file=sys.stderr)
    feedback_rows = fetch_not_helpful(client, args.log_group, start_time, end_time)
    print(f"  {len(feedback_rows)} not_helpful feedback event(s)", file=sys.stderr)

    request_ids = [r["request_id"] for r in feedback_rows if r.get("request_id")]
    qa_by_id = fetch_qa_turns(client, args.log_group, request_ids, start_time, end_time)
    missing = [rid for rid in request_ids if rid not in qa_by_id]
    if missing:
        print(f"  warning: {len(missing)} feedback event(s) had no matching qa/2 record in "
              f"this window (try a larger --days): {missing}", file=sys.stderr)

    new_items = build_queue(feedback_rows, qa_by_id)

    out_path = Path(args.out)
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8")).get("items", [])
    merged, added = merge_queue(existing, new_items)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "log_group": args.log_group,
        "window_days": args.days,
        "items": merged,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pending = sum(1 for i in merged if i["review_status"] == "pending")
    print(f"\nWrote {len(merged)} item(s) to {out_path} ({added} new, {pending} pending review).",
          file=sys.stderr)
    if pending:
        print("Review each pending item in that file, then set review_status to \"promoted\" "
              "or \"dismissed\" (fill in category/source_section/expected_terms for promoted "
              "items) and run promote_feedback.py.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
