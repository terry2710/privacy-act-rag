"""Promote a human-reviewed feedback item into the retrieval regression dataset.

Roadmap step 1.2, second half of the loop: feedback_review.py pulls production "not helpful"
turns into data/feedback_queue.json for review. Once a human has read the question/answer/
retrieved chunks, decided which section *should* have been retrieved, and written
expected_terms that pin down a correct retrieval, this script appends that case into
data/retrieval_eval.json so check_eval_regression.py (and every future embedding/model change)
is tested against it forever after.

Workflow:
    1. python feedback_review.py                     # writes/updates data/feedback_queue.json
    2. Open data/feedback_queue.json; for each item worth turning into a regression case, edit
       it in place:
           "review_status": "promoted",
           "category": "production-feedback",      # or a more specific category
           "source_section": "APP 6",               # the section that should have been retrieved
           "expected_terms": ["a distinctive phrase from that section"],
       Set review_status to "dismissed" for feedback that isn't a real retrieval defect
       (ambiguous question, user error, etc.) so it stops showing as pending.
    3. python promote_feedback.py                    # appends promoted items, bumps
                                                      #   dataset_version, marks the queue item
                                                      #   "promoted_committed" so a rerun never
                                                      #   double-appends it.
    4. Run check_eval_regression.py to confirm the new case passes, then commit both files.

Deliberately dependency-free (stdlib only) so reviewing/promoting never requires the ML venv.
"""
import argparse
import json
import sys
from pathlib import Path

QUEUE_PATH = Path(__file__).parent / "data" / "feedback_queue.json"
DATASET_PATH = Path(__file__).parent / "data" / "retrieval_eval.json"
REQUIRED_ANNOTATIONS = ("category", "source_section", "expected_terms")
REQUIRED_CASE_FIELDS = {"id", "category", "question", "source_section", "expected_terms"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bump_version(version):
    """"1.0" -> "1.1". Falls back to appending a suffix for any version string we don't parse
    as "<major>.<minor>", rather than raising - a promotion should never be blocked by this."""
    try:
        major, minor = str(version).split(".", 1)
        return f"{major}.{int(minor) + 1}"
    except (ValueError, AttributeError):
        return f"{version}-fb"


def validate_dataset(dataset):
    """Mirrors rag_evaluation.load_dataset()'s checks, plus a duplicate-id guard, so a bad
    promotion is caught here rather than at the next `pytest` or app startup."""
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must contain a non-empty cases list")
    seen_ids = set()
    for case in cases:
        missing = REQUIRED_CASE_FIELDS.difference(case)
        if missing:
            raise ValueError(f"case {case.get('id', '<unknown>')} is missing {sorted(missing)}")
        if not case["expected_terms"]:
            raise ValueError(f"case {case['id']} has no expected terms")
        if case["id"] in seen_ids:
            raise ValueError(f"duplicate case id {case['id']}")
        seen_ids.add(case["id"])


def promote(queue, dataset):
    """Pure transform: (queue dict, dataset dict) -> (dataset, queue, messages, promoted_count).

    Only mutates review_status to "promoted_committed" on items it actually appends; anything
    "pending", "dismissed", or already "promoted_committed" is left untouched.
    """
    existing_ids = {case["id"] for case in dataset["cases"]}
    messages = []
    promoted = 0

    for item in queue.get("items", []):
        if item.get("review_status") != "promoted":
            continue

        case_id = f"fb-{item['request_id'][:8]}"
        if case_id in existing_ids:
            item["review_status"] = "promoted_committed"
            continue

        missing = [f for f in REQUIRED_ANNOTATIONS if not item.get(f)]
        if not item.get("question"):
            missing.append("question (no matching qa/2 record - re-run feedback_review.py "
                            "with a larger --days)")
        if missing:
            messages.append(f"skipping {item['request_id']}: marked promoted but missing "
                             f"{missing}")
            continue

        dataset["cases"].append({
            "id": case_id,
            "category": item["category"],
            "question": item["question"],
            "source_section": item["source_section"],
            "expected_terms": item["expected_terms"],
            "origin": "production_feedback",
            "origin_request_id": item["request_id"],
            "promoted_at": queue.get("generated_at"),
        })
        existing_ids.add(case_id)
        item["review_status"] = "promoted_committed"
        promoted += 1
        messages.append(f"promoted {item['request_id']} -> {case_id} ({item['source_section']})")

    if promoted:
        dataset["dataset_version"] = bump_version(dataset.get("dataset_version", "1.0"))
        validate_dataset(dataset)

    return dataset, queue, messages, promoted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue-file", default=str(QUEUE_PATH))
    parser.add_argument("--dataset-file", default=str(DATASET_PATH))
    args = parser.parse_args(argv)

    queue_path = Path(args.queue_file)
    dataset_path = Path(args.dataset_file)

    if not queue_path.exists():
        print(f"No {queue_path} - run feedback_review.py first.", file=sys.stderr)
        return 1

    queue = load_json(queue_path)
    dataset = load_json(dataset_path)

    dataset, queue, messages, promoted = promote(queue, dataset)
    for m in messages:
        print(m, file=sys.stderr)

    if promoted:
        save_json(dataset_path, dataset)
        save_json(queue_path, queue)
        print(f"\nAppended {promoted} case(s) to {dataset_path} "
              f"(dataset_version -> {dataset['dataset_version']}).", file=sys.stderr)
        print("Run check_eval_regression.py to confirm the new case passes before committing.",
              file=sys.stderr)
    else:
        print("Nothing to promote (no items with review_status=\"promoted\" and all required "
              "annotations filled in).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
