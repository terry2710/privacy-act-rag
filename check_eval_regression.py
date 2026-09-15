"""Free-tier CI regression gate for retrieval quality.

Builds the open embedding index (BGE, CPU, no AWS calls at all) from the public Privacy Act
PDF, runs the same evaluate_retrieval() the app's own Benchmark tab uses against
data/retrieval_eval.json, and fails if headline metrics regress beyond tolerance versus
data/eval_baseline.json.

This is what closes roadmap step 1.2's loop ("production feedback -> regression test -> CI
gate"): promote_feedback.py adds a case for something a real user hit in production, and from
that point on every pull request is checked against it here - for free - before it can reach
main/production.

Forces PRV_EMBEDDING_PROVIDER=bge and disables S3 regardless of ambient environment, so this
script can never accidentally rack up Bedrock or S3 cost, whatever is configured in the
running shell or CI secrets.

Usage:
    python check_eval_regression.py                   # compare against data/eval_baseline.json
    python check_eval_regression.py --update-baseline  # after a deliberate, reviewed change

Exit codes: 0 = no regression, 1 = regression, 2 = setup/config error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "data" / "eval_baseline.json"

# Small absolute tolerance: given a fixed model + dataset, BGE/CPU results are deterministic,
# so any drop at all usually means either a real regression or the dataset changed - not noise.
# This only absorbs floating point / library version jitter, never a real miss.
HIT_RATE_TOLERANCE = 0.02
MRR_TOLERANCE = 0.02


def load_baseline(path=BASELINE_PATH):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_baseline(summary, provider, model_id, dataset_version, path=BASELINE_PATH):
    Path(path).write_text(json.dumps({
        "provider": provider,
        "model_id": model_id,
        "dataset_version": dataset_version,
        "cases": summary["cases"],
        "hit_rate": summary["hit_rate"],
        "mrr": summary["mrr"],
    }, indent=2) + "\n", encoding="utf-8")


def check_regression(summary, baseline, fb_miss_ids,
                      hit_rate_tolerance=HIT_RATE_TOLERANCE, mrr_tolerance=MRR_TOLERANCE):
    """Pure comparison: -> list of failure message strings (empty = pass).

    Every case whose id starts with "fb-" (i.e. promoted from real production feedback) must
    always pass, with no tolerance: that is the entire point of the loop, and an aggregate-score
    tolerance could otherwise silently hide exactly the regression such a case exists to catch.
    """
    failures = []
    if summary["hit_rate"] < baseline["hit_rate"] - hit_rate_tolerance:
        failures.append(f"hit_rate regressed: {summary['hit_rate']:.3f} < baseline "
                         f"{baseline['hit_rate']:.3f} - {hit_rate_tolerance}")
    if summary["mrr"] < baseline["mrr"] - mrr_tolerance:
        failures.append(f"mrr regressed: {summary['mrr']:.3f} < baseline "
                         f"{baseline['mrr']:.3f} - {mrr_tolerance}")
    if fb_miss_ids:
        failures.append(f"{len(fb_miss_ids)} production-feedback case(s) regressed: {fb_miss_ids}")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true",
                         help="write the freshly-measured metrics as the new baseline instead of checking against it")
    args = parser.parse_args(argv)

    # Forced, not setdefault: this gate must never run against a paid provider or touch S3,
    # no matter what is already set in the shell / CI secrets.
    os.environ["PRV_EMBEDDING_PROVIDER"] = "bge"
    os.environ["PRV_S3_BUCKET"] = ""
    os.environ.setdefault("PRV_LOG_GROUP", "")

    import rag_backend
    import rag_evaluation

    if rag_backend.EMBEDDING_PROVIDER != "bge":
        print(f"refusing to run: provider={rag_backend.EMBEDDING_PROVIDER!r} (expected 'bge' - "
              "this gate must stay free of AWS cost)", file=sys.stderr)
        return 2

    print(f"Building {rag_backend.EMBEDDING_PROVIDER} index "
          f"(model={rag_backend.OPEN_EMBEDDING_MODEL_ID})...", file=sys.stderr)
    index = rag_backend.prv_index(progress_callback=lambda stage, done, total: print(
        f"  {stage}" + (f" ({done}/{total})" if total else ""), file=sys.stderr))

    dataset = rag_evaluation.load_dataset()
    summary, rows = rag_evaluation.evaluate_retrieval(index, cases=dataset["cases"])

    print(f"\nHit@4: {summary['hits']}/{summary['cases']} ({summary['hit_rate']:.1%})   "
          f"MRR: {summary['mrr']:.3f}   dataset_version={dataset.get('dataset_version')}")

    misses = [r for r in rows if r["result"] == "Miss"]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            tag = " [production feedback]" if r["id"].startswith("fb-") else ""
            print(f"  - {r['id']}{tag} (source {r['source_section']}): {r['question']}")

    if args.update_baseline:
        save_baseline(summary, rag_backend.EMBEDDING_PROVIDER, rag_backend.OPEN_EMBEDDING_MODEL_ID,
                      dataset.get("dataset_version"))
        print(f"\nWrote new baseline to {BASELINE_PATH}.", file=sys.stderr)
        return 0

    baseline = load_baseline()
    if baseline is None:
        print(f"\nNo baseline at {BASELINE_PATH} yet - writing this run's metrics as the first one.",
              file=sys.stderr)
        save_baseline(summary, rag_backend.EMBEDDING_PROVIDER, rag_backend.OPEN_EMBEDDING_MODEL_ID,
                      dataset.get("dataset_version"))
        return 0

    fb_miss_ids = [r["id"] for r in misses if r["id"].startswith("fb-")]
    failures = check_regression(summary, baseline, fb_miss_ids)

    if failures:
        print("\nREGRESSION:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nBaseline: hit_rate={baseline['hit_rate']:.3f} mrr={baseline['mrr']:.3f} "
              f"(dataset_version={baseline.get('dataset_version')})", file=sys.stderr)
        print("If this drop is expected and reviewed, re-run with --update-baseline.",
              file=sys.stderr)
        return 1

    print(f"\nOK - within tolerance of baseline (hit_rate={baseline['hit_rate']:.3f}, "
          f"mrr={baseline['mrr']:.3f}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
