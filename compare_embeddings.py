"""Standalone comparison: production Titan embeddings vs. an open sentence-transformers model.

Usage:
    python compare_embeddings.py                                  # builds/evaluates the open (bge) index
    PRV_EMBEDDING_PROVIDER=bedrock python compare_embeddings.py    # re-run the Titan path (needs AWS creds)

Prints Hit@4 / MRR for whichever provider is active. When the open model is what's being
evaluated, the already-recorded Titan baseline is printed alongside it so the two numbers land
side by side without re-running both every time (Titan costs real Bedrock API calls; the open
path is free and runs on CPU).

Roadmap: step 1.1, "开源 Embedding 替换与对比评测".
"""
import os
import sys

os.environ.setdefault("PRV_EMBEDDING_PROVIDER", "bge")
os.environ.setdefault("PRV_LOG_GROUP", "")  # no CloudWatch needed for a local comparison run

import rag_backend
import rag_evaluation

# From the project handoff (2026-09-15): dense retrieval baseline over the same 25-case dataset.
RECORDED_BASELINES = {
    "bedrock": {"model": rag_backend.EMBEDDING_MODEL_ID, "hit_rate": 0.88, "mrr": 0.670},
}


def main():
    provider = rag_backend.EMBEDDING_PROVIDER
    model_id = (
        rag_backend.EMBEDDING_MODEL_ID if provider == "bedrock" else rag_backend.OPEN_EMBEDDING_MODEL_ID
    )
    print(f"Building index with provider={provider!r} model={model_id!r} -> {rag_backend.INDEX_DIR}/",
          file=sys.stderr)
    index = rag_backend.prv_index(progress_callback=lambda stage, done, total: print(
        f"  {stage}" + (f" ({done}/{total})" if total else ""), file=sys.stderr))

    summary, rows = rag_evaluation.evaluate_retrieval(index)
    print()
    print(f"=== {provider} ({model_id}) ===")
    print(f"Hit@4: {summary['hits']}/{summary['cases']} ({summary['hit_rate']:.1%})")
    print(f"MRR:   {summary['mrr']:.3f}")

    misses = [r for r in rows if r["result"] == "Miss"]
    if misses:
        print(f"\nMisses ({len(misses)}):")
        for r in misses:
            print(f"  - {r['id']} (source {r['source_section']}): {r['question']}")

    baseline = RECORDED_BASELINES.get("bedrock")
    if baseline and provider != "bedrock":
        print(f"\nRecorded baseline - bedrock ({baseline['model']}): "
              f"Hit@4 {baseline['hit_rate']:.1%}, MRR {baseline['mrr']:.3f}")


if __name__ == "__main__":
    main()
