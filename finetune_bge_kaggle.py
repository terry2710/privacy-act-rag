"""Roadmap step 1.3: contrastive fine-tuning of the open embedding model on Kaggle's free GPU.

Run this in a Kaggle Notebook, NOT locally and NOT in any Claude sandbox - it needs a GPU
(Settings -> Accelerator -> GPU T4 x2 or P100) and outbound internet (Settings -> Internet -> On)
to install sentence-transformers, clone this repo for its training data, and optionally push the
result to the Hugging Face Hub. A CPU-only fine-tune of a 109M-parameter model, even on 36
examples, is unnecessarily slow next to a free Kaggle GPU session - this is why the roadmap
explicitly chose Kaggle over the local Mac for this step.

What this trains on and why:
    data/finetune_confirmed.json (36 (question, passage) pairs, all review_status="approved")
    is used here - and ONLY here. data/retrieval_eval.json (the 25-case benchmark that produced
    the Titan 88.0%/0.670 and base-BGE 64.0%/0.450 numbers already on record) must stay untouched
    by training so the post-fine-tune Hit@4/MRR comparison against those two numbers stays a fair,
    uncontaminated comparison. Do not add retrieval_eval.json cases here even to get more data.

Why no query instruction prefix:
    BAAI/bge-base-en-v1.5 is usually used with a query-side instruction prefix
    ("Represent this sentence for searching relevant passages: "). This script deliberately does
    NOT add one, because rag_backend.py's _make_embeddings() calls plain
    langchain_huggingface.HuggingFaceEmbeddings with no query_instruction configured - production
    embeds queries with the bare question text. Training with a prefix production never sends
    would fine-tune the model for an input distribution it will never actually see at inference
    time. If a future step adds the instruction prefix to rag_backend.py, this script's PREFIX
    constant should be updated to match, and the model re-trained.

Usage (inside a Kaggle notebook cell):
    !git clone --depth 1 https://github.com/terry2710/privacy-act-rag.git repo
    !pip install -q sentence-transformers
    !python repo/finetune_bge_kaggle.py --data repo/data/finetune_confirmed.json --out /kaggle/working/bge-privacy-act-ft

    # Optional: push straight to the Hugging Face Hub from Kaggle (avoids downloading the model
    # and re-uploading from a laptop). Add HF_TOKEN as a Kaggle secret first (a token with write
    # access to the greecehalf namespace), then:
    !python repo/finetune_bge_kaggle.py --data repo/data/finetune_confirmed.json \\
        --out /kaggle/working/bge-privacy-act-ft --push-to-hub greecehalf/bge-base-privacy-act-ft

After training, evaluation happens back on the local Mac (it already has langchain + the
retrieval_eval.json harness): point PRV_OPEN_EMBEDDING_MODEL_ID at the pushed Hub repo id (or a
downloaded local path) and re-run compare_embeddings.py. That is a separate, later step - this
script's only job is producing the fine-tuned model.
"""
import argparse
import json
import sys
from pathlib import Path


# Deliberately empty - see "Why no query instruction prefix" above. Kept as a named constant
# (rather than just hardcoding "") so a future change is a one-line diff with a clear reason
# attached, not a silent edit buried in the training loop.
QUERY_PREFIX = ""

BASE_MODEL = "BAAI/bge-base-en-v1.5"

# Small dataset (36 examples), so conservative hyperparameters: a low learning rate and few
# epochs to adapt the model toward this domain's vocabulary and numbered-section phrasing
# without catastrophically forgetting what BGE already knows about general-purpose retrieval.
# There is no internal validation split - see the module docstring for why (retrieval_eval.json
# is the only benchmark, and it must not be touched by anything at training time, including for
# a "best epoch" selection).
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.1


def load_training_pairs(data_path):
    """Return (question, passage_text) tuples for every approved confirmed candidate.

    Raises if any confirmed entry is not review_status == "approved" - training on a candidate
    nobody signed off on would defeat the point of the review step in prepare_finetune_data.py /
    the human-in-the-loop check that came after it.
    """
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    confirmed = data["confirmed"]
    if not confirmed:
        raise ValueError(f"no confirmed candidates in {data_path}")

    pairs = []
    not_approved = []
    for cand in confirmed:
        if cand.get("review_status") != "approved":
            not_approved.append(cand["id"])
            continue
        pairs.append((cand["question"], cand["passage_text"], cand["id"]))

    if not_approved:
        raise ValueError(
            f"{len(not_approved)} confirmed candidate(s) are not review_status='approved', "
            f"refusing to train until reviewed: {not_approved}"
        )
    return pairs


def build_examples(pairs):
    from sentence_transformers import InputExample

    examples = []
    for question, passage_text, cand_id in pairs:
        examples.append(InputExample(texts=[QUERY_PREFIX + question, passage_text]))
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/finetune_confirmed.json",
                         help="path to the confirmed training pairs (default: %(default)s)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--out", default="./bge-privacy-act-ft",
                         help="local directory to save the fine-tuned model to")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", metavar="REPO_ID", default=None,
                         help="e.g. greecehalf/bge-base-privacy-act-ft - pushes the trained model "
                              "to this Hub repo (needs HF_TOKEN set, e.g. via a Kaggle secret). "
                              "Omit to only save locally to --out.")
    args = parser.parse_args(argv)

    import torch
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found - this will be slow. On Kaggle, check "
              "Settings -> Accelerator is set to a GPU.", file=sys.stderr)
    print(f"device={device}", file=sys.stderr)

    pairs = load_training_pairs(args.data)
    print(f"Loaded {len(pairs)} approved training pairs from {args.data}", file=sys.stderr)
    for question, _, cand_id in pairs:
        print(f"  - {cand_id}: {question}", file=sys.stderr)

    examples = build_examples(pairs)

    print(f"Loading base model {args.base_model} ...", file=sys.stderr)
    model = SentenceTransformer(args.base_model, device=device)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"Training: {args.epochs} epoch(s), batch_size={args.batch_size}, lr={args.lr}, "
          f"warmup_steps={warmup_steps}/{total_steps}", file=sys.stderr)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=args.out,
    )
    print(f"\nSaved fine-tuned model to {args.out}", file=sys.stderr)

    if args.push_to_hub:
        print(f"Pushing to Hugging Face Hub: {args.push_to_hub} ...", file=sys.stderr)
        model.push_to_hub(args.push_to_hub, exist_ok=True)
        print(f"Pushed. Set PRV_OPEN_EMBEDDING_MODEL_ID={args.push_to_hub} locally to evaluate it "
              f"against data/retrieval_eval.json.", file=sys.stderr)
    else:
        print(f"\nNot pushed to the Hub (no --push-to-hub given). Download {args.out}/ from the "
              f"Kaggle output, or re-run with --push-to-hub <namespace>/<name> to publish "
              f"directly from this notebook.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
