import argparse
import sys

from core import rag_backend

PROGRESS_EVERY = 50


def print_progress(stage, done=None, total=None):
    if total:
        if done not in (0, total) and done % PROGRESS_EVERY:
            return
        print(f"{stage}: {done}/{total}", flush=True)
    else:
        print(stage, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Make the retrieval index available: load it from the local cache or S3, "
        "or build it with the configured embedding provider if neither has one."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether a cached index can be loaded. Never builds.",
    )
    args = parser.parse_args(argv)

    if args.check:
        available = rag_backend.prv_load_cached_index(print_progress) is not None
        print("cached index available" if available else "no cached index")
        return 0 if available else 1

    rag_backend.prv_index(print_progress)
    print(f"index ready in {rag_backend.INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
