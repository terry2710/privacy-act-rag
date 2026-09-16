"""Confirm candidate training questions against the real production chunks, before fine-tuning.

Roadmap step 1.3: fine-tuning an embedding model needs (question, positive_passage) pairs where
positive_passage is a real chunk of the source document - not a hand-typed excerpt. Hand-typed
excerpts risk being wrong (secondary sources paraphrase, chunk boundaries differ from where you'd
guess) and using them anyway would mean the model is trained to associate a question with text
that doesn't actually match what a retriever would return in production.

This script loads and splits the Privacy Act PDF exactly the way rag_backend.prv_index() does
(same PyPDFLoader call, same RecursiveCharacterTextSplitter separators/chunk_size/chunk_overlap),
then for each candidate in data/finetune_candidates.json searches the real chunks for any of its
short "anchor" phrases. A candidate is "confirmed" only if at least one real chunk actually
contains one of its anchors - the output pairs it with that chunk's real, verbatim text. A
candidate with zero matches is reported separately rather than silently dropped, so a human can
decide whether the anchor wording was just wrong (fixable) or the topic isn't in this compilation
at all.

Needs only what's already in requirements.txt (langchain-community, langchain-text-splitters,
pypdf) - no heavy ML deps, since this step does no embedding, just text search.

Usage:
    python prepare_finetune_data.py
    python prepare_finetune_data.py --candidates data/finetune_candidates.json --out data/finetune_confirmed.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_URL = "https://www.legislation.gov.au/C2004A03712/2026-06-04/2026-06-04/text/original/pdf"
DEFAULT_CANDIDATES = Path(__file__).parent / "data" / "finetune_candidates.json"
DEFAULT_OUT = Path(__file__).parent / "data" / "finetune_confirmed.json"


def normalize_text(text):
    # Same normalization as rag_evaluation._normalize_text, so a match here means
    # evaluate_retrieval() would also recognise this chunk as a hit for the same anchor.
    return " ".join(text.lower().split())


_TOC_DOT_LEADER = re.compile(r"\.{4,}\s*\d+")


def is_toc_chunk(raw_text):
    """True if a chunk looks like a table-of-contents / index page rather than
    operative text. TOC pages list many section numbers in one place (e.g. "26WA
    Guide to this Part ... 202"), so a short anchor like a bare section number
    ("26wa") matches them at least as well as the real section - and since one TOC
    chunk contains dozens of section numbers, it can out-score the correct chunk for
    several different candidates at once. Found by inspecting confirmed matches:
    15/36 first-pass "confirmed" pairs had landed on TOC chunks instead of the
    actual provision text."""
    return bool(_TOC_DOT_LEADER.search(raw_text)) or raw_text.count("...") >= 3


def load_chunks():
    """Same PDF, same loader, same splitter params as rag_backend.prv_index() - the exact
    chunk boundaries a real retrieval index would have."""
    print(f"Loading PDF from {PDF_URL} ...", file=sys.stderr)
    documents = PyPDFLoader(PDF_URL).load()
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"  {len(documents)} PDF page(s) -> {len(chunks)} chunk(s)", file=sys.stderr)
    return chunks


def find_best_match(chunks, normalized_chunks, anchors):
    """Return (chunk_index, matched_anchors) for the chunk hitting the most anchors, or None.

    Skips chunks that look like a table of contents (see is_toc_chunk) - a TOC entry
    for a section is not the section itself, and letting it win produces a
    confidently-wrong training pair rather than a legitimate "unmatched" candidate."""
    best_index, best_anchors = None, []
    for i, norm_text in enumerate(normalized_chunks):
        if is_toc_chunk(chunks[i].page_content):
            continue
        hits = [a for a in anchors if a in norm_text]
        if len(hits) > len(best_anchors):
            best_index, best_anchors = i, hits
    if best_index is None:
        return None
    return best_index, best_anchors


def confirm_candidates(candidates, chunks):
    normalized_chunks = [normalize_text(c.page_content) for c in chunks]
    confirmed, unmatched = [], []

    for cand in candidates:
        anchors = [normalize_text(a) for a in cand["anchors"]]
        match = find_best_match(chunks, normalized_chunks, anchors)
        if match is None:
            unmatched.append(cand)
            continue
        chunk_index, matched_anchors = match
        chunk = chunks[chunk_index]
        confirmed.append({
            "id": cand["id"],
            "category": cand["category"],
            "source_section": cand["source_section"],
            "question": cand["question"],
            "passage_text": chunk.page_content,
            "page": chunk.metadata.get("page"),
            "chunk_index": chunk_index,
            "matched_anchors": matched_anchors,
            # Filled in by a human during review, same pattern as data/feedback_queue.json.
            "review_status": "pending",  # pending | approved | rejected
            "notes": "",
        })

    return confirmed, unmatched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"No {candidates_path} found.", file=sys.stderr)
        return 1

    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = data["candidates"]

    chunks = load_chunks()
    confirmed, unmatched = confirm_candidates(candidates, chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "source_url": PDF_URL,
        "total_chunks": len(chunks),
        "confirmed": confirmed,
        "unmatched": unmatched,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{len(confirmed)}/{len(candidates)} candidates confirmed against a real chunk, "
          f"written to {out_path}", file=sys.stderr)
    if unmatched:
        print(f"\n{len(unmatched)} candidate(s) found NO matching chunk (anchor wording was "
              f"probably wrong, not necessarily that the topic is absent) - review before "
              f"re-running:", file=sys.stderr)
        for cand in unmatched:
            print(f"  - {cand['id']} ({cand['source_section']}): {cand['question']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
