# Privacy Act Q&A with RAG 🎯

A Streamlit app that answers questions about the **Australian Privacy Act 1988** using
retrieval-augmented generation over the official legislation PDF, with Amazon Bedrock for
embeddings and generation and FAISS as the vector store.

Every question and answer is logged to CloudWatch as structured JSON, so retrieval quality,
latency and token spend can be inspected in Logs Insights.

---

## How it works

```
legislation.gov.au PDF
        │  PyPDFLoader
        ▼
  RecursiveCharacterTextSplitter        chunk_size=1500, overlap=200
        │                                → 860 chunks
        ▼
  Bedrock Titan Embeddings v2           1024-dim vectors
        │
        ▼
  FAISS index  ──► cached to ./faiss_index  ──► (optional) backed up to S3
        │
        │  similarity_search_with_score(question, k=4)
        ▼
  top-4 chunks ──► prompt ──► Bedrock Nova Pro ──► answer
        │
        └──────────────────► rag_logging ──► CloudWatch Logs (JSON)
```

Retrieval is invoked explicitly rather than piped through an LCEL chain, so the retrieved
chunks and their similarity scores are available to the logger.

## Repo layout

| File | Purpose |
|---|---|
| `rag_frontend.py` | Streamlit UI — index bootstrap with a progress bar, question box, answer pane |
| `rag_backend.py` | Index build/load/cache, Bedrock clients, the RAG call itself |
| `rag_logging.py` | Structured JSON logging of each Q&A turn to CloudWatch |
| `requirements.txt` | Python dependencies |
| `data_load_test.py`, `data_split_test.py`, `diag_index.py` | Early scratch/diagnostic scripts from building the pipeline. Not used by the app and not kept up to date (`diag_index.py` still references Titan v1). |

## Prerequisites

- Python 3.10+
- An AWS account with **Amazon Bedrock model access** granted in your region for:
  - `amazon.titan-embed-text-v2:0` (embeddings)
  - `amazon.nova-pro-v1:0` (generation)
- IAM permissions:
  - `bedrock:InvokeModel`
  - `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`, `logs:PutRetentionPolicy` — for Q&A logging
  - `s3:GetObject`, `s3:PutObject` on your bucket — only if you enable the S3 index cache

> **Note on the model choice.** The app uses Amazon Nova Pro as a stand-in because Anthropic
> models are pending the "Anthropic use case details" form on this account (Bedrock console →
> Model access). Once approved, change `LLM_MODEL_ID` in `rag_backend.py` to an Anthropic model
> id or inference-profile id.

## Setup

```bash
conda activate aws          # this project runs in the `aws` conda env, not base
pip install -r requirements.txt
```

Credentials come from boto3's default chain — environment variables first, then
`~/.aws/credentials [default]`. No profile name is hardcoded, so the same code works locally
and on Streamlit Cloud.

## Running

```bash
streamlit run rag_frontend.py
```

On first launch the app looks for an index in this order:

1. **`./faiss_index/`** on local disk — instant.
2. **S3**, if `PRV_S3_BUCKET` is set — downloaded to local disk.
3. **Rebuild from the source PDF** — downloads the legislation, splits it, and embeds all 860
   chunks, then saves to disk and uploads to S3.

A rebuild is slow by default: `prv_index()` uses `max_workers=1` with a 1-second pause between
embedding calls, which keeps new/free-tier Bedrock accounts under their TPS quota. Raise
`max_workers` and drop `min_interval` if your quota allows. The Bedrock client also uses
adaptive retries (10 attempts) so throttling backs off rather than failing outright.

`faiss_index/` is gitignored — it is regenerated or restored from S3, never committed.

## Configuration

All settings are environment variables. On Streamlit Community Cloud, set them in
**Settings → Secrets**; `rag_frontend.py` copies the recognised keys into environment
variables at startup so boto3 and the backend pick them up.

| Variable | Default | Meaning |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | Only needed where there is no `~/.aws/credentials` |
| `AWS_DEFAULT_REGION` | `us-east-1` | Region for Bedrock, S3 and CloudWatch |
| `PRV_S3_BUCKET` | *(unset)* | Enables S3 backup/restore of the index. Unset = local cache only |
| `PRV_RETRIEVER_K` | `4` | Chunks retrieved per question |
| `PRV_LOG_GROUP` | `/privacy-act-rag/qa` | CloudWatch log group. Set to `""` to disable logging |
| `PRV_LOG_RETENTION_DAYS` | `30` | Retention applied when the group is created. `0` = never expire |
| `PRV_LOG_CHUNK_CHARS` | `800` | Chars of each retrieved chunk to log. `0` = omit text, `full` = all |
| `PRV_LOG_ANSWER_CHARS` | `full` | Chars of the answer to log |
| `PRV_QA_LOG_FILE` | *(unset)* | Also append each event to this local JSONL file — handy offline and for tests |

## Q&A logging

`rag_logging.py` writes **one JSON object per Q&A turn**, tagged `schema: "qa/2"`. Each event
carries a `request_id`, the Streamlit `session_id` (so one browser session's turns group
together), retrieval/LLM/total timings, token usage, stop reason, and per-chunk rank, `score`
(cosine similarity), `l2_sq` (the raw FAISS distance), page, source and text. Failures are
logged as `event: "qa_error"` with the failing `stage` tagged, then re-raised unchanged.

Logging is **best-effort and asynchronous**: it runs on a background thread, creates the log
group and stream lazily, and swallows every failure after a single stderr warning — a missing
IAM permission degrades logging to a no-op but cannot break the app.

Events are typically 5–7 KB. Chunk text is progressively truncated if an event would approach
the CloudWatch 256 KB per-event cap, and such events are flagged `size_capped: true`.

### Querying in Logs Insights

The JSON fields are discovered automatically — no `parse` statement needed. The `chunks` array
is flattened into indexed fields, so the top-ranked chunk is `chunks.0.score`, the second is
`chunks.1.score`, and so on. (`unnest` does **not** work on these auto-discovered arrays — it
returns empty grouping keys.)

```sql
-- recent turns
fields @timestamp, question, total_ms, chunk_count, input_tokens, output_tokens
| filter event = "qa"
| sort @timestamp desc

-- latency trend
filter event = "qa"
| stats avg(llm_ms) as avg_llm_ms, max(llm_ms) as max_llm_ms by bin(1h)

-- failures by stage
filter event = "qa_error"
| stats count(*) by stage, error_type
```

Retrieval quality, using the cosine `score` of the best-matching chunk:

```sql
-- how well the best chunk matched each question, worst first
fields question, chunks.0.score as best_cosine, chunks.0.page as best_page
| filter event = "qa" and schema = "qa/2"
| sort best_cosine asc

-- questions the corpus answers poorly - candidates for re-chunking or a bigger k
fields question, chunks.0.score as best_cosine
| filter event = "qa" and schema = "qa/2" and chunks.0.score < 0.6
| sort best_cosine asc

-- which chunks get retrieved most, and how strong those matches are
filter event = "qa" and schema = "qa/2"
| stats count(*) as hits, avg(chunks.0.score) as avg_cosine
    by chunks.0.chunk_id as chunk, chunks.0.page as page
| sort hits desc

-- retrieval confidence over time
filter event = "qa" and schema = "qa/2"
| stats avg(chunks.0.score) as avg_best, min(chunks.0.score) as worst_best by bin(1h)

-- score decay across the four retrieved ranks; a flat curve means k could be trimmed
filter event = "qa" and schema = "qa/2"
| stats avg(chunks.0.score) as r1, avg(chunks.1.score) as r2,
        avg(chunks.2.score) as r3, avg(chunks.3.score) as r4
```

**Always filter on `schema = "qa/2"` when aggregating scores.** `qa/1` events logged `score` as
squared L2 distance (lower = better); `qa/2` logs cosine similarity (higher = better). Averaging
across both mixes two incompatible scales.

As a reference point from a live run: an on-topic question scored `0.82` on its best chunk,
while a deliberately off-topic one ("what is the recipe for chocolate cake?") scored `0.13`.

To try the pipeline without touching CloudWatch or IAM, set `PRV_LOG_GROUP=""` and
`PRV_QA_LOG_FILE=/tmp/qa.jsonl` — events go to the local file only.

> **Privacy note.** Questions and answers are logged in full by default. Since users may paste
> personal information into a Privacy Act Q&A box, consider lowering `PRV_LOG_ANSWER_CHARS`, or
> disabling logging entirely, before exposing this publicly. Log retention defaults to 30 days.

## Tests

```bash
pip install -r requirements-dev.txt

pytest                      # offline only - no AWS calls, no cost (~1s)
PRV_TEST_AWS=1 pytest       # everything, including Bedrock and CloudWatch (~1min)
```

AWS-backed tests skip by default, use a disposable `/privacy-act-rag/qa-test/*` log group that
is deleted afterwards, and never rebuild the index. See [`tests/README.md`](tests/README.md).

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub and create an app pointing at `rag_frontend.py`.
2. In **Settings → Secrets**, add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_DEFAULT_REGION`, and `PRV_S3_BUCKET`.
3. Set `PRV_S3_BUCKET` and pre-upload the index. Streamlit Cloud containers start with empty
   disks, so without the S3 cache every cold start would rebuild the index from the PDF.
4. Confirm the deployed credentials carry the `logs:*` permissions above, or Q&A logging will
   silently degrade to a no-op while the app keeps working.

## Notes

- The index is `IndexFlatL2`, which returns **squared** L2 distance. Titan v2 normalises both
  document and query embeddings to unit length, and for unit vectors
  `‖q−d‖² = 2 − 2·cos(q,d)` — so L2 and cosine are an exact monotonic bijection here and rank
  results identically (verified across all 860 vectors: Spearman correlation exactly 1.0).
  The logger therefore reports `score` as cosine similarity (higher = better, `[0,1]` for text)
  and keeps the raw distance alongside as `l2_sq`. If the embedding model is ever swapped for
  one that does not normalise, rebuild with `MAX_INNER_PRODUCT` + `normalize_L2=True` instead.
- The index is loaded with `allow_dangerous_deserialization=True`, which is required to read a
  FAISS pickle. Only load index files you produced yourself.
- The source PDF is a dated compilation of the Privacy Act 1988 (compilation 104, dated
  2026-06-04). Rebuilding against a newer compilation means deleting `faiss_index/` and the S3
  copy, then updating the URL in `rag_backend.py`.

## Credits

The initial frontend scaffold follows the AWS/Streamlit RAG tutorial pattern; index caching,
concurrent embedding, progress reporting and the CloudWatch logging layer were added on top.