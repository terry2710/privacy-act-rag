---
title: Privacy Act Rag Evaluation Lab
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
---

# Privacy Act RAG Evaluation Lab

A retrieval-augmented generation application for answering questions against the Australian
Privacy Act 1988. The project exposes the evidence behind each answer and includes a labeled
retrieval benchmark, so retrieval quality can be measured instead of judged by demos alone.

## What it demonstrates

- Amazon Titan embeddings and a FAISS vector index cached in Amazon S3.
- Amazon Nova Pro generation through Amazon Bedrock.
- Source-level evidence with PDF pages, stable chunk IDs, and cosine similarity scores.
- A versioned 25-case retrieval benchmark reporting Hit@4 and mean reciprocal rank (MRR).
- Dense versus BM25 reciprocal-rank-fusion comparison over the same embedded queries.
- Structured Q&A telemetry in Amazon CloudWatch with latency and token usage.
- Answer-level user feedback linked to the originating request for production error analysis.
- GitHub Actions CI/CD that tests every change before deploying `main` to Hugging Face Spaces.
- A Gradio application deployed on Hugging Face Spaces.

## Request flow

1. Restore the FAISS index from S3 or load the local cache.
2. Embed the question and retrieve the four nearest chunks.
3. Send only the retrieved context and question to the generation model.
4. Return the answer, evidence, and retrieval diagnostics.
5. Emit a structured best-effort telemetry event without blocking the response.
6. Capture Helpful/Not helpful feedback using the same request and browser-session identifiers.

## Retrieval evaluation

Open the **Benchmark** tab and run the labeled test set. Each case identifies an expected
Privacy Act provision and records its first retrieved rank. The dataset is stored in
[`data/retrieval_eval.json`](data/retrieval_eval.json) with its source compilation metadata.

- **Hit@4**: proportion of cases where an expected provision appears in the top four chunks.
- **MRR**: mean reciprocal rank of the first relevant chunk; higher values reward better ordering.

The benchmark performs retrieval only. It does not invoke the answer-generation model. Dense
and hybrid results are shown side by side; the production answer path remains dense retrieval
until the benchmark demonstrates that the reranker improves quality without regressions.

### Open-vs-proprietary embedding comparison

`compare_embeddings.py` builds a second FAISS index with an open sentence-transformers model and
runs it through the same 25-case benchmark, so the production Titan embeddings can be compared
against an open alternative without touching the production index or cache:

```bash
pip install -r requirements-ml.txt                             # torch/sentence-transformers - not in requirements.txt
python compare_embeddings.py                                  # builds/evaluates BAAI/bge-base-en-v1.5
PRV_EMBEDDING_PROVIDER=bedrock python compare_embeddings.py    # re-run the Titan path (needs AWS creds)
```

| Embedding | Hit@4 | MRR |
|---|---:|---:|
| `amazon.titan-embed-text-v2:0` (production) | 22/25 (88.0%) | 0.670 |
| `BAAI/bge-base-en-v1.5` (open) | 16/25 (64.0%) | 0.450 |

The open model trails the production baseline on this dataset, and the gap is not spread evenly
across question types. Of the 9 open-model misses, nearly all reference a specific numbered
Australian Privacy Principle (APP 1, APP 3, APP 5, APP 10, APP 11.1, APP 11.2, APP 12) or section
(s 15, the NDB eligible-breach case), while free-text questions with no provision number are
answered correctly at roughly the same rate as Titan. This points to BAAI/bge-base-en-v1.5's
general-purpose pretraining carrying a weaker signal for numbered legal citations specifically,
rather than a broad quality gap - a plausible target for a small contrastive fine-tune on this
project's own labeled cases, rather than a reason to dismiss open embeddings generally.

`PRV_EMBEDDING_PROVIDER` and `PRV_OPEN_EMBEDDING_MODEL_ID` (see `rag_backend.py`) control which
embedding model `prv_index()` builds. The index cache directory is namespaced per provider
(`faiss_index` for Bedrock, `faiss_index_<provider>` otherwise), so switching providers can never
load an index that was built with a different, dimensionally incompatible embedding model.

## Production feedback loop

Every answer has "Helpful" / "Not helpful" buttons (`app.py` -> `rag_backend.rag_logging.emit_feedback`).
"Not helpful" clicks are the project's real-world signal that retrieval or generation went
wrong, so they are turned into permanent regression tests rather than left to sit unread in
CloudWatch:

1. **`feedback_review.py`** queries CloudWatch Logs Insights for recent `not_helpful` feedback
   events and joins each one to its original Q&A turn by `request_id`, writing a human-readable
   queue to `data/feedback_queue.json` (gitignored - it is pulled fresh from production logs
   each run, not committed).

   ```bash
   python feedback_review.py              # last 7 days
   python feedback_review.py --days 30
   ```

2. A human reviews each pending item in that file: reads the question, the answer, and which
   chunk was actually retrieved, decides which Privacy Act section *should* have been retrieved,
   and annotates the item in place:

   ```json
   "review_status": "promoted",
   "category": "production-feedback",
   "source_section": "APP 4",
   "expected_terms": ["destroy or de-identify"]
   ```

   Feedback that turns out not to be a real retrieval defect (an ambiguous question, a one-off)
   is instead set to `"review_status": "dismissed"`.

3. **`promote_feedback.py`** appends every `"promoted"` item into
   [`data/retrieval_eval.json`](data/retrieval_eval.json) as a new case (tagged
   `"origin": "production_feedback"`), bumps `dataset_version`, and marks the queue item
   `"promoted_committed"` so re-running it is idempotent.

   ```bash
   python promote_feedback.py
   ```

4. **`check_eval_regression.py`** is the CI gate this loop feeds: it builds the free, CPU-only
   BGE index (no AWS calls - `PRV_EMBEDDING_PROVIDER`/`PRV_S3_BUCKET` are forced regardless of
   ambient environment), runs the same `evaluate_retrieval()` the Benchmark tab uses against the
   full dataset, and fails the build if aggregate Hit@4/MRR regress beyond a small tolerance
   versus [`data/eval_baseline.json`](data/eval_baseline.json) - *or* if any single case promoted
   from production feedback (`id` prefixed `fb-`) misses at all, with no tolerance. That last
   rule is the point of the loop: a defect a real user hit in production can never silently
   regress again once it has a case.

   ```bash
   python check_eval_regression.py                    # compare against the stored baseline
   python check_eval_regression.py --update-baseline   # after a deliberate, reviewed change
   ```

   `data/eval_baseline.json` is currently seeded from the measured open-embedding comparison
   above (Hit@4 64.0%, MRR 0.450, 25 cases) - the same free BGE path this gate runs in CI.

Cost: Logs Insights bills per GB of log data scanned (roughly USD 0.005/GB as of Sep 2026); at
this project's traffic volume `feedback_review.py`'s two queries scan well under 1 MB per run,
so realistic cost per run is a small fraction of a cent. `check_eval_regression.py` makes no AWS
calls at all.

## Delivery workflow

GitHub is the source of truth. Pull requests and pushes run the offline test suite, then the
free retrieval regression gate described above. A successful push to `main` is then deployed to
the Hugging Face Space by `.github/workflows/ci.yml`.

Add a write-enabled Hugging Face token to the GitHub repository as an Actions secret named
`HF_TOKEN`. After that, normal development only needs a push to GitHub; the workflow performs
the test gate and Space deployment. AWS-backed tests remain opt-in and never run in CI.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`requirements.txt` is the production dependency set - it is what both live deployments install
(this repo backs both the Gradio app on this HF Space and a separate Streamlit Community Cloud
deployment of `rag_frontend.py`, sharing this same `main` branch). The open-embedding tooling
(`compare_embeddings.py`, `check_eval_regression.py`) needs its own extra dependencies -
`sentence-transformers` pulls in `torch`, which is large and irrelevant to either production
app - so those live in `requirements-ml.txt` instead and are never installed by either
deployment:

```bash
pip install -r requirements-ml.txt   # only needed for the open-embedding tooling below
```

Required environment variables:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=us-east-1
PRV_S3_BUCKET=privacy-act-rag-index
```

Keep credentials in environment variables or Hugging Face Space secrets. Never commit them.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Offline tests are the default. AWS-backed integration tests require the explicit
`PRV_TEST_AWS=1` opt-in because they can incur Bedrock and CloudWatch charges. See
[`tests/README.md`](tests/README.md) for the test matrix and safety controls.
