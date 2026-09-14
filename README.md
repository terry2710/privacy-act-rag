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

## Delivery workflow

GitHub is the source of truth. Pull requests and pushes run the offline test suite. A successful
push to `main` is then deployed to the Hugging Face Space by `.github/workflows/ci.yml`.

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
