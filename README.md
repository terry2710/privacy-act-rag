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
- A repeatable five-case retrieval benchmark reporting Hit@4 and mean reciprocal rank (MRR).
- Structured Q&A telemetry in Amazon CloudWatch with latency and token usage.
- A Gradio application deployed on Hugging Face Spaces.

## Request flow

1. Restore the FAISS index from S3 or load the local cache.
2. Embed the question and retrieve the four nearest chunks.
3. Send only the retrieved context and question to the generation model.
4. Return the answer, evidence, and retrieval diagnostics.
5. Emit a structured best-effort telemetry event without blocking the response.

## Retrieval evaluation

Open the **Benchmark** tab and run the labeled test set. Each case identifies an expected
Privacy Act provision and records its first retrieved rank.

- **Hit@4**: proportion of cases where an expected provision appears in the top four chunks.
- **MRR**: mean reciprocal rank of the first relevant chunk; higher values reward better ordering.

The benchmark performs retrieval only. It does not invoke the answer-generation model.

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
