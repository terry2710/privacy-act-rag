# Current State

Date: 2026-10-14. Describes what exists in the repository today, not the target design.

## Application

- Backend logic lives in rag_backend.py. The function prv_rag_response(index, question, k, session_id, return_details) is independent of any UI.
- Two front ends share this backend: a Gradio app (app.py, Hugging Face Space) and a Streamlit app (rag_frontend.py).
- Retrieval: FAISS dense search with an optional BM25 hybrid reranker (rag_retrieval.py).
- Embedding providers: Amazon Titan Embed v2 via Bedrock (default), BAAI/bge-base-en-v1.5, and the fine-tuned greecehalf/bge-base-privacy-act-ft.
- Generation: Amazon Nova Pro via Bedrock.
- Index cache: local directory namespaced by provider and model id, with optional S3 backup and restore.

## Observability and feedback

- rag_logging.py emits structured request records and feedback events, with an optional CloudWatch sink.
- feedback_review.py and promote_feedback.py turn production feedback into candidate evaluation cases.

## Evaluation

- Retrieval evaluation and comparison: rag_evaluation.py, data/retrieval_eval.json.
- Regression gate: check_eval_regression.py against data/eval_baseline.json, run in CI with open embeddings and no AWS calls.

## Tests and delivery

- Test files: 14 under tests/, with markers separating AWS-backed and index-dependent tests from offline tests.
- CI: compile check, offline tests, retrieval regression gate, then push to the Hugging Face Space on main.
- Deployment credential: a long-lived Hugging Face token stored as a repository secret.

## Not present

- No HTTP API layer, authentication, database, TypeScript front end, container image, Kubernetes manifests or infrastructure as code.

## Open questions

- Size and provenance of the evaluation set, and any overlap with the LoRA project test set.
- Contents of the S3 index bucket.
- Streamlit deployment configuration.
