# Architecture

Target architecture for the full-stack upgrade. Each component carries a status, and nothing is claimed as working until its status says so.

## Components

| Component | Role | Status |
|---|---|---|
| Retrieval and generation core (core/) | FAISS retrieval, Bedrock embeddings and generation | Built, pre-existing |
| HTTP API (api/) | Query and feedback endpoints, request IDs, health probes | Built, offline tests passing |
| Database (Postgres) | Documents, queries, feedback, reports | Initial migration; queries and feedback persisted, best effort |
| Authentication (Cognito) | User identity, ownership checks | Planned |
| Ingestion (S3, SQS, worker) | Asynchronous document parsing and indexing | Skeleton only |
| Compliance agent (LangGraph, MCP) | Clause-by-clause analysis with citations | Planned |
| Web front end (React, TypeScript) | Upload, progress, streamed answers, citations | Scaffold and typed client only |
| Containers and Kubernetes | kind for development, EKS for the demo release | Manifests drafted |
| Infrastructure as code (Terraform) | ECR now, cluster and identity later | Skeleton only |
| Gradio app (demos/gradio) | Always-on public demo on Hugging Face Spaces | Built, pre-existing |
| Streamlit demo (demos/streamlit) | Detached demo on Streamlit Community Cloud, outside the product path | Built, relocated |

## Request path

Browser to the ingress, then to the API for /api and /health or to the front end for everything else. The API validates the request, calls the retrieval and generation core, and returns an answer with citations and a request ID. The same ID links the response, the structured log record and any later feedback.

## Boundaries

The retrieval core lives in the core package and is imported by the API, the worker, both demos and the offline tools. Offline tools live in tools/ and run from the repository root as modules, for example python -m tools.check_eval_regression. Dependency sets live in requirements/, except requirements.txt, which stays at the root because the Hugging Face Space installs it from there.
