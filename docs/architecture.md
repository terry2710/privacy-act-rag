# Architecture

Target architecture for the full-stack upgrade. Each component carries a status, and nothing is claimed as working until its status says so.

## Components

| Component | Role | Status |
|---|---|---|
| Retrieval and generation core (rag_backend.py) | FAISS retrieval, Bedrock embeddings and generation | Built, pre-existing |
| HTTP API (api/) | Query and feedback endpoints, request IDs, health probes | Built, offline tests passing |
| Database (Postgres) | Documents, queries, feedback, reports | Models defined, migrations pending |
| Authentication (Cognito) | User identity, ownership checks | Planned |
| Ingestion (S3, SQS, worker) | Asynchronous document parsing and indexing | Skeleton only |
| Compliance agent (LangGraph, MCP) | Clause-by-clause analysis with citations | Planned |
| Web front end (React, TypeScript) | Upload, progress, streamed answers, citations | Scaffold and typed client only |
| Containers and Kubernetes | kind for development, EKS for the demo release | Manifests drafted |
| Infrastructure as code (Terraform) | ECR now, cluster and identity later | Skeleton only |

## Request path

Browser to the ingress, then to the API for /api and /health or to the front end for everything else. The API validates the request, calls the retrieval and generation core, and returns an answer with citations and a request ID. The same ID links the response, the structured log record and any later feedback.

## Boundaries

The retrieval core stays at the repository root and is imported by the API and the worker. Moving it into a package is deliberately postponed so the Hugging Face deployment keeps working unchanged.
