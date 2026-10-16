# Runbook

## Start locally

1. Copy .env.example to .env and set values.
2. Start Postgres and the local queue with the helper script.
3. Start the API with the helper script and check the liveness and readiness endpoints.

## Trace a failed request

1. Take the x-request-id response header, or the request_id returned in a successful response.
2. Search the structured Q&A log, in CloudWatch or the local log file, for that identifier.
3. A record with event qa_error names the failing stage, retrieval or llm, and the error type.

Known limitation: a failed query does not return the backend request identifier, so the response header is the only handle until the backend accepts a caller-supplied identifier.


## Index not ready

Symptom: /health/ready returns 503 and queries return 503 with error index_not_ready.

1. The API loads the index in the background at startup, from the local cache or from S3. It never builds one on a request.
2. Check whether a cached index can be loaded with prag build-index --check.
3. If none is available, build it once with prag build-index. This calls the embedding provider for every chunk and uploads the result to S3 when PRV_S3_BUCKET is set.
4. A running API retries loading on the next query, so a restart is not required once the index exists.



## Local Kubernetes (kind)

Run in this order on a fresh machine:

1. prag kind-up creates the cluster.
2. prag k-secrets creates the namespace, the database URL secret and a secret holding the local AWS credentials file.
3. prag k-deploy builds the image, tags it with the git revision, loads it into kind and rolls it out.
4. prag k-migrate runs the Alembic migration as a Job. The API starts without it because persistence is best effort.
5. prag k-forward exposes the API on localhost; prag live shows the deployed version and prag ready turns ready once the index has loaded from S3.

prag k-status and prag k-logs show the state. prag kind-down deletes the cluster and everything in it.

The AWS credentials secret is for local use only. On EKS the API uses an IAM role through IRSA and no credentials are stored in the cluster.
