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
