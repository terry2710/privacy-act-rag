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
