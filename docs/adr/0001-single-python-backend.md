# ADR 0001: Single Python backend with a TypeScript front end

Status: accepted

## Context

The retrieval, evaluation and logging code is Python and already tested. The front end needs a typed client for a stable API contract.

## Decision

Keep one FastAPI service in Python and build the front end in React with TypeScript. No Node back end is added.

## Consequences

The retrieval core is reused unchanged. The API contract is the only coupling between the two halves, so it needs its own tests and runtime validation on the client.
