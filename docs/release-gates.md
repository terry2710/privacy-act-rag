# Release gates

Defined before measuring, so results cannot be selected after the fact.

1. Offline tests pass, including the API tests.
2. The retrieval regression gate shows no unapproved drop against the committed baseline.
3. Negative authorisation tests pass: unauthenticated access, access to another user's data, forged user identifiers and expired tokens are all rejected.
4. The container image builds and the readiness probe reports ready.
5. Terraform validates and the Helm chart lints cleanly.
6. A rollback to the previous image tag has been performed and verified for the release being shipped.

Latency and cost targets are set only after a baseline has been measured under recorded conditions, and are added to this file with those conditions.
