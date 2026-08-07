# Tests

```bash
conda activate aws
pip install -r requirements-dev.txt

pytest                      # offline only - no AWS calls, no cost
PRV_TEST_AWS=1 pytest       # everything, including Bedrock and CloudWatch
```

## What runs when

| File | Needs | Covers |
|---|---|---|
| `test_logging_offline.py` | nothing | Event schema, cosine conversion, truncation, error paths, the 256 KB size cap. Stubbed index and LLM, fully deterministic. |
| `test_rag_e2e.py` | `PRV_TEST_AWS=1`, `faiss_index/` | A real retrieval + Bedrock turn: token usage, cosine ordering, unit-norm embeddings, on- vs off-topic score separation. |
| `test_cloudwatch.py` | `PRV_TEST_AWS=1` | The sink: group/stream creation, delivery and ordering, retention, and silent degradation on bad credentials. |
| `test_insights_queries.py` | `PRV_TEST_AWS=1` | Every Logs Insights query in the main README, run for real against seeded events. |
| `test_app_boot.py` | `PRV_TEST_AWS=1`, `faiss_index/` | Boots `rag_frontend.py` through Streamlit's `AppTest`, asks a question, checks an answer renders and the turn is logged. |

AWS-backed tests **skip by default** — they call Bedrock and CloudWatch, which costs money.
Setting `PRV_TEST_AWS=1` without working credentials skips them too, with the reason shown.

## Safety

- Tests never touch the app's real log group. They create `/privacy-act-rag/qa-test`
  (override with `PRV_TEST_LOG_GROUP`), set 1-day retention, and delete it afterwards. The
  teardown refuses to delete a group whose name does not end in `-test`.
- `conftest.py` sets `PRV_LOG_GROUP=""` before any project module is imported, so an
  unconfigured test cannot write to CloudWatch by accident.
- Tests never rebuild the FAISS index — that means ~860 Bedrock embedding calls and 15–20
  minutes. Anything needing the index skips if `faiss_index/` is missing.

## Gotchas

`rag_logging` reads `PRV_LOG_*` **at import time** and builds its sink there. Changing that
configuration mid-session requires the `reload_logging()` context manager in `conftest.py`,
which reloads the module, rebinds it on `rag_backend`, and restores the previous state on exit.
Setting the environment variable alone has no effect once the module is imported.
