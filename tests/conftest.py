"""Shared fixtures and helpers for the test suite.

Importing this module disables CloudWatch logging process-wide before any project module is
imported. rag_logging reads its configuration from the environment at import time, so tests
that want a different configuration must go through reload_logging() below.
"""
import contextlib
import importlib
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set before the first `import rag_logging` anywhere in the session: no test may write to the
# real log group by accident.
os.environ["PRV_LOG_GROUP"] = ""
os.environ.pop("PRV_QA_LOG_FILE", None)

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
# Deliberately not the app's /privacy-act-rag/qa - tests get their own disposable group.
TEST_LOG_GROUP = os.environ.get("PRV_TEST_LOG_GROUP", "/privacy-act-rag/qa-test")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "aws: needs real AWS credentials and spends money; opt in with PRV_TEST_AWS=1")
    config.addinivalue_line("markers", "index: needs a prebuilt faiss_index/ directory")


@pytest.fixture(scope="session")
def aws():
    """Skip unless the caller opted into real AWS calls and credentials actually work."""
    if os.environ.get("PRV_TEST_AWS") != "1":
        pytest.skip("set PRV_TEST_AWS=1 to run tests that call AWS (Bedrock/CloudWatch cost money)")
    import boto3
    try:
        boto3.Session(region_name=AWS_REGION).client("sts").get_caller_identity()
    except Exception as e:
        pytest.skip(f"PRV_TEST_AWS=1 but credentials do not work: {e}")
    return boto3.Session(region_name=AWS_REGION)


@pytest.fixture(scope="session")
def index_dir():
    """Skip if the index is absent - building it takes 15-20 minutes and many Bedrock calls."""
    path = os.path.join(PROJECT_ROOT, "faiss_index")
    if not os.path.isdir(path):
        pytest.skip("no faiss_index/ - run the app once to build it before running index tests")
    return path


@contextlib.contextmanager
def reload_logging(**env):
    """Re-import rag_logging with the given env vars applied, then restore the previous state.

    rag_logging captures PRV_LOG_* at import time and builds its CloudWatch sink there, so
    changing configuration means reloading the module and rebinding it on rag_backend.
    """
    import rag_backend
    import rag_logging

    previous = {k: os.environ.get(k) for k in env}
    old_sink = getattr(rag_logging, "_sink", None)
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        if old_sink is not None:
            old_sink.shutdown()
        importlib.reload(rag_logging)
        rag_backend.rag_logging = rag_logging
        yield rag_logging
    finally:
        new_sink = getattr(rag_logging, "_sink", None)
        if new_sink is not None and new_sink is not old_sink:
            new_sink.shutdown()
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(rag_logging)
        rag_backend.rag_logging = rag_logging


@pytest.fixture
def jsonl_logging(tmp_path):
    """rag_logging configured to write only to a local JSONL file. No AWS involved."""
    log_file = tmp_path / "qa.jsonl"
    with reload_logging(PRV_LOG_GROUP="", PRV_QA_LOG_FILE=str(log_file)) as mod:
        def read():
            import json
            if not log_file.exists():
                return []
            with open(log_file, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        mod.read_events = read
        mod.log_file = log_file
        yield mod


@pytest.fixture(scope="module")
def cw_log_group(aws, request):
    """A disposable CloudWatch log group, one per test module, deleted afterwards.

    Per-module rather than per-session because these tests query by event shape, not by id:
    one module's probe records would otherwise show up in another module's aggregates.
    """
    name = f"{TEST_LOG_GROUP}/{request.module.__name__}"
    client = aws.client("logs")
    try:
        client.create_log_group(logGroupName=name)
    except client.exceptions.ResourceAlreadyExistsException:
        pass
    client.put_retention_policy(logGroupName=name, retentionInDays=1)
    yield name
    # Never delete anything outside the test namespace, however TEST_LOG_GROUP is overridden.
    if "qa-test" in name and name.startswith(TEST_LOG_GROUP):
        with contextlib.suppress(Exception):
            client.delete_log_group(logGroupName=name)
