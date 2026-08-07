"""Drives the real Streamlit frontend headlessly.

Regression guard for the boot failure where `"key" in st.secrets` raised FileNotFoundError with
no secrets.toml present, killing the script before any widget rendered - so the app produced
neither answers nor logs, with nothing in the log group to explain why.
"""
import pytest

pytestmark = [pytest.mark.aws, pytest.mark.index]

QUESTION = "What is a credit reporting body?"


@pytest.fixture(scope="module")
def app_run(aws, index_dir, tmp_path_factory):
    """Boot rag_frontend.py, ask one question, return the AppTest plus what got logged."""
    import json
    import os
    from streamlit.testing.v1 import AppTest
    from conftest import PROJECT_ROOT, reload_logging

    log_file = tmp_path_factory.mktemp("app") / "qa.jsonl"
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)   # the app resolves faiss_index/ relative to the working directory
    try:
        with reload_logging(PRV_LOG_GROUP="", PRV_QA_LOG_FILE=str(log_file)):
            at = AppTest.from_file(os.path.join(PROJECT_ROOT, "rag_frontend.py"),
                                   default_timeout=600)
            at.run()
            boot_exception = list(at.exception)
            at.text_area[0].set_value(QUESTION)
            at.button[0].click().run()
            events = [json.loads(l) for l in
                      log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    finally:
        os.chdir(cwd)
    return at, boot_exception, events


def test_app_boots_without_secrets_toml(app_run):
    _, boot_exception, _ = app_run
    assert boot_exception == [], f"the app raised on startup: {boot_exception}"


def test_widgets_render(app_run):
    at, _, _ = app_run
    assert len(at.text_area) == 1, "the question box did not render"
    assert len(at.button) == 1


def test_asking_a_question_does_not_raise(app_run):
    at, _, _ = app_run
    assert list(at.exception) == []


def test_an_answer_is_displayed(app_run):
    at, _, _ = app_run
    rendered = " ".join(m.value for m in at.markdown)
    assert QUESTION not in rendered or len(rendered) > 200
    assert len(rendered.strip()) > 100, "no answer was written to the page"


def test_session_id_is_a_generated_hex(app_run):
    at, _, _ = app_run
    session_id = at.session_state["session_id"]
    assert len(session_id) == 12
    int(session_id, 16)   # must be hex


def test_the_turn_was_logged_under_the_app_session(app_run):
    at, _, events = app_run
    assert len(events) == 1, f"expected one logged turn, got {len(events)}"
    e = events[0]
    assert e["question"] == QUESTION
    assert e["session_id"] == at.session_state["session_id"], \
        "the log must be tied to the browser session, not a hardcoded id"
    assert e["schema"] == "qa/2"
    assert e["chunk_count"] > 0
    assert e["answer_chars"] > 0


def test_set_page_config_is_the_first_streamlit_call():
    """Reading st.secrets emits a Streamlit command, so set_page_config must precede it."""
    import ast
    import os
    from conftest import PROJECT_ROOT

    # Parsed rather than string-searched: the surrounding comments mention st.secrets too.
    tree = ast.parse(open(os.path.join(PROJECT_ROOT, "rag_frontend.py"), encoding="utf-8").read())
    page_config = [n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "set_page_config"]
    secrets = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Attribute) and n.attr == "secrets"
               and isinstance(n.value, ast.Name) and n.value.id == "st"]

    assert page_config, "rag_frontend.py no longer calls st.set_page_config"
    assert secrets, "rag_frontend.py no longer reads st.secrets"
    assert min(page_config) < min(secrets), \
        "set_page_config must come before the st.secrets read or Streamlit rejects it"
