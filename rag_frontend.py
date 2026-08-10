# The below frontend code is provided by AWS and Streamlit. I have only modified it to make it look attractive.
import os
import uuid
import streamlit as st

# Must be the first Streamlit command in the script - reading st.secrets below can emit one,
# which would make set_page_config() fail if it came afterwards.
st.set_page_config(page_title="Privacy Act Q&A with RAG")  ### Modify Heading

# On Streamlit Community Cloud, AWS credentials live in st.secrets (set via the app's Settings ->
# Secrets UI), not in a local ~/.aws/credentials file. Copy them into env vars so boto3's default
# credential chain picks them up.
#
# Locally there is no secrets.toml, and st.secrets is NOT an empty mapping in that case - since
# Streamlit 1.4x even `"key" in st.secrets` raises FileNotFoundError, which would kill the script
# before any UI renders. Read it defensively so the local ~/.aws/credentials [default] profile
# keeps working unchanged.
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}

for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "PRV_S3_BUCKET",
             "PRV_LOG_GROUP", "PRV_LOG_RETENTION_DAYS", "PRV_LOG_CHUNK_CHARS", "PRV_LOG_ANSWER_CHARS",
             "PRV_RETRIEVER_K"):
    if _key in _secrets:
        os.environ[_key] = str(_secrets[_key])  # TOML may type these as int; env vars must be str

# Imported only after the secrets copy above: rag_backend and rag_logging read their
# configuration from environment variables at import time.
import rag_backend as demo  ### replace rag_backend with your backend filename

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Privacy Act Q&A with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)  ### Modify Title

# Ties every Q&A log event from this browser session together in CloudWatch.
if 'session_id' not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if 'vector_index' not in st.session_state:
    status_text = st.empty()
    progress_bar = st.progress(0)

    def _update_progress(stage, done, total):
        if total:
            status_text.text(f"📀 {stage}... ({done}/{total})")
            progress_bar.progress(min(done / total, 1.0))
        else:
            status_text.text(f"📀 {stage}...")

    st.session_state.vector_index = demo.prv_index(progress_callback=_update_progress)  ### Your Index Function name from Backend File
    status_text.empty()
    progress_bar.empty()

input_text = st.text_area("Input text", label_visibility="collapsed")
go_button = st.button("📌Ask me", type="primary")  ### Button Name

if go_button:
    with st.spinner(
            "📢Anytime someone tells me that I can't do something, I want to do it more - Taylor Swift"):  ### Spinner message
        response_content = demo.prv_rag_response(index=st.session_state.vector_index,
                                                question=input_text,
                                                session_id=st.session_state.session_id)  ### replace with RAG Function from backend file
        st.write(response_content)