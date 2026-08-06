# The below frontend code is provided by AWS and Streamlit. I have only modified it to make it look attractive.
import os
import streamlit as st

# On Streamlit Community Cloud, AWS credentials live in st.secrets (set via the app's Settings ->
# Secrets UI), not in a local ~/.aws/credentials file. Copy them into env vars so boto3's default
# credential chain picks them up. Locally there's no secrets.toml, so st.secrets is empty and this
# is a no-op - the existing ~/.aws/credentials [default] profile keeps working unchanged.
for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "PRV_S3_BUCKET"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

import rag_backend as demo  ### replace rag_backend with your backend filename

st.set_page_config(page_title="Privacy Act Q&A with RAG")  ### Modify Heading

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Privacy Act Q&A with RAG 🎯</p>'
st.markdown(new_title, unsafe_allow_html=True)  ### Modify Title

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
                                                question=input_text)  ### replace with RAG Function from backend file
        st.write(response_content)