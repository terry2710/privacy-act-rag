import os
import threading
import uuid

import gradio as gr
import spaces

import rag_backend


_vector_index = None
_index_lock = threading.Lock()


def get_index():
    global _vector_index

    if _vector_index is None:
        with _index_lock:
            if _vector_index is None:
                _vector_index = rag_backend.prv_index()

    return _vector_index


@spaces.GPU(duration=120)
def answer_question(question):
    question = (question or "").strip()
    if not question:
        return "Please enter a question."

    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return "AWS configuration is missing: " + ", ".join(missing)

    try:
        return rag_backend.prv_rag_response(
            index=get_index(),
            question=question,
            session_id=uuid.uuid4().hex[:12],
        )
    except Exception as exc:
        return f"Request failed: {type(exc).__name__}: {exc}"


with gr.Blocks(title="Privacy Act RAG Evaluation Lab") as demo:
    gr.Markdown(
        """
# Privacy Act RAG Evaluation Lab

Ask questions grounded in the Australian Privacy Act 1988.
        """
    )

    question = gr.Textbox(
        label="Question",
        placeholder="What are the Australian Privacy Principles?",
        lines=3,
    )
    ask_button = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")

    ask_button.click(answer_question, inputs=question, outputs=answer)
    question.submit(answer_question, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch()
