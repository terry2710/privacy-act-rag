import os
import threading
import uuid

import gradio as gr
import spaces

import rag_backend
import rag_evaluation


_vector_index = None
_index_lock = threading.Lock()


def get_index():
    global _vector_index

    if _vector_index is None:
        with _index_lock:
            if _vector_index is None:
                _vector_index = rag_backend.prv_index()

    return _vector_index


@spaces.GPU(duration=1)
def zero_gpu_probe():
    return "ready"


def format_evidence(chunks):
    sections = []
    rows = []

    for chunk in chunks:
        page = chunk.get("page")
        display_page = page + 1 if isinstance(page, int) else "Unknown"
        score = chunk.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "Unknown"
        text = chunk.get("text") or "Text omitted from logging configuration."
        sections.append(
            f"### Source {chunk.get('rank')} | PDF page {display_page} | cosine {score_text}\n\n{text}"
        )
        rows.append(
            [chunk.get("rank"), score, display_page, chunk.get("chunk_id"), chunk.get("chars")]
        )

    return "\n\n---\n\n".join(sections), rows


def new_session_id():
    return uuid.uuid4().hex[:12]


def answer_question(question, session_id):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", [], None, ""

    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return "AWS configuration is missing: " + ", ".join(missing), "", [], None, ""

    try:
        session_id = session_id or new_session_id()
        answer, details = rag_backend.prv_rag_response(
            index=get_index(),
            question=question,
            session_id=session_id,
            return_details=True,
        )
        evidence, rows = format_evidence(details.get("chunks", []))
        turn = {
            "request_id": details["request_id"],
            "session_id": session_id,
        }
        return answer, evidence, rows, turn, ""
    except Exception as exc:
        return f"Request failed: {type(exc).__name__}: {exc}", "", [], None, ""


def submit_feedback(turn, rating):
    if not turn or not turn.get("request_id"):
        return "Ask a question before submitting feedback."
    try:
        rag_backend.rag_logging.emit_feedback(
            request_id=turn["request_id"],
            session_id=turn.get("session_id"),
            rating=rating,
        )
        return "Feedback recorded."
    except Exception as exc:
        return f"Feedback failed: {type(exc).__name__}: {exc}"


def run_retrieval_benchmark():
    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return "AWS configuration is missing: " + ", ".join(missing), []

    try:
        dense, hybrid, results = rag_evaluation.compare_retrieval(get_index())
    except Exception as exc:
        return f"Benchmark failed: {type(exc).__name__}: {exc}", []

    headline = (
        f"## Dense: {dense['hits']}/{dense['cases']} Hit@4 ({dense['hit_rate']:.0%}), "
        f"MRR {dense['mrr']:.3f}"
        f"\n\n## Hybrid: {hybrid['hits']}/{hybrid['cases']} Hit@4 "
        f"({hybrid['hit_rate']:.0%}), MRR {hybrid['mrr']:.3f}"
    )
    rows = [
        [
            result["id"],
            result["category"],
            result["source_section"],
            result["dense_result"],
            result["dense_rank"],
            result["hybrid_result"],
            result["hybrid_rank"],
            result["top_cosine"],
            result["question"],
        ]
        for result in results
    ]
    return headline, rows


with gr.Blocks(title="Privacy Act RAG Evaluation Lab") as demo:
    session_id = gr.State()
    current_turn = gr.State()
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

    with gr.Tabs():
        with gr.Tab("Answer"):
            answer = gr.Markdown()
            with gr.Row():
                helpful_button = gr.Button("Helpful", size="sm")
                not_helpful_button = gr.Button("Not helpful", size="sm")
            feedback_status = gr.Markdown()
        with gr.Tab("Evidence"):
            evidence = gr.Markdown()
        with gr.Tab("Diagnostics"):
            diagnostics = gr.Dataframe(
                headers=["Rank", "Cosine similarity", "PDF page", "Chunk ID", "Characters"],
                datatype=["number", "number", "number", "str", "number"],
                value=[],
                interactive=False,
            )
        with gr.Tab("Benchmark"):
            benchmark_button = gr.Button("Run retrieval benchmark", variant="primary")
            benchmark_summary = gr.Markdown()
            benchmark_results = gr.Dataframe(
                headers=["Case", "Category", "Section", "Dense", "Dense rank", "Hybrid", "Hybrid rank", "Top cosine", "Question"],
                datatype=["str", "str", "str", "str", "number", "str", "number", "number", "str"],
                value=[],
                interactive=False,
            )

    outputs = [answer, evidence, diagnostics, current_turn, feedback_status]
    ask_button.click(answer_question, inputs=[question, session_id], outputs=outputs)
    question.submit(answer_question, inputs=[question, session_id], outputs=outputs)
    helpful_button.click(
        lambda turn: submit_feedback(turn, "helpful"),
        inputs=current_turn,
        outputs=feedback_status,
    )
    not_helpful_button.click(
        lambda turn: submit_feedback(turn, "not_helpful"),
        inputs=current_turn,
        outputs=feedback_status,
    )
    benchmark_button.click(
        run_retrieval_benchmark,
        outputs=[benchmark_summary, benchmark_results],
    )
    demo.load(new_session_id, outputs=session_id)


if __name__ == "__main__":
    demo.launch()
