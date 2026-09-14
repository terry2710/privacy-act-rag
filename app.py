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


def answer_question(question):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", []

    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return "AWS configuration is missing: " + ", ".join(missing), "", []

    try:
        answer, details = rag_backend.prv_rag_response(
            index=get_index(),
            question=question,
            session_id=uuid.uuid4().hex[:12],
            return_details=True,
        )
        evidence, rows = format_evidence(details.get("chunks", []))
        return answer, evidence, rows
    except Exception as exc:
        return f"Request failed: {type(exc).__name__}: {exc}", "", []


def run_retrieval_benchmark():
    required = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return "AWS configuration is missing: " + ", ".join(missing), []

    try:
        summary, results = rag_evaluation.evaluate_retrieval(get_index())
    except Exception as exc:
        return f"Benchmark failed: {type(exc).__name__}: {exc}", []

    headline = (
        f"## Hit@4: {summary['hits']}/{summary['cases']} ({summary['hit_rate']:.0%})"
        f"\n\n**MRR:** {summary['mrr']:.3f}"
    )
    rows = [
        [
            result["id"],
            result["result"],
            result["first_rank"],
            result["reciprocal_rank"],
            result["top_cosine"],
            result["question"],
        ]
        for result in results
    ]
    return headline, rows


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

    with gr.Tabs():
        with gr.Tab("Answer"):
            answer = gr.Markdown()
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
                headers=["Case", "Result", "First rank", "Reciprocal rank", "Top cosine", "Question"],
                datatype=["str", "str", "number", "number", "number", "str"],
                value=[],
                interactive=False,
            )

    outputs = [answer, evidence, diagnostics]
    ask_button.click(answer_question, inputs=question, outputs=outputs)
    question.submit(answer_question, inputs=question, outputs=outputs)
    benchmark_button.click(
        run_retrieval_benchmark,
        outputs=[benchmark_summary, benchmark_results],
    )


if __name__ == "__main__":
    demo.launch()
