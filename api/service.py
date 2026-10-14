import threading
import uuid


class RagService:
    def __init__(self):
        self._index = None
        self._lock = threading.Lock()

    def _get_index(self):
        if self._index is None:
            with self._lock:
                if self._index is None:
                    import rag_backend

                    self._index = rag_backend.prv_index()
        return self._index

    def is_ready(self):
        if self._index is not None:
            return True
        import rag_backend

        return rag_backend._index_is_cached()

    def answer(self, question, session_id=None):
        import rag_backend

        session_id = session_id or uuid.uuid4().hex[:12]
        answer, details = rag_backend.prv_rag_response(
            index=self._get_index(),
            question=question,
            session_id=session_id,
            return_details=True,
        )
        return {
            "request_id": details["request_id"],
            "session_id": session_id,
            "answer": answer,
            "chunks": details.get("chunks", []),
            "pipeline_version": details.get("pipeline_version", ""),
        }

    def record_feedback(self, request_id, session_id, rating):
        import rag_backend

        rag_backend.rag_logging.emit_feedback(
            request_id=request_id,
            session_id=session_id,
            rating=rating,
        )
