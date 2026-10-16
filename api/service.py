import logging
import threading
import uuid

logger = logging.getLogger("api.service")


class IndexNotReady(Exception):
    pass


def load_cached_index():
    from core import rag_backend

    return rag_backend.prv_load_cached_index()


class RagService:
    def __init__(self, index_loader=load_cached_index):
        self._index = None
        self._index_loader = index_loader
        self._lock = threading.Lock()

    def warm_up(self, wait=True):
        if self._index is not None:
            return True
        if not self._lock.acquire(blocking=wait):
            return False
        try:
            if self._index is None:
                self._index = self._index_loader()
                if self._index is None:
                    logger.warning("no cached index available; run python -m tools.build_index")
        except Exception:
            logger.warning("index warm-up failed", exc_info=True)
        finally:
            self._lock.release()
        return self._index is not None

    def is_ready(self):
        return self._index is not None

    def answer(self, question, session_id=None):
        if not self.warm_up(wait=False):
            raise IndexNotReady()
        from core import rag_backend

        session_id = session_id or uuid.uuid4().hex[:12]
        answer, details = rag_backend.prv_rag_response(
            index=self._index,
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
        from core import rag_backend

        rag_backend.rag_logging.emit_feedback(
            request_id=request_id,
            session_id=session_id,
            rating=rating,
        )
