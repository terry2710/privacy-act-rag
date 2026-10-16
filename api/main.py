import logging
import threading
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.repository import ANONYMOUS_OWNER, build_repository
from api.schemas import (
    Citation,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
)
from api.service import IndexNotReady, RagService

logger = logging.getLogger("api")

RETRY_AFTER_SECONDS = "30"


def create_app(service=None, repository=None):
    @asynccontextmanager
    async def lifespan(app):
        threading.Thread(
            target=app.state.service.warm_up, name="index-warm-up", daemon=True
        ).start()
        yield

    app = FastAPI(title="Privacy Act RAG API", version="0.3.0", lifespan=lifespan)
    app.state.service = service or RagService()
    app.state.repository = repository or build_repository()

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:64]
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health/live")
    def live():
        return {"status": "ok", "version": get_settings().app_version}

    @app.get("/health/ready")
    def ready():
        if app.state.service.is_ready():
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @app.post(
        "/api/queries",
        response_model=QueryResponse,
        responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def create_query(body: QueryRequest, background_tasks: BackgroundTasks):
        try:
            result = app.state.service.answer(body.question, body.session_id)
        except IndexNotReady:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "index_not_ready",
                    "message": "The search index is still loading. Please retry shortly.",
                },
                headers={"Retry-After": RETRY_AFTER_SECONDS},
            )
        except Exception:
            logger.exception("query failed")
            return JSONResponse(
                status_code=502,
                content={
                    "error": "upstream_error",
                    "message": "The answer could not be generated. Please try again.",
                },
            )
        response = QueryResponse(
            request_id=result["request_id"],
            session_id=result["session_id"],
            answer=result["answer"],
            citations=[
                Citation(
                    rank=chunk.get("rank"),
                    chunk_id=chunk.get("chunk_id"),
                    page=chunk.get("page"),
                    source=chunk.get("source"),
                    score=chunk.get("score"),
                    text=chunk.get("text"),
                )
                for chunk in result["chunks"]
            ],
            outcome="answered",
            pipeline_version=result["pipeline_version"],
        )
        background_tasks.add_task(
            app.state.repository.save_query,
            request_id=response.request_id,
            session_id=response.session_id,
            owner_sub=ANONYMOUS_OWNER,
            question=body.question,
            answer=response.answer,
            outcome=response.outcome,
            pipeline_version=response.pipeline_version,
        )
        return response

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def create_feedback(body: FeedbackRequest):
        app.state.service.record_feedback(body.request_id, body.session_id, body.rating)
        persisted = app.state.repository.save_feedback(body.request_id, ANONYMOUS_OWNER, body.rating)
        return FeedbackResponse(status="recorded", persisted=persisted)

    return app


app = create_app()
