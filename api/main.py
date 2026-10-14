import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    Citation,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
)
from api.service import RagService

logger = logging.getLogger("api")


def create_app(service=None):
    app = FastAPI(title="Privacy Act RAG API", version="0.1.0")
    app.state.service = service or RagService()

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = (request.headers.get("x-request-id") or uuid.uuid4().hex)[:64]
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        if app.state.service.is_ready():
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @app.post(
        "/api/queries",
        response_model=QueryResponse,
        responses={502: {"model": ErrorResponse}},
    )
    def create_query(body: QueryRequest):
        try:
            result = app.state.service.answer(body.question, body.session_id)
        except Exception:
            logger.exception("query failed")
            return JSONResponse(
                status_code=502,
                content={
                    "error": "upstream_error",
                    "message": "The answer could not be generated. Please try again.",
                },
            )
        citations = [
            Citation(
                rank=chunk.get("rank"),
                chunk_id=chunk.get("chunk_id"),
                page=chunk.get("page"),
                source=chunk.get("source"),
                score=chunk.get("score"),
                text=chunk.get("text"),
            )
            for chunk in result["chunks"]
        ]
        return QueryResponse(
            request_id=result["request_id"],
            session_id=result["session_id"],
            answer=result["answer"],
            citations=citations,
            outcome="answered",
            pipeline_version=result["pipeline_version"],
        )

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def create_feedback(body: FeedbackRequest):
        app.state.service.record_feedback(body.request_id, body.session_id, body.rating)
        return FeedbackResponse(status="recorded")

    return app


app = create_app()
