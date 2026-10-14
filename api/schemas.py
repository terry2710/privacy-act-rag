from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Citation(BaseModel):
    rank: int
    chunk_id: str
    page: Optional[int] = None
    source: Optional[str] = None
    score: float
    text: Optional[str] = None


class QueryResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    citations: List[Citation]
    outcome: Literal["answered", "insufficient_evidence", "blocked"]
    pipeline_version: str


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    session_id: Optional[str] = Field(default=None, max_length=64)
    rating: Literal["helpful", "not_helpful"]


class FeedbackResponse(BaseModel):
    status: Literal["recorded"]
    persisted: bool


class ErrorResponse(BaseModel):
    error: str
    message: str
