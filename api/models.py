import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


def new_id():
    return uuid.uuid4().hex


def utcnow():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_sub: Mapped[str] = mapped_column(String(128), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    s3_key: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    owner_sub: Mapped[str] = mapped_column(String(128), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(32))
    pipeline_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), index=True)
    owner_sub: Mapped[str] = mapped_column(String(128), index=True)
    rating: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    owner_sub: Mapped[str] = mapped_column(String(128), index=True)
    content: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
