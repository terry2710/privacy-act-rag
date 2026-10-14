import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from api.config import Settings
from api.db import Base, make_session_factory
from api.models import Feedback, Query
from api.repository import NullRepository, SqlRepository, build_repository


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return make_session_factory(engine)


@pytest.fixture
def repository(session_factory):
    return SqlRepository(session_factory)


def save_sample_query(repository, owner_sub="anonymous"):
    return repository.save_query(
        request_id="req123",
        session_id="sess1",
        owner_sub=owner_sub,
        question="What is APP 1?",
        answer="Open and transparent management of personal information.",
        outcome="answered",
        pipeline_version="fixed-dense-v1",
    )


def test_save_query_persists_row(repository, session_factory):
    assert save_sample_query(repository) is True
    with session_factory() as session:
        row = session.get(Query, "req123")
    assert row.question == "What is APP 1?"
    assert row.owner_sub == "anonymous"
    assert row.created_at is not None


def test_duplicate_request_id_is_reported_not_raised(repository):
    assert save_sample_query(repository) is True
    assert save_sample_query(repository) is False


def test_save_feedback_for_known_query(repository, session_factory):
    save_sample_query(repository)
    assert repository.save_feedback("req123", "anonymous", "helpful") is True
    with session_factory() as session:
        ratings = session.scalars(select(Feedback.rating)).all()
    assert ratings == ["helpful"]


def test_save_feedback_rejects_unknown_query(repository):
    assert repository.save_feedback("missing", "anonymous", "helpful") is False


def test_save_feedback_rejects_other_owner(repository):
    save_sample_query(repository, owner_sub="user-a")
    assert repository.save_feedback("req123", "user-b", "helpful") is False


def test_database_errors_are_swallowed():
    def broken_factory():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    repository = SqlRepository(broken_factory)
    assert save_sample_query(repository) is False
    assert repository.save_feedback("req123", "anonymous", "helpful") is False


def test_null_repository_never_persists():
    repository = NullRepository()
    assert save_sample_query(repository) is False
    assert repository.save_feedback("req123", "anonymous", "helpful") is False


def test_build_repository_without_database_url_returns_null():
    settings = Settings(_env_file=None, database_url="")
    assert isinstance(build_repository(settings), NullRepository)


def test_build_repository_with_database_url_returns_sql():
    settings = Settings(_env_file=None, database_url="sqlite://")
    assert isinstance(build_repository(settings), SqlRepository)
