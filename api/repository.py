import logging

from sqlalchemy.exc import SQLAlchemyError

from api.config import get_settings
from api.db import make_engine, make_session_factory
from api.models import Feedback, Query

logger = logging.getLogger("api.repository")

ANONYMOUS_OWNER = "anonymous"


class NullRepository:
    def save_query(self, request_id, session_id, owner_sub, question, answer, outcome, pipeline_version):
        return False

    def save_feedback(self, request_id, owner_sub, rating):
        return False


class SqlRepository:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def save_query(self, request_id, session_id, owner_sub, question, answer, outcome, pipeline_version):
        try:
            with self._session_factory() as session:
                session.add(
                    Query(
                        id=request_id,
                        session_id=session_id,
                        owner_sub=owner_sub,
                        question=question,
                        answer=answer,
                        outcome=outcome,
                        pipeline_version=pipeline_version,
                    )
                )
                session.commit()
            return True
        except SQLAlchemyError:
            logger.warning("could not persist query %s", request_id, exc_info=True)
            return False

    def save_feedback(self, request_id, owner_sub, rating):
        try:
            with self._session_factory() as session:
                query = session.get(Query, request_id)
                if query is None or query.owner_sub != owner_sub:
                    return False
                session.add(Feedback(query_id=request_id, owner_sub=owner_sub, rating=rating))
                session.commit()
            return True
        except SQLAlchemyError:
            logger.warning("could not persist feedback for %s", request_id, exc_info=True)
            return False


def build_repository(settings=None):
    settings = settings or get_settings()
    if not settings.database_url:
        return NullRepository()
    return SqlRepository(make_session_factory(make_engine(settings.database_url)))
