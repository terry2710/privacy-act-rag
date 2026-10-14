from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from api.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine(url=None):
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
