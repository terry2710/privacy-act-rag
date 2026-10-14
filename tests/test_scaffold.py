from sqlalchemy import inspect

from api import models
from api.config import Settings
from api.db import Base, make_engine


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.sqs_queue_name == "ingestion"
    assert settings.max_upload_bytes == 10 * 1024 * 1024


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("SQS_QUEUE_NAME", "custom")
    assert Settings(_env_file=None).sqs_queue_name == "custom"


def test_models_create_tables_in_memory():
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    tables = set(inspect(engine).get_table_names())
    assert tables == {"documents", "queries", "feedback", "reports"}
    assert models.Document.__tablename__ == "documents"
