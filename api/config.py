from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    sqs_endpoint_url: Optional[str] = None
    sqs_queue_name: str = "ingestion"
    upload_bucket: str = ""
    max_upload_bytes: int = 10 * 1024 * 1024
    cognito_region: str = ""
    cognito_user_pool_id: str = ""
    cognito_app_client_id: str = ""


@lru_cache
def get_settings():
    return Settings()
