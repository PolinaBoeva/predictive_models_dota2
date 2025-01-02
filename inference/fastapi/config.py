from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_file: str = "logs/fastapi/app.log"
    log_max_bytes: int = 1000000
    log_backup_count: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class FastAPIConfig(BaseSettings):
    title: str = "Predictive Models Dota2 API"
    version: str = "1.0"
    description: str = (
        "API для обучения моделей предсказания исхода игр (Radiant/Dire)."
    )
    docs_url: str = "/api/openapi"
    openapi_url: str = "/api/openapi.json"

    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Config(BaseSettings):
    log_config: LoggingConfig = LoggingConfig()
    fastapi_config: FastAPIConfig = FastAPIConfig()


@lru_cache
def get_config():
    return Config()
