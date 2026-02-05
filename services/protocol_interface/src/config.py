"""Configuration for Protocol Interface service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Protocol Interface configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Service
    service_name: str = "protocol-interface"
    service_port: int = 8001
    log_level: str = "INFO"

    # Core Engine
    core_engine_url: str = "http://localhost:8000"

    # Auth
    auth_provider: str = "abstract"
    auth_enabled: bool = False

    # Timeouts
    request_timeout_seconds: int = 30

    # A2A discovery
    a2a_registry_url: str | None = None
    a2a_static_agents_json: str = "[]"


settings = Settings()
