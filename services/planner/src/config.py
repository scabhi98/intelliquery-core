"""Configuration for Planner Agent service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for Planner Agent."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    service_name: str = "planner-agent"
    service_port: int = 9000
    log_level: str = "INFO"

    agent_id: str = "workflow_planner"
    agent_name: str = "Workflow Planner Agent"
    agent_version: str = "2.0.0"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout_seconds: int = 30

    default_step_timeout_seconds: int = 60
    max_parallel_agents: int = 5


settings = Settings()
