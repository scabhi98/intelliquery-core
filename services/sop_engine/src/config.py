"""Configuration for SOP Engine service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SOP Engine configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # Service
    service_name: str = "sop-engine"
    service_port: int = 8002
    log_level: str = "INFO"

    # Ollama
    ollama_host: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"

    # ChromaDB
    chroma_db_path: str = "./chroma_db"
    chroma_collection: str = "sop_procedures"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Search
    default_top_k: int = 5

    # Upload
    max_upload_size_mb: int = 10


settings = Settings()
