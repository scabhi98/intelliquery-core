"""Core Engine Service Configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Core Engine configuration from environment variables.
    
    Environment variables can be set in .env file or system environment.
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        env_prefix='CORE_ENGINE_'
    )
    
    # Service configuration
    service_name: str = "core-engine"
    service_port: int = 8000
    log_level: str = "INFO"
    
    # Orchestrator configuration
    max_retries: int = 3
    agent_timeout_seconds: int = 60
    
    # A2A Agent endpoints (Phase 2: Mock agents)
    planner_agent_url: str = "http://localhost:9000"
    
    # Knowledge Provider agents
    sop_knowledge_url: str = "http://localhost:9010"
    error_knowledge_url: str = "http://localhost:9011"
    wiki_knowledge_url: str = "http://localhost:9012"
    
    # Data agents
    kql_data_url: str = "http://localhost:9020"
    spl_data_url: str = "http://localhost:9021"
    sql_data_url: str = "http://localhost:9022"
    
    # MCP Server endpoints (Phase 2: Mock servers)
    chromadb_mcp_url: str = "http://localhost:9100"
    azure_la_mcp_url: str = "http://localhost:9101"
    splunk_mcp_url: str = "http://localhost:9102"
    sql_mcp_url: str = "http://localhost:9103"
    errordb_mcp_url: str = "http://localhost:9104"

    # Protocol Interface (Phase 4)
    protocol_interface_url: str = "http://localhost:8001"
    
    # Redis (for production journey storage)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Auth (placeholder for Phase 4)
    auth_enabled: bool = False
    auth_provider: str = "abstract"
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    
    # Cost tracking
    cost_tracking_enabled: bool = True
    cost_alert_threshold_usd: float = 1.0  # Alert if journey exceeds $1
    
    # Feature flags
    enable_mcp_server: bool = True
    enable_a2a_discovery: bool = False  # Phase 4: Dynamic agent discovery
    enable_query_execution: bool = False  # Phase 2: Only generation, Phase 4: Execution
    
    # Performance
    max_concurrent_agents: int = 5
    workflow_timeout_seconds: int = 300  # 5 minutes max per workflow


# Global settings instance
settings = Settings()
