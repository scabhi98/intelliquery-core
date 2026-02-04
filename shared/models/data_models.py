"""Data connector and execution related data models."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    """Configuration for data source connection."""
    
    connection_id: str = Field(..., description="Connection identifier")
    platform: str = Field(..., description="Platform type (kql, spl, sql)")
    endpoint: str = Field(..., description="Service endpoint URL")
    authentication: Dict[str, Any] = Field(
        ...,
        description="Authentication credentials (encrypted)"
    )
    workspace_id: Optional[str] = Field(
        None,
        description="Workspace/database identifier"
    )
    timeout: int = Field(
        default=300,
        description="Query timeout in seconds"
    )
    max_results: Optional[int] = Field(
        None,
        description="Maximum number of results"
    )
    ssl_verify: bool = Field(default=True, description="Verify SSL certificates")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "connection_id": "conn_azure_001",
                "platform": "kql",
                "endpoint": "https://api.loganalytics.io",
                "authentication": {"type": "oauth2", "token": "***"},
                "workspace_id": "workspace-123",
                "timeout": 300,
                "max_results": 10000
            }
        }


class QueryExecution(BaseModel):
    """Query execution request."""
    
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = Field(..., description="Query to execute")
    platform: str = Field(..., description="Target platform")
    config: DataSourceConfig = Field(..., description="Data source configuration")
    journey_id: UUID = Field(..., description="Journey identifier")
    timeout: int = Field(default=300, description="Query timeout in seconds")
    dry_run: bool = Field(default=False, description="Validate without executing")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "exec_001",
                "query": "SigninLogs | where TimeGenerated > ago(1h)",
                "platform": "kql",
                "timeout": 300,
                "dry_run": False
            }
        }


class ExecutionResult(BaseModel):
    """Result of query execution."""
    
    execution_id: str = Field(..., description="Execution identifier")
    success: bool = Field(..., description="Whether execution succeeded")
    data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Query results"
    )
    row_count: int = Field(default=0, description="Number of rows returned")
    columns: List[str] = Field(
        default_factory=list,
        description="Column names"
    )
    execution_time_ms: float = Field(
        ...,
        description="Execution time in milliseconds"
    )
    bytes_processed: Optional[int] = Field(
        None,
        description="Bytes processed by query"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Execution warnings"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "execution_id": "exec_001",
                "success": True,
                "row_count": 42,
                "columns": ["TimeGenerated", "UserPrincipalName", "ResultType"],
                "execution_time_ms": 456.2,
                "bytes_processed": 1048576,
                "warnings": []
            }
        }
