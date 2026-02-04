"""A2A protocol specific models."""

from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class A2AAgentDescriptor(BaseModel):
    """Descriptor for an A2A agent discovered via registry."""
    
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: Literal["planner", "knowledge", "data"] = Field(
        ...,
        description="Type of agent"
    )
    agent_name: str = Field(..., description="Human-readable agent name")
    endpoint: str = Field(..., description="A2A endpoint URL")
    capabilities: List[str] = Field(..., description="List of supported actions/tasks")
    platform: Optional[str] = Field(None, description="Platform for data agents (kql, spl, sql)")
    version: str = Field("1.0.0", description="Agent version")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "kql_data_agent_001",
                "agent_type": "data",
                "agent_name": "KQL Data Agent",
                "endpoint": "http://kql-data-a2a:9020",
                "capabilities": ["generate_query", "execute_query"],
                "platform": "kql",
                "version": "1.0.0"
            }
        }


class A2ATaskRequest(BaseModel):
    """Request to invoke an A2A agent task."""
    
    task_id: UUID = Field(default_factory=uuid4)
    journey_id: UUID = Field(..., description="Journey identifier for tracking")
    agent_id: str = Field(..., description="Target agent ID")
    action: str = Field(..., description="Action/task to perform")
    parameters: Dict[str, Any] = Field(..., description="Task parameters")
    timeout_seconds: int = Field(60, description="Task timeout")
    
    class Config:
        json_schema_extra = {
            "example": {
                "journey_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "sop_knowledge_agent",
                "action": "retrieve_sop_context",
                "parameters": {
                    "query": "failed login procedures",
                    "top_k": 5
                },
                "timeout_seconds": 30
            }
        }


class A2ATaskResponse(BaseModel):
    """Response from an A2A agent task."""
    
    task_id: UUID = Field(..., description="Original task ID")
    journey_id: UUID = Field(..., description="Journey identifier")
    agent_id: str = Field(..., description="Agent that executed the task")
    status: Literal["success", "error", "timeout"] = Field(..., description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    cost_info: Optional[Dict[str, Any]] = Field(None, description="Cost information from agent")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "650e8400-e29b-41d4-a716-446655440000",
                "journey_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "sop_knowledge_agent",
                "status": "success",
                "result": {
                    "procedures": [
                        {"id": "sop_001", "title": "Incident Response"}
                    ]
                },
                "cost_info": {
                    "llm_tokens": 250,
                    "mcp_calls": 1,
                    "cost_usd": 0.003
                },
                "execution_time_ms": 450.5
            }
        }


class KnowledgeContext(BaseModel):
    """Knowledge context retrieved from knowledge provider agents."""
    
    source: str = Field(..., description="Knowledge source (sop, errors, wiki, docs)")
    agent_id: str = Field(..., description="Agent that provided this context")
    content: Dict[str, Any] = Field(..., description="Knowledge content")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "sop",
                "agent_id": "sop_knowledge_agent",
                "content": {
                    "procedures": [
                        {
                            "id": "sop_001",
                            "title": "Incident Response: Failed Logins",
                            "steps": ["Check logs", "Verify user", "Lock account"]
                        }
                    ]
                },
                "relevance_score": 0.87
            }
        }


class QueryResult(BaseModel):
    """Result from a data agent (query generation and/or execution)."""
    
    platform: str = Field(..., description="Platform (kql, spl, sql)")
    agent_id: str = Field(..., description="Data agent that generated the query")
    query: str = Field(..., description="Generated query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    explanation: Optional[str] = Field(None, description="Explanation of query")
    optimizations: List[str] = Field(default_factory=list, description="Applied optimizations")
    
    # Execution results (if execute=true)
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Query execution results")
    row_count: Optional[int] = Field(None, description="Number of rows returned")
    execution_time_ms: Optional[float] = Field(None, description="Query execution time")
    
    # Cost information
    cost_info: Optional[Dict[str, Any]] = Field(None, description="Cost breakdown")
    
    # Warnings/metadata
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "platform": "kql",
                "agent_id": "kql_data_agent",
                "query": "SigninLogs | where TimeGenerated > ago(1h) | where Status == 'Failure'",
                "confidence": 0.89,
                "explanation": "Query retrieves failed sign-ins from the last hour",
                "optimizations": ["time_filter_first", "index_aware"],
                "data": [
                    {"TimeGenerated": "2026-02-05T10:30:00Z", "UserPrincipalName": "user@example.com"}
                ],
                "row_count": 42,
                "execution_time_ms": 1250.5,
                "cost_info": {
                    "llm_tokens": 450,
                    "mcp_calls": 1,
                    "cost_usd": 0.008
                }
            }
        }


class MCPToolCall(BaseModel):
    """Record of an MCP tool invocation."""
    
    tool_server: str = Field(..., description="MCP server name")
    tool_name: str = Field(..., description="Tool/method name")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: float = Field(..., description="Execution time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
