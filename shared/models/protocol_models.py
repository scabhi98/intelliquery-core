"""Protocol (MCP and A2A) related data models."""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    """Model Context Protocol request."""
    
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    method: str = Field(..., description="MCP method name")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method parameters"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context"
    )
    auth_token: Optional[str] = Field(None, description="Authentication token")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "mcp_req_001",
                "method": "query.generate",
                "params": {
                    "natural_language": "Show failed logins",
                    "platform": "kql"
                },
                "context": {"workspace": "production"}
            }
        }


class MCPResponse(BaseModel):
    """Model Context Protocol response."""
    
    request_id: str = Field(..., description="Original request ID")
    success: bool = Field(..., description="Whether request succeeded")
    result: Optional[Dict[str, Any]] = Field(
        None,
        description="Result data"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "mcp_req_001",
                "success": True,
                "result": {
                    "query": "SigninLogs | where ResultType != 0",
                    "confidence": 0.95
                },
                "processing_time_ms": 234.5
            }
        }


class A2AMessage(BaseModel):
    """Agent-to-Agent protocol message."""
    
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    sender_id: str = Field(..., description="Sending agent identifier")
    receiver_id: str = Field(..., description="Receiving agent identifier")
    message_type: Literal["request", "response", "notification", "error"] = Field(
        ...,
        description="Message type"
    )
    action: str = Field(..., description="Requested action or response type")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation thread ID"
    )
    reply_to: Optional[str] = Field(
        None,
        description="Message ID this is replying to"
    )
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal",
        description="Message priority"
    )
    auth_token: Optional[str] = Field(None, description="Authentication token")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(
        None,
        description="Message expiration time"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "a2a_msg_001",
                "sender_id": "query-generator",
                "receiver_id": "sop-engine",
                "message_type": "request",
                "action": "search_procedures",
                "payload": {
                    "query": "incident response",
                    "top_k": 5
                },
                "priority": "normal"
            }
        }
