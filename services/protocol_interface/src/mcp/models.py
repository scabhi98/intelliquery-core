"""MCP models for tool registry and execution."""

from typing import Any, Awaitable, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


ToolHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class MCPTool(BaseModel):
    """Definition of an MCP tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for tool input"
    )
    tags: List[str] = Field(default_factory=list, description="Tool tags")
    handler: Optional[ToolHandler] = Field(default=None, exclude=True)


class MCPToolCallRequest(BaseModel):
    """Tool call request payload."""

    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict)


class MCPToolCallResponse(BaseModel):
    """Tool call response payload."""

    content: List[Dict[str, Any]] = Field(default_factory=list)
    is_error: bool = Field(default=False)
    error_message: Optional[str] = Field(default=None)
