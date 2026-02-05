"""Registry for MCP tools."""

from typing import Dict, List

from .models import MCPTool


class ToolRegistry:
    """In-memory registry for MCP tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> MCPTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[MCPTool]:
        """List all registered tools."""
        return list(self._tools.values())
