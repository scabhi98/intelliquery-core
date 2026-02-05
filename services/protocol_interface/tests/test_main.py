"""Protocol Interface Service Tests."""

import pytest
from httpx import AsyncClient

from services.protocol_interface.src.main import app


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "protocol-interface"


@pytest.mark.asyncio
async def test_mcp_tools_list():
    """Test MCP tools listing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/mcp/tools/list")
    
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert len(data["tools"]) > 0


@pytest.mark.asyncio
async def test_a2a_discovery_empty():
    """Test A2A discovery with empty registry."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/a2a/agents/discover")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_a2a_registry_stats():
    """Test A2A registry statistics."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/a2a/registry/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert "total_agents" in data
    assert "agents" in data
