"""Agent registry for A2A discovery."""

import json
import os
from pathlib import Path
from typing import Dict, List

import httpx
import structlog

from shared.models.a2a_models import A2AAgentDescriptor
from shared.utils.exceptions import A2AProtocolException


logger = structlog.get_logger()


class AgentRegistry:
    """Persistent JSON file-based registry for A2A agents."""

    def __init__(self, timeout_seconds: int = 30, registry_file: str = "agent_registry.json") -> None:
        self._agents: Dict[str, A2AAgentDescriptor] = {}
        self.timeout_seconds = timeout_seconds
        self.registry_file = Path(registry_file)
        self._load_from_file()

    async def register_from_uri(self, service_uri: str) -> A2AAgentDescriptor:
        """Fetch agent descriptor from service and register it.
        
        The agent service must expose GET /a2a/descriptor endpoint.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{service_uri}/a2a/descriptor")
            
            if response.status_code != 200:
                raise A2AProtocolException(
                    f"Failed to fetch agent descriptor from {service_uri}",
                    details={"status_code": response.status_code}
                )
            
            descriptor = A2AAgentDescriptor.model_validate(response.json())
            self._agents[descriptor.agent_id] = descriptor
            self._save_to_file()
            
            logger.info(
                "Agent registered",
                agent_id=descriptor.agent_id,
                agent_type=descriptor.agent_type,
                endpoint=descriptor.endpoint
            )
            
            return descriptor
        except Exception as exc:
            logger.error("Agent registration failed", service_uri=service_uri, error=str(exc))
            raise A2AProtocolException(
                f"Agent registration failed: {str(exc)}",
                details={"service_uri": service_uri}
            )

    def register_descriptor(self, descriptor: A2AAgentDescriptor) -> None:
        """Directly register an agent descriptor."""
        self._agents[descriptor.agent_id] = descriptor
        self._save_to_file()
        logger.info("Agent registered directly", agent_id=descriptor.agent_id)

    def get_agent(self, agent_id: str) -> A2AAgentDescriptor | None:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[A2AAgentDescriptor]:
        """List all registered agents."""
        return list(self._agents.values())

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._save_to_file()
            logger.info("Agent unregistered", agent_id=agent_id)
            return True
        return False

    def get_count(self) -> int:
        """Get count of registered agents."""
        return len(self._agents)
    
    def _load_from_file(self) -> None:
        """Load registry from JSON file."""
        if not self.registry_file.exists():
            logger.info("Registry file not found, starting with empty registry", file=str(self.registry_file))
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            for agent_data in data:
                descriptor = A2AAgentDescriptor(**agent_data)
                self._agents[descriptor.agent_id] = descriptor
            
            logger.info(
                "Registry loaded from file",
                file=str(self.registry_file),
                agents_count=len(self._agents)
            )
        except Exception as exc:
            logger.error(
                "Failed to load registry from file",
                file=str(self.registry_file),
                error=str(exc)
            )
    
    def _save_to_file(self) -> None:
        """Save registry to JSON file."""
        try:
            # Ensure directory exists
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert agents to dict for JSON serialization
            data = [agent.model_dump(mode="json") for agent in self._agents.values()]
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(
                "Registry saved to file",
                file=str(self.registry_file),
                agents_count=len(self._agents)
            )
        except Exception as exc:
            logger.error(
                "Failed to save registry to file",
                file=str(self.registry_file),
                error=str(exc)
            )
