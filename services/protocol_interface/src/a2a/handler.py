"""A2A protocol handler (HTTP fallback implementation)."""

import json
import time
from typing import List

import httpx
import structlog

from shared.models.a2a_models import A2AAgentDescriptor, A2ATaskRequest, A2ATaskResponse
from shared.utils.exceptions import A2AProtocolException

from ..config import Settings


logger = structlog.get_logger()


class A2AHandler:
    """Minimal A2A handler using HTTP-based agent endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timeout_seconds = settings.request_timeout_seconds

    async def invoke_task(self, agent: A2AAgentDescriptor, request: A2ATaskRequest) -> A2ATaskResponse:
        """Invoke an A2A task via agent endpoint."""
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{agent.endpoint}/a2a/task",
                    json=request.model_dump(mode="json")
                )

            if response.status_code != 200:
                raise A2AProtocolException(
                    f"Agent {agent.agent_id} call failed",
                    details={"status_code": response.status_code, "body": response.text}
                )

            payload = response.json()
            return A2ATaskResponse(**payload)
        except Exception as exc:
            logger.error("A2A task invocation failed", agent_id=agent.agent_id, error=str(exc))
            raise A2AProtocolException(str(exc))
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info("A2A task invocation completed", agent_id=agent.agent_id, elapsed_ms=elapsed_ms)

    async def discover_agents(self, internal_registry=None) -> List[A2AAgentDescriptor]:
        """Discover agents using internal registry, external registry, or static configuration."""
        agents: List[A2AAgentDescriptor] = []

        # Priority 1: Internal registry
        if internal_registry:
            agents = internal_registry.list_agents()
            if agents:
                return agents

        # Priority 2: External A2A registry
        if self.settings.a2a_registry_url:
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(
                        f"{self.settings.a2a_registry_url}/agents"
                    )
                if response.status_code == 200:
                    payload = response.json()
                    agents = [A2AAgentDescriptor(**item) for item in payload]
                    return agents
            except Exception as exc:
                logger.warning("A2A registry discovery failed", error=str(exc))

        # Priority 3: Static configuration
        if not agents:
            try:
                static_list = json.loads(self.settings.a2a_static_agents_json or "[]")
                agents = [A2AAgentDescriptor(**item) for item in static_list]
            except Exception as exc:
                logger.warning("Static A2A agent parse failed", error=str(exc))

        return agents
