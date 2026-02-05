"""MCP tool handlers for Protocol Interface service."""

from typing import Any, Dict
from uuid import UUID

import httpx
import structlog

from ..config import Settings


logger = structlog.get_logger()


def _resource_response(uri: str, mime_type: str, text: str) -> Dict[str, Any]:
    return {
        "type": "resource",
        "resource": {
            "uri": uri,
            "mimeType": mime_type,
            "text": text
        }
    }


def _text_response(text: str) -> Dict[str, Any]:
    return {
        "type": "text",
        "text": text
    }


class MCPToolHandlers:
    """Collection of MCP tool handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def process_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Proxy query processing to Core Engine."""
        payload = {
            "natural_language": arguments.get("natural_language"),
            "platform": arguments.get("platform"),
            "context": arguments.get("context")
        }
        if not payload["natural_language"] or not payload["platform"]:
            return {
                "content": [_text_response("Missing required fields: natural_language, platform")],
                "isError": True
            }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.core_engine_url}/api/v1/query",
                json=payload
            )

        if response.status_code != 200:
            return {
                "content": [_text_response("Core Engine query failed")],
                "isError": True
            }

        data = response.json()
        journey_id = data.get("journey_id")
        return {
            "content": [
                _text_response(f"Query processed successfully. Journey ID: {journey_id}"),
                _resource_response(
                    uri=f"lexi://journey/{journey_id}",
                    mime_type="application/json",
                    text=response.text
                )
            ]
        }

    async def get_journey_status(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get journey status from Core Engine."""
        journey_id = arguments.get("journey_id")
        if not journey_id:
            return {
                "content": [_text_response("Missing required field: journey_id")],
                "isError": True
            }

        try:
            UUID(journey_id)
        except ValueError:
            return {
                "content": [_text_response("Invalid journey_id format")],
                "isError": True
            }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.get(
                f"{self.settings.core_engine_url}/api/v1/journey/{journey_id}"
            )

        if response.status_code != 200:
            return {
                "content": [_text_response(f"Journey {journey_id} not found")],
                "isError": True
            }

        cost_response = None
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            cost_response = await client.get(
                f"{self.settings.core_engine_url}/api/v1/journey/{journey_id}/cost"
            )

        status_text = f"Journey {journey_id} is active"
        if cost_response is None or cost_response.status_code != 200:
            return {
                "content": [_text_response(status_text)]
            }

        return {
            "content": [
                _text_response(status_text),
                _resource_response(
                    uri=f"lexi://journey/{journey_id}/cost",
                    mime_type="application/json",
                    text=cost_response.text
                )
            ]
        }
