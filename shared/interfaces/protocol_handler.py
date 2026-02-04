"""Protocol Handler Interface - MCP and A2A protocol implementation."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from uuid import UUID

from shared.models.protocol_models import MCPRequest, MCPResponse, A2AMessage


class ProtocolHandlerInterface(ABC):
    """Interface for handling MCP and A2A protocols.
    
    This service handles:
    - Model Context Protocol (MCP) server and client
    - Agent-to-Agent (A2A) communication
    - Abstract authorization (Google/GCP/Azure)
    - OAuth2 OBO flow
    """
    
    @abstractmethod
    async def handle_mcp_request(
        self,
        request: MCPRequest,
        journey_id: UUID
    ) -> MCPResponse:
        """Handle incoming MCP request.
        
        Args:
            request: MCP request from client
            journey_id: Journey identifier for cost tracking
            
        Returns:
            MCP response
            
        Raises:
            MCPProtocolException: If protocol handling fails
            AuthorizationException: If authorization fails
        """
        pass
    
    @abstractmethod
    async def send_mcp_request(
        self,
        target_service: str,
        request: MCPRequest,
        journey_id: UUID
    ) -> MCPResponse:
        """Send MCP request to another service.
        
        Args:
            target_service: Target service identifier
            request: MCP request to send
            journey_id: Journey identifier for cost tracking
            
        Returns:
            MCP response from target service
            
        Raises:
            MCPClientException: If request fails
            ServiceUnavailableException: If target unavailable
        """
        pass
    
    @abstractmethod
    async def handle_a2a_message(
        self,
        message: A2AMessage,
        journey_id: UUID
    ) -> A2AMessage:
        """Handle incoming A2A message.
        
        Args:
            message: A2A message from another agent
            journey_id: Journey identifier for cost tracking
            
        Returns:
            A2A response message
            
        Raises:
            A2AProtocolException: If protocol handling fails
            AuthorizationException: If authorization fails
        """
        pass
    
    @abstractmethod
    async def send_a2a_message(
        self,
        target_agent: str,
        message: A2AMessage,
        journey_id: UUID
    ) -> A2AMessage:
        """Send A2A message to another agent.
        
        Args:
            target_agent: Target agent identifier
            message: A2A message to send
            journey_id: Journey identifier for cost tracking
            
        Returns:
            A2A response from target agent
            
        Raises:
            A2AClientException: If sending fails
            AgentUnavailableException: If target unavailable
        """
        pass
    
    @abstractmethod
    async def authenticate(
        self,
        provider: str,
        credentials: Dict[str, Any],
        journey_id: UUID
    ) -> Dict[str, Any]:
        """Authenticate using abstract auth provider.
        
        Args:
            provider: Auth provider (google, gcp, azure)
            credentials: Provider-specific credentials
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Authentication tokens and metadata
            
        Raises:
            AuthenticationException: If authentication fails
            UnsupportedProviderException: If provider not supported
        """
        pass
    
    @abstractmethod
    async def validate_token(
        self,
        token: str,
        provider: str,
        journey_id: UUID
    ) -> bool:
        """Validate authentication token.
        
        Args:
            token: Token to validate
            provider: Auth provider
            journey_id: Journey identifier for cost tracking
            
        Returns:
            True if token is valid
            
        Raises:
            TokenValidationException: If validation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check service health and dependencies.
        
        Returns:
            Health status dictionary
        """
        pass
