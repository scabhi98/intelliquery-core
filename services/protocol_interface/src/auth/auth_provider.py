"""Abstract authentication provider interface."""

from abc import ABC, abstractmethod
from typing import Optional

from shared.utils.exceptions import AuthenticationException


class AuthProvider(ABC):
    """Authentication provider interface."""

    @abstractmethod
    async def authenticate(self) -> str:
        """Get authenticated access token."""
        raise NotImplementedError

    @abstractmethod
    async def exchange_token_obo(self, user_token: str, target_service: str) -> str:
        """Exchange token for OBO flow."""
        raise NotImplementedError

    @abstractmethod
    async def validate_token(self, token: str) -> bool:
        """Validate access token."""
        raise NotImplementedError


class NoAuthProvider(AuthProvider):
    """No-op auth provider for local development."""

    async def authenticate(self) -> str:
        raise AuthenticationException("Authentication provider not configured")

    async def exchange_token_obo(self, user_token: str, target_service: str) -> str:
        raise AuthenticationException("OBO exchange not supported")

    async def validate_token(self, token: str) -> bool:
        return False
