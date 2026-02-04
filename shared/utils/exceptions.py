"""Custom exception hierarchy for LexiQuery."""

from typing import Dict, Any, Optional


class LexiQueryException(Exception):
    """Base exception for all LexiQuery errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "error_code": self.error_code,
        }


# Service-level exceptions
class ServiceUnavailableException(LexiQueryException):
    """Raised when a required service is unavailable."""
    pass


class ServiceTimeoutException(LexiQueryException):
    """Raised when a service call times out."""
    pass


# Query-related exceptions
class QueryGenerationException(LexiQueryException):
    """Raised when query generation fails."""
    pass


class QueryExecutionException(LexiQueryException):
    """Raised when query execution fails."""
    pass


class QueryValidationException(LexiQueryException):
    """Raised when query validation fails."""
    pass


class UnsupportedPlatformException(LexiQueryException):
    """Raised when a platform is not supported."""
    pass


# SOP-related exceptions
class DocumentParsingException(LexiQueryException):
    """Raised when SOP document parsing fails."""
    pass


class SchemaExtractionException(LexiQueryException):
    """Raised when GraphRAG schema extraction fails."""
    pass


# Vector store exceptions
class VectorStoreException(LexiQueryException):
    """Raised when vector database operations fail."""
    pass


class EmbeddingException(LexiQueryException):
    """Raised when embedding generation fails."""
    pass


# Cache-related exceptions
class CacheStorageException(LexiQueryException):
    """Raised when cache storage operations fail."""
    pass


class CacheRetrievalException(LexiQueryException):
    """Raised when cache retrieval operations fail."""
    pass


class PIIDetectionException(LexiQueryException):
    """Raised when PII detection fails."""
    pass


# Authentication/Authorization exceptions
class AuthenticationException(LexiQueryException):
    """Raised when authentication fails."""
    pass


class AuthorizationException(LexiQueryException):
    """Raised when authorization fails."""
    pass


class TokenValidationException(LexiQueryException):
    """Raised when token validation fails."""
    pass


class UnsupportedProviderException(LexiQueryException):
    """Raised when auth provider is not supported."""
    pass


# Protocol exceptions
class MCPProtocolException(LexiQueryException):
    """Raised when MCP protocol handling fails."""
    pass


class A2AProtocolException(LexiQueryException):
    """Raised when A2A protocol handling fails."""
    pass


# Data connector exceptions
class ConnectionException(LexiQueryException):
    """Raised when data source connection fails."""
    pass


class TimeoutException(LexiQueryException):
    """Raised when operation exceeds timeout."""
    pass


# Cost tracking exceptions
class CostTrackingException(LexiQueryException):
    """Raised when cost tracking operations fail."""
    pass


class JourneyNotFoundException(LexiQueryException):
    """Raised when journey is not found."""
    pass


# Configuration exceptions
class ConfigurationException(LexiQueryException):
    """Raised when configuration is invalid or missing."""
    pass
