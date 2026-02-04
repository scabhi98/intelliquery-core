"""Query Generator Interface - Multi-platform query generation."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from shared.models.query_models import QueryRequest, QueryResponse, QueryValidation


class QueryGeneratorInterface(ABC):
    """Interface for generating platform-specific queries.
    
    This service handles:
    - Natural language to query translation
    - Platform-specific optimization (KQL, SPL, SQL)
    - GraphRAG-enhanced schema awareness
    - Query validation and testing
    """
    
    @abstractmethod
    async def generate(
        self,
        request: QueryRequest,
        journey_id: UUID
    ) -> QueryResponse:
        """Generate platform-specific query from natural language.
        
        Args:
            request: Query request with NL query and platform
            journey_id: Journey identifier for cost tracking
            
        Returns:
            QueryResponse with generated query and metadata
            
        Raises:
            QueryGenerationException: If generation fails
            UnsupportedPlatformException: If platform not supported
        """
        pass
    
    @abstractmethod
    async def validate_query(
        self,
        query: str,
        platform: str,
        journey_id: UUID
    ) -> QueryValidation:
        """Validate generated query syntax and semantics.
        
        Args:
            query: Query to validate
            platform: Target platform (kql, spl, sql)
            journey_id: Journey identifier for cost tracking
            
        Returns:
            QueryValidation with validation results
            
        Raises:
            ValidationException: If validation process fails
        """
        pass
    
    @abstractmethod
    async def optimize_query(
        self,
        query: str,
        platform: str,
        journey_id: UUID
    ) -> QueryResponse:
        """Optimize existing query for performance.
        
        Args:
            query: Query to optimize
            platform: Target platform
            journey_id: Journey identifier for cost tracking
            
        Returns:
            QueryResponse with optimized query
            
        Raises:
            OptimizationException: If optimization fails
        """
        pass
    
    @abstractmethod
    async def explain_query(
        self,
        query: str,
        platform: str,
        journey_id: UUID
    ) -> dict:
        """Generate natural language explanation of query.
        
        Args:
            query: Query to explain
            platform: Target platform
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Dictionary with explanation and metadata
            
        Raises:
            ExplanationException: If explanation generation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check service health and dependencies.
        
        Returns:
            Health status dictionary
        """
        pass
