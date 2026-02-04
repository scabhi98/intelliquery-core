"""Cache Service Interface - Pattern storage and learning."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from shared.models.cache_models import CacheEntry, PatternMatch, LearningInsight


class CacheServiceInterface(ABC):
    """Interface for caching and learning from query patterns.
    
    This service handles:
    - Query pattern caching with PII elimination
    - Pattern matching and retrieval
    - Learning from successful queries
    - Performance optimization through caching
    """
    
    @abstractmethod
    async def store_pattern(
        self,
        entry: CacheEntry,
        journey_id: UUID
    ) -> str:
        """Store query pattern with PII elimination.
        
        Args:
            entry: Cache entry with query and result
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Cache key/ID
            
        Raises:
            CacheStorageException: If storage fails
            PIIDetectionException: If PII detection fails
        """
        pass
    
    @abstractmethod
    async def find_similar_patterns(
        self,
        query: str,
        platform: str,
        journey_id: UUID,
        threshold: float = 0.8
    ) -> List[PatternMatch]:
        """Find similar cached query patterns.
        
        Args:
            query: Natural language query
            platform: Target platform
            journey_id: Journey identifier for cost tracking
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of matching patterns with similarity scores
            
        Raises:
            CacheRetrievalException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_cached_result(
        self,
        cache_key: str,
        journey_id: UUID
    ) -> Optional[CacheEntry]:
        """Retrieve cached result by key.
        
        Args:
            cache_key: Cache entry identifier
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Cache entry or None if not found/expired
            
        Raises:
            CacheRetrievalException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def invalidate_pattern(
        self,
        cache_key: str,
        journey_id: UUID
    ) -> bool:
        """Invalidate cached pattern.
        
        Args:
            cache_key: Cache entry identifier
            journey_id: Journey identifier for cost tracking
            
        Returns:
            True if invalidated, False if not found
            
        Raises:
            CacheInvalidationException: If invalidation fails
        """
        pass
    
    @abstractmethod
    async def get_learning_insights(
        self,
        platform: Optional[str] = None,
        journey_id: Optional[UUID] = None
    ) -> List[LearningInsight]:
        """Get insights from learned patterns.
        
        Args:
            platform: Optional platform filter
            journey_id: Optional journey identifier
            
        Returns:
            List of learning insights
            
        Raises:
            InsightGenerationException: If insight generation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check service health and dependencies.
        
        Returns:
            Health status dictionary
        """
        pass
