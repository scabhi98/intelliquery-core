"""Cost Tracker Interface - LLM and embedding cost tracking."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from shared.models.cost_models import ModelUsage, JourneyCostSummary


class CostTrackerInterface(ABC):
    """Interface for tracking LLM and embedding costs.
    
    This service handles:
    - Per-operation cost tracking
    - Journey-level cost aggregation
    - Token usage monitoring
    - Cost optimization insights
    """
    
    @abstractmethod
    async def track_usage(
        self,
        usage: ModelUsage
    ) -> None:
        """Record a model usage event.
        
        Args:
            usage: Model usage metrics including tokens, latency, cost
            
        Raises:
            CostTrackingException: If tracking fails
        """
        pass
    
    @abstractmethod
    async def get_journey_cost(
        self,
        journey_id: UUID
    ) -> JourneyCostSummary:
        """Get cost summary for a journey.
        
        Args:
            journey_id: Journey identifier
            
        Returns:
            Aggregate cost summary with breakdowns
            
        Raises:
            JourneyNotFoundException: If journey not found
            CostRetrievalException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_recent_journeys(
        self,
        limit: int = 10,
        service_filter: Optional[str] = None
    ) -> List[JourneyCostSummary]:
        """Get recent journey cost summaries.
        
        Args:
            limit: Maximum number of journeys to return
            service_filter: Optional service name filter
            
        Returns:
            List of journey summaries
            
        Raises:
            CostRetrievalException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_service_costs(
        self,
        service_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> dict:
        """Get aggregated costs for a service.
        
        Args:
            service_name: Service identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Service cost aggregates
            
        Raises:
            CostRetrievalException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_model_costs(
        self,
        model_name: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> dict:
        """Get aggregated costs for a model.
        
        Args:
            model_name: Model identifier (e.g., llama3, nomic-embed-text)
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Model cost aggregates
            
        Raises:
            CostRetrievalException: If retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_optimization_recommendations(
        self,
        journey_id: UUID
    ) -> List[dict]:
        """Get cost optimization recommendations for a journey.
        
        Args:
            journey_id: Journey identifier
            
        Returns:
            List of optimization recommendations
            
        Raises:
            OptimizationException: If analysis fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check service health and dependencies.
        
        Returns:
            Health status dictionary
        """
        pass
