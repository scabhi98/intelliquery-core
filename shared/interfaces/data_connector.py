"""Data Connector Interface - Platform-specific data source integration."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from uuid import UUID

from shared.models.data_models import QueryExecution, DataSourceConfig, ExecutionResult


class DataConnectorInterface(ABC):
    """Interface for connecting to data platforms.
    
    This service handles:
    - Azure Log Analytics (KQL) integration
    - Splunk (SPL) integration
    - SQL database connectivity
    - Query execution and result handling
    """
    
    @abstractmethod
    async def execute_query(
        self,
        query: str,
        platform: str,
        config: DataSourceConfig,
        journey_id: UUID,
        timeout: Optional[int] = 300
    ) -> ExecutionResult:
        """Execute query on target platform.
        
        Args:
            query: Platform-specific query
            platform: Target platform (kql, spl, sql)
            config: Data source configuration
            journey_id: Journey identifier for cost tracking
            timeout: Query timeout in seconds
            
        Returns:
            ExecutionResult with data and metadata
            
        Raises:
            QueryExecutionException: If execution fails
            TimeoutException: If query exceeds timeout
            AuthenticationException: If authentication fails
        """
        pass
    
    @abstractmethod
    async def test_connection(
        self,
        platform: str,
        config: DataSourceConfig,
        journey_id: UUID
    ) -> bool:
        """Test connection to data source.
        
        Args:
            platform: Target platform
            config: Data source configuration
            journey_id: Journey identifier for cost tracking
            
        Returns:
            True if connection successful
            
        Raises:
            ConnectionException: If connection test fails
        """
        pass
    
    @abstractmethod
    async def get_schema_info(
        self,
        platform: str,
        config: DataSourceConfig,
        journey_id: UUID
    ) -> Dict[str, Any]:
        """Get schema information from data source.
        
        Args:
            platform: Target platform
            config: Data source configuration
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Schema information dictionary
            
        Raises:
            SchemaRetrievalException: If schema retrieval fails
        """
        pass
    
    @abstractmethod
    async def validate_query_syntax(
        self,
        query: str,
        platform: str,
        config: DataSourceConfig,
        journey_id: UUID
    ) -> Dict[str, Any]:
        """Validate query syntax without execution.
        
        Args:
            query: Query to validate
            platform: Target platform
            config: Data source configuration
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Validation results
            
        Raises:
            ValidationException: If validation fails
        """
        pass
    
    @abstractmethod
    async def cancel_query(
        self,
        execution_id: str,
        platform: str,
        journey_id: UUID
    ) -> bool:
        """Cancel running query.
        
        Args:
            execution_id: Query execution identifier
            platform: Target platform
            journey_id: Journey identifier for cost tracking
            
        Returns:
            True if cancelled successfully
            
        Raises:
            CancellationException: If cancellation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check service health and dependencies.
        
        Returns:
            Health status dictionary
        """
        pass
