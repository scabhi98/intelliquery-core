"""SOP Engine Interface - Document processing and semantic search."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from shared.models.sop_models import SOPDocument, ProcessedSOP, Procedure


class SOPEngineInterface(ABC):
    """Interface for SOP processing and semantic search.
    
    This service handles:
    - Parsing and chunking SOP documents
    - Storing documents in vector database (ChromaDB)
    - Semantic search for relevant procedures
    - GraphRAG integration for schema understanding
    """
    
    @abstractmethod
    async def process_sop(
        self,
        document: SOPDocument,
        journey_id: UUID
    ) -> ProcessedSOP:
        """Parse and store SOP document in vector database.
        
        Args:
            document: SOP document with content and metadata
            journey_id: Journey identifier for cost tracking
            
        Returns:
            ProcessedSOP with embedding IDs and status
            
        Raises:
            VectorStoreException: If storage fails
            DocumentParsingException: If document cannot be parsed
        """
        pass
    
    @abstractmethod
    async def search_procedures(
        self,
        query: str,
        journey_id: UUID,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[Procedure]:
        """Semantic search for relevant procedures.
        
        Args:
            query: Natural language query
            journey_id: Journey identifier for cost tracking
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of relevant procedures with similarity scores
            
        Raises:
            VectorStoreException: If search fails
        """
        pass
    
    @abstractmethod
    async def get_schema_context(
        self,
        platform: str,
        journey_id: UUID
    ) -> dict:
        """Get schema context using GraphRAG.
        
        Args:
            platform: Target platform (kql, spl, sql)
            journey_id: Journey identifier for cost tracking
            
        Returns:
            Schema context with entities and relationships
            
        Raises:
            SchemaExtractionException: If schema extraction fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict:
        """Check service health and dependencies.
        
        Returns:
            Health status dictionary
        """
        pass
