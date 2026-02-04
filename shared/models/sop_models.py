"""SOP (Standard Operating Procedure) related data models."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SOPDocument(BaseModel):
    """SOP document for processing."""
    
    id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    version: str = Field(default="1.0", description="Document version")
    author: Optional[str] = Field(None, description="Document author")
    department: Optional[str] = Field(None, description="Department/team")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "sop_001",
                "title": "Incident Response: Service Outage",
                "content": "1. Identify affected service...",
                "version": "2.1",
                "department": "DevOps",
                "category": "incident-response",
                "tags": ["incident", "outage", "critical"]
            }
        }


class ProcessedSOP(BaseModel):
    """Result of SOP processing."""
    
    document_id: str = Field(..., description="Original document ID")
    status: str = Field(..., description="Processing status")
    embedding_ids: List[str] = Field(
        default_factory=list,
        description="Vector database embedding IDs"
    )
    chunks_count: int = Field(..., description="Number of chunks created")
    entities_extracted: List[str] = Field(
        default_factory=list,
        description="Extracted entities from GraphRAG"
    )
    relationships: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Extracted relationships"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "sop_001",
                "status": "processed",
                "embedding_ids": ["emb_001", "emb_002", "emb_003"],
                "chunks_count": 12,
                "entities_extracted": ["ServiceMonitor", "AlertManager"],
                "relationships": [
                    {"from": "ServiceMonitor", "to": "AlertManager", "type": "triggers"}
                ],
                "processing_time_ms": 1250.5
            }
        }


class Procedure(BaseModel):
    """Individual procedure from SOP."""
    
    id: str = Field(..., description="Procedure identifier")
    document_id: str = Field(..., description="Source document ID")
    title: str = Field(..., description="Procedure title")
    description: Optional[str] = Field(None, description="Procedure description")
    steps: List[str] = Field(default_factory=list, description="Procedure steps")
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites for execution"
    )
    expected_outcome: Optional[str] = Field(
        None,
        description="Expected outcome"
    )
    tags: List[str] = Field(default_factory=list, description="Procedure tags")
    similarity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Similarity score from search"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "proc_001",
                "document_id": "sop_001",
                "title": "Identify Service Outage",
                "steps": [
                    "Check monitoring dashboard",
                    "Verify service health endpoints",
                    "Review recent deployments"
                ],
                "prerequisites": ["Access to monitoring tools"],
                "expected_outcome": "Root cause identified",
                "tags": ["diagnostic", "outage"],
                "similarity_score": 0.92
            }
        }
