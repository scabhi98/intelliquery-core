"""Cache and learning related data models."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """Cached query pattern entry."""
    
    cache_id: str = Field(default_factory=lambda: str(uuid4()))
    query_hash: str = Field(..., description="Hash of the query pattern")
    natural_language: str = Field(
        ...,
        description="Original NL query (PII eliminated)"
    )
    platform: str = Field(..., description="Target platform")
    generated_query: str = Field(..., description="Generated query")
    success: bool = Field(..., description="Whether query was successful")
    execution_time_ms: Optional[float] = Field(
        None,
        description="Query execution time"
    )
    result_count: Optional[int] = Field(
        None,
        description="Number of results returned"
    )
    user_feedback: Optional[str] = Field(
        None,
        description="User feedback (positive/negative/neutral)"
    )
    usage_count: int = Field(default=1, description="Number of times used")
    last_used: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(
        None,
        description="Cache expiration time"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cache_id": "cache_123",
                "query_hash": "a1b2c3d4e5f6",
                "natural_language": "Show failed logins in last hour",
                "platform": "kql",
                "generated_query": "SigninLogs | where TimeGenerated > ago(1h)",
                "success": True,
                "execution_time_ms": 320.5,
                "result_count": 42,
                "user_feedback": "positive",
                "usage_count": 5
            }
        }


class PatternMatch(BaseModel):
    """Similar pattern match from cache."""
    
    cache_id: str = Field(..., description="Cache entry ID")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score"
    )
    natural_language: str = Field(..., description="Cached NL query")
    platform: str = Field(..., description="Target platform")
    generated_query: str = Field(..., description="Generated query")
    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Historical success rate"
    )
    usage_count: int = Field(..., description="Number of times used")
    last_used: datetime = Field(..., description="Last usage timestamp")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cache_id": "cache_123",
                "similarity_score": 0.89,
                "natural_language": "Show failed logins in last hour",
                "platform": "kql",
                "generated_query": "SigninLogs | where ResultType != 0",
                "success_rate": 0.95,
                "usage_count": 12
            }
        }


class LearningInsight(BaseModel):
    """Learning insight from query patterns."""
    
    insight_id: str = Field(default_factory=lambda: str(uuid4()))
    insight_type: str = Field(
        ...,
        description="Type of insight (performance, pattern, optimization)"
    )
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed description")
    platform: Optional[str] = Field(None, description="Platform filter")
    affected_patterns: int = Field(
        ...,
        description="Number of patterns affected"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in insight"
    )
    recommendation: str = Field(..., description="Recommended action")
    impact: str = Field(
        ...,
        description="Expected impact (high/medium/low)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "insight_id": "insight_001",
                "insight_type": "optimization",
                "title": "Common time filter pattern",
                "description": "85% of queries use 1-hour time window",
                "platform": "kql",
                "affected_patterns": 127,
                "confidence": 0.95,
                "recommendation": "Pre-cache common time windows",
                "impact": "high"
            }
        }
