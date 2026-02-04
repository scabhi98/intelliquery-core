"""Mock A2A Knowledge Provider Agents for Phase 2 Testing."""

from typing import Dict, List
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import structlog

from shared.models.a2a_models import A2ATaskRequest, A2ATaskResponse, KnowledgeContext


logger = structlog.get_logger()


# Mock SOP knowledge data
MOCK_SOP_PROCEDURES = [
    {
        "id": "sop_001",
        "title": "Incident Response: Failed Login Investigation",
        "description": "Standard procedure for investigating failed authentication attempts",
        "steps": [
            "Query SigninLogs table for failed authentication events",
            "Identify user accounts with multiple consecutive failures",
            "Check IP addresses against known threat intelligence",
            "Verify if failures correlate with unusual times or locations",
            "Lock account if suspicious activity detected",
            "Notify security team if pattern indicates attack"
        ],
        "relevant_tables": ["SigninLogs", "AuditLogs", "AADSignInEventsBeta"],
        "relevant_columns": ["UserPrincipalName", "IPAddress", "Status", "ResultType", "Location"],
        "query_templates": [
            "SigninLogs | where ResultType != 0 | where TimeGenerated > ago(1h)"
        ],
        "tags": ["security", "authentication", "incident", "login"],
        "priority": "high"
    },
    {
        "id": "sop_002",
        "title": "Performance Degradation Analysis",
        "description": "Procedure for analyzing service performance issues",
        "steps": [
            "Check metrics for CPU, memory, and response time",
            "Query application logs for errors or warnings",
            "Identify slow database queries",
            "Check for recent deployments or configuration changes",
            "Review resource utilization trends"
        ],
        "relevant_tables": ["PerformanceMetrics", "ApplicationLogs", "ResourceUtilization"],
        "relevant_columns": ["Timestamp", "ServiceName", "ResponseTime", "CPU", "Memory"],
        "query_templates": [
            "PerformanceMetrics | where ResponseTime > 1000 | summarize avg(ResponseTime) by ServiceName"
        ],
        "tags": ["performance", "monitoring", "troubleshooting"],
        "priority": "medium"
    },
    {
        "id": "sop_003",
        "title": "Database Connection Error Troubleshooting",
        "description": "Steps to diagnose and resolve database connectivity issues",
        "steps": [
            "Verify database server is running and accessible",
            "Check connection pool metrics",
            "Review application error logs for connection exceptions",
            "Validate connection string configuration",
            "Test network connectivity to database server"
        ],
        "relevant_tables": ["ApplicationLogs", "DatabaseMetrics", "ConnectionPool"],
        "relevant_columns": ["ExceptionMessage", "ConnectionCount", "FailureReason"],
        "query_templates": [
            "ApplicationLogs | where ExceptionMessage contains 'connection' | summarize count() by ExceptionType"
        ],
        "tags": ["database", "connectivity", "errors"],
        "priority": "high"
    }
]


# Mock known errors data
MOCK_KNOWN_ERRORS = [
    {
        "error_id": "err_001",
        "error_pattern": "50126.*conditional access policy",
        "error_type": "AuthenticationError",
        "description": "Conditional access policy blocking sign-in",
        "common_causes": [
            "User device not compliant",
            "Sign-in from blocked location",
            "MFA not completed",
            "App not approved for access"
        ],
        "resolution_steps": [
            "Check conditional access policies in Azure AD",
            "Verify user's device compliance status",
            "Review sign-in location restrictions",
            "Ensure MFA is properly configured"
        ],
        "query_suggestions": [
            "SigninLogs | where ResultType == 50126 | project TimeGenerated, UserPrincipalName, ConditionalAccessPolicies"
        ],
        "tags": ["authentication", "conditional-access", "azure-ad"]
    },
    {
        "error_id": "err_002",
        "error_pattern": "connection timeout|network unreachable",
        "error_type": "NetworkError",
        "description": "Network connectivity issues between services",
        "common_causes": [
            "Network firewall blocking traffic",
            "Service endpoint unavailable",
            "DNS resolution failure",
            "Load balancer health check failing"
        ],
        "resolution_steps": [
            "Verify network security group rules",
            "Check service endpoint health",
            "Test DNS resolution",
            "Review load balancer configuration"
        ],
        "query_suggestions": [
            "ApplicationLogs | where ExceptionMessage matches regex @'connection.*timeout|network.*unreachable'"
        ],
        "tags": ["network", "connectivity", "timeout"]
    }
]


# SOP Knowledge Agent
app_sop = FastAPI(
    title="Mock SOP Knowledge Agent",
    description="A2A agent providing SOP procedure context (Phase 2 Mock)",
    version="1.0.0-phase2"
)


@app_sop.get("/")
async def sop_root():
    """SOP agent information."""
    return {
        "agent_id": "mock_sop_knowledge",
        "agent_name": "Mock SOP Knowledge Provider",
        "agent_type": "knowledge",
        "knowledge_domain": "sop_procedures",
        "version": "1.0.0-phase2-mock",
        "capabilities": ["retrieve_sop_context", "search_procedures"],
        "status": "operational",
        "mock": True,
        "data_sources": ["mock_sop_database"]
    }


@app_sop.get("/health")
async def sop_health():
    """Health check."""
    return {
        "status": "healthy",
        "agent_id": "mock_sop_knowledge",
        "procedures_count": len(MOCK_SOP_PROCEDURES)
    }


@app_sop.post("/a2a/task", response_model=A2ATaskResponse)
async def sop_process_task(request: A2ATaskRequest):
    """Process A2A task for SOP knowledge retrieval."""
    logger.info(
        "SOP knowledge request received",
        task_id=request.task_id,
        action=request.action
    )
    
    if request.action == "retrieve_sop_context":
        return await retrieve_sop_context(request)
    elif request.action == "search_procedures":
        return await search_procedures(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


async def retrieve_sop_context(request: A2ATaskRequest) -> A2ATaskResponse:
    """Retrieve relevant SOP procedures based on query."""
    parameters = request.parameters or {}
    query = parameters.get("query", "").lower()
    top_k = parameters.get("top_k", 5)
    
    logger.info(
        "Retrieving SOP context",
        query_preview=query[:100],
        top_k=top_k
    )
    
    # Simple keyword matching (Phase 2)
    # Phase 4: Will use ChromaDB + embeddings for semantic search
    relevant_procedures = []
    for procedure in MOCK_SOP_PROCEDURES:
        # Score based on keyword matches
        score = 0.0
        query_words = query.split()
        
        for word in query_words:
            if word in procedure["title"].lower():
                score += 0.3
            if word in procedure["description"].lower():
                score += 0.2
            if word in " ".join(procedure["tags"]).lower():
                score += 0.15
        
        if score > 0:
            relevant_procedures.append({
                "procedure": procedure,
                "relevance_score": min(score, 1.0)
            })
    
    # Sort by relevance and limit
    relevant_procedures.sort(key=lambda x: x["relevance_score"], reverse=True)
    relevant_procedures = relevant_procedures[:top_k]
    
    # Build knowledge context
    knowledge_context = KnowledgeContext(
        source="sop",
        agent_id="mock_sop_knowledge",
        content={
            "procedures": [p["procedure"] for p in relevant_procedures],
            "relevance_scores": [p["relevance_score"] for p in relevant_procedures],
            "total_procedures_searched": len(MOCK_SOP_PROCEDURES)
        },
        relevance_score=relevant_procedures[0]["relevance_score"] if relevant_procedures else 0.0,
        metadata={
            "mock": True,
            "search_method": "keyword_matching"
        }
    )
    
    logger.info(
        "SOP context retrieved",
        procedures_found=len(relevant_procedures),
        top_relevance=knowledge_context.relevance_score
    )
    
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id="mock_sop_knowledge",
        status="completed",
        result={
            "knowledge_context": knowledge_context.model_dump(),
            "procedures_found": len(relevant_procedures)
        },
        cost_info={
            "llm_calls": 0,
            "llm_tokens": 0,
            "mcp_calls": 1,  # MCP call to ChromaDB (mock)
            "embedding_calls": 1,  # Mock embedding
            "execution_time_ms": 350.0,
            "cost_usd": 0.003
        }
    )


async def search_procedures(request: A2ATaskRequest) -> A2ATaskResponse:
    """Search procedures by tags or keywords."""
    parameters = request.parameters or {}
    tags = parameters.get("tags", [])
    
    matching_procedures = [
        p for p in MOCK_SOP_PROCEDURES
        if any(tag in p["tags"] for tag in tags)
    ]
    
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id="mock_sop_knowledge",
        status="completed",
        result={
            "procedures": matching_procedures,
            "count": len(matching_procedures)
        },
        cost_info={
            "llm_calls": 0,
            "llm_tokens": 0,
            "mcp_calls": 1,
            "execution_time_ms": 100.0,
            "cost_usd": 0.001
        }
    )


# Error Knowledge Agent
app_error = FastAPI(
    title="Mock Error Knowledge Agent",
    description="A2A agent providing known error context (Phase 2 Mock)",
    version="1.0.0-phase2"
)


@app_error.get("/")
async def error_root():
    """Error agent information."""
    return {
        "agent_id": "mock_error_knowledge",
        "agent_name": "Mock Error Knowledge Provider",
        "agent_type": "knowledge",
        "knowledge_domain": "known_errors",
        "version": "1.0.0-phase2-mock",
        "capabilities": ["retrieve_error_context", "search_errors"],
        "status": "operational",
        "mock": True,
        "data_sources": ["mock_error_database"]
    }


@app_error.get("/health")
async def error_health():
    """Health check."""
    return {
        "status": "healthy",
        "agent_id": "mock_error_knowledge",
        "errors_count": len(MOCK_KNOWN_ERRORS)
    }


@app_error.post("/a2a/task", response_model=A2ATaskResponse)
async def error_process_task(request: A2ATaskRequest):
    """Process A2A task for error knowledge retrieval."""
    logger.info(
        "Error knowledge request received",
        task_id=request.task_id,
        action=request.action
    )
    
    if request.action == "retrieve_error_context":
        return await retrieve_error_context(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


async def retrieve_error_context(request: A2ATaskRequest) -> A2ATaskResponse:
    """Retrieve relevant known errors based on query."""
    parameters = request.parameters or {}
    query = parameters.get("query", "").lower()
    top_k = parameters.get("top_k", 3)
    
    logger.info(
        "Retrieving error context",
        query_preview=query[:100],
        top_k=top_k
    )
    
    # Simple keyword matching
    relevant_errors = []
    for error in MOCK_KNOWN_ERRORS:
        score = 0.0
        query_words = query.split()
        
        for word in query_words:
            if word in error["description"].lower():
                score += 0.25
            if word in error["error_type"].lower():
                score += 0.3
            if word in " ".join(error["tags"]).lower():
                score += 0.15
        
        if score > 0:
            relevant_errors.append({
                "error": error,
                "relevance_score": min(score, 1.0)
            })
    
    relevant_errors.sort(key=lambda x: x["relevance_score"], reverse=True)
    relevant_errors = relevant_errors[:top_k]
    
    knowledge_context = KnowledgeContext(
        source="known_errors",
        agent_id="mock_error_knowledge",
        content={
            "errors": [e["error"] for e in relevant_errors],
            "relevance_scores": [e["relevance_score"] for e in relevant_errors],
            "total_errors_searched": len(MOCK_KNOWN_ERRORS)
        },
        relevance_score=relevant_errors[0]["relevance_score"] if relevant_errors else 0.0,
        metadata={
            "mock": True,
            "search_method": "keyword_matching"
        }
    )
    
    logger.info(
        "Error context retrieved",
        errors_found=len(relevant_errors),
        top_relevance=knowledge_context.relevance_score
    )
    
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id="mock_error_knowledge",
        status="completed",
        result={
            "knowledge_context": knowledge_context.model_dump(),
            "errors_found": len(relevant_errors)
        },
        cost_info={
            "llm_calls": 0,
            "llm_tokens": 0,
            "mcp_calls": 1,  # MCP call to ErrorDB (mock)
            "embedding_calls": 1,
            "execution_time_ms": 280.0,
            "cost_usd": 0.002
        }
    )


# Export both apps for separate deployment
__all__ = ["app_sop", "app_error"]


# For running SOP agent standalone
if __name__ == "__main__":
    import uvicorn
    import sys
    
    agent_type = sys.argv[1] if len(sys.argv) > 1 else "sop"
    
    if agent_type == "sop":
        uvicorn.run(
            "main:app_sop",
            host="0.0.0.0",
            port=9010,
            reload=True,
            log_level="info"
        )
    elif agent_type == "error":
        uvicorn.run(
            "main:app_error",
            host="0.0.0.0",
            port=9011,
            reload=True,
            log_level="info"
        )
    else:
        print(f"Unknown agent type: {agent_type}")
        print("Usage: python main.py [sop|error]")
