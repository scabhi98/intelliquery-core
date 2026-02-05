"""Mock A2A Data Agents for Phase 2 Testing."""

from typing import Dict, List
from uuid import UUID
import os

from fastapi import FastAPI, HTTPException
import structlog

from shared.models.a2a_models import A2ATaskRequest, A2ATaskResponse, QueryResult
from shared.utils.logging_config import setup_logging

# Setup structured logging with file output
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
log_file = os.path.join(project_root, "logs", "data-agents.log")
setup_logging(log_file=log_file)
logger = structlog.get_logger()


# KQL Data Agent
app_kql = FastAPI(
    title="Mock KQL Data Agent",
    description="A2A agent for KQL query generation (Phase 2 Mock)",
    version="1.0.0-phase2"
)


@app_kql.get("/")
async def kql_root():
    """KQL agent information."""
    return {
        "agent_id": "mock_kql_data",
        "agent_name": "Mock KQL Data Agent",
        "agent_type": "data",
        "platform": "kql",
        "version": "1.0.0-phase2-mock",
        "capabilities": ["generate_query", "execute_query", "generate_and_execute"],
        "status": "operational",
        "mock": True
    }


@app_kql.get("/health")
async def kql_health():
    """Health check."""
    return {"status": "healthy", "agent_id": "mock_kql_data"}


@app_kql.post("/a2a/task", response_model=A2ATaskResponse)
async def kql_process_task(request: A2ATaskRequest):
    """Process A2A task for KQL query generation."""
    logger.info(
        "KQL data agent request received",
        task_id=request.task_id,
        action=request.action
    )
    
    if request.action == "generate_and_execute":
        return await generate_kql_query(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


async def generate_kql_query(request: A2ATaskRequest) -> A2ATaskResponse:
    """Generate KQL query based on request and knowledge context."""
    parameters = request.parameters or {}
    natural_language = parameters.get("natural_language", "")
    knowledge_context = parameters.get("knowledge_context", {})
    execute = parameters.get("execute", False)
    
    logger.info(
        "Generating KQL query",
        query_preview=natural_language[:100],
        has_knowledge=bool(knowledge_context),
        execute=execute
    )
    
    # Extract relevant tables from SOP knowledge
    relevant_tables = []
    if "sop" in knowledge_context:
        sop_data = knowledge_context["sop"].get("content", {})
        procedures = sop_data.get("procedures", [])
        for proc in procedures:
            relevant_tables.extend(proc.get("relevant_tables", []))
    
    # Generate KQL query based on keywords
    query = generate_kql_template(natural_language, relevant_tables)
    
    # Build query result
    query_result = QueryResult(
        platform="kql",
        agent_id="mock_kql_data",
        query=query,
        confidence=0.78,
        explanation=f"Generated KQL query using {len(relevant_tables)} table(s) from SOP knowledge",
        optimizations=["table_filtering", "time_window"],
        data=None,  # Phase 2: No execution
        row_count=None,
        execution_time_ms=None,
        cost_info={
            "llm_tokens": 520,
            "mcp_calls": 0,
            "cost_usd": 0.006
        },
        warnings=[],
        metadata={
            "mock": True,
            "tables_used": relevant_tables[:3] if relevant_tables else ["SigninLogs"],
            "knowledge_sources": list(knowledge_context.keys())
        }
    )
    
    logger.info(
        "KQL query generated",
        confidence=query_result.confidence,
        query_length=len(query)
    )
    
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id="mock_kql_data",
        status="completed",
        result={
            "query_result": query_result.model_dump()
        },
        cost_info={
            "llm_calls": 1,
            "llm_tokens": 520,
            "mcp_calls": 0,
            "execution_time_ms": 850.0,
            "cost_usd": 0.006
        }
    )


def generate_kql_template(natural_language: str, tables: List[str]) -> str:
    """Generate KQL query template based on keywords."""
    nl_lower = natural_language.lower()
    
    # Determine primary table
    table = tables[0] if tables else "SigninLogs"
    
    # Build query based on keywords
    query_lines = [f"{table}"]
    
    # Time window
    if "hour" in nl_lower:
        query_lines.append("| where TimeGenerated > ago(1h)")
    elif "day" in nl_lower or "today" in nl_lower:
        query_lines.append("| where TimeGenerated > ago(1d)")
    else:
        query_lines.append("| where TimeGenerated > ago(24h)")
    
    # Filters based on keywords
    if "fail" in nl_lower or "error" in nl_lower:
        if table == "SigninLogs":
            query_lines.append("| where ResultType != 0")
        else:
            query_lines.append("| where Status == 'Failure' or Level == 'Error'")
    
    if "success" in nl_lower:
        if table == "SigninLogs":
            query_lines.append("| where ResultType == 0")
        else:
            query_lines.append("| where Status == 'Success'")
    
    # Aggregations
    if "count" in nl_lower or "how many" in nl_lower:
        if "user" in nl_lower:
            query_lines.append("| summarize count() by UserPrincipalName")
        elif "ip" in nl_lower:
            query_lines.append("| summarize count() by IPAddress")
        else:
            query_lines.append("| summarize count()")
    else:
        # Default projection
        query_lines.append("| project TimeGenerated, UserPrincipalName, IPAddress, Status, ResultType")
    
    # Limit
    if "summarize" not in query_lines[-1]:
        query_lines.append("| take 100")
    
    # Add comment with original query
    query_lines.append(f"// Generated for: {natural_language[:80]}")
    
    return "\n".join(query_lines)


# SPL Data Agent
app_spl = FastAPI(
    title="Mock SPL Data Agent",
    description="A2A agent for SPL query generation (Phase 2 Mock)",
    version="1.0.0-phase2"
)


@app_spl.get("/")
async def spl_root():
    """SPL agent information."""
    return {
        "agent_id": "mock_spl_data",
        "agent_name": "Mock SPL Data Agent",
        "agent_type": "data",
        "platform": "spl",
        "version": "1.0.0-phase2-mock",
        "capabilities": ["generate_query", "execute_query", "generate_and_execute"],
        "status": "operational",
        "mock": True
    }


@app_spl.get("/health")
async def spl_health():
    """Health check."""
    return {"status": "healthy", "agent_id": "mock_spl_data"}


@app_spl.post("/a2a/task", response_model=A2ATaskResponse)
async def spl_process_task(request: A2ATaskRequest):
    """Process A2A task for SPL query generation."""
    logger.info(
        "SPL data agent request received",
        task_id=request.task_id,
        action=request.action
    )
    
    if request.action == "generate_and_execute":
        return await generate_spl_query(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


async def generate_spl_query(request: A2ATaskRequest) -> A2ATaskResponse:
    """Generate SPL query based on request and knowledge context."""
    parameters = request.parameters or {}
    natural_language = parameters.get("natural_language", "")
    knowledge_context = parameters.get("knowledge_context", {})
    
    logger.info(
        "Generating SPL query",
        query_preview=natural_language[:100]
    )
    
    query = generate_spl_template(natural_language)
    
    query_result = QueryResult(
        platform="spl",
        agent_id="mock_spl_data",
        query=query,
        confidence=0.72,
        explanation="Generated SPL query using mock templates",
        optimizations=["index_specification", "time_range"],
        data=None,
        row_count=None,
        execution_time_ms=None,
        cost_info={
            "llm_tokens": 480,
            "mcp_calls": 0,
            "cost_usd": 0.005
        },
        warnings=[],
        metadata={"mock": True}
    )
    
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id="mock_spl_data",
        status="completed",
        result={
            "query_result": query_result.model_dump()
        },
        cost_info={
            "llm_calls": 1,
            "llm_tokens": 480,
            "mcp_calls": 0,
            "execution_time_ms": 820.0,
            "cost_usd": 0.005
        }
    )


def generate_spl_template(natural_language: str) -> str:
    """Generate SPL query template."""
    nl_lower = natural_language.lower()
    
    query_parts = ["index=main"]
    
    # Source type
    if "auth" in nl_lower or "login" in nl_lower:
        query_parts.append('sourcetype="authentication"')
    else:
        query_parts.append('sourcetype="application"')
    
    # Time range
    if "hour" in nl_lower:
        query_parts.append("earliest=-1h")
    else:
        query_parts.append("earliest=-24h")
    
    # Filters
    if "fail" in nl_lower or "error" in nl_lower:
        query_parts.append('Status="Failure" OR Level="ERROR"')
    
    # Build search
    search_line = " ".join(query_parts)
    
    # Stats
    if "count" in nl_lower:
        if "user" in nl_lower:
            stats_line = "| stats count by user"
        elif "ip" in nl_lower:
            stats_line = "| stats count by src_ip"
        else:
            stats_line = "| stats count"
        
        query = f"{search_line}\n{stats_line}\n| sort -count"
    else:
        query = f"{search_line}\n| head 100"
    
    query += f"\n# Generated for: {natural_language[:70]}"
    
    return query


# SQL Data Agent
app_sql = FastAPI(
    title="Mock SQL Data Agent",
    description="A2A agent for SQL query generation (Phase 2 Mock)",
    version="1.0.0-phase2"
)


@app_sql.get("/")
async def sql_root():
    """SQL agent information."""
    return {
        "agent_id": "mock_sql_data",
        "agent_name": "Mock SQL Data Agent",
        "agent_type": "data",
        "platform": "sql",
        "version": "1.0.0-phase2-mock",
        "capabilities": ["generate_query", "execute_query", "generate_and_execute"],
        "status": "operational",
        "mock": True
    }


@app_sql.get("/health")
async def sql_health():
    """Health check."""
    return {"status": "healthy", "agent_id": "mock_sql_data"}


@app_sql.post("/a2a/task", response_model=A2ATaskResponse)
async def sql_process_task(request: A2ATaskRequest):
    """Process A2A task for SQL query generation."""
    logger.info(
        "SQL data agent request received",
        task_id=request.task_id,
        action=request.action
    )
    
    if request.action == "generate_and_execute":
        return await generate_sql_query(request)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {request.action}"
        )


async def generate_sql_query(request: A2ATaskRequest) -> A2ATaskResponse:
    """Generate SQL query based on request and knowledge context."""
    parameters = request.parameters or {}
    natural_language = parameters.get("natural_language", "")
    knowledge_context = parameters.get("knowledge_context", {})
    
    logger.info(
        "Generating SQL query",
        query_preview=natural_language[:100]
    )
    
    query = generate_sql_template(natural_language)
    
    query_result = QueryResult(
        platform="sql",
        agent_id="mock_sql_data",
        query=query,
        confidence=0.80,
        explanation="Generated SQL query using mock templates",
        optimizations=["indexed_columns", "time_partition"],
        data=None,
        row_count=None,
        execution_time_ms=None,
        cost_info={
            "llm_tokens": 510,
            "mcp_calls": 0,
            "cost_usd": 0.0055
        },
        warnings=[],
        metadata={"mock": True}
    )
    
    return A2ATaskResponse(
        task_id=request.task_id,
        agent_id="mock_sql_data",
        status="completed",
        result={
            "query_result": query_result.model_dump()
        },
        cost_info={
            "llm_calls": 1,
            "llm_tokens": 510,
            "mcp_calls": 0,
            "execution_time_ms": 800.0,
            "cost_usd": 0.0055
        }
    )


def generate_sql_template(natural_language: str) -> str:
    """Generate SQL query template."""
    nl_lower = natural_language.lower()
    
    # Determine table
    if "auth" in nl_lower or "login" in nl_lower:
        table = "authentication_logs"
        columns = "timestamp, username, ip_address, status"
    else:
        table = "application_logs"
        columns = "timestamp, service_name, level, message"
    
    # Build WHERE clause
    where_parts = []
    
    # Time filter
    if "hour" in nl_lower:
        where_parts.append("timestamp > NOW() - INTERVAL '1 HOUR'")
    else:
        where_parts.append("timestamp > NOW() - INTERVAL '24 HOURS'")
    
    # Status filter
    if "fail" in nl_lower or "error" in nl_lower:
        where_parts.append("status = 'FAILURE' OR level = 'ERROR'")
    elif "success" in nl_lower:
        where_parts.append("status = 'SUCCESS'")
    
    where_clause = "\n  AND ".join(where_parts)
    
    # Build query
    if "count" in nl_lower:
        if "user" in nl_lower:
            query = f"""SELECT username, COUNT(*) as count
FROM {table}
WHERE {where_clause}
GROUP BY username
ORDER BY count DESC
LIMIT 100"""
        else:
            query = f"""SELECT COUNT(*) as total_count
FROM {table}
WHERE {where_clause}"""
    else:
        query = f"""SELECT {columns}
FROM {table}
WHERE {where_clause}
ORDER BY timestamp DESC
LIMIT 100"""
    
    query += f"\n-- Generated for: {natural_language[:70]}"
    
    return query


# Export all apps
__all__ = ["app_kql", "app_spl", "app_sql"]


if __name__ == "__main__":
    import uvicorn
    import sys
    
    platform = sys.argv[1] if len(sys.argv) > 1 else "kql"
    
    if platform == "kql":
        uvicorn.run("main:app_kql", host="0.0.0.0", port=9020, reload=True)
    elif platform == "spl":
        uvicorn.run("main:app_spl", host="0.0.0.0", port=9021, reload=True)
    elif platform == "sql":
        uvicorn.run("main:app_sql", host="0.0.0.0", port=9022, reload=True)
    else:
        print(f"Unknown platform: {platform}")
        print("Usage: python main.py [kql|spl|sql]")
