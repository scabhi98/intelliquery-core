# LexiQuery System Architecture

This document provides a structured view of the LexiQuery Phase 2 architecture using five focused diagrams: high-level architecture, data flow, component connectivity and extensibility, data processing flow, and observability.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph clients[Clients]
        ui["Web UI / API Clients"]
    end

    subgraph core[Core Engine]
        api["FastAPI Service"] --> orch["A2A Orchestrator"]
        orch --> cost["Cost Tracker"]
        orch --> cache["Cache Learning"]
    end

    subgraph agents[Agent Layer]
        planner["Planner Agent"]
        knowledge["Knowledge Agents (SOP, Error)"]
        data_agents["Data Agents (KQL, SPL, SQL)"]
    end

    subgraph mcp[MCP Tool Servers]
        chroma["ChromaDB"]
        errdb["Error DB"]
        azure["Azure Log Analytics"]
        splunk["Splunk"]
        sql["SQL DB"]
    end

    ui --> api
    orch --> planner
    orch --> knowledge
    orch --> data_agents
    knowledge --> chroma
    knowledge --> errdb
    data_agents --> azure
    data_agents --> splunk
    data_agents --> sql

    api --> ui
```

## Data Flow Diagram

```mermaid
flowchart TD
    request["User Request"] --> validate["Request Validation"]
    validate --> journey["Create Journey Context"]
    journey --> plan["Invoke Planner Agent"]
    plan --> knowledge["Fetch Knowledge Context"]
    knowledge --> generate["Invoke Data Agents"]
    generate --> validate_q["Validate & Rank Queries"]
    validate_q --> respond["Response to Client"]

    cache_hit["Cache Hit"] -.-> respond
    cache_check["Check Cache"] --> cache_hit
    validate --> cache_check

    cost_track["Track Cost & Usage"]
    plan --> cost_track
    knowledge --> cost_track
    generate --> cost_track
```

## Component Connections and Extensibility

```mermaid
flowchart LR
    subgraph shared[Shared Contracts]
        interfaces["Python ABC Interfaces"]
        models["Pydantic Models"]
    end

    subgraph a2a[A2A Network]
        core_engine["Core Engine"]
        planner["Planner Agent"]
        knowledge["Knowledge Agents"]
        data_agents["Data Agents"]
    end

    subgraph mcp[MCP Tooling]
        tool_api["MCP Tool Interface"]
        tools["Existing Tools"]
        add_tools["New Tool Adapters"]
    end

    interfaces --> core_engine
    interfaces --> planner
    interfaces --> knowledge
    interfaces --> data_agents
    models --> core_engine

    core_engine --> planner
    core_engine --> knowledge
    core_engine --> data_agents

    tool_api --> tools
    tool_api -.-> add_tools

    knowledge --> tool_api
    data_agents --> tool_api
```

## Data Processing Flow

```mermaid
flowchart TB
    start["Receive Query"] --> pending["State: PENDING"]
    pending --> discovering["State: DISCOVERING"]
    discovering --> planning["State: PLANNING"]
    planning --> gathering["State: GATHERING_KNOWLEDGE"]
    gathering --> generating["State: GENERATING_QUERIES"]
    generating --> validating["State: VALIDATING"]
    validating --> completed["State: COMPLETED"]
    validating --> failed["State: FAILED"]

    completed --> response["Return Query Response"]
    failed --> error_resp["Return Error Response"]
```

## Observability and Operations

```mermaid
flowchart LR
    subgraph runtime[Runtime Services]
        core["Core Engine"]
        agents["A2A Agents"]
        tools["MCP Tools"]
    end

    subgraph telemetry[Telemetry Signals]
        logs["Structured Logs"]
        metrics["Metrics"]
        traces["Traces"]
        health["Health Checks"]
        cost["Cost Events"]
    end

    subgraph sinks[Operational Sinks]
        files["Log Files"]
        dashboard["Ops Dashboard"]
        alerts["Alerting Rules"]
        reports["Cost Reports"]
    end

    core --> logs
    core --> metrics
    core --> traces
    core --> health
    core --> cost

    agents --> logs
    agents --> metrics
    agents --> traces
    agents --> health
    agents --> cost

    tools --> logs
    tools --> metrics
    tools --> traces
    tools --> health

    logs --> files
    metrics --> dashboard
    traces --> dashboard
    health --> alerts
    cost --> reports
```

## Notes

- Agents communicate via A2A protocol using shared interfaces and models.
- MCP tool servers provide platform-specific access (log analytics, vector DBs, SQL).
- Cost tracking aggregates usage per journey, per service, and per model.
- Observability is structured around logs, metrics, traces, and health endpoints.
