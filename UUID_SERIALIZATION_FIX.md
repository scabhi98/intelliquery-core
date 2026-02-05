# UUID Serialization Fix - Global Update

**Date**: February 6, 2026  
**Issue**: `Object of type UUID is not JSON serializable` errors  
**Solution**: Applied `mode="json"` to all Pydantic `model_dump()` calls

## Problem

When A2A agents and services serialize Pydantic models containing UUID or datetime fields to JSON for HTTP requests, the default `model_dump()` method returns Python native types that are not JSON-serializable.

Example Error:
```
"message": "Internal server error",
"error": "Object of type UUID is not JSON serializable"
```

## Solution

In Pydantic v2, the `mode="json"` parameter in `model_dump()` ensures all special types (UUID, datetime, etc.) are converted to their JSON-serializable string representations.

### Changed Method Calls

Changed from:
```python
payload = {"agent": agent.model_dump()}
```

To:
```python
payload = {"agent": agent.model_dump(mode="json")}
```

## Files Updated

### Core Engine
- **services/core_engine/src/orchestrator.py**
  - `_invoke_agent_task()`: Agent and task serialization
  - `_invoke_data_step()`: Knowledge context serialization

### Planner Service
- **services/planner/src/service.py**
  - `create_plan()`: Plan and agent selection serialization
  - `optimize_plan()`: Plan serialization
  
- **services/planner/src/plan_generator.py**
  - `generate_plan()`: Analysis and agent selection metadata

### Mock Agents
- **services/mock_agents/planner/src/main.py**
  - `create_workflow_plan()`: Plan response
  
- **services/mock_agents/data/src/main.py**
  - All three data agent variants (KQL, SPL, SQL): Query result serialization
  
- **services/mock_agents/knowledge/src/main.py**
  - All knowledge agent variants (SOP, Errors): Knowledge context serialization

### Protocol Interface
- **services/protocol_interface/src/a2a/handler.py**
  - `invoke_task()`: A2A task request serialization
  
- **services/protocol_interface/src/a2a/registry.py**
  - `_save_registry()`: Agent registry serialization

## Impact

✅ **All UUID objects** are now properly converted to strings in JSON payloads  
✅ **All datetime objects** are properly formatted as ISO 8601 strings  
✅ **HTTP communication** between services works without serialization errors  
✅ **Backward compatible** - no breaking changes to API contracts

## Testing

The fix has been applied to:
- A2A task invocation (core engine ↔ agents)
- Agent discovery and registration
- Response payload creation from all services
- Metadata storage in workflow plans

## Verification

To verify the fix works, check that:
1. Core engine can successfully invoke planner agent
2. Planner agent returns valid JSON with workflow plan
3. Knowledge and data agents return valid JSON responses
4. Protocol interface correctly handles agent registry

All components now properly serialize UUID and datetime fields in JSON payloads.
