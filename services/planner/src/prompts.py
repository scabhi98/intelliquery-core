"""Prompt templates for Planner LLM."""

QUERY_ANALYSIS_PROMPT = """You are a query planning expert for a multi-platform log and database query system.

Analyze the following user query and provide structured analysis.

USER QUERY:
{natural_language}

AVAILABLE PLATFORMS:
{platforms_summary}

SCHEMA HINTS:
{schema_hints}

USER CONTEXT:
{user_context}

Return ONLY valid JSON in this format:
{{
  "intent": "diagnostic|monitoring|security|analytics|investigation",
  "complexity": "simple|moderate|complex",
  "required_platforms": ["kql", "spl", "sql"],
  "confidence": 0.0,
  "reasoning": "short explanation",
  "entities": {{
    "tables": [],
    "fields": [],
    "time_range": ""
  }},
  "knowledge_requirements": {{
    "requires_sop": true,
    "requires_error_knowledge": false,
    "requires_schema": false
  }},
  "query_characteristics": {{
    "is_aggregation": false,
    "is_time_series": false,
    "has_joins": false,
    "complexity_factors": []
  }}
}}
"""
