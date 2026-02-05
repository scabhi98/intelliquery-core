# Mock Knowledge Agents Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy shared dependencies
COPY shared/ ./shared/

# Copy service code
COPY services/mock-agents/knowledge/ ./services/mock-agents/knowledge/

# Install dependencies
RUN pip install --no-cache-dir -r shared/requirements.txt
RUN pip install --no-cache-dir -r services/mock-agents/knowledge/requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Expose ports (9010 for SOP, 9011 for Error)
EXPOSE 9010 9011

# Default to SOP agent
ENV AGENT_TYPE=sop

# Run application based on AGENT_TYPE
CMD uvicorn services.mock_agents.knowledge.src.main:app_${AGENT_TYPE} --host 0.0.0.0 --port 901${AGENT_TYPE:0:1}
