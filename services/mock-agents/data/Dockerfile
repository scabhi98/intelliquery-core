# Mock Data Agents Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy shared dependencies
COPY shared/ ./shared/

# Copy service code
COPY services/mock-agents/data/ ./services/mock-agents/data/

# Install dependencies
RUN pip install --no-cache-dir -r shared/requirements.txt
RUN pip install --no-cache-dir -r services/mock-agents/data/requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Expose ports (9020 for KQL, 9021 for SPL, 9022 for SQL)
EXPOSE 9020 9021 9022

# Default to KQL agent
ENV PLATFORM=kql

# Run application based on PLATFORM
CMD uvicorn services.mock_agents.data.src.main:app_${PLATFORM} --host 0.0.0.0 --port 902${PLATFORM:0:1}
