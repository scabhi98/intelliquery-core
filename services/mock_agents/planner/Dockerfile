# Mock Planner Agent Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy shared dependencies
COPY shared/ ./shared/

# Copy service code
COPY services/mock-agents/planner/ ./services/mock-agents/planner/

# Install dependencies
RUN pip install --no-cache-dir -r shared/requirements.txt
RUN pip install --no-cache-dir -r services/mock-agents/planner/requirements.txt

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

# Run application
CMD ["uvicorn", "services.mock_agents.planner.src.main:app", "--host", "0.0.0.0", "--port", "9000"]
