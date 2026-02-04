"""Logging configuration using structlog for structured logging."""

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    log_level: str = "INFO",
    service_name: Optional[str] = None,
    json_format: bool = True
) -> None:
    """Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service_name: Name of the service for log context
        json_format: Whether to use JSON format (True for production)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )
    
    # Build processor chain
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add service name if provided
    if service_name:
        processors.insert(
            0,
            structlog.processors.add_log_level
        )
    
    # Add appropriate renderer
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Set service name in context if provided
    if service_name:
        structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: Optional[str] = None, **initial_values: Any) -> structlog.BoundLogger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        **initial_values: Initial context values to bind
        
    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    
    if initial_values:
        logger = logger.bind(**initial_values)
    
    return logger


# Example usage patterns:
"""
# Basic usage
logger = get_logger(__name__)
logger.info("service_started", port=8000, version="1.0.0")

# With context
logger = get_logger(__name__, service="query-generator")
logger.info("query_generated", platform="kql", latency_ms=234.5)

# With journey tracking
logger.info(
    "llm_called",
    journey_id=str(journey_id),
    model="llama3",
    tokens=225,
    cost_usd=0.002
)

# Error logging
try:
    result = await service.call()
except Exception as e:
    logger.error(
        "service_call_failed",
        service="sop-engine",
        error=str(e),
        exc_info=True
    )
"""
