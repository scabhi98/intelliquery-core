"""Lightweight schema analyzer with optional GraphRAG integration."""

import re
from typing import Dict, List

import structlog

from shared.utils.exceptions import SchemaExtractionException


logger = structlog.get_logger()


class SchemaAnalyzer:
    """Extract schema context from SOP documents."""

    def __init__(self) -> None:
        self._entities: Dict[str, Dict[str, str]] = {}
        self._relationships: List[Dict[str, str]] = []

    def update_from_text(self, content: str) -> None:
        """Update schema entities from content text."""
        try:
            table_matches = re.findall(r"\bTable\s+([A-Za-z0-9_]+)", content)
            for table in table_matches:
                self._entities.setdefault(table, {"type": "table"})

            column_matches = re.findall(r"\b([A-Za-z0-9_]+)\s*\(([^)]+)\)", content)
            for table, columns in column_matches:
                self._entities.setdefault(table, {"type": "table"})
                for column in columns.split(","):
                    col_name = column.strip().split()[0]
                    self._entities.setdefault(col_name, {"type": "column"})
                    self._relationships.append(
                        {
                            "source": table,
                            "target": col_name,
                            "type": "has_column"
                        }
                    )
        except Exception as exc:
            logger.error("Schema extraction failed", error=str(exc))
            raise SchemaExtractionException("Schema extraction failed", details={"error": str(exc)})

    def get_context(self) -> Dict[str, List[Dict[str, str]]]:
        """Return current schema context."""
        return {
            "entities": [
                {"name": name, **meta}
                for name, meta in self._entities.items()
            ],
            "relationships": self._relationships
        }
