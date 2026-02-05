"""Procedure extraction helper."""

import re
from typing import List
from uuid import uuid4

from shared.models.sop_models import SOPDocument, Procedure


_HEADING_PATTERN = re.compile(r"^(#{1,6}\s+|Procedure:\s+|Step\s+\d+\s*:)" , re.IGNORECASE)
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


def _split_sections(lines: List[str]) -> List[List[str]]:
    sections: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if _HEADING_PATTERN.match(line) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append(current)
    return sections


def extract_procedures(document: SOPDocument, content: str) -> List[Procedure]:
    """Extract procedure objects from content text."""
    lines = [line.rstrip() for line in content.splitlines() if line.strip()]
    if not lines:
        return [
            Procedure(
                id=str(uuid4()),
                document_id=document.id,
                title=document.title,
                description=None,
                steps=[],
                tags=document.tags,
                metadata=document.metadata
            )
        ]

    sections = _split_sections(lines)
    procedures: List[Procedure] = []

    for section in sections:
        heading = section[0].lstrip("# ") if section else document.title
        title = heading.replace("Procedure:", "").strip() or document.title
        steps: List[str] = []
        description_parts: List[str] = []

        for line in section[1:]:
            if _LIST_ITEM_PATTERN.match(line):
                steps.append(_LIST_ITEM_PATTERN.sub("", line).strip())
            else:
                description_parts.append(line.strip())

        description = " ".join(description_parts).strip() if description_parts else None

        procedures.append(
            Procedure(
                id=str(uuid4()),
                document_id=document.id,
                title=title,
                description=description,
                steps=steps,
                tags=document.tags,
                metadata=document.metadata
            )
        )

    return procedures
