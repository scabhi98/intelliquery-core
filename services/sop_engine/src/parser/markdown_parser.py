"""Markdown document parser."""

from typing import List

from shared.models.sop_models import SOPDocument, Procedure
from .document_parser import DocumentParser
from .procedure_extractor import extract_procedures


class MarkdownParser(DocumentParser):
    """Parse Markdown SOP documents."""

    async def parse(self, document: SOPDocument) -> List[Procedure]:
        return extract_procedures(document, document.content)
