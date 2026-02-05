"""DOCX document parser."""

from typing import List

from docx import Document as DocxDocument

from shared.models.sop_models import SOPDocument, Procedure
from .document_parser import DocumentParser
from .procedure_extractor import extract_procedures


class DocxParser(DocumentParser):
    """Parse Word documents into procedures."""

    async def parse(self, document: SOPDocument) -> List[Procedure]:
        doc = DocxDocument(document.content)
        lines: List[str] = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                lines.append(paragraph.text.strip())
        text = "\n".join(lines)
        return extract_procedures(document, text)
