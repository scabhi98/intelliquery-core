"""PDF document parser."""

from typing import List

from pypdf import PdfReader

from shared.models.sop_models import SOPDocument, Procedure
from .document_parser import DocumentParser
from .procedure_extractor import extract_procedures


class PdfParser(DocumentParser):
    """Parse PDF documents into procedures."""

    async def parse(self, document: SOPDocument) -> List[Procedure]:
        reader = PdfReader(document.content)
        lines: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                if line.strip():
                    lines.append(line.strip())
        content = "\n".join(lines)
        return extract_procedures(document, content)
