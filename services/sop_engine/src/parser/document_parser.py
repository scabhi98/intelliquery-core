"""Abstract document parser interface."""

from abc import ABC, abstractmethod
from typing import List

from shared.models.sop_models import SOPDocument, Procedure


class DocumentParser(ABC):
    """Base class for document parsers."""

    @abstractmethod
    async def parse(self, document: SOPDocument) -> List[Procedure]:
        """Parse a document into procedures."""
        raise NotImplementedError
