"""Semantic search engine for SOP procedures."""

from typing import List, Optional

from shared.models.sop_models import Procedure

from ..vector_store.vector_store import VectorStoreManager


class SemanticSearchEngine:
    """Semantic search over SOP procedures."""

    def __init__(self, vector_store: VectorStoreManager) -> None:
        self.vector_store = vector_store

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> List[Procedure]:
        results = await self.vector_store.search(query, top_k)
        if not filters:
            return results

        filtered: List[Procedure] = []
        for proc in results:
            if self._matches_filters(proc, filters):
                filtered.append(proc)
        return filtered

    @staticmethod
    def _matches_filters(procedure: Procedure, filters: dict) -> bool:
        tags = set(procedure.tags or [])
        filter_tags = set(filters.get("tags", []))
        if filter_tags and not filter_tags.intersection(tags):
            return False
        return True
