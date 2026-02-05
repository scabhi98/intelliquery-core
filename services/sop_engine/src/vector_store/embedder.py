"""Ollama embedding client."""

import asyncio
from typing import List

import ollama
import structlog

from shared.utils.exceptions import EmbeddingException


logger = structlog.get_logger()


class OllamaEmbedder:
    """Embedding client backed by Ollama."""

    def __init__(self, host: str, model: str) -> None:
        self.model = model
        self.client = ollama.Client(host=host)

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text string."""
        try:
            response = await asyncio.to_thread(
                self.client.embeddings,
                model=self.model,
                prompt=text
            )
            return response.get("embedding", [])
        except Exception as exc:
            logger.error("Embedding generation failed", error=str(exc))
            raise EmbeddingException("Embedding generation failed", details={"error": str(exc)})

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(await self.embed_text(text))
        return embeddings
