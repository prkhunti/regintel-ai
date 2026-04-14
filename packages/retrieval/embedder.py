"""
Provider-abstracted embedding generator.

Supports:
  - OpenAI (text-embedding-3-small / text-embedding-3-large)
  - Anthropic voyage embeddings (stub — add voyage-ai client when needed)
  - Local sentence-transformers (stub)
  - Random (deterministic stub — for testing without any API key)

All providers expose the same interface:
    embed_texts(texts: list[str]) -> list[list[float]]
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY_S = 2.0


def _with_retry(fn, *args, attempts: int = _RETRY_ATTEMPTS, delay: float = _RETRY_DELAY_S, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.warning("Embedding attempt %d/%d failed: %s — retrying in %.1fs", attempt, attempts, exc, delay)
            time.sleep(delay * attempt)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (same order)."""

    def embed_single(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package is required: pip install openai") from e

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            # OpenAI replaces newlines with spaces internally, but let's be explicit
            batch = [t.replace("\n", " ") for t in batch]

            response = _with_retry(
                self._client.embeddings.create,
                input=batch,
                model=self._model,
            )
            batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            all_embeddings.extend(batch_embeddings)
            logger.debug("Embedded batch %d-%d via OpenAI", i, i + len(batch))

        return all_embeddings


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedder — no API key required. Useful for dev / air-gapped envs."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("sentence-transformers package is required") from e

        self._model = SentenceTransformer(model)
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class RandomEmbedder(BaseEmbedder):
    """
    Stub embedder — returns deterministic random vectors.
    Zero external dependencies, zero API calls.
    Use LLM_PROVIDER=random in .env for development / CI without API keys.
    """

    def __init__(self, embedding_dim: int = 1536, seed: int = 42) -> None:
        import random as _random
        self._dim = embedding_dim
        self._rng = _random.Random(seed)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[self._rng.uniform(-1.0, 1.0) for _ in range(self._dim)] for _ in texts]


def get_embedder(
    provider: str = "openai",
    model: str | None = None,
    api_key: str | None = None,
    batch_size: int = 100,
) -> BaseEmbedder:
    """
    Factory — returns the correct embedder for the configured provider.

    Args:
        provider: "openai" | "local" | "random"
        model: Model name override (uses provider defaults if None).
        api_key: API key override (falls back to env vars).
        batch_size: Texts per API call.
    """
    if provider == "openai":
        return OpenAIEmbedder(
            model=model or "text-embedding-3-small",
            batch_size=batch_size,
            api_key=api_key,
        )
    if provider == "local":
        return SentenceTransformerEmbedder(
            model=model or "BAAI/bge-small-en-v1.5",
            batch_size=batch_size,
        )
    if provider == "random":
        return RandomEmbedder(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
        )
    raise ValueError(f"Unknown embedding provider: {provider!r}. Choose 'openai', 'local', or 'random'.")
