"""
Cross-encoder reranker — second-stage relevance scoring.

Role in the pipeline
--------------------
Hybrid retrieval (dense + sparse) optimises for recall: it returns a broad
candidate set cheaply.  The reranker optimises for precision: it scores each
(query, chunk) pair jointly with a cross-encoder, which is much more accurate
but too slow to run over the full corpus.

Typical usage
-------------
    hybrid_hits = await hybrid_retriever.search(query, db, top_k=40)
    final_hits  = reranker.rerank(query, hybrid_hits, top_n=10)

Two backends
------------
CrossEncoderReranker
    Local sentence-transformers cross-encoder.  No API key; runs on CPU/GPU.
    Recommended models:
      - "cross-encoder/ms-marco-MiniLM-L-6-v2"   fast, good quality
      - "BAAI/bge-reranker-base"                  better quality, ~2× slower
    Raw logits are sigmoid-normalised to [0, 1] before returning.

CohereReranker
    Cohere Rerank API.  Higher quality than small local models; requires an
    API key.  Scores are already in [0, 1].

Both backends tag returned hits with ``source="reranked"`` and update each
hit's ``score`` to the reranker's relevance score.
"""
from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod

from .dense import DenseHit

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        hits: list[DenseHit],
        top_n: int,
    ) -> list[DenseHit]:
        """Rerank retrieved hits for a query.

        Parameters
        ----------
        query
            Free-text query string.
        hits
            Candidate hits to rerank in place.
        top_n
            Maximum number of reranked hits to return.

        Returns
        -------
        list[DenseHit]
            Top reranked hits sorted by descending relevance.
        """


# ── Cross-encoder (local) ─────────────────────────────────────────────────────

class CrossEncoderReranker(BaseReranker):
    """
    Reranker backed by a sentence-transformers ``CrossEncoder``.

    Args:
        model: HuggingFace model ID.
        batch_size: Pairs per forward pass.  Reduce if OOM on CPU.
        max_length: Token limit per (query, document) pair.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder as _CE
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoderReranker: "
                "pip install sentence-transformers"
            ) from e

        self._model = _CE(model, max_length=max_length)
        self._batch_size = batch_size
        logger.info("CrossEncoderReranker loaded model %r", model)

    def rerank(self, query: str, hits: list[DenseHit], top_n: int) -> list[DenseHit]:
        if not hits:
            return []

        t0 = time.perf_counter()
        pairs = [(query, hit.text) for hit in hits]

        raw_scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        for hit, raw in zip(hits, raw_scores):
            hit.score = _sigmoid(float(raw))
            hit.source = "reranked"

        hits.sort(key=lambda h: h.score, reverse=True)
        result = hits[:top_n]

        logger.debug(
            "CrossEncoder reranked %d → %d hits in %d ms (top=%.4f)",
            len(hits), len(result),
            int((time.perf_counter() - t0) * 1000),
            result[0].score if result else 0,
        )
        return result


# ── Cohere Rerank API ─────────────────────────────────────────────────────────

class CohereReranker(BaseReranker):
    """
    Reranker backed by the Cohere Rerank API.

    Args:
        model: Cohere rerank model name.
        api_key: Overrides the ``COHERE_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: str | None = None,
    ) -> None:
        try:
            import cohere
        except ImportError as e:
            raise RuntimeError(
                "cohere package is required for CohereReranker: pip install cohere"
            ) from e

        import os
        self._client = cohere.Client(api_key or os.getenv("COHERE_API_KEY", ""))
        self._model = model
        logger.info("CohereReranker using model %r", model)

    def rerank(self, query: str, hits: list[DenseHit], top_n: int) -> list[DenseHit]:
        if not hits:
            return []

        t0 = time.perf_counter()
        documents = [hit.text for hit in hits]

        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_n=top_n,
        )

        result: list[DenseHit] = []
        for r in response.results:
            hit = hits[r.index]
            hit.score = float(r.relevance_score)  # already in [0, 1]
            hit.source = "reranked"
            result.append(hit)

        logger.debug(
            "CohereReranker reranked %d → %d hits in %d ms (top=%.4f)",
            len(hits), len(result),
            int((time.perf_counter() - t0) * 1000),
            result[0].score if result else 0,
        )
        return result


# ── No-op reranker (passthrough) ──────────────────────────────────────────────

class IdentityReranker(BaseReranker):
    """
    Passthrough — returns the first *top_n* hits unchanged.
    Useful as a default when no reranker is configured, or in tests that
    want to isolate retrieval from reranking.
    """

    def rerank(self, query: str, hits: list[DenseHit], top_n: int) -> list[DenseHit]:
        return hits[:top_n]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_reranker(
    backend: str = "cross-encoder",
    model: str | None = None,
    **kwargs,
) -> BaseReranker:
    """Create a reranker backend.

    Parameters
    ----------
    backend
        Backend name. Supported values are ``"cross-encoder"``, ``"cohere"``, and ``"none"``.
    model
        Optional backend-specific model override.
    **kwargs
        Additional keyword arguments forwarded to the backend constructor.

    Returns
    -------
    BaseReranker
        Configured reranker implementation.

    Raises
    ------
    ValueError
        If ``backend`` is not a supported reranker type.
    """
    if backend == "cross-encoder":
        return CrossEncoderReranker(
            model=model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            **kwargs,
        )
    if backend == "cohere":
        return CohereReranker(model=model or "rerank-english-v3.0", **kwargs)
    if backend == "none":
        return IdentityReranker()
    raise ValueError(
        f"Unknown reranker backend: {backend!r}. "
        "Choose 'cross-encoder', 'cohere', or 'none'."
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Map a raw cross-encoder logit to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))
