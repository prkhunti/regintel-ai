"""
Hybrid retriever — fuses dense (pgvector) and sparse (BM25 / FTS) results.

Fusion strategies
-----------------
RRF  (Reciprocal Rank Fusion)  — default
    score = Σ 1 / (k + rank_i)   for each retriever that returned the chunk.
    k=60 is the standard constant from Cormack et al. 2009.
    Does NOT require normalised scores; rank position is the only input.
    Best choice when dense and sparse scores live on different scales.

Weighted sum
    score = alpha * dense_score + (1 - alpha) * sparse_score
    Both inputs must already be in [0, 1]:
      - dense scores are cosine similarities (always [0, 1])
      - sparse scores are normalised by SparseRetriever before returning
    Use when you want explicit control over the dense/sparse balance.

Returned hits have ``source="hybrid"`` and ``score`` set to the fused value.
Hits that appear in only one retriever are still included — their missing-side
score is treated as 0 for weighted sum, and their rank contribution from the
missing side is simply absent for RRF.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .dense import DenseHit, DenseRetriever
from .sparse import SparseRetriever

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class HybridConfig:
    fusion: str = "rrf"      # "rrf" | "weighted"
    alpha: float = 0.7       # dense weight in weighted mode (sparse = 1 - alpha)
    rrf_k: int = 60          # RRF constant — higher k reduces the impact of top ranks
    dense_top_k: int = 20    # candidates fetched from each retriever before fusion
    sparse_top_k: int = 20


# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines a ``DenseRetriever`` and a ``SparseRetriever`` into a single
    ranked result list using configurable score fusion.

    Args:
        dense:  Configured ``DenseRetriever``.
        sparse: Configured ``SparseRetriever``.
        config: Fusion parameters.  Defaults to RRF with k=60.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        config: HybridConfig | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.config = config or HybridConfig()

    # ── Async (API path) ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 10,
        document_ids: list[uuid.UUID] | None = None,
        document_type_filter: list[str] | None = None,
    ) -> list[DenseHit]:
        """
        Run both retrievers concurrently and return fused results.

        Args:
            query: Free-text query string.
            db: Open ``AsyncSession``.
            top_k: Final result count after fusion.
            document_ids: Restrict both retrievers to these document UUIDs.
            document_type_filter: Restrict by document type.

        Returns:
            Fused, ranked list of ``DenseHit`` with ``source="hybrid"``.
        """
        cfg = self.config

        dense_hits, sparse_hits = await asyncio.gather(
            self.dense.search(query, db, cfg.dense_top_k, document_ids, document_type_filter),
            self.sparse.search(query, db, cfg.sparse_top_k, document_ids, document_type_filter),
        )

        logger.debug(
            "Hybrid pre-fusion: dense=%d sparse=%d query=%r",
            len(dense_hits), len(sparse_hits), query[:60],
        )
        return self._fuse(dense_hits, sparse_hits, top_k)

    # ── Sync (worker / test path) ─────────────────────────────────────────────

    def search_sync(
        self,
        query: str,
        db: Session,
        top_k: int = 10,
        document_ids: list[uuid.UUID] | None = None,
        document_type_filter: list[str] | None = None,
    ) -> list[DenseHit]:
        """Synchronous variant — runs retrievers sequentially."""
        cfg = self.config
        dense_hits = self.dense.search_sync(query, db, cfg.dense_top_k, document_ids, document_type_filter)
        sparse_hits = self.sparse.search_sync(query, db, cfg.sparse_top_k, document_ids, document_type_filter)
        return self._fuse(dense_hits, sparse_hits, top_k)

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(
        self,
        dense_hits: list[DenseHit],
        sparse_hits: list[DenseHit],
        top_k: int,
    ) -> list[DenseHit]:
        if self.config.fusion == "rrf":
            return _fuse_rrf(dense_hits, sparse_hits, top_k, self.config.rrf_k)
        if self.config.fusion == "weighted":
            return _fuse_weighted(dense_hits, sparse_hits, top_k, self.config.alpha)
        raise ValueError(f"Unknown fusion strategy: {self.config.fusion!r}. Use 'rrf' or 'weighted'.")


# ── Fusion implementations ────────────────────────────────────────────────────

def _fuse_rrf(
    dense_hits: list[DenseHit],
    sparse_hits: list[DenseHit],
    top_k: int,
    k: int,
) -> list[DenseHit]:
    """
    Reciprocal Rank Fusion.

    For each chunk, accumulate  1 / (k + rank)  from every retriever list
    that contains it.  rank is 0-based.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    hit_registry: dict[str, DenseHit] = {}

    for rank, hit in enumerate(dense_hits):
        cid = str(hit.chunk_id)
        rrf_scores[cid] += 1.0 / (k + rank + 1)
        hit_registry[cid] = hit

    for rank, hit in enumerate(sparse_hits):
        cid = str(hit.chunk_id)
        rrf_scores[cid] += 1.0 / (k + rank + 1)
        hit_registry.setdefault(cid, hit)  # keep dense version if both present

    return _assemble(rrf_scores, hit_registry, top_k)


def _fuse_weighted(
    dense_hits: list[DenseHit],
    sparse_hits: list[DenseHit],
    top_k: int,
    alpha: float,
) -> list[DenseHit]:
    """
    Weighted linear combination:  alpha * dense_score + (1-alpha) * sparse_score.
    Chunks that appear in only one list receive 0 from the missing side.
    """
    dense_map = {str(h.chunk_id): h for h in dense_hits}
    sparse_map = {str(h.chunk_id): h for h in sparse_hits}
    all_ids = set(dense_map) | set(sparse_map)

    fused_scores: dict[str, float] = {}
    hit_registry: dict[str, DenseHit] = {}

    for cid in all_ids:
        d_score = dense_map[cid].score if cid in dense_map else 0.0
        s_score = sparse_map[cid].score if cid in sparse_map else 0.0
        fused_scores[cid] = alpha * d_score + (1.0 - alpha) * s_score
        hit_registry[cid] = dense_map.get(cid) or sparse_map[cid]

    return _assemble(fused_scores, hit_registry, top_k)


def _assemble(
    scores: dict[str, float],
    registry: dict[str, DenseHit],
    top_k: int,
) -> list[DenseHit]:
    """Sort by score, tag as hybrid, return top_k."""
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    results = []
    for cid in sorted_ids:
        hit = registry[cid]
        hit.score = scores[cid]
        hit.source = "hybrid"
        results.append(hit)

    logger.debug("Hybrid post-fusion: %d results (top score=%.4f)", len(results), results[0].score if results else 0)
    return results
