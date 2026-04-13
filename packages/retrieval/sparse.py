"""
Sparse (keyword) retriever — BM25 and PostgreSQL FTS.

Returns ``DenseHit`` objects with ``source="sparse"`` so results can be
merged directly with dense retrieval hits in the hybrid retriever.

Score normalisation
-------------------
BM25 scores are unbounded. Before returning, each batch is normalised to
[0, 1] by dividing by the maximum score in the result set.  This makes
dense and sparse scores comparable during score fusion.

Backends
--------
``BM25Index``  — in-memory rank_bm25, loaded from per-document pkl files
                 via ``BM25IndexRegistry``.  Best for dev and testing.

``PostgresFTSIndexer`` — uses pg ``ts_rank_cd``, already in [0, 1].
                         Production path; no extra service required.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .dense import DenseHit
from .indexer import BaseIndex, BM25Index

logger = logging.getLogger(__name__)


# ── BM25 index registry ───────────────────────────────────────────────────────

class BM25IndexRegistry:
    """
    Loads all per-document ``<uuid>.bm25.pkl`` files from a directory and
    merges them into a single global ``BM25Index`` for cross-corpus search.

    Usage
    -----
    registry = BM25IndexRegistry("/tmp/regintel/indexes")
    registry.load_all()
    index = registry.index          # pass to SparseRetriever
    registry.refresh()              # call after a new document is ingested
    """

    def __init__(self, index_dir: str | Path) -> None:
        self._dir = Path(index_dir)
        self._index = BM25Index()

    @property
    def index(self) -> BM25Index:
        return self._index

    def load_all(self) -> int:
        """Scan directory, merge all .bm25.pkl files. Returns number of docs loaded."""
        pkl_files = sorted(self._dir.glob("*.bm25.pkl"))
        if not pkl_files:
            logger.warning("BM25IndexRegistry: no index files found in %s", self._dir)
            return 0

        all_ids: list[str] = []
        all_texts: list[str] = []

        for pkl in pkl_files:
            try:
                tmp = BM25Index.load(pkl)
                all_ids.extend(tmp._chunk_ids)
                all_texts.extend(tmp._texts)
            except Exception as exc:
                logger.warning("Failed to load BM25 index %s: %s", pkl.name, exc)

        if all_ids:
            self._index.build(all_ids, all_texts)
            logger.info(
                "BM25IndexRegistry loaded %d chunks from %d document(s)",
                len(all_ids), len(pkl_files),
            )
        return len(pkl_files)

    def refresh(self) -> int:
        """Reload all indices — call after a new document finishes ingestion."""
        return self.load_all()


# ── Sparse retriever ──────────────────────────────────────────────────────────

class SparseRetriever:
    """
    Keyword retrieval via BM25 (``BM25Index``) or PostgreSQL FTS
    (``PostgresFTSIndexer``).

    Both paths return ``DenseHit`` objects with ``source="sparse"`` and
    scores normalised to [0, 1].

    Args:
        index: Any ``BaseIndex`` — typically a ``BM25Index`` loaded via
               ``BM25IndexRegistry``, or a ``PostgresFTSIndexer``.
    """

    def __init__(self, index: BaseIndex) -> None:
        self.index = index

    # ── Async (API path) ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        document_type_filter: list[str] | None = None,
    ) -> list[DenseHit]:
        raw = self.index.search(query, top_k=top_k * 2)  # over-fetch before filter
        if not raw:
            return []

        chunk_uuids, score_map = _parse_raw_hits(raw)
        rows = await _fetch_chunks_async(db, chunk_uuids, document_ids, document_type_filter)
        hits = _rows_to_hits(rows, score_map)
        return _normalise_and_trim(hits, top_k)

    # ── Sync (worker / test path) ─────────────────────────────────────────────

    def search_sync(
        self,
        query: str,
        db: Session,
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        document_type_filter: list[str] | None = None,
    ) -> list[DenseHit]:
        raw = self.index.search(query, top_k=top_k * 2)
        if not raw:
            return []

        chunk_uuids, score_map = _parse_raw_hits(raw)
        rows = _fetch_chunks_sync(db, chunk_uuids, document_ids, document_type_filter)
        hits = _rows_to_hits(rows, score_map)
        return _normalise_and_trim(hits, top_k)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_raw_hits(raw) -> tuple[list[uuid.UUID], dict[str, float]]:
    """Extract UUIDs and build a score lookup from raw BM25/FTS hits."""
    chunk_uuids: list[uuid.UUID] = []
    score_map: dict[str, float] = {}
    for hit in raw:
        try:
            uid = uuid.UUID(hit.chunk_id)
            chunk_uuids.append(uid)
            score_map[str(uid)] = hit.score
        except (ValueError, AttributeError):
            logger.debug("Skipping non-UUID chunk_id %r from index", hit.chunk_id)
    return chunk_uuids, score_map


def _build_chunk_stmt(
    chunk_uuids: list[uuid.UUID],
    document_ids: list[uuid.UUID] | None,
    document_type_filter: list[str] | None,
):
    from apps.api.app.models.document import Chunk, Document

    stmt = (
        select(Chunk, Document.title.label("doc_title"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(chunk_uuids))
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    if document_type_filter:
        stmt = stmt.where(Document.document_type.in_(document_type_filter))
    return stmt


async def _fetch_chunks_async(db, chunk_uuids, document_ids, document_type_filter):
    stmt = _build_chunk_stmt(chunk_uuids, document_ids, document_type_filter)
    result = await db.execute(stmt)
    return result.all()


def _fetch_chunks_sync(db, chunk_uuids, document_ids, document_type_filter):
    stmt = _build_chunk_stmt(chunk_uuids, document_ids, document_type_filter)
    return db.execute(stmt).all()


def _rows_to_hits(rows, score_map: dict[str, float]) -> list[DenseHit]:
    hits = []
    for row in rows:
        chunk = row.Chunk
        score = score_map.get(str(chunk.id), 0.0)
        hits.append(DenseHit(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            score=score,
            text=chunk.text,
            section_title=chunk.section_title,
            heading_path=list(chunk.heading_path or []),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            document_title=row.doc_title,
            source="sparse",
        ))
    return hits


def _normalise_and_trim(hits: list[DenseHit], top_k: int) -> list[DenseHit]:
    """Normalise scores to [0, 1] and return top_k sorted by descending score."""
    if not hits:
        return []
    max_score = max(h.score for h in hits)
    if max_score > 0:
        for h in hits:
            h.score = h.score / max_score
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
