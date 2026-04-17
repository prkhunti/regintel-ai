"""
Dense retriever — pgvector cosine-similarity search.

Flow
----
1. Embed the query string with the configured embedder.
2. Run an ANN search against the ``chunks.embedding`` column using pgvector's
   ``<=>`` cosine-distance operator.
3. Optionally filter by document_ids or document_type.
4. Return a ranked list of DenseHit objects (score = cosine similarity, not distance).

The retriever is async-first (FastAPI / SQLAlchemy asyncio path).
A sync helper ``search_sync`` is provided for Celery tasks and tests.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .embedder import BaseEmbedder

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DenseHit:
    """Dense retrieval result enriched with document metadata.

    Parameters
    ----------
    chunk_id
        Identifier of the matched chunk.
    document_id
        Identifier of the parent document.
    score
        Cosine-similarity score where higher is more similar.
    text
        Chunk text returned to the caller.
    section_title
        Section heading associated with the chunk, if available.
    heading_path
        Hierarchical heading path for the chunk.
    page_start
        First page number contributing to the chunk.
    page_end
        Last page number contributing to the chunk.
    document_title
        Human-readable document title, when loaded from the database.
    source
        Retrieval source label such as ``"dense"``, ``"sparse"``, or ``"hybrid"``.
    """
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    score: float            # cosine similarity in [0, 1]; higher = more similar
    text: str
    section_title: str | None
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    document_title: str | None = None
    source: str = "dense"   # "dense" | "sparse" — set by the retriever that produced it


# ── Retriever ─────────────────────────────────────────────────────────────────

class DenseRetriever:
    """
    Retrieves chunks by semantic similarity using pgvector.

    Args:
        embedder: Any ``BaseEmbedder``.  Its ``embed_single`` method is called
                  per query; the result is sent to pgvector as a filter vector.
        embedding_dim: Must match the ``Vector(n)`` declared on ``Chunk.embedding``.
    """

    def __init__(self, embedder: BaseEmbedder, embedding_dim: int = 1536) -> None:
        self.embedder = embedder
        self.embedding_dim = embedding_dim

    # ── Async (API path) ──────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        document_type_filter: list[str] | None = None,
    ) -> list[DenseHit]:
        """Run async nearest-neighbour search over embedded chunks.

        Parameters
        ----------
        query
            Free-text query string.
        db
            Open SQLAlchemy async session.
        top_k
            Maximum number of results to return.
        document_ids
            Optional document identifiers used to constrain the search space.
        document_type_filter
            Optional document type values used to filter joined document rows.

        Returns
        -------
        list[DenseHit]
            Ranked dense retrieval hits sorted by descending similarity.
        """
        query_vec = self.embedder.embed_single(query)
        stmt = self._build_stmt(query_vec, top_k, document_ids, document_type_filter)
        result = await db.execute(stmt)
        return self._to_hits(result.all())

    # ── Sync (worker / test path) ─────────────────────────────────────────────

    def search_sync(
        self,
        query: str,
        db: Session,
        top_k: int = 20,
        document_ids: list[uuid.UUID] | None = None,
        document_type_filter: list[str] | None = None,
    ) -> list[DenseHit]:
        """Run synchronous nearest-neighbour search.

        Parameters
        ----------
        query
            Free-text query string.
        db
            Open SQLAlchemy sync session.
        top_k
            Maximum number of results to return.
        document_ids
            Optional document identifiers used to constrain the search space.
        document_type_filter
            Optional document type values used to filter joined document rows.

        Returns
        -------
        list[DenseHit]
            Ranked dense retrieval hits sorted by descending similarity.
        """
        query_vec = self.embedder.embed_single(query)
        stmt = self._build_stmt(query_vec, top_k, document_ids, document_type_filter)
        result = db.execute(stmt)
        return self._to_hits(result.all())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_stmt(
        self,
        query_vec: list[float],
        top_k: int,
        document_ids: list[uuid.UUID] | None,
        document_type_filter: list[str] | None,
    ):
        # Import here to keep module importable without a live DB connection
        from apps.api.app.models.document import Chunk, Document

        # cosine_distance = 1 - cosine_similarity  →  similarity = 1 - distance
        distance_col = Chunk.embedding.cosine_distance(query_vec).label("distance")
        score_col = (1 - Chunk.embedding.cosine_distance(query_vec)).label("score")

        stmt = (
            select(Chunk, Document.title.label("doc_title"), score_col)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.embedding.is_not(None))
        )

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        if document_type_filter:
            stmt = stmt.where(Document.document_type.in_(document_type_filter))

        # Order by distance ASC (closest first) — same as score DESC
        stmt = stmt.order_by(distance_col).limit(top_k)
        return stmt

    @staticmethod
    def _to_hits(rows) -> list[DenseHit]:
        hits = []
        for row in rows:
            chunk = row.Chunk
            hits.append(DenseHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                score=float(row.score),
                text=chunk.text,
                section_title=chunk.section_title,
                heading_path=list(chunk.heading_path or []),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                document_title=row.doc_title,
            ))
        logger.debug("Dense search returned %d hits", len(hits))
        return hits
