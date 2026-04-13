"""
ChunkingPipeline — orchestrates the full knowledge-indexing flow.

    ParsedDocument
        ↓  chunk_document()
    list[TextChunk]
        ↓  embedder.embed_texts()
    list[TextChunk + embedding]
        ↓  indexer.build()
    BM25Index (or PostgresFTS)
        ↓  returns
    PipelineResult

The pipeline is intentionally stateless: it receives all dependencies via
constructor injection so it can be unit-tested without a database or API key.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from .chunker import ChunkingConfig, TextChunk, chunk_document
from .embedder import BaseEmbedder
from .indexer import BaseIndex, BM25Index

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EnrichedChunk:
    """TextChunk enriched with its embedding vector."""
    chunk: TextChunk
    embedding: list[float]


@dataclass
class PipelineResult:
    document_id: str
    enriched_chunks: list[EnrichedChunk]
    index: BaseIndex
    page_count: int
    stats: dict = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        return len(self.enriched_chunks)

    @property
    def table_chunk_count(self) -> int:
        return sum(1 for ec in self.enriched_chunks if ec.chunk.is_table_chunk)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ChunkingPipeline:
    """
    Orchestrates: parse output → chunk → embed → index.

    Args:
        embedder: Any ``BaseEmbedder`` implementation.
        indexer:  Any ``BaseIndex`` implementation.  Defaults to BM25Index.
        config:   ``ChunkingConfig``.  Uses defaults when omitted.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        indexer: BaseIndex | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.indexer = indexer or BM25Index()
        self.config = config or ChunkingConfig()

    def run(self, parsed_doc, document_id: str) -> PipelineResult:
        """
        Execute the full pipeline for one parsed document.

        Args:
            parsed_doc: ``ParsedDocument`` from ``parser.parse_pdf()``.
            document_id: UUID string for this document (used in index keys).

        Returns:
            ``PipelineResult`` with enriched chunks and a built index.
        """
        t0 = time.perf_counter()
        logger.info("Pipeline started for document %s (%d pages)", document_id, parsed_doc.page_count)

        # ── 1. Chunk ──────────────────────────────────────────────────────────
        t_chunk = time.perf_counter()
        chunks = chunk_document(parsed_doc.pages, self.config)
        chunk_ms = int((time.perf_counter() - t_chunk) * 1000)
        logger.info("  Chunking: %d chunks in %d ms", len(chunks), chunk_ms)

        if not chunks:
            logger.warning("No chunks produced for document %s — pipeline aborted", document_id)
            return PipelineResult(
                document_id=document_id,
                enriched_chunks=[],
                index=self.indexer,
                page_count=parsed_doc.page_count,
                stats={"chunk_ms": chunk_ms},
            )

        # ── 2. Embed ──────────────────────────────────────────────────────────
        t_embed = time.perf_counter()
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        embed_ms = int((time.perf_counter() - t_embed) * 1000)
        logger.info("  Embedding: %d vectors in %d ms", len(embeddings), embed_ms)

        enriched = [
            EnrichedChunk(chunk=chunk, embedding=emb)
            for chunk, emb in zip(chunks, embeddings)
        ]

        # ── 3. Build keyword index ────────────────────────────────────────────
        t_index = time.perf_counter()
        chunk_ids = [f"{document_id}:{c.chunk_index}" for c in chunks]
        self.indexer.build(chunk_ids, texts)
        index_ms = int((time.perf_counter() - t_index) * 1000)
        logger.info("  Indexing: %d docs in %d ms", len(chunks), index_ms)

        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Pipeline complete for %s: %d chunks, %d table chunks, %d ms total",
            document_id, len(enriched),
            sum(1 for ec in enriched if ec.chunk.is_table_chunk),
            total_ms,
        )

        return PipelineResult(
            document_id=document_id,
            enriched_chunks=enriched,
            index=self.indexer,
            page_count=parsed_doc.page_count,
            stats={
                "chunk_ms": chunk_ms,
                "embed_ms": embed_ms,
                "index_ms": index_ms,
                "total_ms": total_ms,
                "chunk_count": len(enriched),
                "table_chunk_count": sum(1 for ec in enriched if ec.chunk.is_table_chunk),
                "ocr_pages": sum(1 for p in parsed_doc.pages if p.via_ocr),
            },
        )

    # ── Convenience: run on plain text (for testing) ──────────────────────────

    def run_on_text(self, text: str, document_id: str = "test") -> PipelineResult:
        """
        Run the pipeline on a plain string — useful for unit tests that
        don't need a real PDF.
        """
        from .parser import ParsedDocument, ParsedPage

        page = ParsedPage(page_number=1, text=text)
        doc = ParsedDocument(pages=[page], checksum="test", page_count=1)
        return self.run(doc, document_id)
