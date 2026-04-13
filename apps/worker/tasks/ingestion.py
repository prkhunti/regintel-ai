"""
Ingestion pipeline Celery task.

Steps
-----
1. Load document record → validate status
2. Parse PDF (text extraction + OCR fallback)
3. Section-aware chunking
4. Batch embedding generation
5. Persist chunks + embeddings to PostgreSQL (pgvector)
6. Update document status and page count
7. Emit audit event
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from uuid import UUID

# Make packages importable when running as a Celery worker.
# Walk up from this file until we find a directory that contains packages/.
# In Docker (WORKDIR /app, packages at /app/packages/) this resolves to /app.
# In local dev (monorepo root) this resolves to the repo root.
for _p in Path(__file__).resolve().parents:
    if (_p / "packages").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from celery import Task
from sqlalchemy.orm import Session

from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


class _IngestionTask(Task):
    """Base task with lazy-loaded pipeline (one instance per worker process)."""
    _pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from packages.retrieval.embedder import get_embedder
            from packages.retrieval.pipeline import ChunkingPipeline
            from packages.retrieval.chunker import ChunkingConfig

            embedder = get_embedder(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.getenv("OPENAI_API_KEY"),
                batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            )
            config = ChunkingConfig(
                target_tokens=int(os.getenv("CHUNK_TARGET_TOKENS", "512")),
                overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "64")),
            )
            # BM25 index is built post-DB-save (to use real UUIDs), not inside the pipeline
            self._pipeline = ChunkingPipeline(embedder=embedder, indexer=None, config=config)
        return self._pipeline


@celery_app.task(
    bind=True,
    base=_IngestionTask,
    name="tasks.ingestion.ingest_document",
    max_retries=3,
    default_retry_delay=30,
)
def ingest_document(self, document_id: str) -> dict:
    """
    Full ingestion pipeline for a single document.

    Args:
        document_id: UUID string of the Document record.

    Returns:
        Summary dict with chunk count and processing stats.
    """
    doc_uuid = UUID(document_id)
    logger.info("Starting ingestion for document %s", document_id)

    try:
        return _run_pipeline(self, doc_uuid)
    except Exception as exc:
        logger.exception("Ingestion failed for document %s: %s", document_id, exc)
        _mark_failed(doc_uuid, str(exc))
        raise self.retry(exc=exc)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _run_pipeline(task: _IngestionTask, doc_uuid: UUID) -> dict:
    from database import get_db
    from apps.api.app.models import Document, Chunk, AuditEvent
    from packages.schemas.common import ProcessingStatus
    from packages.retrieval.parser import parse_pdf

    with get_db() as db:
        doc = _load_document(db, doc_uuid)
        _set_status(db, doc, ProcessingStatus.PROCESSING)

        # ── 1. Parse ──────────────────────────────────────────────────────────
        file_path = Path(doc.metadata_.get("file_path", ""))
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        ocr_threshold = int(os.getenv("OCR_TEXT_THRESHOLD", "50"))
        parsed = parse_pdf(file_path, ocr_threshold=ocr_threshold)
        logger.info("Parsed %d pages for document %s", parsed.page_count, doc_uuid)

        # ── 2. Chunk → embed → index (via ChunkingPipeline) ───────────────────
        result = task.pipeline.run(parsed, str(doc_uuid))
        logger.info(
            "Pipeline produced %d chunks (%d table) for document %s",
            result.chunk_count, result.table_chunk_count, doc_uuid,
        )

        # ── 3. Persist enriched chunks to PostgreSQL ──────────────────────────
        # Generate UUIDs in Python so they are available immediately after
        # flush — before commit — for use as BM25 chunk identifiers.
        import uuid as _uuid
        db_chunks = [
            Chunk(
                id=_uuid.uuid4(),
                document_id=doc_uuid,
                chunk_index=ec.chunk.chunk_index,
                section_title=ec.chunk.section_title,
                heading_path=ec.chunk.heading_path,
                page_start=ec.chunk.page_start,
                page_end=ec.chunk.page_end,
                text=ec.chunk.text,
                token_count=ec.chunk.token_count,
                source_hash=ec.chunk.source_hash,
                embedding=ec.embedding,
            )
            for ec in result.enriched_chunks
        ]
        db.add_all(db_chunks)
        db.flush()  # assigns IDs without committing

        # ── 4. Build + persist BM25 index with real chunk UUIDs ──────────────
        _save_bm25_index(
            chunk_ids=[str(c.id) for c in db_chunks],
            texts=[ec.chunk.text for ec in result.enriched_chunks],
            doc_uuid=doc_uuid,
        )

        # ── 5. Finalise document ──────────────────────────────────────────────
        doc.page_count = parsed.page_count
        doc.checksum = parsed.checksum
        _set_status(db, doc, ProcessingStatus.COMPLETE)

        # ── 6. Audit event ────────────────────────────────────────────────────
        db.add(AuditEvent(
            event_type="document_processed",
            resource_type="document",
            resource_id=doc_uuid,
            detail=result.stats,
        ))

        db.commit()

    summary = {
        "document_id": str(doc_uuid),
        **result.stats,
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


def _save_bm25_index(chunk_ids: list[str], texts: list[str], doc_uuid: UUID) -> None:
    """Build a per-document BM25 index from real chunk UUIDs and persist to disk."""
    from packages.retrieval.indexer import BM25Index
    index_dir = Path(os.getenv("INDEX_DIR", "/tmp/regintel/indexes"))
    index_path = index_dir / f"{doc_uuid}.bm25.pkl"
    try:
        idx = BM25Index()
        idx.build(chunk_ids, texts)
        idx.save(index_path)
        logger.info("BM25 index saved for document %s (%d chunks)", doc_uuid, len(chunk_ids))
    except Exception as exc:
        logger.warning("Could not save BM25 index for %s: %s", doc_uuid, exc)


def _load_document(db: Session, doc_uuid: UUID):
    from apps.api.app.models import Document
    from packages.schemas.common import ProcessingStatus

    doc = db.get(Document, doc_uuid)
    if doc is None:
        raise ValueError(f"Document {doc_uuid} not found")
    if doc.status == ProcessingStatus.COMPLETE:
        logger.warning("Document %s already processed — skipping", doc_uuid)
        raise ValueError("Document already processed")
    return doc


def _set_status(db: Session, doc, status: str) -> None:
    from packages.schemas.common import ProcessingStatus
    doc.status = status
    db.flush()


def _mark_failed(doc_uuid: UUID, reason: str) -> None:
    try:
        from database import get_db
        from apps.api.app.models import Document
        from packages.schemas.common import ProcessingStatus

        with get_db() as db:
            doc = db.get(Document, doc_uuid)
            if doc:
                doc.status = ProcessingStatus.FAILED
                doc.metadata_["failure_reason"] = reason
                db.commit()
    except Exception as exc:
        logger.error("Failed to mark document %s as failed: %s", doc_uuid, exc)
