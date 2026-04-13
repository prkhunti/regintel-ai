import uuid as _uuid
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..models import AuditEvent, Document
from ..services import document_service as svc
from packages.schemas import (
    ChunkRead,
    DocumentCreate,
    DocumentRead,
    DocumentSummary,
)
from packages.schemas.common import DocumentType

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentSummary, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to ingest"),
    title: str = Form(...),
    document_type: DocumentType = Form(...),
    version: str = Form(default="1.0"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document and enqueue the ingestion pipeline.

    Returns a DocumentSummary with status=pending. Poll GET /documents/{id}
    to track processing progress.
    """
    _validate_upload(file)

    file_path = await svc.save_upload(file)

    payload = DocumentCreate(
        title=title,
        document_type=document_type,
        version=version,
        source_filename=file.filename or file_path.name,
    )
    doc = await svc.create_document(db, payload, file_path)
    await db.commit()

    db.add(AuditEvent(
        id=_uuid.uuid4(),
        event_type="document_uploaded",
        resource_type="document",
        resource_id=doc.id,
        detail={
            "title": doc.title,
            "document_type": str(doc.document_type),
            "version": doc.version,
            "filename": file.filename,
        },
    ))

    # Fire ingestion task
    _enqueue_ingestion(str(doc.id))

    chunk_count = 0  # not yet processed
    return _to_summary(doc, chunk_count)


@router.post("/{document_id}/process", response_model=DocumentSummary, status_code=status.HTTP_202_ACCEPTED)
async def reprocess_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Re-trigger ingestion for an existing document (e.g. after a failure)."""
    doc = await svc.get_document(db, document_id)
    _enqueue_ingestion(str(document_id))
    chunk_count = await svc.get_chunk_count(db, document_id)
    return _to_summary(doc, chunk_count)


@router.get("", response_model=list[DocumentSummary])
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    docs = await svc.list_documents(db, skip=skip, limit=limit)
    results = []
    for doc in docs:
        count = await svc.get_chunk_count(db, doc.id)
        results.append(_to_summary(doc, count))
    return results


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    doc = await svc.get_document(db, document_id)
    chunk_count = await svc.get_chunk_count(db, document_id)
    return DocumentRead(
        id=doc.id,
        title=doc.title,
        document_type=doc.document_type,
        version=doc.version,
        status=doc.status,
        uploaded_at=doc.created_at,
        chunk_count=chunk_count,
        checksum=doc.checksum,
        page_count=doc.page_count,
        metadata=doc.metadata_,
    )


@router.get("/{document_id}/chunks", response_model=list[ChunkRead])
async def list_chunks(
    document_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    chunks = await svc.list_chunks(db, document_id, skip=skip, limit=limit)
    return [
        ChunkRead(
            id=c.id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            section_title=c.section_title,
            heading_path=c.heading_path,
            page_start=c.page_start,
            page_end=c.page_end,
            text=c.text,
            token_count=c.token_count,
            source_hash=c.source_hash,
        )
        for c in chunks
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {suffix!r} not allowed. Accepted: {settings.allowed_extensions}",
        )


def _enqueue_ingestion(document_id: str) -> None:
    try:
        from celery import Celery
        import os

        app = Celery(broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))
        app.send_task("tasks.ingestion.ingest_document", args=[document_id], queue="ingestion")
    except Exception as exc:
        # Log but don't fail the HTTP response — the user can re-trigger via /process
        import logging
        logging.getLogger(__name__).error("Failed to enqueue ingestion for %s: %s", document_id, exc)


def _to_summary(doc: Document, chunk_count: int) -> DocumentSummary:
    return DocumentSummary(
        id=doc.id,
        title=doc.title,
        document_type=doc.document_type,
        version=doc.version,
        status=doc.status,
        uploaded_at=doc.created_at,
        chunk_count=chunk_count,
    )
