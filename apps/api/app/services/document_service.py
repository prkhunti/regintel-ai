"""DB operations for the document ingestion domain."""
from __future__ import annotations

import shutil
import uuid as _uuid
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..models import Chunk, Document
from packages.schemas import (  # noqa: F401 — re-exported for router convenience
    ChunkRead,
    DocumentCreate,
    DocumentRead,
    DocumentSummary,
)
from packages.schemas.common import ProcessingStatus

settings = get_settings()


async def save_upload(file: UploadFile) -> Path:
    """Persist uploaded file to the configured upload directory.

    Prefixes the filename with a UUID so concurrent uploads of files with
    the same name do not overwrite each other.
    """
    safe_name = f"{_uuid.uuid4().hex}_{file.filename}"
    dest = settings.upload_path / safe_name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return dest


async def create_document(
    db: AsyncSession,
    payload: DocumentCreate,
    file_path: Path,
) -> Document:
    doc = Document(
        title=payload.title,
        document_type=payload.document_type,
        version=payload.version,
        source_filename=payload.source_filename,
        # Use a unique placeholder so concurrent uploads don't collide on the
        # unique checksum constraint. The worker replaces this with the real
        # SHA-256 once parsing is complete.
        checksum=f"pending-{_uuid.uuid4().hex}",
        status=ProcessingStatus.PENDING,
        metadata_={"file_path": str(file_path)},
    )
    db.add(doc)
    await db.flush()   # get the generated ID
    return doc


async def get_document(db: AsyncSession, document_id: UUID) -> Document:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


async def list_documents(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> list[Document]:
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    )
    return list(result.scalars().all())


async def list_chunks(
    db: AsyncSession,
    document_id: UUID,
    skip: int = 0,
    limit: int = 100,
) -> list[Chunk]:
    await get_document(db, document_id)  # 404 if missing
    result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_chunk_count(db: AsyncSession, document_id: UUID) -> int:
    result = await db.execute(
        select(func.count()).where(Chunk.document_id == document_id)
    )
    return result.scalar_one()
