from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .common import DocumentType, ProcessingStatus


class DocumentCreate(BaseModel):
    title: str
    document_type: DocumentType
    version: str = "1.0"
    source_filename: str


class DocumentSummary(BaseModel):
    id: UUID
    title: str
    document_type: DocumentType
    version: str
    status: ProcessingStatus
    uploaded_at: datetime
    chunk_count: int = 0

    model_config = {"from_attributes": True}


class DocumentRead(DocumentSummary):
    checksum: str
    page_count: int | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentVersionRead(BaseModel):
    id: UUID
    document_id: UUID
    version: str
    checksum: str
    created_at: datetime
    notes: str | None = None

    model_config = {"from_attributes": True}


class ChunkCreate(BaseModel):
    document_id: UUID
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    token_count: int
    source_hash: str
    heading_path: list[str] = Field(default_factory=list)


class ChunkRead(ChunkCreate):
    id: UUID
    chunk_index: int

    model_config = {"from_attributes": True}
