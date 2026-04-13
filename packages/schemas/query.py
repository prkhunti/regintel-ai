from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .common import DocumentType, QueryType


class QueryCreate(BaseModel):
    user_query: str
    query_type: QueryType = QueryType.EVIDENCE_EXTRACTION
    document_type_filter: list[DocumentType] | None = None
    document_ids: list[UUID] | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class QueryRead(BaseModel):
    id: UUID
    user_query: str
    query_type: QueryType
    rewritten_query: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RetrievalRunRead(BaseModel):
    id: UUID
    query_id: UUID
    retriever_config: dict
    top_k: int
    dense_hits: int
    sparse_hits: int
    reranked_hits: int
    latency_ms: int
    created_at: datetime

    model_config = {"from_attributes": True}


class RetrievedChunkRead(BaseModel):
    id: UUID
    retrieval_run_id: UUID
    chunk_id: UUID
    rank: int
    dense_score: float | None = None
    sparse_score: float | None = None
    reranker_score: float | None = None
    final_score: float

    # Chunk payload (joined)
    chunk_text: str | None = None
    section_title: str | None = None
    document_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None

    model_config = {"from_attributes": True}
