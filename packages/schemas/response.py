from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from .common import RiskLevel


class CitationRead(BaseModel):
    id: UUID
    chunk_id: UUID
    document_title: str
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    quote: str
    relevance_score: float

    model_config = {"from_attributes": True}


class ResponseCreate(BaseModel):
    query_id: UUID
    model_name: str
    prompt_version: str
    answer_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    refusal_reason: str | None = None
    uncited_claim_count: int = 0
    pii_detected: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    latency_ms: int | None = None


class ResponseRead(ResponseCreate):
    id: UUID
    citations: list[CitationRead] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}


class AnswerPayload(BaseModel):
    """Full response returned to the client — the primary API output shape."""
    query_id: UUID
    response_id: UUID
    query_text: str
    answer: str
    confidence: float
    risk_level: RiskLevel
    citations: list[CitationRead]
    evidence_snippets: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    latency_ms: int | None = None
