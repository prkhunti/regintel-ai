from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel


class AuditEventType(StrEnum):
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSED = "document_processed"
    QUERY_SUBMITTED = "query_submitted"
    ANSWER_GENERATED = "answer_generated"
    ANSWER_REFUSED = "answer_refused"
    PII_DETECTED = "pii_detected"
    INJECTION_DETECTED = "injection_detected"
    EVAL_RUN_STARTED = "eval_run_started"
    EVAL_RUN_COMPLETED = "eval_run_completed"


class AuditEventRead(BaseModel):
    id: UUID
    event_type: AuditEventType
    actor: str | None = None
    resource_type: str | None = None
    resource_id: UUID | None = None
    detail: dict = {}
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackCreate(BaseModel):
    response_id: UUID
    rating: int  # 1–5
    comment: str | None = None
    flagged_hallucination: bool = False
    flagged_missing_citation: bool = False


class FeedbackRead(FeedbackCreate):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}
