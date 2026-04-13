from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class EvalCaseCreate(BaseModel):
    query: str
    expected_chunk_ids: list[UUID] = Field(default_factory=list)
    expected_answer_pattern: str | None = None
    is_insufficient: bool = False
    notes: str | None = None


class EvalCaseRead(EvalCaseCreate):
    id: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalRunCreate(BaseModel):
    label: str
    model_name: str
    retriever_config: dict
    case_ids: list[UUID] | None = None  # None = run all cases


class EvalRunRead(BaseModel):
    id: UUID
    label: str
    model_name: str
    retriever_config: dict
    total_cases: int
    recall_at_10: float | None = None
    precision_at_10: float | None = None
    mrr: float | None = None
    faithfulness_score: float | None = None
    hallucination_rate: float | None = None
    refusal_accuracy: float | None = None
    mean_latency_ms: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
