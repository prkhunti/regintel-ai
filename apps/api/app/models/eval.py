import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, new_uuid


class EvalCase(Base, TimestampMixin):
    __tablename__ = "eval_cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_chunk_ids: Mapped[list] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=False, default=list)
    expected_answer_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_insufficient: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class EvalRun(Base, TimestampMixin):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    retriever_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Retrieval metrics
    recall_at_10: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_10: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Answer quality metrics
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucination_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    refusal_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
