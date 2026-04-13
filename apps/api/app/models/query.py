import uuid

from sqlalchemy import ForeignKey, Integer, String, Text, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, new_uuid


class Query(Base, TimestampMixin):
    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(64), nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)

    retrieval_runs: Mapped[list["RetrievalRun"]] = relationship(back_populates="query", cascade="all, delete-orphan")
    responses: Mapped[list["Response"]] = relationship(back_populates="query", cascade="all, delete-orphan")  # noqa: F821


class RetrievalRun(Base, TimestampMixin):
    __tablename__ = "retrieval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    query_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False)
    retriever_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    dense_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sparse_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reranked_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    query: Mapped["Query"] = relationship(back_populates="retrieval_runs")
    retrieved_chunks: Mapped[list["RetrievedChunk"]] = relationship(back_populates="retrieval_run", cascade="all, delete-orphan")


class RetrievedChunk(Base):
    __tablename__ = "retrieved_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_uuid)
    retrieval_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("retrieval_runs.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    dense_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sparse_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reranker_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)

    retrieval_run: Mapped["RetrievalRun"] = relationship(back_populates="retrieved_chunks")
    chunk: Mapped["Chunk"] = relationship(back_populates="retrieved_in")  # noqa: F821
