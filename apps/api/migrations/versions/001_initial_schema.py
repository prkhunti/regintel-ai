"""Initial schema — all tables + pgvector extension.

Revision ID: 001
Revises:
Create Date: 2026-04-07
"""
from __future__ import annotations
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── pgvector extension ────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── documents ─────────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("document_type", sa.String(64), nullable=False),
        sa.Column("version", sa.String(64), nullable=False, server_default="1.0"),
        sa.Column("source_filename", sa.String(512), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False, unique=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("page_count", sa.Integer, nullable=True),
        sa.Column("metadata", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_document_type", "documents", ["document_type"])
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    # ── document_versions ─────────────────────────────────────────────────────
    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── chunks ────────────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("section_title", sa.String(512), nullable=True),
        sa.Column("heading_path", postgresql.ARRAY(sa.String), nullable=False, server_default="{}"),
        sa.Column("page_start", sa.Integer, nullable=True),
        sa.Column("page_end", sa.Integer, nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("source_hash", sa.String(64), nullable=False),
        # pgvector column — declared as Text here; the real Vector type is added below
        sa.Column("embedding", sa.Text, nullable=True),
    )
    # Replace the placeholder Text column with the proper vector type
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536) USING NULL")
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_source_hash", "chunks", ["source_hash"])

    # ── queries ───────────────────────────────────────────────────────────────
    op.create_table(
        "queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_query", sa.Text, nullable=False),
        sa.Column("query_type", sa.String(64), nullable=False),
        sa.Column("rewritten_query", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_queries_created_at", "queries", ["created_at"])

    # ── retrieval_runs ────────────────────────────────────────────────────────
    op.create_table(
        "retrieval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("queries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("retriever_config", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("top_k", sa.Integer, nullable=False),
        sa.Column("dense_hits", sa.Integer, nullable=False, server_default="0"),
        sa.Column("sparse_hits", sa.Integer, nullable=False, server_default="0"),
        sa.Column("reranked_hits", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── retrieved_chunks ──────────────────────────────────────────────────────
    op.create_table(
        "retrieved_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("retrieval_run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("retrieval_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("dense_score", sa.Float, nullable=True),
        sa.Column("sparse_score", sa.Float, nullable=True),
        sa.Column("reranker_score", sa.Float, nullable=True),
        sa.Column("final_score", sa.Float, nullable=False),
    )

    # ── responses ─────────────────────────────────────────────────────────────
    op.create_table(
        "responses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("queries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("answer_text", sa.Text, nullable=False),
        sa.Column("confidence_score", sa.Float, nullable=False),
        sa.Column("refusal_reason", sa.Text, nullable=True),
        sa.Column("uncited_claim_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("pii_detected", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("risk_level", sa.String(16), nullable=False, server_default="low"),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── citations ─────────────────────────────────────────────────────────────
    op.create_table(
        "citations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("response_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("responses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("quote", sa.Text, nullable=False),
        sa.Column("quote_start", sa.Integer, nullable=True),
        sa.Column("quote_end", sa.Integer, nullable=True),
        sa.Column("relevance_score", sa.Float, nullable=False),
    )

    # ── eval_cases ────────────────────────────────────────────────────────────
    op.create_table(
        "eval_cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("expected_chunk_ids",
                  postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
                  nullable=False, server_default="{}"),
        sa.Column("expected_answer_pattern", sa.Text, nullable=True),
        sa.Column("is_insufficient", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── eval_runs ─────────────────────────────────────────────────────────────
    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("retriever_config", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("total_cases", sa.Integer, nullable=False, server_default="0"),
        sa.Column("recall_at_10", sa.Float, nullable=True),
        sa.Column("precision_at_10", sa.Float, nullable=True),
        sa.Column("mrr", sa.Float, nullable=True),
        sa.Column("faithfulness_score", sa.Float, nullable=True),
        sa.Column("hallucination_rate", sa.Float, nullable=True),
        sa.Column("refusal_accuracy", sa.Float, nullable=True),
        sa.Column("mean_latency_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # ── audit_events ──────────────────────────────────────────────────────────
    op.create_table(
        "audit_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(256), nullable=True),
        sa.Column("resource_type", sa.String(64), nullable=True),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("detail", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_resource_id", "audit_events", ["resource_id"])

    # ── feedback ──────────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("response_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("responses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column("flagged_hallucination", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("flagged_missing_citation", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("audit_events")
    op.drop_table("eval_runs")
    op.drop_table("eval_cases")
    op.drop_table("citations")
    op.drop_table("responses")
    op.drop_table("retrieved_chunks")
    op.drop_table("retrieval_runs")
    op.drop_table("queries")
    op.drop_table("chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
