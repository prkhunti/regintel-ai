"""Add HNSW index on chunks.embedding for fast ANN search.

This is a separate migration from the initial schema because:
  - CREATE INDEX CONCURRENTLY cannot run inside a transaction.
  - Building the index on an empty table is instant; on an existing table
    it can take minutes — it should be applied after data has been loaded
    and the system is stable.

Revision ID: 002
Revises: 001
Create Date: 2026-04-07
"""
from __future__ import annotations
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HNSW index — best recall/throughput trade-off for cosine similarity.
    # m=16, ef_construction=64 are conservative defaults; tune upward if recall
    # is insufficient once the corpus exceeds ~100k chunks.
    #
    # vector_cosine_ops matches the `<=>` operator used in DenseRetriever.
    #
    # Note: CONCURRENTLY is omitted here — it cannot run inside a transaction
    # (which Alembic always uses). For zero-downtime production deployments,
    # run this DDL manually outside Alembic after the migration is stamped.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw"
        " ON chunks"
        " USING hnsw (embedding vector_cosine_ops)"
        " WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
