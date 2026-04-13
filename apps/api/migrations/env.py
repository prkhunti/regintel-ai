"""Alembic environment — async SQLAlchemy + auto-detecting models."""
from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ── Make project importable ───────────────────────────────────────────────────
# Works in two environments:
#
#   Local (monorepo root):
#     __file__ = .../regintel-ai/apps/api/migrations/env.py
#     _API_ROOT = .../regintel-ai/apps/api
#     _MONO_ROOT = .../regintel-ai        ← contains packages/
#     Import: from apps.api.app.models import Base
#
#   Docker (WORKDIR /app, app copied to /app):
#     __file__ = /app/migrations/env.py
#     _API_ROOT = /app
#     _MONO_ROOT = /  (no packages/ there)
#     /app/packages/ exists  ← packages were COPY'd in
#     Import: from app.models import Base

_API_ROOT = Path(__file__).resolve().parents[1]  # directory containing alembic.ini

# Always add api root — lets Docker resolve `from app.models` and `from packages.x`
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

# Add monorepo root when detectable (local dev) — enables `from apps.api.app.models`
_MONO_ROOT = _API_ROOT.parents[1] if len(_API_ROOT.parts) >= 3 else None
if _MONO_ROOT and (_MONO_ROOT / "packages").exists():
    if str(_MONO_ROOT) not in sys.path:
        sys.path.insert(0, str(_MONO_ROOT))

# ── Import all models so Alembic can detect them ──────────────────────────────
try:
    from apps.api.app.models import Base  # local / monorepo context  # noqa: E402
except ImportError:
    from app.models import Base  # Docker container context  # noqa: E402

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Override sqlalchemy.url with the DATABASE_URL env var (sync driver for migrations)
_db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://regintel:regintel@localhost:5432/regintel",
)
config.set_main_option("sqlalchemy.url", _db_url)


# ── Offline mode (generate SQL without connecting) ───────────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode (connect and migrate) ────────────────────────────────────────
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # pgvector columns use a custom type — tell Alembic not to try to diff them
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
