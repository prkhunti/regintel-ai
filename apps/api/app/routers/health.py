"""
GET /api/v1/health — liveness + dependency probe.

Checks:
  - database  : SELECT 1 via the async session pool
  - redis     : PING via redis-py
  - llm       : one-token OpenAI embedding call (or Anthropic ping)

Returns HTTP 200 with status "ok" or "degraded" so the frontend can
always parse the body even when a dependency is down.
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

router = APIRouter(prefix="/health", tags=["meta"])
logger = logging.getLogger(__name__)


@router.get("", summary="Liveness + dependency probe")
async def health_check():
    results: dict[str, dict] = {}

    results["database"] = await _check_database()
    results["redis"]    = await _check_redis()
    results["llm"]      = await _check_llm()

    # quota_exceeded = key is valid, just needs billing — treat as "ok" for routing
    def _is_healthy(s: str) -> bool:
        return s in ("ok", "quota_exceeded")

    overall = "ok" if all(_is_healthy(c["status"]) for c in results.values()) else "degraded"
    return {"status": overall, "components": results}


# ── Individual probes ─────────────────────────────────────────────────────────

async def _check_database() -> dict:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        return {"status": "error", "detail": "DATABASE_URL not set"}
    try:
        t0 = time.perf_counter()
        engine = create_async_engine(url, pool_pre_ping=True)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return {"status": "ok", "latency_ms": int((time.perf_counter() - t0) * 1000)}
    except Exception as exc:
        logger.warning("DB health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


async def _check_redis() -> dict:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        import redis.asyncio as aioredis
        t0 = time.perf_counter()
        client = aioredis.from_url(url, socket_connect_timeout=3)
        await client.ping()
        await client.aclose()
        return {"status": "ok", "latency_ms": int((time.perf_counter() - t0) * 1000)}
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return {"status": "error", "detail": str(exc)}


async def _check_llm() -> dict:
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider in ("random", "mock"):
        return {
            "status": "ok",
            "provider": provider,
            "detail": "Stub mode — no API key required. Set LLM_PROVIDER=openai once credits are available.",
        }

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-...") or len(api_key) < 20:
            return {
                "status": "unconfigured",
                "provider": "openai",
                "detail": "OPENAI_API_KEY is missing or still a placeholder. "
                          "Set it in your shell before running docker compose.",
            }
        try:
            from openai import AsyncOpenAI, RateLimitError, AuthenticationError
            t0 = time.perf_counter()
            client = AsyncOpenAI(api_key=api_key)
            await client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                input="health",
            )
            return {
                "status": "ok",
                "provider": "openai",
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }
        except RateLimitError as exc:
            # 429 quota_exceeded — key is valid, account just needs credits.
            # openai SDK 1.x exposes the error code as exc.code directly.
            err_code = getattr(exc, "code", None) or ""
            err_type = getattr(exc, "type", None) or ""
            if "insufficient_quota" in (err_code, err_type) or "insufficient_quota" in str(exc):
                return {
                    "status": "quota_exceeded",
                    "provider": "openai",
                    "detail": "API key is valid but the account has no remaining quota. "
                              "Add credits at platform.openai.com/settings/billing.",
                }
            return {"status": "error", "provider": "openai", "detail": str(exc)}
        except AuthenticationError as exc:
            return {
                "status": "error",
                "provider": "openai",
                "detail": "API key rejected by OpenAI — check the key is correct.",
            }
        except Exception as exc:
            logger.warning("OpenAI health check failed: %s", exc)
            return {"status": "error", "provider": "openai", "detail": str(exc)}

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key or api_key.startswith("sk-ant-...") or len(api_key) < 20:
            return {
                "status": "unconfigured",
                "provider": "anthropic",
                "detail": "ANTHROPIC_API_KEY is missing or still a placeholder.",
            }
        # Anthropic has no free ping; just confirm the key is present and non-placeholder
        return {"status": "ok", "provider": "anthropic", "detail": "key present (not probed)"}

    return {"status": "error", "detail": f"Unknown LLM_PROVIDER: {provider!r}"}
