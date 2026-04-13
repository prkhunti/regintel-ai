import logging
import os

from celery import Celery
from celery.signals import worker_ready

logger = logging.getLogger(__name__)

celery_app = Celery(
    "regintel",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=["tasks.ingestion"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,          # re-queue on worker crash
    worker_prefetch_multiplier=1, # one task at a time per worker (heavy ingestion jobs)
    task_routes={
        "tasks.ingestion.ingest_document": {"queue": "ingestion"},
    },
)


@worker_ready.connect
def _check_api_keys(**kwargs):
    """Log API key availability when the worker boots so misconfiguration is obvious."""
    provider = os.getenv("LLM_PROVIDER", "openai")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if provider == "openai":
        if not openai_key or openai_key.startswith("sk-...") or len(openai_key) < 20:
            logger.error(
                "OPENAI_API_KEY is missing or still a placeholder. "
                "Ingestion will fail at the embedding step. "
                "Set the key in your shell and restart with: "
                "docker compose up -d --force-recreate api worker"
            )
        else:
            logger.info("OPENAI_API_KEY is set (len=%d) — embedding ready.", len(openai_key))
    elif provider == "anthropic":
        if not anthropic_key or anthropic_key.startswith("sk-ant-...") or len(anthropic_key) < 20:
            logger.error(
                "ANTHROPIC_API_KEY is missing or still a placeholder. "
                "Ingestion will fail at the embedding step."
            )
        else:
            logger.info("ANTHROPIC_API_KEY is set (len=%d) — embedding ready.", len(anthropic_key))

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    logger.info("Worker ready. provider=%s embedding_model=%s", provider, embedding_model)
