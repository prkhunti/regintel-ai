"""
Evaluation harness endpoints.

POST /eval/cases              — create an eval case
GET  /eval/cases              — list eval cases
GET  /eval/cases/{id}         — get a single eval case
DELETE /eval/cases/{id}       — delete a case

POST /eval/runs               — trigger an eval run (synchronous, returns results)
GET  /eval/runs               — list completed runs
GET  /eval/runs/{id}          — get a run's aggregate + per-case breakdown
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..models import AuditEvent, EvalCase, EvalRun
from ..services.answer_service import AnswerGenerator
from ..services.llm_client import get_llm_client
from packages.evals.runner import EvalCaseInput, EvalRunner
from packages.retrieval.dense import DenseRetriever
from packages.retrieval.embedder import get_embedder
from packages.retrieval.hybrid import HybridConfig, HybridRetriever
from packages.retrieval.reranker import get_reranker
from packages.retrieval.sparse import BM25IndexRegistry, SparseRetriever
from packages.schemas.eval import (
    EvalCaseCreate,
    EvalCaseRead,
    EvalRunCreate,
    EvalRunRead,
)

settings = get_settings()
router = APIRouter(prefix="/eval", tags=["eval"])
logger = logging.getLogger(__name__)


# ── Eval cases ────────────────────────────────────────────────────────────────

@router.post("/cases", response_model=EvalCaseRead, status_code=status.HTTP_201_CREATED)
async def create_eval_case(
    payload: EvalCaseCreate,
    db: AsyncSession = Depends(get_db),
):
    row = EvalCase(
        id=uuid.uuid4(),
        query=payload.query,
        expected_chunk_ids=[str(cid) for cid in payload.expected_chunk_ids],
        expected_answer_pattern=payload.expected_answer_pattern,
        is_insufficient=payload.is_insufficient,
        notes=payload.notes,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return EvalCaseRead.model_validate(row)


@router.get("/cases", response_model=list[EvalCaseRead])
async def list_eval_cases(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(EvalCase).order_by(EvalCase.created_at.desc()).offset(offset).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    return [EvalCaseRead.model_validate(r) for r in rows]


@router.get("/cases/{case_id}", response_model=EvalCaseRead)
async def get_eval_case(case_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = await db.get(EvalCase, case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Eval case not found")
    return EvalCaseRead.model_validate(row)


@router.delete("/cases/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_eval_case(case_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = await db.get(EvalCase, case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Eval case not found")
    await db.delete(row)
    await db.commit()


# ── Eval runs ─────────────────────────────────────────────────────────────────

@router.post("/runs", response_model=EvalRunRead, status_code=status.HTTP_200_OK)
async def trigger_eval_run(
    payload: EvalRunCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Run the evaluation pipeline over the selected cases (or all cases if
    case_ids is null) and return the aggregated metrics.

    The run is executed synchronously. For large case sets, consider moving
    this to a Celery task.
    """
    # Load cases from DB
    if payload.case_ids:
        stmt = select(EvalCase).where(EvalCase.id.in_([str(cid) for cid in payload.case_ids]))
    else:
        stmt = select(EvalCase).order_by(EvalCase.created_at.asc())
    rows = (await db.execute(stmt)).scalars().all()

    if not rows:
        raise HTTPException(status_code=400, detail="No eval cases found. Create some first via POST /eval/cases.")

    cases = [
        EvalCaseInput(
            id=str(r.id),
            query=r.query,
            expected_chunk_ids=[str(cid) for cid in r.expected_chunk_ids],
            is_insufficient=r.is_insufficient,
            notes=r.notes,
        )
        for r in rows
    ]

    # Build retriever stack (mirrors query router)
    embedder = get_embedder(
        provider=settings.llm_provider,
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        batch_size=settings.embedding_batch_size,
    )
    dense = DenseRetriever(embedder=embedder, embedding_dim=settings.embedding_dim)
    index_dir = os.getenv("INDEX_DIR", "/tmp/regintel/indexes")
    registry = BM25IndexRegistry(index_dir)
    registry.load_all()
    sparse = SparseRetriever(index=registry.index)
    hybrid = HybridRetriever(dense=dense, sparse=sparse, config=HybridConfig())
    reranker = get_reranker(backend=os.getenv("RERANKER_BACKEND", "none"))

    class _Retriever:
        async def search(self, query: str, top_k: int):
            hits = await hybrid.search(query=query, db=db, top_k=top_k * 4)
            return reranker.rerank(query, hits, top_n=top_k)

    # Build LLM generator
    llm_client = get_llm_client(
        provider=settings.llm_provider,
        model=payload.model_name,
        api_key=settings.openai_api_key or settings.anthropic_api_key or None,
    )
    generator = AnswerGenerator(llm_client=llm_client)

    # Emit start audit event
    run_id = uuid.uuid4()
    db.add(AuditEvent(
        id=uuid.uuid4(),
        event_type="eval_run_started",
        resource_type="eval_run",
        resource_id=run_id,
        detail={"label": payload.label, "total_cases": len(cases), "model": payload.model_name},
    ))
    await db.flush()

    # Execute runner
    runner = EvalRunner(retriever=_Retriever(), generator=generator, top_k=10)
    result = await runner.run(cases=cases, label=payload.label, model_name=payload.model_name)

    # Persist EvalRun record
    run_row = EvalRun(
        id=run_id,
        label=result.label,
        model_name=result.model_name,
        retriever_config={
            "fusion": payload.retriever_config.get("fusion", "rrf"),
            "reranker": os.getenv("RERANKER_BACKEND", "none"),
        },
        total_cases=result.total_cases,
        recall_at_10=result.recall_at_10,
        precision_at_10=result.precision_at_10,
        mrr=result.mrr,
        faithfulness_score=result.citation_recall,
        refusal_accuracy=result.refusal_accuracy,
        mean_latency_ms=result.mean_latency_ms,
    )
    db.add(run_row)

    db.add(AuditEvent(
        id=uuid.uuid4(),
        event_type="eval_run_completed",
        resource_type="eval_run",
        resource_id=run_id,
        detail={
            "total_cases": result.total_cases,
            "recall_at_10": round(result.recall_at_10, 4),
            "mrr": round(result.mrr, 4),
            "citation_recall": round(result.citation_recall, 4),
            "refusal_accuracy": round(result.refusal_accuracy, 4),
        },
    ))

    await db.commit()
    await db.refresh(run_row)
    return EvalRunRead.model_validate(run_row)


@router.get("/runs", response_model=list[EvalRunRead])
async def list_eval_runs(
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(EvalRun).order_by(EvalRun.created_at.desc()).offset(offset).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    return [EvalRunRead.model_validate(r) for r in rows]


@router.get("/runs/{run_id}", response_model=EvalRunRead)
async def get_eval_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = await db.get(EvalRun, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return EvalRunRead.model_validate(row)
