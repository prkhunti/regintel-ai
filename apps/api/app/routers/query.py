"""
Query + answer endpoints.

POST /query          — full pipeline: retrieve → rerank → generate answer
GET  /query/{id}     — fetch a stored query + response by ID
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..database import get_db
from ..models import AuditEvent, Query, Response, Citation, RetrievalRun, RetrievedChunk
from ..services.answer_service import AnswerGenerator, InlineCitation
from ..services.confidence import ConfidenceScorer
from ..services.llm_client import get_llm_client
from packages.retrieval.dense import DenseHit, DenseRetriever
from packages.retrieval.embedder import get_embedder
from packages.retrieval.hybrid import HybridConfig, HybridRetriever
from packages.retrieval.reranker import IdentityReranker, get_reranker
from packages.retrieval.sparse import BM25IndexRegistry, SparseRetriever
from packages.schemas import AnswerPayload, QueryCreate, QueryRead
from packages.schemas.common import QueryType

settings = get_settings()
router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)


@router.post("", response_model=AnswerPayload, status_code=status.HTTP_200_OK)
async def ask(
    payload: QueryCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Full retrieval + answer pipeline.

    1. Embed query → hybrid retrieve → rerank
    2. Generate grounded answer with inline citations
    3. Persist query, retrieval run, response, citations, audit event
    4. Return AnswerPayload
    """
    t_start = time.perf_counter()

    # ── Build retriever stack ─────────────────────────────────────────────────
    embedder = get_embedder(
        provider=settings.llm_provider,
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        batch_size=settings.embedding_batch_size,
    )
    dense = DenseRetriever(embedder=embedder, embedding_dim=settings.embedding_dim)

    # BM25 index loaded from disk (best-effort; falls back to dense-only if unavailable)
    import os
    index_dir = os.getenv("INDEX_DIR", "/tmp/regintel/indexes")
    registry = BM25IndexRegistry(index_dir)
    registry.load_all()
    sparse = SparseRetriever(index=registry.index)

    hybrid = HybridRetriever(dense=dense, sparse=sparse, config=HybridConfig())
    reranker = get_reranker(backend=os.getenv("RERANKER_BACKEND", "none"))

    # ── Retrieve ──────────────────────────────────────────────────────────────
    t_ret = time.perf_counter()
    hybrid_hits = await hybrid.search(
        query=payload.user_query,
        db=db,
        top_k=40,
        document_ids=payload.document_ids,
        document_type_filter=[str(t) for t in payload.document_type_filter] if payload.document_type_filter else None,
    )
    reranked_hits = reranker.rerank(payload.user_query, hybrid_hits, top_n=payload.top_k)
    retrieval_ms = int((time.perf_counter() - t_ret) * 1000)

    # ── Generate answer ───────────────────────────────────────────────────────
    llm_client = get_llm_client(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key or settings.anthropic_api_key or None,
    )
    generator = AnswerGenerator(llm_client=llm_client)
    answer = await generator.generate(query=payload.user_query, hits=reranked_hits)

    # ── Score confidence ──────────────────────────────────────────────────────
    confidence = ConfidenceScorer().score(answer, reranked_hits)
    logger.info(
        "Confidence: overall=%.3f top_chunk=%.3f citation=%.3f coverage=%.3f risk=%s",
        confidence.overall, confidence.top_chunk_score,
        confidence.citation_density, confidence.coverage_ratio,
        confidence.risk_level,
    )

    total_ms = int((time.perf_counter() - t_start) * 1000)

    # ── Persist ───────────────────────────────────────────────────────────────
    query_record = Query(
        id=uuid.uuid4(),
        user_query=payload.user_query,
        query_type=str(payload.query_type),
    )
    db.add(query_record)
    await db.flush()

    retrieval_run = RetrievalRun(
        id=uuid.uuid4(),
        query_id=query_record.id,
        retriever_config={"fusion": "rrf", "reranker": os.getenv("RERANKER_BACKEND", "none")},
        top_k=payload.top_k,
        dense_hits=len([h for h in hybrid_hits if h.source == "dense"]),
        sparse_hits=len([h for h in hybrid_hits if h.source == "sparse"]),
        reranked_hits=len(reranked_hits),
        latency_ms=retrieval_ms,
    )
    db.add(retrieval_run)
    await db.flush()

    for rank, hit in enumerate(reranked_hits):
        db.add(RetrievedChunk(
            id=uuid.uuid4(),
            retrieval_run_id=retrieval_run.id,
            chunk_id=hit.chunk_id,
            rank=rank,
            final_score=hit.score,
        ))

    response_record = Response(
        id=uuid.uuid4(),
        query_id=query_record.id,
        model_name=answer.model_name,
        prompt_version=answer.prompt_version,
        answer_text=answer.answer_text,
        confidence_score=confidence.overall,
        refusal_reason=answer.refusal_reason,
        risk_level=str(confidence.risk_level),
        latency_ms=answer.latency_ms,
    )
    db.add(response_record)
    await db.flush()

    for cit in answer.citations:
        db.add(Citation(
            id=uuid.uuid4(),
            response_id=response_record.id,
            chunk_id=cit.chunk_id,
            quote=cit.quote,
            relevance_score=cit.relevance_score,
        ))

    db.add(AuditEvent(
        id=uuid.uuid4(),
        event_type="answer_generated" if not answer.refused else "answer_refused",
        resource_type="response",
        resource_id=response_record.id,
        detail={"total_ms": total_ms, "chunks_retrieved": len(reranked_hits)},
    ))

    await db.commit()

    # ── Build response payload ────────────────────────────────────────────────
    from packages.schemas import CitationRead
    from packages.schemas.common import RiskLevel

    citation_reads = [
        CitationRead(
            id=uuid.uuid4(),
            chunk_id=c.chunk_id,
            document_title=c.document_title or "",
            section_title=c.section_title,
            page_start=c.page_start,
            page_end=c.page_end,
            quote=c.quote,
            relevance_score=c.relevance_score,
        )
        for c in answer.citations
    ]

    return AnswerPayload(
        query_id=query_record.id,
        response_id=response_record.id,
        query_text=payload.user_query,
        answer=answer.answer_text,
        confidence=confidence.overall,
        risk_level=confidence.risk_level,
        citations=citation_reads,
        evidence_snippets=[h.text[:200] for h in reranked_hits[:3]],
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        latency_ms=total_ms,
    )


@router.get("/{query_id}", response_model=QueryRead)
async def get_query(query_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    record = await db.get(Query, query_id)
    if not record:
        raise HTTPException(status_code=404, detail="Query not found")
    return QueryRead(
        id=record.id,
        user_query=record.user_query,
        query_type=record.query_type,
        rewritten_query=record.rewritten_query,
        created_at=record.created_at,
    )
