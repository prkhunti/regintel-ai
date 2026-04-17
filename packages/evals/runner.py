"""
Evaluation runner.

Orchestrates the full retrieve → generate → score pipeline over a set of
EvalCase records and returns an EvalRunResult with aggregate metrics.

Design
------
* Pure-Python dataclasses — no SQLAlchemy imports here.  The API router
  handles all DB reads/writes; the runner is given plain objects.
* Every case produces a CaseResult.  Aggregate metrics are computed from
  the full list at the end.
* Designed for both async API invocation and sync CLI/test use.

Metrics computed
----------------
Retrieval (requires expected_chunk_ids on the case):
  recall@10, precision@10, MRR

Answer quality:
  citation_recall  — fraction of expected chunks cited in the answer
  refusal_accuracy — 1.0 if the refusal matches is_insufficient, else 0.0
  mean_latency_ms
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

from packages.evals.metrics import (
    RetrievalMetrics,
    compute_all,
    mrr as mrr_fn,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


# ── Input / output dataclasses ────────────────────────────────────────────────

@dataclass
class EvalCaseInput:
    """Flat eval-case input consumed by :class:`EvalRunner`.

    Parameters
    ----------
    id
        Stable identifier for the eval case.
    query
        User query to run through retrieval and generation.
    expected_chunk_ids
        Chunk identifiers treated as relevant for retrieval scoring.
    is_insufficient
        Indicates whether the gold answer should be a refusal.
    notes
        Optional free-form evaluator notes.
    """
    id: str
    query: str
    expected_chunk_ids: list[str]   # string IDs (as stored in DB / gold set)
    is_insufficient: bool = False
    notes: str | None = None


@dataclass
class CaseResult:
    """Per-case evaluation output.

    Parameters
    ----------
    case_id
        Stable identifier of the eval case.
    query
        Query text used for the evaluation.
    retrieved_ids
        Retrieved chunk identifiers in ranked order.
    cited_ids
        Chunk identifiers cited by the generated answer.
    refused
        Indicates whether the generator refused to answer.
    recall_at_10
        Recall at rank 10 for retrieved chunks.
    precision_at_10
        Precision at rank 10 for retrieved chunks.
    mrr
        Reciprocal rank for the first relevant retrieved chunk.
    citation_recall
        Fraction of expected chunks that were cited in the answer.
    refusal_correct
        Whether the refusal decision matched the gold label.
    latency_ms
        End-to-end latency for the case in milliseconds.
    """
    case_id: str
    query: str
    retrieved_ids: list[str]
    cited_ids: list[str]
    refused: bool

    # Retrieval
    recall_at_10: float
    precision_at_10: float
    mrr: float

    # Answer quality
    citation_recall: float    # fraction of expected chunks that were cited
    refusal_correct: bool     # True if refused == is_insufficient

    latency_ms: int


@dataclass
class EvalRunResult:
    """Aggregate evaluation output for a batch of cases.

    Parameters
    ----------
    label
        Human-readable label for the eval run.
    model_name
        Model or system name under evaluation.
    total_cases
        Number of cases included in the run.
    recall_at_10
        Mean recall at rank 10.
    precision_at_10
        Mean precision at rank 10.
    mrr
        Mean reciprocal rank.
    citation_recall
        Mean citation recall across generated answers.
    refusal_accuracy
        Fraction of cases where refusal behaviour matched expectations.
    mean_latency_ms
        Mean end-to-end latency across all cases.
    per_case
        Detailed results for each eval case.
    """
    label: str
    model_name: str
    total_cases: int

    # Retrieval (macro-averaged)
    recall_at_10: float
    precision_at_10: float
    mrr: float

    # Answer quality (macro-averaged)
    citation_recall: float       # proxy for faithfulness
    refusal_accuracy: float      # fraction of refusal decisions that are correct
    mean_latency_ms: int

    per_case: list[CaseResult] = field(default_factory=list, repr=False)


# ── Retriever / generator protocols ──────────────────────────────────────────

class RetrieverProtocol(Protocol):
    async def search(self, query: str, top_k: int) -> list:
        """Return list of DenseHit-like objects with .chunk_id and .score."""
        ...


class GeneratorProtocol(Protocol):
    async def generate(self, query: str, hits: list) -> object:
        """Return a GeneratedAnswer-like object with .citations, .refused."""
        ...


# ── Runner ────────────────────────────────────────────────────────────────────

class EvalRunner:
    """
    Runs eval cases through the retrieval + answer pipeline.

    Parameters
    ----------
    retriever
        Anything with ``async search(query, top_k) -> list[DenseHit]``.
        Pass ``None`` to skip retrieval-dependent metrics.
    generator
        Anything with ``async generate(query, hits) -> GeneratedAnswer``.
        Pass ``None`` to skip answer quality metrics.
    top_k
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol | None,
        generator: GeneratorProtocol | None,
        top_k: int = 10,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    async def run(
        self,
        cases: list[EvalCaseInput],
        label: str,
        model_name: str,
    ) -> EvalRunResult:
        """Run retrieval and generation evaluation across multiple cases.

        Parameters
        ----------
        cases
            Eval cases to execute.
        label
            Human-readable label for the run.
        model_name
            Model or system name associated with the run.

        Returns
        -------
        EvalRunResult
            Aggregate evaluation metrics and per-case outputs.
        """
        if not cases:
            return EvalRunResult(
                label=label,
                model_name=model_name,
                total_cases=0,
                recall_at_10=0.0,
                precision_at_10=0.0,
                mrr=0.0,
                citation_recall=0.0,
                refusal_accuracy=0.0,
                mean_latency_ms=0,
            )

        case_results: list[CaseResult] = []
        for case in cases:
            result = await self._run_case(case)
            case_results.append(result)
            logger.info(
                "Eval case %s: recall@10=%.3f citation_recall=%.3f refusal_correct=%s latency=%dms",
                case.id, result.recall_at_10, result.citation_recall,
                result.refusal_correct, result.latency_ms,
            )

        return _aggregate(label, model_name, case_results)

    async def _run_case(self, case: EvalCaseInput) -> CaseResult:
        t0 = time.perf_counter()

        # Retrieve
        hits: list = []
        if self._retriever is not None:
            hits = await self._retriever.search(case.query, top_k=self._top_k)

        retrieved_ids = [str(h.chunk_id) for h in hits]

        # Generate answer
        refused = False
        cited_ids: list[str] = []
        if self._generator is not None:
            answer = await self._generator.generate(case.query, hits)
            refused = answer.refused
            cited_ids = [str(c.chunk_id) for c in answer.citations]

        latency_ms = int((time.perf_counter() - t0) * 1000)

        relevant = set(case.expected_chunk_ids)

        # Retrieval metrics
        r10   = recall_at_k(retrieved_ids, relevant, 10)
        p10   = precision_at_k(retrieved_ids, relevant, 10)
        mrr_v = mrr_fn(retrieved_ids, relevant)

        # Citation recall: how many expected chunks appear in answer citations?
        cit_recall = (
            len(relevant & set(cited_ids)) / len(relevant)
            if relevant else 0.0
        )

        return CaseResult(
            case_id=case.id,
            query=case.query,
            retrieved_ids=retrieved_ids,
            cited_ids=cited_ids,
            refused=refused,
            recall_at_10=r10,
            precision_at_10=p10,
            mrr=mrr_v,
            citation_recall=cit_recall,
            refusal_correct=(refused == case.is_insufficient),
            latency_ms=latency_ms,
        )


# ── Aggregation helper ────────────────────────────────────────────────────────

def _aggregate(label: str, model_name: str, results: list[CaseResult]) -> EvalRunResult:
    n = len(results)
    if n == 0:
        return EvalRunResult(label=label, model_name=model_name, total_cases=0,
                             recall_at_10=0.0, precision_at_10=0.0, mrr=0.0,
                             citation_recall=0.0, refusal_accuracy=0.0, mean_latency_ms=0)

    return EvalRunResult(
        label=label,
        model_name=model_name,
        total_cases=n,
        recall_at_10=sum(r.recall_at_10 for r in results) / n,
        precision_at_10=sum(r.precision_at_10 for r in results) / n,
        mrr=sum(r.mrr for r in results) / n,
        citation_recall=sum(r.citation_recall for r in results) / n,
        refusal_accuracy=sum(1 for r in results if r.refusal_correct) / n,
        mean_latency_ms=int(sum(r.latency_ms for r in results) / n),
        per_case=results,
    )
