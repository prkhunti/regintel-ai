"""
Retrieval quality metrics.

All functions operate on plain Python lists/sets of string IDs so they can be
used independently of the database or retrieval implementation.

Conventions
-----------
retrieved   : ordered list of chunk IDs returned by the retriever (rank 0 = best)
relevant    : set of chunk IDs that are known relevant for the query
k           : cut-off depth
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Individual metrics ────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute recall at rank ``k``.

    Parameters
    ----------
    retrieved
        Ranked retrieved chunk identifiers.
    relevant
        Set of known relevant chunk identifiers.
    k
        Depth cutoff applied to ``retrieved``.

    Returns
    -------
    float
        Fraction of relevant identifiers present in the top-``k`` retrieved results.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute precision at rank ``k``.

    Parameters
    ----------
    retrieved
        Ranked retrieved chunk identifiers.
    relevant
        Set of known relevant chunk identifiers.
    k
        Depth cutoff applied to ``retrieved``.

    Returns
    -------
    float
        Fraction of the top-``k`` retrieved identifiers that are relevant.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute hit rate at rank ``k``.

    Parameters
    ----------
    retrieved
        Ranked retrieved chunk identifiers.
    relevant
        Set of known relevant chunk identifiers.
    k
        Depth cutoff applied to ``retrieved``.

    Returns
    -------
    float
        ``1.0`` when at least one relevant identifier appears in the top-``k`` results, else ``0.0``.
    """
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Compute reciprocal rank for a single query.

    Parameters
    ----------
    retrieved
        Ranked retrieved chunk identifiers.
    relevant
        Set of known relevant chunk identifiers.

    Returns
    -------
    float
        Reciprocal of the first relevant rank, or ``0.0`` when no relevant item is retrieved.
    """
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Compute average precision for a single query.

    Parameters
    ----------
    retrieved
        Ranked retrieved chunk identifiers.
    relevant
        Set of known relevant chunk identifiers.

    Returns
    -------
    float
        Average precision over the ranked list, or ``0.0`` when ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute binary NDCG at rank ``k``.

    Parameters
    ----------
    retrieved
        Ranked retrieved chunk identifiers.
    relevant
        Set of known relevant chunk identifiers.
    k
        Depth cutoff applied to ``retrieved``.

    Returns
    -------
    float
        Normalised discounted cumulative gain for the top-``k`` results.
    """
    if not relevant or k == 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Aggregate result dataclass ────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Macro-averaged retrieval metrics across a query set.

    Parameters
    ----------
    num_queries
        Number of queries included in the aggregate.
    recall_at_1
        Mean recall at rank 1.
    recall_at_5
        Mean recall at rank 5.
    recall_at_10
        Mean recall at rank 10.
    precision_at_1
        Mean precision at rank 1.
    precision_at_5
        Mean precision at rank 5.
    precision_at_10
        Mean precision at rank 10.
    hit_rate_at_5
        Mean hit rate at rank 5.
    hit_rate_at_10
        Mean hit rate at rank 10.
    mrr
        Mean reciprocal rank.
    map_score
        Mean average precision.
    ndcg_at_5
        Mean NDCG at rank 5.
    ndcg_at_10
        Mean NDCG at rank 10.
    per_query
        Optional per-query metric breakdowns.
    """
    num_queries: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    hit_rate_at_5: float = 0.0
    hit_rate_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0       # Mean Average Precision
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    per_query: list[dict] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics over {self.num_queries} queries\n"
            f"  Recall     @1={self.recall_at_1:.3f}  @5={self.recall_at_5:.3f}  @10={self.recall_at_10:.3f}\n"
            f"  Precision  @1={self.precision_at_1:.3f}  @5={self.precision_at_5:.3f}  @10={self.precision_at_10:.3f}\n"
            f"  Hit Rate   @5={self.hit_rate_at_5:.3f}  @10={self.hit_rate_at_10:.3f}\n"
            f"  MRR        {self.mrr:.3f}\n"
            f"  MAP        {self.map_score:.3f}\n"
            f"  NDCG       @5={self.ndcg_at_5:.3f}  @10={self.ndcg_at_10:.3f}"
        )

    def to_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "precision_at_1": round(self.precision_at_1, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "hit_rate_at_5": round(self.hit_rate_at_5, 4),
            "hit_rate_at_10": round(self.hit_rate_at_10, 4),
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
        }


def compute_all(
    results: list[tuple[list[str], set[str]]],
) -> RetrievalMetrics:
    """Aggregate retrieval metrics over multiple queries.

    Parameters
    ----------
    results
        List of ``(retrieved_ids, relevant_ids)`` pairs, one entry per query.

    Returns
    -------
    RetrievalMetrics
        Macro-averaged retrieval metrics and optional per-query breakdowns.
    """
    n = len(results)
    if n == 0:
        return RetrievalMetrics()

    agg = {
        "recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0,
        "precision_at_1": 0.0, "precision_at_5": 0.0, "precision_at_10": 0.0,
        "hit_rate_at_5": 0.0, "hit_rate_at_10": 0.0,
        "mrr": 0.0, "map_score": 0.0,
        "ndcg_at_5": 0.0, "ndcg_at_10": 0.0,
    }
    per_query = []

    for retrieved, relevant in results:
        q = {
            "recall_at_1":     recall_at_k(retrieved, relevant, 1),
            "recall_at_5":     recall_at_k(retrieved, relevant, 5),
            "recall_at_10":    recall_at_k(retrieved, relevant, 10),
            "precision_at_1":  precision_at_k(retrieved, relevant, 1),
            "precision_at_5":  precision_at_k(retrieved, relevant, 5),
            "precision_at_10": precision_at_k(retrieved, relevant, 10),
            "hit_rate_at_5":   hit_rate_at_k(retrieved, relevant, 5),
            "hit_rate_at_10":  hit_rate_at_k(retrieved, relevant, 10),
            "mrr":             mrr(retrieved, relevant),
            "map_score":       average_precision(retrieved, relevant),
            "ndcg_at_5":       ndcg_at_k(retrieved, relevant, 5),
            "ndcg_at_10":      ndcg_at_k(retrieved, relevant, 10),
        }
        for key in agg:
            agg[key] += q[key]
        per_query.append(q)

    return RetrievalMetrics(
        num_queries=n,
        per_query=per_query,
        **{k: v / n for k, v in agg.items()},
    )
