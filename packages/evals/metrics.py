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
    """
    Fraction of relevant documents found in the top-k results.

        Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Returns 0.0 when *relevant* is empty.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of the top-k results that are relevant.

        Precision@k = |retrieved[:k] ∩ relevant| / k

    Returns 0.0 when k == 0.
    """
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    1.0 if at least one relevant document appears in the top-k results, else 0.0.
    Also known as Recall@k with |relevant|=1, but useful as a standalone signal.
    """
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """
    Mean Reciprocal Rank for a single query.

        MRR = 1 / rank_of_first_relevant

    rank is 1-based.  Returns 0.0 if no relevant document is retrieved.
    """
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Average Precision (AP) for a single query — area under the P-R curve.

        AP = (1/|relevant|) * Σ Precision@k * rel(k)

    where rel(k) = 1 if the k-th retrieved item is relevant, else 0.
    Returns 0.0 when *relevant* is empty.
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
    """
    Normalised Discounted Cumulative Gain at depth k.

    Binary relevance: rel(i) = 1 if retrieved[i] ∈ relevant, else 0.

        DCG@k  = Σ rel(i) / log2(i + 2)   (i is 0-based)
        IDCG@k = Σ 1      / log2(i + 2)   for min(|relevant|, k) ideal results
        NDCG@k = DCG@k / IDCG@k
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
    """Aggregate metrics across a set of queries."""
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
    """
    Compute aggregate retrieval metrics over a set of queries.

    Args:
        results: List of (retrieved_ids, relevant_ids) pairs — one per query.

    Returns:
        ``RetrievalMetrics`` with macro-averaged scores.
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
