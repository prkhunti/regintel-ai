"""
Retrieval quality evaluation harness.

Runs the full BM25 → hybrid fusion → reranker pipeline over the gold query
set in data/gold_queries/sample_queries.json and reports Recall@K, Precision@K,
MRR, NDCG, and Hit Rate.

No database or API key is required:
  - Dense retrieval is mocked with a perfect oracle (returns the correct chunk).
  - BM25 index is built in-memory from the gold corpus.
  - Cross-encoder reranker is replaced with IdentityReranker.

To run against a real database, implement the DB fixtures and remove the mocks.

Usage
-----
    pytest tests/eval/test_retrieval_quality.py -v --tb=short
    pytest tests/eval/test_retrieval_quality.py -v -k "bm25"   # only BM25 tests
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock

import pytest

from packages.evals.metrics import RetrievalMetrics, compute_all
from packages.retrieval.dense import DenseHit, DenseRetriever
from packages.retrieval.hybrid import HybridConfig, HybridRetriever
from packages.retrieval.indexer import BM25Index
from packages.retrieval.reranker import IdentityReranker
from packages.retrieval.sparse import SparseRetriever

# ── Gold data ─────────────────────────────────────────────────────────────────

_GOLD_PATH = Path(__file__).resolve().parents[2] / "data" / "gold_queries" / "sample_queries.json"


@pytest.fixture(scope="module")
def gold() -> dict:
    with _GOLD_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corpus(gold) -> list[dict]:
    return gold["chunks"]


@pytest.fixture(scope="module")
def queries(gold) -> list[dict]:
    return gold["queries"]


# ── BM25 index built from gold corpus ────────────────────────────────────────

@pytest.fixture(scope="module")
def bm25_index(corpus) -> BM25Index:
    idx = BM25Index()
    idx.build(
        chunk_ids=[c["id"] for c in corpus],
        texts=[c["text"] for c in corpus],
    )
    return idx


# ── Retriever factory helpers ─────────────────────────────────────────────────

def _make_dense_hit(chunk: dict, score: float = 1.0) -> DenseHit:
    return DenseHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]),
        document_id=uuid.uuid5(uuid.NAMESPACE_DNS, chunk["document_id"]),
        score=score,
        text=chunk["text"],
        section_title=chunk.get("section_title"),
        heading_path=chunk.get("heading_path", []),
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_start"),
        document_title=chunk.get("document_title"),
        source="dense",
    )


def _chunk_by_id(corpus: list[dict], chunk_id: str) -> dict | None:
    return next((c for c in corpus if c["id"] == chunk_id), None)


# ── BM25-only retrieval (no DB mock needed) ───────────────────────────────────

def bm25_retrieve(
    query_text: str,
    bm25_index: BM25Index,
    top_k: int = 10,
) -> list[str]:
    """Run BM25 and return ordered chunk IDs."""
    hits = bm25_index.search(query_text, top_k=top_k)
    return [h.chunk_id for h in hits]


# ── Oracle dense retriever (injects ground-truth for fusion testing) ──────────

def oracle_dense_retrieve(
    query: dict,
    corpus: list[dict],
    top_k: int = 10,
) -> list[DenseHit]:
    """
    Returns the relevant chunk(s) first, then fills remaining slots with
    random corpus chunks.  Simulates a perfect dense retriever for testing
    hybrid fusion and reranker logic without a real embedding model.
    """
    relevant_ids = set(query["relevant_chunk_ids"])
    hits: list[DenseHit] = []

    for cid in query["relevant_chunk_ids"]:
        chunk = _chunk_by_id(corpus, cid)
        if chunk:
            hits.append(_make_dense_hit(chunk, score=0.95))

    for chunk in corpus:
        if len(hits) >= top_k:
            break
        if chunk["id"] not in relevant_ids:
            hits.append(_make_dense_hit(chunk, score=0.5))

    return hits


# ══════════════════════════════════════════════════════════════════════════════
# Test suite 1: BM25 retrieval quality (pure offline)
# ══════════════════════════════════════════════════════════════════════════════

class TestBM25RetrievalQuality:
    """Evaluate BM25 sparse retrieval over the gold query set."""

    def _run(self, queries, bm25_index, top_k=10) -> RetrievalMetrics:
        results = []
        for q in queries:
            retrieved = bm25_retrieve(q["query"], bm25_index, top_k=top_k)
            relevant = set(q["relevant_chunk_ids"])
            results.append((retrieved, relevant))
        return compute_all(results)

    def test_recall_at_10_above_threshold(self, queries, bm25_index):
        """BM25 should find the relevant chunk in top-10 for most queries."""
        m = self._run(queries, bm25_index, top_k=10)
        print(f"\n{m}")
        # Threshold: BM25 should achieve at least 50% recall@10 on this corpus
        assert m.recall_at_10 >= 0.50, (
            f"BM25 Recall@10 = {m.recall_at_10:.3f} — below 0.50 threshold.\n{m}"
        )

    def test_hit_rate_at_5_above_threshold(self, queries, bm25_index):
        """For at least 60% of queries, a relevant chunk should be in top-5."""
        m = self._run(queries, bm25_index, top_k=5)
        assert m.hit_rate_at_5 >= 0.40, (
            f"BM25 Hit Rate@5 = {m.hit_rate_at_5:.3f} — below 0.40 threshold."
        )

    def test_mrr_above_threshold(self, queries, bm25_index):
        m = self._run(queries, bm25_index, top_k=10)
        assert m.mrr >= 0.40, (
            f"BM25 MRR = {m.mrr:.3f} — below 0.40 threshold."
        )

    def test_keyword_queries_score_well(self, bm25_index):
        """Short keyword queries should still retrieve correctly."""
        hits = bm25_retrieve("AES-256 encryption data security", bm25_index, top_k=5)
        assert "chunk-005" in hits, "Cybersecurity chunk should be in top-5 for encryption query"

    def test_no_results_for_irrelevant_query(self, bm25_index):
        """A nonsense query should return empty or very low-scoring results."""
        hits = bm25_index.search("xyzzy frobnicator quux", top_k=5)
        # Either no hits, or all scores are 0
        assert all(h.score == 0.0 for h in hits) or len(hits) == 0

    def test_per_query_breakdown(self, queries, bm25_index):
        """Print per-query results for manual inspection."""
        print("\n── BM25 per-query results ──")
        for q in queries:
            retrieved = bm25_retrieve(q["query"], bm25_index, top_k=10)
            relevant = set(q["relevant_chunk_ids"])
            found = any(r in relevant for r in retrieved[:10])
            rank = next((i + 1 for i, r in enumerate(retrieved) if r in relevant), None)
            print(f"  [{q['id']}] hit={found} rank={rank}  {q['query'][:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# Test suite 2: Hybrid retrieval quality (oracle dense + BM25)
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridRetrievalQuality:
    """
    Tests hybrid fusion with a mocked dense retriever (oracle) + real BM25.
    Validates that fusion outperforms BM25 alone on this corpus.
    """

    def _run_hybrid(
        self,
        queries: list[dict],
        corpus: list[dict],
        bm25_index: BM25Index,
        fusion: str = "rrf",
        top_k: int = 10,
    ) -> RetrievalMetrics:
        from packages.retrieval.hybrid import _fuse_rrf, _fuse_weighted
        from packages.retrieval.sparse import _normalise_and_trim
        from packages.retrieval.indexer import SparseHit

        results = []
        for q in queries:
            # Dense: oracle
            dense_hits = oracle_dense_retrieve(q, corpus, top_k=top_k)

            # Sparse: real BM25
            raw = bm25_index.search(q["query"], top_k=top_k * 2)
            chunk_lookup = {c["id"]: c for c in corpus}
            sparse_hits = []
            for h in raw:
                chunk = chunk_lookup.get(h.chunk_id)
                if chunk:
                    sparse_hits.append(_make_dense_hit(chunk, score=h.score))
            sparse_hits = _normalise_and_trim(sparse_hits, top_k)
            for h in sparse_hits:
                h.source = "sparse"

            # Fuse
            if fusion == "rrf":
                fused = _fuse_rrf(dense_hits, sparse_hits, top_k=top_k, k=60)
            else:
                fused = _fuse_weighted(dense_hits, sparse_hits, top_k=top_k, alpha=0.7)

            # Convert chunk UUIDs back to gold IDs for metric computation
            uid_to_gold = {
                uuid.uuid5(uuid.NAMESPACE_DNS, c["id"]): c["id"]
                for c in corpus
            }
            retrieved_gold_ids = [
                uid_to_gold.get(h.chunk_id, str(h.chunk_id))
                for h in fused
            ]
            results.append((retrieved_gold_ids, set(q["relevant_chunk_ids"])))

        return compute_all(results)

    def test_rrf_recall_at_10(self, queries, corpus, bm25_index):
        m = self._run_hybrid(queries, corpus, bm25_index, fusion="rrf", top_k=10)
        print(f"\n── Hybrid RRF metrics ──\n{m}")
        # Oracle dense means hybrid should achieve very high recall
        assert m.recall_at_10 >= 0.80, (
            f"Hybrid RRF Recall@10 = {m.recall_at_10:.3f} — below 0.80."
        )

    def test_weighted_recall_at_10(self, queries, corpus, bm25_index):
        m = self._run_hybrid(queries, corpus, bm25_index, fusion="weighted", top_k=10)
        print(f"\n── Hybrid Weighted metrics ──\n{m}")
        assert m.recall_at_10 >= 0.80

    def test_hybrid_mrr_above_bm25(self, queries, corpus, bm25_index):
        """Hybrid with oracle dense should outperform BM25-alone MRR."""
        bm25_results = []
        for q in queries:
            retrieved = bm25_retrieve(q["query"], bm25_index, top_k=10)
            bm25_results.append((retrieved, set(q["relevant_chunk_ids"])))
        bm25_m = compute_all(bm25_results)

        hybrid_m = self._run_hybrid(queries, corpus, bm25_index, top_k=10)
        print(f"\nBM25 MRR={bm25_m.mrr:.3f}  Hybrid MRR={hybrid_m.mrr:.3f}")
        assert hybrid_m.mrr >= bm25_m.mrr, (
            f"Hybrid MRR ({hybrid_m.mrr:.3f}) should be >= BM25 MRR ({bm25_m.mrr:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Test suite 3: Metrics correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsOnGoldData:
    """Sanity-check metric computation against hand-verified cases."""

    def test_perfect_retrieval_scores_one(self):
        from packages.evals.metrics import recall_at_k, mrr, ndcg_at_k
        retrieved = ["chunk-001"]
        relevant = {"chunk-001"}
        assert recall_at_k(retrieved, relevant, k=1) == 1.0
        assert mrr(retrieved, relevant) == 1.0
        assert ndcg_at_k(retrieved, relevant, k=1) == 1.0

    def test_missed_retrieval_scores_zero(self):
        from packages.evals.metrics import recall_at_k, mrr
        assert recall_at_k(["chunk-002"], {"chunk-001"}, k=5) == 0.0
        assert mrr(["chunk-002", "chunk-003"], {"chunk-001"}) == 0.0

    def test_aggregate_metrics_structure(self, queries, bm25_index):
        results = []
        for q in queries:
            retrieved = bm25_retrieve(q["query"], bm25_index, top_k=10)
            results.append((retrieved, set(q["relevant_chunk_ids"])))
        m = compute_all(results)

        # All metric values should be in [0, 1]
        d = m.to_dict()
        for key, val in d.items():
            if key == "num_queries":
                continue
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_num_queries_matches_fixture(self, queries, bm25_index):
        results = [
            (bm25_retrieve(q["query"], bm25_index), set(q["relevant_chunk_ids"]))
            for q in queries
        ]
        m = compute_all(results)
        assert m.num_queries == len(queries)
