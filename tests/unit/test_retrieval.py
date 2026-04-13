"""
Unit tests for retrieval components.

All tests are offline — no database, no API keys required.
Dense/sparse DB queries are mocked; BM25 runs in-memory.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.retrieval.dense import DenseHit
from packages.retrieval.hybrid import HybridConfig, _fuse_rrf, _fuse_weighted
from packages.retrieval.indexer import BM25Index
from packages.retrieval.reranker import CrossEncoderReranker, IdentityReranker, get_reranker
from packages.retrieval.sparse import BM25IndexRegistry, _normalise_and_trim, _parse_raw_hits
from packages.evals.metrics import (
    average_precision, hit_rate_at_k, mrr, ndcg_at_k,
    precision_at_k, recall_at_k, compute_all,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_hit(
    chunk_id: str | None = None,
    score: float = 1.0,
    source: str = "dense",
    text: str = "sample text",
) -> DenseHit:
    return DenseHit(
        chunk_id=uuid.UUID(chunk_id) if chunk_id else uuid.uuid4(),
        document_id=uuid.uuid4(),
        score=score,
        text=text,
        section_title=None,
        heading_path=[],
        page_start=1,
        page_end=1,
        document_title="Test Doc",
        source=source,
    )


ID_A = str(uuid.uuid4())
ID_B = str(uuid.uuid4())
ID_C = str(uuid.uuid4())
ID_D = str(uuid.uuid4())
ID_E = str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k([ID_A, ID_B], {ID_A, ID_B}, k=2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k([ID_C], {ID_A, ID_B}, k=1) == 0.0

    def test_partial_recall(self):
        assert recall_at_k([ID_A, ID_C], {ID_A, ID_B}, k=2) == pytest.approx(0.5)

    def test_k_smaller_than_list(self):
        # Only top-k is considered
        assert recall_at_k([ID_C, ID_A], {ID_A}, k=1) == 0.0
        assert recall_at_k([ID_A, ID_C], {ID_A}, k=1) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k([ID_A], set(), k=5) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([ID_A, ID_B], {ID_A, ID_B}, k=2) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([ID_C, ID_D], {ID_A, ID_B}, k=2) == 0.0

    def test_half_relevant(self):
        assert precision_at_k([ID_A, ID_C], {ID_A, ID_B}, k=2) == pytest.approx(0.5)

    def test_k_zero(self):
        assert precision_at_k([ID_A], {ID_A}, k=0) == 0.0


class TestHitRateAtK:
    def test_hit_in_top_k(self):
        assert hit_rate_at_k([ID_C, ID_A], {ID_A}, k=2) == 1.0

    def test_hit_outside_top_k(self):
        assert hit_rate_at_k([ID_C, ID_A], {ID_A}, k=1) == 0.0

    def test_no_hit(self):
        assert hit_rate_at_k([ID_C, ID_D], {ID_A}, k=5) == 0.0


class TestMRR:
    def test_first_result_relevant(self):
        assert mrr([ID_A, ID_B], {ID_A}) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mrr([ID_C, ID_A], {ID_A}) == pytest.approx(0.5)

    def test_third_result_relevant(self):
        assert mrr([ID_C, ID_D, ID_A], {ID_A}) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert mrr([ID_C, ID_D], {ID_A}) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision([ID_A, ID_B], {ID_A, ID_B}) == pytest.approx(1.0)

    def test_reversed_order(self):
        # Both retrieved, but relevant one is second
        ap = average_precision([ID_C, ID_A], {ID_A})
        assert ap == pytest.approx(0.5)

    def test_empty_relevant(self):
        assert average_precision([ID_A], set()) == 0.0


class TestNDCG:
    def test_perfect(self):
        assert ndcg_at_k([ID_A, ID_B], {ID_A, ID_B}, k=2) == pytest.approx(1.0)

    def test_reversed_penalised(self):
        perfect = ndcg_at_k([ID_A, ID_B], {ID_A, ID_B}, k=2)
        reversed_ = ndcg_at_k([ID_B, ID_A], {ID_A}, k=2)
        assert perfect >= reversed_

    def test_k_zero(self):
        assert ndcg_at_k([ID_A], {ID_A}, k=0) == 0.0

    def test_none_relevant_at_k(self):
        assert ndcg_at_k([ID_C, ID_D], {ID_A}, k=2) == 0.0


class TestComputeAll:
    def test_perfect_retrieval(self):
        results = [([ID_A, ID_B], {ID_A, ID_B})]
        m = compute_all(results)
        assert m.recall_at_10 == pytest.approx(1.0)
        assert m.mrr == pytest.approx(1.0)
        assert m.num_queries == 1

    def test_empty_results(self):
        m = compute_all([])
        assert m.num_queries == 0
        assert m.mrr == 0.0

    def test_macro_average(self):
        results = [
            ([ID_A], {ID_A}),   # MRR = 1.0
            ([ID_B, ID_A], {ID_A}),  # MRR = 0.5
        ]
        m = compute_all(results)
        assert m.mrr == pytest.approx(0.75)

    def test_to_dict_keys(self):
        m = compute_all([([ID_A], {ID_A})])
        d = m.to_dict()
        assert "recall_at_10" in d
        assert "mrr" in d
        assert "ndcg_at_10" in d


# ══════════════════════════════════════════════════════════════════════════════
# BM25 Index
# ══════════════════════════════════════════════════════════════════════════════

class TestBM25Index:
    def _build(self) -> BM25Index:
        idx = BM25Index()
        idx.build(
            chunk_ids=[ID_A, ID_B, ID_C, ID_D],
            texts=[
                "The device is intended for use in adult patients with cardiac conditions.",
                "Risk management activities were performed per ISO 14971.",
                "Software architecture follows IEC 62304 Class B requirements.",
                "Clinical evaluation report confirms substantial equivalence.",
            ],
        )
        return idx

    def test_returns_hits(self):
        idx = self._build()
        hits = idx.search("cardiac device intended use", top_k=2)
        assert len(hits) > 0

    def test_top_hit_is_most_relevant(self):
        idx = self._build()
        hits = idx.search("intended use adult patients", top_k=4)
        assert hits[0].chunk_id == ID_A

    def test_top_k_respected(self):
        idx = self._build()
        hits = idx.search("device", top_k=2)
        assert len(hits) <= 2

    def test_empty_query_returns_empty(self):
        idx = self._build()
        hits = idx.search("", top_k=5)
        assert hits == []

    def test_stopword_only_query_returns_empty(self):
        idx = self._build()
        hits = idx.search("the is for with", top_k=5)
        assert hits == []

    def test_save_and_load(self, tmp_path):
        idx = self._build()
        path = tmp_path / "test.bm25.pkl"
        idx.save(path)
        loaded = BM25Index.load(path)
        hits_orig = idx.search("risk management", top_k=4)
        hits_loaded = loaded.search("risk management", top_k=4)
        assert [h.chunk_id for h in hits_orig] == [h.chunk_id for h in hits_loaded]

    def test_add_incremental(self):
        idx = BM25Index()
        idx.build([ID_A], ["cardiac device evaluation"])
        idx.add(ID_B, "software architecture IEC 62304")
        hits = idx.search("software IEC", top_k=2)
        assert hits[0].chunk_id == ID_B


# ══════════════════════════════════════════════════════════════════════════════
# BM25IndexRegistry
# ══════════════════════════════════════════════════════════════════════════════

class TestBM25IndexRegistry:
    def test_load_all(self, tmp_path):
        # Write two per-document pkl files
        for doc_id, text in [(ID_A, "cardiac risk"), (ID_B, "software IEC 62304")]:
            idx = BM25Index()
            idx.build([str(uuid.uuid4())], [text])
            idx.save(tmp_path / f"{doc_id}.bm25.pkl")

        registry = BM25IndexRegistry(tmp_path)
        n = registry.load_all()
        assert n == 2

        hits = registry.index.search("cardiac", top_k=5)
        assert len(hits) > 0

    def test_empty_dir_returns_zero(self, tmp_path):
        registry = BM25IndexRegistry(tmp_path)
        assert registry.load_all() == 0


# ══════════════════════════════════════════════════════════════════════════════
# Sparse retriever helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestSparseHelpers:
    def test_parse_raw_hits_valid_uuids(self):
        from packages.retrieval.indexer import SparseHit
        raw = [
            SparseHit(chunk_id=ID_A, score=2.0),
            SparseHit(chunk_id=ID_B, score=1.0),
        ]
        uuids, score_map = _parse_raw_hits(raw)
        assert len(uuids) == 2
        assert score_map[ID_A] == pytest.approx(2.0)

    def test_parse_raw_hits_skips_non_uuid(self):
        from packages.retrieval.indexer import SparseHit
        raw = [
            SparseHit(chunk_id="not-a-uuid", score=1.0),
            SparseHit(chunk_id=ID_A, score=0.5),
        ]
        uuids, score_map = _parse_raw_hits(raw)
        assert len(uuids) == 1

    def test_normalise_and_trim(self):
        hits = [make_hit(score=4.0), make_hit(score=2.0), make_hit(score=1.0)]
        result = _normalise_and_trim(hits, top_k=2)
        assert len(result) == 2
        assert result[0].score == pytest.approx(1.0)   # 4/4
        assert result[1].score == pytest.approx(0.5)   # 2/4

    def test_normalise_empty(self):
        assert _normalise_and_trim([], top_k=5) == []


# ══════════════════════════════════════════════════════════════════════════════
# Hybrid fusion
# ══════════════════════════════════════════════════════════════════════════════

class TestRRFFusion:
    def _hits(self, ids_scores: list[tuple[str, float]], source="dense") -> list[DenseHit]:
        return [make_hit(chunk_id=cid, score=s, source=source) for cid, s in ids_scores]

    def test_chunk_in_both_lists_scores_higher(self):
        dense  = self._hits([(ID_A, 0.9), (ID_B, 0.7)])
        sparse = self._hits([(ID_A, 0.8), (ID_C, 0.6)], source="sparse")
        result = _fuse_rrf(dense, sparse, top_k=3, k=60)
        ids = [str(h.chunk_id) for h in result]
        assert ids[0] == ID_A   # ID_A got contributions from both

    def test_output_tagged_hybrid(self):
        dense  = self._hits([(ID_A, 0.9)])
        sparse = self._hits([(ID_B, 0.8)], source="sparse")
        result = _fuse_rrf(dense, sparse, top_k=2, k=60)
        assert all(h.source == "hybrid" for h in result)

    def test_top_k_respected(self):
        dense  = self._hits([(ID_A, 0.9), (ID_B, 0.8), (ID_C, 0.7)])
        sparse = self._hits([(ID_D, 0.6), (ID_E, 0.5)], source="sparse")
        result = _fuse_rrf(dense, sparse, top_k=2, k=60)
        assert len(result) == 2

    def test_empty_dense(self):
        sparse = self._hits([(ID_A, 0.8)], source="sparse")
        result = _fuse_rrf([], sparse, top_k=5, k=60)
        assert len(result) == 1
        assert str(result[0].chunk_id) == ID_A

    def test_empty_both(self):
        assert _fuse_rrf([], [], top_k=5, k=60) == []

    def test_scores_decrease_monotonically(self):
        dense  = self._hits([(ID_A, 0.9), (ID_B, 0.8), (ID_C, 0.7)])
        sparse = self._hits([(ID_C, 0.9), (ID_B, 0.7), (ID_D, 0.5)], source="sparse")
        result = _fuse_rrf(dense, sparse, top_k=4, k=60)
        scores = [h.score for h in result]
        assert scores == sorted(scores, reverse=True)


class TestWeightedFusion:
    def _hits(self, ids_scores, source="dense") -> list[DenseHit]:
        return [make_hit(chunk_id=cid, score=s, source=source) for cid, s in ids_scores]

    def test_weighted_combination(self):
        dense  = self._hits([(ID_A, 1.0)])
        sparse = self._hits([(ID_A, 0.0)], source="sparse")
        result = _fuse_weighted(dense, sparse, top_k=1, alpha=0.7)
        assert result[0].score == pytest.approx(0.7)

    def test_chunk_only_in_sparse(self):
        dense  = self._hits([(ID_A, 1.0)])
        sparse = self._hits([(ID_B, 0.8)], source="sparse")
        result = _fuse_weighted(dense, sparse, top_k=2, alpha=0.5)
        ids = [str(h.chunk_id) for h in result]
        assert ID_B in ids
        b_hit = next(h for h in result if str(h.chunk_id) == ID_B)
        assert b_hit.score == pytest.approx(0.5 * 0.0 + 0.5 * 0.8)

    def test_tagged_hybrid(self):
        dense  = self._hits([(ID_A, 0.9)])
        sparse = self._hits([(ID_A, 0.5)], source="sparse")
        result = _fuse_weighted(dense, sparse, top_k=1, alpha=0.7)
        assert result[0].source == "hybrid"


# ══════════════════════════════════════════════════════════════════════════════
# Reranker
# ══════════════════════════════════════════════════════════════════════════════

class TestIdentityReranker:
    def test_passthrough(self):
        hits = [make_hit(score=0.9), make_hit(score=0.7), make_hit(score=0.5)]
        result = IdentityReranker().rerank("query", hits, top_n=2)
        assert len(result) == 2
        assert result[0].score == pytest.approx(0.9)

    def test_empty(self):
        assert IdentityReranker().rerank("query", [], top_n=5) == []


class TestCrossEncoderReranker:
    """Test reranker logic with a mocked CrossEncoder to avoid loading weights."""

    def _make_reranker(self, scores: list[float]) -> CrossEncoderReranker:
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model = mock_model
        reranker._batch_size = 32
        return reranker

    def test_reranks_by_score(self):
        hits = [
            make_hit(chunk_id=ID_A, score=0.5),
            make_hit(chunk_id=ID_B, score=0.9),
            make_hit(chunk_id=ID_C, score=0.1),
        ]
        # Model gives highest score to ID_C
        reranker = self._make_reranker([0.0, -1.0, 5.0])
        result = reranker.rerank("query", hits, top_n=3)
        assert str(result[0].chunk_id) == ID_C

    def test_top_n_respected(self):
        hits = [make_hit() for _ in range(5)]
        reranker = self._make_reranker([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reranker.rerank("query", hits, top_n=2)
        assert len(result) == 2

    def test_source_tagged_reranked(self):
        hits = [make_hit(source="hybrid")]
        reranker = self._make_reranker([1.0])
        result = reranker.rerank("query", hits, top_n=1)
        assert result[0].source == "reranked"

    def test_scores_sigmoid_normalised(self):
        hits = [make_hit()]
        reranker = self._make_reranker([0.0])   # sigmoid(0) = 0.5
        result = reranker.rerank("query", hits, top_n=1)
        assert result[0].score == pytest.approx(0.5)

    def test_empty_hits(self):
        reranker = self._make_reranker([])
        assert reranker.rerank("query", [], top_n=5) == []


class TestGetReranker:
    def test_none_returns_identity(self):
        assert isinstance(get_reranker("none"), IdentityReranker)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="Unknown reranker backend"):
            get_reranker("nonexistent")
