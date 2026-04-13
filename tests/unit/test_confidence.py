"""Unit tests for confidence scoring — no DB, no LLM required."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import pytest

from apps.api.app.services.confidence import (
    ConfidenceScorer,
    _citation_density,
    _content_terms,
    _coverage_ratio,
    _retrieval_score,
    _to_risk_level,
    _top_chunk_score,
)
from apps.api.app.services.answer_service import GeneratedAnswer, InlineCitation
from packages.retrieval.dense import DenseHit
from packages.schemas.common import RiskLevel


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_answer(
    text: str = "The device is intended for cardiac monitoring [1].",
    refused: bool = False,
    refusal_reason: str | None = None,
    citations: list | None = None,
) -> GeneratedAnswer:
    return GeneratedAnswer(
        query="What is the intended use?",
        answer_text=text,
        citations=citations or [],
        refused=refused,
        refusal_reason=refusal_reason,
        model_name="gpt-4o",
        prompt_version="v1",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
    )


def make_hit(score: float = 0.9, text: str = "cardiac monitoring intended use adult patients") -> DenseHit:
    return DenseHit(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        score=score,
        text=text,
        section_title="1 Intended Use",
        heading_path=[],
        page_start=1,
        page_end=1,
        document_title="CER",
        source="reranked",
    )


# ── _top_chunk_score ──────────────────────────────────────────────────────────

class TestTopChunkScore:
    def test_returns_max(self):
        hits = [make_hit(0.9), make_hit(0.7), make_hit(0.5)]
        assert _top_chunk_score(hits) == pytest.approx(0.9)

    def test_single_hit(self):
        assert _top_chunk_score([make_hit(0.6)]) == pytest.approx(0.6)

    def test_empty(self):
        assert _top_chunk_score([]) == 0.0


# ── _retrieval_score ──────────────────────────────────────────────────────────

class TestRetrievalScore:
    def test_mean(self):
        hits = [make_hit(1.0), make_hit(0.0)]
        assert _retrieval_score(hits) == pytest.approx(0.5)

    def test_empty(self):
        assert _retrieval_score([]) == 0.0


# ── _citation_density ─────────────────────────────────────────────────────────

class TestCitationDensity:
    def test_all_cited(self):
        text = "Cardiac monitoring is indicated [1]. Pacemakers are contraindicated [2]."
        assert _citation_density(text) == pytest.approx(1.0)

    def test_none_cited(self):
        text = "The device is intended for adults. It should not be used with pacemakers."
        assert _citation_density(text) == pytest.approx(0.0)

    def test_partial(self):
        text = "Cardiac monitoring is indicated [1]. No further information was provided."
        density = _citation_density(text)
        assert 0.0 < density < 1.0

    def test_empty_text(self):
        assert _citation_density("") == 0.0

    def test_multiple_citations_in_one_sentence_counts_once(self):
        text = "Approved per ISO 14971 and IEC 62304 [1][2]. No further information."
        assert _citation_density(text) == pytest.approx(0.5)


# ── _coverage_ratio ───────────────────────────────────────────────────────────

class TestCoverageRatio:
    def test_perfect_overlap(self):
        hit = make_hit(text="cardiac monitoring intended adults")
        answer = make_answer(text="cardiac monitoring intended adults")
        ratio = _coverage_ratio(answer.answer_text, [hit])
        assert ratio > 0.5

    def test_zero_overlap(self):
        hit = make_hit(text="completely unrelated text about software architecture")
        answer = make_answer(text="biocompatibility cytotoxicity sensitisation testing")
        ratio = _coverage_ratio(answer.answer_text, [hit])
        assert ratio == pytest.approx(0.0)

    def test_empty_hits(self):
        answer = make_answer()
        assert _coverage_ratio(answer.answer_text, []) == 0.0


# ── _to_risk_level ────────────────────────────────────────────────────────────

class TestToRiskLevel:
    def test_low(self):
        assert _to_risk_level(0.80) == RiskLevel.LOW
        assert _to_risk_level(0.75) == RiskLevel.LOW

    def test_medium(self):
        assert _to_risk_level(0.60) == RiskLevel.MEDIUM
        assert _to_risk_level(0.50) == RiskLevel.MEDIUM

    def test_high(self):
        assert _to_risk_level(0.40) == RiskLevel.HIGH
        assert _to_risk_level(0.30) == RiskLevel.HIGH

    def test_critical(self):
        assert _to_risk_level(0.29) == RiskLevel.CRITICAL
        assert _to_risk_level(0.0) == RiskLevel.CRITICAL


# ── ConfidenceScorer ──────────────────────────────────────────────────────────

class TestConfidenceScorer:
    scorer = ConfidenceScorer()

    def test_refusal_gives_zero_and_critical(self):
        answer = make_answer(refused=True, refusal_reason="No relevant information found.")
        hits = [make_hit(0.9)]
        bd = self.scorer.score(answer, hits)
        assert bd.overall == 0.0
        assert bd.risk_level == RiskLevel.CRITICAL

    def test_well_cited_high_score_answer(self):
        answer = make_answer(
            text=(
                "The device is intended for cardiac monitoring in adult patients [1]. "
                "Risk management was performed per ISO 14971 [2]."
            )
        )
        hits = [make_hit(0.95, "cardiac monitoring intended adult patients"), make_hit(0.88, "risk management ISO 14971")]
        bd = self.scorer.score(answer, hits)
        assert bd.overall > 0.5
        assert bd.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_uncited_answer_penalised(self):
        well_cited = make_answer("Device for cardiac monitoring [1]. Used in adults [2].")
        uncited = make_answer("Device for cardiac monitoring. Used in adults.")
        hits = [make_hit(0.9)]
        bd_cited = self.scorer.score(well_cited, hits)
        bd_uncited = self.scorer.score(uncited, hits)
        assert bd_cited.overall > bd_uncited.overall

    def test_low_retrieval_score_lowers_confidence(self):
        answer = make_answer("Device for cardiac monitoring [1].")
        strong_hits = [make_hit(0.95)]
        weak_hits = [make_hit(0.10)]
        bd_strong = self.scorer.score(answer, strong_hits)
        bd_weak = self.scorer.score(answer, weak_hits)
        assert bd_strong.overall > bd_weak.overall

    def test_overall_in_unit_interval(self):
        answer = make_answer()
        hits = [make_hit(0.7)]
        bd = self.scorer.score(answer, hits)
        assert 0.0 <= bd.overall <= 1.0

    def test_breakdown_fields_populated(self):
        answer = make_answer("Cardiac monitoring [1].")
        hits = [make_hit(0.85, "cardiac monitoring adult patients")]
        bd = self.scorer.score(answer, hits)
        assert bd.top_chunk_score == pytest.approx(0.85)
        assert bd.retrieval_score == pytest.approx(0.85)
        assert bd.citation_density > 0.0
        assert bd.risk_level is not None

    def test_empty_hits_gives_low_confidence(self):
        answer = make_answer("Device for cardiac monitoring [1].")
        bd = self.scorer.score(answer, [])
        assert bd.overall < 0.5
