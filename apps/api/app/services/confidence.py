"""
Confidence scorer for generated answers.

Four signals, weighted into a single [0, 1] score:

  top_chunk_score   (0.30) — quality of the single best retrieved chunk.
                             Answers grounded in a near-perfect match get
                             credit here.

  citation_density  (0.35) — fraction of answer sentences that carry at
                             least one [N] citation.  Uncited sentences
                             are a hallucination risk.

  retrieval_score   (0.25) — mean score of the top-k chunks.  A strong
                             average indicates the query is well-covered
                             by the corpus.

  coverage_ratio    (0.10) — fraction of content terms in the answer that
                             also appear in the source chunks.  Low overlap
                             suggests the model introduced out-of-context
                             content.

A refusal (INSUFFICIENT_CONTEXT) short-circuits to confidence=0.0 and
risk_level=CRITICAL.

Risk levels
-----------
  >= 0.75   LOW
  >= 0.50   MEDIUM
  >= 0.30   HIGH
  <  0.30   CRITICAL
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

from packages.retrieval.dense import DenseHit
from packages.schemas.common import RiskLevel
from .answer_service import GeneratedAnswer

# ── Constants ─────────────────────────────────────────────────────────────────

_W_TOP_CHUNK   = 0.30
_W_CITATION    = 0.35
_W_RETRIEVAL   = 0.25
_W_COVERAGE    = 0.10

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION_IN_SENT = re.compile(r"\[\d+\]")
_STOPWORDS = frozenset(
    "a an the and or but in on at to for of with by from is are was were be "
    "been being have has had do does did will would could should may might "
    "shall this that these those it its i we you he she they".split()
)
_PUNCT = re.compile(r"[" + re.escape(string.punctuation) + r"]")


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ConfidenceBreakdown:
    """Granular breakdown of each contributing signal."""
    top_chunk_score: float
    citation_density: float
    retrieval_score: float
    coverage_ratio: float
    overall: float
    risk_level: RiskLevel


# ── Scorer ────────────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Scores a ``GeneratedAnswer`` against its supporting ``DenseHit`` list.

    Usage::

        scorer = ConfidenceScorer()
        breakdown = scorer.score(answer, hits)
        print(breakdown.overall, breakdown.risk_level)
    """

    def score(
        self,
        answer: GeneratedAnswer,
        hits: list[DenseHit],
    ) -> ConfidenceBreakdown:
        """
        Compute a confidence score for *answer* given *hits*.

        Returns:
            ``ConfidenceBreakdown`` with individual signals and the weighted
            overall score.
        """
        # Refusal → zero confidence regardless of other signals
        if answer.refused:
            return ConfidenceBreakdown(
                top_chunk_score=0.0,
                citation_density=0.0,
                retrieval_score=0.0,
                coverage_ratio=0.0,
                overall=0.0,
                risk_level=RiskLevel.CRITICAL,
            )

        top   = _top_chunk_score(hits)
        cit   = _citation_density(answer.answer_text)
        ret   = _retrieval_score(hits)
        cov   = _coverage_ratio(answer.answer_text, hits)

        overall = (
            _W_TOP_CHUNK  * top
            + _W_CITATION * cit
            + _W_RETRIEVAL * ret
            + _W_COVERAGE  * cov
        )
        overall = max(0.0, min(1.0, overall))

        return ConfidenceBreakdown(
            top_chunk_score=round(top, 4),
            citation_density=round(cit, 4),
            retrieval_score=round(ret, 4),
            coverage_ratio=round(cov, 4),
            overall=round(overall, 4),
            risk_level=_to_risk_level(overall),
        )


# ── Signal implementations ────────────────────────────────────────────────────

def _top_chunk_score(hits: list[DenseHit]) -> float:
    """Max retriever score across all hits; already in [0, 1]."""
    if not hits:
        return 0.0
    return max(h.score for h in hits)


def _retrieval_score(hits: list[DenseHit]) -> float:
    """Mean retriever score across top hits."""
    if not hits:
        return 0.0
    return sum(h.score for h in hits) / len(hits)


def _citation_density(answer_text: str) -> float:
    """
    Fraction of non-empty answer sentences that contain at least one [N] marker.

    Examples::
        "The device is for adults [1]. It must not be used with pacemakers [2]."
        → 2 cited / 2 sentences = 1.0

        "The device is intended for adults [1]. No other information available."
        → 1 cited / 2 sentences = 0.5
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer_text) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITATION_IN_SENT.search(s))
    return cited / len(sentences)


def _coverage_ratio(answer_text: str, hits: list[DenseHit]) -> float:
    """
    Fraction of content terms in the answer that appear in at least one chunk.

    Content terms = non-stopword tokens of length > 3.
    """
    answer_terms = _content_terms(answer_text)
    if not answer_terms:
        return 0.0

    source_terms: set[str] = set()
    for hit in hits:
        source_terms.update(_content_terms(hit.text))

    covered = answer_terms & source_terms
    return len(covered) / len(answer_terms)


def _content_terms(text: str) -> set[str]:
    text = _PUNCT.sub(" ", text.lower())
    return {t for t in text.split() if len(t) > 3 and t not in _STOPWORDS}


def _to_risk_level(score: float) -> RiskLevel:
    if score >= 0.75:
        return RiskLevel.LOW
    if score >= 0.50:
        return RiskLevel.MEDIUM
    if score >= 0.30:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL
