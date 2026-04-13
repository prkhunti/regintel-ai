"""
Grounded answer generation with structured JSON output.

Flow
----
1. Number the retrieved chunks: [1], [2], … [N].
2. Build a prompt that instructs the model to return StructuredAnswerOutput JSON.
3. Call LLM via complete_structured() — OpenAI JSON mode or Anthropic tool use.
4. Pydantic validates the response; no regex needed.
5. Map CitationOutput.chunk_index back to the source DenseHit.
6. Return a GeneratedAnswer ready to be scored, stored, and returned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import uuid

from packages.retrieval.dense import DenseHit
from packages.schemas.llm_output import CitationOutput, StructuredAnswerOutput
from .llm_client import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Walk up until we find the directory that contains packages/ (works in both
# Docker and local-monorepo contexts without hardcoding depth).
_PACKAGES_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "packages").exists()
)
_TEMPLATE_PATH = _PACKAGES_ROOT / "packages" / "prompts" / "templates" / "answer_structured.txt"
_PROMPT_VERSION = "answer_structured_v1"


# ── Output dataclasses ────────────────────────────────────────────────────────

@dataclass
class InlineCitation:
    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str | None
    section_title: str | None
    quote: str
    page_start: int | None
    page_end: int | None
    relevance_score: float


@dataclass
class GeneratedAnswer:
    query: str
    answer_text: str
    citations: list[InlineCitation]
    refused: bool
    refusal_reason: str | None
    model_name: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


# ── Answer generator ──────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates a grounded, structured answer with validated citations.

    Uses ``complete_structured()`` so the output is a Pydantic model —
    no regex, no string parsing.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_context_chunks: int = 10,
        max_chunk_chars: int = 800,
    ) -> None:
        self._llm = llm_client
        self._max_chunks = max_context_chunks
        self._max_chars = max_chunk_chars
        self._template = _TEMPLATE_PATH.read_text()

    async def generate(
        self,
        query: str,
        hits: list[DenseHit],
        max_tokens: int = 1024,
    ) -> GeneratedAnswer:
        chunks = hits[: self._max_chunks]
        context_block = _build_context(chunks, self._max_chars)
        prompt = self._template.format(context=context_block, question=query)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical document intelligence assistant for regulated "
                    "medical device environments. Answer only from the provided context. "
                    "Return your response as a JSON object matching the required schema."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        structured: StructuredAnswerOutput
        llm_resp: LLMResponse
        structured, llm_resp = await self._llm.complete_structured(
            messages, StructuredAnswerOutput, temperature=0.0, max_tokens=max_tokens
        )

        citations = _map_citations(structured.citations, chunks)

        return GeneratedAnswer(
            query=query,
            answer_text=structured.answer,
            citations=citations,
            refused=structured.insufficient_context,
            refusal_reason=structured.refusal_reason,
            model_name=llm_resp.model,
            prompt_version=_PROMPT_VERSION,
            prompt_tokens=llm_resp.prompt_tokens,
            completion_tokens=llm_resp.completion_tokens,
            latency_ms=llm_resp.latency_ms,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context(chunks: list[DenseHit], max_chars: int) -> str:
    lines = []
    for i, hit in enumerate(chunks, start=1):
        text = hit.text
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " …"
        header = f"[{i}]"
        if hit.document_title:
            header += f" {hit.document_title}"
        if hit.section_title:
            header += f" › {hit.section_title}"
        if hit.page_start:
            header += f" (p.{hit.page_start})"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def _map_citations(
    raw: list[CitationOutput],
    chunks: list[DenseHit],
) -> list[InlineCitation]:
    citations = []
    for c in raw:
        idx = c.chunk_index - 1   # 1-based → 0-based
        if idx < 0 or idx >= len(chunks):
            logger.warning("Citation chunk_index %d out of range (have %d)", c.chunk_index, len(chunks))
            continue
        hit = chunks[idx]
        citations.append(InlineCitation(
            index=c.chunk_index,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            document_title=hit.document_title,
            section_title=hit.section_title,
            quote=c.quote,
            page_start=hit.page_start,
            page_end=hit.page_end,
            relevance_score=hit.score,
        ))
    return citations
