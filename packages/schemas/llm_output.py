"""
Pydantic schemas for structured LLM output.

These are used as the JSON contract between the LLM and the application —
not the API contract with the client (that lives in response.py).

The model is asked to return a JSON object matching StructuredAnswerOutput.
Having a schema here lets us validate, type-check, and test the LLM output
without relying on regex parsing of free text.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CitationOutput(BaseModel):
    """A single citation produced by the LLM."""
    chunk_index: int = Field(
        description="1-based index of the context block this citation refers to.",
        ge=1,
    )
    quote: str = Field(
        description="Verbatim or near-verbatim sentence from the referenced chunk.",
        min_length=5,
    )

    @field_validator("quote")
    @classmethod
    def strip_quote(cls, v: str) -> str:
        return v.strip()


class StructuredAnswerOutput(BaseModel):
    """
    Full structured output returned by the LLM.

    The prompt asks the model to populate this object directly.
    The application validates it with Pydantic before using it.
    """
    answer: str = Field(
        description=(
            "Concise, professional answer to the question. "
            "Must be grounded in the context; do not add outside knowledge. "
            "Empty string when insufficient_context is true."
        ),
    )
    citations: list[CitationOutput] = Field(
        default_factory=list,
        description="All context blocks cited in the answer, deduplicated.",
    )
    insufficient_context: bool = Field(
        description="True when the provided context does not contain enough information to answer.",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="One sentence explaining what information is missing. Null when insufficient_context is false.",
    )

    @field_validator("citations")
    @classmethod
    def deduplicate_citations(cls, v: list[CitationOutput]) -> list[CitationOutput]:
        seen: set[int] = set()
        deduped = []
        for c in v:
            if c.chunk_index not in seen:
                seen.add(c.chunk_index)
                deduped.append(c)
        return sorted(deduped, key=lambda c: c.chunk_index)
