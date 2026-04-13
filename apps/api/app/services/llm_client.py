"""
Provider-abstracted async LLM client.

Supports OpenAI (gpt-4o, gpt-4o-mini, …) and Anthropic (claude-*) behind a
common interface with two completion modes:

  complete()            — free-text response (legacy / simple use-cases)
  complete_structured() — returns a validated Pydantic model via JSON mode
                          (OpenAI) or tool use (Anthropic)

Usage
-----
    client = get_llm_client()
    text   = await client.complete(messages)
    output = await client.complete_structured(messages, StructuredAnswerOutput)
"""
from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send messages and return the assistant response as plain text."""

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[T, LLMResponse]:
        """
        Send messages and parse the response into *schema*.

        Returns (parsed_model, raw_llm_response).
        Raises ``ValueError`` if the model output cannot be parsed.
        """


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError("openai package required: pip install openai") from e
        self._client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._model = model

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )

    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[T, LLMResponse]:
        """Use OpenAI JSON mode + Pydantic validation."""
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        choice = response.choices[0]
        raw_text = choice.message.content or "{}"

        llm_resp = LLMResponse(
            text=raw_text,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )
        try:
            parsed = schema.model_validate_json(raw_text)
        except Exception as exc:
            raise ValueError(f"OpenAI returned invalid JSON for schema {schema.__name__}: {exc}\n{raw_text}") from exc
        return parsed, llm_resp


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic package required: pip install anthropic") from e
        self._client = anthropic.AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._model = model

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        return (await self._call(messages, temperature, max_tokens))[1]

    async def complete_structured(
        self,
        messages: list[dict],
        schema: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> tuple[T, LLMResponse]:
        """Use Anthropic tool use to get a structured JSON response."""
        tool = {
            "name": "structured_answer",
            "description": "Return the answer in the required JSON format.",
            "input_schema": schema.model_json_schema(),
        }
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]

        t0 = time.perf_counter()
        kwargs: dict = dict(
            model=self._model,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": "structured_answer"},
        )
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Extract the tool_use block
        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            raise ValueError("Anthropic did not return a tool_use block for structured output")

        raw_input = tool_block.input  # already a dict
        llm_resp = LLMResponse(
            text=json.dumps(raw_input),
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
        try:
            parsed = schema.model_validate(raw_input)
        except Exception as exc:
            raise ValueError(f"Anthropic tool output invalid for {schema.__name__}: {exc}") from exc
        return parsed, llm_resp

    async def _call(self, messages, temperature, max_tokens) -> tuple[str, LLMResponse]:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_messages = [m for m in messages if m["role"] != "system"]
        t0 = time.perf_counter()
        kwargs: dict = dict(model=self._model, messages=user_messages,
                            temperature=temperature, max_tokens=max_tokens)
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = response.content[0].text if response.content else ""
        return text, LLMResponse(
            text=text, model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> BaseLLMClient:
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL", "gpt-4o")

    if provider == "openai":
        return OpenAIClient(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'openai' or 'anthropic'.")
