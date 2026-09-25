"""Thin async client for OpenRouter (OpenAI-compatible chat completions).

Calls Qwen 3 8B (< 10 B params) via OpenRouter.  Uses httpx2 which is already
a project dependency, so no extra packages are needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """All fields come straight from the .env file."""

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen3-8b"
    temperature: float = 0.0
    max_tokens: int = 2048


class LLMClient:
    """Async wrapper around the OpenRouter chat-completions endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._http: httpx2.AsyncClient | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def _client(self) -> httpx2.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx2.AsyncClient(
                base_url=self._config.base_url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx2.Timeout(120.0, connect=30.0),
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # core call
    # ------------------------------------------------------------------
    async def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> str:
        """Send a chat completion request and return the assistant text.

        Retries on transient HTTP errors with exponential back-off.
        """
        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                client = await self._client()
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices", [])
                if not choices:
                    raise ValueError("OpenRouter returned no choices")
                content = choices[0].get("message", {}).get("content", "")
                # Qwen 3 with thinking enabled wraps reasoning in <think>…</think> tags.
                # Strip those so callers receive only the final answer.
                content = _strip_think_tags(content)
                logger.debug(
                    "LLM response (attempt %d): model=%s tokens=%s",
                    attempt + 1,
                    body.get("model"),
                    body.get("usage"),
                )
                return content.strip()
            except (httpx2.HTTPStatusError, httpx2.ConnectError, httpx2.ReadTimeout) as exc:
                last_error = exc
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, exc)
                if attempt + 1 < retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call chat() and parse the response as JSON.

        If the model wraps the JSON in a markdown code fence, strip it first.
        """
        raw = await self.chat(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cleaned = _extract_json(raw)
        return json.loads(cleaned)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks that Qwen 3 emits when thinking is on."""
    import re

    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Extract JSON from possible markdown code fences."""
    import re

    # Try ```json ... ``` first
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to the whole text
    return text.strip()

