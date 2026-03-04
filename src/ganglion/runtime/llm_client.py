"""OpenAI-compatible LLM client with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment, misc]
    APIError = Exception  # type: ignore[assignment, misc]
    RateLimitError = Exception  # type: ignore[assignment, misc]
    APIConnectionError = Exception  # type: ignore[assignment, misc]


class LLMClient:
    """Thin wrapper around an OpenAI-compatible async client with retry logic."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Send a chat completion request with exponential backoff on transient errors."""
        request_kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if tools:
            request_kwargs["tools"] = tools

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.monotonic()
                response = await self.client.chat.completions.create(**request_kwargs)
                elapsed = time.monotonic() - start
                logger.debug(
                    "LLM request completed in %.2fs (attempt %d)",
                    elapsed,
                    attempt + 1,
                )
                return self._parse_response(response)
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(
                        "Retryable error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
            except APIError as e:
                if e.status_code and e.status_code >= 500:
                    last_error = e
                    if attempt < self.max_retries:
                        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                        logger.warning(
                            "Server error (attempt %d/%d): %s. Retrying in %.1fs",
                            attempt + 1,
                            self.max_retries + 1,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    def _parse_response(self, response: Any) -> dict:
        """Parse the OpenAI response into a standardized dict."""
        choice = response.choices[0]
        message = choice.message

        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]

        result["finish_reason"] = choice.finish_reason
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        return result
