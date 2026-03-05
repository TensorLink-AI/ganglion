"""OpenAI-compatible LLM client with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ganglion.config import LLMBackendConfig

logger = logging.getLogger(__name__)

try:
    from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    APIError = Exception  # type: ignore[assignment,misc]
    RateLimitError = Exception  # type: ignore[assignment,misc]
    APIConnectionError = Exception  # type: ignore[assignment,misc]


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
        request_timeout: float = 120.0,
    ):
        if AsyncOpenAI is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
        )
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.request_timeout = request_timeout

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    logger.warning(
                        "Retryable error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
            except APIError as e:
                if getattr(e, "status_code", None) and e.status_code >= 500:  # type: ignore[attr-defined]
                    last_error = e  # type: ignore[assignment]
                    if attempt < self.max_retries:
                        delay = min(self.base_delay * (2**attempt), self.max_delay)
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

    def _parse_response(self, response: Any) -> dict[str, Any]:
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


class LLMClientFactory:
    """Creates and caches LLMClient instances from named backend configs.

    Claw bot selects backends by name; API keys never leave this factory.
    """

    def __init__(self, backends: dict[str, LLMBackendConfig]) -> None:
        self._backends = dict(backends)
        self._clients: dict[str, LLMClient] = {}

    def get(
        self, backend_name: str = "default", model_override: str | None = None
    ) -> LLMClient:
        """Get or create an LLMClient for the named backend."""
        cache_key = f"{backend_name}:{model_override or ''}"
        if cache_key not in self._clients:
            cfg = self._backends.get(backend_name)
            if cfg is None:
                raise ValueError(
                    f"Unknown LLM backend '{backend_name}'. "
                    f"Available: {sorted(self._backends)}"
                )
            self._clients[cache_key] = LLMClient(
                api_key=cfg.api_key,
                base_url=cfg.base_url or None,
                model=model_override or cfg.model,
                max_retries=cfg.max_retries,
                base_delay=cfg.base_delay,
                max_delay=cfg.max_delay,
                request_timeout=cfg.request_timeout,
            )
        return self._clients[cache_key]

    def list_backends(self) -> list[dict[str, str]]:
        """Return backend names and models — no secrets exposed."""
        return [
            {"name": cfg.name, "model": cfg.model, "base_url": cfg.base_url}
            for cfg in self._backends.values()
        ]

    def has_backend(self, name: str) -> bool:
        """Check if a backend name is registered."""
        return name in self._backends
