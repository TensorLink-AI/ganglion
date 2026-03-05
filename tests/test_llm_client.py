"""Tests for the LLM client with retry logic."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ganglion.config import LLMBackendConfig
from ganglion.runtime.llm_client import LLMClient, LLMClientFactory


class TestLLMClient:
    def test_init_requires_openai(self):
        """LLMClient should be constructable when openai is available."""
        client = LLMClient(api_key="test-key", model="gpt-4o")
        assert client.model == "gpt-4o"
        assert client.max_retries == 5
        assert client.base_delay == 1.0

    def test_custom_parameters(self):
        client = LLMClient(
            api_key="test-key",
            model="gpt-3.5-turbo",
            max_retries=3,
            base_delay=0.5,
            max_delay=30.0,
            request_timeout=60.0,
        )
        assert client.model == "gpt-3.5-turbo"
        assert client.max_retries == 3
        assert client.base_delay == 0.5
        assert client.max_delay == 30.0
        assert client.request_timeout == 60.0

    def test_parse_response_basic(self):
        client = LLMClient(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = "Hello"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        result = client._parse_response(mock_response)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10

    def test_parse_response_with_tool_calls(self):
        client = LLMClient(api_key="test-key")

        mock_func = MagicMock()
        mock_func.name = "my_tool"
        mock_func.arguments = '{"x": 1}'

        mock_tc = MagicMock()
        mock_tc.id = "tc_1"
        mock_tc.function = mock_func

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=0, completion_tokens=0)

        result = client._parse_response(mock_response)
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_chat_completion_success(self):
        """Test successful chat completion call."""
        client = LLMClient(api_key="test-key", model="gpt-4o")

        mock_message = MagicMock()
        mock_message.content = "response text"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert result["content"] == "response text"
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self):
        """Test chat completion with tools parameter."""
        client = LLMClient(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=3)

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
            temperature=0.5,
        )
        assert result["role"] == "assistant"
        # Verify tools were passed
        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_completion_with_model_override(self):
        """Test chat completion with model override."""
        client = LLMClient(api_key="test-key", model="gpt-4o")

        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

        client.client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-3.5-turbo",
        )
        call_kwargs = client.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_chat_completion_retry_on_rate_limit(self):
        """Test that RateLimitError triggers retry."""
        from openai import RateLimitError

        client = LLMClient(
            api_key="test-key",
            max_retries=1,
            base_delay=0.01,
        )

        mock_message = MagicMock()
        mock_message.content = "ok"
        mock_message.tool_calls = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

        mock_rate_error = MagicMock(spec=RateLimitError)
        mock_rate_error.__class__ = RateLimitError

        # First call raises, second succeeds
        client.client.chat.completions.create = AsyncMock(
            side_effect=[
                RateLimitError("rate limited", response=MagicMock(), body=None),
                mock_response,
            ]
        )

        result = await client.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result["content"] == "ok"
        assert client.client.chat.completions.create.call_count == 2

    def test_parse_response_no_usage(self):
        """Test parsing response when usage is None."""
        client = LLMClient(api_key="test-key")

        mock_message = MagicMock()
        mock_message.content = "Hello"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = client._parse_response(mock_response)
        assert result["usage"]["prompt_tokens"] == 0
        assert result["usage"]["completion_tokens"] == 0


class TestLLMClientFactory:
    def _make_backends(self):
        return {
            "default": LLMBackendConfig(
                name="default", api_key="sk-default", model="gpt-4o"
            ),
            "fast": LLMBackendConfig(
                name="fast", api_key="sk-fast", model="gpt-4o-mini"
            ),
            "reasoning": LLMBackendConfig(
                name="reasoning",
                api_key="sk-reason",
                base_url="https://reason.api.com/v1",
                model="o1",
            ),
        }

    def test_get_default_backend(self):
        factory = LLMClientFactory(self._make_backends())
        client = factory.get()
        assert client.model == "gpt-4o"

    def test_get_named_backend(self):
        factory = LLMClientFactory(self._make_backends())
        client = factory.get("fast")
        assert client.model == "gpt-4o-mini"

    def test_get_with_model_override(self):
        factory = LLMClientFactory(self._make_backends())
        client = factory.get("fast", model_override="gpt-4o-mini-2025")
        assert client.model == "gpt-4o-mini-2025"

    def test_get_unknown_backend_raises(self):
        factory = LLMClientFactory(self._make_backends())
        with pytest.raises(ValueError, match="Unknown LLM backend 'nonexistent'"):
            factory.get("nonexistent")

    def test_caches_clients(self):
        factory = LLMClientFactory(self._make_backends())
        client1 = factory.get("fast")
        client2 = factory.get("fast")
        assert client1 is client2

    def test_different_overrides_get_different_clients(self):
        factory = LLMClientFactory(self._make_backends())
        client1 = factory.get("fast")
        client2 = factory.get("fast", model_override="other-model")
        assert client1 is not client2

    def test_list_backends_no_secrets(self):
        factory = LLMClientFactory(self._make_backends())
        backends = factory.list_backends()
        assert len(backends) == 3
        names = {b["name"] for b in backends}
        assert names == {"default", "fast", "reasoning"}
        # Ensure no api_key is exposed
        for b in backends:
            assert "api_key" not in b

    def test_has_backend(self):
        factory = LLMClientFactory(self._make_backends())
        assert factory.has_backend("fast") is True
        assert factory.has_backend("nonexistent") is False
