"""Tests for the LLM client with retry logic."""

from unittest.mock import MagicMock

from ganglion.runtime.llm_client import LLMClient


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
