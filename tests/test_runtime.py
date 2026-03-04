"""Tests for Layer 1: Runtime."""

from unittest.mock import AsyncMock, MagicMock

from ganglion.runtime.agent import SimpleAgent
from ganglion.runtime.coerce import (
    CoercionPipeline,
    coerce_empty_to_list,
    coerce_json_strings,
    coerce_string_bools,
    coerce_string_numbers,
)
from ganglion.runtime.types import AgentResult, ToolCall, ToolResult


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(success=True)
        assert r.success is True
        assert r.structured is None
        assert r.raw_text == ""
        assert r.messages == []
        assert r.turns_used == 0

    def test_with_data(self):
        r = AgentResult(
            success=False,
            structured={"error": "oops"},
            raw_text="Something failed",
            turns_used=3,
        )
        assert r.success is False
        assert r.structured["error"] == "oops"
        assert r.turns_used == 3


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(id="tc_1", name="my_tool", arguments={"x": 1})
        assert tc.id == "tc_1"
        assert tc.name == "my_tool"
        assert tc.arguments == {"x": 1}


class TestToolResult:
    def test_basic(self):
        tr = ToolResult(tool_call_id="tc_1", name="my_tool", content="result")
        assert tr.is_finish is False
        assert tr.structured is None

    def test_finish(self):
        tr = ToolResult(
            tool_call_id="tc_1",
            name="finish",
            content="done",
            is_finish=True,
            structured={"success": True},
        )
        assert tr.is_finish is True


class TestCoercionFunctions:
    def test_coerce_json_strings_dict(self):
        value, modified = coerce_json_strings("arg", '{"key": "value"}', None)
        assert modified is True
        assert value == {"key": "value"}

    def test_coerce_json_strings_list(self):
        value, modified = coerce_json_strings("arg", "[1, 2, 3]", None)
        assert modified is True
        assert value == [1, 2, 3]

    def test_coerce_json_strings_non_json(self):
        value, modified = coerce_json_strings("arg", "hello world", None)
        assert modified is False
        assert value == "hello world"

    def test_coerce_json_strings_non_string(self):
        value, modified = coerce_json_strings("arg", 42, None)
        assert modified is False
        assert value == 42

    def test_coerce_empty_to_list(self):
        value, modified = coerce_empty_to_list("arg", "", list)
        assert modified is True
        assert value == []

    def test_coerce_empty_to_list_none(self):
        value, modified = coerce_empty_to_list("arg", None, list)
        assert modified is True
        assert value == []

    def test_coerce_empty_to_list_no_match(self):
        value, modified = coerce_empty_to_list("arg", "hello", list)
        assert modified is False

    def test_coerce_string_bools_true(self):
        value, modified = coerce_string_bools("arg", "true", bool)
        assert modified is True
        assert value is True

    def test_coerce_string_bools_false(self):
        value, modified = coerce_string_bools("arg", "False", bool)
        assert modified is True
        assert value is False

    def test_coerce_string_bools_no_match(self):
        value, modified = coerce_string_bools("arg", "maybe", bool)
        assert modified is False

    def test_coerce_string_numbers_int(self):
        value, modified = coerce_string_numbers("arg", "42", int)
        assert modified is True
        assert value == 42

    def test_coerce_string_numbers_float(self):
        value, modified = coerce_string_numbers("arg", "3.14", float)
        assert modified is True
        assert abs(value - 3.14) < 0.001

    def test_coerce_json_strings_invalid_json(self):
        value, modified = coerce_json_strings("arg", "{bad json", None)
        assert modified is False
        assert value == "{bad json"

    def test_coerce_string_numbers_invalid_int(self):
        value, modified = coerce_string_numbers("arg", "not_a_number", int)
        assert modified is False
        assert value == "not_a_number"

    def test_coerce_string_numbers_invalid_float(self):
        value, modified = coerce_string_numbers("arg", "not_a_float", float)
        assert modified is False
        assert value == "not_a_float"


class TestCoercionPipeline:
    def test_default_pipeline(self):
        pipeline = CoercionPipeline()
        result = pipeline.apply(
            {"data": '{"x": 1}', "flag": "true", "items": ""},
            {"data": dict, "flag": bool, "items": list},
        )
        assert result["data"] == {"x": 1}
        assert result["flag"] is True
        assert result["items"] == []

    def test_custom_pipeline(self):
        pipeline = CoercionPipeline([coerce_json_strings])
        result = pipeline.apply({"data": '{"x": 1}', "flag": "maybe"}, {"flag": bool})
        assert result["data"] == {"x": 1}
        # String bools NOT coerced because we only have json coercion
        assert result["flag"] == "maybe"

    def test_no_modification(self):
        pipeline = CoercionPipeline()
        result = pipeline.apply({"x": 42, "y": "hello"})
        assert result == {"x": 42, "y": "hello"}


def _make_llm_client(**kwargs):
    """Create a mock LLMClient."""
    client = MagicMock()
    client.chat_completion = AsyncMock(**kwargs)
    return client


class TestSimpleAgent:
    async def test_no_tool_calls_returns_early(self):
        llm = _make_llm_client(return_value={"content": "Hello!", "finish_reason": "stop"})
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="You are a test agent.",
            tools_schema=[],
            tool_handlers={},
        )
        result = await agent.run()
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert result.raw_text == "Hello!"
        assert result.turns_used == 1

    async def test_tool_call_then_finish(self):
        tool_response = {
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "finish",
                        "arguments": '{"success": true, "summary": "done"}',
                    },
                }
            ],
        }
        llm = _make_llm_client(return_value=tool_response)
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[{"type": "function", "function": {"name": "finish"}}],
            tool_handlers={"finish": lambda **kw: kw},
        )
        result = await agent.run()
        assert result.success is True
        assert result.turns_used == 1

    async def test_regular_tool_then_finish(self):
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "my_tool", "arguments": '{"x": 42}'},
                        }
                    ],
                }
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_2",
                        "function": {
                            "name": "finish",
                            "arguments": '{"success": true, "result": 42}',
                        },
                    }
                ],
            }

        llm = MagicMock()
        llm.chat_completion = mock_completion

        def my_tool(x=0):
            return f"result: {x}"

        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={"my_tool": my_tool, "finish": lambda **kw: kw},
        )
        result = await agent.run()
        assert result.success is True
        assert result.turns_used == 2

    async def test_max_turns_reached(self):
        tool_response = {
            "content": "",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {"name": "my_tool", "arguments": "{}"},
                }
            ],
        }
        llm = _make_llm_client(return_value=tool_response)
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={"my_tool": lambda: "ok"},
            max_turns=2,
        )
        result = await agent.run()
        assert result.success is False
        assert result.turns_used == 2
        assert "Max turns" in result.raw_text

    async def test_unknown_tool(self):
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "unknown_tool", "arguments": "{}"},
                        }
                    ],
                }
            return {"content": "gave up", "finish_reason": "stop"}

        llm = MagicMock()
        llm.chat_completion = mock_completion
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={"finish": lambda **kw: kw},
            max_turns=3,
        )
        result = await agent.run()
        assert result.turns_used == 2

    async def test_tool_exception(self):
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "bad_tool", "arguments": "{}"},
                        }
                    ],
                }
            return {"content": "recovered", "finish_reason": "stop"}

        def bad_tool():
            raise ValueError("tool exploded")

        llm = MagicMock()
        llm.chat_completion = mock_completion
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={"bad_tool": bad_tool, "finish": lambda **kw: kw},
            max_turns=3,
        )
        result = await agent.run()
        assert result.turns_used == 2

    async def test_invalid_json_arguments(self):
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "my_tool", "arguments": "not valid json"},
                        }
                    ],
                }
            return {"content": "done", "finish_reason": "stop"}

        llm = MagicMock()
        llm.chat_completion = mock_completion
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={"my_tool": lambda: "ok", "finish": lambda **kw: kw},
            max_turns=3,
        )
        result = await agent.run()
        assert result.turns_used == 2

    async def test_context_messages(self):
        llm = _make_llm_client(return_value={"content": "Hello!", "finish_reason": "stop"})
        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={},
            context_messages=[{"role": "user", "content": "context msg"}],
        )
        assert len(agent.messages) == 2
        assert agent.messages[1]["content"] == "context msg"

    async def test_tool_returning_tool_output(self):
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "my_tool", "arguments": "{}"},
                        }
                    ],
                }
            return {"content": "done", "finish_reason": "stop"}

        llm = MagicMock()
        llm.chat_completion = mock_completion

        class FakeOutput:
            content = "tool output"
            structured = {"key": "val"}

        agent = SimpleAgent(
            llm_client=llm,
            system_prompt="Test",
            tools_schema=[],
            tool_handlers={"my_tool": lambda: FakeOutput(), "finish": lambda **kw: kw},
            max_turns=3,
        )
        result = await agent.run()
        assert result.turns_used == 2
