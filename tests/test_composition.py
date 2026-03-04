"""Tests for Layer 2: Composition."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.composition.prompt import PromptBuilder
from ganglion.composition.tool_registry import (
    _infer_schema,
    build_toolset,
    clear_global_registry,
    get_all_tools,
    tool,
)
from ganglion.composition.tool_returns import ExperimentResult, ToolOutput, ValidationResult
from ganglion.runtime.types import AgentResult


class TestToolReturns:
    def test_tool_output(self):
        out = ToolOutput(content="result", structured={"key": "value"})
        assert out.content == "result"
        assert out.structured == {"key": "value"}

    def test_experiment_result(self):
        out = ExperimentResult(
            content="done",
            experiment_id="exp_1",
            metrics={"accuracy": 0.95},
            artifact_path="/models/best.pt",
        )
        assert out.experiment_id == "exp_1"
        assert out.metrics["accuracy"] == 0.95
        assert isinstance(out, ToolOutput)

    def test_validation_result(self):
        out = ValidationResult(
            content="checked",
            is_passed=False,
            errors=["shape mismatch"],
            warnings=["high loss"],
        )
        assert out.is_passed is False
        assert len(out.errors) == 1


class TestToolDecorator:
    def setup_method(self):
        clear_global_registry()

    def test_basic_registration(self):
        @tool("test_tool")
        def test_tool(x: int, y: str = "default") -> str:
            """A test tool."""
            return f"{x}:{y}"

        tools = get_all_tools()
        assert "test_tool" in tools
        td = tools["test_tool"]
        assert td.name == "test_tool"
        assert td.description == "A test tool."
        assert "x" in td.parameters_schema["properties"]

    def test_schema_inference(self):
        def my_func(name: str, count: int = 5, active: bool = True) -> dict:
            """Do something."""
            pass

        schema = _infer_schema(my_func)
        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["active"]["type"] == "boolean"
        assert schema["required"] == ["name"]

    def test_build_toolset(self):
        @tool("tool_a")
        def tool_a(x: int) -> str:
            """Tool A."""
            return str(x)

        @tool("tool_b")
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        schemas, handlers = build_toolset("tool_a", "tool_b")
        # Should include tool_a, tool_b, and finish
        assert len(schemas) == 3
        assert "tool_a" in handlers
        assert "tool_b" in handlers
        assert "finish" in handlers

        names = [s["function"]["name"] for s in schemas]
        assert "tool_a" in names
        assert "tool_b" in names
        assert "finish" in names

    def test_build_toolset_missing_tool(self):
        schemas, handlers = build_toolset("nonexistent")
        # Should still have finish
        assert len(schemas) == 1
        assert "finish" in handlers


class TestPromptBuilder:
    def test_basic_build(self):
        prompt = (
            PromptBuilder()
            .section("role", "You are a test agent.")
            .section("context", "Testing context.")
            .build()
        )
        assert "## role" in prompt
        assert "You are a test agent." in prompt
        assert "## context" in prompt

    def test_empty_sections_skipped(self):
        prompt = PromptBuilder().section("role", "Agent").section("empty", "   ").build()
        assert "## empty" not in prompt

    def test_remove_section(self):
        builder = (
            PromptBuilder().section("role", "Agent").section("extra", "Extra stuff").remove("extra")
        )
        assert not builder.has_section("extra")
        assert builder.has_section("role")

    def test_replace_section(self):
        prompt = PromptBuilder().section("role", "Original").replace("role", "Replaced").build()
        assert "Replaced" in prompt
        assert "Original" not in prompt

    def test_section_names(self):
        builder = PromptBuilder().section("a", "A").section("b", "B").section("c", "C")
        assert builder.section_names() == ["a", "b", "c"]


class TestBaseAgentWrapper:
    def test_build_system_prompt_not_implemented(self):
        agent = BaseAgentWrapper()
        with pytest.raises(NotImplementedError):
            agent.build_system_prompt({})

    def test_build_tools_not_implemented(self):
        agent = BaseAgentWrapper()
        with pytest.raises(NotImplementedError):
            agent.build_tools({})

    def test_build_context_default(self):
        agent = BaseAgentWrapper()
        assert agent.build_context({}) == []

    def test_post_process_default(self):
        agent = BaseAgentWrapper()
        result = AgentResult(success=True, raw_text="done")
        assert agent.post_process(result, {}) is result

    def test_describe(self):
        agent = BaseAgentWrapper(max_turns=10, temperature=0.5, model="gpt-4")
        desc = agent.describe()
        assert desc["class"] == "BaseAgentWrapper"
        assert desc["max_turns"] == 10
        assert desc["temperature"] == 0.5
        assert desc["model"] == "gpt-4"

    async def test_run_without_llm_client_raises(self):
        agent = BaseAgentWrapper()
        agent.build_system_prompt = MagicMock(return_value="prompt")
        agent.build_tools = MagicMock(return_value=([], {}))
        with pytest.raises(RuntimeError, match="llm_client"):
            await agent.run({})

    async def test_run_with_llm_client(self):
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(
            return_value={"content": "done", "finish_reason": "stop"}
        )
        agent = BaseAgentWrapper(llm_client=mock_llm, extra_system_context="extra info")
        agent.build_system_prompt = MagicMock(return_value="You are a test agent.")
        agent.build_tools = MagicMock(return_value=([], {}))
        agent.build_context = MagicMock(return_value=[{"role": "user", "content": "hi"}])
        result = await agent.run({"task": "test"})
        assert isinstance(result, AgentResult)
        agent.build_system_prompt.assert_called_once_with({"task": "test"})
        agent.build_tools.assert_called_once_with({"task": "test"})
