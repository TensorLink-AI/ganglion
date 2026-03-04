"""Tests for the State/Mutability layer."""

import pytest

from ganglion.orchestration.errors import (
    AgentNotFoundError,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
)
from ganglion.state.agent_registry import AgentRegistry
from ganglion.state.mutation import Mutation, MutationResult
from ganglion.state.tool_registry import ToolRegistry
from ganglion.state.validator import MutationValidator


class TestMutationValidator:
    def setup_method(self):
        self.validator = MutationValidator()

    def test_valid_tool(self):
        code = '''
from ganglion.composition.tool_registry import tool

@tool("my_tool")
def my_tool(x: int, y: str = "hi") -> str:
    """A valid tool."""
    return f"{x}:{y}"
'''
        result = self.validator.validate_tool(code)
        assert result.is_passed is True
        assert result.errors == []

    def test_tool_missing_decorator(self):
        code = '''
def my_tool(x: int) -> str:
    """No decorator."""
    return str(x)
'''
        result = self.validator.validate_tool(code)
        assert result.is_passed is False
        assert any("@tool" in e for e in result.errors)

    def test_tool_missing_type_hint(self):
        code = '''
from ganglion.composition.tool_registry import tool

@tool("my_tool")
def my_tool(x, y: str = "hi") -> str:
    """Missing type hint on x."""
    return str(x)
'''
        result = self.validator.validate_tool(code)
        assert result.is_passed is False
        assert any("type hint" in e for e in result.errors)

    def test_tool_missing_docstring(self):
        code = '''
from ganglion.composition.tool_registry import tool

@tool("my_tool")
def my_tool(x: int) -> str:
    return str(x)
'''
        result = self.validator.validate_tool(code)
        assert result.is_passed is False
        assert any("docstring" in e for e in result.errors)

    def test_tool_blocked_import(self):
        code = '''
import subprocess
from ganglion.composition.tool_registry import tool

@tool("my_tool")
def my_tool(x: int) -> str:
    """Uses subprocess."""
    return str(x)
'''
        result = self.validator.validate_tool(code)
        assert result.is_passed is False
        assert any("Blocked" in e for e in result.errors)

    def test_tool_syntax_error(self):
        code = "def broken(:"
        result = self.validator.validate_tool(code)
        assert result.is_passed is False
        assert any("Syntax" in e for e in result.errors)

    def test_valid_agent(self):
        code = '''
from ganglion.composition.base_agent import BaseAgentWrapper

class MyAgent(BaseAgentWrapper):
    def build_system_prompt(self, task):
        return "prompt"

    def build_tools(self, task):
        return [], {}
'''
        result = self.validator.validate_agent(code)
        assert result.is_passed is True

    def test_agent_missing_class(self):
        code = '''
def not_an_agent():
    pass
'''
        result = self.validator.validate_agent(code)
        assert result.is_passed is False
        assert any("BaseAgentWrapper" in e for e in result.errors)

    def test_agent_missing_methods(self):
        code = '''
from ganglion.composition.base_agent import BaseAgentWrapper

class MyAgent(BaseAgentWrapper):
    pass
'''
        result = self.validator.validate_agent(code)
        assert result.is_passed is False
        assert any("build_system_prompt" in e for e in result.errors)
        assert any("build_tools" in e for e in result.errors)


class TestToolRegistry:
    def test_register_and_has(self):
        reg = ToolRegistry()
        reg.register("my_tool", lambda: None, "desc", {"type": "object"})
        assert reg.has("my_tool") is True
        assert reg.has("other") is False

    def test_duplicate_register(self):
        reg = ToolRegistry()
        reg.register("my_tool", lambda: None, "desc", {"type": "object"})
        with pytest.raises(ToolAlreadyRegisteredError):
            reg.register("my_tool", lambda: None, "desc2", {"type": "object"})

    def test_unregister(self):
        reg = ToolRegistry()
        reg.register("my_tool", lambda: None, "desc", {"type": "object"})
        reg.unregister("my_tool")
        assert reg.has("my_tool") is False

    def test_unregister_missing(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.unregister("nonexistent")

    def test_list_all(self):
        reg = ToolRegistry()
        reg.register("a", lambda: None, "A", {"type": "object"}, category="cat1")
        reg.register("b", lambda: None, "B", {"type": "object"}, category="cat2")
        all_tools = reg.list_all()
        assert len(all_tools) == 2

        cat1_tools = reg.list_all(category="cat1")
        assert len(cat1_tools) == 1
        assert cat1_tools[0]["name"] == "a"

    def test_build_toolset(self):
        reg = ToolRegistry()
        reg.register("tool_a", lambda: "a", "Tool A", {"type": "object"})
        schemas, handlers = reg.build_toolset("tool_a")
        assert len(schemas) == 2  # tool_a + finish
        assert "tool_a" in handlers
        assert "finish" in handlers


class TestAgentRegistry:
    def test_register_and_has(self):
        from ganglion.composition.base_agent import BaseAgentWrapper

        reg = AgentRegistry()
        reg.register("MyAgent", BaseAgentWrapper)
        assert reg.has("MyAgent") is True
        assert reg.get("MyAgent") is BaseAgentWrapper

    def test_unregister(self):
        from ganglion.composition.base_agent import BaseAgentWrapper

        reg = AgentRegistry()
        reg.register("MyAgent", BaseAgentWrapper)
        reg.unregister("MyAgent")
        assert reg.has("MyAgent") is False

    def test_unregister_missing(self):
        reg = AgentRegistry()
        with pytest.raises(AgentNotFoundError):
            reg.unregister("nonexistent")

    def test_list_all(self):
        from ganglion.composition.base_agent import BaseAgentWrapper

        reg = AgentRegistry()
        reg.register("A", BaseAgentWrapper)
        reg.register("B", BaseAgentWrapper)
        items = reg.list_all()
        assert len(items) == 2

    def test_as_dict(self):
        from ganglion.composition.base_agent import BaseAgentWrapper

        reg = AgentRegistry()
        reg.register("A", BaseAgentWrapper)
        d = reg.as_dict()
        assert "A" in d


class TestMutation:
    def test_to_dict(self):
        m = Mutation(
            mutation_type="write_tool",
            target="my_tool",
            description="Added my_tool",
        )
        d = m.to_dict()
        assert d["mutation_type"] == "write_tool"
        assert d["target"] == "my_tool"


class TestMutationResult:
    def test_success(self):
        r = MutationResult(success=True, path="/tools/my_tool.py")
        assert r.success is True

    def test_failure(self):
        r = MutationResult(success=False, errors=["Bad syntax"])
        assert r.success is False
        assert "Bad syntax" in r.errors
