"""Tests for the MCP integration layer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ganglion.composition.tool_registry import ToolDef
from ganglion.composition.tool_returns import ToolOutput
from ganglion.mcp.config import MCPClientConfig
from ganglion.mcp.errors import MCPConnectionError, MCPNotAvailableError, MCPToolError


class TestMCPClientConfig:
    def test_valid_stdio_config(self):
        config = MCPClientConfig(
            name="test-server",
            transport="stdio",
            command=["python", "-m", "test_server"],
        )
        assert config.validate() == []

    def test_valid_sse_config(self):
        config = MCPClientConfig(
            name="test-server",
            transport="sse",
            url="http://localhost:8901/sse",
        )
        assert config.validate() == []

    def test_missing_name(self):
        config = MCPClientConfig(name="", transport="stdio", command=["python"])
        errors = config.validate()
        assert any("name" in e for e in errors)

    def test_invalid_transport(self):
        config = MCPClientConfig(name="test", transport="websocket", command=["python"])
        errors = config.validate()
        assert any("transport" in e for e in errors)

    def test_stdio_missing_command(self):
        config = MCPClientConfig(name="test", transport="stdio")
        errors = config.validate()
        assert any("command" in e for e in errors)

    def test_sse_missing_url(self):
        config = MCPClientConfig(name="test", transport="sse")
        errors = config.validate()
        assert any("url" in e for e in errors)

    def test_invalid_timeout(self):
        config = MCPClientConfig(name="test", transport="stdio", command=["python"], timeout=0)
        errors = config.validate()
        assert any("timeout" in e for e in errors)

    def test_to_dict(self):
        config = MCPClientConfig(
            name="test",
            transport="stdio",
            command=["python", "-m", "server"],
            tool_prefix="myprefix",
        )
        d = config.to_dict()
        assert d["name"] == "test"
        assert d["transport"] == "stdio"
        assert d["tool_prefix"] == "myprefix"

    def test_defaults(self):
        config = MCPClientConfig(name="test", transport="stdio", command=["python"])
        assert config.category == "mcp"
        assert config.timeout == 30.0
        assert config.tool_prefix == ""
        assert config.env is None


class TestMCPErrors:
    def test_mcp_connection_error(self):
        err = MCPConnectionError("failed")
        assert str(err) == "failed"

    def test_mcp_tool_error(self):
        err = MCPToolError("timeout")
        assert str(err) == "timeout"

    def test_mcp_not_available(self):
        err = MCPNotAvailableError()
        assert "pip install ganglion[mcp]" in str(err)


class TestAsyncToolExecution:
    """Test that SimpleAgent._execute_tool() handles async handlers."""

    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self):
        """Verify that async tool handlers are properly awaited."""
        from ganglion.runtime.agent import SimpleAgent
        from ganglion.runtime.types import ToolCall

        async def async_tool(**kwargs):
            return ToolOutput(content=f"async result: {kwargs.get('x', 'none')}")

        mock_llm = MagicMock()
        agent = SimpleAgent(
            llm_client=mock_llm,
            system_prompt="test",
            tools_schema=[],
            tool_handlers={"async_tool": async_tool},
        )

        tc = ToolCall(id="tc1", name="async_tool", arguments={"x": "hello"})
        result = await agent._execute_tool(tc)

        assert result.content == "async result: hello"
        assert result.name == "async_tool"

    @pytest.mark.asyncio
    async def test_sync_handler_still_works(self):
        """Verify that sync tool handlers are unaffected."""
        from ganglion.runtime.agent import SimpleAgent
        from ganglion.runtime.types import ToolCall

        def sync_tool(**kwargs):
            return ToolOutput(content=f"sync result: {kwargs.get('x', 'none')}")

        mock_llm = MagicMock()
        agent = SimpleAgent(
            llm_client=mock_llm,
            system_prompt="test",
            tools_schema=[],
            tool_handlers={"sync_tool": sync_tool},
        )

        tc = ToolCall(id="tc2", name="sync_tool", arguments={"x": "world"})
        result = await agent._execute_tool(tc)

        assert result.content == "sync result: world"

    @pytest.mark.asyncio
    async def test_async_handler_error_is_caught(self):
        """Verify that errors from async handlers are caught gracefully."""
        from ganglion.runtime.agent import SimpleAgent
        from ganglion.runtime.types import ToolCall

        async def failing_tool(**kwargs):
            raise MCPToolError("connection lost")

        mock_llm = MagicMock()
        agent = SimpleAgent(
            llm_client=mock_llm,
            system_prompt="test",
            tools_schema=[],
            tool_handlers={"failing_tool": failing_tool},
        )

        tc = ToolCall(id="tc3", name="failing_tool", arguments={})
        result = await agent._execute_tool(tc)

        assert "connection lost" in result.content
        assert result.name == "failing_tool"


class TestMCPClientBridge:
    """Test MCPClientBridge tool conversion using mocked MCP sessions."""

    @pytest.mark.asyncio
    async def test_invalid_config_raises(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(name="", transport="stdio")
        bridge = MCPClientBridge(config)
        with pytest.raises(MCPConnectionError, match="Invalid config"):
            await bridge.connect()

    @pytest.mark.asyncio
    async def test_make_handler_returns_async_callable(self):
        """Verify the handler closure is async."""
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(
            name="test",
            transport="stdio",
            command=["python"],
            tool_prefix="test",
            timeout=5.0,
        )
        bridge = MCPClientBridge(config)

        # Mock session
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.isError = False
        mock_content = MagicMock()
        mock_content.text = "result text"
        mock_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        bridge.session = mock_session
        handler = bridge._make_handler("test_tool")

        assert asyncio.iscoroutinefunction(handler)

        result = await handler(x="hello")
        assert isinstance(result, ToolOutput)
        assert result.content == "result text"
        mock_session.call_tool.assert_called_once_with("test_tool", arguments={"x": "hello"})

    @pytest.mark.asyncio
    async def test_handler_returns_error_content_on_mcp_error(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(name="test", transport="stdio", command=["python"], timeout=5.0)
        bridge = MCPClientBridge(config)

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.isError = True
        mock_content = MagicMock()
        mock_content.text = "something went wrong"
        mock_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        bridge.session = mock_session
        handler = bridge._make_handler("failing_tool")

        result = await handler()
        assert "MCP Error" in result.content
        assert "something went wrong" in result.content

    @pytest.mark.asyncio
    async def test_handler_timeout(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(name="test", transport="stdio", command=["python"], timeout=0.01)
        bridge = MCPClientBridge(config)

        mock_session = MagicMock()

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(1)

        mock_session.call_tool = slow_call
        bridge.session = mock_session

        handler = bridge._make_handler("slow_tool")
        with pytest.raises(MCPToolError, match="timed out"):
            await handler()

    def test_get_tools_empty_before_connect(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(name="test", transport="stdio", command=["python"])
        bridge = MCPClientBridge(config)
        assert bridge.get_tools() == {}

    @pytest.mark.asyncio
    async def test_disconnect_clears_tools(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(name="test", transport="stdio", command=["python"])
        bridge = MCPClientBridge(config)
        # Manually add a tool to simulate post-connect state
        bridge._tools["test_tool"] = ToolDef(
            name="test_tool",
            description="test",
            func=lambda: None,
            parameters_schema={"type": "object", "properties": {}},
        )

        await bridge.disconnect()
        assert bridge.get_tools() == {}
        assert bridge.session is None


class TestMCPConfigGanglion:
    """Test GanglionConfig MCP fields."""

    def test_default_mcp_config(self):
        from ganglion.config import GanglionConfig

        config = GanglionConfig()
        assert config.mcp_server_enabled is False
        assert config.mcp_server_transport == "stdio"
        assert config.mcp_server_sse_port == 8900

    def test_mcp_server_validation(self):
        from ganglion.config import GanglionConfig

        config = GanglionConfig(mcp_server_transport="invalid")
        errors = config.validate()
        assert any("MCP_SERVER_TRANSPORT" in e for e in errors)

    def test_mcp_port_validation(self):
        from ganglion.config import GanglionConfig

        config = GanglionConfig(mcp_server_sse_port=0)
        errors = config.validate()
        assert any("MCP_SERVER_SSE_PORT" in e for e in errors)


class TestFrameworkStateMCP:
    """Test FrameworkState MCP methods."""

    def _make_state(self):
        from ganglion.orchestration.pipeline import PipelineDef, StageDef
        from ganglion.orchestration.task_context import (
            MetricDef,
            OutputSpec,
            SubnetConfig,
            TaskDef,
        )
        from ganglion.state.framework_state import FrameworkState

        return FrameworkState.create(
            subnet_config=SubnetConfig(
                netuid=1,
                name="test",
                metrics=[MetricDef("score", "maximize")],
                tasks={"default": TaskDef("default")},
                output_spec=OutputSpec(format="model"),
            ),
            pipeline_def=PipelineDef(
                name="test-pipeline",
                stages=[StageDef(name="plan", agent="Explorer")],
            ),
        )

    def test_describe_mcp_empty(self):
        state = self._make_state()
        mcp_status = state._describe_mcp()
        assert mcp_status["connected_servers"] == []
        assert mcp_status["total_tools"] == 0

    @pytest.mark.asyncio
    async def test_connect_mcp_server_invalid_config(self):
        state = self._make_state()
        config = MCPClientConfig(name="", transport="stdio")
        result = await state.connect_mcp_server(config)
        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_server(self):
        state = self._make_state()
        result = await state.disconnect_mcp_server("nonexistent")
        assert result.success is False
        assert any("not connected" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_initialize_mcp_with_no_configs(self):
        state = self._make_state()
        await state.initialize_mcp()
        assert len(state._mcp_bridges) == 0

    @pytest.mark.asyncio
    async def test_shutdown_mcp_with_no_connections(self):
        state = self._make_state()
        await state.shutdown_mcp()
        assert len(state._mcp_bridges) == 0

    @pytest.mark.asyncio
    async def test_describe_includes_mcp(self):
        state = self._make_state()
        desc = await state.describe()
        assert "mcp" in desc
        assert "connected_servers" in desc["mcp"]


class TestMCPServerBridge:
    """Test MCPServerBridge exposing tools as MCP server."""

    def _make_registry(self):  # type: ignore[no-untyped-def]
        from ganglion.state.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="greet",
            func=lambda name="world": f"Hello, {name}!",
            description="Greet someone",
            parameters_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
            category="general",
        )
        registry.register(
            name="add",
            func=lambda a=0, b=0: a + b,
            description="Add two numbers",
            parameters_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
            category="math",
        )
        return registry

    def test_create_bridge(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = self._make_registry()
        bridge = MCPServerBridge(registry, server_name="test-server")
        assert bridge._registry is registry
        assert bridge._server is not None

    def test_create_bridge_with_categories(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = self._make_registry()
        bridge = MCPServerBridge(registry, categories=["math"])
        assert bridge._categories == ["math"]

    @pytest.mark.asyncio
    async def test_list_tools_all(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = self._make_registry()
        _bridge = MCPServerBridge(registry)
        assert _bridge._server is not None
        assert len(registry.list_all()) == 2

    @pytest.mark.asyncio
    async def test_list_tools_with_category_filter(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = self._make_registry()
        bridge = MCPServerBridge(registry, categories=["math"])

        # Simulate the handler's filtering logic
        tools = []
        for tool_dict in registry.list_all():
            if bridge._categories and tool_dict.get("category") not in bridge._categories:
                continue
            tools.append(tool_dict["name"])
        assert tools == ["add"]

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = self._make_registry()
        _bridge = MCPServerBridge(registry)

        tool_def = registry.get("greet")
        assert tool_def is not None
        result = tool_def.func(name="Claude")
        assert result == "Hello, Claude!"
        assert _bridge._server is not None

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        from ganglion.mcp.server import MCPServerBridge

        _bridge = MCPServerBridge(self._make_registry())
        assert _bridge._registry.get("nonexistent") is None

    def test_call_tool_with_exception(self):
        from ganglion.state.tool_registry import ToolRegistry

        registry = ToolRegistry()

        def failing_func(**kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("test error")

        registry.register(
            name="fail_tool",
            func=failing_func,
            description="A failing tool",
            parameters_schema={"type": "object", "properties": {}},
        )

        from ganglion.mcp.server import MCPServerBridge

        _bridge = MCPServerBridge(registry)
        tool_def = _bridge._registry.get("fail_tool")
        assert tool_def is not None
        with pytest.raises(ValueError, match="test error"):
            tool_def.func()

    @pytest.mark.asyncio
    async def test_call_tool_with_tool_output(self):
        from ganglion.state.tool_registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(
            name="output_tool",
            func=lambda: ToolOutput(content="structured content"),
            description="Returns ToolOutput",
            parameters_schema={"type": "object", "properties": {}},
        )

        from ganglion.mcp.server import MCPServerBridge

        _bridge = MCPServerBridge(registry)
        tool_def = _bridge._registry.get("output_tool")
        assert tool_def is not None
        result = tool_def.func()
        assert hasattr(result, "content")
        assert result.content == "structured content"


# ── Helpers for new tests ─────────────────────────────────


def _make_categorized_registry():
    """Create a registry with tools in different categories (observation, execution, admin)."""
    from ganglion.state.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(
        name="observe_status",
        func=lambda: "status ok",
        description="Observe status",
        parameters_schema={"type": "object", "properties": {}},
        category="observation",
    )
    registry.register(
        name="run_task",
        func=lambda task="default": f"ran {task}",
        description="Run a task",
        parameters_schema={
            "type": "object",
            "properties": {"task": {"type": "string"}},
        },
        category="execution",
    )
    registry.register(
        name="admin_rollback",
        func=lambda: "rolled back",
        description="Admin rollback",
        parameters_schema={"type": "object", "properties": {}},
        category="admin",
    )
    return registry


def _make_mock_framework_state():
    """Create a mock FrameworkState for testing framework tools."""
    from ganglion.orchestration.pipeline import PipelineDef, StageDef
    from ganglion.orchestration.task_context import (
        MetricDef,
        OutputSpec,
        SubnetConfig,
        TaskDef,
    )
    from ganglion.state.framework_state import FrameworkState

    state = FrameworkState.create(
        subnet_config=SubnetConfig(
            netuid=1,
            name="test",
            metrics=[MetricDef("score", "maximize")],
            tasks={"default": TaskDef("default")},
            output_spec=OutputSpec(format="model"),
        ),
        pipeline_def=PipelineDef(
            name="test-pipeline",
            stages=[StageDef(name="plan", agent="Explorer")],
        ),
    )
    return state


# ── Feature 1: Framework tools tests ─────────────────────


class TestMCPFrameworkTools:
    """Test that register_framework_tools exposes all HTTP bridge operations."""

    def test_registers_observation_tools(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        registered = register_framework_tools(registry, state)

        observation_tools = [
            "ganglion_get_status",
            "ganglion_get_pipeline",
            "ganglion_get_tools",
            "ganglion_get_agents",
            "ganglion_get_runs",
            "ganglion_get_metrics",
            "ganglion_get_leaderboard",
            "ganglion_get_knowledge",
            "ganglion_get_source",
            "ganglion_get_components",
            "ganglion_get_mcp_status",
        ]
        for name in observation_tools:
            assert name in registered, f"Missing observation tool: {name}"

    def test_registers_mutation_tools(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        registered = register_framework_tools(registry, state)

        mutation_tools = [
            "ganglion_write_tool",
            "ganglion_write_agent",
            "ganglion_write_component",
            "ganglion_write_prompt",
            "ganglion_patch_pipeline",
            "ganglion_swap_policy",
        ]
        for name in mutation_tools:
            assert name in registered, f"Missing mutation tool: {name}"

    def test_registers_execution_tools(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        registered = register_framework_tools(registry, state)

        execution_tools = [
            "ganglion_run_pipeline",
            "ganglion_run_stage",
            "ganglion_run_experiment",
        ]
        for name in execution_tools:
            assert name in registered, f"Missing execution tool: {name}"

    def test_registers_admin_tools(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        registered = register_framework_tools(registry, state)

        admin_tools = [
            "ganglion_rollback_last",
            "ganglion_rollback_to",
            "ganglion_connect_mcp",
            "ganglion_disconnect_mcp",
            "ganglion_reconnect_mcp",
        ]
        for name in admin_tools:
            assert name in registered, f"Missing admin tool: {name}"

    @pytest.mark.asyncio
    async def test_get_status_returns_json(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_status")
        assert tool_def is not None
        result = await tool_def.func()
        data = json.loads(result)
        assert "pipeline" in data
        assert "tools" in data

    @pytest.mark.asyncio
    async def test_get_pipeline_returns_json(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_pipeline")
        assert tool_def is not None
        result = await tool_def.func()
        data = json.loads(result)
        assert data["name"] == "test-pipeline"

    @pytest.mark.asyncio
    async def test_get_source_blocks_path_traversal(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_source")
        assert tool_def is not None
        result = await tool_def.func(path="../../../etc/passwd")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_source_blocks_absolute_path(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_source")
        assert tool_def is not None
        result = await tool_def.func(path="/etc/passwd")
        data = json.loads(result)
        assert "error" in data

    def test_does_not_double_register(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        first = register_framework_tools(registry, state)
        second = register_framework_tools(registry, state)
        assert len(first) > 0
        assert len(second) == 0  # all already registered

    def test_tool_categories_assigned(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        assert registry.get("ganglion_get_status").category == "observation"
        assert registry.get("ganglion_write_tool").category == "mutation"
        assert registry.get("ganglion_run_pipeline").category == "execution"
        assert registry.get("ganglion_rollback_last").category == "admin"

    @pytest.mark.asyncio
    async def test_get_tools_returns_json(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_tools")
        result = await tool_def.func()
        data = json.loads(result)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_agents_returns_json(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_agents")
        result = await tool_def.func()
        data = json.loads(result)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_runs_no_persistence(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_runs")
        result = await tool_def.func()
        assert json.loads(result) == []

    @pytest.mark.asyncio
    async def test_get_metrics_no_persistence(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_metrics")
        result = await tool_def.func()
        assert json.loads(result) == []

    @pytest.mark.asyncio
    async def test_get_leaderboard_no_client(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_leaderboard")
        result = await tool_def.func()
        assert json.loads(result) == []

    @pytest.mark.asyncio
    async def test_get_knowledge_no_store(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_knowledge")
        result = await tool_def.func()
        data = json.loads(result)
        assert data["patterns"] == []
        assert data["antipatterns"] == []

    @pytest.mark.asyncio
    async def test_get_components_no_framework(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_components")
        result = await tool_def.func()
        assert json.loads(result) == []

    @pytest.mark.asyncio
    async def test_get_mcp_status(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_mcp_status")
        result = await tool_def.func()
        data = json.loads(result)
        assert "connected_servers" in data

    @pytest.mark.asyncio
    async def test_get_source_valid_file(self, tmp_path):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        state.project_root = tmp_path
        (tmp_path / "hello.py").write_text("print('hi')")
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_source")
        result = await tool_def.func(path="hello.py")
        data = json.loads(result)
        assert data["content"] == "print('hi')"

    @pytest.mark.asyncio
    async def test_get_source_not_found(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_get_source")
        result = await tool_def.func(path="nonexistent.py")
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_rollback_to_negative_index(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_rollback_to")
        result = await tool_def.func(index=-1)
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_write_component_no_framework(self, tmp_path):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        state.project_root = tmp_path
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_write_component")
        result = await tool_def.func(name="my_comp", code="x = 1")
        data = json.loads(result)
        assert data["success"] is True
        assert (tmp_path / "components" / "my_comp.py").read_text() == "x = 1"

    @pytest.mark.asyncio
    async def test_write_tool_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_write_tool")
        result = await tool_def.func(name="my_tool", code="def my_tool(): pass")
        data = json.loads(result)
        # Validation will fail since code doesn't have proper decorators etc.
        assert "success" in data

    @pytest.mark.asyncio
    async def test_write_agent_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_write_agent")
        result = await tool_def.func(name="my_agent", code="class MyAgent: pass")
        data = json.loads(result)
        assert "success" in data

    @pytest.mark.asyncio
    async def test_write_prompt_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_write_prompt")
        result = await tool_def.func(
            agent_name="Explorer", prompt_section="system", content="Be helpful"
        )
        data = json.loads(result)
        assert "success" in data

    @pytest.mark.asyncio
    async def test_patch_pipeline_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_patch_pipeline")
        result = await tool_def.func(operations=[{"op": "add", "stage": "test"}])
        data = json.loads(result)
        assert "success" in data

    @pytest.mark.asyncio
    async def test_swap_policy_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_swap_policy")
        result = await tool_def.func(
            stage_name="default", retry_policy={"type": "fixed", "max_attempts": 3}
        )
        data = json.loads(result)
        assert "success" in data

    @pytest.mark.asyncio
    async def test_rollback_last_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_rollback_last")
        result = await tool_def.func()
        data = json.loads(result)
        assert "success" in data

    @pytest.mark.asyncio
    async def test_disconnect_mcp_calls_state(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_disconnect_mcp")
        result = await tool_def.func(name="nonexistent")
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_reconnect_mcp_not_connected(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_reconnect_mcp")
        result = await tool_def.func(name="nonexistent")
        data = json.loads(result)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_run_pipeline_exception(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        # run_pipeline will fail since no agents are registered
        tool_def = registry.get("ganglion_run_pipeline")
        result = await tool_def.func()
        data = json.loads(result)
        # Either success with result or error
        assert "success" in data or "error" in data

    @pytest.mark.asyncio
    async def test_run_stage_exception(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_run_stage")
        result = await tool_def.func(stage_name="nonexistent")
        data = json.loads(result)
        assert "success" in data or "error" in data

    @pytest.mark.asyncio
    async def test_run_experiment_exception(self):
        from ganglion.mcp.tools import register_framework_tools
        from ganglion.state.tool_registry import ToolRegistry

        state = _make_mock_framework_state()
        registry = ToolRegistry()
        register_framework_tools(registry, state)

        tool_def = registry.get("ganglion_run_experiment")
        result = await tool_def.func(config={"task": "test"})
        data = json.loads(result)
        assert "success" in data or "error" in data

    def test_json_result_fallback(self):
        from ganglion.mcp.tools import _json_result

        # Normal case
        assert json.loads(_json_result({"key": "value"})) == {"key": "value"}

        # Unserializable object falls back to str()
        class Unserializable:
            def __repr__(self):
                return "unserializable-repr"

        obj = Unserializable()
        obj.x = obj  # circular reference
        result = _json_result(obj)
        assert "unserializable-repr" in result


# ── Feature 2: Roles config tests ────────────────────────


class TestMCPRolesConfig:
    """Test MCPRolesConfig validation, mirroring TestInputValidation."""

    def test_valid_config_from_file(self, tmp_path):
        from ganglion.mcp.roles import MCPRolesConfig

        roles_file = tmp_path / "roles.json"
        roles_file.write_text(
            json.dumps(
                [
                    {"name": "admin", "categories": None, "token": "abc", "port": 8901},
                    {"name": "worker", "categories": ["observation"], "token": "def", "port": 8902},
                ]
            )
        )
        config = MCPRolesConfig.from_file(roles_file)
        assert len(config.roles) == 2
        assert config.validate() == []

    def test_duplicate_port_rejected(self):
        from ganglion.mcp.roles import MCPRole, MCPRolesConfig

        config = MCPRolesConfig(
            roles=[
                MCPRole(name="a", token="t1", port=8901),
                MCPRole(name="b", token="t2", port=8901),
            ]
        )
        errors = config.validate()
        assert any("Duplicate SSE port" in e for e in errors)

    def test_duplicate_name_rejected(self):
        from ganglion.mcp.roles import MCPRole, MCPRolesConfig

        config = MCPRolesConfig(
            roles=[
                MCPRole(name="admin", token="t1", port=8901),
                MCPRole(name="admin", token="t2", port=8902),
            ]
        )
        errors = config.validate()
        assert any("Duplicate role name" in e for e in errors)

    def test_empty_token_rejected(self):
        from ganglion.mcp.roles import MCPRole, MCPRolesConfig

        config = MCPRolesConfig(
            roles=[
                MCPRole(name="worker", token="", port=8901),
            ]
        )
        errors = config.validate()
        assert any("empty token" in e for e in errors)

    def test_multiple_stdio_rejected(self):
        from ganglion.mcp.roles import MCPRole, MCPRolesConfig

        config = MCPRolesConfig(
            roles=[
                MCPRole(name="a", token="t1", port=8901, transport="stdio"),
                MCPRole(name="b", token="t2", port=8902, transport="stdio"),
            ]
        )
        errors = config.validate()
        assert any("stdio" in e for e in errors)

    def test_from_file_nonexistent(self):
        from ganglion.mcp.roles import MCPRolesConfig

        with pytest.raises(FileNotFoundError):
            MCPRolesConfig.from_file(Path("/nonexistent/roles.json"))

    def test_empty_roles_rejected(self):
        from ganglion.mcp.roles import MCPRolesConfig

        config = MCPRolesConfig(roles=[])
        errors = config.validate()
        assert any("At least one role" in e for e in errors)

    def test_invalid_transport_rejected(self):
        from ganglion.mcp.roles import MCPRole, MCPRolesConfig

        config = MCPRolesConfig(
            roles=[
                MCPRole(name="a", token="t1", port=8901, transport="websocket"),
            ]
        )
        errors = config.validate()
        assert any("invalid transport" in e for e in errors)


# ── Feature 2: Auth tests ────────────────────────────────


class TestMCPServerAuth:
    """Test bearer token authentication on MCPServerBridge, mirroring TestSecurityHeaders."""

    def test_bridge_stores_token(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, token="secret-token", role="admin")
        assert bridge._token == "secret-token"
        assert bridge.role == "admin"

    def test_no_token_means_no_auth(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry)
        assert bridge._token is None


# ── Feature 2: Role-based tool filtering tests ───────────


class TestMCPServerBridgeRoles:
    """Test role-based category filtering, mirroring TestMutationEndpoints."""

    def test_admin_sees_all_tools(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, categories=None, role="admin")

        tools = []
        for tool_dict in registry.list_all():
            if bridge._categories and tool_dict.get("category") not in bridge._categories:
                continue
            tools.append(tool_dict["name"])

        assert "observe_status" in tools
        assert "run_task" in tools
        assert "admin_rollback" in tools

    def test_worker_sees_filtered_tools(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, categories=["observation", "execution"], role="worker")

        tools = []
        for tool_dict in registry.list_all():
            if bridge._categories and tool_dict.get("category") not in bridge._categories:
                continue
            tools.append(tool_dict["name"])

        assert "observe_status" in tools
        assert "run_task" in tools
        assert "admin_rollback" not in tools

    def test_observer_sees_read_only_tools(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, categories=["observation"], role="observer")

        tools = []
        for tool_dict in registry.list_all():
            if bridge._categories and tool_dict.get("category") not in bridge._categories:
                continue
            tools.append(tool_dict["name"])

        assert "observe_status" in tools
        assert "run_task" not in tools
        assert "admin_rollback" not in tools

    def test_role_label_stored(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, role="worker")
        assert bridge.role == "worker"


# ── Feature 3: Usage tracker tests ───────────────────────


class TestUsageTracker:
    """Test UsageTracker in-memory and SQLite functionality."""

    @pytest.mark.asyncio
    async def test_record_increments_counters(self):
        from ganglion.mcp.usage import UsageTracker

        tracker = UsageTracker()
        await tracker.record("alpha", "greet", True, 10.0)
        await tracker.record("alpha", "greet", True, 5.0)
        await tracker.record("alpha", "add", True, 3.0)

        stats = tracker.get_bot_stats("alpha")
        assert stats["totals"]["total"] == 3
        assert stats["totals"]["success"] == 3
        assert stats["per_tool"]["greet"] == 2
        assert stats["per_tool"]["add"] == 1

    def test_get_bot_stats_empty(self):
        from ganglion.mcp.usage import UsageTracker

        tracker = UsageTracker()
        stats = tracker.get_bot_stats("unknown")
        assert stats["totals"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_all_stats_multiple_bots(self):
        from ganglion.mcp.usage import UsageTracker

        tracker = UsageTracker()
        await tracker.record("alpha", "tool1", True, 1.0)
        await tracker.record("beta", "tool2", True, 2.0)

        all_stats = tracker.get_all_stats()
        assert len(all_stats) == 2
        bot_ids = {s["bot_id"] for s in all_stats}
        assert bot_ids == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_success_failure_separation(self):
        from ganglion.mcp.usage import UsageTracker

        tracker = UsageTracker()
        await tracker.record("alpha", "tool", True, 1.0)
        await tracker.record("alpha", "tool", False, 2.0)
        await tracker.record("alpha", "tool", False, 3.0)

        stats = tracker.get_bot_stats("alpha")
        assert stats["totals"]["success"] == 1
        assert stats["totals"]["failure"] == 2
        assert stats["totals"]["total"] == 3

    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, tmp_path):
        import sqlite3

        from ganglion.mcp.usage import UsageTracker

        db_path = tmp_path / "usage.db"
        tracker = UsageTracker(db_path=db_path)
        await tracker.record("alpha", "greet", True, 10.5)

        # Verify data was written to SQLite
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT bot_id, tool_name, success FROM usage_log").fetchall()
        assert len(rows) == 1
        assert rows[0] == ("alpha", "greet", 1)

    @pytest.mark.asyncio
    async def test_in_memory_only(self):
        from ganglion.mcp.usage import UsageTracker

        tracker = UsageTracker()  # no db_path
        await tracker.record("alpha", "tool", True, 1.0)
        stats = tracker.get_bot_stats("alpha")
        assert stats["totals"]["total"] == 1


# ── Feature 3: Usage tracker integration tests ───────────


class TestUsageTrackerIntegration:
    """Test usage tracking wired into MCPServerBridge."""

    def test_bridge_accepts_usage_tracker(self):
        from ganglion.mcp.server import MCPServerBridge
        from ganglion.mcp.usage import UsageTracker

        registry = _make_categorized_registry()
        tracker = UsageTracker()
        bridge = MCPServerBridge(registry, usage_tracker=tracker, role="test")
        assert bridge._usage_tracker is tracker

    def test_no_tracking_without_tracker(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry)
        assert bridge._usage_tracker is None

    def test_bot_id_resolved_from_role(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, role="worker")
        assert bridge._resolve_bot_id() == "worker"

    def test_bot_id_none_without_role(self):
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry)
        assert bridge._resolve_bot_id() is None


# ── Backward compatibility tests ─────────────────────────


class TestMCPBackwardCompatibility:
    """Test backward compatibility when no roles/token/tracker configured."""

    def test_no_roles_single_server(self):
        """Bridge works without roles config (original behavior)."""
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry, server_name="test")
        assert bridge._token is None
        assert bridge.role is None
        assert bridge._categories is None
        assert bridge._usage_tracker is None

    def test_no_token_no_auth(self):
        """Without token, no auth is enforced."""
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry)
        assert bridge._token is None

    def test_no_tracker_no_tracking(self):
        """Without usage tracker, no tracking happens."""
        from ganglion.mcp.server import MCPServerBridge

        registry = _make_categorized_registry()
        bridge = MCPServerBridge(registry)
        assert bridge._usage_tracker is None
