"""Tests for the MCP integration layer."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

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
        config = MCPClientConfig(
            name="test", transport="websocket", command=["python"]
        )
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
        config = MCPClientConfig(
            name="test", transport="stdio", command=["python"], timeout=0
        )
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
        config = MCPClientConfig(
            name="test", transport="stdio", command=["python"]
        )
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
        mock_session.call_tool.assert_called_once_with(
            "test_tool", arguments={"x": "hello"}
        )

    @pytest.mark.asyncio
    async def test_handler_returns_error_content_on_mcp_error(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(
            name="test", transport="stdio", command=["python"], timeout=5.0
        )
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

        config = MCPClientConfig(
            name="test", transport="stdio", command=["python"], timeout=0.01
        )
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

        config = MCPClientConfig(
            name="test", transport="stdio", command=["python"]
        )
        bridge = MCPClientBridge(config)
        assert bridge.get_tools() == {}

    @pytest.mark.asyncio
    async def test_disconnect_clears_tools(self):
        from ganglion.mcp.client import MCPClientBridge

        config = MCPClientConfig(
            name="test", transport="stdio", command=["python"]
        )
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
