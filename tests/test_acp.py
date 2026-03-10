"""Tests for the ACP integration layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ganglion.acp.config import ACPClientConfig, ACPServerConfig
from ganglion.acp.errors import ACPConnectionError, ACPNotAvailableError, ACPRunError


class TestACPClientConfig:
    def test_valid_config(self):
        config = ACPClientConfig(
            name="test-server",
            url="http://localhost:8950",
        )
        assert config.validate() == []

    def test_missing_name(self):
        config = ACPClientConfig(name="", url="http://localhost:8950")
        errors = config.validate()
        assert any("name" in e for e in errors)

    def test_missing_url(self):
        config = ACPClientConfig(name="test", url="")
        errors = config.validate()
        assert any("url" in e for e in errors)

    def test_invalid_timeout(self):
        config = ACPClientConfig(name="test", url="http://localhost:8950", timeout=0)
        errors = config.validate()
        assert any("timeout" in e for e in errors)

    def test_negative_timeout(self):
        config = ACPClientConfig(name="test", url="http://localhost:8950", timeout=-1)
        errors = config.validate()
        assert any("timeout" in e for e in errors)

    def test_to_dict_excludes_token(self):
        config = ACPClientConfig(
            name="test",
            url="http://localhost:8950",
            token="secret-token",
        )
        d = config.to_dict()
        assert "token" not in d
        assert d["name"] == "test"
        assert d["url"] == "http://localhost:8950"

    def test_defaults(self):
        config = ACPClientConfig(name="test", url="http://localhost:8950")
        assert config.timeout == 120.0
        assert config.agent_prefix == ""
        assert config.token is None
        assert config.headers == {}


class TestACPServerConfig:
    def test_valid_config(self):
        config = ACPServerConfig(host="0.0.0.0", port=8950)
        assert config.validate() == []

    def test_invalid_port_zero(self):
        config = ACPServerConfig(port=0)
        errors = config.validate()
        assert any("port" in e for e in errors)

    def test_invalid_port_high(self):
        config = ACPServerConfig(port=70000)
        errors = config.validate()
        assert any("port" in e for e in errors)

    def test_defaults(self):
        config = ACPServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8950
        assert config.token is None


class TestACPErrors:
    def test_connection_error(self):
        err = ACPConnectionError("connection refused")
        assert "connection refused" in str(err)

    def test_run_error(self):
        err = ACPRunError("agent timed out")
        assert "agent timed out" in str(err)

    def test_not_available_error(self):
        err = ACPNotAvailableError()
        assert "aiohttp" in str(err)
        assert "pip install" in str(err)


class TestACPClientBridge:
    """Tests for ACPClientBridge using mocked aiohttp."""

    @pytest.fixture
    def config(self):
        return ACPClientConfig(
            name="test-acp",
            url="http://localhost:8950",
            agent_prefix="remote",
        )

    def test_invalid_config_raises(self, config):
        """Bridge rejects invalid config on connect."""
        from ganglion.acp.client import ACPClientBridge

        bad_config = ACPClientConfig(name="", url="")
        bridge = ACPClientBridge(bad_config)
        with pytest.raises(ACPConnectionError, match="Invalid config"):
            pytest.importorskip("asyncio").get_event_loop().run_until_complete(bridge.connect())

    async def test_discover_agents_creates_tools(self, config):
        """Discovered ACP agents become ToolDef objects."""
        from ganglion.acp.client import ACPClientBridge

        mock_agents = [
            {"id": "summarizer", "name": "summarizer", "description": "Summarizes text"},
            {"id": "translator", "name": "translator", "description": "Translates text"},
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_agents)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False

        bridge = ACPClientBridge(config)
        bridge._session = mock_session

        await bridge._discover_agents()

        tools = bridge.get_tools()
        assert len(tools) == 2
        assert "remote_summarizer" in tools
        assert "remote_translator" in tools
        assert tools["remote_summarizer"].category == "acp"
        assert tools["remote_summarizer"].description == "Summarizes text"

    async def test_discover_agents_with_nested_format(self, config):
        """Handles ACP response with agents nested under 'agents' key."""
        from ganglion.acp.client import ACPClientBridge

        mock_agents = {
            "agents": [
                {"id": "agent1", "name": "agent1", "description": "Agent 1"},
            ]
        }

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_agents)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False

        bridge = ACPClientBridge(config)
        bridge._session = mock_session

        await bridge._discover_agents()
        assert len(bridge.get_tools()) == 1
        assert "remote_agent1" in bridge.get_tools()

    async def test_handler_sends_run_request(self, config):
        """Tool handler sends correct ACP run request."""
        from ganglion.acp.client import ACPClientBridge

        run_response = {
            "id": "run-123",
            "status": "completed",
            "output": [
                {"parts": [{"content": "Hello from agent", "content_type": "text/plain"}]}
            ],
        }

        mock_run_resp = AsyncMock()
        mock_run_resp.raise_for_status = MagicMock()
        mock_run_resp.json = AsyncMock(return_value=run_response)
        mock_run_resp.__aenter__ = AsyncMock(return_value=mock_run_resp)
        mock_run_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_run_resp)
        mock_session.closed = False

        bridge = ACPClientBridge(config)
        bridge._session = mock_session

        handler = bridge._make_handler("agent1", "agent1")
        result = await handler(message="Test input")

        assert result.content == "Hello from agent"
        # Verify the POST was called with correct body
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "/runs"
        body = call_args[1]["json"]
        assert body["agent_id"] == "agent1"
        assert body["input"][0]["parts"][0]["content"] == "Test input"

    async def test_handler_returns_error_on_failed_run(self, config):
        """Tool handler returns error content for failed ACP runs."""
        from ganglion.acp.client import ACPClientBridge

        run_response = {
            "id": "run-456",
            "status": "failed",
            "error": "Agent crashed",
            "output": [{"parts": [{"content": "Agent crashed", "content_type": "text/plain"}]}],
        }

        mock_run_resp = AsyncMock()
        mock_run_resp.raise_for_status = MagicMock()
        mock_run_resp.json = AsyncMock(return_value=run_response)
        mock_run_resp.__aenter__ = AsyncMock(return_value=mock_run_resp)
        mock_run_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_run_resp)
        mock_session.closed = False

        bridge = ACPClientBridge(config)
        bridge._session = mock_session

        handler = bridge._make_handler("agent1", "agent1")
        result = await handler(message="Test")

        assert "ACP Error" in result.content

    async def test_disconnect_clears_state(self, config):
        """Disconnect clears tools, agents, and closes session."""
        from ganglion.acp.client import ACPClientBridge

        mock_session = AsyncMock()
        mock_session.closed = False

        bridge = ACPClientBridge(config)
        bridge._session = mock_session
        bridge._tools["test"] = MagicMock()
        bridge._agents["test"] = {}

        await bridge.disconnect()

        assert bridge._session is None
        assert len(bridge._tools) == 0
        assert len(bridge._agents) == 0


class TestACPServerBridge:
    """Tests for ACPServerBridge using FastAPI test client."""

    @pytest.fixture
    def agent_registry(self):
        from ganglion.state.agent_registry import AgentRegistry

        return AgentRegistry()

    @pytest.fixture
    def tool_registry(self):
        from ganglion.state.tool_registry import ToolRegistry

        return ToolRegistry()

    @pytest.fixture
    def server_bridge(self, agent_registry, tool_registry):
        from ganglion.acp.server import ACPServerBridge

        return ACPServerBridge(
            agent_registry=agent_registry,
            tool_registry=tool_registry,
            server_name="test-ganglion",
        )

    def test_healthz(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["server"] == "test-ganglion"

    def test_list_agents_empty(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.get("/agents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_agents_with_registered(self, agent_registry, tool_registry):
        from ganglion.acp.server import ACPServerBridge
        from ganglion.composition.base_agent import BaseAgentWrapper

        class TestAgent(BaseAgentWrapper):
            def build_system_prompt(self, task):
                return "Test prompt"

            def build_tools(self, task):
                return [], {}

        agent_registry.register("TestAgent", TestAgent)
        bridge = ACPServerBridge(
            agent_registry=agent_registry,
            tool_registry=tool_registry,
            server_name="test",
        )

        from starlette.testclient import TestClient

        client = TestClient(bridge._app)
        resp = client.get("/agents")
        assert resp.status_code == 200
        agents = resp.json()
        assert len(agents) == 1
        assert agents[0]["id"] == "TestAgent"
        assert agents[0]["name"] == "TestAgent"

    def test_get_agent_not_found(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.get("/agents/nonexistent")
        assert resp.status_code == 404

    def test_create_run_missing_agent_id(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.post("/runs", json={})
        assert resp.status_code == 400

    def test_create_run_agent_not_found(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.post("/runs", json={"agent_id": "nonexistent"})
        assert resp.status_code == 404

    def test_get_run_not_found(self, server_bridge):
        from starlette.testclient import TestClient

        client = TestClient(server_bridge._app)
        resp = client.get("/runs/nonexistent")
        assert resp.status_code == 404

    def test_auth_required_when_token_set(self, agent_registry, tool_registry):
        from ganglion.acp.server import ACPServerBridge

        bridge = ACPServerBridge(
            agent_registry=agent_registry,
            tool_registry=tool_registry,
            server_name="test",
            token="secret-123",
        )

        from starlette.testclient import TestClient

        client = TestClient(bridge._app)

        # No auth header -> 401
        resp = client.get("/agents")
        assert resp.status_code == 401

        # Wrong token -> 401
        resp = client.get("/agents", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

        # Correct token -> 200
        resp = client.get("/agents", headers={"Authorization": "Bearer secret-123"})
        assert resp.status_code == 200


class TestExtractRunOutput:
    def test_basic_output(self):
        from ganglion.acp.client import _extract_run_output

        result = {
            "output": [
                {"parts": [{"content": "Hello world"}]},
            ]
        }
        assert _extract_run_output(result) == "Hello world"

    def test_multi_part_output(self):
        from ganglion.acp.client import _extract_run_output

        result = {
            "output": [
                {"parts": [{"content": "Part 1"}, {"content": "Part 2"}]},
            ]
        }
        assert _extract_run_output(result) == "Part 1\nPart 2"

    def test_multi_message_output(self):
        from ganglion.acp.client import _extract_run_output

        result = {
            "output": [
                {"parts": [{"content": "Message 1"}]},
                {"parts": [{"content": "Message 2"}]},
            ]
        }
        assert _extract_run_output(result) == "Message 1\nMessage 2"

    def test_empty_output_falls_back_to_error(self):
        from ganglion.acp.client import _extract_run_output

        result = {"output": [], "error": "Something went wrong"}
        assert _extract_run_output(result) == "Something went wrong"

    def test_non_string_content(self):
        from ganglion.acp.client import _extract_run_output

        result = {
            "output": [
                {"parts": [{"content": {"key": "value"}}]},
            ]
        }
        output = _extract_run_output(result)
        assert "key" in output
        assert "value" in output


class TestExtractInputText:
    def test_basic_input(self):
        from ganglion.acp.server import _extract_input_text

        messages = [{"parts": [{"content": "Hello"}]}]
        assert _extract_input_text(messages) == "Hello"

    def test_multi_part_input(self):
        from ganglion.acp.server import _extract_input_text

        messages = [{"parts": [{"content": "Part 1"}, {"content": "Part 2"}]}]
        assert _extract_input_text(messages) == "Part 1\nPart 2"

    def test_empty_input(self):
        from ganglion.acp.server import _extract_input_text

        assert _extract_input_text([]) == ""

    def test_non_string_content(self):
        from ganglion.acp.server import _extract_input_text

        messages = [{"parts": [{"content": {"data": 42}}]}]
        output = _extract_input_text(messages)
        assert "42" in output
