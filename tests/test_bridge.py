"""Tests for the HTTP bridge server (Layer 5)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import ganglion.bridge.server as srv
from ganglion.bridge.server import app, configure
from ganglion.state.framework_state import FrameworkState
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, OutputSpec, TaskDef
from ganglion.state.tool_registry import ToolRegistry
from ganglion.state.agent_registry import AgentRegistry


def make_mock_state() -> MagicMock:
    """Create a mock FrameworkState for testing."""
    state = MagicMock(spec=FrameworkState)
    state.pipeline_def = PipelineDef(
        name="test-pipeline",
        stages=[StageDef(name="train", agent="Trainer")],
    )
    state.tool_registry = ToolRegistry()
    state.agent_registry = AgentRegistry()
    state.knowledge = None
    state.persistence = None
    state.project_root = MagicMock()
    state.project_root.resolve.return_value = "/test/project"
    state.mutations = []

    async def mock_describe():
        return {
            "subnet": {},
            "pipeline": state.pipeline_def.to_dict(),
            "tools": [],
            "agents": [],
            "knowledge": None,
            "mutations": 0,
            "running": False,
        }

    state.describe = mock_describe
    return state


@pytest.fixture
def client():
    """Create a test client with a configured mock state.

    Sets state directly on the module to avoid re-adding CORS middleware.
    """
    mock_state = make_mock_state()
    srv._state = mock_state
    srv._config = None
    return TestClient(app)


class TestHealthEndpoints:
    def test_liveness(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness_when_configured(self, client):
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestObservationEndpoints:
    def test_get_status(self, client):
        response = client.get("/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_pipeline(self, client):
        response = client.get("/v1/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["name"] == "test-pipeline"

    def test_get_tools(self, client):
        response = client.get("/v1/tools")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_agents(self, client):
        response = client.get("/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_knowledge_empty(self, client):
        response = client.get("/v1/knowledge")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["patterns"] == []
        assert data["antipatterns"] == []

    def test_get_components(self, client):
        response = client.get("/v1/components")
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []


class TestResponseEnvelope:
    def test_success_envelope(self, client):
        response = client.get("/v1/pipeline")
        data = response.json()
        assert "data" in data

    def test_error_on_invalid_source_path(self, client):
        response = client.get("/v1/source/nonexistent_file.py")
        # Either 403 (path escape) or 404 (not found) depending on mock
        assert response.status_code in (403, 404)
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "code" in data["detail"]["error"]


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        response = client.get("/healthz")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "X-Request-ID" in response.headers

    def test_request_id_passthrough(self, client):
        response = client.get("/healthz", headers={"X-Request-ID": "test-123"})
        assert response.headers.get("X-Request-ID") == "test-123"


class TestBackwardCompatibility:
    def test_unversioned_status(self, client):
        response = client.get("/status")
        assert response.status_code == 200

    def test_unversioned_pipeline(self, client):
        response = client.get("/pipeline")
        assert response.status_code == 200

    def test_unversioned_tools(self, client):
        response = client.get("/tools")
        assert response.status_code == 200


class TestMutationEndpoints:
    def test_write_tool_validation_error(self, client):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["Missing @tool decorator"]

        async def mock_write(*args, **kwargs):
            return mock_result

        import ganglion.bridge.server as srv
        srv._state.write_and_register_tool = mock_write

        response = client.post("/v1/tools", json={
            "name": "bad_tool",
            "code": "def bad(): pass",
        })
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]

    def test_write_tool_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.path = "/tools/good_tool.py"

        async def mock_write(*args, **kwargs):
            return mock_result

        import ganglion.bridge.server as srv
        srv._state.write_and_register_tool = mock_write

        response = client.post("/v1/tools", json={
            "name": "good_tool",
            "code": "@tool('good_tool')\ndef good_tool(x: int) -> str:\n    '''doc'''\n    return str(x)",
        })
        assert response.status_code == 201


class TestInputValidation:
    def test_empty_tool_name_rejected(self, client):
        response = client.post("/v1/tools", json={
            "name": "",
            "code": "some code",
        })
        assert response.status_code == 422

    def test_empty_code_rejected(self, client):
        response = client.post("/v1/tools", json={
            "name": "tool",
            "code": "",
        })
        assert response.status_code == 422
