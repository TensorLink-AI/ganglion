"""Tests for the HTTP bridge server (Layer 5)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import ganglion.bridge.server as srv
from ganglion.bridge.server import app
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.state.agent_registry import AgentRegistry
from ganglion.state.framework_state import FrameworkState
from ganglion.state.tool_registry import ToolRegistry


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

        response = client.post(
            "/v1/tools",
            json={
                "name": "bad_tool",
                "code": "def bad(): pass",
            },
        )
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

        response = client.post(
            "/v1/tools",
            json={
                "name": "good_tool",
                "code": (
                    "@tool('good_tool')\n"
                    "def good_tool(x: int) -> str:\n"
                    "    '''doc'''\n"
                    "    return str(x)"
                ),
            },
        )
        assert response.status_code == 201


class TestInputValidation:
    def test_empty_tool_name_rejected(self, client):
        response = client.post(
            "/v1/tools",
            json={
                "name": "",
                "code": "some code",
            },
        )
        assert response.status_code == 422

    def test_empty_code_rejected(self, client):
        response = client.post(
            "/v1/tools",
            json={
                "name": "tool",
                "code": "",
            },
        )
        assert response.status_code == 422


class TestWriteAgentEndpoint:
    def test_write_agent_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.path = "/agents/test_agent.py"

        async def mock_write(*args, **kwargs):
            return mock_result

        srv._state.write_and_register_agent = mock_write
        response = client.post(
            "/v1/agents",
            json={"name": "TestAgent", "code": "class TestAgent: pass"},
        )
        assert response.status_code == 201

    def test_write_agent_validation_error(self, client):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["Missing BaseAgentWrapper subclass"]

        async def mock_write(*args, **kwargs):
            return mock_result

        srv._state.write_and_register_agent = mock_write
        response = client.post(
            "/v1/agents",
            json={"name": "BadAgent", "code": "def bad(): pass"},
        )
        assert response.status_code == 400


class TestWriteComponentEndpoint:
    def test_write_component_fallback(self, client):
        """Test write_component when no training_framework is set."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            srv._state.project_root = Path(tmp)
            response = client.post(
                "/v1/components",
                json={
                    "name": "test_backbone",
                    "code": "class Backbone: pass",
                    "component_type": "backbone",
                },
            )
            assert response.status_code == 201
            data = response.json()["data"]
            assert "path" in data


class TestWritePromptEndpoint:
    def test_write_prompt_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.path = "/prompts/agent.py"

        async def mock_update(*args, **kwargs):
            return mock_result

        srv._state.update_prompt = mock_update
        response = client.post(
            "/v1/prompts",
            json={
                "agent_name": "trainer",
                "prompt_section": "role",
                "content": "You are a trainer agent.",
            },
        )
        assert response.status_code == 200


class TestPatchPipelineEndpoint:
    def test_patch_pipeline_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.pipeline = {"name": "test", "stages": []}

        async def mock_patch(*args, **kwargs):
            return mock_result

        srv._state.apply_pipeline_patch = mock_patch
        response = client.patch(
            "/v1/pipeline",
            json={"operations": [{"op": "add_stage", "stage": {"name": "s", "agent": "A"}}]},
        )
        assert response.status_code == 200

    def test_patch_pipeline_error(self, client):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["Invalid operation"]

        async def mock_patch(*args, **kwargs):
            return mock_result

        srv._state.apply_pipeline_patch = mock_patch
        response = client.patch(
            "/v1/pipeline",
            json={"operations": [{"op": "bad_op"}]},
        )
        assert response.status_code == 400


class TestSwapPolicyEndpoint:
    def test_swap_policy_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True

        async def mock_swap(*args, **kwargs):
            return mock_result

        srv._state.swap_policy = mock_swap
        response = client.put(
            "/v1/policies/train",
            json={"retry_policy": {"type": "fixed", "max_attempts": 3}},
        )
        assert response.status_code == 200

    def test_swap_default_policy(self, client):
        mock_result = MagicMock()
        mock_result.success = True

        async def mock_swap(*args, **kwargs):
            return mock_result

        srv._state.swap_policy = mock_swap
        response = client.put(
            "/v1/policies/default",
            json={"retry_policy": {"type": "none"}},
        )
        assert response.status_code == 200

    def test_swap_policy_error(self, client):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["Stage 'missing' not found"]

        async def mock_swap(*args, **kwargs):
            return mock_result

        srv._state.swap_policy = mock_swap
        response = client.put(
            "/v1/policies/missing",
            json={"retry_policy": {"type": "fixed", "max_attempts": 3}},
        )
        assert response.status_code == 400


class TestRollbackEndpoints:
    def test_rollback_last_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True

        async def mock_rollback():
            return mock_result

        srv._state.rollback_last = mock_rollback
        response = client.post("/v1/rollback/last")
        assert response.status_code == 200

    def test_rollback_last_error(self, client):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["No mutations to rollback"]

        async def mock_rollback():
            return mock_result

        srv._state.rollback_last = mock_rollback
        response = client.post("/v1/rollback/last")
        assert response.status_code == 400

    def test_rollback_to_success(self, client):
        mock_result = MagicMock()
        mock_result.success = True

        async def mock_rollback(index):
            return mock_result

        srv._state.rollback_to = mock_rollback
        response = client.post("/v1/rollback/2")
        assert response.status_code == 200

    def test_rollback_to_negative_index(self, client):
        response = client.post("/v1/rollback/-1")
        assert response.status_code == 400


class TestRunEndpoints:
    def test_run_pipeline_success(self, client):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "results": {}}

        async def mock_run(**kwargs):
            return mock_result

        srv._state.run_pipeline = mock_run
        response = client.post("/v1/run/pipeline")
        assert response.status_code == 200
        assert response.json()["data"]["success"] is True

    def test_run_pipeline_with_overrides(self, client):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "results": {}}

        async def mock_run(**kwargs):
            return mock_result

        srv._state.run_pipeline = mock_run
        response = client.post(
            "/v1/run/pipeline",
            json={"overrides": {"key": "value"}},
        )
        assert response.status_code == 200

    def test_run_stage_success(self, client):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"success": True, "attempts": 1}

        async def mock_run(stage_name, context=None):
            return mock_result

        srv._state.run_single_stage = mock_run
        response = client.post("/v1/run/stage/train")
        assert response.status_code == 200

    def test_run_experiment_success(self, client):
        async def mock_run(config):
            return {"success": True, "content": "done"}

        srv._state.run_direct_experiment = mock_run
        response = client.post(
            "/v1/run/experiment",
            json={"config": {"param": "value"}},
        )
        assert response.status_code == 200
        assert response.json()["data"]["success"] is True


class TestMCPEndpoints:
    def test_get_mcp_status(self, client):
        srv._state._describe_mcp = MagicMock(
            return_value={"connected_servers": [], "total_tools": 0}
        )
        response = client.get("/v1/mcp")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["connected_servers"] == []

    def test_disconnect_mcp_server_not_found(self, client):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["MCP server 'x' not connected"]

        async def mock_disconnect(name):
            return mock_result

        srv._state.disconnect_mcp_server = mock_disconnect
        response = client.delete("/v1/mcp/servers/x")
        assert response.status_code == 400


class TestRunHistoryAndMetrics:
    def test_get_run_history_no_persistence(self, client):
        response = client.get("/v1/runs")
        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_get_metrics_no_persistence(self, client):
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_get_leaderboard_no_client(self, client):
        response = client.get("/v1/leaderboard")
        assert response.status_code == 200
        assert response.json()["data"] == []

    def test_get_run_history_invalid_n(self, client):
        response = client.get("/v1/runs?n=0")
        assert response.status_code == 400


class TestBackwardCompatibilityExtended:
    def test_unversioned_agents(self, client):
        response = client.get("/agents")
        assert response.status_code == 200

    def test_unversioned_knowledge(self, client):
        response = client.get("/knowledge")
        assert response.status_code == 200


class TestNotConfigured:
    def test_readiness_not_configured(self):
        srv._state = None
        srv._config = None
        unconfigured_client = TestClient(app)
        response = unconfigured_client.get("/readyz")
        assert response.status_code == 503

    def test_status_not_configured(self):
        srv._state = None
        srv._config = None
        unconfigured_client = TestClient(app)
        response = unconfigured_client.get("/v1/status")
        assert response.status_code == 503


class TestArtifactEndpoints:
    @pytest.fixture(autouse=True)
    def setup_artifact_store(self):
        """Set up a real LocalArtifactStore for artifact tests."""
        import tempfile
        from pathlib import Path

        from ganglion.compute.artifacts import LocalArtifactStore

        self.tmpdir = tempfile.mkdtemp()
        mock_state = make_mock_state()
        mock_state.artifact_store = LocalArtifactStore(root=Path(self.tmpdir))

        async def mock_store_artifact(**kwargs):
            from ganglion.compute.artifacts import ArtifactMeta

            meta = ArtifactMeta(
                key=kwargs["key"],
                run_id=kwargs.get("run_id", ""),
                experiment_id=kwargs.get("experiment_id", ""),
                stage=kwargs.get("stage", ""),
                content_type=kwargs.get("content_type", ""),
            )
            await mock_state.artifact_store.put(kwargs["key"], kwargs["data"], meta)

        mock_state.store_artifact = mock_store_artifact
        srv._state = mock_state
        srv._config = None
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_list_artifacts_empty(self):
        client = TestClient(app)
        response = client.get("/v1/artifacts")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["count"] == 0
        assert data["artifacts"] == []

    def test_store_and_list_artifacts(self):
        client = TestClient(app)
        # Store a text artifact
        response = client.post(
            "/v1/artifacts",
            json={
                "key": "run-1/config.json",
                "content": '{"lr": 0.001}',
                "run_id": "run-1",
                "experiment_id": "exp-42",
                "content_type": "application/json",
            },
        )
        assert response.status_code == 201
        assert response.json()["data"]["key"] == "run-1/config.json"

        # List all
        response = client.get("/v1/artifacts")
        data = response.json()["data"]
        assert data["count"] == 1
        assert data["artifacts"][0]["run_id"] == "run-1"

    def test_store_and_get_artifact(self):
        client = TestClient(app)
        client.post(
            "/v1/artifacts",
            json={
                "key": "run-1/train.py",
                "content": "print('hello')",
                "run_id": "run-1",
            },
        )

        response = client.get("/v1/artifacts/run-1/train.py")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["content"] == "print('hello')"
        assert data["encoding"] == "utf-8"
        assert data["size_bytes"] == len("print('hello')")

    def test_get_artifact_not_found(self):
        client = TestClient(app)
        response = client.get("/v1/artifacts/nonexistent/file.txt")
        assert response.status_code == 404

    def test_store_and_get_base64_artifact(self):
        import base64

        client = TestClient(app)
        binary_data = bytes(range(256))
        b64 = base64.b64encode(binary_data).decode("ascii")

        client.post(
            "/v1/artifacts",
            json={
                "key": "run-1/model.pt",
                "content": b64,
                "encoding": "base64",
                "run_id": "run-1",
                "content_type": "model/pytorch",
            },
        )

        response = client.get("/v1/artifacts/run-1/model.pt?encoding=base64")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["encoding"] == "base64"
        assert base64.b64decode(data["content"]) == binary_data

    def test_list_artifacts_by_run_id(self):
        client = TestClient(app)
        client.post(
            "/v1/artifacts",
            json={"key": "run-1/a.txt", "content": "a", "run_id": "run-1"},
        )
        client.post(
            "/v1/artifacts",
            json={"key": "run-2/b.txt", "content": "b", "run_id": "run-2"},
        )

        response = client.get("/v1/artifacts?run_id=run-1")
        data = response.json()["data"]
        assert data["count"] == 1
        assert data["artifacts"][0]["key"] == "run-1/a.txt"

    def test_list_artifacts_no_store(self):
        srv._state.artifact_store = None
        client = TestClient(app)
        response = client.get("/v1/artifacts")
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["count"] == 0
