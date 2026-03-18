"""Tests for FrameworkState."""

import tempfile
from pathlib import Path

import pytest

from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import (
    MetricDef,
    OutputSpec,
    SubnetConfig,
    TaskDef,
)
from ganglion.policies.retry import FixedRetry, NoRetry
from ganglion.state.agent_registry import AgentRegistry
from ganglion.state.framework_state import FrameworkState
from ganglion.state.tool_registry import ToolRegistry


def make_config() -> SubnetConfig:
    return SubnetConfig(
        netuid=99,
        name="Test",
        metrics=[MetricDef("accuracy", "maximize")],
        tasks={"main": TaskDef("main")},
        output_spec=OutputSpec(format="test"),
    )


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def state(tmp_dir):
    from ganglion.composition.base_agent import BaseAgentWrapper

    pipeline = PipelineDef(
        name="test",
        stages=[StageDef(name="train", agent="Trainer")],
    )
    agent_registry = AgentRegistry()
    agent_registry.register("Trainer", BaseAgentWrapper)
    return FrameworkState(
        subnet_config=make_config(),
        pipeline_def=pipeline,
        tool_registry=ToolRegistry(),
        agent_registry=agent_registry,
        project_root=tmp_dir,
    )


class TestFrameworkState:
    @pytest.mark.asyncio
    async def test_describe(self, state):
        desc = await state.describe()
        assert desc["subnet"]["name"] == "Test"
        assert desc["running"] is False
        assert desc["mutations"] == 0

    @pytest.mark.asyncio
    async def test_write_and_register_tool(self, state):
        code = '''
from ganglion.composition.tool_registry import tool

@tool("my_tool")
def my_tool(x: int) -> str:
    """A test tool."""
    return str(x)
'''
        result = await state.write_and_register_tool("my_tool", code, "general")
        assert result.success is True
        assert state.tool_registry.has("my_tool")
        assert len(state.mutations) == 1

    @pytest.mark.asyncio
    async def test_write_tool_invalid_code(self, state):
        code = "def broken(:"
        result = await state.write_and_register_tool("bad_tool", code, "general")
        assert result.success is False
        assert not state.tool_registry.has("bad_tool")

    @pytest.mark.asyncio
    async def test_write_tool_missing_decorator(self, state):
        code = '''
def my_tool(x: int) -> str:
    """No decorator."""
    return str(x)
'''
        result = await state.write_and_register_tool("my_tool", code, "general")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_apply_pipeline_patch_add(self, state):
        from ganglion.composition.base_agent import BaseAgentWrapper

        state.agent_registry.register("NewAgent", BaseAgentWrapper)

        result = await state.apply_pipeline_patch(
            [
                {"op": "add_stage", "stage": {"name": "new_stage", "agent": "NewAgent"}},
            ]
        )
        assert result.success is True
        assert state.pipeline_def.get_stage("new_stage") is not None

    @pytest.mark.asyncio
    async def test_apply_pipeline_patch_invalid(self, state):
        result = await state.apply_pipeline_patch(
            [
                {"op": "add_stage", "stage": {"name": "bad", "agent": "NonexistentAgent"}},
            ]
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_swap_policy(self, state):
        new_policy = FixedRetry(max_attempts=5)
        result = await state.swap_policy("train", new_policy)
        assert result.success is True
        stage = state.pipeline_def.get_stage("train")
        assert isinstance(stage.retry, FixedRetry)

    @pytest.mark.asyncio
    async def test_swap_policy_default(self, state):
        new_policy = FixedRetry(max_attempts=3)
        result = await state.swap_policy(None, new_policy)
        assert result.success is True
        assert isinstance(state.pipeline_def.default_retry, FixedRetry)

    @pytest.mark.asyncio
    async def test_swap_policy_missing_stage(self, state):
        result = await state.swap_policy("nonexistent", NoRetry())
        assert result.success is False

    @pytest.mark.asyncio
    async def test_rollback_last(self, state):
        from ganglion.composition.base_agent import BaseAgentWrapper

        state.agent_registry.register("NewAgent", BaseAgentWrapper)

        await state.apply_pipeline_patch(
            [
                {"op": "add_stage", "stage": {"name": "added", "agent": "NewAgent"}},
            ]
        )
        assert state.pipeline_def.get_stage("added") is not None

        result = await state.rollback_last()
        assert result.success is True
        assert state.pipeline_def.get_stage("added") is None

    @pytest.mark.asyncio
    async def test_rollback_empty(self, state):
        result = await state.rollback_last()
        assert result.success is False

    def test_create_factory(self, tmp_dir):
        state = FrameworkState.create(
            subnet_config=make_config(),
            pipeline_def=PipelineDef(name="test", stages=[]),
            project_root=tmp_dir,
        )
        assert state.subnet_config.name == "Test"
        assert len(state.pipeline_def.stages) == 0

    def test_load_from_project_dir(self, tmp_dir):
        """Test FrameworkState.load() discovers config, tools, and agents."""
        # Write a config.py
        config_code = """
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, TaskDef, OutputSpec
from ganglion.orchestration.pipeline import PipelineDef, StageDef

subnet_config = SubnetConfig(
    netuid=50,
    name="LoadTest",
    metrics=[MetricDef("crps", "minimize")],
    tasks={"btc": TaskDef("btc")},
    output_spec=OutputSpec(format="pytorch_tensor"),
)

pipeline = PipelineDef(
    name="load-test",
    stages=[
        StageDef(name="train", agent="LoadTrainer"),
    ],
)
"""
        (tmp_dir / "config.py").write_text(config_code)

        # Write a tool
        tools_dir = tmp_dir / "tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text('''
from ganglion.composition.tool_registry import tool

@tool("load_test_tool")
def load_test_tool(x: int) -> str:
    """A tool discovered by load()."""
    return str(x)
''')

        # Write an agent
        agents_dir = tmp_dir / "agents"
        agents_dir.mkdir()
        (agents_dir / "trainer.py").write_text("""
from ganglion.composition.base_agent import BaseAgentWrapper

class LoadTrainer(BaseAgentWrapper):
    def build_system_prompt(self, task):
        return "test"

    def build_tools(self, task):
        return [], {}
""")

        state = FrameworkState.load(tmp_dir)
        assert state.subnet_config.name == "LoadTest"
        assert state.subnet_config.netuid == 50
        assert state.pipeline_def.name == "load-test"
        assert state.tool_registry.has("load_test_tool")
        assert state.agent_registry.has("LoadTrainer")

    def test_load_missing_config(self, tmp_dir):
        """load() raises FileNotFoundError when config.py is missing."""
        with pytest.raises(FileNotFoundError):
            FrameworkState.load(tmp_dir)

    @pytest.mark.asyncio
    async def test_update_prompt(self, state):
        result = await state.update_prompt("trainer", "role", "You are a trainer.")
        assert result.success is True
        assert len(state.mutations) == 1
        assert state.mutations[0].mutation_type == "write_prompt"

    @pytest.mark.asyncio
    async def test_update_prompt_replace_existing(self, state):
        """Test updating a prompt section that already exists."""
        await state.update_prompt("trainer", "role", "Original role.")
        result = await state.update_prompt("trainer", "role", "Updated role.")
        assert result.success is True
        assert len(state.mutations) == 2
        # Read the file and verify the section was replaced
        path = state.project_root / "prompts" / "trainer.py"
        content = path.read_text()
        assert "Updated role." in content

    @pytest.mark.asyncio
    async def test_write_and_register_agent(self, state):
        code = '''
from ganglion.composition.base_agent import BaseAgentWrapper

class MyAgent(BaseAgentWrapper):
    """A test agent."""

    def build_system_prompt(self, task):
        return "test prompt"

    def build_tools(self, task):
        return [], {}
'''
        result = await state.write_and_register_agent("MyAgent", code)
        assert result.success is True
        assert len(state.mutations) == 1
        assert state.mutations[0].mutation_type == "write_agent"

    @pytest.mark.asyncio
    async def test_write_agent_invalid_code(self, state):
        code = "def not_a_class(): pass"
        result = await state.write_and_register_agent("BadAgent", code)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_rollback_to(self, state):
        from ganglion.composition.base_agent import BaseAgentWrapper

        state.agent_registry.register("A1", BaseAgentWrapper)
        state.agent_registry.register("A2", BaseAgentWrapper)

        await state.apply_pipeline_patch(
            [{"op": "add_stage", "stage": {"name": "s1", "agent": "A1"}}]
        )
        await state.apply_pipeline_patch(
            [{"op": "add_stage", "stage": {"name": "s2", "agent": "A2"}}]
        )
        assert len(state.mutations) == 2

        result = await state.rollback_to(1)
        assert result.success is True
        assert len(state.mutations) == 1
        assert state.pipeline_def.get_stage("s2") is None

    @pytest.mark.asyncio
    async def test_rollback_to_zero(self, state):
        from ganglion.composition.base_agent import BaseAgentWrapper

        state.agent_registry.register("A1", BaseAgentWrapper)
        await state.apply_pipeline_patch(
            [{"op": "add_stage", "stage": {"name": "s1", "agent": "A1"}}]
        )
        result = await state.rollback_to(0)
        assert result.success is True
        assert len(state.mutations) == 0

    @pytest.mark.asyncio
    async def test_run_direct_experiment_no_tool(self, state):
        result = await state.run_direct_experiment({"param": "value"})
        assert result["success"] is False
        assert "No 'run_experiment' tool registered" in result["error"]

    @pytest.mark.asyncio
    async def test_run_direct_experiment_success(self, state):
        from ganglion.composition.tool_returns import ToolOutput

        def run_experiment(**kwargs):
            return ToolOutput(content="experiment done")

        state.tool_registry.register(
            name="run_experiment",
            func=run_experiment,
            description="Run an experiment",
            parameters_schema={"type": "object", "properties": {}},
        )
        result = await state.run_direct_experiment({})
        assert result["success"] is True
        assert result["content"] == "experiment done"

    @pytest.mark.asyncio
    async def test_run_direct_experiment_plain_result(self, state):
        state.tool_registry.register(
            name="run_experiment",
            func=lambda **kwargs: "plain result",
            description="Run an experiment",
            parameters_schema={"type": "object", "properties": {}},
        )
        result = await state.run_direct_experiment({})
        assert result["success"] is True
        assert result["content"] == "plain result"

    @pytest.mark.asyncio
    async def test_run_direct_experiment_error(self, state):
        def run_experiment(**kwargs):
            raise ValueError("experiment failed")

        state.tool_registry.register(
            name="run_experiment",
            func=run_experiment,
            description="Run an experiment",
            parameters_schema={"type": "object", "properties": {}},
        )
        result = await state.run_direct_experiment({})
        assert result["success"] is False
        assert "experiment failed" in result["error"]

    def test_run_test_success(self, state):
        result = state._run_test("assert 1 + 1 == 2")
        assert result.is_passed is True

    def test_run_test_assertion_error(self, state):
        result = state._run_test("assert False, 'failed'")
        assert result.is_passed is False
        assert any("failed" in e for e in result.errors)

    def test_run_test_unexpected_error(self, state):
        result = state._run_test("raise RuntimeError('boom')")
        assert result.is_passed is False
        assert any("boom" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_rollback_write_tool(self, state):
        """Test rollback of a write_tool mutation restores previous state."""
        code = '''
from ganglion.composition.tool_registry import tool

@tool("rollback_tool")
def rollback_tool(x: int) -> str:
    """A test tool."""
    return str(x)
'''
        result = await state.write_and_register_tool("rollback_tool", code, "general")
        assert result.success is True
        assert state.tool_registry.has("rollback_tool")

        rollback = await state.rollback_last()
        assert rollback.success is True
        assert not state.tool_registry.has("rollback_tool")

    @pytest.mark.asyncio
    async def test_rollback_write_agent(self, state):
        """Test rollback of a write_agent mutation."""
        code = '''
from ganglion.composition.base_agent import BaseAgentWrapper

class RollbackAgent(BaseAgentWrapper):
    """A test agent."""
    def build_system_prompt(self, task):
        return "test"
    def build_tools(self, task):
        return [], {}
'''
        result = await state.write_and_register_agent("RollbackAgent", code)
        assert result.success is True

        rollback = await state.rollback_last()
        assert rollback.success is True

    @pytest.mark.asyncio
    async def test_rollback_swap_policy(self, state):
        """Test rollback of a swap_policy mutation."""
        original_retry = state.pipeline_def.get_stage("train").retry
        new_policy = FixedRetry(max_attempts=10)
        await state.swap_policy("train", new_policy)
        assert isinstance(state.pipeline_def.get_stage("train").retry, FixedRetry)

        rollback = await state.rollback_last()
        assert rollback.success is True
        # Policy should be restored
        assert state.pipeline_def.get_stage("train").retry is original_retry

    @pytest.mark.asyncio
    async def test_rollback_swap_default_policy(self, state):
        """Test rollback of a default swap_policy mutation."""
        original = state.pipeline_def.default_retry
        await state.swap_policy(None, FixedRetry(max_attempts=7))
        rollback = await state.rollback_last()
        assert rollback.success is True
        assert state.pipeline_def.default_retry is original

    @pytest.mark.asyncio
    async def test_write_tool_with_test(self, state):
        """Test writing a tool with test_code that passes."""
        code = '''
from ganglion.composition.tool_registry import tool

@tool("tested_tool")
def tested_tool(x: int) -> str:
    """A tested tool."""
    return str(x)
'''
        test_code = "assert 1 + 1 == 2"
        result = await state.write_and_register_tool("tested_tool", code, "general", test_code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_write_tool_with_failing_test(self, state):
        """Test writing a tool with test_code that fails."""
        code = '''
from ganglion.composition.tool_registry import tool

@tool("bad_tested_tool")
def bad_tested_tool(x: int) -> str:
    """A tool with a failing test."""
    return str(x)
'''
        test_code = "assert False, 'test should fail'"
        result = await state.write_and_register_tool("bad_tested_tool", code, "general", test_code)
        assert result.success is False
        assert any("Test failed" in e for e in result.errors)


class TestArtifactSourceBot:
    """Tests for source_bot auto-fill in store_artifact (Rule 2 compliance)."""

    @pytest.mark.asyncio
    async def test_source_bot_auto_filled_from_knowledge(self, tmp_dir):
        """store_artifact fills source_bot from knowledge.bot_id."""
        from unittest.mock import AsyncMock, MagicMock

        from ganglion.compute.artifacts import LocalArtifactStore

        store = LocalArtifactStore(root=tmp_dir / "artifacts")
        knowledge = MagicMock()
        knowledge.bot_id = "claw-bot-1"

        fs = FrameworkState(
            subnet_config=make_config(),
            pipeline_def=PipelineDef(name="test", stages=[]),
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            project_root=tmp_dir,
            knowledge=knowledge,
            artifact_store=store,
        )

        await fs.store_artifact(
            key="run-1/model.pt",
            data=b"fake weights",
            run_id="run-1",
        )

        meta = await store.get_meta("run-1/model.pt")
        assert meta is not None
        assert meta.source_bot == "claw-bot-1"

    @pytest.mark.asyncio
    async def test_source_bot_explicit_overrides_knowledge(self, tmp_dir):
        """Explicit source_bot takes precedence over knowledge.bot_id."""
        from unittest.mock import MagicMock

        from ganglion.compute.artifacts import LocalArtifactStore

        store = LocalArtifactStore(root=tmp_dir / "artifacts")
        knowledge = MagicMock()
        knowledge.bot_id = "claw-bot-1"

        fs = FrameworkState(
            subnet_config=make_config(),
            pipeline_def=PipelineDef(name="test", stages=[]),
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            project_root=tmp_dir,
            knowledge=knowledge,
            artifact_store=store,
        )

        await fs.store_artifact(
            key="run-1/model.pt",
            data=b"fake weights",
            run_id="run-1",
            source_bot="other-bot",
        )

        meta = await store.get_meta("run-1/model.pt")
        assert meta is not None
        assert meta.source_bot == "other-bot"

    @pytest.mark.asyncio
    async def test_source_bot_none_without_knowledge(self, tmp_dir):
        """source_bot is None when no knowledge store is configured."""
        from ganglion.compute.artifacts import LocalArtifactStore

        store = LocalArtifactStore(root=tmp_dir / "artifacts")

        fs = FrameworkState(
            subnet_config=make_config(),
            pipeline_def=PipelineDef(name="test", stages=[]),
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            project_root=tmp_dir,
            artifact_store=store,
        )

        await fs.store_artifact(
            key="run-1/model.pt",
            data=b"fake weights",
            run_id="run-1",
        )

        meta = await store.get_meta("run-1/model.pt")
        assert meta is not None
        assert meta.source_bot is None
