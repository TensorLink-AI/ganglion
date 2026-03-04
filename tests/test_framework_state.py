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

        result = await state.apply_pipeline_patch([
            {"op": "add_stage", "stage": {"name": "new_stage", "agent": "NewAgent"}},
        ])
        assert result.success is True
        assert state.pipeline_def.get_stage("new_stage") is not None

    @pytest.mark.asyncio
    async def test_apply_pipeline_patch_invalid(self, state):
        result = await state.apply_pipeline_patch([
            {"op": "add_stage", "stage": {"name": "bad", "agent": "NonexistentAgent"}},
        ])
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

        await state.apply_pipeline_patch([
            {"op": "add_stage", "stage": {"name": "added", "agent": "NewAgent"}},
        ])
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
        config_code = '''
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
'''
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
        (agents_dir / "trainer.py").write_text('''
from ganglion.composition.base_agent import BaseAgentWrapper

class LoadTrainer(BaseAgentWrapper):
    def build_system_prompt(self, task):
        return "test"

    def build_tools(self, task):
        return [], {}
''')

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
