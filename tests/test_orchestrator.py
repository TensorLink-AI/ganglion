"""Tests for PipelineOrchestrator."""

import pytest

from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.orchestration.events import PipelineEvent
from ganglion.orchestration.orchestrator import PipelineOrchestrator
from ganglion.orchestration.pipeline import PipelineDef, StageDef, ToolStageDef
from ganglion.orchestration.task_context import (
    MetricDef,
    OutputSpec,
    SubnetConfig,
    TaskContext,
    TaskDef,
)
from ganglion.policies.retry import FixedRetry
from ganglion.runtime.types import AgentResult


def make_config() -> SubnetConfig:
    return SubnetConfig(
        netuid=99,
        name="Test",
        metrics=[MetricDef("accuracy", "maximize")],
        tasks={"main": TaskDef("main")},
        output_spec=OutputSpec(format="test"),
    )


class SuccessAgent(BaseAgentWrapper):
    """Agent that always succeeds."""

    def build_system_prompt(self, task):
        return "Test agent"

    def build_tools(self, task):
        return [], {}

    async def run(self, task):
        task.set("output", "success_data", stage="success_agent")
        return AgentResult(success=True, structured={"result": "ok"}, raw_text="Done")


class FailAgent(BaseAgentWrapper):
    """Agent that always fails."""

    def build_system_prompt(self, task):
        return "Test agent"

    def build_tools(self, task):
        return [], {}

    async def run(self, task):
        return AgentResult(success=False, raw_text="Something went wrong")


class ConditionalAgent(BaseAgentWrapper):
    """Agent that succeeds on the Nth attempt."""

    _attempt_count = 0

    def build_system_prompt(self, task):
        return "Test agent"

    def build_tools(self, task):
        return [], {}

    async def run(self, task):
        ConditionalAgent._attempt_count += 1
        if ConditionalAgent._attempt_count >= 2:
            return AgentResult(success=True, raw_text="Succeeded on retry")
        return AgentResult(success=False, raw_text="Failed first attempt")


@pytest.mark.asyncio
class TestPipelineOrchestrator:
    async def test_single_stage_success(self):
        pipeline = PipelineDef(
            name="test",
            stages=[StageDef(name="step1", agent="SuccessAgent")],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"SuccessAgent": SuccessAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is True
        assert "step1" in result.results
        assert result.results["step1"].success is True

    async def test_single_stage_failure(self):
        pipeline = PipelineDef(
            name="test",
            stages=[StageDef(name="step1", agent="FailAgent")],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"FailAgent": FailAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is False
        assert result.failed_stage == "step1"

    async def test_multi_stage_pipeline(self):
        pipeline = PipelineDef(
            name="test",
            stages=[
                StageDef(name="plan", agent="SuccessAgent", output_keys=["output"]),
                StageDef(
                    name="train",
                    agent="SuccessAgent",
                    depends_on=["plan"],
                    input_keys=["output"],
                ),
            ],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"SuccessAgent": SuccessAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is True
        assert len(result.results) == 2

    async def test_dependency_failure_stops_pipeline(self):
        pipeline = PipelineDef(
            name="test",
            stages=[
                StageDef(name="step1", agent="FailAgent"),
                StageDef(name="step2", agent="SuccessAgent", depends_on=["step1"]),
            ],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"FailAgent": FailAgent, "SuccessAgent": SuccessAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is False
        assert result.failed_stage == "step1"
        # step2 should not have been attempted
        assert "step2" not in result.results

    async def test_optional_stage_skipped(self):
        pipeline = PipelineDef(
            name="test",
            stages=[
                StageDef(name="step1", agent="FailAgent"),
                StageDef(
                    name="step2",
                    agent="SuccessAgent",
                    depends_on=["step1"],
                    is_optional=True,
                ),
                StageDef(name="step3", agent="SuccessAgent"),
            ],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"FailAgent": FailAgent, "SuccessAgent": SuccessAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        # Pipeline fails because step1 is not optional
        assert result.success is False

    async def test_retry_with_fixed_policy(self):
        ConditionalAgent._attempt_count = 0
        pipeline = PipelineDef(
            name="test",
            stages=[
                StageDef(
                    name="step1",
                    agent="ConditionalAgent",
                    retry=FixedRetry(max_attempts=3),
                ),
            ],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"ConditionalAgent": ConditionalAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is True
        assert result.results["step1"].attempts == 2

    async def test_missing_agent(self):
        pipeline = PipelineDef(
            name="test",
            stages=[StageDef(name="step1", agent="NonexistentAgent")],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is False

    async def test_event_emission(self):
        events: list[PipelineEvent] = []

        pipeline = PipelineDef(
            name="test",
            stages=[StageDef(name="step1", agent="SuccessAgent")],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"SuccessAgent": SuccessAgent},
            event_handler=events.append,
        )
        await orchestrator.run(TaskContext(make_config()))

        event_types = [type(e).__name__ for e in events]
        assert "PipelineStarted" in event_types
        assert "StageStarted" in event_types
        assert "StageCompleted" in event_types
        assert "PipelineCompleted" in event_types

    async def test_invalid_pipeline_raises(self):
        pipeline = PipelineDef(
            name="test",
            stages=[
                StageDef(name="a", agent="A", depends_on=["b"]),
                StageDef(name="b", agent="B", depends_on=["a"]),
            ],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"A": SuccessAgent, "B": SuccessAgent},
        )
        from ganglion.orchestration.errors import PipelineValidationError

        with pytest.raises(PipelineValidationError):
            await orchestrator.run(TaskContext(make_config()))

    async def test_tool_stage_success(self):
        async def fetch_data(task):
            task.set("data", {"prices": [100, 200]}, stage="fetch_data")
            return AgentResult(success=True, structured={"prices": [100, 200]}, raw_text="OK")

        pipeline = PipelineDef(
            name="test",
            stages=[ToolStageDef(name="fetch", fn=fetch_data)],
        )
        orchestrator = PipelineOrchestrator(pipeline=pipeline, agents={})
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is True
        assert result.results["fetch"].success is True
        assert result.results["fetch"].attempts == 1

    async def test_tool_stage_failure(self):
        async def bad_fetch(task):
            return AgentResult(success=False, raw_text="Connection timeout")

        pipeline = PipelineDef(
            name="test",
            stages=[ToolStageDef(name="fetch", fn=bad_fetch)],
        )
        orchestrator = PipelineOrchestrator(pipeline=pipeline, agents={})
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is False
        assert result.failed_stage == "fetch"

    async def test_tool_stage_exception(self):
        async def exploding_fetch(task):
            raise RuntimeError("Network error")

        pipeline = PipelineDef(
            name="test",
            stages=[ToolStageDef(name="fetch", fn=exploding_fetch)],
        )
        orchestrator = PipelineOrchestrator(pipeline=pipeline, agents={})
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is False
        assert "Network error" in result.results["fetch"].error

    async def test_mixed_pipeline_tool_then_agent(self):
        async def fetch_data(task):
            task.set("data", {"prices": [100]}, stage="fetch_data")
            return AgentResult(success=True, structured={"prices": [100]}, raw_text="OK")

        pipeline = PipelineDef(
            name="test",
            stages=[
                ToolStageDef(name="fetch", fn=fetch_data, output_keys=["data"]),
                StageDef(
                    name="plan",
                    agent="SuccessAgent",
                    depends_on=["fetch"],
                    input_keys=["data"],
                ),
            ],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={"SuccessAgent": SuccessAgent},
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is True
        assert result.results["fetch"].success is True
        assert result.results["plan"].success is True

    async def test_tool_stage_retry(self):
        call_count = 0

        async def flaky_fetch(task):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return AgentResult(success=False, raw_text="Temporary failure")
            return AgentResult(success=True, raw_text="OK")

        pipeline = PipelineDef(
            name="test",
            stages=[
                ToolStageDef(
                    name="fetch",
                    fn=flaky_fetch,
                    retry=FixedRetry(max_attempts=3),
                ),
            ],
        )
        orchestrator = PipelineOrchestrator(pipeline=pipeline, agents={})
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success is True
        assert result.results["fetch"].attempts == 2

    async def test_tool_stage_events(self):
        events: list[PipelineEvent] = []

        async def fetch(task):
            return AgentResult(success=True, raw_text="OK")

        pipeline = PipelineDef(
            name="test",
            stages=[ToolStageDef(name="fetch", fn=fetch)],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={},
            event_handler=events.append,
        )
        await orchestrator.run(TaskContext(make_config()))

        event_types = [type(e).__name__ for e in events]
        assert "PipelineStarted" in event_types
        assert "StageStarted" in event_types
        assert "StageCompleted" in event_types
        assert "PipelineCompleted" in event_types


class TestOrchestratorArtifactPersistence:
    """Rule 1: Every action leaves a trace — artifacts from stages are persisted."""

    @pytest.mark.asyncio
    async def test_artifacts_in_structured_result_are_stored(self):
        """When a stage returns artifacts in structured output, they get stored."""
        import tempfile
        from pathlib import Path

        from ganglion.compute.artifacts import LocalArtifactStore

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=Path(d))

            async def produce_artifacts(task):
                return AgentResult(
                    success=True,
                    raw_text="trained",
                    structured={"artifacts": {"model.pt": b"weights", "config.json": b"{}"}},
                )

            pipeline = PipelineDef(
                name="train-run",
                stages=[ToolStageDef(name="train", fn=produce_artifacts)],
            )
            orchestrator = PipelineOrchestrator(
                pipeline=pipeline,
                agents={},
                artifact_store=store,
            )
            result = await orchestrator.run(TaskContext(make_config()))
            assert result.success

            keys = await store.list()
            assert len(keys) == 2
            assert "train-run/model.pt" in keys
            assert "train-run/config.json" in keys

            meta = await store.get_meta("train-run/model.pt")
            assert meta is not None
            assert meta.stage == "train"

    @pytest.mark.asyncio
    async def test_no_artifacts_no_error(self):
        """Stages without artifacts don't cause errors."""
        async def simple_stage(task):
            return AgentResult(success=True, raw_text="done", structured={"score": 0.9})

        pipeline = PipelineDef(
            name="test",
            stages=[ToolStageDef(name="eval", fn=simple_stage)],
        )
        orchestrator = PipelineOrchestrator(
            pipeline=pipeline,
            agents={},
            artifact_store=None,
        )
        result = await orchestrator.run(TaskContext(make_config()))
        assert result.success

    @pytest.mark.asyncio
    async def test_source_bot_from_knowledge(self):
        """source_bot is pulled from knowledge.bot_id into artifact metadata."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        from ganglion.compute.artifacts import LocalArtifactStore

        with tempfile.TemporaryDirectory() as d:
            store = LocalArtifactStore(root=Path(d))
            knowledge = MagicMock()
            knowledge.bot_id = "claw-bot-42"
            knowledge.record_success = None
            knowledge.record_failure = None

            async def produce_artifact(task):
                return AgentResult(
                    success=True,
                    raw_text="done",
                    structured={"artifacts": {"output.txt": b"result"}},
                )

            pipeline = PipelineDef(
                name="bot-run",
                stages=[ToolStageDef(name="generate", fn=produce_artifact)],
            )
            orchestrator = PipelineOrchestrator(
                pipeline=pipeline,
                agents={},
                artifact_store=store,
                knowledge=knowledge,
            )
            result = await orchestrator.run(TaskContext(make_config()))
            assert result.success

            meta = await store.get_meta("bot-run/output.txt")
            assert meta is not None
            assert meta.source_bot == "claw-bot-42"
