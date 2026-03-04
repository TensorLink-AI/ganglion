"""Tests for PipelineOrchestrator."""

import pytest

from ganglion.orchestration.orchestrator import PipelineOrchestrator, PipelineResult, StageResult
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import (
    TaskContext,
    SubnetConfig,
    MetricDef,
    TaskDef,
    OutputSpec,
)
from ganglion.orchestration.events import PipelineEvent, StageStarted, StageCompleted
from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.runtime.types import AgentResult
from ganglion.policies.retry import FixedRetry, NoRetry


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
                StageDef(name="train", agent="SuccessAgent", depends_on=["plan"], input_keys=["output"]),
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
                StageDef(name="step2", agent="SuccessAgent", depends_on=["step1"], is_optional=True),
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
