"""Tests for Layer 3: Orchestration."""

import pytest

from ganglion.orchestration.errors import (
    AgentError,
    EnvironmentError,
    InfrastructureError,
    PipelineOperationError,
    PipelineValidationError,
)
from ganglion.orchestration.events import (
    StageCompleted,
    StageStarted,
)
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import (
    MetricDef,
    OutputSpec,
    SubnetConfig,
    TaskContext,
    TaskDef,
)


def make_config() -> SubnetConfig:
    return SubnetConfig(
        netuid=99,
        name="Test Subnet",
        metrics=[MetricDef("accuracy", "maximize")],
        tasks={"main": TaskDef("main")},
        output_spec=OutputSpec(format="pytorch_model", description="A model"),
    )


class TestSubnetConfig:
    def test_to_prompt_section(self):
        config = make_config()
        section = config.to_prompt_section()
        assert "Test Subnet" in section
        assert "accuracy" in section
        assert "maximize" in section

    def test_to_dict(self):
        config = make_config()
        d = config.to_dict()
        assert d["netuid"] == 99
        assert d["name"] == "Test Subnet"
        assert len(d["metrics"]) == 1

    def test_metric_is_better(self):
        maximize = MetricDef("acc", "maximize")
        assert maximize.is_better(0.9, 0.8) is True
        assert maximize.is_better(0.7, 0.8) is False

        minimize = MetricDef("loss", "minimize")
        assert minimize.is_better(0.1, 0.2) is True
        assert minimize.is_better(0.3, 0.2) is False


class TestTaskContext:
    def test_set_and_get(self):
        ctx = TaskContext(make_config())
        ctx.set("result", {"score": 0.95}, stage="train")
        assert ctx.get("result") == {"score": 0.95}

    def test_get_missing_key_with_default(self):
        ctx = TaskContext(make_config())
        assert ctx.get("missing", "fallback") == "fallback"

    def test_get_missing_key_raises(self):
        ctx = TaskContext(make_config())
        with pytest.raises(KeyError, match="missing"):
            ctx.get("missing")

    def test_has(self):
        ctx = TaskContext(make_config())
        assert ctx.has("x") is False
        ctx.set("x", 1, stage="test")
        assert ctx.has("x") is True

    def test_keys(self):
        ctx = TaskContext(make_config())
        ctx.set("a", 1, stage="s1")
        ctx.set("b", 2, stage="s2")
        assert set(ctx.keys()) == {"a", "b"}

    def test_snapshot(self):
        ctx = TaskContext(make_config())
        ctx.set("x", 42, stage="test", description="A number")
        snap = ctx.snapshot()
        assert snap["x"]["value"] == 42
        assert snap["x"]["meta"]["written_by"] == "test"

    def test_to_agent_context(self):
        ctx = TaskContext(make_config())
        ctx.set("plan", {"steps": ["a", "b"]}, stage="planner")
        text = ctx.to_agent_context(["plan"])
        assert "## plan" in text
        assert "steps" in text

    def test_initial_data(self):
        ctx = TaskContext(make_config(), initial={"seed": 42})
        assert ctx.get("seed") == 42


class TestStageDef:
    def test_to_dict(self):
        s = StageDef(
            name="train",
            agent="Trainer",
            depends_on=["plan"],
            input_keys=["plan"],
            output_keys=["model"],
        )
        d = s.to_dict()
        assert d["name"] == "train"
        assert d["agent"] == "Trainer"
        assert d["depends_on"] == ["plan"]


class TestPipelineDef:
    def test_validate_valid_pipeline(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="plan", agent="Planner", output_keys=["plan"]),
                StageDef(
                    name="train",
                    agent="Trainer",
                    depends_on=["plan"],
                    input_keys=["plan"],
                    output_keys=["model"],
                ),
            ],
        )
        assert p.validate() == []

    def test_validate_duplicate_names(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="dup", agent="A"),
                StageDef(name="dup", agent="B"),
            ],
        )
        errors = p.validate()
        assert any("Duplicate" in e for e in errors)

    def test_validate_missing_dependency(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="train", agent="Trainer", depends_on=["nonexistent"]),
            ],
        )
        errors = p.validate()
        assert any("nonexistent" in e for e in errors)

    def test_validate_cycle(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="a", agent="A", depends_on=["b"]),
                StageDef(name="b", agent="B", depends_on=["a"]),
            ],
        )
        errors = p.validate()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_missing_input_key(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="train", agent="Trainer", input_keys=["plan"]),
            ],
        )
        errors = p.validate()
        assert any("plan" in e for e in errors)

    def test_copy(self):
        p = PipelineDef(
            name="test",
            stages=[StageDef(name="a", agent="A", depends_on=["b"])],
        )
        copy = p.copy()
        assert copy.name == p.name
        assert copy.stages[0].name == "a"
        # Verify it's a deep copy
        copy.stages[0].name = "modified"
        assert p.stages[0].name == "a"

    def test_apply_add_stage(self):
        p = PipelineDef(name="test", stages=[StageDef(name="a", agent="A")])
        p2 = p.apply_operation({"op": "add_stage", "stage": {"name": "b", "agent": "B"}})
        assert len(p2.stages) == 2
        assert p2.stages[1].name == "b"

    def test_apply_add_duplicate_stage(self):
        p = PipelineDef(name="test", stages=[StageDef(name="a", agent="A")])
        with pytest.raises(PipelineOperationError, match="already exists"):
            p.apply_operation({"op": "add_stage", "stage": {"name": "a", "agent": "B"}})

    def test_apply_remove_stage(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="a", agent="A"),
                StageDef(name="b", agent="B", depends_on=["a"]),
            ],
        )
        p2 = p.apply_operation({"op": "remove_stage", "stage_name": "a"})
        assert len(p2.stages) == 1
        # Dependency on "a" should be cleaned up
        assert p2.stages[0].depends_on == []

    def test_apply_update_stage(self):
        p = PipelineDef(
            name="test",
            stages=[StageDef(name="a", agent="A", is_optional=False)],
        )
        p2 = p.apply_operation(
            {"op": "update_stage", "stage_name": "a", "updates": {"is_optional": True}}
        )
        assert p2.stages[0].is_optional is True

    def test_apply_update_invalid_field(self):
        p = PipelineDef(name="test", stages=[StageDef(name="a", agent="A")])
        with pytest.raises(PipelineOperationError, match="no field"):
            p.apply_operation(
                {"op": "update_stage", "stage_name": "a", "updates": {"nonexistent": True}}
            )

    def test_apply_unknown_operation(self):
        p = PipelineDef(name="test", stages=[])
        with pytest.raises(PipelineOperationError, match="Unknown"):
            p.apply_operation({"op": "invalid_op"})

    def test_get_stage(self):
        p = PipelineDef(
            name="test",
            stages=[StageDef(name="a", agent="A"), StageDef(name="b", agent="B")],
        )
        assert p.get_stage("a") is not None
        assert p.get_stage("a").agent == "A"
        assert p.get_stage("nonexistent") is None

    def test_topological_order(self):
        p = PipelineDef(
            name="test",
            stages=[
                StageDef(name="c", agent="C", depends_on=["a", "b"]),
                StageDef(name="a", agent="A"),
                StageDef(name="b", agent="B", depends_on=["a"]),
            ],
        )
        order = p._topological_order()
        names = [s.name for s in order]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("c")

    def test_to_dict(self):
        p = PipelineDef(
            name="test",
            stages=[StageDef(name="a", agent="A")],
        )
        d = p.to_dict()
        assert d["name"] == "test"
        assert len(d["stages"]) == 1


class TestEvents:
    def test_stage_started(self):
        e = StageStarted(stage="train")
        assert e.stage == "train"
        assert e.timestamp is not None

    def test_stage_completed(self):
        e = StageCompleted(stage="train", result="success")
        assert e.result == "success"


class TestErrors:
    def test_retryable_flags(self):
        assert AgentError.is_retryable is True
        assert EnvironmentError.is_retryable is False
        assert InfrastructureError.is_retryable is True

    def test_error_hierarchy(self):
        assert issubclass(EnvironmentError, AgentError)
        assert issubclass(PipelineValidationError, AgentError)
