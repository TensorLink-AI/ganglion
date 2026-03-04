"""Tests for the CLI entry point."""

import json

import pytest

from ganglion.__main__ import main


class TestCLI:
    def test_no_command_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_unknown_command_exits(self, capsys):
        with pytest.raises(SystemExit):
            main(["nonexistent"])

    def test_init_refuses_overwrite(self, tmp_path):
        # Create existing config.py
        (tmp_path / "config.py").write_text("# existing")

        with pytest.raises(SystemExit):
            main(["init", str(tmp_path)])

    def test_init_scaffolds(self, tmp_path):
        target = tmp_path / "new-project"
        main(["init", str(target)])
        assert (target / "config.py").exists()

    def test_status_command(self, tmp_path, capsys):
        # Create a minimal project
        (tmp_path / "config.py").write_text("""
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, OutputSpec, TaskDef
from ganglion.orchestration.pipeline import PipelineDef, StageDef

subnet_config = SubnetConfig(
    netuid=0, name="Test", metrics=[MetricDef("acc", "maximize")],
    tasks={"main": TaskDef("main", weight=1.0)},
    output_spec=OutputSpec(format="model", description="test"),
)
pipeline = PipelineDef(name="test", stages=[])
""")

        main(["status", str(tmp_path)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "subnet" in data
        assert "pipeline" in data

    def test_tools_command(self, tmp_path, capsys):
        (tmp_path / "config.py").write_text("""
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, OutputSpec, TaskDef
from ganglion.orchestration.pipeline import PipelineDef, StageDef

subnet_config = SubnetConfig(
    netuid=0, name="Test", metrics=[MetricDef("acc", "maximize")],
    tasks={"main": TaskDef("main", weight=1.0)},
    output_spec=OutputSpec(format="model", description="test"),
)
pipeline = PipelineDef(name="test", stages=[])
""")

        main(["tools", str(tmp_path)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)

    def test_pipeline_command(self, tmp_path, capsys):
        (tmp_path / "config.py").write_text("""
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, OutputSpec, TaskDef
from ganglion.orchestration.pipeline import PipelineDef, StageDef

subnet_config = SubnetConfig(
    netuid=0, name="Test", metrics=[MetricDef("acc", "maximize")],
    tasks={"main": TaskDef("main", weight=1.0)},
    output_spec=OutputSpec(format="model", description="test"),
)
pipeline = PipelineDef(name="test", stages=[])
""")

        main(["pipeline", str(tmp_path)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["name"] == "test"

    def test_run_invalid_overrides(self, tmp_path):
        (tmp_path / "config.py").write_text("""
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, OutputSpec, TaskDef
from ganglion.orchestration.pipeline import PipelineDef, StageDef

subnet_config = SubnetConfig(
    netuid=0, name="Test", metrics=[MetricDef("acc", "maximize")],
    tasks={"main": TaskDef("main", weight=1.0)},
    output_spec=OutputSpec(format="model", description="test"),
)
pipeline = PipelineDef(name="test", stages=[])
""")

        with pytest.raises(SystemExit):
            main(["run", str(tmp_path), "--overrides", "not-json"])
