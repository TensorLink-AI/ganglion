"""Tests for subnet templates, including the SN50 Synth City template."""

import pytest

from ganglion.templates import (
    GENERIC_TEMPLATE,
    SN50_SYNTH_CITY_TEMPLATE,
    get_template,
)


class TestGetTemplate:
    def test_generic_lookup(self):
        t = get_template("generic")
        assert t is GENERIC_TEMPLATE

    def test_sn50_lookup(self):
        t = get_template("sn50")
        assert t is SN50_SYNTH_CITY_TEMPLATE

    def test_synth_lookup(self):
        t = get_template("synth")
        assert t is SN50_SYNTH_CITY_TEMPLATE

    def test_synth_city_lookup(self):
        t = get_template("synth-city")
        assert t is SN50_SYNTH_CITY_TEMPLATE

    def test_unknown_returns_customized_generic(self):
        t = get_template("my-custom-subnet")
        assert t.name == "my-custom-subnet"
        assert t.slug == "my-custom-subnet"
        assert t.netuid == 0  # generic default


class TestSN50Template:
    def test_identity(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert t.netuid == 50
        assert t.name == "Synth City"
        assert t.slug == "synth-city"

    def test_metrics(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        metric_names = [m["name"] for m in t.metrics]
        assert "crps" in metric_names
        assert "calibration" in metric_names
        assert "sharpness" in metric_names
        # CRPS should be the primary metric (weight=1.0, minimize)
        crps = next(m for m in t.metrics if m["name"] == "crps")
        assert crps["direction"] == "minimize"
        assert crps["weight"] == 1.0

    def test_tasks(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert "btc_forecast" in t.tasks
        assert "eth_forecast" in t.tasks
        assert t.tasks["btc_forecast"]["weight"] == 0.6
        assert t.tasks["eth_forecast"]["weight"] == 0.4

    def test_output_format(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert t.output_format == "price_paths_json"

    def test_starter_agent_name(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert t.starter_agent_name == "Forecaster"

    def test_has_search_strategies(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert len(t.search_strategies) > 0
        # Should mention GBM as a baseline
        assert any("GBM" in s or "gbm" in s.lower() for s in t.search_strategies)

    def test_has_known_pitfalls(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert len(t.known_pitfalls) > 0

    def test_has_domain_context(self):
        t = SN50_SYNTH_CITY_TEMPLATE
        assert "CRPS" in t.domain_context
        assert "Monte Carlo" in t.domain_context


class TestSN50Rendering:
    def test_render_config(self):
        config_code = SN50_SYNTH_CITY_TEMPLATE.render_config()
        assert "netuid=50" in config_code
        assert '"crps"' in config_code
        assert "Forecaster" in config_code
        assert "PipelineDef" in config_code
        # Should be valid Python (no syntax errors)
        compile(config_code, "<config>", "exec")

    def test_render_starter_agent(self):
        agent_code = SN50_SYNTH_CITY_TEMPLATE.render_starter_agent()
        assert "class Forecaster" in agent_code
        assert "BaseAgentWrapper" in agent_code
        assert "build_system_prompt" in agent_code
        assert "build_tools" in agent_code
        compile(agent_code, "<agent>", "exec")

    def test_render_skill_md(self):
        skill_md = SN50_SYNTH_CITY_TEMPLATE.render_skill_md()
        assert "ganglion-synth-city" in skill_md
        assert "netuid 50" in skill_md
        assert "crps" in skill_md.lower()
        assert "Search strategies" in skill_md
        assert "Known pitfalls" in skill_md
        assert "Bootstrap workflow" in skill_md
        # Should contain openclaw metadata
        assert "openclaw" in skill_md

    def test_scaffold(self, tmp_path):
        created = SN50_SYNTH_CITY_TEMPLATE.scaffold(tmp_path / "sn50")
        assert len(created) > 0

        target = tmp_path / "sn50"
        assert (target / "config.py").exists()
        assert (target / "tools" / "run_experiment.py").exists()
        assert (target / "agents" / "forecaster.py").exists()
        assert (target / "skill" / "SKILL.md").exists()

    def test_starter_tool_has_sn50_logic(self):
        tool_code = SN50_SYNTH_CITY_TEMPLATE.starter_tools["run_experiment"]
        assert "run_experiment" in tool_code
        assert "Monte Carlo" in tool_code or "price" in tool_code.lower()
        assert "crps" in tool_code.lower()
        compile(tool_code, "<tool>", "exec")
