"""Project scaffolding templates for `ganglion init`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SubnetTemplate:
    """Everything needed to scaffold a subnet project directory."""

    # Identity
    netuid: int
    name: str
    slug: str  # filesystem-safe name, e.g. "sn9-pretrain"

    # SubnetConfig fields
    metrics: list[dict[str, Any]]
    tasks: dict[str, dict[str, Any]]
    output_format: str
    output_description: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)

    # Starter code
    starter_tools: dict[str, str] = field(default_factory=dict)  # name -> code
    starter_agent_code: str = ""
    starter_agent_name: str = "Explorer"

    # Domain knowledge for the Claw Hub skill
    domain_context: str = ""
    search_strategies: list[str] = field(default_factory=list)
    known_pitfalls: list[str] = field(default_factory=list)

    def render_config(self) -> str:
        """Render config.py for the project."""
        metrics_lines = []
        for m in self.metrics:
            metrics_lines.append(
                f'    MetricDef(name="{m["name"]}", direction="{m["direction"]}", '
                f'weight={m.get("weight", 1.0)}, description="{m.get("description", "")}"),'
            )

        tasks_lines = []
        for tname, tdef in self.tasks.items():
            meta = tdef.get("metadata", {})
            meta_str = repr(meta) if meta else "{}"
            tasks_lines.append(
                f'    "{tname}": TaskDef(name="{tname}",'
                f" weight={tdef.get('weight', 1.0)}, metadata={meta_str}),"
            )

        constraints_str = repr(self.constraints) if self.constraints else "{}"

        return f'''"""Subnet configuration for {self.name} (netuid {self.netuid})."""

from ganglion.orchestration.task_context import MetricDef, OutputSpec, SubnetConfig, TaskDef
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.policies.retry import EscalatingRetry, FixedRetry

subnet_config = SubnetConfig(
    netuid={self.netuid},
    name="{self.name}",
    metrics=[
{chr(10).join(metrics_lines)}
    ],
    tasks={{
{chr(10).join(tasks_lines)}
    }},
    output_spec=OutputSpec(
        format="{self.output_format}",
        description="{self.output_description}",
    ),
    constraints={constraints_str},
)

pipeline = PipelineDef(
    name="{self.slug}-pipeline",
    stages=[
        StageDef(
            name="plan",
            agent="{self.starter_agent_name}",
            output_keys=["plan"],
            retry=FixedRetry(max_attempts=2),
        ),
        StageDef(
            name="experiment",
            agent="{self.starter_agent_name}",
            depends_on=["plan"],
            input_keys=["plan"],
            output_keys=["experiment_result"],
            retry=EscalatingRetry(max_attempts=3, base_temp=0.2, temp_step=0.15),
        ),
        StageDef(
            name="evaluate",
            agent="{self.starter_agent_name}",
            depends_on=["experiment"],
            input_keys=["experiment_result"],
            output_keys=["evaluation"],
        ),
    ],
)
'''

    def render_starter_agent(self) -> str:
        """Render the starter agent class."""
        return f'''"""Starter agent for {self.name}."""

from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.composition.tool_registry import build_toolset


class {self.starter_agent_name}(BaseAgentWrapper):
    """Bootstrap agent that plans and runs experiments for {self.name}."""

    def build_system_prompt(self, task):
        subnet_info = task.subnet_config.to_prompt_section()
        return f"""You are a mining agent for the {{task.subnet_config.name}} Bittensor subnet.

{{subnet_info}}

Your job is to search for configurations and approaches that maximize the
subnet's scoring metrics. Use the tools available to run experiments,
analyze results, and iterate toward better scores.

When you have a result, call finish(success=true, result={{...}}, summary="...").
If you cannot make progress, call finish(success=false, summary="...").
"""

    def build_tools(self, task):
        return build_toolset("run_experiment", "finish")
'''

    def render_starter_tool(self, name: str, code: str) -> str:
        """Return the tool code (already complete)."""
        return code

    def _render_constraints(self) -> str:
        """Render constraints as markdown list."""
        if not self.constraints:
            return "None specified."
        return "\n".join(f"- **{k}**: {v}" for k, v in self.constraints.items())

    def _render_metrics_list(self) -> str:
        """Render metrics as markdown list."""
        lines = []
        for m in self.metrics:
            weight = m.get("weight", 1.0)
            desc = m.get("description", "")
            lines.append(f"- **{m['name']}** ({m['direction']}, weight={weight}): {desc}")
        return "\n".join(lines)

    def render_skill_md(self) -> str:
        """Render a subnet-specific SKILL.md for Claw Hub."""
        strategies_section = ""
        if self.search_strategies:
            items = "\n".join(f"- {s}" for s in self.search_strategies)
            strategies_section = f"\n## Search strategies\n\n{items}\n"

        pitfalls_section = ""
        if self.known_pitfalls:
            items = "\n".join(f"- {p}" for p in self.known_pitfalls)
            pitfalls_section = f"\n## Known pitfalls\n\n{items}\n"

        desc = (
            f"Domain knowledge and bootstrap strategies"
            f" for mining {self.name} (netuid {self.netuid})"
            f" with Ganglion."
        )
        return f"""---
name: ganglion-{self.slug}
description: {desc}
homepage: https://github.com/TensorLink-AI/ganglion
metadata: {{"openclaw":{{"emoji":"\\u26d3","requires":{{"anyBins":["ganglion","curl"]}}}}}}
---

# {self.name} (netuid {self.netuid})

{self.domain_context}

This skill provides domain knowledge. Use the `ganglion` skill for API commands.
When `GANGLION_PROJECT` is set (or `GANGLION_URL` is not set), use local CLI commands.
When `GANGLION_URL` is set, use curl against the HTTP bridge (all endpoints are under `/v1/`).
Responses use a standard envelope: `{{"data": <payload>}}` on success.

## Metrics

The validator scores miners on:

{self._render_metrics_list()}

## Output format

{self.output_format}: {self.output_description}
{strategies_section}{pitfalls_section}
## Bootstrap workflow

Local mode (same machine as OpenClaw):

1. `export GANGLION_PROJECT=./{self.slug}`
2. `ganglion status $GANGLION_PROJECT` — check state
3. `ganglion knowledge $GANGLION_PROJECT` — review prior knowledge
4. `ganglion tools $GANGLION_PROJECT` — review starter tools
5. `ganglion run $GANGLION_PROJECT` — run the pipeline
6. Iterate: edit tools/agents in the project directory and re-run

Remote mode (separate server):

1. `ganglion serve ./{self.slug} --bot-id {{{{bot_id}}}} --port 8899`
2. `export GANGLION_URL=http://127.0.0.1:8899`
3. Verify readiness: `curl -s "$GANGLION_URL/readyz" | jq`
4. Use `/v1/` curl commands from the `ganglion` skill

## Constraints

{self._render_constraints()}
"""

    def scaffold(self, target: Path) -> list[str]:
        """Write all files to the target directory. Returns list of created paths."""
        created: list[str] = []
        target.mkdir(parents=True, exist_ok=True)

        # config.py
        config_path = target / "config.py"
        config_path.write_text(self.render_config())
        created.append(str(config_path))

        # tools/
        tools_dir = target / "tools"
        tools_dir.mkdir(exist_ok=True)
        for tool_name, tool_code in self.starter_tools.items():
            tool_path = tools_dir / f"{tool_name}.py"
            tool_path.write_text(tool_code)
            created.append(str(tool_path))

        # agents/
        agents_dir = target / "agents"
        agents_dir.mkdir(exist_ok=True)
        agent_path = agents_dir / f"{self.starter_agent_name.lower()}.py"
        agent_path.write_text(self.render_starter_agent())
        created.append(str(agent_path))

        # skill/ (subnet-specific Claw Hub skill)
        skill_dir = target / "skill"
        skill_dir.mkdir(exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(self.render_skill_md())
        created.append(str(skill_path))

        return created


# ── Built-in templates ─────────────────────────────────────────

_GENERIC_RUN_EXPERIMENT = '''\
"""Generic experiment runner — replace with subnet-specific logic."""

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult


@tool("run_experiment", category="training")
def run_experiment(config: dict) -> ExperimentResult:
    """Run a training experiment with the given config.

    Replace this with actual training/inference logic for your subnet.
    """
    # TODO: Replace with real experiment logic
    return ExperimentResult(
        content=f"Experiment completed with config: {config}",
        experiment_id="exp-001",
        metrics={"score": 0.0},
    )
'''

GENERIC_TEMPLATE = SubnetTemplate(
    netuid=0,
    name="My Subnet",
    slug="my-subnet",
    metrics=[
        {
            "name": "score",
            "direction": "maximize",
            "weight": 1.0,
            "description": "Primary scoring metric",
        },
    ],
    tasks={
        "default": {"weight": 1.0},
    },
    output_format="model_weights",
    output_description="Serialized model weights or predictions",
    starter_tools={"run_experiment": _GENERIC_RUN_EXPERIMENT},
    starter_agent_name="Explorer",
    domain_context="This is a generic template. Replace with subnet-specific domain knowledge.",
    search_strategies=[
        "Start with small experiments to map the search space",
        "Use knowledge from prior runs to guide new experiments",
        "Vary one parameter at a time to understand sensitivity",
    ],
    known_pitfalls=[
        "Running expensive experiments before understanding the scoring function",
        "Ignoring knowledge from failed runs",
    ],
)


_SN50_RUN_EXPERIMENT = '''\
"""Price-path forecasting experiment for Synth (SN50).

Generates Monte Carlo simulated price paths and evaluates them
against the validator's CRPS scoring function.
"""

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult


@tool("run_experiment", category="training")
def run_experiment(config: dict) -> ExperimentResult:
    """Run a price-path forecasting experiment.

    Expected config keys:
        asset: str          — target asset ("BTC", "ETH")
        model_type: str     — forecasting model ("gbm", "jump_diffusion", "regime_switching", "neural_sde")
        n_paths: int        — number of Monte Carlo paths to simulate (default 1000)
        horizon_hours: int  — forecast horizon in hours (default 24)
        volatility_model: str — vol estimator ("ewma", "garch", "realized")
        drift_estimator: str  — drift method ("historical", "risk_neutral", "ml_predicted")
    """
    asset = config.get("asset", "BTC")
    model_type = config.get("model_type", "gbm")
    n_paths = config.get("n_paths", 1000)
    horizon = config.get("horizon_hours", 24)

    # TODO: Replace with real price-path generation logic
    # This stub returns a placeholder result
    return ExperimentResult(
        content=(
            f"Generated {n_paths} price paths for {asset} "
            f"using {model_type} model over {horizon}h horizon. "
            f"CRPS: 0.0 (placeholder)"
        ),
        experiment_id=f"sn50-{asset.lower()}-{model_type}-001",
        metrics={"crps": 0.0, "calibration": 0.0, "sharpness": 0.0},
        artifact_path=None,
    )
'''

SN50_SYNTH_CITY_TEMPLATE = SubnetTemplate(
    netuid=50,
    name="Synth City",
    slug="synth-city",
    metrics=[
        {
            "name": "crps",
            "direction": "minimize",
            "weight": 1.0,
            "description": (
                "Continuous Ranked Probability Score — measures quality of the "
                "full predicted probability distribution against realized prices. "
                "Lower is better."
            ),
        },
        {
            "name": "calibration",
            "direction": "minimize",
            "weight": 0.3,
            "description": (
                "How well the predicted quantiles match observed frequencies. "
                "Perfect calibration = 0."
            ),
        },
        {
            "name": "sharpness",
            "direction": "minimize",
            "weight": 0.2,
            "description": (
                "Width of prediction intervals — sharper (narrower) distributions "
                "score better, but only if well-calibrated."
            ),
        },
    ],
    tasks={
        "btc_forecast": {
            "weight": 0.6,
            "metadata": {"asset": "BTC", "horizon_hours": 24},
        },
        "eth_forecast": {
            "weight": 0.4,
            "metadata": {"asset": "ETH", "horizon_hours": 24},
        },
    },
    output_format="price_paths_json",
    output_description=(
        "JSON array of simulated price paths. Each path is a list of "
        "(timestamp, price) pairs. The validator evaluates the full "
        "distribution of paths against realized prices using CRPS."
    ),
    constraints={
        "min_paths": "100 paths per submission",
        "max_paths": "10000 paths per submission",
        "horizon": "24 hours from submission time",
        "assets": "BTC and ETH (weighted 60/40)",
        "update_cadence": "Validators request new forecasts every ~30 minutes",
    },
    starter_tools={"run_experiment": _SN50_RUN_EXPERIMENT},
    starter_agent_name="Forecaster",
    domain_context=(
        "Synth (SN50) is a probabilistic price forecasting subnet on Bittensor. "
        "Miners generate Monte Carlo simulated price paths for crypto assets "
        "(BTC, ETH). The validator scores submissions using CRPS — a proper "
        "scoring rule that evaluates the full predicted distribution, not just "
        "point estimates. Good miners produce distributions that are both "
        "well-calibrated (quantiles match observed frequencies) and sharp "
        "(tight intervals around the realized price).\n\n"
        "The competitive dynamic: miners who simply widen their distributions "
        "to cover all outcomes get poor sharpness scores. Miners who make "
        "overconfident narrow predictions get poor calibration scores. The "
        "winning strategy balances both — capturing real volatility structure "
        "without unnecessary uncertainty."
    ),
    search_strategies=[
        "Start with Geometric Brownian Motion (GBM) as baseline — it's fast and establishes a CRPS floor",
        "Try jump-diffusion models (Merton) to capture sudden price moves that GBM misses",
        "Experiment with regime-switching models (bull/bear/sideways) to adapt volatility",
        "Use GARCH or EWMA for volatility estimation instead of constant vol assumptions",
        "Neural SDEs can learn drift/diffusion functions from data but need careful regularization",
        "Ensemble multiple model types — blend GBM paths with jump-diffusion paths",
        "Calibrate on recent data windows (7-30 days) rather than long historical periods",
        "Test different path counts: 500 paths is fast for screening, 5000+ for final submissions",
    ],
    known_pitfalls=[
        "Constant volatility assumption (flat vol GBM) fails during high-volatility regimes",
        "Overfitting to recent price action produces narrow distributions that fail on regime changes",
        "Too few paths (< 200) leads to noisy CRPS estimates — score variance masks real improvements",
        "Ignoring overnight/weekend vol patterns degrades BTC forecasts (24/7 market has structure)",
        "Neural SDE training is unstable with small datasets — use at least 90 days of minute-level data",
        "Submitting the same distribution for BTC and ETH ignores correlation structure",
    ],
)


def get_template(subnet: str) -> SubnetTemplate:
    """Look up a built-in template by name or return a customized generic."""
    registry: dict[str, SubnetTemplate] = {
        "generic": GENERIC_TEMPLATE,
        "sn50": SN50_SYNTH_CITY_TEMPLATE,
        "synth": SN50_SYNTH_CITY_TEMPLATE,
        "synth-city": SN50_SYNTH_CITY_TEMPLATE,
    }

    if subnet in registry:
        return registry[subnet]

    # Return generic with the subnet name filled in
    from dataclasses import replace

    return replace(
        GENERIC_TEMPLATE,
        name=subnet,
        slug=subnet.lower().replace(" ", "-"),
    )
