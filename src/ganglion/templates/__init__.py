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
                f'    "{tname}": TaskDef(name="{tname}", weight={tdef.get("weight", 1.0)}, metadata={meta_str}),'
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

        return f'''---
name: ganglion-{self.slug}
description: Domain knowledge and bootstrap strategies for mining {self.name} (netuid {self.netuid}) with Ganglion.
homepage: https://github.com/TensorLink-AI/ganglion
metadata: {{"openclaw":{{"emoji":"\\u26d3","requires":{{"env":["GANGLION_URL"]}}}}}}
---

# {self.name} (netuid {self.netuid})

{self.domain_context}

## Metrics

The validator scores miners on:

{chr(10).join(f"- **{m['name']}** ({m['direction']}, weight={m.get('weight', 1.0)}): {m.get('description', '')}" for m in self.metrics)}

## Output format

{self.output_format}: {self.output_description}
{strategies_section}{pitfalls_section}
## Bootstrap workflow

1. Start the Ganglion bridge: `ganglion serve ./{self.slug} --bot-id {{{{bot_id}}}} --port 8899`
2. Check current knowledge: `curl -s "$GANGLION_URL/knowledge" | jq`
3. Review starter tools: `curl -s "$GANGLION_URL/tools" | jq`
4. Run the pipeline: `curl -s -X POST "$GANGLION_URL/run/pipeline" | jq`
5. Check metrics: `curl -s "$GANGLION_URL/metrics" | jq`
6. Iterate: write new tools, adjust agents, patch the pipeline based on results

## Constraints

{chr(10).join(f"- **{k}**: {v}" for k, v in self.constraints.items()) if self.constraints else "None specified."}
'''

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
        {"name": "score", "direction": "maximize", "weight": 1.0, "description": "Primary scoring metric"},
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


def get_template(subnet: str) -> SubnetTemplate:
    """Look up a built-in template by name or return a customized generic."""
    registry: dict[str, SubnetTemplate] = {
        "generic": GENERIC_TEMPLATE,
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
