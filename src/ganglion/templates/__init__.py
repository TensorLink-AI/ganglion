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

# MCP — uncomment to connect to external tool servers
# from ganglion.mcp.config import MCPClientConfig

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

# MCP client connections (optional)
# Uncomment and configure to connect to external MCP servers.
# Their tools will be available to agents during pipeline execution.
#
# mcp_clients = [
#     MCPClientConfig(
#         name="example-server",
#         transport="stdio",
#         command=["python", "-m", "example_mcp_server"],
#         tool_prefix="example",
#     ),
# ]

# Docker prefabs (optional)
# Pre-built container templates for compute stages. Pass to SubnetConfig
# via docker_prefabs={{...}}. Stages reference prefabs by name; the prefab
# seeds a JobSpec with the command supplied at call time.
#
# from ganglion.compute.protocol import DockerPrefab
#
# docker_prefabs = {{
#     "train": DockerPrefab(
#         name="train",
#         image="my-registry/trainer:latest",
#         gpu_type="A100",
#         gpu_count=1,
#         memory_gb=32,
#     ),
# }}
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

## MCP Integration

This subnet supports MCP (Model Context Protocol) for tool interoperability.

**As a client:** Configure external MCP servers in `config.py` via `mcp_clients`.
Tools from connected servers appear as regular Ganglion tools with a prefix.
Clawbot can also add MCP servers at runtime via `POST /v1/mcp/servers`.

**As a server:** Expose this subnet's tools via MCP for Claude Desktop or other clients:
```bash
ganglion mcp-serve ./{self.slug} --transport stdio
```
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
