# Ganglion

Domain-specific execution engine for Bittensor subnet mining. Ganglion provides a layered pipeline framework for building, orchestrating, and evolving autonomous mining agents.

## Architecture

Ganglion is a 5-layer pipeline engine — not an agent platform. Agent lifecycle, multi-agent routing, and communication are handled by [OpenClaw](https://github.com/openclaw). Ganglion exposes itself as tools, skills, and an HTTP API.

```
Layer 1: Runtime          — SimpleAgent turn-by-turn execution kernel
Layer 2: Composition      — BaseAgentWrapper, @tool decorator, PromptBuilder
Layer 3: Orchestration    — PipelineDef/StageDef DAG, PipelineOrchestrator, TaskContext
Layer 4: Policies         — Retry, stall detection, model escalation (all swappable)
Layer 5: Subnet Domain    — Developer-written config, tools, prompts, agents
```

## Install

```bash
pip install ganglion
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Define your subnet config

```python
from ganglion.orchestration.task_context import SubnetConfig, MetricDef, OutputSpec, TaskDef

config = SubnetConfig(
    netuid=99,
    name="My Subnet",
    metrics=[MetricDef("accuracy", "maximize")],
    tasks={"main": TaskDef("main", weight=1.0)},
    output_spec=OutputSpec(format="pytorch_model", description="A classifier"),
)
```

### 2. Write tools

```python
from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult

@tool("train_model")
def train_model(architecture: str, learning_rate: float = 1e-3, epochs: int = 10) -> ExperimentResult:
    """Train a model with the given config."""
    # ... training code ...
    return ExperimentResult(
        content=f"Training complete. Accuracy: {accuracy:.4f}",
        experiment_id="exp_001",
        metrics={"accuracy": accuracy},
        artifact_path="/models/exp_001.pt",
    )
```

### 3. Write agents

```python
from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.composition.tool_registry import build_toolset

class Trainer(BaseAgentWrapper):
    def build_system_prompt(self, task):
        return f"You are a model trainer for {task.subnet_config.name}."

    def build_tools(self, task):
        return build_toolset("train_model", "finish")
```

### 4. Define a pipeline

```python
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.policies.retry import FixedRetry

pipeline = PipelineDef(
    name="my-optimization",
    stages=[
        StageDef(name="train", agent="Trainer",
                 output_keys=["model_path", "metrics"],
                 retry=FixedRetry(max_attempts=3)),
        StageDef(name="publish", agent="Publisher",
                 depends_on=["train"],
                 input_keys=["model_path"]),
    ],
)
```

### 5. Run

```python
import asyncio
from ganglion.orchestration.orchestrator import PipelineOrchestrator
from ganglion.orchestration.task_context import TaskContext

async def main():
    task = TaskContext(subnet_config=config)
    orchestrator = PipelineOrchestrator(
        pipeline=pipeline,
        agents={"Trainer": Trainer, "Publisher": Publisher},
    )
    result = await orchestrator.run(task)
    print(f"Pipeline {'succeeded' if result.success else 'failed'}")

asyncio.run(main())
```

## Key Concepts

### Pipeline DAG

Stages declare dependencies via `depends_on` and data flow via `input_keys`/`output_keys`. The framework validates the graph is a DAG and that all data dependencies are satisfied before execution.

### Retry Policies

Swappable strategies plugged into any stage:

- **NoRetry** — single attempt
- **FixedRetry** — retry N times with same config
- **EscalatingRetry** — increase temperature on each retry, with optional stall detection
- **ModelEscalationRetry** — climb a model ladder (haiku -> sonnet -> opus)

### Knowledge Store

Cross-run strategic memory that compounds over time. Records patterns (what worked) and antipatterns (what failed), then injects relevant history into agent prompts:

```python
from ganglion.knowledge.store import KnowledgeStore
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend

knowledge = KnowledgeStore(backend=JsonKnowledgeBackend("./knowledge/"))
```

### FrameworkState (Runtime Mutation)

For dynamic systems (OpenClaw agents, external controllers) that observe and mutate the framework at runtime:

```python
from ganglion.state.framework_state import FrameworkState

state = FrameworkState.create(
    subnet_config=config,
    pipeline_def=pipeline,
    project_root="./my-subnet",
)

# Mutations are validated and audited
await state.write_and_register_tool("new_tool", code, "training")
await state.apply_pipeline_patch([{"op": "add_stage", "stage": {...}}])
await state.swap_policy("train", EscalatingRetry(max_attempts=5))

# Execution blocks mutations; mutations block execution
result = await state.run_pipeline()
```

### HTTP Bridge

FastAPI server at `:8377` exposing the full API for OpenClaw integration:

```
GET  /pipeline        — current pipeline definition
GET  /tools           — registered tools
GET  /knowledge       — accumulated patterns and antipatterns
POST /tools           — write and register a new tool
PATCH /pipeline       — apply pipeline mutations
POST /run/pipeline    — execute the full pipeline
POST /run/stage/{name} — execute a single stage
```

## Design Principles

1. **Extract from two, not one.** Abstractions are validated against at least two dissimilar subnets before promotion to framework code.
2. **Hooks over hardcodes.** Recovery, escalation, and detection are overridable hooks, not locked-in strategies.
3. **Contracts over conventions.** Typed interfaces for shared data, not string-keyed bags.
4. **Thin orchestration.** The orchestrator sequences stages and routes data. It has no opinions about temperature, stall detection, or debugging strategy.

## Testing

```bash
pytest tests/ -v
```

## License

MIT
