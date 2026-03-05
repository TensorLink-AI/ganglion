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

## Prerequisites

- Python 3.11 or later
- An OpenAI API key (or compatible provider)

## Install

```bash
pip install ganglion
```

For development:

```bash
pip install -e ".[dev]"
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER_API_KEY` | Yes (for LLM) | — | LLM provider API key |
| `LLM_PROVIDER_BASE_URL` | No | — | Custom API base URL |
| `GANGLION_LLM_MODEL` | No | `gpt-4o` | LLM model name |
| `GANGLION_HOST` | No | `127.0.0.1` | Server bind host |
| `GANGLION_PORT` | No | `8899` | Server bind port |
| `GANGLION_CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated allowed origins |
| `GANGLION_LOG_LEVEL` | No | `INFO` | Logging level |

See `.env.example` for the full list.

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
- **ModelEscalationRetry** — climb a model cost ladder (cheap → standard → expensive)

### Knowledge Store

Cross-run strategic memory that compounds over time. Records patterns (what worked) and antipatterns (what failed), then injects relevant history into agent prompts. All methods are async.

```python
from ganglion.knowledge.store import KnowledgeStore
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend

knowledge = KnowledgeStore(backend=JsonKnowledgeBackend("./knowledge/"))

# Record outcomes (automatically injected into future agent prompts)
await knowledge.record_success(capability="train", description="Conv+Gaussian head", metric_value=0.85, metric_name="crps")
await knowledge.record_failure(capability="train", error_summary="LSTM diverged", failure_mode="numerical_instability")

# Retrieve formatted context for prompt injection
ctx = await knowledge.to_prompt_context("train")
```

#### Multi-bot shared knowledge

Multiple OpenClaw sessions can share a knowledge backend. Each session sets a `bot_id` so entries are tagged by source and sessions can read each other's discoveries while filtering out their own:

```python
knowledge = KnowledgeStore(
    backend=JsonKnowledgeBackend("./shared-knowledge/"),
    bot_id="alpha",
)

# Own knowledge (patterns + antipatterns from all bots)
ctx = await knowledge.to_prompt_context("train")

# Foreign knowledge only (excludes this bot's entries)
foreign = await knowledge.to_foreign_prompt_context("train")
```

#### Federated backend (distributed peers)

When sessions run on different hosts, the `FederatedKnowledgeBackend` splits writes (local only) from reads (local + all peers). The `PeerDiscovery` protocol is pluggable — `FilesystemPeerDiscovery` is the built-in implementation for same-machine deployments:

```python
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend
from ganglion.knowledge.backends.federated import FederatedKnowledgeBackend, FilesystemPeerDiscovery

backend = FederatedKnowledgeBackend(
    local=JsonKnowledgeBackend("./shared/alpha/"),
    peers=FilesystemPeerDiscovery(base_dir="./shared/", local_bot_id="alpha"),
)
knowledge = KnowledgeStore(backend=backend, bot_id="alpha")
```

Other `PeerDiscovery` implementations (S3, HTTP, gossip) can be plugged in by implementing `query_all_patterns()` and `query_all_antipatterns()`.

### FrameworkState (Runtime Mutation)

For dynamic systems (OpenClaw agents, external controllers) that observe and mutate the framework at runtime:

```python
from ganglion.state.framework_state import FrameworkState

# Load from a project directory (discovers tools/, agents/, config.py)
state = FrameworkState.load("./my-subnet", bot_id="alpha")

# Or create programmatically
state = FrameworkState.create(
    subnet_config=config,
    pipeline_def=pipeline,
    project_root="./my-subnet",
)

# Mutations are validated and audited
await state.write_and_register_tool("new_tool", code, "training", test_code="assert 1+1==2")
await state.write_and_register_agent("NewAgent", agent_code)
await state.apply_pipeline_patch([{"op": "add_stage", "stage": {...}}])
await state.swap_policy("train", EscalatingRetry(max_attempts=5))
await state.update_prompt("trainer", "constraints", "Max 10 experiments per run.")

# Execution blocks mutations; mutations block execution
result = await state.run_pipeline()
stage_result = await state.run_single_stage("train")

# Rollback any mutation
await state.rollback_last()
```

### HTTP Bridge

FastAPI server exposing the full API for OpenClaw integration:

```
# Observation
GET  /status               — full framework state snapshot
GET  /pipeline             — current pipeline definition
GET  /tools                — registered tools (optional ?category= filter)
GET  /agents               — registered agents
GET  /knowledge            — patterns and antipatterns (optional ?capability=&max_entries=)
GET  /runs                 — past pipeline runs (optional ?n=)
GET  /metrics              — experiment metrics
GET  /leaderboard          — Bittensor subnet leaderboard
GET  /source/{path}        — read any project file
GET  /components           — available model components

# Mutation
POST  /tools               — write and register a new tool
POST  /agents              — write and register a new agent
POST  /components          — write a model component
POST  /prompts             — write or replace an agent prompt section
PATCH /pipeline            — apply pipeline mutations
PUT   /policies/{stage}    — swap retry policy for a stage

# Execution
POST /run/pipeline         — execute the full pipeline
POST /run/stage/{name}     — execute a single stage
POST /run/experiment       — run a single experiment directly

# Rollback
POST /rollback/last        — undo the most recent mutation
POST /rollback/{index}     — undo all mutations back to index
```

## How Ganglion Thinks About Mining

A subnet's validator defines a scoring function. Mining is finding outputs that score well. Ganglion is search infrastructure — it doesn't know what a good model looks like, it knows how to search for one.

Fitting the framework to a subnet's incentive mechanism means answering four questions:

1. **What does the validator measure?** This defines your metrics and output spec.
2. **What can you change?** Architecture, hyperparameters, data preprocessing, ensembling — these become your tools.
3. **What should you try?** Strategic exploration guided by accumulated knowledge — this is what agents decide.
4. **How do you know it's working?** Metrics recorded per run, compared against history — this is the knowledge store.

### The funnel: cost efficiency through progressive filtering

Not all experiments are worth finishing. Ganglion pipelines are designed as funnels — cheap stages generate volume, expensive stages select winners:

1. **Plan** — an agent proposes experiment configurations (fast, zero compute)
2. **Screen** — quick heuristic checks reject obviously bad configs (seconds)
3. **Prototype** — short training runs on small data to estimate potential (minutes)
4. **Train** — full training on promising candidates only (hours)
5. **Validate** — final evaluation against the validator's actual scoring (minutes)

Most value comes from screening, not training. A pipeline that screens 100 ideas and trains 3 beats one that trains 10 ideas blindly.

### Knowledge compounds across runs

The knowledge store records patterns (what worked) and antipatterns (what failed) after every run. Run 30 is better than run 1 because the agent has 29 runs of accumulated evidence about what to try and what to avoid. This gives an incumbent miner a structural advantage over newcomers starting from zero — the knowledge is the moat.

### Multi-bot: emergent cooperation from diversity

The mechanism comes from Weis et al. 2026 ("Multi-agent cooperation through in-context co-player inference"). The key insight: agents trained against a diverse mix of co-players naturally develop cooperation — without explicit coordination machinery.

Multiple OpenClaw sessions share a knowledge pool. Each session has a unique `bot_id`. When one session discovers a pattern, others see it automatically through the shared backend. Cooperation emerges from the pool, not from sessions talking to each other.

This works best when sessions are equipped with different skills:

- **Architecture explorer** — broad search across model families
- **Hyperparameter specialist** — deep optimization of promising configs
- **Data strategist** — preprocessing, augmentation, feature engineering
- **Baseline sentinel** — maintains and defends the current best submission

Each session's discoveries flow into the shared knowledge pool. The architecture explorer finds that transformers outperform CNNs on this subnet — the hyperparameter specialist sees that pattern and focuses its search on transformer configurations. No explicit routing. No coordinator. The shared knowledge pool is the only coupling.

### Self-organizing information sharing

Not all knowledge is equally shareable. Antipatterns are always worth sharing — revealing a dead end costs nothing and saves everyone compute. Superseded patterns (approaches that once worked but have been beaten) are cheap to share. Current best approaches stay private — that's competition. The knowledge store's `source_bot` tagging and `exclude_source` filtering make this natural: each session controls what it queries and how it uses foreign knowledge.

## Design Principles

1. **Extract from two, not one.** Abstractions are validated against at least two dissimilar subnets before promotion to framework code.
2. **Hooks over hardcodes.** Recovery, escalation, and detection are overridable hooks, not locked-in strategies.
3. **Contracts over conventions.** Typed interfaces for shared data, not string-keyed bags.
4. **Thin orchestration.** The orchestrator sequences stages and routes data. It has no opinions about temperature, stall detection, or debugging strategy.

## Running Locally

```bash
# Scaffold a new project
ganglion init ./my-subnet --subnet sn9 --netuid 9

# Start the HTTP bridge server
ganglion serve ./my-subnet --bot-id alpha

# Or use local mode (no server)
ganglion status ./my-subnet
ganglion run ./my-subnet
```

## Testing

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=ganglion --cov-report=term-missing
```

## Deployment

### Docker

```bash
docker build -t ganglion .
docker run -p 8899:8899 -e LLM_PROVIDER_API_KEY=sk-... -v ./my-subnet:/app/project ganglion
```

### From Source

```bash
pip install .
ganglion serve ./my-subnet --bot-id alpha
```

## License

MIT
