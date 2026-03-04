# Setting Up Ganglion with Claude Code

This guide walks you through using [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (the `claude` CLI) to develop, operate, and iterate on Ganglion subnet mining projects.

## Why Claude Code?

Ganglion is a pipeline engine — it provides the layers (runtime, composition, orchestration, policies, knowledge), but the domain-specific work lives in your tools, agents, and config. Claude Code is how you author and evolve that domain layer without switching between editor, terminal, and docs.

What Claude Code gives you here:

- **Scaffold projects** — `ganglion init` generates the boilerplate, Claude Code fills in the domain logic
- **Write tools and agents** — describe what your subnet scores on, Claude Code writes the `@tool` functions and `BaseAgentWrapper` subclasses
- **Iterate on pipelines** — adjust stages, retry policies, and knowledge injection through conversation
- **Debug runs** — read pipeline output, trace failures, fix tools, re-run
- **Multi-file edits** — when a pipeline change requires updates across config, tools, agents, and prompts

## Prerequisites

- Python 3.11+
- Claude Code installed ([install guide](https://docs.anthropic.com/en/docs/claude-code/overview))
- Ganglion installed:
  ```bash
  pip install ganglion
  # or for development:
  pip install -e ".[dev]"
  ```

## Quick Start

### 1. Scaffold a project

```bash
ganglion init ./my-subnet --subnet sn9 --netuid 9
cd my-subnet
```

This creates:

```
my-subnet/
├── config.py              # SubnetConfig + PipelineDef
├── tools/
│   └── run_experiment.py  # Starter @tool function
├── agents/
│   └── explorer.py        # Starter BaseAgentWrapper
└── skill/
    └── SKILL.md           # OpenClaw skill descriptor
```

### 2. Start Claude Code in the project

```bash
cd my-subnet
claude
```

Claude Code picks up the project context from the directory structure. From here, you work through conversation.

### 3. Tell Claude Code about your subnet

Give it the key details up front so it can write useful tools and agents:

```
The validator for SN9 scores miners on CRPS (continuous ranked probability
score, lower is better) for probabilistic time-series forecasting. Miners
submit prediction distributions as quantile arrays. The validator evaluates
on 8-hour forecast windows over financial data.
```

Claude Code will use this to:

- Update `config.py` with the right `MetricDef` and `OutputSpec`
- Write tools that match the subnet's evaluation format
- Build agents with domain-aware system prompts

### 4. Develop tools

Ask Claude Code to write your experiment tools. Be specific about what the tool should do:

```
Write a tool called "train_forecaster" that:
- Takes architecture (str), lookback_window (int), and learning_rate (float)
- Trains a time-series model on the data in ./data/train.parquet
- Returns an ExperimentResult with CRPS as the metric
- Saves the model checkpoint to ./models/
```

Claude Code writes the `@tool` function in `tools/`, following Ganglion's conventions:

```python
@tool("train_forecaster", category="training")
def train_forecaster(architecture: str, lookback_window: int = 168,
                     learning_rate: float = 1e-3) -> ExperimentResult:
    ...
```

### 5. Develop agents

```
Write a Planner agent that:
- Reviews knowledge from prior runs
- Proposes 5 experiment configs ranked by expected improvement
- Uses train_forecaster and finish tools
```

Claude Code writes the `BaseAgentWrapper` subclass with the right `build_system_prompt()`, `build_tools()`, and optionally `build_context()` and `post_process()` hooks.

### 6. Configure the pipeline

```
Update the pipeline to have 3 stages:
1. "plan" — Planner agent, FixedRetry(2)
2. "train" — Trainer agent, depends on plan, EscalatingRetry(5)
3. "evaluate" — Evaluator agent, depends on train, NoRetry
```

Claude Code edits `config.py` to wire up the `PipelineDef` and `StageDef` entries.

### 7. Run and iterate

```bash
# Run the full pipeline
ganglion run . --bot-id alpha

# Run a single stage
ganglion run . --stage plan

# Check knowledge accumulated so far
ganglion knowledge .
```

After a run, paste the output or errors back to Claude Code:

```
The train stage failed with: ValueError: quantile array must have shape
(n_samples, n_quantiles). Got (64, 1).

Fix the train_forecaster tool to output the correct quantile format.
```

Claude Code reads the tool, identifies the bug, and fixes it.

## What to Use Claude Code For

### Writing domain logic

This is where Claude Code helps most. The framework layers (runtime, orchestration, policies) are already built — you need to fill in:

| What | Where | Claude Code prompt |
|---|---|---|
| Subnet config | `config.py` | "Set up config for SN9 — CRPS metric, minimize, quantile output" |
| Experiment tools | `tools/*.py` | "Write a tool that trains a transformer and returns CRPS" |
| Agent logic | `agents/*.py` | "Write an agent that plans experiments using knowledge context" |
| Pipeline stages | `config.py` | "Add a screening stage between plan and train" |
| Retry policies | `config.py` | "Use ModelEscalationRetry for the train stage" |

### Debugging pipeline runs

```
Here's the output from `ganglion run .`:
<paste output>

Why did the plan stage stall? Fix it.
```

Claude Code can read the pipeline events, agent turn logs, and tool outputs to diagnose issues.

### Knowledge store management

```
Show me what patterns the knowledge store has for "training".
The antipattern about LSTM divergence is outdated — we fixed that. Remove it.
```

### Multi-bot setup

```
Set up a 3-bot configuration:
- alpha: architecture explorer (broad search)
- beta: hyperparameter tuner (deep optimization)
- gamma: baseline sentinel (defends current best)

All sharing knowledge via federated backend.
```

Claude Code writes the `FederatedKnowledgeBackend` config and bot-specific agent variants.

## CLAUDE.md (Optional)

Drop a `CLAUDE.md` in your project root to give Claude Code persistent context:

```markdown
# CLAUDE.md

## Project
Ganglion subnet miner for SN9 (time-series forecasting).

## Commands
- `ganglion run . --bot-id alpha` — run pipeline
- `ganglion status .` — check state
- `ganglion knowledge .` — review knowledge
- `pytest tests/ -v` — run tests

## Architecture
- Tools in `tools/` — each file has one @tool function
- Agents in `agents/` — each file has one BaseAgentWrapper subclass
- Config in `config.py` — SubnetConfig + PipelineDef
- Knowledge in `knowledge/` — JSON backend, shared across bots

## Conventions
- Tool names are snake_case
- Agent class names are PascalCase
- All tools return ExperimentResult
- Metrics: CRPS (lower is better)
```

This ensures Claude Code understands your project structure without re-explanation each session.

## Remote Mode (HTTP Bridge)

If Ganglion runs on a separate machine (GPU server, cloud instance), start the bridge:

```bash
ganglion serve ./my-subnet --bot-id alpha --port 8899
```

Then use Claude Code locally with curl commands against the bridge:

```
Check the status of my Ganglion server at http://gpu-server:8899
```

Claude Code will use `curl` to interact with the observation, mutation, and execution endpoints.

## Typical Session Flow

1. **Start**: `cd my-subnet && claude`
2. **Orient**: "Show me the current pipeline, tools, and last run results"
3. **Analyze**: "What patterns does the knowledge store have? What failed?"
4. **Plan**: "Based on what worked, write a new tool that tries X approach"
5. **Wire**: "Add a screening stage to the pipeline that filters bad configs early"
6. **Run**: `ganglion run . --bot-id alpha`
7. **Debug**: paste failures back, Claude Code fixes them
8. **Record**: successful patterns get recorded in the knowledge store automatically
9. **Iterate**: repeat from step 3

## Tips

- **Be specific about your subnet** — the more detail about the validator's scoring function, the better Claude Code's tools and agents will be.
- **Paste run output** — Claude Code can't see your terminal. Copy-paste pipeline output, errors, and metrics.
- **Use `ganglion init`** — don't start from scratch. The scaffold gives Claude Code the right file structure to work with.
- **One tool per file** — keeps edits scoped and easy for Claude Code to reason about.
- **Review generated code** — especially experiment tools that touch GPUs or external APIs. Claude Code writes the structure, you verify the domain logic.
- **Commit after each working iteration** — use `/commit` in Claude Code to snapshot progress before trying something risky.
