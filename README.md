# Ganglion

**Give your agent hands.**

Ganglion is the execution layer for autonomous agents on [Bittensor](https://bittensor.com). It turns an agent that can *think* into one that can *mine* — observe a subnet, write code, run experiments, learn from results, and improve over time. All without human intervention.

## The idea

A Bittensor subnet defines a scoring function. Mining is search: find outputs that score well. An agent is the searcher. But an agent needs more than an LLM — it needs infrastructure to observe what's happening, tools to act on the world, memory to learn from experience, and compute to run experiments.

Ganglion is that infrastructure.

```
Agent sees the subnet          →  ganglion_get_status, ganglion_get_leaderboard
Agent reads past experiments   →  ganglion_get_knowledge, ganglion_get_metrics
Agent writes new code          →  ganglion_write_tool, ganglion_write_agent
Agent rewires the pipeline     →  ganglion_patch_pipeline, ganglion_swap_policy
Agent runs experiments         →  ganglion_run_pipeline, ganglion_run_experiment
Agent learns from results      →  Knowledge store records patterns automatically
Agent does it again, better    →  Run 30 knows what runs 1–29 discovered
```

An agent connected to Ganglion can do everything a human miner does — except it doesn't sleep.

## How agents connect

Ganglion exposes itself three ways. Use whichever fits your agent framework:

**MCP Server** — Native tool integration for Claude, OpenClaw, or any MCP-compatible agent. The agent sees Ganglion's capabilities as callable tools with typed schemas. Role-based filtering controls what each agent can see and do.

**HTTP API** — REST endpoints for agents built on any stack. Same capabilities as MCP: observe state, mutate code, execute pipelines, roll back mistakes.

**Python SDK** — Direct `FrameworkState` access for agents running in the same process. Zero overhead, full control.

All three surfaces expose the same four capabilities:

| Capability | What the agent can do |
|---|---|
| **Observe** | Read subnet config, pipeline state, tools, agents, knowledge, metrics, leaderboard, source code |
| **Mutate** | Write tools, write agents, patch pipelines, swap retry policies, update prompts |
| **Execute** | Run full pipelines, run individual stages, run one-off experiments |
| **Recover** | Roll back any mutation to a previous state |

## What makes this work

### Agents that write their own tools

An agent isn't limited to the tools it starts with. It can read source code, understand what tools exist, and write new ones — validated, tested, and registered at runtime. A training tool that doesn't exist at 2am can exist at 2:01am because the agent decided it needed one.

### Knowledge that compounds

The knowledge store records what worked (patterns) and what failed (antipatterns) after every run. This isn't logging — it's strategic memory that gets injected into future agent prompts. Run 50 is better than run 1 because the agent carries 49 runs of accumulated evidence about what to try and what to avoid. For an incumbent miner, this knowledge is the moat.

### Pipelines as funnels

Not all experiments are worth finishing. Ganglion pipelines are designed as cost-efficient funnels — cheap stages generate volume, expensive stages select winners:

1. **Plan** — propose experiment configs (zero compute)
2. **Screen** — reject obviously bad configs (seconds)
3. **Prototype** — short runs to estimate potential (minutes)
4. **Train** — full training on winners only (hours)
5. **Validate** — score against the validator (minutes)

An agent that screens 100 ideas and trains 3 beats one that trains 10 blindly.

### Multi-agent cooperation without coordination

Multiple agents share a knowledge pool. When one discovers a pattern, others see it. An architecture explorer finds that transformers outperform CNNs — a hyperparameter specialist sees that and focuses on transformer configs. No coordinator. No message passing. The shared knowledge pool is the only coupling.

This follows the insight from Weis et al. 2026 — agents trained against diverse co-players naturally develop cooperation without explicit coordination machinery.

### Compute that scales

Agents can dispatch work to local machines, SSH hosts, or cloud GPUs (RunPod). The compute router picks the cheapest backend that meets the job's requirements. Artifacts flow back automatically.

## Quick start

```bash
pip install ganglion
```

### Scaffold a project

```bash
ganglion init ./my-subnet --subnet sn9 --netuid 9
```

### Start as an MCP server (for agent integration)

```bash
ganglion serve ./my-subnet --bot-id alpha
```

### Or use the Python SDK directly

```python
from ganglion.state.framework_state import FrameworkState

state = FrameworkState.load("./my-subnet", bot_id="alpha")

# An agent can observe...
status = await state.describe()

# ...mutate...
await state.write_and_register_tool("new_tool", code, "training")

# ...execute...
result = await state.run_pipeline()

# ...and recover
await state.rollback_last()
```

## Architecture

Ganglion is a 5-layer pipeline engine. Agent lifecycle, multi-agent routing, and communication are handled upstream by your agent framework (e.g. [OpenClaw](https://github.com/openclaw)). Ganglion is the execution substrate.

```
Layer 1: Runtime          — Turn-by-turn LLM execution kernel
Layer 2: Composition      — Tools, agents, and prompt building
Layer 3: Orchestration    — Pipeline DAG with typed data flow between stages
Layer 4: Policies         — Retry, stall detection, model escalation (all swappable)
Layer 5: Subnet Domain    — Your config, tools, prompts, and agents
```

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER_API_KEY` | Yes | — | LLM provider API key |
| `LLM_PROVIDER_BASE_URL` | No | — | Custom API base URL |
| `GANGLION_LLM_MODEL` | No | `gpt-4o` | LLM model name |
| `GANGLION_HOST` | No | `127.0.0.1` | Server bind host |
| `GANGLION_PORT` | No | `8899` | Server bind port |

See `.env.example` for the full list.

## Testing

```bash
pytest tests/ -v
```

## License

MIT
