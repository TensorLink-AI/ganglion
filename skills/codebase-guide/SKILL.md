---
name: codebase-guide
description: "Use for every task involving this project. Covers running Ganglion, its CLI commands, HTTP bridge API, pipeline execution, knowledge queries, configuration, and operational workflows. Trigger phrases: 'run the pipeline', 'start the server', 'check status', 'query knowledge', 'configure', 'call the API', 'scaffold a project', 'check metrics', 'rollback', 'swap policy'."
metadata: {"openclaw": {"emoji": "📘", "requires": {"bins": ["python3", "ganglion"], "env": ["OPENAI_API_KEY"]}, "always": true}}
---

# Ganglion — Operator's Manual

Ganglion is a domain-specific execution engine for Bittensor subnet mining. It provides a pipeline framework for orchestrating autonomous mining agents that search for optimal model configurations. It exposes a CLI, an HTTP bridge API, and a Python library. Ganglion is search infrastructure — it doesn't know what a good model looks like, it knows how to search for one.

## Quick Reference

```bash
# Scaffold a new project
ganglion init ./my-subnet --subnet sn9 --netuid 9

# Check state (local mode)
ganglion status ./my-subnet
ganglion tools ./my-subnet
ganglion agents ./my-subnet
ganglion knowledge ./my-subnet --capability training --max-entries 10
ganglion pipeline ./my-subnet

# Run (local mode)
ganglion run ./my-subnet
ganglion run ./my-subnet --stage plan
ganglion run ./my-subnet --overrides '{"target_metric":"accuracy"}'

# Start HTTP bridge (remote mode)
ganglion serve ./my-subnet --bot-id alpha --port 8899

# Check state (remote mode)
curl -s "$GANGLION_URL/status" | jq
```

## Mode Detection

Ganglion supports two modes. **Always check which mode applies before running commands.**

- **Local mode**: No `GANGLION_URL` set, or `GANGLION_PROJECT` is set. Use `ganglion <command> <project_dir>` directly.
- **Remote mode**: `GANGLION_URL` is set. Use `curl` against the HTTP bridge.

```bash
if [ -n "$GANGLION_PROJECT" ] || [ -z "$GANGLION_URL" ]; then
  echo "local"
else
  echo "remote"
fi
```

## How to Run

**Prerequisites:** Python >= 3.11, `OPENAI_API_KEY` set (used by the LLM runtime).

**Install:** `pip install ganglion`

**Scaffold a project:**
```bash
ganglion init ./my-subnet --subnet sn9 --netuid 9
```
This creates `config.py`, `tools/`, `agents/`, and `skill/` in the target directory.

**Start in local mode:**
```bash
export GANGLION_PROJECT=./my-subnet
ganglion status $GANGLION_PROJECT
```

**Start in remote mode:**
```bash
ganglion serve ./my-subnet --bot-id alpha --port 8899
export GANGLION_URL=http://127.0.0.1:8899
```

The project directory must contain a `config.py` that defines `subnet_config` (SubnetConfig) and `pipeline` (PipelineDef). See `{baseDir}/references/configuration.md` for full config details.

## Key Features

### Observe State
Query the current framework state — registered tools, agents, pipeline definition, knowledge, metrics, and run history. Local mode uses CLI commands; remote mode uses GET endpoints.

Full reference: `{baseDir}/references/commands.md`

### Execute Pipelines
Run the full pipeline or a single stage. The orchestrator executes stages in dependency order, applies retry policies, injects accumulated knowledge into agent prompts, and records outcomes.

```bash
# Local
ganglion run ./my-subnet
ganglion run ./my-subnet --stage plan

# Remote
curl -s -X POST "$GANGLION_URL/run/pipeline" -H "Content-Type: application/json" -d '{}' | jq
curl -s -X POST "$GANGLION_URL/run/stage/plan" -H "Content-Type: application/json" -d '{}' | jq
```

### Mutate at Runtime (Remote Only)
Register new tools, agents, and components; patch the pipeline; swap retry policies; update prompts. All mutations are validated, audited, and reversible.

```bash
# Register a tool
curl -s -X POST "$GANGLION_URL/tools" -H "Content-Type: application/json" \
  -d '{"name":"my_tool","code":"<code>","category":"training"}' | jq

# Patch pipeline
curl -s -X PATCH "$GANGLION_URL/pipeline" -H "Content-Type: application/json" \
  -d '{"operations":[{"op":"add_stage","stage":{"name":"validate","agent":"Validator","depends_on":["train"]}}]}' | jq
```

Pipeline operations: `add_stage`, `remove_stage`, `update_stage`. See `{baseDir}/references/commands.md` for all mutation endpoints.

### Knowledge Store
Cross-run strategic memory that compounds over time. Records patterns (what worked) and antipatterns (what failed), then automatically injects relevant history into agent prompts. Knowledge is queried by capability and filtered by bot_id for multi-bot setups.

```bash
# Local
ganglion knowledge ./my-subnet --bot-id alpha --capability training

# Remote
curl -s "$GANGLION_URL/knowledge?capability=training&max_entries=10" | jq
```

### Rollback
Undo any mutation. Every mutation is recorded in an audit log with rollback data.

```bash
curl -s -X POST "$GANGLION_URL/rollback/last" | jq
curl -s -X POST "$GANGLION_URL/rollback/0" | jq    # undo ALL mutations
```

### Multi-Bot Workflows
Multiple OpenClaw sessions share a knowledge pool via `--bot-id`. Each bot's discoveries flow into the shared pool. Cooperation emerges from shared knowledge, not explicit coordination.

```bash
# Two local sessions
ganglion run ./my-subnet --bot-id alpha
ganglion run ./my-subnet --bot-id beta

# Two remote servers
ganglion serve ./my-subnet --bot-id alpha --port 8899
ganglion serve ./my-subnet --bot-id beta  --port 8900
```

## Common Workflows

See `{baseDir}/examples/common-workflows.md` for full step-by-step guides.

1. **First run**: `ganglion init` → edit `config.py` → `ganglion run`
2. **Iterative mining**: check status → review knowledge → run pipeline → check metrics → repeat
3. **Dynamic mutation**: observe tools/agents → register new tool via API → patch pipeline → run
4. **Multi-bot setup**: start multiple servers with different `--bot-id` values on the same project

## When Things Go Wrong

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: No config.py` | Wrong project path | Verify path contains `config.py` |
| `OPENAI_API_KEY` errors | Missing or invalid API key | `export OPENAI_API_KEY=sk-...` |
| `ConcurrentMutationError` | Mutating during a pipeline run | Wait for the run to finish |
| `PipelineValidationError` | Invalid pipeline DAG (cycles, missing deps) | Check `ganglion pipeline` output |
| Agent stuck / max turns reached | Agent cannot make progress | Review knowledge, swap retry policy, adjust prompts |

Full troubleshooting: `{baseDir}/references/troubleshooting.md`

## Retry Policies

Four built-in policies control how stages retry on failure:
- **NoRetry** — single attempt
- **FixedRetry** — retry N times (default: 3)
- **EscalatingRetry** — increase temperature per attempt, optional stall detection
- **ModelEscalationRetry** — climb a model cost ladder (cheap → expensive)

Three presets: `SN50_PRESET` (escalating + stall detection), `SIMPLE_PRESET` (fixed), `AGGRESSIVE_PRESET` (model escalation).

## Additional Resources

- Full CLI & API reference: `{baseDir}/references/commands.md`
- Configuration guide: `{baseDir}/references/configuration.md`
- Operational procedures: `{baseDir}/references/operations.md`
- Troubleshooting: `{baseDir}/references/troubleshooting.md`
- Workflow examples: `{baseDir}/examples/common-workflows.md`
- Sample API requests: `{baseDir}/examples/sample-requests.md`
- Health check script: `{baseDir}/scripts/healthcheck.sh`
