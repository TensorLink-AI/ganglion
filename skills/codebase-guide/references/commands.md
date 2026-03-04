# Commands & API Reference

## CLI Commands

All CLI commands are invoked as `ganglion <command> [args]`.

### ganglion init

Scaffold a new subnet project directory.

```bash
ganglion init <target_dir> [--subnet <name>] [--netuid <id>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `target_dir` | Yes | — | Directory to create |
| `--subnet` | No | `generic` | Subnet name or built-in template |
| `--netuid` | No | `0` | Subnet network UID |

**Creates:**
- `config.py` — subnet configuration
- `tools/run_experiment.py` — starter tool
- `agents/explorer.py` — starter agent
- `skill/SKILL.md` — subnet-specific OpenClaw skill

**Errors:**
- Refuses to overwrite if `config.py` already exists in target

---

### ganglion serve

Start the HTTP bridge server for remote mode.

```bash
ganglion serve <project_dir> [--bot-id <id>] [--host <addr>] [--port <port>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `project_dir` | Yes | — | Path to project directory (must contain `config.py`) |
| `--bot-id` | No | `None` | Bot identifier for multi-bot knowledge |
| `--host` | No | `127.0.0.1` | Host to bind |
| `--port` | No | `8899` | Port to bind |

**Startup output:** Project root, pipeline name, tool count, agent count, bot ID, endpoint URL.

---

### ganglion status

Show full framework state snapshot.

```bash
ganglion status <project_dir> [--bot-id <id>]
```

**Output (JSON):**
```json
{
  "subnet": {"netuid": 9, "name": "...", "metrics": [...]},
  "pipeline": {"name": "...", "stages": [...]},
  "tools": [...],
  "agents": [...],
  "knowledge": {"patterns": 0, "antipatterns": 0},
  "mutations": 0,
  "running": false
}
```

---

### ganglion tools

List registered tools.

```bash
ganglion tools <project_dir> [--category <category>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `project_dir` | Yes | — | Path to project directory |
| `--category` | No | `None` | Filter by category (e.g. `training`) |

---

### ganglion agents

List registered agents.

```bash
ganglion agents <project_dir>
```

---

### ganglion knowledge

Show knowledge store contents.

```bash
ganglion knowledge <project_dir> [--bot-id <id>] [--capability <cap>] [--max-entries <n>]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `project_dir` | Yes | — | Path to project directory |
| `--bot-id` | No | `None` | Bot identifier for filtering |
| `--capability` | No | `None` | Filter by capability |
| `--max-entries` | No | `20` | Max entries per type |

**Output (JSON):**
```json
{
  "patterns": [{"capability": "...", "description": "...", "metric_value": 0.85, ...}],
  "antipatterns": [{"capability": "...", "error_summary": "...", "failure_mode": "...", ...}],
  "summary": {"patterns": 5, "antipatterns": 3}
}
```

---

### ganglion pipeline

Show current pipeline definition.

```bash
ganglion pipeline <project_dir>
```

**Output (JSON):**
```json
{
  "name": "my-pipeline",
  "stages": [
    {"name": "plan", "agent": "Explorer", "depends_on": [], "input_keys": [], "output_keys": ["plan"], "retry": "FixedRetry(max_attempts=2)"}
  ],
  "default_retry": null
}
```

---

### ganglion run

Run the pipeline or a single stage.

```bash
ganglion run <project_dir> [--bot-id <id>] [--stage <name>] [--overrides '<json>']
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `project_dir` | Yes | — | Path to project directory |
| `--bot-id` | No | `None` | Bot identifier |
| `--stage` | No | `None` | Run only this stage |
| `--overrides` | No | `None` | JSON string of TaskContext overrides |

**Output (JSON):**
```json
{
  "success": true,
  "failed_stage": null,
  "reason": null,
  "results": {
    "plan": {"success": true, "attempts": 1, "error": null, "structured": {...}},
    "experiment": {"success": true, "attempts": 2, "error": null, "structured": {...}}
  }
}
```

---

## HTTP Bridge API

Base URL: `http://<host>:<port>` (default `http://127.0.0.1:8899`).

All responses are JSON. Errors return HTTP 400 with `{"errors": [...]}` or HTTP 404/500.

### Observation Endpoints

#### GET /status
Full framework state snapshot. Same output as `ganglion status`.

#### GET /pipeline
Current pipeline definition.

#### GET /tools
Registered tools.

| Query Param | Default | Description |
|-------------|---------|-------------|
| `category` | — | Filter by category |

#### GET /agents
Registered agents.

#### GET /runs
Past pipeline runs.

| Query Param | Default | Description |
|-------------|---------|-------------|
| `n` | `10` | Number of runs to return |

Returns `[]` if no persistence backend configured.

#### GET /metrics
Experiment metrics.

| Query Param | Default | Description |
|-------------|---------|-------------|
| `experiment_id` | — | Filter by experiment ID |

Returns `[]` if no persistence backend configured.

#### GET /leaderboard
Current Bittensor subnet leaderboard. Returns `[]` if no subnet client configured.

#### GET /knowledge
Knowledge store contents.

| Query Param | Default | Description |
|-------------|---------|-------------|
| `capability` | — | Filter by capability |
| `max_entries` | `20` | Max entries per type |

#### GET /source/{path}
Read any file in the project directory.

| Path Param | Description |
|------------|-------------|
| `path` | Relative path within project root |

**Response:** `{"path": "tools/train.py", "content": "..."}`

Returns 404 if file not found.

#### GET /components
Available model components. Returns `[]` if no training framework configured.

---

### Mutation Endpoints

All mutations are blocked while a pipeline is running (`ConcurrentMutationError`).

#### POST /tools
Write and register a new tool.

**Request:**
```json
{
  "name": "my_tool",
  "code": "from ganglion.composition.tool_registry import tool\nfrom ganglion.composition.tool_returns import ExperimentResult\n\n@tool(\"my_tool\", category=\"training\")\ndef my_tool(param: str) -> ExperimentResult:\n    \"\"\"Docstring required.\"\"\"\n    return ExperimentResult(content=\"done\")",
  "category": "training",
  "test_code": "assert 1 + 1 == 2"
}
```

**Validation rules:**
- Valid Python syntax
- Must have `@tool` decorator
- All parameters must have type hints
- Docstring required
- No blocked imports: `subprocess`, `os.system`, `shutil.rmtree`, `socket`, `http.server`

**Success:** `{"success": true, "path": "/path/to/tools/my_tool.py"}`

**Failure (400):** `{"errors": ["No @tool decorator found", "Parameter 'x' missing type hint"]}`

#### POST /agents
Write and register a new agent.

**Request:**
```json
{
  "name": "MyAgent",
  "code": "from ganglion.composition.base_agent import BaseAgentWrapper\nfrom ganglion.composition.tool_registry import build_toolset\n\nclass MyAgent(BaseAgentWrapper):\n    def build_system_prompt(self, task):\n        return \"You are an agent.\"\n    def build_tools(self, task):\n        return build_toolset(\"run_experiment\", \"finish\")",
  "test_task": null
}
```

**Validation rules:**
- Valid Python syntax
- Must inherit from `BaseAgentWrapper`
- Must implement `build_system_prompt` and `build_tools`
- No blocked imports

**Success:** `{"success": true, "path": "/path/to/agents/myagent.py"}`

#### POST /components
Write a model component.

**Request:**
```json
{
  "name": "my_backbone",
  "code": "class MyBackbone: ...",
  "component_type": "backbone"
}
```

**Success:** `{"success": true, "path": "/path/to/components/my_backbone.py"}`

#### POST /prompts
Write or replace a prompt section for an agent.

**Request:**
```json
{
  "agent_name": "Planner",
  "prompt_section": "strategy",
  "content": "Focus on low parameter count architectures."
}
```

**Success:** `{"success": true, "path": "/path/to/prompts/planner.py"}`

Creates/updates `prompts/{agent_name}.py` with section markers.

#### PATCH /pipeline
Apply pipeline modifications atomically.

**Request:**
```json
{
  "operations": [
    {"op": "add_stage", "stage": {"name": "validate", "agent": "Validator", "depends_on": ["train"], "input_keys": ["model_path"], "output_keys": ["validation"]}},
    {"op": "remove_stage", "stage_name": "old_stage"},
    {"op": "update_stage", "stage_name": "train", "updates": {"optional": true, "agent": "NewTrainer"}}
  ]
}
```

**Operations:**

| Operation | Required Fields | Description |
|-----------|----------------|-------------|
| `add_stage` | `stage` (object with `name`, `agent`) | Add a new stage to the pipeline |
| `remove_stage` | `stage_name` | Remove a stage and clean up dependency refs |
| `update_stage` | `stage_name`, `updates` (dict) | Update fields on an existing stage |

**Validation:** Pipeline must remain a valid DAG with all agents registered.

**Success:** `{"success": true, "pipeline": {...}}`

#### PUT /policies/{stage_name}
Swap retry policy for a stage. Use `"default"` as stage_name for the pipeline default.

**Request:**
```json
{
  "retry_policy": {"type": "escalating", "max_attempts": 5, "temperature_step": 0.1}
}
```

**Success:** `{"success": true}`

---

### Execution Endpoints

#### POST /run/pipeline
Execute the full pipeline.

**Request (optional):**
```json
{
  "overrides": {"target_metric": "accuracy"}
}
```

**Response:**
```json
{
  "success": true,
  "failed_stage": null,
  "reason": null,
  "results": {
    "plan": {"success": true, "attempts": 1, "error": null, "structured": {...}}
  }
}
```

#### POST /run/stage/{stage_name}
Execute a single pipeline stage in isolation.

**Request (optional):**
```json
{
  "context": {"model": "resnet18"}
}
```

**Response:**
```json
{
  "success": true,
  "attempts": 1,
  "error": null,
  "structured": {...}
}
```

#### POST /run/experiment
Run a single experiment directly (bypasses the pipeline). Calls the registered `run_experiment` tool.

**Request:**
```json
{
  "config": {"learning_rate": 0.001, "epochs": 10}
}
```

**Response:**
```json
{
  "success": true,
  "content": "Experiment completed. Accuracy: 0.85",
  "structured": {...},
  "metrics": {"accuracy": 0.85}
}
```

Returns `{"success": false, "error": "No 'run_experiment' tool registered"}` if no such tool exists.

---

### Rollback Endpoints

#### POST /rollback/last
Undo the most recent mutation.

**Response:** `{"success": true}`

Returns 400 with `{"errors": ["No mutations to rollback"]}` if no mutations exist.

#### POST /rollback/{index}
Undo all mutations back to the given index (0 = undo everything).

**Response:** `{"success": true}`

**What gets rolled back per mutation type:**
- `write_tool` / `write_agent` — file restored or deleted, unregistered from registry
- `patch_pipeline` — previous pipeline definition restored
- `swap_policy` — previous retry policy restored
- `write_prompt` — prompt file restored or deleted

---

## Retry Policy Reference

### NoRetry
Single attempt, no retries.

### FixedRetry
```
FixedRetry(max_attempts=3)
```
Retries up to `max_attempts` times with the same configuration.

### EscalatingRetry
```
EscalatingRetry(max_attempts=5, base_temp=0.1, temp_step=0.1, stall_detector=None)
```
Temperature increases per attempt: `base_temp + (attempt * temp_step)`. If a `stall_detector` is provided and detects repeated outputs, a divergence prompt is injected.

### ModelEscalationRetry
```
ModelEscalationRetry(model_ladder=["haiku-3.5", "sonnet-3.5", "opus-3"], attempts_per_model=2)
```
Tries each model in the ladder for `attempts_per_model` attempts before escalating.

### Presets
- **SN50_PRESET** — EscalatingRetry with ConfigComparisonStallDetector
- **SIMPLE_PRESET** — FixedRetry(max_attempts=3)
- **AGGRESSIVE_PRESET** — ModelEscalationRetry with haiku → sonnet → opus ladder

### Stall Detectors
- **ConfigComparisonStallDetector** — detects when the agent produces the same experiment config twice
- **OutputHashStallDetector** — detects when the agent output text repeats (configurable `max_repeats`, default 2)
