# Running Ganglion via the OpenClaw Bot

This guide walks through how to set up and operate Ganglion from an OpenClaw bot session. OpenClaw is the autonomous agent orchestrator; Ganglion is the execution engine it drives. OpenClaw handles agent lifecycle and multi-agent routing — Ganglion handles the mining pipeline, tools, knowledge, and experiments.

## Prerequisites

- Python 3.10+
- An OpenClaw installation with skill support
- An `OPENAI_API_KEY` (or compatible LLM endpoint) for Ganglion's agent runtime

Install Ganglion:

```bash
pip install ganglion
```

Verify the CLI is available:

```bash
ganglion --help
```

## How the integration works

Ganglion exposes itself to OpenClaw in two ways:

1. **As an OpenClaw Skill** — a `SKILL.md` manifest that tells OpenClaw what commands are available and how to use them.
2. **As an HTTP bridge** — a FastAPI server that OpenClaw can call via `curl` when running remotely.

OpenClaw reads the skill manifest to learn what Ganglion can do, then issues CLI commands (local mode) or HTTP requests (remote mode) to drive it.

## Step 1: Scaffold a subnet project

Create a new project directory with starter config, tools, agents, and the OpenClaw skill manifest:

```bash
ganglion init ./my-subnet --subnet sn9 --netuid 9
```

This generates:

```
my-subnet/
├── config.py              # SubnetConfig + PipelineDef
├── tools/
│   └── run_experiment.py  # Starter experiment tool
├── agents/
│   └── explorer.py        # Starter agent
└── skill/
    └── SKILL.md           # OpenClaw skill manifest
```

Edit `config.py` to match your target subnet's metrics, tasks, and output format. Replace the starter tool in `tools/run_experiment.py` with your actual training/inference logic.

## Step 2: Install the skill into OpenClaw

Copy the generated skill manifest into your OpenClaw skills directory:

```bash
cp my-subnet/skill/SKILL.md ~/.openclaw/skills/ganglion-sn9/SKILL.md
```

Also install the base Ganglion skill (ships with the package) so that OpenClaw has the full command reference:

```bash
cp skills/ganglion/SKILL.md ~/.openclaw/skills/ganglion/SKILL.md
```

Once installed, OpenClaw will see Ganglion as an available skill and know the full set of observe/execute/mutate/rollback commands.

## Step 3: Choose a mode

### Local mode (OpenClaw and Ganglion on the same machine)

No server needed. Set the project path and use the `ganglion` CLI directly:

```bash
export GANGLION_PROJECT=./my-subnet
```

OpenClaw will run commands like:

```bash
ganglion status $GANGLION_PROJECT
ganglion tools $GANGLION_PROJECT
ganglion run $GANGLION_PROJECT
```

### Remote mode (OpenClaw connects over HTTP)

Start the Ganglion bridge server:

```bash
ganglion serve ./my-subnet --bot-id alpha --port 8899
```

Output:

```
Ganglion bridge starting on 127.0.0.1:8899
  Project:  /path/to/my-subnet
  Pipeline: sn9-pipeline
  Tools:    1
  Agents:   1
  Bot ID:   alpha

OpenClaw can connect at:
  http://127.0.0.1:8899
```

Set the URL so OpenClaw uses the bridge:

```bash
export GANGLION_URL=http://127.0.0.1:8899
```

OpenClaw will issue `curl` requests against the bridge instead of running CLI commands.

## Step 4: The typical OpenClaw workflow

Once the skill is installed and the mode is configured, an OpenClaw bot session follows this loop:

### 1. Observe current state

```bash
# Local
ganglion status $GANGLION_PROJECT
ganglion knowledge $GANGLION_PROJECT
ganglion tools $GANGLION_PROJECT
ganglion pipeline $GANGLION_PROJECT

# Remote
curl -s "$GANGLION_URL/status" | jq
curl -s "$GANGLION_URL/knowledge" | jq
curl -s "$GANGLION_URL/tools" | jq
curl -s "$GANGLION_URL/pipeline" | jq
```

The bot reads the knowledge store to see what worked and what failed in prior runs, reviews registered tools and agents, and checks the pipeline definition.

### 2. Mutate tools, agents, or pipeline (remote mode)

OpenClaw can write new tools, agents, or pipeline stages at runtime via the HTTP bridge. These mutations are validated and audited.

**Write a new tool:**

```bash
curl -s -X POST "$GANGLION_URL/tools" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "train_transformer",
    "code": "<python code>",
    "category": "training"
  }' | jq
```

**Write a new agent:**

```bash
curl -s -X POST "$GANGLION_URL/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "TransformerTrainer",
    "code": "<agent class code>"
  }' | jq
```

**Patch the pipeline** (add a stage, remove a stage, change dependencies):

```bash
curl -s -X PATCH "$GANGLION_URL/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "operations": [
      {
        "op": "add_stage",
        "stage": {
          "name": "validate",
          "agent": "Validator",
          "depends_on": ["train"]
        }
      }
    ]
  }' | jq
```

**Swap a retry policy:**

```bash
curl -s -X PUT "$GANGLION_URL/policies/train" \
  -H "Content-Type: application/json" \
  -d '{
    "retry_policy": {
      "type": "escalating",
      "max_attempts": 5,
      "temperature_step": 0.1
    }
  }' | jq
```

**Update an agent prompt:**

```bash
curl -s -X POST "$GANGLION_URL/prompts" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "Explorer",
    "prompt_section": "strategy",
    "content": "Focus on low parameter count architectures."
  }' | jq
```

### 3. Execute the pipeline

```bash
# Local — full pipeline
ganglion run $GANGLION_PROJECT

# Local — single stage
ganglion run $GANGLION_PROJECT --stage plan

# Local — with overrides
ganglion run $GANGLION_PROJECT --overrides '{"target_metric":"accuracy"}'

# Remote — full pipeline
curl -s -X POST "$GANGLION_URL/run/pipeline" \
  -H "Content-Type: application/json" \
  -d '{}' | jq

# Remote — single stage
curl -s -X POST "$GANGLION_URL/run/stage/plan" \
  -H "Content-Type: application/json" \
  -d '{"context":{"model":"resnet18"}}' | jq

# Remote — direct experiment (bypass pipeline)
curl -s -X POST "$GANGLION_URL/run/experiment" \
  -H "Content-Type: application/json" \
  -d '{"config":{"learning_rate":0.001,"epochs":10}}' | jq
```

### 4. Review results and iterate

```bash
# Remote — check run history and metrics
curl -s "$GANGLION_URL/runs?n=5" | jq
curl -s "$GANGLION_URL/metrics" | jq
curl -s "$GANGLION_URL/leaderboard" | jq

# Local — check updated knowledge
ganglion knowledge $GANGLION_PROJECT
```

The knowledge store automatically records patterns (what worked) and antipatterns (what failed) after each run. The bot uses this accumulated evidence to inform the next iteration.

### 5. Rollback if needed (remote mode)

If a mutation causes problems, undo it:

```bash
# Undo the most recent mutation
curl -s -X POST "$GANGLION_URL/rollback/last" | jq

# Undo all mutations back to a specific index
curl -s -X POST "$GANGLION_URL/rollback/0" | jq
```

## Multi-bot operation

Multiple OpenClaw bot sessions can run against the same subnet project with different `--bot-id` values. Each bot's discoveries are tagged and shared through the knowledge store.

### Start multiple bots

**Local mode** (two terminals):

```bash
# Terminal 1
ganglion run ./my-subnet --bot-id alpha

# Terminal 2
ganglion run ./my-subnet --bot-id beta
```

**Remote mode** (two bridge servers):

```bash
# Terminal 1
ganglion serve ./my-subnet --bot-id alpha --port 8899

# Terminal 2
ganglion serve ./my-subnet --bot-id beta --port 8900
```

Each OpenClaw session connects to its own bridge:

```bash
# Session 1
export GANGLION_URL=http://127.0.0.1:8899

# Session 2
export GANGLION_URL=http://127.0.0.1:8900
```

### How knowledge sharing works

When bot `alpha` discovers that transformers outperform CNNs, that pattern is recorded with `source_bot=alpha`. Bot `beta` sees the pattern in its knowledge queries and can use it to focus its own search — without explicit coordination.

Recommended specializations across bots:

| Bot | Role | Focus |
|-----|------|-------|
| alpha | Architecture explorer | Broad search across model families |
| beta | Hyperparameter specialist | Deep optimization of promising configs |
| gamma | Data strategist | Preprocessing, augmentation, feature engineering |
| delta | Baseline sentinel | Maintains and defends the current best submission |

Cooperation emerges from the shared knowledge pool. No coordinator is needed.

## Environment variables reference

| Variable | Purpose |
|----------|---------|
| `GANGLION_PROJECT` | Path to the subnet project directory (local mode) |
| `GANGLION_URL` | URL of the Ganglion bridge server (remote mode) |
| `OPENAI_API_KEY` | API key for the LLM backend used by Ganglion agents |

## CLI command reference

| Command | Description |
|---------|-------------|
| `ganglion init <dir>` | Scaffold a new subnet project |
| `ganglion serve <dir>` | Start the HTTP bridge server |
| `ganglion status <dir>` | Show framework state |
| `ganglion tools <dir>` | List registered tools |
| `ganglion agents <dir>` | List registered agents |
| `ganglion knowledge <dir>` | Show knowledge store contents |
| `ganglion pipeline <dir>` | Show pipeline definition |
| `ganglion run <dir>` | Execute pipeline or a single stage |

Common flags:

- `--bot-id <id>` — Bot identifier for multi-bot knowledge tagging (used with `serve`, `status`, `knowledge`, `run`)
- `--stage <name>` — Run only a specific pipeline stage (used with `run`)
- `--overrides '<json>'` — JSON overrides for the pipeline run (used with `run`)
- `--port <n>` — Port for the bridge server (used with `serve`, default: 8899)
- `--host <addr>` — Host to bind the bridge to (used with `serve`, default: 127.0.0.1)
- `--category <name>` — Filter tools by category (used with `tools`)
- `--capability <name>` — Filter knowledge by capability (used with `knowledge`)
- `--max-entries <n>` — Limit knowledge entries returned (used with `knowledge`, default: 20)

## HTTP bridge endpoint reference

### Observation (GET)

| Endpoint | Description |
|----------|-------------|
| `/status` | Full framework state snapshot |
| `/pipeline` | Current pipeline definition |
| `/tools` | Registered tools (`?category=` filter) |
| `/agents` | Registered agents |
| `/knowledge` | Patterns and antipatterns (`?capability=`, `?max_entries=`) |
| `/runs` | Past pipeline runs (`?n=` limit) |
| `/metrics` | Experiment metrics (`?experiment_id=` filter) |
| `/leaderboard` | Bittensor subnet leaderboard |
| `/source/{path}` | Read any project file |
| `/components` | Available model components |

### Mutation (POST/PATCH/PUT)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools` | POST | Write and register a new tool |
| `/agents` | POST | Write and register a new agent |
| `/components` | POST | Write a model component |
| `/prompts` | POST | Write or replace an agent prompt section |
| `/pipeline` | PATCH | Apply pipeline mutations |
| `/policies/{stage}` | PUT | Swap retry policy for a stage |

### Execution (POST)

| Endpoint | Description |
|----------|-------------|
| `/run/pipeline` | Execute the full pipeline |
| `/run/stage/{name}` | Execute a single stage |
| `/run/experiment` | Run a single experiment directly |

### Rollback (POST)

| Endpoint | Description |
|----------|-------------|
| `/rollback/last` | Undo the most recent mutation |
| `/rollback/{index}` | Undo all mutations back to a given index |
