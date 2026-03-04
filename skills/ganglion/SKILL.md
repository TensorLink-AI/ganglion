---
name: ganglion
description: Drive Bittensor subnet mining pipelines via the Ganglion execution engine. Observe state, mutate tools/agents/pipelines, run experiments, and query cross-bot shared knowledge.
homepage: https://github.com/TensorLink-AI/ganglion
metadata: {"openclaw":{"emoji":"🧠","requires":{"anyBins":["ganglion","curl"]},"install":[{"id":"pip","kind":"command","command":"pip install ganglion","label":"Install Ganglion (pip)"}]}}
---

# Ganglion

Drive Bittensor subnet mining from OpenClaw via Ganglion.

## Mode detection

Ganglion supports two modes. **Always check which mode applies before running commands.**

- **Local mode** (same machine): `GANGLION_PROJECT` is set, or no `GANGLION_URL` is set. Use `ganglion <command> $GANGLION_PROJECT` directly — no server needed.
- **Remote mode**: `GANGLION_URL` is set. Use `curl` against the HTTP bridge.

Check which mode you are in:

```
if [ -n "$GANGLION_PROJECT" ] || [ -z "$GANGLION_URL" ]; then
  echo "local"
else
  echo "remote"
fi
```

## Response format

All remote (HTTP bridge) responses use a standard envelope:

- **Success**: `{"data": <payload>}`
- **Error**: `{"error": {"code": "ERROR_CODE", "message": "Human-readable message"}}`

Use `jq .data` to extract the payload from successful responses.

Interactive API docs are available at `$GANGLION_URL/v1/docs` (Swagger UI).

> **Note:** Unversioned routes (e.g. `/status`, `/tools`) still work for backward compatibility but are deprecated. Always prefer the `/v1/` prefix.

## Setup

**Local mode** — set the project path, no server required:

```
export GANGLION_PROJECT=./my-subnet
```

**Remote mode** — start the bridge server and set the URL:

```
ganglion serve ./my-subnet --bot-id <your-bot-id> --port 8899
export GANGLION_URL=http://127.0.0.1:8899
```

## Health checks (remote only)

```
# Liveness — is the process alive?
curl -s "$GANGLION_URL/healthz" | jq

# Readiness — is the bridge configured and ready?
curl -s "$GANGLION_URL/readyz" | jq
```

## Observe

### Status

```
# local
ganglion status $GANGLION_PROJECT

# remote
curl -s "$GANGLION_URL/v1/status" | jq .data
```

### Pipeline

```
# local
ganglion pipeline $GANGLION_PROJECT

# remote
curl -s "$GANGLION_URL/v1/pipeline" | jq .data
```

### Tools

```
# local
ganglion tools $GANGLION_PROJECT
ganglion tools $GANGLION_PROJECT --category training

# remote
curl -s "$GANGLION_URL/v1/tools" | jq .data
curl -s "$GANGLION_URL/v1/tools?category=training" | jq .data
```

### Agents

```
# local
ganglion agents $GANGLION_PROJECT

# remote
curl -s "$GANGLION_URL/v1/agents" | jq .data
```

### Knowledge

```
# local
ganglion knowledge $GANGLION_PROJECT
ganglion knowledge $GANGLION_PROJECT --capability training --max-entries 10

# remote
curl -s "$GANGLION_URL/v1/knowledge" | jq .data
curl -s "$GANGLION_URL/v1/knowledge?capability=training&max_entries=10" | jq .data
```

### Runs and metrics (remote only)

```
curl -s "$GANGLION_URL/v1/runs?n=5" | jq .data
curl -s "$GANGLION_URL/v1/metrics" | jq .data
curl -s "$GANGLION_URL/v1/metrics?experiment_id=exp-001" | jq .data
curl -s "$GANGLION_URL/v1/leaderboard" | jq .data
```

### Components (remote only)

```
curl -s "$GANGLION_URL/v1/components" | jq .data
```

### Source code (remote only)

```
curl -s "$GANGLION_URL/v1/source/tools/train.py" | jq .data.content -r
```

## Execute

### Run full pipeline

```
# local
ganglion run $GANGLION_PROJECT
ganglion run $GANGLION_PROJECT --overrides '{"target_metric":"accuracy"}'

# remote
curl -s -X POST "$GANGLION_URL/v1/run/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"overrides":{"target_metric":"accuracy"}}' | jq .data
```

### Run a single stage

```
# local
ganglion run $GANGLION_PROJECT --stage plan

# remote
curl -s -X POST "$GANGLION_URL/v1/run/stage/plan" \
  -H "Content-Type: application/json" \
  -d '{"context":{"model":"resnet18"}}' | jq .data
```

### Run direct experiment (remote only)

```
curl -s -X POST "$GANGLION_URL/v1/run/experiment" \
  -H "Content-Type: application/json" \
  -d '{"config":{"learning_rate":0.001,"epochs":10}}' | jq .data
```

## Mutate (remote only)

Mutations require the HTTP bridge. Start the server first.

Write and register a new tool (optional `test_code` runs inline tests):

```
curl -s -X POST "$GANGLION_URL/v1/tools" \
  -H "Content-Type: application/json" \
  -d '{"name":"my_tool","code":"<tool code>","category":"training","test_code":"<optional test code>"}' | jq .data
```

Write and register a new agent (optional `test_task` validates the agent):

```
curl -s -X POST "$GANGLION_URL/v1/agents" \
  -H "Content-Type: application/json" \
  -d '{"name":"MyAgent","code":"<agent class code>","test_task":{"input":"test"}}' | jq .data
```

Write and register a new component:

```
curl -s -X POST "$GANGLION_URL/v1/components" \
  -H "Content-Type: application/json" \
  -d '{"name":"MyBackbone","code":"<component code>","component_type":"backbone"}' | jq .data
```

Patch the pipeline:

```
curl -s -X PATCH "$GANGLION_URL/v1/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"operations":[{"op":"add_stage","stage":{"name":"validate","agent":"Validator","depends_on":["train"]}}]}' | jq .data
```

Swap retry policy:

```
curl -s -X PUT "$GANGLION_URL/v1/policies/train" \
  -H "Content-Type: application/json" \
  -d '{"retry_policy":{"type":"escalating","max_attempts":5,"temperature_step":0.1}}' | jq .data
```

Update a prompt section:

```
curl -s -X POST "$GANGLION_URL/v1/prompts" \
  -H "Content-Type: application/json" \
  -d '{"agent_name":"Planner","prompt_section":"strategy","content":"Focus on low parameter count."}' | jq .data
```

## Rollback (remote only)

```
curl -s -X POST "$GANGLION_URL/v1/rollback/last" | jq .data
curl -s -X POST "$GANGLION_URL/v1/rollback/0" | jq .data
```

## Multi-bot workflow

When multiple OpenClaw sessions use different `--bot-id` values, the knowledge store tags entries by source bot. Each bot sees its own patterns plus other bots' discoveries.

```
# local — two terminals, same project
ganglion run ./my-subnet --bot-id alpha
ganglion run ./my-subnet --bot-id beta

# remote — two servers, different ports
ganglion serve ./my-subnet --bot-id alpha --port 8899
ganglion serve ./my-subnet --bot-id beta  --port 8900
```

## Typical workflow

1. Check health and readiness (`/healthz`, `/readyz`)
2. Check status and knowledge from prior runs
3. Review what tools, agents, and components are registered
4. Write or adjust tools and agents based on what worked/failed
5. Patch the pipeline if needed
6. Run the pipeline
7. Check metrics and leaderboard
8. Record insights and iterate
