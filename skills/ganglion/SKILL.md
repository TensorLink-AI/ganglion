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

## Observe

### Status

```
# local
ganglion status $GANGLION_PROJECT

# remote
curl -s "$GANGLION_URL/status" | jq
```

### Pipeline

```
# local
ganglion pipeline $GANGLION_PROJECT

# remote
curl -s "$GANGLION_URL/pipeline" | jq
```

### Tools

```
# local
ganglion tools $GANGLION_PROJECT
ganglion tools $GANGLION_PROJECT --category training

# remote
curl -s "$GANGLION_URL/tools" | jq
curl -s "$GANGLION_URL/tools?category=training" | jq
```

### Agents

```
# local
ganglion agents $GANGLION_PROJECT

# remote
curl -s "$GANGLION_URL/agents" | jq
```

### Knowledge

```
# local
ganglion knowledge $GANGLION_PROJECT
ganglion knowledge $GANGLION_PROJECT --capability training --max-entries 10

# remote
curl -s "$GANGLION_URL/knowledge" | jq
curl -s "$GANGLION_URL/knowledge?capability=training&max_entries=10" | jq
```

### Runs and metrics (remote only)

```
curl -s "$GANGLION_URL/runs?n=5" | jq
curl -s "$GANGLION_URL/metrics" | jq
curl -s "$GANGLION_URL/leaderboard" | jq
```

### Source code (remote only)

```
curl -s "$GANGLION_URL/source/tools/train.py" | jq .content -r
```

## Execute

### Run full pipeline

```
# local
ganglion run $GANGLION_PROJECT
ganglion run $GANGLION_PROJECT --overrides '{"target_metric":"accuracy"}'

# remote
curl -s -X POST "$GANGLION_URL/run/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"overrides":{"target_metric":"accuracy"}}' | jq
```

### Run a single stage

```
# local
ganglion run $GANGLION_PROJECT --stage plan

# remote
curl -s -X POST "$GANGLION_URL/run/stage/plan" \
  -H "Content-Type: application/json" \
  -d '{"context":{"model":"resnet18"}}' | jq
```

### Run direct experiment (remote only)

```
curl -s -X POST "$GANGLION_URL/run/experiment" \
  -H "Content-Type: application/json" \
  -d '{"config":{"learning_rate":0.001,"epochs":10}}' | jq
```

## Mutate (remote only)

Mutations require the HTTP bridge. Start the server first.

Write and register a new tool:

```
curl -s -X POST "$GANGLION_URL/tools" \
  -H "Content-Type: application/json" \
  -d '{"name":"my_tool","code":"<tool code>","category":"training"}' | jq
```

Write and register a new agent:

```
curl -s -X POST "$GANGLION_URL/agents" \
  -H "Content-Type: application/json" \
  -d '{"name":"MyAgent","code":"<agent class code>"}' | jq
```

Patch the pipeline:

```
curl -s -X PATCH "$GANGLION_URL/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"operations":[{"op":"add_stage","stage":{"name":"validate","agent":"Validator","depends_on":["train"]}}]}' | jq
```

Swap retry policy:

```
curl -s -X PUT "$GANGLION_URL/policies/train" \
  -H "Content-Type: application/json" \
  -d '{"retry_policy":{"type":"escalating","max_attempts":5,"temperature_step":0.1}}' | jq
```

Update a prompt section:

```
curl -s -X POST "$GANGLION_URL/prompts" \
  -H "Content-Type: application/json" \
  -d '{"agent_name":"Planner","prompt_section":"strategy","content":"Focus on low parameter count."}' | jq
```

## Rollback (remote only)

```
curl -s -X POST "$GANGLION_URL/rollback/last" | jq
curl -s -X POST "$GANGLION_URL/rollback/0" | jq
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

1. Check status and knowledge from prior runs
2. Review what tools and agents are registered
3. Write or adjust tools and agents based on what worked/failed
4. Patch the pipeline if needed
5. Run the pipeline
6. Check metrics and leaderboard
7. Record insights and iterate
