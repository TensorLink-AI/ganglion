---
name: ganglion
description: Drive Bittensor subnet mining pipelines via the Ganglion execution engine. Observe state, mutate tools/agents/pipelines, run experiments, and query cross-bot shared knowledge.
homepage: https://github.com/TensorLink-AI/ganglion
metadata: {"openclaw":{"emoji":"🧠","requires":{"bins":["curl","jq"],"env":["GANGLION_URL"]},"primaryEnv":"GANGLION_URL","install":[{"id":"pip","kind":"command","command":"pip install ganglion","label":"Install Ganglion (pip)"}]}}
---

# Ganglion

Drive Bittensor subnet mining from OpenClaw via the Ganglion bridge API.

## Setup

Start the Ganglion bridge server in a separate terminal before using this skill:

```
ganglion serve ./my-subnet --bot-id <your-bot-id> --port 8899
```

Set `GANGLION_URL` to the server address (default: `http://127.0.0.1:8899`).

## Conventions

- All endpoints return JSON. Pipe through `jq` for readability.
- Use `$GANGLION_URL` as the base for every request.
- Mutations are blocked while a pipeline is running.
- Every mutation is recorded and can be rolled back.

## Observe

Check framework status (pipeline, tools, agents, knowledge, run state):

```
curl -s "$GANGLION_URL/status" | jq
```

View the current pipeline definition:

```
curl -s "$GANGLION_URL/pipeline" | jq
```

List registered tools (optionally filter by category):

```
curl -s "$GANGLION_URL/tools" | jq
curl -s "$GANGLION_URL/tools?category=training" | jq
```

List registered agents:

```
curl -s "$GANGLION_URL/agents" | jq
```

View past pipeline runs:

```
curl -s "$GANGLION_URL/runs?n=5" | jq
```

Query experiment metrics:

```
curl -s "$GANGLION_URL/metrics" | jq
curl -s "$GANGLION_URL/metrics?experiment_id=exp-42" | jq
```

View the subnet leaderboard:

```
curl -s "$GANGLION_URL/leaderboard" | jq
```

Query the knowledge store (accumulated patterns and antipatterns):

```
curl -s "$GANGLION_URL/knowledge" | jq
curl -s "$GANGLION_URL/knowledge?capability=training&max_entries=10" | jq
```

Read source code of any project file:

```
curl -s "$GANGLION_URL/source/tools/train.py" | jq .content -r
```

List available model components:

```
curl -s "$GANGLION_URL/components" | jq
```

## Mutate

Write and register a new tool:

```
curl -s -X POST "$GANGLION_URL/tools" \
  -H "Content-Type: application/json" \
  -d '{"name":"my_tool","code":"from ganglion.composition.tool_registry import tool\nfrom ganglion.composition.tool_returns import ExperimentResult\n\n@tool()\ndef my_tool(config: dict) -> ExperimentResult:\n    return ExperimentResult(content=\"result\", metrics={\"score\": 0.95})","category":"training"}' | jq
```

Write and register a new agent:

```
curl -s -X POST "$GANGLION_URL/agents" \
  -H "Content-Type: application/json" \
  -d '{"name":"MyAgent","code":"<agent class code>"}' | jq
```

Write a model component:

```
curl -s -X POST "$GANGLION_URL/components" \
  -H "Content-Type: application/json" \
  -d '{"name":"my_backbone","code":"<component code>","component_type":"backbone"}' | jq
```

Update or add a prompt section for an agent:

```
curl -s -X POST "$GANGLION_URL/prompts" \
  -H "Content-Type: application/json" \
  -d '{"agent_name":"Planner","prompt_section":"strategy","content":"Focus on architectures with low parameter count."}' | jq
```

Patch the pipeline (add/remove/reorder stages):

```
curl -s -X PATCH "$GANGLION_URL/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"operations":[{"op":"add_stage","stage":{"name":"validate","agent":"Validator","depends_on":["train"]}}]}' | jq
```

Swap the retry policy for a stage (or "default" for the pipeline-wide policy):

```
curl -s -X PUT "$GANGLION_URL/policies/train" \
  -H "Content-Type: application/json" \
  -d '{"retry_policy":{"type":"escalating","max_attempts":5,"temperature_step":0.1}}' | jq
```

## Execute

Run the full pipeline:

```
curl -s -X POST "$GANGLION_URL/run/pipeline" \
  -H "Content-Type: application/json" \
  -d '{"overrides":{"target_metric":"accuracy"}}' | jq
```

Run a single pipeline stage:

```
curl -s -X POST "$GANGLION_URL/run/stage/train" \
  -H "Content-Type: application/json" \
  -d '{"context":{"model":"resnet18"}}' | jq
```

Run a direct experiment (bypasses pipeline):

```
curl -s -X POST "$GANGLION_URL/run/experiment" \
  -H "Content-Type: application/json" \
  -d '{"config":{"learning_rate":0.001,"epochs":10}}' | jq
```

## Rollback

Undo the most recent mutation:

```
curl -s -X POST "$GANGLION_URL/rollback/last" | jq
```

Undo all mutations back to a specific index:

```
curl -s -X POST "$GANGLION_URL/rollback/0" | jq
```

## Multi-bot workflow

When multiple OpenClaw sessions run with different `--bot-id` values against the same project, the knowledge store automatically tags entries by source bot. Each bot sees its own patterns plus foreign patterns from other bots, enabling emergent cooperation without explicit coordination.

Start multiple servers on different ports:

```
ganglion serve ./my-subnet --bot-id alpha --port 8899
ganglion serve ./my-subnet --bot-id beta  --port 8900
```

## Typical workflow

1. `/ganglion` to check status
2. Observe knowledge from prior runs
3. Write or adjust tools and agents based on what worked/failed
4. Patch the pipeline if needed
5. Run the pipeline
6. Check metrics and leaderboard
7. Record insights and iterate
