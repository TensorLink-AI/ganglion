# Knowledge Injection Format

The MCP plugin injects knowledge into tool response envelopes using the format
below.  This matches the output of `KnowledgeStore.to_prompt_context()` and
`KnowledgeStore.to_foreign_prompt_context()` from
`src/ganglion/knowledge/store.py`.

## Own knowledge (to_prompt_context)

```
## Accumulated Knowledge

### Known Good Approaches
- {description} (achieved {metric_name}={metric_value})

### Known Failures (avoid these)
- {error_summary}
  Failure mode: {failure_mode}

### Agent Designs That Worked
- {agent_class} with tools [{tools_str}] ({metric_name}={metric_value})
```

**Rules:**
- The metric suffix ` (achieved {metric_name}={metric_value})` is only appended
  when `metric_value` is not null.
- The `Failure mode:` line is only appended when `failure_mode` is not null.
- The "Agent Designs That Worked" section is only included when there are
  matching agent design records.
- Returns empty string when there are no patterns and no antipatterns.

## Foreign knowledge (to_foreign_prompt_context)

```
## Discoveries from other bots

### Approaches that worked for others
- {description} (achieved {metric_name}={metric_value})

### Dead ends found by others (avoid these)
- {error_summary}
  Failure mode: {failure_mode}
```

**Rules:**
- Same conditional formatting as own knowledge.
- Queries exclude the current bot's own entries (`exclude_source=bot_id`).
- Returns empty string when `bot_id` is not set (single-bot mode) or when
  there are no foreign patterns/antipatterns.

## Combined injection

In a multi-bot MCP deployment, both blocks are concatenated and injected into
the `_knowledge_context` field of tool response envelopes:

```
## Accumulated Knowledge

### Known Good Approaches
- Used EWMA volatility with lambda=0.92 for BTC (achieved crps=42.3)
- GBM with regime switching for SOL (achieved crps=38.7)

### Known Failures (avoid these)
- Fixed sigma=0.02 produces terrible calibration on high-vol days
  Failure mode: underestimates tail risk

## Discoveries from other bots

### Approaches that worked for others
- Heston stochastic-vol model for ETH (achieved crps=35.1)

### Dead ends found by others (avoid these)
- ARIMA on raw prices diverges after 2h horizon
  Failure mode: non-stationary input
```
