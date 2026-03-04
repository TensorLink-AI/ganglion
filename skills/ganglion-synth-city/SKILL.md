---
name: ganglion-synth-city
description: Domain knowledge and bootstrap strategies for mining Synth City (netuid 50) with Ganglion.
homepage: https://github.com/TensorLink-AI/ganglion
metadata: {"openclaw": {"emoji": "\u26d3", "requires": {"anyBins": ["ganglion", "curl"]}}}
---

# Synth City (netuid 50)

Synth (SN50) is a probabilistic price forecasting subnet on Bittensor. Miners generate Monte Carlo simulated price paths for crypto assets (BTC, ETH). The validator scores submissions using CRPS — a proper scoring rule that evaluates the full predicted distribution, not just point estimates. Good miners produce distributions that are both well-calibrated (quantiles match observed frequencies) and sharp (tight intervals around the realized price).

The competitive dynamic: miners who simply widen their distributions to cover all outcomes get poor sharpness scores. Miners who make overconfident narrow predictions get poor calibration scores. The winning strategy balances both — capturing real volatility structure without unnecessary uncertainty.

This skill provides domain knowledge. Use the `ganglion` skill for API commands.
When `GANGLION_PROJECT` is set (or `GANGLION_URL` is not set), use local CLI commands.
When `GANGLION_URL` is set, use curl against the HTTP bridge (all endpoints are under `/v1/`).
Responses use a standard envelope: `{"data": <payload>}` on success.

## Metrics

The validator scores miners on:

- **crps** (minimize, weight=1.0): Continuous Ranked Probability Score — measures quality of the full predicted probability distribution against realized prices. Lower is better.
- **calibration** (minimize, weight=0.3): How well the predicted quantiles match observed frequencies. Perfect calibration = 0.
- **sharpness** (minimize, weight=0.2): Width of prediction intervals — sharper (narrower) distributions score better, but only if well-calibrated.

## Output format

price_paths_json: JSON array of simulated price paths. Each path is a list of (timestamp, price) pairs. The validator evaluates the full distribution of paths against realized prices using CRPS.

## Search strategies

- Start with Geometric Brownian Motion (GBM) as baseline — it's fast and establishes a CRPS floor
- Try jump-diffusion models (Merton) to capture sudden price moves that GBM misses
- Experiment with regime-switching models (bull/bear/sideways) to adapt volatility
- Use GARCH or EWMA for volatility estimation instead of constant vol assumptions
- Neural SDEs can learn drift/diffusion functions from data but need careful regularization
- Ensemble multiple model types — blend GBM paths with jump-diffusion paths
- Calibrate on recent data windows (7-30 days) rather than long historical periods
- Test different path counts: 500 paths is fast for screening, 5000+ for final submissions

## Known pitfalls

- Constant volatility assumption (flat vol GBM) fails during high-volatility regimes
- Overfitting to recent price action produces narrow distributions that fail on regime changes
- Too few paths (< 200) leads to noisy CRPS estimates — score variance masks real improvements
- Ignoring overnight/weekend vol patterns degrades BTC forecasts (24/7 market has structure)
- Neural SDE training is unstable with small datasets — use at least 90 days of minute-level data
- Submitting the same distribution for BTC and ETH ignores correlation structure

## Multi-bot conflict avoidance

When running multiple bots on SN50, specialize each bot to avoid convergence conflicts:

| Bot | Role | Focus |
|-----|------|-------|
| alpha | Vol model explorer | GARCH, EWMA, realized vol, neural vol |
| beta | Path generator explorer | GBM, jump-diffusion, regime-switching, neural SDE |
| gamma | Ensemble builder | Combines best models from alpha and beta |
| delta | Baseline sentinel | Maintains current best submission, defends rank |

Conflict resolution rules:
1. **Read before exploring** — always check `ganglion knowledge` for existing patterns before planning
2. **Share antipatterns freely** — dead ends save everyone compute
3. **Specialize by capability** — each bot tags knowledge with its own `bot_id`
4. **Stall detection catches re-exploration** — the `SN50_PRESET` retry policy includes `ConfigComparisonStallDetector`

## Bootstrap workflow

Local mode (same machine as OpenClaw):

1. `export GANGLION_PROJECT=./synth-city`
2. `ganglion status $GANGLION_PROJECT` — check state
3. `ganglion knowledge $GANGLION_PROJECT` — review prior knowledge
4. `ganglion tools $GANGLION_PROJECT` — review starter tools
5. `ganglion run $GANGLION_PROJECT` — run the pipeline
6. Iterate: edit tools/agents in the project directory and re-run

Remote mode (separate server):

1. `ganglion serve ./synth-city --bot-id {{bot_id}} --port 8899`
2. `export GANGLION_URL=http://127.0.0.1:8899`
3. Verify readiness: `curl -s "$GANGLION_URL/readyz" | jq`
4. Use `/v1/` curl commands from the `ganglion` skill

## Constraints

- **min_paths**: 100 paths per submission
- **max_paths**: 10000 paths per submission
- **horizon**: 24 hours from submission time
- **assets**: BTC and ETH (weighted 60/40)
- **update_cadence**: Validators request new forecasts every ~30 minutes

## Quick experiment (remote mode)

```bash
# Run a single GBM experiment for BTC
curl -s -X POST "$GANGLION_URL/v1/run/experiment" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "asset": "BTC",
      "model_type": "gbm",
      "n_paths": 1000,
      "horizon_hours": 24,
      "volatility_model": "garch"
    }
  }' | jq .data

# Check CRPS score
curl -s "$GANGLION_URL/v1/metrics" | jq '.data[] | select(.experiment_id | startswith("sn50"))'

# Review what worked and what failed
curl -s "$GANGLION_URL/v1/knowledge?capability=generate_paths" | jq .data
```

## Recommended first pipeline run

1. Screen volatility models: constant vs EWMA vs GARCH (3 experiments, ~minutes)
2. Screen path generators: GBM vs jump-diffusion vs regime-switching (3 experiments with best vol model)
3. Full run: combine best vol + path model, generate 5000 paths, evaluate CRPS
4. Record results and iterate

The SN50_PRESET retry policy is pre-configured for this subnet — it uses escalating temperature with stall detection to avoid the agent re-submitting the same config.
