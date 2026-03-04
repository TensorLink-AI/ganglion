---
name: ganglion-synth-city
description: "Mine Bittensor Subnet 50 (Synth) with Ganglion.  Covers price-path simulation, CRPS scoring, volatility estimation, and multi-asset forecasting."
metadata: {"openclaw": {"emoji": "\u26d3", "requires": {"anyBins": ["ganglion", "curl"]}, "always": false}}
---

# Synth City — SN50 Mining Skill

Synth (netuid 50) is a probabilistic price-forecasting subnet on Bittensor.
Miners generate 1 000 Monte Carlo price paths per asset.  Validators score
submissions using CRPS (Continuous Ranked Probability Score) across multiple
time increments.  Emissions split 50/50 between low-frequency (24 h) and
high-frequency (1 h) competitions.

This skill provides SN50 domain knowledge.  Use the `ganglion` skill for
generic Ganglion API commands.

## How SN50 works

```
Validator                              Miner
─────────                              ─────
1. SimulationInput ──────────────────► 2. Fetch live price (Pyth oracle)
   (asset, start_time,                    Generate 1 000 price paths
    time_increment, time_length,
    num_simulations=1000)
                                       3. Return tuple:
                                    ◄──── (timestamp, increment,
                                           [path_1], …, [path_1000])

4. Wait for horizon to elapse
5. Fetch realised prices (Pyth benchmarks)
6. Score with CRPS per interval
7. 90th-percentile cap → normalise → 10-day rolling avg → softmax → weights
```

## Competitions

| | Low-frequency | High-frequency |
|---|---|---|
| Assets | BTC ETH SOL XAU SPYX NVDAX TSLAX AAPLX GOOGLX | BTC ETH SOL XAU |
| Horizon | 24 h (86 400 s) | 1 h (3 600 s) |
| Increment | 5 min (300 s) | 1 min (60 s) |
| Points / path | 289 | 61 |
| Paths | 1 000 | 1 000 |
| Scoring intervals | 5 min, 30 min, 3 h, 24 h abs | 1 min → 60 min (18 intervals) |
| Rolling window | 10 days (5-day half-life) | 3 days |
| Softmax β | −0.1 | −0.2 |
| Emission share | 50 % | 50 % |

## Submission format

```python
(
    start_timestamp,          # int — unix seconds, must match request
    time_increment,           # int — seconds, must match request
    [p0, p1, …, pT],         # path 1 — T = time_length / time_increment + 1
    [p0, p1, …, pT],         # path 2
    …                        # 1 000 paths total
)
# Max 8 digits per price value.
```

## CRPS scoring

CRPS is computed per time increment using `properscoring.crps_ensemble`.
Price changes are in **basis points**: `(diff / price) × 10 000`.
Absolute intervals (`_abs`) normalise by `real_price[-1] × 10 000`.
Lower CRPS = better.

## Per-asset coefficients (weight normalisation)

```
BTC: 1.000   ETH: 0.672   SOL: 0.588   XAU: 2.262
SPYX: 2.991  NVDAX: 1.389 TSLAX: 1.420 AAPLX: 1.865  GOOGLX: 1.431
```

## Reference sigma values (hourly, GBM baseline)

```
BTC: 0.00472  ETH: 0.00695  SOL: 0.00782  XAU: 0.00208
SPYX: 0.00156 NVDAX: 0.00342 TSLAX: 0.00332 AAPLX: 0.00250 GOOGLX: 0.00332
```

Competitive miners should NOT use fixed sigma.  Estimate from recent data.

## Tools in this project

| Tool | Category | Purpose |
|------|----------|---------|
| `run_experiment` | training | Generate Monte Carlo paths with configurable model, sigma, asset |
| `fetch_price` | data | Get live spot price from Pyth Hermes oracle |
| `estimate_volatility` | training | Estimate sigma from recent returns (realized vol or EWMA) |
| `score_paths` | evaluation | Compute CRPS against realised prices locally |

## Search strategies

1. Start with GBM + reference sigma as baseline — establishes a CRPS floor
2. Replace fixed sigma with EWMA or realised vol from recent 5-min returns
3. Try GARCH(1,1) for volatility clustering — this is what CRPS rewards
4. Add jump-diffusion (Merton) for sudden price moves GBM misses
5. Experiment with regime-switching (bull/bear/sideways) for adaptive vol
6. Use Student-t or skewed-t innovations instead of normal (fat tails)
7. Ensemble: blend paths from multiple models (e.g. 500 GBM + 500 jump)
8. Calibrate on recent windows (7-30 days) rather than long history
9. Tune per-asset: BTC and SOL need different models than XAU or SPYX

## Known pitfalls

- Constant sigma fails during high-volatility regimes — CRPS will spike
- Overfitting to recent action → narrow distributions that blow up on regime change
- Too few paths (< 500) → noisy CRPS estimates; use 1 000 (the required amount)
- Ignoring 24/7 BTC vol structure (weekends, overnight) degrades scores
- Neural SDE training is unstable with < 90 days of minute-level data
- Same distribution for BTC and ETH ignores different vol characteristics
- Exceeding 8 digits per price value → validator rejects the submission

## Bootstrap workflow

```bash
# Scaffold (if using ganglion init)
ganglion init ./synth-city --subnet sn50 --netuid 50

# Or just copy this example directory and run:
export GANGLION_PROJECT=./examples/synth-city
ganglion status $GANGLION_PROJECT
ganglion tools $GANGLION_PROJECT
ganglion run $GANGLION_PROJECT

# Remote mode
ganglion serve ./examples/synth-city --bot-id alpha --port 8899
export GANGLION_URL=http://127.0.0.1:8899
```

## Quick experiment (remote mode)

```bash
# GBM baseline for BTC
curl -s -X POST "$GANGLION_URL/v1/run/experiment" \
  -H "Content-Type: application/json" \
  -d '{"config": {"asset": "BTC", "model_type": "gbm", "num_simulations": 1000}}' \
  | jq .data

# Check knowledge for what has been tried
curl -s "$GANGLION_URL/v1/knowledge?capability=simulate" | jq .data
```

## Key dependencies

- `numpy` — path generation, array ops
- `properscoring` — CRPS calculation (`pip install properscoring`)
- Pyth Hermes API — live prices (`https://hermes.pyth.network/v2/updates/price/latest`)
