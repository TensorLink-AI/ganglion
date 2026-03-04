"""Price-path simulation tool for Synth (SN50).

Generates Monte Carlo price paths using configurable stochastic models.
The baseline is Geometric Brownian Motion (GBM) with constant sigma —
competitive miners should replace the simulation engine with models that
capture volatility clustering and fat-tailed distributions.

─── SN50 Subnet Reference ───────────────────────────────────────────

Repo:     https://github.com/mode-network/synth-subnet/tree/main/synth
Netuid:   50
Emission: 50 % low-frequency (24 h) + 50 % high-frequency (1 h)

Competitions:
  LOW-FREQUENCY (24 h)
    Assets:         BTC ETH SOL XAU SPYX NVDAX TSLAX AAPLX GOOGLX
    time_increment: 300 s (5 min)
    time_length:    86 400 s (24 h)
    Points/path:    289  (86400 / 300 + 1)
    num_simulations:1 000
    Scoring:        5 min, 30 min, 3 h, 24 h absolute
    Rolling window: 10 days, 5-day half-life
    Softmax beta:   -0.1
    Cycle:          every 60 min

  HIGH-FREQUENCY (1 h)
    Assets:         BTC ETH SOL XAU
    time_increment: 60 s (1 min)
    time_length:    3 600 s (1 h)
    Points/path:    61  (3600 / 60 + 1)
    num_simulations:1 000
    Scoring:        1 min → 60 min (18 intervals incl. _abs and _gap)
    Rolling window: 3 days
    Softmax beta:   -0.2
    Cycle:          every 12 min

Submission format (synapse.simulation_output):
  (start_timestamp_int, time_increment_int, [path1], [path2], …, [path1000])
  - Each path: list of floats, length = time_length // time_increment + 1
  - Max 8 significant digits per price value
  - start_timestamp must match the validator's SimulationInput.start_time
  - time_increment must match SimulationInput.time_increment

Scoring pipeline:
  1. Validator waits for horizon to elapse, fetches realised prices from
     Pyth Benchmarks API (1-min resolution)
  2. Price changes computed in basis points: (diff / price) × 10 000
     Exception: intervals ending in '_abs' use raw absolute prices
     normalised by real_price[-1] × 10 000
  3. CRPS computed per interval via properscoring.crps_ensemble
  4. NaN gaps handled by label_observed_blocks — only scored over
     consecutive non-NaN blocks
  5. Per-miner total CRPS = sum across all intervals
  6. Percentile cap: scores > 90th percentile capped; invalid (-1) → p90
  7. Normalised: shifted so best miner = 0
  8. Per-asset coefficient applied (BTC=1.0, ETH=0.672, XAU=2.262, …)
  9. 10-day rolling average with 5-day half-life exponential decay
  10. Softmax with negative beta: lower CRPS → higher reward weight
  11. Low-freq + high-freq weights summed → final weight → set_weights on chain

Per-asset coefficients:
  BTC: 1.000  ETH: 0.672  SOL: 0.588  XAU: 2.262
  SPYX: 2.991 NVDAX: 1.389 TSLAX: 1.420 AAPLX: 1.865 GOOGLX: 1.431

Reference sigma (hourly, upstream GBM baseline):
  BTC: 0.00472  ETH: 0.00695  SOL: 0.00782  XAU: 0.00208
  SPYX: 0.00156 NVDAX: 0.00342 TSLAX: 0.00332 AAPLX: 0.00250 GOOGLX: 0.00332

Price oracle (miner-side): Pyth Hermes
  https://hermes.pyth.network/v2/updates/price/latest?ids[]=<feed_id>

Price oracle (validator-side): Pyth Benchmarks (TradingView shim)
  https://benchmarks.pyth.network/v1/shims/tradingview/history
  Params: symbol, resolution=1, from, to
  Symbols: Crypto.BTC/USD, Crypto.ETH/USD, etc.

Key libraries: numpy, properscoring, httpx/urllib
──────────────────────────────────────────────────────────────────────
"""

import math

import numpy as np

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult

# Per-asset hourly sigma from upstream synth/miner/simulations.py SIGMA_MAP.
# Competitive miners should estimate their own from recent price data using
# the estimate_volatility or fetch_historical_prices tools.
SIGMA_MAP: dict[str, float] = {
    "BTC": 0.00472,
    "ETH": 0.00695,
    "SOL": 0.00782,
    "XAU": 0.00208,
    "SPYX": 0.00156,
    "NVDAX": 0.00342,
    "TSLAX": 0.00332,
    "AAPLX": 0.00250,
    "GOOGLX": 0.00332,
}


def _simulate_gbm(
    current_price: float,
    sigma: float,
    time_increment: int,
    time_length: int,
    num_simulations: int,
) -> np.ndarray:
    """Geometric Brownian Motion — the SN50 baseline model.

    Matches the upstream implementation in synth/miner/price_simulation.py:
      dt = time_increment / 3600
      pct_change = N(0, sigma * sqrt(dt))
      price[t+1] = price[t] * (1 + pct_change)

    Returns an (num_simulations, num_steps+1) array of price paths.
    """
    dt = time_increment / 3600
    num_steps = time_length // time_increment
    std = sigma * math.sqrt(dt)

    pct_changes = np.random.normal(0, std, size=(num_simulations, num_steps))
    cum_returns = np.cumprod(1 + pct_changes, axis=1)

    paths = np.empty((num_simulations, num_steps + 1))
    paths[:, 0] = current_price
    paths[:, 1:] = current_price * cum_returns
    return paths


@tool("run_experiment", category="training")
def run_experiment(config: dict) -> ExperimentResult:
    """Run a price-path forecasting experiment for SN50.

    Expected config keys:
        asset: str             — target asset (default "BTC")
                                 Supported: BTC ETH SOL XAU SPYX NVDAX
                                 TSLAX AAPLX GOOGLX
        model_type: str        — simulation model: "gbm" (default)
        current_price: float   — spot price at simulation start
        time_increment: int    — seconds between steps (default 300)
        time_length: int       — total horizon in seconds (default 86400)
        num_simulations: int   — number of paths (default 1000)
        sigma: float | None    — override per-asset sigma
    """
    asset = config.get("asset", "BTC")
    model_type = config.get("model_type", "gbm")
    current_price = config.get("current_price", 100_000.0)
    time_increment = config.get("time_increment", 300)
    time_length = config.get("time_length", 86400)
    num_simulations = config.get("num_simulations", 1000)
    sigma = config.get("sigma") or SIGMA_MAP.get(asset, 0.005)

    num_steps = time_length // time_increment

    if model_type == "gbm":
        paths = _simulate_gbm(current_price, sigma, time_increment, time_length, num_simulations)
    else:
        return ExperimentResult(
            content=f"Unknown model_type '{model_type}'.  Supported: gbm",
            experiment_id=None,
            metrics={},
        )

    # Compute self-assessment metrics (approximate — real CRPS needs realized prices)
    final_prices = paths[:, -1]
    mean_final = float(np.mean(final_prices))
    std_final = float(np.std(final_prices))
    pct_range_90 = float(np.percentile(final_prices, 95) - np.percentile(final_prices, 5))

    return ExperimentResult(
        content=(
            f"Generated {num_simulations} {model_type.upper()} price paths "
            f"for {asset} ({num_steps + 1} steps, {time_increment}s increments, "
            f"{time_length // 3600}h horizon).  "
            f"Mean final price: {mean_final:,.2f}, Std: {std_final:,.2f}, "
            f"90% range: {pct_range_90:,.2f}"
        ),
        experiment_id=f"sn50-{asset.lower()}-{model_type}-{num_simulations}p",
        metrics={
            "mean_final_price": round(mean_final, 2),
            "std_final_price": round(std_final, 2),
            "pct_range_90": round(pct_range_90, 2),
            "num_paths": num_simulations,
            "num_steps": num_steps + 1,
        },
        artifact_path=None,
    )
