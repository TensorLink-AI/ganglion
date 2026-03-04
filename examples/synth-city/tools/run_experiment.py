"""Price-path simulation tool for Synth (SN50).

Generates Monte Carlo price paths using configurable stochastic models.
The baseline is Geometric Brownian Motion (GBM) with constant sigma —
competitive miners should replace the simulation engine with models that
capture volatility clustering and fat-tailed distributions (GARCH,
Heston, regime-switching, etc.).

Reference implementation:
  https://github.com/mode-network/synth-subnet/blob/main/synth/miner/price_simulation.py
"""

import math

import numpy as np

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult

# Per-asset annualised sigma values from the upstream miner reference.
# Competitive miners should estimate their own from recent price data.
SIGMA_MAP: dict[str, float] = {
    "BTC": 0.00472,
    "ETH": 0.00695,
    "SOL": 0.00782,
    "XAU": 0.00208,
    "SPYX": 0.00156,
    "NVDAX": 0.00342,
    "TSLAX": 0.00285,
    "AAPLX": 0.00231,
    "GOOGLX": 0.00247,
}


def _simulate_gbm(
    current_price: float,
    sigma: float,
    time_increment: int,
    time_length: int,
    num_simulations: int,
) -> np.ndarray:
    """Geometric Brownian Motion — the SN50 baseline model.

    Returns an (num_simulations, num_steps+1) array of price paths.
    """
    dt = time_increment / 3600
    num_steps = time_length // time_increment
    std = sigma * math.sqrt(dt)

    # Vectorised path generation
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
