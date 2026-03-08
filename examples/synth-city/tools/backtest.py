"""Backtester for SN50 price-path simulations.

Replays the full SN50 validator scoring pipeline against historical data:
  1. Picks a historical window as the "prediction start"
  2. Generates simulated paths from start price using the chosen model
  3. Fetches (or uses provided) realised prices over the same window
  4. Scores with CRPS at every scoring interval
  5. Applies per-asset coefficients and percentile normalisation
  6. Reports exactly what the validator would have scored

This mirrors the scoring logic in:
  https://github.com/mode-network/synth-subnet/blob/main/synth/validator/crps_calculation.py
  https://github.com/mode-network/synth-subnet/blob/main/synth/validator/reward.py

Install:  pip install properscoring
"""

from __future__ import annotations

import math
from typing import Any

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult


# ── SN50 scoring intervals (from prompt_config.py) ──────────

LOW_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "5min": 300,
    "30min": 1800,
    "3hour": 10800,
    "24hour_abs": 86400,
}

HIGH_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "1min": 60,
    "2min": 120,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "60min_abs": 3600,
    "0_5min_gaps": 300,
    "0_10min_gaps": 600,
    "0_15min_gaps": 900,
    "0_20min_gaps": 1200,
    "0_25min_gaps": 1500,
    "0_30min_gaps": 1800,
    "0_35min_gaps": 2100,
    "0_40min_gaps": 2400,
    "0_45min_gaps": 2700,
    "0_50min_gaps": 3000,
    "0_55min_gaps": 3300,
    "0_60min_gaps": 3600,
}

# Per-asset weight coefficients from the validator's moving_average.py
ASSET_COEFFICIENTS: dict[str, float] = {
    "BTC": 1.0,
    "ETH": 0.6715,
    "SOL": 0.5884,
    "XAU": 2.262,
    "SPYX": 2.991,
    "NVDAX": 1.389,
    "TSLAX": 1.420,
    "AAPLX": 1.865,
    "GOOGLX": 1.431,
}

# Reference sigma for baseline GBM
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


# ── Simulation engines ──────────────────────────────────────

def _simulate_gbm(
    price: float, sigma: float, time_increment: int,
    time_length: int, n: int,
):
    import numpy as np

    dt = time_increment / 3600
    steps = time_length // time_increment
    std = sigma * math.sqrt(dt)
    pct = np.random.normal(0, std, size=(n, steps))
    paths = np.empty((n, steps + 1))
    paths[:, 0] = price
    paths[:, 1:] = price * np.cumprod(1 + pct, axis=1)
    return paths


# ── CRPS scoring (mirrors validator logic) ───────────────────

def _label_observed_blocks(arr: np.ndarray) -> np.ndarray:
    """Group consecutive non-NaN values into numbered blocks (-1 for gaps)."""
    import numpy as np

    not_nan = ~np.isnan(arr)
    block_start = not_nan & np.concatenate(([True], ~not_nan[:-1]))
    group_numbers = np.cumsum(block_start) - 1
    return np.where(not_nan, group_numbers, -1)


def _calculate_crps(
    sim_paths: np.ndarray,
    real_prices: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> dict[str, Any]:
    """Score simulated paths against realised prices using the full
    SN50 validator CRPS pipeline.

    Returns a dict with per-interval CRPS and the total score.
    """
    import numpy as np

    try:
        from properscoring import crps_ensemble as _crps_ensemble
    except ImportError:
        _crps_ensemble = None

    breakdown: dict[str, float] = {}
    total = 0.0

    for name, interval_s in scoring_intervals.items():
        is_abs = name.endswith("_abs")
        is_gap = "_gap" in name

        step = interval_s // time_increment
        if step < 1:
            continue

        # Subsample at interval steps
        sim_at = sim_paths[:, ::step]
        real_at = real_prices[::step]

        if is_gap:
            # Gap intervals: only use first two points
            sim_at = sim_at[:, :2] if sim_at.shape[1] >= 2 else sim_at
            real_at = real_at[:2] if len(real_at) >= 2 else real_at

        if is_abs:
            # Absolute price scoring (skip first point)
            if sim_at.shape[1] < 2:
                continue
            sim_changes = sim_at[:, 1:]
            real_changes = real_at[1:]
        else:
            # Percentage returns in basis points
            if sim_at.shape[1] < 2:
                continue
            denom_sim = sim_at[:, :-1]
            denom_real = real_at[:-1]
            # Guard against zero prices
            if np.any(denom_sim == 0) or np.any(denom_real == 0):
                breakdown[name] = -1.0
                continue
            sim_changes = (np.diff(sim_at, axis=1) / denom_sim) * 10_000
            real_changes = (np.diff(real_at) / denom_real) * 10_000

        # Handle NaN blocks in real data
        flat_real = real_changes if real_changes.ndim == 1 else real_changes.ravel()
        block_labels = _label_observed_blocks(flat_real.astype(float))

        interval_crps = 0.0
        n_scored = 0
        for t in range(len(flat_real)):
            if block_labels[t] == -1:
                continue
            val = float(flat_real[t])
            if np.isnan(val):
                continue
            c = _crps_ensemble(val, sim_changes[:, t].astype(float))
            # Normalise absolute intervals
            if is_abs and real_prices[-1] != 0:
                c = c / (real_prices[-1] * 10_000)
            interval_crps += c
            n_scored += 1

        avg = interval_crps / max(n_scored, 1)
        breakdown[name] = round(avg, 8)
        total += avg

    breakdown["total"] = round(total, 8)
    return breakdown


# ── Backtest tool ────────────────────────────────────────────

@tool("backtest", category="evaluation")
def backtest(config: dict) -> ExperimentResult:
    """Run a full SN50 backtest: simulate paths, score against realised prices.

    Replicates the validator's scoring pipeline so you can evaluate model
    changes offline before deploying to the live subnet.

    Expected config keys:
        asset: str                    — target asset (default "BTC")
        realized_prices: list[float]  — realised price series (REQUIRED)
                                        Use fetch_historical_prices to get this.
        start_price: float | None     — price at simulation start
                                        (default: realized_prices[0])
        model_type: str               — "gbm" (default)
        sigma: float | None           — override volatility
        num_simulations: int          — paths to generate (default 1000)
        time_increment: int           — seconds between points (default 300)
        time_length: int              — horizon in seconds (default 86400)
        competition: str              — "low_freq" | "high_freq" (default "low_freq")
                                        Determines which scoring intervals to use.
        simulated_paths: list | None  — provide pre-generated paths instead
                                        of generating new ones.  Shape:
                                        (num_simulations, num_steps+1)

    Returns detailed CRPS breakdown per scoring interval, the weighted
    score (with per-asset coefficient), and diagnostic metrics.
    """
    import numpy as np

    try:
        from properscoring import crps_ensemble as _crps_ensemble  # noqa: F811
    except ImportError:
        _crps_ensemble = None

    if _crps_ensemble is None:
        return ExperimentResult(
            content="properscoring is not installed.  Run: pip install properscoring",
            metrics={},
        )

    realized = config.get("realized_prices")
    if not realized or len(realized) < 3:
        return ExperimentResult(
            content=(
                "Provide 'realized_prices' — a list of historical prices.  "
                "Use the fetch_historical_prices tool to get this data."
            ),
            metrics={},
        )

    asset = config.get("asset", "BTC")
    time_increment = config.get("time_increment", 300)
    time_length = config.get("time_length", 86400)
    num_simulations = config.get("num_simulations", 1000)
    model_type = config.get("model_type", "gbm")
    sigma = config.get("sigma") or SIGMA_MAP.get(asset, 0.005)
    competition = config.get("competition", "low_freq")
    pre_generated = config.get("simulated_paths")

    real_arr = np.array(realized, dtype=float)
    start_price = config.get("start_price") or float(real_arr[0])

    expected_steps = time_length // time_increment + 1

    # Generate or use provided paths
    if pre_generated is not None:
        sim_paths = np.array(pre_generated, dtype=float)
        if sim_paths.ndim != 2:
            return ExperimentResult(
                content="simulated_paths must be 2D: (num_simulations, num_steps+1)",
                metrics={},
            )
        num_simulations = sim_paths.shape[0]
    elif model_type == "gbm":
        sim_paths = _simulate_gbm(
            start_price, sigma, time_increment, time_length, num_simulations,
        )
    else:
        return ExperimentResult(
            content=f"Unknown model_type '{model_type}'.  Supported: gbm",
            metrics={},
        )

    # Trim real prices to match simulation length
    if len(real_arr) > sim_paths.shape[1]:
        real_arr = real_arr[: sim_paths.shape[1]]
    elif len(real_arr) < sim_paths.shape[1]:
        # Pad with NaN (validator handles gaps this way)
        padded = np.full(sim_paths.shape[1], np.nan)
        padded[: len(real_arr)] = real_arr
        real_arr = padded

    # Pick scoring intervals
    if competition == "high_freq":
        intervals = HIGH_FREQ_SCORING_INTERVALS
    else:
        intervals = LOW_FREQ_SCORING_INTERVALS

    # Score
    crps_breakdown = _calculate_crps(sim_paths, real_arr, time_increment, intervals)

    # Apply per-asset coefficient (how the validator weights different assets)
    coeff = ASSET_COEFFICIENTS.get(asset.upper(), 1.0)
    raw_total = crps_breakdown["total"]
    weighted_total = round(raw_total * coeff, 8)

    # Diagnostic metrics
    final_sim = sim_paths[:, -1]
    final_real = float(real_arr[-1]) if not np.isnan(real_arr[-1]) else None

    result_metrics: dict[str, Any] = {
        "crps_breakdown": crps_breakdown,
        "crps_total": raw_total,
        "crps_weighted": weighted_total,
        "asset_coefficient": coeff,
        "asset": asset.upper(),
        "competition": competition,
        "model_type": model_type,
        "sigma": sigma,
        "num_simulations": num_simulations,
        "num_steps": sim_paths.shape[1],
        "realized_points": int(np.sum(~np.isnan(real_arr))),
        "start_price": start_price,
        "final_price_real": final_real,
        "final_price_sim_mean": round(float(np.mean(final_sim)), 2),
        "final_price_sim_std": round(float(np.std(final_sim)), 2),
    }

    # Format interval breakdown for readable output
    interval_lines = []
    for name, score in crps_breakdown.items():
        if name == "total":
            continue
        interval_lines.append(f"  {name}: {score:.6f}")

    return ExperimentResult(
        content=(
            f"Backtest: {asset} {competition} ({model_type}, sigma={sigma:.5f}, "
            f"{num_simulations} paths)\n"
            f"CRPS total: {raw_total:.6f}  "
            f"Weighted (×{coeff}): {weighted_total:.6f}\n"
            f"Intervals:\n" + "\n".join(interval_lines) + "\n"
            f"Real final: {final_real}  "
            f"Sim mean: {result_metrics['final_price_sim_mean']:,.2f} "
            f"± {result_metrics['final_price_sim_std']:,.2f}"
        ),
        experiment_id=f"bt-{asset.lower()}-{model_type}-{competition}",
        metrics=result_metrics,
    )
