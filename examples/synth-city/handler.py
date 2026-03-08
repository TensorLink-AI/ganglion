"""RunPod training worker for Synth City (SN50).

This is NOT an MCP server. It's a headless job worker that:
  1. Boots with datasets baked into the image (historical prices, sigma maps)
  2. Receives a job spec via HTTP POST or env-var config
  3. Runs simulate → backtest → writes results to /outputs
  4. Returns a JSON result on stdout (for RunPodBackend.collect())

The ganglion MCP server dispatches jobs here via RunPodBackend.submit().
Agents never talk to this container directly.

Modes:
  train    — simulate price paths with a given model config, save checkpoints
  validate — load checkpoints, score against baked-in or fetched historical data
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("synth-worker")

OUTPUT_DIR = Path(os.environ.get("GANGLION_OUTPUT_DIR", "/outputs"))
CHECKPOINT_DIR = Path(os.environ.get("GANGLION_CHECKPOINT_DIR", "/app/checkpoints"))
DATASET_DIR = Path("/app/datasets")

# ── Baked-in constants (from SN50 spec) ────────────────────

LF_ASSETS = ["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
HF_ASSETS = ["BTC", "ETH", "SOL", "XAU"]

SIGMA_MAP: dict[str, float] = {
    "BTC": 0.00472, "ETH": 0.00695, "SOL": 0.00782, "XAU": 0.00208,
    "SPYX": 0.00156, "NVDAX": 0.00342, "TSLAX": 0.00332,
    "AAPLX": 0.00250, "GOOGLX": 0.00332,
}

ASSET_COEFFICIENTS: dict[str, float] = {
    "BTC": 1.0, "ETH": 0.6715, "SOL": 0.5884, "XAU": 2.262,
    "SPYX": 2.991, "NVDAX": 1.389, "TSLAX": 1.420,
    "AAPLX": 1.865, "GOOGLX": 1.431,
}

LOW_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "5min": 300, "30min": 1800, "3hour": 10800, "24hour_abs": 86400,
}
HIGH_FREQ_SCORING_INTERVALS: dict[str, int] = {
    "1min": 60, "2min": 120, "5min": 300, "15min": 900,
    "30min": 1800, "60min_abs": 3600,
}


def _competition_params(comp: str) -> dict[str, int]:
    if comp == "high_freq":
        return {"time_increment": 60, "time_length": 3600}
    return {"time_increment": 300, "time_length": 86400}


# ── Simulation ─────────────────────────────────────────────


def simulate_gbm(
    price: float, sigma: float, time_increment: int,
    time_length: int, n: int,
) -> np.ndarray:
    """Geometric Brownian Motion — the SN50 baseline."""
    dt = time_increment / 3600
    steps = time_length // time_increment
    std = sigma * math.sqrt(dt)
    pct = np.random.normal(0, std, size=(n, steps))
    paths = np.empty((n, steps + 1))
    paths[:, 0] = price
    paths[:, 1:] = price * np.cumprod(1 + pct, axis=1)
    return paths


# ── CRPS scoring (mirrors SN50 validator) ──────────────────


def calculate_crps(
    sim_paths: np.ndarray,
    real_prices: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> dict[str, Any]:
    """Score simulated paths against realised prices using CRPS."""
    try:
        from properscoring import crps_ensemble
    except ImportError:
        return {"error": "properscoring not installed", "total": -1.0}

    breakdown: dict[str, float] = {}
    total = 0.0

    for name, interval_s in scoring_intervals.items():
        is_abs = name.endswith("_abs")
        step = interval_s // time_increment
        if step < 1:
            continue

        sim_at = sim_paths[:, ::step]
        real_at = real_prices[::step]

        if sim_at.shape[1] < 2:
            continue

        if is_abs:
            sim_changes = sim_at[:, 1:]
            real_changes = real_at[1:]
        else:
            denom_sim = sim_at[:, :-1]
            denom_real = real_at[:-1]
            if np.any(denom_sim == 0) or np.any(denom_real == 0):
                breakdown[name] = -1.0
                continue
            sim_changes = (np.diff(sim_at, axis=1) / denom_sim) * 10_000
            real_changes = (np.diff(real_at) / denom_real) * 10_000

        flat_real = real_changes if real_changes.ndim == 1 else real_changes.ravel()
        interval_crps = 0.0
        n_scored = 0
        for t in range(len(flat_real)):
            val = float(flat_real[t])
            if np.isnan(val):
                continue
            c = crps_ensemble(val, sim_changes[:, t].astype(float))
            if is_abs and real_prices[-1] != 0:
                c = c / (real_prices[-1] * 10_000)
            interval_crps += c
            n_scored += 1

        avg = interval_crps / max(n_scored, 1)
        breakdown[name] = round(avg, 8)
        total += avg

    breakdown["total"] = round(total, 8)
    return breakdown


# ── Dataset loading ────────────────────────────────────────


def load_baked_prices(asset: str, competition: str) -> np.ndarray | None:
    """Load pre-baked historical prices from /app/datasets/.

    Expected format: .npy file with shape (num_points,) of float64 prices.
    Naming: datasets/{asset}_{competition}_prices.npy
    """
    path = DATASET_DIR / f"{asset.upper()}_{competition}_prices.npy"
    if path.exists():
        prices = np.load(str(path))
        logger.info("Loaded baked prices: %s (%d points)", path.name, len(prices))
        return prices
    logger.info("No baked prices at %s", path)
    return None


# ── Train ──────────────────────────────────────────────────


def run_train(spec: dict[str, Any]) -> dict[str, Any]:
    """Run simulation for specified assets. Saves .npy checkpoints.

    Spec keys:
        assets: list[str]       — assets to simulate (default: all for competition)
        competition: str        — "low_freq" | "high_freq" (default: "low_freq")
        num_simulations: int    — paths per asset (default: 1000)
        model_type: str         — "gbm" (default, more models pluggable)
        sigma_overrides: dict   — per-asset sigma overrides
        current_prices: dict    — per-asset spot prices (required for real runs)
    """
    competition = spec.get("competition", "low_freq")
    params = _competition_params(competition)
    num_sims = spec.get("num_simulations", 1000)
    model_type = spec.get("model_type", "gbm")
    sigma_overrides = spec.get("sigma_overrides", {})
    current_prices = spec.get("current_prices", {})

    default_assets = HF_ASSETS if competition == "high_freq" else LF_ASSETS
    assets = spec.get("assets", default_assets)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    for asset in assets:
        asset = asset.upper()
        start = time.monotonic()
        sigma = sigma_overrides.get(asset, SIGMA_MAP.get(asset, 0.005))
        price = current_prices.get(asset, 100_000.0)

        logger.info("Train: %s competition=%s model=%s sigma=%.5f price=%.2f",
                     asset, competition, model_type, sigma, price)

        if model_type != "gbm":
            results[asset] = {"error": f"Unknown model: {model_type}"}
            continue

        paths = simulate_gbm(
            price=price,
            sigma=sigma,
            time_increment=params["time_increment"],
            time_length=params["time_length"],
            n=num_sims,
        )

        # Save checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{asset}_{competition}_paths.npy"
        np.save(str(ckpt_path), paths)

        final = paths[:, -1]
        elapsed = time.monotonic() - start

        results[asset] = {
            "status": "ok",
            "model": model_type,
            "sigma": sigma,
            "start_price": price,
            "num_simulations": num_sims,
            "num_steps": paths.shape[1],
            "mean_final": round(float(np.mean(final)), 2),
            "std_final": round(float(np.std(final)), 2),
            "p5": round(float(np.percentile(final, 5)), 2),
            "p95": round(float(np.percentile(final, 95)), 2),
            "checkpoint": str(ckpt_path),
            "duration_s": round(elapsed, 2),
        }
        logger.info("  %s done in %.1fs — mean=%.2f std=%.2f",
                     asset, elapsed, results[asset]["mean_final"], results[asset]["std_final"])

    return {"mode": "train", "competition": competition, "assets": results}


# ── Validate ───────────────────────────────────────────────


def run_validate(spec: dict[str, Any]) -> dict[str, Any]:
    """Score checkpoints against historical data (baked-in or provided).

    Spec keys:
        assets: list[str]          — assets to validate
        competition: str           — "low_freq" | "high_freq"
        realized_prices: dict      — optional per-asset price lists
                                     (if omitted, uses baked-in datasets)
    """
    competition = spec.get("competition", "low_freq")
    params = _competition_params(competition)
    provided_prices = spec.get("realized_prices", {})

    default_assets = HF_ASSETS if competition == "high_freq" else LF_ASSETS
    assets = spec.get("assets", default_assets)

    intervals = HIGH_FREQ_SCORING_INTERVALS if competition == "high_freq" else LOW_FREQ_SCORING_INTERVALS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for asset in assets:
        asset = asset.upper()

        # Load checkpoint
        ckpt_path = CHECKPOINT_DIR / f"{asset}_{competition}_paths.npy"
        if not ckpt_path.exists():
            results[asset] = {"error": f"No checkpoint at {ckpt_path}"}
            continue

        paths = np.load(str(ckpt_path))
        logger.info("Validate: %s — loaded %s (%s paths, %d steps)",
                     asset, ckpt_path.name, paths.shape[0], paths.shape[1])

        # Get realized prices: provided > baked-in > fallback to first path
        if asset in provided_prices:
            real = np.array(provided_prices[asset], dtype=float)
        else:
            real = load_baked_prices(asset, competition)

        if real is None:
            logger.warning("  No realized prices for %s, using first simulated path as proxy", asset)
            real = paths[0, :].copy()

        # Align lengths
        min_len = min(len(real), paths.shape[1])
        real = real[:min_len]
        paths = paths[:, :min_len]

        # Score
        start = time.monotonic()
        crps = calculate_crps(paths, real, params["time_increment"], intervals)
        elapsed = time.monotonic() - start

        coeff = ASSET_COEFFICIENTS.get(asset, 1.0)
        raw_total = crps.get("total", -1.0)
        weighted = round(raw_total * coeff, 8) if raw_total >= 0 else -1.0

        results[asset] = {
            "status": "ok",
            "crps_breakdown": crps,
            "crps_total": raw_total,
            "crps_weighted": weighted,
            "asset_coefficient": coeff,
            "realized_source": "provided" if asset in provided_prices
                               else "baked" if load_baked_prices(asset, competition) is not None
                               else "proxy",
            "num_paths": paths.shape[0],
            "num_steps": paths.shape[1],
            "duration_s": round(elapsed, 2),
        }
        logger.info("  %s CRPS=%.6f weighted=%.6f (%.1fs)", asset, raw_total, weighted, elapsed)

    return {"mode": "validate", "competition": competition, "assets": results}


# ── Job entrypoint ─────────────────────────────────────────


def run_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Single dispatch point. Called with the full job spec."""
    mode = spec.get("mode", "train")
    if mode == "train":
        return run_train(spec)
    elif mode == "validate":
        return run_validate(spec)
    elif mode == "train_and_validate":
        train_result = run_train(spec)
        val_result = run_validate(spec)
        return {"train": train_result, "validate": val_result}
    else:
        return {"error": f"Unknown mode: {mode}"}


def main() -> None:
    """Entry point. Reads job spec from:
      1. GANGLION_JOB_SPEC env var (JSON string) — set by RunPodBackend
      2. /input/spec.json — mounted by RunPod volume
      3. Fallback defaults from individual env vars
    """
    spec: dict[str, Any] = {}

    # Priority 1: env var (how RunPodBackend passes the spec via dockerArgs)
    env_spec = os.environ.get("GANGLION_JOB_SPEC", "").strip()
    if env_spec:
        try:
            spec = json.loads(env_spec)
            logger.info("Loaded job spec from GANGLION_JOB_SPEC env var")
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in GANGLION_JOB_SPEC: %s", e)
            sys.exit(1)

    # Priority 2: mounted file
    if not spec:
        spec_file = Path("/input/spec.json")
        if spec_file.exists():
            spec = json.loads(spec_file.read_text())
            logger.info("Loaded job spec from %s", spec_file)

    # Priority 3: env var fallback
    if not spec:
        spec = {
            "mode": os.environ.get("GANGLION_MODE", "train"),
            "competition": os.environ.get("GANGLION_COMPETITION", "low_freq"),
            "num_simulations": int(os.environ.get("GANGLION_NUM_SIMULATIONS", "1000")),
        }
        assets_env = os.environ.get("GANGLION_ASSETS", "").strip()
        if assets_env:
            spec["assets"] = [a.strip().upper() for a in assets_env.split(",") if a.strip()]
        logger.info("Using env-var fallback spec")

    logger.info("Job spec: %s", json.dumps(spec, default=str)[:500])

    result = run_job(spec)

    # Write result to /outputs for RunPodBackend.collect()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))

    # Also print to stdout (RunPodBackend reads stdout from collect())
    print(json.dumps(result, default=str))

    logger.info("Job complete. Results at %s", result_path)


if __name__ == "__main__":
    main()
