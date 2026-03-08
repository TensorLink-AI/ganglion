"""Lightweight RunPod entrypoint for Synth City (SN50).

Modes:
  train    — run the full pipeline: fetch prices → calibrate → simulate → backtest
  validate — backtest-only against a checkpoint or live model
  serve    — start ganglion MCP server for remote agent access

Designed to be minimal: no heavy frameworks, just ganglion + numpy.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("synth-handler")

PROJECT_DIR = "/app/project"
OUTPUT_DIR = Path(os.environ.get("GANGLION_OUTPUT_DIR", "/app/outputs"))

# ── Asset lists per competition ────────────────────────────

LF_ASSETS = ["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
HF_ASSETS = ["BTC", "ETH", "SOL", "XAU"]


def _resolve_assets() -> list[tuple[str, str]]:
    """Return list of (asset, competition) pairs to process."""
    env_assets = os.environ.get("GANGLION_ASSETS", "").strip()
    competition = os.environ.get("GANGLION_COMPETITION", "low_freq").strip()

    if env_assets:
        assets = [a.strip().upper() for a in env_assets.split(",") if a.strip()]
    elif competition == "high_freq":
        assets = HF_ASSETS
    elif competition == "both":
        assets = list(dict.fromkeys(LF_ASSETS + HF_ASSETS))
    else:
        assets = LF_ASSETS

    pairs = []
    for asset in assets:
        if competition == "both":
            if asset in LF_ASSETS:
                pairs.append((asset, "low_freq"))
            if asset in HF_ASSETS:
                pairs.append((asset, "high_freq"))
        elif competition == "high_freq":
            pairs.append((asset, "high_freq"))
        else:
            pairs.append((asset, "low_freq"))
    return pairs


def _competition_params(comp: str) -> dict[str, int]:
    """Return time_increment and time_length for a competition type."""
    if comp == "high_freq":
        return {"time_increment": 60, "time_length": 3600}
    return {"time_increment": 300, "time_length": 86400}


# ── Train mode ─────────────────────────────────────────────


async def run_train() -> dict[str, Any]:
    """Run the full ganglion pipeline for each asset.

    Uses ganglion's pipeline runner when an LLM key is available,
    otherwise falls back to direct tool invocation (no agent needed).
    """
    from ganglion.state.framework_state import FrameworkState

    pairs = _resolve_assets()
    num_sims = int(os.environ.get("GANGLION_NUM_SIMULATIONS", "1000"))
    has_llm = bool(os.environ.get("LLM_PROVIDER_API_KEY", "").strip())

    results: dict[str, Any] = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if has_llm:
        # Full pipeline mode — agent-driven
        logger.info("LLM key detected, running full agent pipeline")
        state = FrameworkState.load(PROJECT_DIR)
        await state.initialize_mcp()
        try:
            for asset, comp in pairs:
                logger.info("Pipeline: %s (%s)", asset, comp)
                params = _competition_params(comp)
                overrides = {
                    "target_asset": asset,
                    "competition": comp,
                    "num_simulations": num_sims,
                    **params,
                }
                result = await state.run_pipeline(overrides=overrides)
                results[f"{asset}_{comp}"] = result.to_dict()
        finally:
            await state.shutdown_mcp()
    else:
        # Direct tool mode — no LLM, just simulate + backtest
        logger.info("No LLM key, running direct simulate → backtest")
        results = _run_direct(pairs, num_sims)

    # Write results
    out_path = OUTPUT_DIR / "train_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Results written to %s", out_path)
    return results


def _run_direct(
    pairs: list[tuple[str, str]], num_sims: int
) -> dict[str, Any]:
    """Simulate + backtest without an LLM agent."""
    import numpy as np

    sys.path.insert(0, os.path.join(PROJECT_DIR, "tools"))

    from run_experiment import SIGMA_MAP, _simulate_gbm

    try:
        from backtest import backtest
    except ImportError:
        backtest = None

    results: dict[str, Any] = {}

    for asset, comp in pairs:
        logger.info("Direct run: %s (%s)", asset, comp)
        start = time.monotonic()
        params = _competition_params(comp)
        sigma = SIGMA_MAP.get(asset, 0.005)

        # Simulate
        paths = _simulate_gbm(
            current_price=100_000.0,  # placeholder — real price from fetch tool
            sigma=sigma,
            time_increment=params["time_increment"],
            time_length=params["time_length"],
            num_simulations=num_sims,
        )

        final_prices = paths[:, -1]
        entry: dict[str, Any] = {
            "asset": asset,
            "competition": comp,
            "model": "gbm",
            "sigma": sigma,
            "num_simulations": num_sims,
            "mean_final": round(float(np.mean(final_prices)), 2),
            "std_final": round(float(np.std(final_prices)), 2),
            "duration_s": round(time.monotonic() - start, 2),
        }

        # Save paths for checkpoint
        checkpoint_dir = Path(os.environ.get("GANGLION_CHECKPOINT_DIR", "/app/checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{asset}_{comp}_paths.npy"
        np.save(str(ckpt_path), paths)
        entry["checkpoint"] = str(ckpt_path)

        results[f"{asset}_{comp}"] = entry
        logger.info(
            "  %s: mean=%.2f std=%.2f (%.1fs)",
            asset, entry["mean_final"], entry["std_final"], entry["duration_s"],
        )

    return results


# ── Validate mode ──────────────────────────────────────────


async def run_validate() -> dict[str, Any]:
    """Backtest saved checkpoints or live-generated paths against scoring."""
    import numpy as np

    sys.path.insert(0, os.path.join(PROJECT_DIR, "tools"))

    from backtest import backtest

    pairs = _resolve_assets()
    checkpoint_dir = Path(os.environ.get("GANGLION_CHECKPOINT_DIR", "/app/checkpoints"))
    results: dict[str, Any] = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for asset, comp in pairs:
        ckpt_path = checkpoint_dir / f"{asset}_{comp}_paths.npy"
        params = _competition_params(comp)

        if ckpt_path.exists():
            logger.info("Validating checkpoint: %s", ckpt_path)
            paths = np.load(str(ckpt_path))
        else:
            logger.warning("No checkpoint for %s_%s, generating fresh paths", asset, comp)
            from run_experiment import SIGMA_MAP, _simulate_gbm

            sigma = SIGMA_MAP.get(asset, 0.005)
            paths = _simulate_gbm(
                current_price=100_000.0,
                sigma=sigma,
                time_increment=params["time_increment"],
                time_length=params["time_length"],
                num_simulations=int(os.environ.get("GANGLION_NUM_SIMULATIONS", "1000")),
            )

        # Generate synthetic realized prices for validation
        # In production, you'd fetch real prices via fetch_historical_prices
        num_steps = paths.shape[1]
        realized = paths[0, :].tolist()  # use first path as "realized" for offline testing

        result = backtest({
            "asset": asset,
            "competition": comp,
            "realized_prices": realized,
            "simulated_paths": paths.tolist(),
            "time_increment": params["time_increment"],
            "time_length": params["time_length"],
        })

        results[f"{asset}_{comp}"] = {
            "content": result.content,
            "metrics": result.metrics,
        }
        logger.info("  %s (%s): CRPS=%.6f", asset, comp, result.metrics.get("crps_total", -1))

    out_path = OUTPUT_DIR / "validate_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Validation results written to %s", out_path)
    return results


# ── Serve mode ─────────────────────────────────────────────


async def run_serve() -> None:
    """Start ganglion as an MCP server on SSE transport."""
    from ganglion.mcp.server import MCPServerBridge
    from ganglion.mcp.tools import register_framework_tools
    from ganglion.mcp.usage import UsageTracker
    from ganglion.state.framework_state import FrameworkState

    state = FrameworkState.load(PROJECT_DIR)
    await state.initialize_mcp()

    register_framework_tools(state.tool_registry, state)

    token = os.environ.get("GANGLION_MCP_TOKEN", "").strip() or None
    usage_db = Path(PROJECT_DIR) / ".ganglion" / "usage.db"
    usage_db.parent.mkdir(parents=True, exist_ok=True)
    tracker = UsageTracker(db_path=usage_db)

    bridge = MCPServerBridge(
        tool_registry=state.tool_registry,
        server_name="ganglion-synth-city",
        token=token,
        role="runpod-worker",
        usage_tracker=tracker,
    )

    port = int(os.environ.get("GANGLION_MCP_PORT", "8900"))
    logger.info("Starting MCP server on 0.0.0.0:%d (token=%s)", port, "set" if token else "NONE")
    if not token:
        logger.warning("No GANGLION_MCP_TOKEN set — server is open to anyone who can reach it!")

    await bridge.run_sse(host="0.0.0.0", port=port)


# ── Main ───────────────────────────────────────────────────


def main() -> None:
    mode = os.environ.get("GANGLION_MODE", "train").strip().lower()
    logger.info("Synth City handler starting in '%s' mode", mode)

    if mode == "train":
        results = asyncio.run(run_train())
        logger.info("Train complete: %d asset runs", len(results))
    elif mode == "validate":
        results = asyncio.run(run_validate())
        logger.info("Validate complete: %d asset runs", len(results))
    elif mode == "serve":
        asyncio.run(run_serve())
    else:
        logger.error("Unknown GANGLION_MODE='%s'. Use: train, validate, serve", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
