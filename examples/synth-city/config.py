"""Synth City — Ganglion project config for mining Bittensor SN50 (Synth).

Synth is a probabilistic price-forecasting subnet. Miners generate 1 000
Monte Carlo price paths per asset. Validators score submissions using CRPS
across multiple time increments (5 min → 24 h).  Emissions are split 50/50
between low-frequency (24 h, 9 assets) and high-frequency (1 h, 4 assets)
competitions.

Reference: https://github.com/mode-network/synth-subnet
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

_PROJECT_DIR = str(Path(__file__).resolve().parent)

from ganglion.compute.protocol import DockerPrefab
from ganglion.compute.router import ComputeRoute, ComputeRouter
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend
from ganglion.knowledge.store import KnowledgeStore
from ganglion.mcp.config import MCPClientConfig
from ganglion.orchestration.pipeline import PipelineDef, StageDef, ToolStageDef
from ganglion.orchestration.task_context import (
    MetricDef,
    OutputSpec,
    SubnetConfig,
    TaskDef,
)
from ganglion.policies.presets import SN50_PRESET

if TYPE_CHECKING:
    from ganglion.orchestration.task_context import TaskContext
    from ganglion.runtime.types import AgentResult

# ── Subnet configuration ────────────────────────────────────

subnet_config = SubnetConfig(
    netuid=50,
    name="Synth City",
    metrics=[
        MetricDef(
            "crps",
            "minimize",
            weight=1.0,
            description=(
                "Continuous Ranked Probability Score — the primary scoring "
                "metric.  Computed per time increment (5 min, 30 min, 3 h, "
                "24 h absolute) in basis points.  Lower is better."
            ),
        ),
        MetricDef(
            "calibration",
            "minimize",
            weight=0.3,
            description=(
                "How well predicted quantiles match observed frequencies.  "
                "Implied by CRPS but tracked separately for diagnostics."
            ),
        ),
        MetricDef(
            "sharpness",
            "minimize",
            weight=0.2,
            description=(
                "Width of prediction intervals.  Sharper (narrower) "
                "distributions score better — but only if well-calibrated."
            ),
        ),
    ],
    tasks={
        # Low-frequency competition (50 % of emissions)
        "lf_btc": TaskDef("lf_btc", weight=1.0, metadata={"asset": "BTC", "time_increment": 300, "time_length": 86400}),
        "lf_eth": TaskDef("lf_eth", weight=0.67, metadata={"asset": "ETH", "time_increment": 300, "time_length": 86400}),
        "lf_sol": TaskDef("lf_sol", weight=1.0, metadata={"asset": "SOL", "time_increment": 300, "time_length": 86400}),
        "lf_xau": TaskDef("lf_xau", weight=2.26, metadata={"asset": "XAU", "time_increment": 300, "time_length": 86400}),
        "lf_spyx": TaskDef("lf_spyx", weight=1.4, metadata={"asset": "SPYX", "time_increment": 300, "time_length": 86400}),
        "lf_nvdax": TaskDef("lf_nvdax", weight=2.99, metadata={"asset": "NVDAX", "time_increment": 300, "time_length": 86400}),
        "lf_tslax": TaskDef("lf_tslax", weight=2.0, metadata={"asset": "TSLAX", "time_increment": 300, "time_length": 86400}),
        "lf_aaplx": TaskDef("lf_aaplx", weight=2.0, metadata={"asset": "AAPLX", "time_increment": 300, "time_length": 86400}),
        "lf_googlx": TaskDef("lf_googlx", weight=2.0, metadata={"asset": "GOOGLX", "time_increment": 300, "time_length": 86400}),
        # High-frequency competition (50 % of emissions)
        "hf_btc": TaskDef("hf_btc", weight=1.0, metadata={"asset": "BTC", "time_increment": 60, "time_length": 3600}),
        "hf_eth": TaskDef("hf_eth", weight=0.67, metadata={"asset": "ETH", "time_increment": 60, "time_length": 3600}),
        "hf_sol": TaskDef("hf_sol", weight=1.0, metadata={"asset": "SOL", "time_increment": 60, "time_length": 3600}),
        "hf_xau": TaskDef("hf_xau", weight=2.26, metadata={"asset": "XAU", "time_increment": 60, "time_length": 3600}),
    },
    output_spec=OutputSpec(
        format="price_paths_tuple",
        shape_constraints={
            "lf_steps": 289,   # (86400 / 300) + 1
            "hf_steps": 61,    # (3600 / 60) + 1
            "num_simulations": 1000,
            "max_digits_per_price": 8,
        },
        description=(
            "Tuple: (start_timestamp, time_increment, [path_1], ..., [path_1000]).  "
            "Each path is a list of floats with (time_length / time_increment) + 1 "
            "price points.  Max 8 digits per value."
        ),
    ),
    constraints={
        "num_simulations": 1000,
        "lf_time_increment_s": 300,
        "lf_time_length_s": 86400,
        "hf_time_increment_s": 60,
        "hf_time_length_s": 3600,
        "scoring_intervals_lf": "5min, 30min, 3h, 24h_abs",
        "scoring_intervals_hf": "1min → 60min (18 intervals)",
        "moving_average_window_days": 10,
        "moving_average_halflife_days": 5,
        "emission_split": "50% low-freq / 50% high-freq",
    },
    docker_prefabs={
        # The synth-city training worker — handles train, validate, backtest,
        # and agent-submitted scripts. Agents dispatch jobs to this image via
        # RunPodBackend; the GANGLION_JOB_SPEC env var carries the full spec.
        # Build:  docker build -f examples/synth-city/Dockerfile.runpod -t synth-city-worker .
        "synth-worker": DockerPrefab(
            name="synth-worker",
            image="synth-city-worker:latest",
            gpu_type="A10G",
            gpu_count=1,
            memory_gb=16,
            timeout_seconds=1800,
            env={"GANGLION_NUM_SIMULATIONS": "1000"},
            artifacts_dir="/outputs",
        ),
        # CPU-only variant for calibration / validation (no GPU needed)
        "synth-worker-cpu": DockerPrefab(
            name="synth-worker-cpu",
            image="synth-city-worker:latest",
            cpu_cores=4,
            memory_gb=8,
            timeout_seconds=600,
            artifacts_dir="/outputs",
        ),
    },
)

# ── MCP tool servers ─────────────────────────────────────────
# Connect to external data feeds via MCP.

mcp_clients = [
    MCPClientConfig(
        name="market-data",
        transport="stdio",
        command=["python", "-m", "synth_market_data_server"],
        tool_prefix="market",
        category="data",
        timeout=15.0,
        cwd=_PROJECT_DIR,
    ),
]

# ── Compute routing ──────────────────────────────────────────
# Route stages to the synth-worker on RunPod or locally.
# The RunPod backend is registered at runtime (requires RUNPOD_API_KEY).
# Job specs are passed via the GANGLION_JOB_SPEC env var in dockerArgs.

compute_router = ComputeRouter(
    backends={},  # Backends registered at runtime via CLI or ganglion_connect_mcp
    routes=[
        # GPU-heavy simulation → RunPod (image resolved from docker_prefabs at runtime)
        ComputeRoute(
            pattern="simulate",
            backend="runpod",
            overrides={"gpu_type": "A10G"},
        ),
        # Backtest can run on RunPod too (for large path sets)
        ComputeRoute(
            pattern="backtest",
            backend="runpod",
            overrides={},
        ),
        # Calibration and everything else stays local
        ComputeRoute(pattern="calibrate", backend="local"),
        ComputeRoute(pattern="default", backend="local"),
    ],
)


# ── Deterministic stages ─────────────────────────────────────
# These are ToolStageDef callbacks — pure data operations, no LLM needed.
# They build job specs and dispatch to the synth-city-worker container
# (via compute_router → RunPodBackend) or run locally as fallback.


async def fetch_prices(ctx: TaskContext) -> AgentResult:
    """Fetch historical prices for the target asset via the Pyth API.

    Uses the fetch_historical_prices tool if registered, otherwise
    returns a placeholder for the agent to fill in.
    """
    from ganglion.runtime.types import AgentResult

    asset = ctx.get("target_asset", "BTC")
    competition = ctx.get("competition", "low_freq")
    time_increment = 60 if competition == "high_freq" else 300
    hours_back = 1 if competition == "high_freq" else 24

    # Try to use the registered fetch_historical_prices tool
    tool_registry = ctx.get("_tool_registry")
    if tool_registry:
        fetch_tool = tool_registry.get("fetch_historical_prices")
        if fetch_tool:
            result = fetch_tool.func({
                "asset": asset,
                "hours_back": hours_back,
                "time_increment": time_increment,
            })
            if hasattr(result, "metrics") and result.metrics.get("prices"):
                return AgentResult(
                    success=True,
                    structured={
                        "asset": asset,
                        "prices": result.metrics["prices"],
                        "timestamps": result.metrics.get("timestamps", []),
                    },
                    raw_text=f"Fetched {result.metrics.get('valid_points', 0)} prices for {asset}",
                )

    # Fallback — the agent or a subsequent stage must provide real prices
    return AgentResult(
        success=True,
        structured={"asset": asset, "prices": [], "source": "placeholder"},
        raw_text=f"No price data fetched for {asset} — agent should provide via tool",
    )


async def score_paths(ctx: TaskContext) -> AgentResult:
    """Dispatch a backtest job to the synth-city worker.

    Builds a job spec from pipeline context and sends it to the
    training worker. If no compute backend is available, falls back
    to the local backtest tool.
    """
    import json

    from ganglion.runtime.types import AgentResult

    asset = ctx.get("target_asset", "BTC")
    competition = ctx.get("competition", "low_freq")
    price_paths = ctx.get("price_paths")
    historical_prices = ctx.get("historical_prices", [])

    # Build the job spec for the training worker
    job_spec = {
        "mode": "backtest",
        "asset": asset,
        "competition": competition,
    }

    if price_paths is not None:
        job_spec["simulated_paths"] = price_paths
    if historical_prices:
        job_spec["realized_prices"] = historical_prices

    # Try to dispatch to compute backend using the declared prefab
    router = ctx.get("_compute_router")
    if router:
        # Use the declared docker prefab rather than hardcoding the image
        prefab = ctx.subnet_config.docker_prefabs.get("synth-worker-cpu")
        if prefab:
            spec = prefab.to_job_spec(
                command=["python", "/app/handler.py"],
                env={"GANGLION_JOB_SPEC": json.dumps(job_spec)},
            )
        else:
            from ganglion.compute.protocol import JobSpec

            spec = JobSpec(
                image="synth-city-worker:latest",
                command=["python", "/app/handler.py"],
                env={"GANGLION_JOB_SPEC": json.dumps(job_spec)},
                artifacts_dir="/outputs",
            )

        backend, spec = router.resolve_with_overrides("backtest", spec)
        try:
            handle = await backend.submit(spec)
            # Poll until done (simplified — real impl should use async polling)
            import asyncio

            for _ in range(60):
                handle = await backend.poll(handle)
                if handle.status.value in ("succeeded", "failed"):
                    break
                await asyncio.sleep(10)

            result = await backend.collect(handle)
            await backend.cleanup(handle)

            if result.stdout:
                try:
                    parsed = json.loads(result.stdout)
                    return AgentResult(
                        success=True,
                        structured=parsed,
                        raw_text=f"Backtest complete for {asset}: CRPS={parsed.get('crps_total', '?')}",
                    )
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            # Fall through to local backtest
            pass

    # Fallback — use local backtest tool if available
    tool_registry = ctx.get("_tool_registry")
    if tool_registry:
        bt_tool = tool_registry.get("backtest")
        if bt_tool:
            result = bt_tool.func({
                "asset": asset,
                "competition": competition,
                "realized_prices": historical_prices,
                "simulated_paths": price_paths,
            })
            return AgentResult(
                success=True,
                structured=result.metrics if hasattr(result, "metrics") else {},
                raw_text=result.content if hasattr(result, "content") else str(result),
            )

    return AgentResult(
        success=False,
        structured={"error": "No compute backend or backtest tool available"},
        raw_text=f"Could not run backtest for {asset}",
    )


# ── Pipeline ─────────────────────────────────────────────────

pipeline = PipelineDef(
    name="synth-city",
    stages=[
        StageDef(
            name="plan",
            agent="Forecaster",
            output_keys=["plan", "target_asset", "model_config"],
            retry=SN50_PRESET["default_retry"],
        ),
        ToolStageDef(
            name="fetch_prices",
            fn=fetch_prices,
            depends_on=["plan"],
            input_keys=["target_asset"],
            output_keys=["historical_prices"],
        ),
        StageDef(
            name="calibrate",
            agent="Forecaster",
            depends_on=["plan", "fetch_prices"],
            input_keys=["plan", "target_asset", "historical_prices"],
            output_keys=["volatility_params"],
            retry=SN50_PRESET["default_retry"],
        ),
        StageDef(
            name="simulate",
            agent="Forecaster",
            depends_on=["calibrate"],
            input_keys=["model_config", "volatility_params", "target_asset"],
            output_keys=["price_paths", "metrics"],
            retry=SN50_PRESET["default_retry"],
        ),
        ToolStageDef(
            name="backtest",
            fn=score_paths,
            depends_on=["simulate", "fetch_prices"],
            input_keys=["price_paths", "historical_prices", "target_asset"],
            output_keys=["crps_breakdown", "crps_weighted"],
        ),
        StageDef(
            name="evaluate",
            agent="Forecaster",
            depends_on=["backtest"],
            input_keys=["crps_breakdown", "crps_weighted", "metrics"],
            output_keys=["summary"],
        ),
    ],
)

# ── Knowledge store ──────────────────────────────────────────
# Cross-run memory: tracks which model configs and volatility params
# produced the best CRPS scores.

knowledge = KnowledgeStore(
    backend=JsonKnowledgeBackend("./knowledge/"),
    max_patterns=500,
    max_antipatterns=500,
)
