"""Subnet conflict example — two bots mining Synth City (SN50) with conflicting strategies.

This example demonstrates how multi-bot mining creates natural conflicts and
how Ganglion's knowledge store + stall detection resolve them without explicit
coordination.

Scenario
--------
Two OpenClaw bots mine SN50 (Synth City) simultaneously:

  - Bot "alpha" explores volatility models (GARCH, EWMA, realized vol)
  - Bot "beta" explores price-path generators (GBM, jump-diffusion, neural SDE)

Conflict arises when both bots converge on the same configuration space:
alpha discovers that GARCH + GBM is the current best, and beta independently
discovers the same. Without conflict detection, they waste compute re-exploring
the same region.

Resolution: The shared knowledge pool makes alpha's discovery visible to beta.
Beta reads the pattern, sees that GARCH + GBM is already explored, and pivots
to GARCH + jump-diffusion instead. No coordinator needed — the knowledge pool
is the only coupling.

Run
---
    python examples/synth-city/subnet_conflict.py
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend
from ganglion.knowledge.store import KnowledgeStore
from ganglion.knowledge.types import KnowledgeQuery
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import (
    MetricDef,
    OutputSpec,
    SubnetConfig,
    TaskContext,
    TaskDef,
)
from ganglion.policies.presets import SN50_PRESET
from ganglion.policies.retry import EscalatingRetry
from ganglion.policies.stall import ConfigComparisonStallDetector


# ── Subnet configuration (SN50 Synth City) ──────────────────

sn50_config = SubnetConfig(
    netuid=50,
    name="Synth City",
    metrics=[
        MetricDef("crps", "minimize", weight=1.0,
                  description="Continuous Ranked Probability Score"),
        MetricDef("calibration", "minimize", weight=0.3,
                  description="Quantile calibration error"),
        MetricDef("sharpness", "minimize", weight=0.2,
                  description="Prediction interval width"),
    ],
    tasks={
        "btc_forecast": TaskDef("btc_forecast", weight=0.6,
                                metadata={"asset": "BTC", "horizon_hours": 24}),
        "eth_forecast": TaskDef("eth_forecast", weight=0.4,
                                metadata={"asset": "ETH", "horizon_hours": 24}),
    },
    output_spec=OutputSpec(
        format="price_paths_json",
        description="JSON array of simulated price paths",
    ),
)


# ── Pipeline with conflict-aware stall detection ─────────────

sn50_pipeline = PipelineDef(
    name="synth-city-pipeline",
    stages=[
        StageDef(
            name="plan",
            agent="Forecaster",
            output_keys=["plan"],
            retry=SN50_PRESET["default_retry"],
        ),
        StageDef(
            name="generate_paths",
            agent="Forecaster",
            depends_on=["plan"],
            input_keys=["plan"],
            output_keys=["price_paths", "metrics"],
            retry=EscalatingRetry(
                max_attempts=5,
                base_temp=0.1,
                temp_step=0.15,
                stall_detector=ConfigComparisonStallDetector(
                    extract_config=lambda r: (
                        r.structured.get("config", {})
                        if isinstance(r.structured, dict)
                        else {}
                    ),
                ),
            ),
        ),
        StageDef(
            name="evaluate",
            agent="Forecaster",
            depends_on=["generate_paths"],
            input_keys=["price_paths", "metrics"],
            output_keys=["crps_score"],
        ),
    ],
)


# ── Simulated multi-bot conflict scenario ────────────────────


async def simulate_conflict():
    """Demonstrate knowledge-mediated conflict resolution between two bots."""

    import tempfile
    from pathlib import Path

    shared_dir = Path(tempfile.mkdtemp(prefix="sn50-shared-"))
    print(f"Shared knowledge directory: {shared_dir}\n")

    # Both bots share a knowledge backend but have different bot_ids
    alpha_knowledge = KnowledgeStore(
        backend=JsonKnowledgeBackend(str(shared_dir / "alpha")),
        bot_id="alpha",
    )
    beta_knowledge = KnowledgeStore(
        backend=JsonKnowledgeBackend(str(shared_dir / "beta")),
        bot_id="beta",
    )

    # ── Phase 1: Alpha discovers GARCH + GBM is good ─────────

    print("=" * 60)
    print("PHASE 1: Alpha explores volatility models")
    print("=" * 60)

    # Alpha tries constant vol + GBM (baseline)
    await alpha_knowledge.record_failure(
        capability="generate_paths",
        error_summary="Constant vol GBM gives CRPS=0.45 — too wide distributions",
        failure_mode="poor_calibration",
        config={"vol_model": "constant", "path_model": "gbm", "crps": 0.45},
    )
    print("[alpha] Constant vol + GBM: CRPS=0.45 (poor calibration)")

    # Alpha tries EWMA + GBM
    await alpha_knowledge.record_success(
        capability="generate_paths",
        description="EWMA vol + GBM gives CRPS=0.31 — decent baseline",
        metric_value=0.31,
        metric_name="crps",
        config={"vol_model": "ewma", "path_model": "gbm"},
    )
    print("[alpha] EWMA + GBM: CRPS=0.31 (decent)")

    # Alpha tries GARCH + GBM
    await alpha_knowledge.record_success(
        capability="generate_paths",
        description="GARCH vol + GBM gives CRPS=0.22 — best so far",
        metric_value=0.22,
        metric_name="crps",
        config={"vol_model": "garch", "path_model": "gbm"},
    )
    print("[alpha] GARCH + GBM: CRPS=0.22 (current best)")

    # ── Phase 2: Beta starts exploring — WITHOUT seeing alpha ─

    print(f"\n{'=' * 60}")
    print("PHASE 2: Beta explores path generators (no shared knowledge)")
    print("=" * 60)

    # Beta independently discovers GBM + GARCH is good
    await beta_knowledge.record_success(
        capability="generate_paths",
        description="GBM + GARCH vol gives CRPS=0.23 — strong result",
        metric_value=0.23,
        metric_name="crps",
        config={"vol_model": "garch", "path_model": "gbm"},
    )
    print("[beta]  GBM + GARCH: CRPS=0.23 (converged to same region as alpha!)")
    print()
    print("  ** CONFLICT: Both bots converged on GARCH + GBM **")
    print("  ** Beta is about to waste compute re-exploring alpha's territory **")

    # ── Phase 3: Enable shared knowledge — conflict resolves ──

    print(f"\n{'=' * 60}")
    print("PHASE 3: Shared knowledge resolves the conflict")
    print("=" * 60)

    # In a real setup, both bots use a FederatedKnowledgeBackend.
    # Here we simulate beta reading alpha's patterns.
    alpha_patterns = await alpha_knowledge.backend.query_patterns(
        KnowledgeQuery(capability="generate_paths")
    )
    alpha_antipatterns = await alpha_knowledge.backend.query_antipatterns(
        KnowledgeQuery(capability="generate_paths")
    )

    print(f"\n[beta reads shared pool]")
    print(f"  Patterns from alpha: {len(alpha_patterns)}")
    for p in alpha_patterns:
        print(f"    - {p.description} (crps={p.metric_value})")
    print(f"  Antipatterns from alpha: {len(alpha_antipatterns)}")
    for a in alpha_antipatterns:
        print(f"    - {a.error_summary}")

    # Beta sees that GARCH + GBM is already explored — pivots
    print(f"\n[beta pivots strategy]")
    print("  Alpha already found GARCH + GBM = 0.22")
    print("  Beta pivots to GARCH + jump-diffusion (unexplored region)")

    await beta_knowledge.record_success(
        capability="generate_paths",
        description="GARCH vol + jump-diffusion gives CRPS=0.18 — new best!",
        metric_value=0.18,
        metric_name="crps",
        config={"vol_model": "garch", "path_model": "jump_diffusion",
                "jump_intensity": 0.1, "jump_size_std": 0.02},
    )
    print("[beta]  GARCH + jump-diffusion: CRPS=0.18 (NEW BEST)")

    # ── Phase 4: Alpha reads beta's discovery ─────────────────

    print(f"\n{'=' * 60}")
    print("PHASE 4: Alpha reads beta's discovery — cooperation emerges")
    print("=" * 60)

    beta_patterns = await beta_knowledge.backend.query_patterns(
        KnowledgeQuery(capability="generate_paths")
    )
    print(f"\n[alpha reads shared pool]")
    for p in beta_patterns:
        print(f"  Pattern from beta: {p.description} (crps={p.metric_value})")

    print("\n  Alpha sees jump-diffusion works — explores jump + neural SDE hybrid")

    await alpha_knowledge.record_success(
        capability="generate_paths",
        description="Neural SDE with jump-diffusion prior gives CRPS=0.15",
        metric_value=0.15,
        metric_name="crps",
        config={"vol_model": "neural_sde", "path_model": "jump_diffusion",
                "jump_intensity": 0.1, "regularization": "kl_divergence"},
    )
    print("[alpha] Neural SDE + jump-diffusion: CRPS=0.15 (NEW BEST)")

    # ── Summary ───────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("SUMMARY: Conflict resolution through shared knowledge")
    print("=" * 60)
    print("""
Without shared knowledge:
  - Both bots spend cycles on GARCH + GBM
  - Redundant exploration of the same region
  - Slower convergence to optimal solution

With shared knowledge:
  - Alpha explores vol models → discovers GARCH + GBM (0.22)
  - Beta sees alpha's result → pivots to jump-diffusion (0.18)
  - Alpha sees beta's result → tries neural SDE + jump prior (0.15)
  - Each run compounds on the last — the knowledge is the moat

Conflict types in multi-bot subnet mining:
  1. CONVERGENCE CONFLICT — bots explore the same region (resolved by shared patterns)
  2. STALL CONFLICT — a bot retries the same config (resolved by stall detectors)
  3. MUTATION CONFLICT — simultaneous pipeline edits (resolved by ConcurrentMutationError)
  4. KNOWLEDGE CONFLICT — contradictory patterns (resolved by recency + metric comparison)
""")

    # Cleanup
    import shutil
    shutil.rmtree(shared_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(simulate_conflict())
