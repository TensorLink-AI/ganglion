"""Local CRPS scoring tool for offline evaluation.

Lets the agent score its own simulated paths against a held-out price
series before submitting to the validator.  Uses the same `properscoring`
library the SN50 validator uses.

Install:  pip install properscoring

Reference:
  https://github.com/mode-network/synth-subnet/blob/main/synth/validator/crps_calculation.py
"""

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult


@tool("score_paths", category="evaluation")
def score_paths(config: dict) -> ExperimentResult:
    """Score simulated price paths against realised prices using CRPS.

    Expected config keys:
        simulated_paths: list[list[float]]  — (num_sims, num_steps) price paths
        realized_prices: list[float]        — actual prices (same num_steps)
        time_increment: int                 — seconds between steps (default 300)
        scoring_intervals: list[int]        — intervals in seconds to evaluate
                                              (default [300, 1800, 10800, 86400])
    """
    import numpy as np

    try:
        from properscoring import crps_ensemble as _crps_ensemble
    except ImportError:
        _crps_ensemble = None

    if _crps_ensemble is None:
        return ExperimentResult(
            content="properscoring is not installed.  Run: pip install properscoring",
            metrics={},
        )

    sim = config.get("simulated_paths")
    real = config.get("realized_prices")
    if sim is None or real is None:
        return ExperimentResult(
            content="Provide 'simulated_paths' and 'realized_prices' in config.",
            metrics={},
        )

    sim_arr = np.array(sim, dtype=float)
    real_arr = np.array(real, dtype=float)
    time_increment = config.get("time_increment", 300)
    scoring_intervals = config.get("scoring_intervals", [300, 1800, 10800, 86400])

    results = {}
    total_crps = 0.0

    for interval in scoring_intervals:
        step = interval // time_increment
        if step < 1 or step >= len(real_arr):
            continue

        # Compute pct changes in basis points (matching validator logic)
        sim_at_interval = sim_arr[:, ::step]
        real_at_interval = real_arr[::step]

        sim_changes = (np.diff(sim_at_interval, axis=1) / sim_at_interval[:, :-1]) * 10_000
        real_changes = (np.diff(real_at_interval) / real_at_interval[:-1]) * 10_000

        interval_crps = 0.0
        n_points = real_changes.shape[0]
        for t in range(n_points):
            interval_crps += _crps_ensemble(real_changes[t], sim_changes[:, t])

        avg_crps = interval_crps / max(n_points, 1)
        label = f"{interval}s"
        results[f"crps_{label}"] = round(avg_crps, 6)
        total_crps += avg_crps

    results["crps_total"] = round(total_crps, 6)

    return ExperimentResult(
        content=(
            f"CRPS evaluation across {len(scoring_intervals)} intervals.  "
            f"Total CRPS: {total_crps:.6f} (lower is better).  "
            f"Breakdown: {results}"
        ),
        experiment_id="crps-eval",
        metrics=results,
    )
