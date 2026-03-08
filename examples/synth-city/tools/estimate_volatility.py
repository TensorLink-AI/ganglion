"""Volatility estimation tool for SN50 miners.

The baseline SN50 miner uses a fixed sigma per asset.  Better miners
estimate volatility from recent price data.  This tool provides EWMA
and simple realised-vol estimators as a starting point.

Competitive approaches to try:
  - GARCH(1,1) / EGARCH for volatility clustering
  - Heston stochastic-vol for mean-reverting vol
  - Regime-switching (Hamilton) for structural breaks
  - Realised vol from tick-level data (5-min returns)
"""

import math

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult


@tool("estimate_volatility", category="training")
def estimate_volatility(config: dict) -> ExperimentResult:
    """Estimate asset volatility from recent returns.

    Expected config keys:
        returns: list[float]     — recent log-returns (required)
        method: str              — "realized" | "ewma" (default "realized")
        ewma_lambda: float       — EWMA decay factor (default 0.94)
        asset: str               — asset name for labelling (default "BTC")
    """
    returns = config.get("returns")
    if not returns or len(returns) < 10:
        return ExperimentResult(
            content="Need at least 10 return observations.  Pass 'returns' as a list of floats.",
            metrics={},
        )

    import numpy as np

    method = config.get("method", "realized")
    asset = config.get("asset", "BTC")
    arr = np.array(returns, dtype=float)

    if method == "ewma":
        lam = config.get("ewma_lambda", 0.94)
        var = arr[0] ** 2
        for r in arr[1:]:
            var = lam * var + (1 - lam) * r**2
        sigma = math.sqrt(var)
    else:  # realized
        sigma = float(np.std(arr, ddof=1))

    return ExperimentResult(
        content=(
            f"{method.upper()} volatility for {asset}: sigma={sigma:.6f} "
            f"(from {len(arr)} observations)"
        ),
        experiment_id=f"vol-{asset.lower()}-{method}",
        metrics={
            "sigma": round(sigma, 8),
            "method": method,
            "n_observations": len(arr),
        },
    )
