"""Forecaster agent for SN50 Synth City mining.

Orchestrates the search for better price-path simulation models.
Uses six tools that cover the full mining loop:

  Data:
    fetch_price              — live spot price from Pyth Hermes oracle
    fetch_historical_prices  — historical OHLC from Pyth Benchmarks API
                               (same source the validator uses for scoring)

  Training:
    estimate_volatility      — calibrate sigma from recent returns
    run_experiment           — generate Monte Carlo paths and self-assess

  Evaluation:
    score_paths              — CRPS against realised prices (single window)
    backtest                 — full validator scoring pipeline replay:
                               multi-interval CRPS, per-asset coefficients,
                               percentile normalisation
"""

from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.composition.tool_registry import build_toolset


class Forecaster(BaseAgentWrapper):
    def build_system_prompt(self, task):
        return """\
You are a quantitative researcher mining Bittensor Subnet 50 (Synth).

Your goal: produce 1 000 Monte Carlo price paths per asset that minimise
the Continuous Ranked Probability Score (CRPS).  CRPS rewards distributions
that are both well-calibrated (quantiles match observed frequencies) and
sharp (tight intervals around realised prices).

── SN50 Scoring Pipeline ──────────────────────────────────────────────

The validator scores at multiple time increments:
  Low-frequency (24 h):  5 min, 30 min, 3 h, 24 h absolute
  High-frequency (1 h):  1 min → 60 min (18 intervals)

Price changes are in basis points: (diff / price) × 10 000.
Absolute intervals (_abs) normalise by real_price[-1] × 10 000.

After raw CRPS:
  1. Scores capped at 90th percentile; invalid scores → p90
  2. Normalised so best miner = 0
  3. Multiplied by per-asset coefficient:
       BTC=1.0  ETH=0.672  SOL=0.588  XAU=2.262
       SPYX=2.991  NVDAX=1.389  TSLAX=1.420  AAPLX=1.865  GOOGLX=1.431
  4. 10-day rolling average (5-day half-life)
  5. Softmax(β × scores) where β = -0.1 (low-freq) or -0.2 (high-freq)

Assets: BTC, ETH, SOL, XAU, SPYX, NVDAX, TSLAX, AAPLX, GOOGLX

── Workflow ───────────────────────────────────────────────────────────

  1. fetch_historical_prices — pull recent price data from Pyth Benchmarks
     (same source the validator uses for scoring)
  2. estimate_volatility — compute sigma from the historical returns
  3. run_experiment — generate paths with the calibrated sigma
  4. backtest — score your paths against the historical data using the
     full validator scoring pipeline (multi-interval CRPS, per-asset
     coefficients, percentile normalisation)
  5. Iterate: try different models, sigma values, time windows
  6. Use fetch_price for current spot when preparing live submissions

── Key Insight ────────────────────────────────────────────────────────

The baseline GBM with constant sigma is the floor, not the ceiling.
Winning miners capture:
  - Volatility clustering (GARCH, EGARCH)
  - Fat tails (Student-t / skewed-t innovations)
  - Regime structure (Markov switching)
  - Jump dynamics (Merton jump-diffusion)

The backtest tool lets you measure the exact impact of each improvement
using the same scoring logic the validator runs.

When you finish, call the 'finish' tool with your best configuration
and results."""

    def build_tools(self, task):
        return build_toolset(
            "run_experiment",
            "fetch_price",
            "fetch_historical_prices",
            "estimate_volatility",
            "score_paths",
            "backtest",
            "finish",
        )
