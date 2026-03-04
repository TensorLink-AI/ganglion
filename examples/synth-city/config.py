"""Synth City — Ganglion project config for mining Bittensor SN50 (Synth).

Synth is a probabilistic price-forecasting subnet. Miners generate 1 000
Monte Carlo price paths per asset. Validators score submissions using CRPS
across multiple time increments (5 min → 24 h).  Emissions are split 50/50
between low-frequency (24 h, 9 assets) and high-frequency (1 h, 4 assets)
competitions.

Reference: https://github.com/mode-network/synth-subnet
"""

from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import (
    MetricDef,
    OutputSpec,
    SubnetConfig,
    TaskDef,
)
from ganglion.policies.presets import SN50_PRESET

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
        StageDef(
            name="calibrate",
            agent="Forecaster",
            depends_on=["plan"],
            input_keys=["plan", "target_asset"],
            output_keys=["volatility_params", "historical_prices"],
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
        StageDef(
            name="backtest",
            agent="Forecaster",
            depends_on=["simulate", "calibrate"],
            input_keys=["price_paths", "historical_prices", "target_asset"],
            output_keys=["crps_breakdown", "crps_weighted"],
            retry=SN50_PRESET["default_retry"],
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
