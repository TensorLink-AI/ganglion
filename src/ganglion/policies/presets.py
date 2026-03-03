"""Pre-built policy bundles for common scenarios."""

from __future__ import annotations

from ganglion.policies.retry import (
    EscalatingRetry,
    FixedRetry,
    ModelEscalationRetry,
)
from ganglion.policies.stall import ConfigComparisonStallDetector

# SN50 preset: escalating temperature with config-comparison stall detection
SN50_PRESET = {
    "default_retry": EscalatingRetry(
        max_attempts=5,
        base_temp=0.1,
        temp_step=0.1,
        stall_detector=ConfigComparisonStallDetector(
            extract_config=lambda r: r.structured.get("config", {})
            if isinstance(r.structured, dict)
            else {},
        ),
    ),
}

# Simple preset: fixed retries, no stall detection
SIMPLE_PRESET = {
    "default_retry": FixedRetry(max_attempts=3),
}

# Aggressive preset: model escalation ladder
AGGRESSIVE_PRESET = {
    "default_retry": ModelEscalationRetry(
        model_ladder=["haiku-3.5", "sonnet-3.5", "opus-3"],
        attempts_per_model=2,
    ),
}
