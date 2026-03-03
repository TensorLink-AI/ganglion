"""MetaStrategy — optional adaptive tuning of retry policies from historical data."""

from __future__ import annotations

from typing import Any

from ganglion.policies.retry import (
    AttemptConfig,
    EscalatingRetry,
    FixedRetry,
    RetryPolicy,
)


class MetaStrategy:
    """Adaptive tuning of retry policies from historical run data.

    NOT a core primitive. Import and use only if you have enough
    run history to benefit from adaptive tuning (typically 20+ runs).
    """

    def __init__(self, persistence: Any):
        self.persistence = persistence

    async def suggest_policy(self, stage_name: str) -> RetryPolicy:
        """Analyze historical runs and suggest optimal retry parameters."""
        history = await self.persistence.load_run_history(n=50)

        if not history:
            return FixedRetry(max_attempts=3)

        # Analyze success rates and attempt distributions for this stage
        stage_results = []
        for run in history:
            results = run.results if hasattr(run, "results") else {}
            if stage_name in results:
                stage_results.append(results[stage_name])

        if not stage_results:
            return FixedRetry(max_attempts=3)

        total = len(stage_results)
        successes = sum(1 for r in stage_results if r.success)
        success_rate = successes / total if total > 0 else 0

        avg_attempts = (
            sum(r.attempts for r in stage_results) / total if total > 0 else 1
        )

        # High success rate + low attempts => simple retry is fine
        if success_rate > 0.8 and avg_attempts < 2:
            return FixedRetry(max_attempts=2)

        # Moderate success rate => escalating retry
        if success_rate > 0.4:
            max_attempts = min(int(avg_attempts * 1.5) + 1, 8)
            return EscalatingRetry(
                max_attempts=max_attempts,
                base_temp=0.1,
                temp_step=0.15,
            )

        # Low success rate => more aggressive escalation
        return EscalatingRetry(
            max_attempts=8,
            base_temp=0.2,
            temp_step=0.2,
        )
