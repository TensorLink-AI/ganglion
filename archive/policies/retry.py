"""RetryPolicy — the single retry abstraction and built-in implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ganglion.runtime.types import AgentResult


@dataclass
class AttemptConfig:
    """Configuration for a single retry attempt."""

    temperature: float = 0.7
    model: str | None = None
    extra_system_context: str | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


class RetryPolicy(Protocol):
    """Protocol for retry policies.

    Implement configure_attempt to control retry behavior.
    Return None to stop retrying.
    """

    def configure_attempt(
        self,
        attempt: int,
        last_result: AgentResult | None,
    ) -> AttemptConfig | None:
        """Configure the next attempt, or return None to stop retrying."""
        ...


class NoRetry:
    """Never retry. One shot."""

    def configure_attempt(
        self, attempt: int, last_result: AgentResult | None
    ) -> AttemptConfig | None:
        return AttemptConfig() if attempt == 0 else None

    def __repr__(self) -> str:
        return "NoRetry()"


class FixedRetry:
    """Retry N times with the same config."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def configure_attempt(
        self, attempt: int, last_result: AgentResult | None
    ) -> AttemptConfig | None:
        if attempt >= self.max_attempts:
            return None
        return AttemptConfig()

    def __repr__(self) -> str:
        return f"FixedRetry(max_attempts={self.max_attempts})"


class EscalatingRetry:
    """Retry with increasing temperature and optional stall detection."""

    def __init__(
        self,
        max_attempts: int = 5,
        base_temp: float = 0.1,
        temp_step: float = 0.1,
        stall_detector: Any | None = None,
    ):
        self.max_attempts = max_attempts
        self.base_temp = base_temp
        self.temp_step = temp_step
        self.stall_detector = stall_detector

    def configure_attempt(
        self, attempt: int, last_result: AgentResult | None
    ) -> AttemptConfig | None:
        if attempt >= self.max_attempts:
            return None

        if last_result and not self._is_retryable(last_result):
            return None

        temp = self.base_temp + (attempt * self.temp_step)
        extra_context = None

        if (
            self.stall_detector
            and last_result
            and self.stall_detector.is_stalled(attempt, last_result)
        ):
            extra_context = self.stall_detector.divergence_prompt()

        return AttemptConfig(temperature=temp, extra_system_context=extra_context)

    def _is_retryable(self, result: AgentResult) -> bool:
        """Check if the error is retryable based on the result."""
        return not result.success

    def __repr__(self) -> str:
        return (
            f"EscalatingRetry(max_attempts={self.max_attempts}, "
            f"base_temp={self.base_temp}, temp_step={self.temp_step})"
        )


class ModelEscalationRetry:
    """Escalate by switching to a more capable (expensive) model."""

    def __init__(
        self,
        model_ladder: list[str],
        attempts_per_model: int = 2,
    ):
        self.model_ladder = model_ladder
        self.attempts_per_model = attempts_per_model

    def configure_attempt(
        self, attempt: int, last_result: AgentResult | None
    ) -> AttemptConfig | None:
        model_idx = attempt // self.attempts_per_model
        if model_idx >= len(self.model_ladder):
            return None
        return AttemptConfig(model=self.model_ladder[model_idx])

    def __repr__(self) -> str:
        return (
            f"ModelEscalationRetry(model_ladder={self.model_ladder}, "
            f"attempts_per_model={self.attempts_per_model})"
        )
