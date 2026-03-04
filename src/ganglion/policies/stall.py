"""StallDetector — pluggable stall detection for retry policies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ganglion.runtime.types import AgentResult


class StallDetector(Protocol):
    """Protocol for stall detection."""

    def is_stalled(self, attempt: int, last_result: AgentResult) -> bool:
        """Returns True if the agent is stuck in a loop."""
        ...

    def divergence_prompt(self) -> str:
        """Returns a prompt injection to force the agent to try something different."""
        ...


class ConfigComparisonStallDetector:
    """Detects stalls by comparing experiment configs across attempts."""

    def __init__(self, extract_config: Callable[[AgentResult], dict]):
        self.extract_config = extract_config
        self.previous_configs: list[dict] = []

    def is_stalled(self, attempt: int, last_result: AgentResult) -> bool:
        try:
            config = self.extract_config(last_result)
        except (KeyError, TypeError, AttributeError):
            return False

        is_duplicate = config in self.previous_configs
        self.previous_configs.append(config)
        return is_duplicate

    def divergence_prompt(self) -> str:
        return (
            "CRITICAL: Your last attempt produced the same configuration as a previous attempt. "
            "You MUST try a fundamentally different approach. Change the architecture, "
            "hyperparameters, or strategy — do not repeat what has already failed."
        )

    def reset(self) -> None:
        """Clear history (e.g., between pipeline runs)."""
        self.previous_configs.clear()


class OutputHashStallDetector:
    """Detects stalls by comparing hashed output text across attempts."""

    def __init__(self, max_repeats: int = 2):
        self.max_repeats = max_repeats
        self.previous_hashes: list[int] = []

    def is_stalled(self, attempt: int, last_result: AgentResult) -> bool:
        h = hash(last_result.raw_text)
        count = self.previous_hashes.count(h)
        self.previous_hashes.append(h)
        return count >= self.max_repeats

    def divergence_prompt(self) -> str:
        return (
            "WARNING: Your output is very similar to previous attempts. "
            "Try a completely different approach or strategy."
        )

    def reset(self) -> None:
        self.previous_hashes.clear()
