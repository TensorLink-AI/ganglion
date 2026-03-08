"""Mutation types and audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Mutation:
    """Record of a single state change."""

    mutation_type: str
    target: str
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    diff: str | None = None
    rollback_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_type": self.mutation_type,
            "target": self.target,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "diff": self.diff,
        }


@dataclass
class MutationResult:
    """Result of attempting a mutation."""

    success: bool
    errors: list[str] = field(default_factory=list)
    path: str | None = None
    pipeline: dict[str, Any] | None = None
