"""TaskContext — typed inter-agent blackboard and SubnetConfig."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

_SENTINEL = object()


@dataclass
class MetricDef:
    """Definition of a metric used to evaluate subnet performance."""

    name: str
    direction: Literal["minimize", "maximize"]
    weight: float = 1.0
    description: str = ""

    def is_better(self, a: float, b: float) -> bool:
        """Returns True if `a` is better than `b` according to this metric."""
        return a < b if self.direction == "minimize" else a > b


@dataclass
class TaskDef:
    """Definition of a named task within a subnet."""

    name: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSpec:
    """What the validator expects from a miner submission."""

    format: str
    shape_constraints: dict[str, Any] = field(default_factory=dict)
    validation_fn: Callable[[Any], Any] | None = None
    description: str = ""


@dataclass
class SubnetConfig:
    """Everything the framework needs to know about a subnet."""

    netuid: int
    name: str
    metrics: list[MetricDef]
    tasks: dict[str, TaskDef]
    output_spec: OutputSpec
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_prompt_section(self) -> str:
        """Format the config as a prompt section for agent consumption."""
        lines = [
            f"Subnet: {self.name} (netuid={self.netuid})",
            "",
            "Metrics:",
        ]
        for m in self.metrics:
            lines.append(f"  - {m.name} ({m.direction}, weight={m.weight})")
            if m.description:
                lines.append(f"    {m.description}")

        lines.append("")
        lines.append("Tasks:")
        for name, task in self.tasks.items():
            lines.append(f"  - {name} (weight={task.weight})")

        lines.append("")
        lines.append(f"Output: {self.output_spec.format}")
        if self.output_spec.description:
            lines.append(f"  {self.output_spec.description}")

        if self.constraints:
            lines.append("")
            lines.append("Constraints:")
            for k, v in self.constraints.items():
                lines.append(f"  - {k}: {v}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "netuid": self.netuid,
            "name": self.name,
            "metrics": [
                {"name": m.name, "direction": m.direction, "weight": m.weight, "description": m.description}
                for m in self.metrics
            ],
            "tasks": {
                name: {"name": t.name, "weight": t.weight, "metadata": t.metadata}
                for name, t in self.tasks.items()
            },
            "output_spec": {
                "format": self.output_spec.format,
                "shape_constraints": self.output_spec.shape_constraints,
                "description": self.output_spec.description,
            },
            "constraints": self.constraints,
        }


@dataclass
class SlotMeta:
    """Metadata about a TaskContext slot write."""

    written_by: str
    written_at: datetime
    description: str = ""
    value_type: str = ""


class TaskContext:
    """Inter-agent shared state with namespaced, typed slots."""

    def __init__(self, subnet_config: SubnetConfig, initial: dict | None = None):
        self.subnet_config = subnet_config
        self._data: dict[str, Any] = initial or {}
        self._metadata: dict[str, SlotMeta] = {}

    def set(self, key: str, value: Any, stage: str, description: str = "") -> None:
        """Write a value. Records which stage wrote it and when."""
        self._data[key] = value
        self._metadata[key] = SlotMeta(
            written_by=stage,
            written_at=datetime.utcnow(),
            description=description,
            value_type=type(value).__name__,
        )

    def get(self, key: str, default: Any = _SENTINEL) -> Any:
        """Read a value. Raises KeyError with helpful message if missing."""
        if key not in self._data:
            if default is not _SENTINEL:
                return default
            available = list(self._data.keys())
            raise KeyError(
                f"TaskContext key '{key}' not found. "
                f"Available keys: {available}. "
                f"Check that an upstream stage declares this in output_keys."
            )
        return self._data[key]

    def has(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data

    def keys(self) -> list[str]:
        """Return all available keys."""
        return list(self._data.keys())

    def snapshot(self) -> dict:
        """Returns a read-only copy of all data (for logging/persistence)."""
        return {
            k: {
                "value": v,
                "meta": {
                    "written_by": self._metadata[k].written_by,
                    "written_at": self._metadata[k].written_at.isoformat(),
                    "description": self._metadata[k].description,
                    "value_type": self._metadata[k].value_type,
                } if k in self._metadata else None,
            }
            for k, v in self._data.items()
        }

    def to_agent_context(self, keys: list[str]) -> str:
        """Format selected keys as a string for injection into agent prompts."""
        parts = []
        for key in keys:
            if key in self._data:
                try:
                    value_str = json.dumps(self._data[key], indent=2, default=str)
                except (TypeError, ValueError):
                    value_str = str(self._data[key])
                parts.append(f"## {key}\n{value_str}")
        return "\n\n".join(parts)
