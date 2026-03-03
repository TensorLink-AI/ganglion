"""Return type contracts for tools."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolOutput:
    """Base return type. Content is passed to the LLM as a string."""

    content: str
    structured: dict | None = None


@dataclass
class ExperimentResult(ToolOutput):
    """Returned by experiment-running tools."""

    experiment_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str | None = None


@dataclass
class ValidationResult(ToolOutput):
    """Returned by validation tools."""

    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
