"""Event types for pipeline observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PipelineEvent:
    """Base event type."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class StageStarted(PipelineEvent):
    """Emitted when a pipeline stage begins execution."""

    stage: str = ""


@dataclass
class StageCompleted(PipelineEvent):
    """Emitted when a pipeline stage finishes (success or failure)."""

    stage: str = ""
    result: Any = None


@dataclass
class StageRetry(PipelineEvent):
    """Emitted when a pipeline stage is being retried."""

    stage: str = ""
    attempt: int = 0
    policy_config: Any = None


@dataclass
class StageSkipped(PipelineEvent):
    """Emitted when a pipeline stage is skipped due to failed dependencies."""

    stage: str = ""
    reason: str = ""


@dataclass
class PipelineStarted(PipelineEvent):
    """Emitted when the full pipeline begins execution."""

    pipeline_name: str = ""


@dataclass
class PipelineCompleted(PipelineEvent):
    """Emitted when the full pipeline finishes."""

    pipeline_name: str = ""
    success: bool = False
