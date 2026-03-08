"""Core types for the knowledge store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Pattern:
    """A strategy or configuration that produced good results."""

    capability: str
    description: str
    config: dict[str, Any] | None = None
    metric_value: float | None = None
    metric_name: str | None = None
    stage: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None = None
    source_bot: str | None = None
    subnet_id: str | None = None
    record_type: str = "strategy"
    confirmation_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "description": self.description,
            "config": self.config,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "source_bot": self.source_bot,
            "subnet_id": self.subnet_id,
            "record_type": self.record_type,
            "confirmation_count": self.confirmation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pattern:
        parsed_timestamp = data.get("timestamp")
        if isinstance(parsed_timestamp, str):
            parsed_timestamp = datetime.fromisoformat(parsed_timestamp)
        elif parsed_timestamp is None:
            parsed_timestamp = datetime.now(UTC)
        return cls(
            capability=data["capability"],
            description=data["description"],
            config=data.get("config"),
            metric_value=data.get("metric_value"),
            metric_name=data.get("metric_name"),
            stage=data.get("stage"),
            timestamp=parsed_timestamp,
            run_id=data.get("run_id"),
            source_bot=data.get("source_bot"),
            subnet_id=data.get("subnet_id"),
            record_type=data.get("record_type", "strategy"),
            confirmation_count=data.get("confirmation_count", 1),
        )


@dataclass
class Antipattern:
    """A strategy or configuration that failed, and why."""

    capability: str
    error_summary: str
    config: dict[str, Any] | None = None
    failure_mode: str | None = None
    stage: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None = None
    source_bot: str | None = None
    subnet_id: str | None = None
    record_type: str = "strategy"
    confirmation_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "error_summary": self.error_summary,
            "config": self.config,
            "failure_mode": self.failure_mode,
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "source_bot": self.source_bot,
            "subnet_id": self.subnet_id,
            "record_type": self.record_type,
            "confirmation_count": self.confirmation_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Antipattern:
        parsed_timestamp = data.get("timestamp")
        if isinstance(parsed_timestamp, str):
            parsed_timestamp = datetime.fromisoformat(parsed_timestamp)
        elif parsed_timestamp is None:
            parsed_timestamp = datetime.now(UTC)
        return cls(
            capability=data["capability"],
            error_summary=data["error_summary"],
            config=data.get("config"),
            failure_mode=data.get("failure_mode"),
            stage=data.get("stage"),
            timestamp=parsed_timestamp,
            run_id=data.get("run_id"),
            source_bot=data.get("source_bot"),
            subnet_id=data.get("subnet_id"),
            record_type=data.get("record_type", "strategy"),
            confirmation_count=data.get("confirmation_count", 1),
        )


@dataclass
class AgentDesignPattern:
    """Records what an agent looked like when it achieved an outcome."""

    capability: str
    agent_class: str
    tools: list[str] = field(default_factory=list)
    model: str | None = None
    metric_value: float | None = None
    metric_name: str | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_id: str | None = None
    source_bot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "agent_class": self.agent_class,
            "tools": self.tools,
            "model": self.model,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "fingerprint": self.fingerprint,
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "source_bot": self.source_bot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentDesignPattern:
        parsed_timestamp = data.get("timestamp")
        if isinstance(parsed_timestamp, str):
            parsed_timestamp = datetime.fromisoformat(parsed_timestamp)
        elif parsed_timestamp is None:
            parsed_timestamp = datetime.now(UTC)
        return cls(
            capability=data["capability"],
            agent_class=data["agent_class"],
            tools=data.get("tools", []),
            model=data.get("model"),
            metric_value=data.get("metric_value"),
            metric_name=data.get("metric_name"),
            fingerprint=data.get("fingerprint", {}),
            stage=data.get("stage"),
            timestamp=parsed_timestamp,
            run_id=data.get("run_id"),
            source_bot=data.get("source_bot"),
        )


@dataclass
class KnowledgeQuery:
    """Filter for retrieving relevant knowledge."""

    capability: str | None = None
    max_entries: int = 10
    since: datetime | None = None
    min_metric: float | None = None
    exclude_source: str | None = None
    subnet_id: str | None = None
    record_type: str | None = None
