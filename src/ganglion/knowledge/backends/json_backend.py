"""File-based JSON knowledge backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ganglion.knowledge.types import AgentDesignPattern, Antipattern, KnowledgeQuery, Pattern

logger = logging.getLogger(__name__)


class JsonKnowledgeBackend:
    """Stores patterns and antipatterns as JSON files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._patterns_path = self.directory / "patterns.json"
        self._antipatterns_path = self.directory / "antipatterns.json"
        self._agent_designs_path = self.directory / "agent_designs.json"

    def _load_patterns(self) -> list[dict[str, Any]]:
        if self._patterns_path.exists():
            try:
                result: list[dict[str, Any]] = json.loads(self._patterns_path.read_text())
                return result
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load patterns: %s", e)
        return []

    def _save_patterns(self, data: list[dict[str, Any]]) -> None:
        self._patterns_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_antipatterns(self) -> list[dict[str, Any]]:
        if self._antipatterns_path.exists():
            try:
                result: list[dict[str, Any]] = json.loads(self._antipatterns_path.read_text())
                return result
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load antipatterns: %s", e)
        return []

    def _save_antipatterns(self, data: list[dict[str, Any]]) -> None:
        self._antipatterns_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_agent_designs(self) -> list[dict[str, Any]]:
        if self._agent_designs_path.exists():
            try:
                result: list[dict[str, Any]] = json.loads(self._agent_designs_path.read_text())
                return result
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load agent designs: %s", e)
        return []

    def _save_agent_designs(self, data: list[dict[str, Any]]) -> None:
        self._agent_designs_path.write_text(json.dumps(data, indent=2, default=str))

    async def save_pattern(self, pattern: Pattern) -> None:
        data = self._load_patterns()
        data.append(pattern.to_dict())
        self._save_patterns(data)

    async def save_antipattern(self, antipattern: Antipattern) -> None:
        data = self._load_antipatterns()
        data.append(antipattern.to_dict())
        self._save_antipatterns(data)

    async def save_agent_design(self, design: AgentDesignPattern) -> None:
        data = self._load_agent_designs()
        data.append(design.to_dict())
        self._save_agent_designs(data)

    async def query_agent_designs(self, query: KnowledgeQuery) -> list[AgentDesignPattern]:
        data = self._load_agent_designs()
        designs = [AgentDesignPattern.from_dict(d) for d in data]

        if query.capability:
            designs = [d for d in designs if d.capability == query.capability]
        if query.since:
            designs = [d for d in designs if d.timestamp >= query.since]
        if query.min_metric is not None:
            designs = [
                d
                for d in designs
                if d.metric_value is not None and d.metric_value >= query.min_metric
            ]
        if query.exclude_source is not None:
            designs = [d for d in designs if d.source_bot != query.exclude_source]

        designs.sort(key=lambda d: d.timestamp, reverse=True)
        return designs[: query.max_entries]

    async def query_patterns(self, query: KnowledgeQuery) -> list[Pattern]:
        data = self._load_patterns()
        patterns = [Pattern.from_dict(d) for d in data]

        if query.capability:
            patterns = [p for p in patterns if p.capability == query.capability]
        if query.since:
            patterns = [p for p in patterns if p.timestamp >= query.since]
        if query.min_metric is not None:
            patterns = [
                p
                for p in patterns
                if p.metric_value is not None and p.metric_value >= query.min_metric
            ]
        if query.exclude_source is not None:
            patterns = [p for p in patterns if p.source_bot != query.exclude_source]
        if query.subnet_id is not None:
            patterns = [p for p in patterns if p.subnet_id == query.subnet_id]
        if query.record_type is not None:
            patterns = [p for p in patterns if p.record_type == query.record_type]

        # Most recent first, limited to max_entries
        patterns.sort(key=lambda p: p.timestamp, reverse=True)
        return patterns[: query.max_entries]

    async def query_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]:
        data = self._load_antipatterns()
        antipatterns = [Antipattern.from_dict(d) for d in data]

        if query.capability:
            antipatterns = [a for a in antipatterns if a.capability == query.capability]
        if query.since:
            antipatterns = [a for a in antipatterns if a.timestamp >= query.since]
        if query.exclude_source is not None:
            antipatterns = [a for a in antipatterns if a.source_bot != query.exclude_source]
        if query.subnet_id is not None:
            antipatterns = [a for a in antipatterns if a.subnet_id == query.subnet_id]
        if query.record_type is not None:
            antipatterns = [a for a in antipatterns if a.record_type == query.record_type]

        antipatterns.sort(key=lambda a: a.timestamp, reverse=True)
        return antipatterns[: query.max_entries]

    async def find_similar_pattern(
        self, capability: str, description: str, record_type: str = "strategy"
    ) -> int | None:
        data = self._load_patterns()
        for i, d in enumerate(data):
            if (
                d.get("capability") == capability
                and d.get("description") == description
                and d.get("record_type", "strategy") == record_type
            ):
                return i
        return None

    async def find_similar_antipattern(
        self, capability: str, error_summary: str, record_type: str = "strategy"
    ) -> int | None:
        data = self._load_antipatterns()
        for i, d in enumerate(data):
            if (
                d.get("capability") == capability
                and d.get("error_summary") == error_summary
                and d.get("record_type", "strategy") == record_type
            ):
                return i
        return None

    async def increment_confirmation(self, table: str, record_id: int) -> None:
        if table == "patterns":
            data = self._load_patterns()
            data[record_id]["confirmation_count"] = data[record_id].get("confirmation_count", 1) + 1
            self._save_patterns(data)
        elif table == "antipatterns":
            data = self._load_antipatterns()
            data[record_id]["confirmation_count"] = data[record_id].get("confirmation_count", 1) + 1
            self._save_antipatterns(data)
        else:
            raise ValueError(f"Invalid table: {table}")

    async def count(self) -> dict[str, int]:
        return {
            "patterns": len(self._load_patterns()),
            "antipatterns": len(self._load_antipatterns()),
            "agent_designs": len(self._load_agent_designs()),
        }

    async def trim(self, max_patterns: int = 500, max_antipatterns: int = 500) -> None:
        """Evict oldest entries when limits are exceeded."""
        patterns = self._load_patterns()
        if len(patterns) > max_patterns:
            # Keep most recent
            patterns.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
            patterns = patterns[:max_patterns]
            self._save_patterns(patterns)

        antipatterns = self._load_antipatterns()
        if len(antipatterns) > max_antipatterns:
            antipatterns.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
            antipatterns = antipatterns[:max_antipatterns]
            self._save_antipatterns(antipatterns)
