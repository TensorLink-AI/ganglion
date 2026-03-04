"""File-based JSON knowledge backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ganglion.knowledge.types import Antipattern, KnowledgeQuery, Pattern

logger = logging.getLogger(__name__)


class JsonKnowledgeBackend:
    """Stores patterns and antipatterns as JSON files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._patterns_path = self.directory / "patterns.json"
        self._antipatterns_path = self.directory / "antipatterns.json"

    def _load_patterns(self) -> list[dict]:
        if self._patterns_path.exists():
            try:
                return json.loads(self._patterns_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load patterns: %s", e)
        return []

    def _save_patterns(self, data: list[dict]) -> None:
        self._patterns_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_antipatterns(self) -> list[dict]:
        if self._antipatterns_path.exists():
            try:
                return json.loads(self._antipatterns_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load antipatterns: %s", e)
        return []

    def _save_antipatterns(self, data: list[dict]) -> None:
        self._antipatterns_path.write_text(json.dumps(data, indent=2, default=str))

    async def save_pattern(self, pattern: Pattern) -> None:
        data = self._load_patterns()
        data.append(pattern.to_dict())
        self._save_patterns(data)

    async def save_antipattern(self, antipattern: Antipattern) -> None:
        data = self._load_antipatterns()
        data.append(antipattern.to_dict())
        self._save_antipatterns(data)

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

        antipatterns.sort(key=lambda a: a.timestamp, reverse=True)
        return antipatterns[: query.max_entries]

    async def count(self) -> dict[str, int]:
        return {
            "patterns": len(self._load_patterns()),
            "antipatterns": len(self._load_antipatterns()),
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
