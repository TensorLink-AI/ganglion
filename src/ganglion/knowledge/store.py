"""KnowledgeStore — cross-run strategic memory."""

from __future__ import annotations

from typing import Any

from ganglion.knowledge.backends.base import KnowledgeBackend
from ganglion.knowledge.types import Antipattern, KnowledgeQuery, Pattern


class KnowledgeStore:
    """Cross-run strategic memory.

    Three things compound over time:
    - Patterns: strategies that worked (injected into prompts as guidance)
    - Antipatterns: strategies that failed (injected into prompts as warnings)
    - Generated tools: code written by agents, tested, and persisted
      (already handled by ToolRegistry + PersistenceBackend)

    The store is append-heavy and query-light. Writes happen at the end of
    each stage. Reads happen at the start of each stage via to_prompt_context().
    """

    def __init__(
        self,
        backend: KnowledgeBackend,
        max_patterns: int = 500,
        max_antipatterns: int = 500,
        bot_id: str | None = None,
    ):
        self.backend = backend
        self.max_patterns = max_patterns
        self.max_antipatterns = max_antipatterns
        self.bot_id = bot_id

    async def record_success(
        self,
        capability: str,
        description: str,
        config: dict[str, Any] | None = None,
        metric_value: float | None = None,
        metric_name: str | None = None,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Record a strategy that worked."""
        await self.backend.save_pattern(
            Pattern(
                capability=capability,
                description=description,
                config=config,
                metric_value=metric_value,
                metric_name=metric_name,
                stage=stage,
                run_id=run_id,
                source_bot=self.bot_id,
            )
        )

    async def record_failure(
        self,
        capability: str,
        error_summary: str,
        config: dict[str, Any] | None = None,
        failure_mode: str | None = None,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Record a strategy that failed."""
        await self.backend.save_antipattern(
            Antipattern(
                capability=capability,
                error_summary=error_summary[:500],
                config=config,
                failure_mode=failure_mode,
                stage=stage,
                run_id=run_id,
                source_bot=self.bot_id,
            )
        )

    async def to_prompt_context(
        self,
        capability: str,
        max_entries: int = 10,
    ) -> str:
        """Format relevant knowledge for injection into an agent's prompt.

        Returns a string block that can be appended to any system prompt.
        Agents receive only knowledge relevant to their capability.
        """
        patterns = await self.backend.query_patterns(
            KnowledgeQuery(capability=capability, max_entries=max_entries)
        )
        antipatterns = await self.backend.query_antipatterns(
            KnowledgeQuery(capability=capability, max_entries=max_entries)
        )

        if not patterns and not antipatterns:
            return ""

        lines = ["\n## Accumulated Knowledge"]

        if patterns:
            lines.append("\n### Known Good Approaches")
            for p in patterns:
                metric_str = (
                    f" (achieved {p.metric_name}={p.metric_value})"
                    if p.metric_value is not None
                    else ""
                )
                lines.append(f"- {p.description}{metric_str}")

        if antipatterns:
            lines.append("\n### Known Failures (avoid these)")
            for a in antipatterns:
                lines.append(f"- {a.error_summary}")
                if a.failure_mode:
                    lines.append(f"  Failure mode: {a.failure_mode}")

        return "\n".join(lines)

    async def to_foreign_prompt_context(
        self,
        capability: str,
        max_entries: int = 10,
    ) -> str:
        """Format knowledge from OTHER bots for injection into an agent's prompt.

        Returns "" if bot_id is not set (single-bot mode, no foreign knowledge).
        Otherwise queries patterns/antipatterns excluding this bot's own entries.
        """
        if self.bot_id is None:
            return ""

        query = KnowledgeQuery(
            capability=capability,
            max_entries=max_entries,
            exclude_source=self.bot_id,
        )
        patterns = await self.backend.query_patterns(query)
        antipatterns = await self.backend.query_antipatterns(query)

        if not patterns and not antipatterns:
            return ""

        lines = ["\n## Discoveries from other bots"]

        if patterns:
            lines.append("\n### Approaches that worked for others")
            for p in patterns:
                metric_str = (
                    f" (achieved {p.metric_name}={p.metric_value})"
                    if p.metric_value is not None
                    else ""
                )
                lines.append(f"- {p.description}{metric_str}")

        if antipatterns:
            lines.append("\n### Dead ends found by others (avoid these)")
            for a in antipatterns:
                lines.append(f"- {a.error_summary}")
                if a.failure_mode:
                    lines.append(f"  Failure mode: {a.failure_mode}")

        return "\n".join(lines)

    async def summary(self) -> dict[str, int]:
        """Snapshot for observation tools."""
        counts = await self.backend.count()
        return {
            "patterns": counts.get("patterns", 0),
            "antipatterns": counts.get("antipatterns", 0),
        }

    async def trim(self) -> None:
        """Evict oldest entries when limits are exceeded.
        Called automatically at the end of each pipeline run."""
        await self.backend.trim(self.max_patterns, self.max_antipatterns)
