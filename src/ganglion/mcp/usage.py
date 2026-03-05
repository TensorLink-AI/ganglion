"""Per-bot usage tracking for MCP server tool calls."""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UsageTracker:
    """Track per-bot tool call counts, success/failure, and duration.

    Maintains in-memory counters for fast access with optional SQLite
    persistence for durability across restarts.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        # In-memory counters: bot_id -> tool_name -> count
        self._calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # bot_id -> {total, success, failure}
        self._bot_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "failure": 0}
        )
        self._db_path = db_path
        if db_path:
            self._init_db()

    def _init_db(self) -> None:
        """Create usage_log table in SQLite."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_bot ON usage_log(bot_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_log(timestamp)")

    async def record(
        self,
        bot_id: str,
        tool_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record a tool call. Updates in-memory counters and optionally persists."""
        self._calls[bot_id][tool_name] += 1
        self._bot_totals[bot_id]["total"] += 1
        self._bot_totals[bot_id]["success" if success else "failure"] += 1

        if self._db_path:
            self._persist(bot_id, tool_name, success, duration_ms)

    def _persist(
        self,
        bot_id: str,
        tool_name: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Write a single record to SQLite."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO usage_log (bot_id, tool_name, success, timestamp, duration_ms)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    bot_id,
                    tool_name,
                    1 if success else 0,
                    datetime.now(timezone.utc).isoformat(),
                    duration_ms,
                ),
            )

    def get_bot_stats(self, bot_id: str) -> dict[str, Any]:
        """Return usage stats for a specific bot."""
        return {
            "bot_id": bot_id,
            "totals": dict(self._bot_totals.get(bot_id, {"total": 0, "success": 0, "failure": 0})),
            "per_tool": dict(self._calls.get(bot_id, {})),
        }

    def get_all_stats(self) -> list[dict[str, Any]]:
        """Return usage stats for all bots."""
        return [self.get_bot_stats(bid) for bid in sorted(self._bot_totals)]
