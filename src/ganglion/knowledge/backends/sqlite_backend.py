"""SQLite-based knowledge backend for larger deployments."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ganglion.knowledge.types import Antipattern, KnowledgeQuery, Pattern

logger = logging.getLogger(__name__)


class SqliteKnowledgeBackend:
    """SQLite-backed knowledge storage with indexed queries."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capability TEXT NOT NULL,
                    description TEXT NOT NULL,
                    config TEXT,
                    metric_value REAL,
                    metric_name TEXT,
                    stage TEXT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT,
                    source_bot TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS antipatterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capability TEXT NOT NULL,
                    error_summary TEXT NOT NULL,
                    config TEXT,
                    failure_mode TEXT,
                    stage TEXT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT,
                    source_bot TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_capability ON patterns(capability)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_antipatterns_capability ON antipatterns(capability)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_source_bot ON patterns(source_bot)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_antipatterns_source_bot ON antipatterns(source_bot)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_timestamp ON patterns(timestamp)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_antipatterns_timestamp ON antipatterns(timestamp)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    async def save_pattern(self, pattern: Pattern) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO patterns
                   (capability, description, config, metric_value,
                    metric_name, stage, timestamp, run_id, source_bot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pattern.capability,
                    pattern.description,
                    json.dumps(pattern.config) if pattern.config else None,
                    pattern.metric_value,
                    pattern.metric_name,
                    pattern.stage,
                    pattern.timestamp.isoformat(),
                    pattern.run_id,
                    pattern.source_bot,
                ),
            )

    async def save_antipattern(self, antipattern: Antipattern) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO antipatterns
                   (capability, error_summary, config, failure_mode,
                    stage, timestamp, run_id, source_bot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    antipattern.capability,
                    antipattern.error_summary,
                    json.dumps(antipattern.config) if antipattern.config else None,
                    antipattern.failure_mode,
                    antipattern.stage,
                    antipattern.timestamp.isoformat(),
                    antipattern.run_id,
                    antipattern.source_bot,
                ),
            )

    async def query_patterns(self, query: KnowledgeQuery) -> list[Pattern]:
        conditions = []
        params: list = []

        if query.capability:
            conditions.append("capability = ?")
            params.append(query.capability)
        if query.since:
            conditions.append("timestamp >= ?")
            params.append(query.since.isoformat())
        if query.min_metric is not None:
            conditions.append("metric_value >= ?")
            params.append(query.min_metric)
        if query.exclude_source is not None:
            conditions.append("(source_bot IS NULL OR source_bot != ?)")
            params.append(query.exclude_source)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM patterns {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(query.max_entries)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_pattern(row) for row in rows]

    async def query_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]:
        conditions = []
        params: list = []

        if query.capability:
            conditions.append("capability = ?")
            params.append(query.capability)
        if query.since:
            conditions.append("timestamp >= ?")
            params.append(query.since.isoformat())
        if query.exclude_source is not None:
            conditions.append("(source_bot IS NULL OR source_bot != ?)")
            params.append(query.exclude_source)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM antipatterns {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(query.max_entries)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_antipattern(row) for row in rows]

    async def count(self) -> dict[str, int]:
        with self._connect() as conn:
            patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            antipatterns = conn.execute("SELECT COUNT(*) FROM antipatterns").fetchone()[0]
        return {"patterns": patterns, "antipatterns": antipatterns}

    async def trim(self, max_patterns: int = 500, max_antipatterns: int = 500) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            if count > max_patterns:
                conn.execute(
                    """DELETE FROM patterns WHERE id NOT IN
                       (SELECT id FROM patterns ORDER BY timestamp DESC LIMIT ?)""",
                    (max_patterns,),
                )

            count = conn.execute("SELECT COUNT(*) FROM antipatterns").fetchone()[0]
            if count > max_antipatterns:
                conn.execute(
                    """DELETE FROM antipatterns WHERE id NOT IN
                       (SELECT id FROM antipatterns ORDER BY timestamp DESC LIMIT ?)""",
                    (max_antipatterns,),
                )

    def _row_to_pattern(self, row: sqlite3.Row) -> Pattern:
        config = json.loads(row["config"]) if row["config"] else None
        from datetime import datetime

        return Pattern(
            capability=row["capability"],
            description=row["description"],
            config=config,
            metric_value=row["metric_value"],
            metric_name=row["metric_name"],
            stage=row["stage"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            run_id=row["run_id"],
            source_bot=row["source_bot"],
        )

    def _row_to_antipattern(self, row: sqlite3.Row) -> Antipattern:
        config = json.loads(row["config"]) if row["config"] else None
        from datetime import datetime

        return Antipattern(
            capability=row["capability"],
            error_summary=row["error_summary"],
            config=config,
            failure_mode=row["failure_mode"],
            stage=row["stage"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            run_id=row["run_id"],
            source_bot=row["source_bot"],
        )
