"""SQLite-based knowledge backend for larger deployments."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from ganglion.knowledge.types import AgentDesignPattern, Antipattern, KnowledgeQuery, Pattern

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
                    source_bot TEXT,
                    subnet_id TEXT,
                    record_type TEXT DEFAULT 'strategy',
                    confirmation_count INTEGER DEFAULT 1
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
                    source_bot TEXT,
                    subnet_id TEXT,
                    record_type TEXT DEFAULT 'strategy',
                    confirmation_count INTEGER DEFAULT 1
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
            # Migration: add new columns to existing tables
            for table in ("patterns", "antipatterns"):
                existing = {
                    row[1]
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                if "subnet_id" not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN subnet_id TEXT")
                if "record_type" not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN record_type TEXT DEFAULT 'strategy'"
                    )
                if "confirmation_count" not in existing:
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN confirmation_count INTEGER DEFAULT 1"
                    )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_designs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capability TEXT NOT NULL,
                    agent_class TEXT NOT NULL,
                    tools TEXT,
                    model TEXT,
                    metric_value REAL,
                    metric_name TEXT,
                    fingerprint TEXT,
                    stage TEXT,
                    timestamp TEXT NOT NULL,
                    run_id TEXT,
                    source_bot TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_agent_designs_capability ON agent_designs(capability)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_agent_designs_source_bot ON agent_designs(source_bot)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_designs_timestamp ON agent_designs(timestamp)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    async def save_pattern(self, pattern: Pattern) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO patterns
                   (capability, description, config, metric_value,
                    metric_name, stage, timestamp, run_id, source_bot,
                    subnet_id, record_type, confirmation_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    pattern.subnet_id,
                    pattern.record_type,
                    pattern.confirmation_count,
                ),
            )

    async def save_antipattern(self, antipattern: Antipattern) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO antipatterns
                   (capability, error_summary, config, failure_mode,
                    stage, timestamp, run_id, source_bot,
                    subnet_id, record_type, confirmation_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    antipattern.capability,
                    antipattern.error_summary,
                    json.dumps(antipattern.config) if antipattern.config else None,
                    antipattern.failure_mode,
                    antipattern.stage,
                    antipattern.timestamp.isoformat(),
                    antipattern.run_id,
                    antipattern.source_bot,
                    antipattern.subnet_id,
                    antipattern.record_type,
                    antipattern.confirmation_count,
                ),
            )

    async def save_agent_design(self, design: AgentDesignPattern) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO agent_designs
                   (capability, agent_class, tools, model, metric_value,
                    metric_name, fingerprint, stage, timestamp, run_id, source_bot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    design.capability,
                    design.agent_class,
                    json.dumps(design.tools),
                    design.model,
                    design.metric_value,
                    design.metric_name,
                    json.dumps(design.fingerprint),
                    design.stage,
                    design.timestamp.isoformat(),
                    design.run_id,
                    design.source_bot,
                ),
            )

    async def query_agent_designs(self, query: KnowledgeQuery) -> list[AgentDesignPattern]:
        conditions = []
        params: list[Any] = []

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
        sql = f"SELECT * FROM agent_designs {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(query.max_entries)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_agent_design(row) for row in rows]

    async def query_patterns(self, query: KnowledgeQuery) -> list[Pattern]:
        conditions = []
        params: list[Any] = []

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
        if query.subnet_id is not None:
            conditions.append("subnet_id = ?")
            params.append(query.subnet_id)
        if query.record_type is not None:
            conditions.append("record_type = ?")
            params.append(query.record_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM patterns {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(query.max_entries)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_pattern(row) for row in rows]

    async def query_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]:
        conditions = []
        params: list[Any] = []

        if query.capability:
            conditions.append("capability = ?")
            params.append(query.capability)
        if query.since:
            conditions.append("timestamp >= ?")
            params.append(query.since.isoformat())
        if query.exclude_source is not None:
            conditions.append("(source_bot IS NULL OR source_bot != ?)")
            params.append(query.exclude_source)
        if query.subnet_id is not None:
            conditions.append("subnet_id = ?")
            params.append(query.subnet_id)
        if query.record_type is not None:
            conditions.append("record_type = ?")
            params.append(query.record_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM antipatterns {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(query.max_entries)

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_antipattern(row) for row in rows]

    async def find_similar_pattern(
        self, capability: str, description: str, record_type: str = "strategy"
    ) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM patterns"
                " WHERE capability = ? AND description = ? AND record_type = ? LIMIT 1",
                (capability, description, record_type),
            ).fetchone()
        return row[0] if row else None

    async def find_similar_antipattern(
        self, capability: str, error_summary: str, record_type: str = "strategy"
    ) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM antipatterns"
                " WHERE capability = ? AND error_summary = ? AND record_type = ? LIMIT 1",
                (capability, error_summary, record_type),
            ).fetchone()
        return row[0] if row else None

    async def increment_confirmation(self, table: str, record_id: int) -> None:
        if table not in ("patterns", "antipatterns"):
            raise ValueError(f"Invalid table: {table}")
        with self._connect() as conn:
            conn.execute(
                f"UPDATE {table} SET confirmation_count = confirmation_count + 1 WHERE id = ?",
                (record_id,),
            )

    async def count(self) -> dict[str, int]:
        with self._connect() as conn:
            patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            antipatterns = conn.execute("SELECT COUNT(*) FROM antipatterns").fetchone()[0]
            agent_designs = conn.execute("SELECT COUNT(*) FROM agent_designs").fetchone()[0]
        return {"patterns": patterns, "antipatterns": antipatterns, "agent_designs": agent_designs}

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
            subnet_id=row["subnet_id"],
            record_type=row["record_type"] or "strategy",
            confirmation_count=row["confirmation_count"] or 1,
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
            subnet_id=row["subnet_id"],
            record_type=row["record_type"] or "strategy",
            confirmation_count=row["confirmation_count"] or 1,
        )

    def _row_to_agent_design(self, row: sqlite3.Row) -> AgentDesignPattern:
        from datetime import datetime

        return AgentDesignPattern(
            capability=row["capability"],
            agent_class=row["agent_class"],
            tools=json.loads(row["tools"]) if row["tools"] else [],
            model=row["model"],
            metric_value=row["metric_value"],
            metric_name=row["metric_name"],
            fingerprint=json.loads(row["fingerprint"]) if row["fingerprint"] else {},
            stage=row["stage"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            run_id=row["run_id"],
            source_bot=row["source_bot"],
        )
