-- Ganglion Knowledge Plugin — SQLite schema
-- Tables for the 8 MCP tools: patterns, antipatterns, shared_tools, subnet_configs

CREATE TABLE patterns (
  id INTEGER PRIMARY KEY,
  capability TEXT NOT NULL,
  description TEXT NOT NULL,
  config TEXT,
  tags TEXT,
  metric_value REAL,
  metric_name TEXT,
  stage TEXT,
  run_id TEXT,
  source_bot TEXT NOT NULL,
  subnet_id TEXT,
  record_type TEXT DEFAULT 'strategy',
  confirmation_count INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE antipatterns (
  id INTEGER PRIMARY KEY,
  capability TEXT NOT NULL,
  error_summary TEXT NOT NULL,
  config TEXT,
  tags TEXT,
  failure_mode TEXT,
  stage TEXT,
  run_id TEXT,
  source_bot TEXT NOT NULL,
  subnet_id TEXT,
  record_type TEXT DEFAULT 'strategy',
  confirmation_count INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE shared_tools (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  code TEXT NOT NULL,
  category TEXT DEFAULT 'general',
  author TEXT NOT NULL,
  subnet_id TEXT,
  version INTEGER DEFAULT 1,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE subnet_configs (
  id INTEGER PRIMARY KEY,
  subnet_id TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  config TEXT NOT NULL,
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_patterns_subnet ON patterns(subnet_id);
CREATE INDEX idx_patterns_capability ON patterns(capability);
CREATE INDEX idx_patterns_source ON patterns(source_bot);
CREATE INDEX idx_patterns_tags ON patterns(tags);
CREATE INDEX idx_antipatterns_subnet ON antipatterns(subnet_id);
CREATE INDEX idx_antipatterns_capability ON antipatterns(capability);
CREATE INDEX idx_antipatterns_tags ON antipatterns(tags);
CREATE INDEX idx_shared_tools_name ON shared_tools(name);
CREATE INDEX idx_subnet_configs_subnet ON subnet_configs(subnet_id);
