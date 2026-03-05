# Plan: Multi-Role MCP Servers & Per-Bot Usage Tracking

## Overview

Ganglion's MCP server currently runs as a single instance with no authentication
and no usage tracking. This plan adds two composing features plus comprehensive
test coverage aligned with the HTTP bridge test patterns.

1. **Multi-Role MCP Servers** — different bots get different tool access levels
2. **Per-Bot Usage Tracking** — track which bot calls which tools and how often
3. **Test Coverage** — mirror `test_bridge.py` patterns for full MCP coverage

All features are **opt-in** and backward-compatible. Without configuration,
the system behaves identically to today.

---

## Composition

```
CLI (_run_mcp_serve)
  │
  ├── loads roles.json            → MCPRolesConfig
  ├── creates shared UsageTracker → SQLite at .ganglion/usage.db
  │
  └── for each role:
        MCPServerBridge(
            tool_registry,
            categories = role.categories,
            token      = role.token,
            role       = role.name,
            usage_tracker = shared,
        )
        └── handle_call_tool(name, args):
              1. Resolve bot_id from role/token
              2. Execute tool, measure duration
              3. Record usage (UsageTracker)
              4. Return result
```

---

## Feature 1: Multi-Role MCP Servers

Run multiple MCP server instances from one process with different category
filters, auth tokens, and ports.

### Example `roles.json`

```json
[
  {"name": "admin",    "categories": null,                        "token": "admin-xyz",    "port": 8901},
  {"name": "worker",   "categories": ["observation","execution"], "token": "worker-abc",   "port": 8902},
  {"name": "observer", "categories": ["observation"],             "token": "observer-def", "port": 8903}
]
```

### Implementation

| Step | File | What |
|------|------|------|
| 1.1 | `src/ganglion/mcp/roles.py` (new) | `MCPRole` + `MCPRolesConfig` dataclasses with `from_file()` and `validate()` |
| 1.2 | `src/ganglion/mcp/server.py` | Add `token`, `role` params; bearer auth middleware in `run_sse()` |
| 1.3 | `src/ganglion/__main__.py` | `--roles` CLI flag; multi-server launcher with `asyncio.gather()` |
| 1.4 | `src/ganglion/mcp/server.py` | Token → role name mapping for bot identification |

---

## Feature 2: Per-Bot Usage Tracking

Track per-bot tool call counts, success/failure, timestamps, and duration.

### Implementation

| Step | File | What |
|------|------|------|
| 2.1 | `src/ganglion/mcp/usage.py` (new) | `UsageTracker` class: in-memory counters + optional SQLite persistence |
| 2.2 | `src/ganglion/mcp/server.py` | `usage_tracker` param; timing + recording in `handle_call_tool` |
| 2.3 | `src/ganglion/mcp/server.py` | `/usage` GET endpoint in SSE transport |
| 2.4 | `src/ganglion/__main__.py` | Shared `UsageTracker` creation, pass to each bridge |

---

## Feature 3: Test Coverage

Comprehensive tests aligned with `tests/test_bridge.py` patterns.

### New test classes in `tests/test_mcp.py`

| Class | Mirrors | Tests |
|-------|---------|-------|
| `TestMCPRolesConfig` | `TestInputValidation` | Config validation: duplicates, empty tokens, multi-stdio |
| `TestMCPServerAuth` | `TestSecurityHeaders` | Bearer token: missing, wrong, correct, no-auth fallback |
| `TestMCPServerBridgeRoles` | `TestMutationEndpoints` | Role-based tool filtering: admin/worker/observer |
| `TestUsageTracker` | (new) | In-memory + SQLite: counting, per-tool, multi-bot |
| `TestUsageTrackerIntegration` | (new) | Bridge integration: tool calls trigger recording |
| `TestUsageEndpoint` | `TestObservationEndpoints` | `/usage` endpoint: all bots, filtered, disabled |
| `TestMCPBackwardCompatibility` | `TestBackwardCompatibility` | No roles/token/tracker → works as today |

---

## Files Summary

| Action | File | What |
|--------|------|------|
| Create | `src/ganglion/mcp/usage.py` | UsageTracker |
| Create | `src/ganglion/mcp/roles.py` | MCPRole, MCPRolesConfig |
| Modify | `src/ganglion/mcp/server.py` | Auth, usage recording, /usage endpoint |
| Modify | `src/ganglion/__main__.py` | --roles flag, multi-server launcher |
| Modify | `src/ganglion/mcp/__init__.py` | Export new classes |
| Modify | `tests/test_mcp.py` | ~30 new tests across 7 test classes |

## Implementation Order

1. Usage Tracking (self-contained, simplest)
2. Multi-Role (establishes bot identity)
3. Tests (after implementation, full coverage)
