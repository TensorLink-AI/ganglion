"""ACP configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ACPClientConfig:
    """Configuration for connecting to one external ACP server.

    ACP uses a REST API for agent discovery and execution:
      GET  /agents       — list available agents
      GET  /agents/{id}  — get agent details
      POST /runs         — create a run (invoke an agent)
    """

    name: str
    url: str = ""  # base URL of the ACP server (e.g. "http://localhost:8080")
    agent_prefix: str = ""  # prepended to agent names: "{prefix}_{name}"
    token: str | None = None  # bearer token for auth
    timeout: float = 120.0  # per-run timeout in seconds
    headers: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate this configuration. Returns list of errors (empty = valid)."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.url:
            errors.append("url is required")
        if self.timeout <= 0:
            errors.append("timeout must be > 0")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes token for safety)."""
        return {
            "name": self.name,
            "url": self.url,
            "agent_prefix": self.agent_prefix,
            "timeout": self.timeout,
        }


@dataclass
class ACPServerConfig:
    """Configuration for exposing Ganglion agents as an ACP server."""

    host: str = "127.0.0.1"
    port: int = 8950
    token: str | None = None  # bearer token required from callers

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.port < 1 or self.port > 65535:
            errors.append("port must be between 1 and 65535")
        return errors
