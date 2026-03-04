"""MCP configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MCPClientConfig:
    """Configuration for connecting to one external MCP server."""

    name: str
    transport: str = "stdio"  # "stdio" or "sse"
    command: list[str] | None = None  # stdio: command + args to spawn the MCP server
    url: str | None = None  # sse: endpoint URL
    env: dict[str, str] | None = None  # extra env vars for stdio subprocess
    tool_prefix: str = ""  # prepended to tool names: "{prefix}_{name}"
    category: str = "mcp"  # ToolDef category for all tools from this server
    timeout: float = 30.0  # per-call timeout in seconds

    def validate(self) -> list[str]:
        """Validate this configuration. Returns list of errors (empty = valid)."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if self.transport not in ("stdio", "sse"):
            errors.append(f"transport must be 'stdio' or 'sse', got '{self.transport}'")
        if self.transport == "stdio" and not self.command:
            errors.append("command is required for stdio transport")
        if self.transport == "sse" and not self.url:
            errors.append("url is required for sse transport")
        if self.timeout <= 0:
            errors.append("timeout must be > 0")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "url": self.url,
            "tool_prefix": self.tool_prefix,
            "category": self.category,
            "timeout": self.timeout,
        }
