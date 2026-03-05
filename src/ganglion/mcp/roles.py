"""Multi-role MCP server configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPRole:
    """A single MCP server role with access control."""

    name: str
    token: str
    port: int
    categories: list[str] | None = None  # None = all categories
    transport: str = "sse"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPRole:
        return cls(
            name=data["name"],
            token=data["token"],
            port=data["port"],
            categories=data.get("categories"),
            transport=data.get("transport", "sse"),
        )


@dataclass
class MCPRolesConfig:
    """Configuration for multiple MCP server roles."""

    roles: list[MCPRole] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path) -> MCPRolesConfig:
        """Load roles from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return cls(roles=[MCPRole.from_dict(d) for d in data])

    def validate(self) -> list[str]:
        """Validate the roles configuration. Returns list of errors (empty = valid)."""
        errors: list[str] = []

        if not self.roles:
            errors.append("At least one role must be defined")
            return errors

        # Unique names
        names = [r.name for r in self.roles]
        if len(names) != len(set(names)):
            seen = set()
            for n in names:
                if n in seen:
                    errors.append(f"Duplicate role name: '{n}'")
                seen.add(n)

        # Unique ports (among SSE roles)
        sse_ports = [r.port for r in self.roles if r.transport == "sse"]
        if len(sse_ports) != len(set(sse_ports)):
            seen_ports: set[int] = set()
            for p in sse_ports:
                if p in seen_ports:
                    errors.append(f"Duplicate SSE port: {p}")
                seen_ports.add(p)

        # Non-empty tokens
        for r in self.roles:
            if not r.token:
                errors.append(f"Role '{r.name}' has empty token")

        # Non-empty names
        for r in self.roles:
            if not r.name:
                errors.append("Role name cannot be empty")

        # At most one stdio
        stdio_count = sum(1 for r in self.roles if r.transport == "stdio")
        if stdio_count > 1:
            errors.append(f"At most 1 role can use stdio transport, found {stdio_count}")

        # Valid transport
        for r in self.roles:
            if r.transport not in ("stdio", "sse"):
                errors.append(f"Role '{r.name}' has invalid transport: '{r.transport}'")

        return errors
