"""MCP-specific error types."""

from __future__ import annotations


class MCPError(Exception):
    """Base class for MCP errors."""

    pass


class MCPConnectionError(MCPError):
    """Failed to connect to an MCP server."""

    pass


class MCPToolError(MCPError):
    """MCP tool returned an error or timed out."""

    pass


class MCPNotAvailableError(MCPError):
    """The mcp package is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "MCP support requires the 'mcp' package. Install it with: pip install ganglion[mcp]"
        )
