"""MCP (Model Context Protocol) integration for Ganglion.

Provides two bridge directions:
  - MCPClientBridge: consume tools from external MCP servers as Ganglion ToolDefs
  - MCPServerBridge: expose Ganglion's ToolRegistry as an MCP server

Multi-role support:
  - MCPRole / MCPRolesConfig: define roles with different access levels
  - UsageTracker: per-bot tool call tracking
  - register_framework_tools: expose all HTTP bridge operations as MCP tools

Install with: pip install ganglion[mcp]
"""

from ganglion.mcp.config import MCPClientConfig
from ganglion.mcp.errors import MCPConnectionError, MCPError, MCPNotAvailableError, MCPToolError
from ganglion.mcp.roles import MCPRole, MCPRolesConfig
from ganglion.mcp.usage import UsageTracker

__all__ = [
    "MCPClientConfig",
    "MCPConnectionError",
    "MCPError",
    "MCPNotAvailableError",
    "MCPRole",
    "MCPRolesConfig",
    "MCPToolError",
    "UsageTracker",
]
