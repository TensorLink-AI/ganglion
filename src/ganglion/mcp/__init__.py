"""MCP (Model Context Protocol) integration for Ganglion.

Provides two bridge directions:
  - MCPClientBridge: consume tools from external MCP servers as Ganglion ToolDefs
  - MCPServerBridge: expose Ganglion's ToolRegistry as an MCP server

Install with: pip install ganglion[mcp]
"""

from ganglion.mcp.config import MCPClientConfig
from ganglion.mcp.errors import MCPConnectionError, MCPError, MCPNotAvailableError, MCPToolError

__all__ = [
    "MCPClientConfig",
    "MCPConnectionError",
    "MCPError",
    "MCPNotAvailableError",
    "MCPToolError",
]
