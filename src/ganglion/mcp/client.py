"""MCPClientBridge — connects to external MCP servers and wraps their tools as ToolDefs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from ganglion.composition.tool_registry import ToolDef
from ganglion.composition.tool_returns import ToolOutput
from ganglion.mcp.config import MCPClientConfig
from ganglion.mcp.errors import MCPConnectionError, MCPNotAvailableError, MCPToolError

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _check_mcp_available() -> None:
    if not MCP_AVAILABLE:
        raise MCPNotAvailableError()


class MCPClientBridge:
    """Connects to an external MCP server and wraps its tools as Ganglion ToolDefs."""

    def __init__(self, config: MCPClientConfig):
        _check_mcp_available()
        self.config = config
        self.session: Any | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: dict[str, ToolDef] = {}

    async def connect(self) -> list[ToolDef]:
        """Establish connection and discover tools. Returns list of wrapped ToolDefs."""
        errors = self.config.validate()
        if errors:
            raise MCPConnectionError(
                f"Invalid config for '{self.config.name}': {'; '.join(errors)}"
            )

        self._exit_stack = AsyncExitStack()
        try:
            if self.config.transport == "stdio":
                await self._connect_stdio()
            elif self.config.transport == "sse":
                await self._connect_sse()

            await self.session.initialize()
            await self._discover_tools()
            return list(self._tools.values())

        except MCPConnectionError:
            raise
        except Exception as e:
            await self._cleanup()
            raise MCPConnectionError(
                f"Failed to connect to MCP server '{self.config.name}': {e}"
            ) from e

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        assert self._exit_stack is not None
        assert self.config.command is not None

        params = StdioServerParameters(
            command=self.config.command[0],
            args=self.config.command[1:] if len(self.config.command) > 1 else [],
            env=self.config.env,
        )
        transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read, write = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

    async def _connect_sse(self) -> None:
        """Connect via SSE transport."""
        assert self._exit_stack is not None

        try:
            from mcp.client.sse import sse_client
        except ImportError as e:
            raise MCPConnectionError(
                "SSE transport requires additional dependencies from the mcp package"
            ) from e

        transport = await self._exit_stack.enter_async_context(
            sse_client(self.config.url)
        )
        read, write = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

    async def _discover_tools(self) -> None:
        """List tools from the MCP server and create ToolDef wrappers."""
        result = await self.session.list_tools()
        prefix = self.config.tool_prefix or self.config.name

        for mcp_tool in result.tools:
            tool_name = f"{prefix}_{mcp_tool.name}" if prefix else mcp_tool.name

            handler = self._make_handler(mcp_tool.name)

            tool_def = ToolDef(
                name=tool_name,
                description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                func=handler,
                parameters_schema=mcp_tool.inputSchema
                or {"type": "object", "properties": {}},
                category=self.config.category,
            )
            self._tools[tool_name] = tool_def

        logger.info(
            "MCP server '%s': discovered %d tools", self.config.name, len(self._tools)
        )

    def _make_handler(self, tool_name: str) -> Callable[..., Any]:
        """Create an async handler closure that calls the MCP tool."""
        session = self.session
        timeout = self.config.timeout
        server_name = self.config.name

        async def handler(**kwargs: Any) -> ToolOutput:
            if session is None:
                raise MCPToolError(f"MCP server '{server_name}' is not connected")

            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=kwargs),
                    timeout=timeout,
                )
            except TimeoutError as e:
                raise MCPToolError(
                    f"MCP tool '{tool_name}' on '{server_name}' timed out after {timeout}s"
                ) from e

            # Extract text content from result blocks
            text_parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            content = "\n".join(text_parts) or str(result.content)

            if result.isError:
                return ToolOutput(content=f"MCP Error: {content}")
            return ToolOutput(content=content)

        return handler

    def get_tools(self) -> dict[str, ToolDef]:
        """Return discovered tools as ToolDefs."""
        return dict(self._tools)

    async def disconnect(self) -> None:
        """Cleanly disconnect from the MCP server."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning(
                    "Error closing MCP connection to '%s': %s", self.config.name, e
                )
            self._exit_stack = None
        self.session = None
        self._tools.clear()
