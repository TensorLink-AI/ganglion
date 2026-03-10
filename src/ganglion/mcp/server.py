"""MCPServerBridge — exposes Ganglion's registered tools as an MCP server."""

from __future__ import annotations

import inspect
import logging
import time
from typing import TYPE_CHECKING, Any

from ganglion.mcp.errors import MCPNotAvailableError
from ganglion.state.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from ganglion.mcp.usage import UsageTracker

logger = logging.getLogger(__name__)

try:
    from mcp.server import Server
    from mcp.types import TextContent, Tool

    MCP_SERVER_AVAILABLE = True
except ImportError:
    MCP_SERVER_AVAILABLE = False


class MCPServerBridge:
    """Expose Ganglion's registered tools as an MCP server."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        server_name: str = "ganglion",
        categories: list[str] | None = None,
        token: str | None = None,
        role: str | None = None,
        usage_tracker: UsageTracker | None = None,
    ):
        if not MCP_SERVER_AVAILABLE:
            raise MCPNotAvailableError()

        self._registry = tool_registry
        self._categories = categories
        self._token = token
        self.role = role
        self._usage_tracker = usage_tracker
        self._server = Server(server_name)
        self._setup_handlers()

    def _resolve_bot_id(self) -> str | None:
        """Resolve the bot identifier from role name."""
        return self.role

    def _setup_handlers(self) -> None:
        """Register list_tools and call_tool handlers on the MCP server."""

        @self._server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def handle_list_tools() -> list[Tool]:
            tools = []
            try:
                tool_list = self._registry.list_all()
            except Exception as e:
                logger.error("Failed to list tools from registry: %s", e, exc_info=True)
                return []

            for tool_dict in tool_list:
                # Apply category filter
                if self._categories and tool_dict.get("category") not in self._categories:
                    continue
                try:
                    tools.append(
                        Tool(
                            name=tool_dict["name"],
                            description=tool_dict.get("description", ""),
                            inputSchema=tool_dict.get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Skipping tool '%s' due to schema error: %s",
                        tool_dict.get("name", "<unknown>"),
                        e,
                    )
            return tools

        @self._server.call_tool()  # type: ignore[untyped-decorator]
        async def handle_call_tool(
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> list[TextContent]:
            tool_def = self._registry.get(name)
            if tool_def is None:
                return [TextContent(type="text", text=f"Error: Tool '{name}' not found")]

            # Enforce category filter on call_tool too
            if self._categories and tool_def.category not in self._categories:
                msg = f"Error: Tool '{name}' not available for this role"
                return [TextContent(type="text", text=msg)]

            bot_id = self._resolve_bot_id()
            start = time.monotonic()
            success = True

            try:
                result = tool_def.func(**(arguments or {}))
                if inspect.isawaitable(result):
                    result = await result

                if hasattr(result, "content"):
                    return [TextContent(type="text", text=str(result.content))]
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                success = False
                logger.error("MCP tool '%s' raised: %s", name, e, exc_info=True)
                return [TextContent(type="text", text=f"Error: {e}")]
            finally:
                if self._usage_tracker and bot_id:
                    try:
                        elapsed_ms = (time.monotonic() - start) * 1000
                        await self._usage_tracker.record(bot_id, name, success, elapsed_ms)
                    except Exception as e:
                        logger.warning("Failed to record usage for tool '%s': %s", name, e)

    async def run_stdio(self) -> None:
        """Run as stdio transport MCP server."""
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await self._server.run(read, write, self._server.create_initialization_options())

    async def run_sse(self, host: str = "127.0.0.1", port: int = 8900) -> None:
        """Run as SSE transport MCP server.

        Serves both the legacy SSE transport (``/sse`` + ``/messages/``)
        and the newer Streamable HTTP transport (``/mcp``) so that old
        and new MCP clients can connect to the same port.
        """
        import contextlib
        from collections.abc import AsyncIterator

        import uvicorn
        from mcp.server.sse import SseServerTransport
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        sse = SseServerTransport("/messages/")

        def _check_auth_asgi(scope: dict[str, Any]) -> bool:
            """Check bearer token from raw ASGI scope headers.

            Returns True if authorised (or no token required).
            """
            if not self._token:
                return True
            headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
            for name, value in headers:
                if name == b"authorization":
                    return value.decode() == f"Bearer {self._token}"
            return False

        # -- SSE and message handlers are raw ASGI applications --------
        # connect_sse and handle_post_message send HTTP responses
        # directly via the ASGI ``send`` callable.  They must NOT be
        # wrapped in Starlette's request→Response pattern (which would
        # double-send), and must NOT use Mount (which rewrites
        # scope["root_path"], causing the MCP SDK to advertise a wrong
        # messages URL to clients).
        #
        # Starlette's Route only treats *classes* as raw ASGI apps and
        # wraps plain functions with request_response().  We therefore
        # wrap each handler in a minimal ASGI class.

        mcp_server = self._server

        class _SSEApp:
            async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
                if not _check_auth_asgi(scope):
                    resp = Response("Unauthorized", status_code=401)
                    await resp(scope, receive, send)
                    return
                try:
                    async with sse.connect_sse(scope, receive, send) as streams:
                        await mcp_server.run(
                            streams[0],
                            streams[1],
                            mcp_server.create_initialization_options(),
                        )
                except Exception as e:
                    logger.error("SSE connection error: %s", e, exc_info=True)

        class _MessageApp:
            async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
                if not _check_auth_asgi(scope):
                    resp = Response("Unauthorized", status_code=401)
                    await resp(scope, receive, send)
                    return
                try:
                    await sse.handle_post_message(scope, receive, send)
                except Exception as e:
                    logger.error("MCP message handling error: %s", e, exc_info=True)

        # -- Streamable HTTP transport (MCP 2025-03-26+) ---------------
        # Modern clients (Claude Desktop, Claude Code, etc.) POST to
        # /mcp with JSON-RPC over Streamable HTTP instead of the legacy
        # SSE two-endpoint pattern.  We delegate to the SDK's
        # StreamableHTTPSessionManager which handles session lifecycle,
        # transport creation, and server.run() orchestration.

        session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            json_response=True,
        )

        class _StreamableApp:
            """Delegates to StreamableHTTPSessionManager with auth."""

            async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
                if not _check_auth_asgi(scope):
                    resp = Response("Unauthorized", status_code=401)
                    await resp(scope, receive, send)
                    return
                await session_manager.handle_request(scope, receive, send)

        # -- Standard request→Response endpoints -----------------------

        async def handle_usage(request: Request) -> Response:
            if not self._usage_tracker:
                return JSONResponse({"error": "Usage tracking not enabled"}, status_code=404)
            bot_id = request.query_params.get("bot_id")
            if bot_id:
                return JSONResponse(self._usage_tracker.get_bot_stats(bot_id))
            return JSONResponse(self._usage_tracker.get_all_stats())

        async def handle_healthz(request: Request) -> Response:
            return JSONResponse({"status": "ok"})

        async def handle_readyz(request: Request) -> Response:
            return JSONResponse({"status": "ready", "role": self.role})

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            async with session_manager.run():
                logger.info("Streamable HTTP session manager started")
                yield

        # Route treats class instances as raw ASGI apps — no
        # request_response wrapping, no Mount path rewriting.
        starlette_app = Starlette(
            routes=[
                Route("/healthz", endpoint=handle_healthz),
                Route("/readyz", endpoint=handle_readyz),
                Route("/sse", endpoint=_SSEApp(), methods=["GET"]),
                Route("/messages/", endpoint=_MessageApp(), methods=["POST"]),
                Route(
                    "/mcp",
                    endpoint=_StreamableApp(),
                    methods=["GET", "POST", "DELETE"],
                ),
                Route("/usage", endpoint=handle_usage),
            ],
            lifespan=lifespan,
        )

        config = uvicorn.Config(starlette_app, host=host, port=port)
        server = uvicorn.Server(config)
        logger.info("MCP server starting on %s:%d (SSE transport, role=%s)", host, port, self.role)
        await server.serve()
