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
        """Run as SSE transport MCP server."""
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Mount, Route

        # Follow the official MCP SDK pattern (FastMCP.sse_app):
        #  - Route("/sse") for the SSE GET endpoint (no Mount, so no root_path
        #    prefix and no 307 trailing-slash redirect)
        #  - Mount("/messages/") for the POST endpoint
        #  - SseServerTransport("/messages/") so clients POST to /messages/
        sse = SseServerTransport("/messages/")

        def _check_auth(request: Request) -> Response | None:
            """Check bearer token. Returns error Response or None."""
            if not self._token:
                return None
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {self._token}":
                return Response("Unauthorized", status_code=401)
            return None

        async def _check_auth_asgi(scope: dict[str, Any]) -> Response | None:
            """Check bearer token from ASGI scope. Returns error Response or None."""
            if not self._token:
                return None
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            if auth != f"Bearer {self._token}":
                return Response("Unauthorized", status_code=401)
            return None

        async def handle_sse(scope: Any, receive: Any, send: Any) -> None:
            """Raw ASGI app for SSE connections.

            Uses raw ASGI interface instead of a Starlette endpoint to avoid
            double-response: connect_sse already sends the HTTP response via
            ``send``, so returning a Starlette Response would trigger
            'Unexpected ASGI message http.response.start sent, after response
            already completed'.
            """
            if scope["type"] != "http":
                return
            request = Request(scope, receive, send)
            auth_error = _check_auth(request)
            if auth_error:
                await auth_error(scope, receive, send)
                return
            try:
                async with sse.connect_sse(scope, receive, send) as streams:
                    await self._server.run(
                        streams[0],
                        streams[1],
                        self._server.create_initialization_options(),
                    )
            except Exception as e:
                logger.error("SSE connection error: %s", e, exc_info=True)

        async def messages_app(scope: Any, receive: Any, send: Any) -> None:
            """Raw ASGI app for message posting."""
            auth_error = await _check_auth_asgi(scope)
            if auth_error:
                await auth_error(scope, receive, send)
                return
            try:
                await sse.handle_post_message(scope, receive, send)
            except Exception as e:
                logger.error("MCP message handling error: %s", e, exc_info=True)
                try:
                    error_resp = Response(f"Internal server error: {e}", status_code=500)
                    await error_resp(scope, receive, send)
                except RuntimeError:
                    # Response was already sent before the exception —
                    # nothing more we can do on this connection.
                    logger.debug("Could not send error response (response already sent)")

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

        starlette_app = Starlette(
            routes=[
                Route("/healthz", endpoint=handle_healthz),
                Route("/readyz", endpoint=handle_readyz),
                Mount("/sse", app=handle_sse),
                Mount("/messages/", app=messages_app),
                Route("/usage", endpoint=handle_usage),
            ]
        )

        config = uvicorn.Config(starlette_app, host=host, port=port)
        server = uvicorn.Server(config)
        logger.info("MCP server starting on %s:%d (SSE transport, role=%s)", host, port, self.role)
        await server.serve()
