"""ACPServerBridge — exposes Ganglion's registered agents as an ACP-compatible server.

Implements the core ACP REST endpoints:
  GET  /agents         — list all agents
  GET  /agents/{id}    — get agent details
  POST /runs           — invoke an agent (synchronous mode)
  GET  /runs/{id}      — get run status/result

This allows external ACP-aware agents and orchestrators to discover and
invoke Ganglion pipeline agents over a standard REST interface.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from ganglion.acp.errors import ACPNotAvailableError

if TYPE_CHECKING:
    from ganglion.state.agent_registry import AgentRegistry
    from ganglion.state.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# ACP server only needs FastAPI + uvicorn (already in core deps)
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    ACP_SERVER_AVAILABLE = True
except ImportError:
    ACP_SERVER_AVAILABLE = False


class ACPServerBridge:
    """Expose Ganglion's registered agents as an ACP-compatible REST server."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        server_name: str = "ganglion",
        token: str | None = None,
    ):
        if not ACP_SERVER_AVAILABLE:
            raise ACPNotAvailableError()

        self._agent_registry = agent_registry
        self._tool_registry = tool_registry
        self._server_name = server_name
        self._token = token

        # In-memory run store (run_id -> run result)
        self._runs: dict[str, dict[str, Any]] = {}

        self._app = FastAPI(title=f"Ganglion ACP — {server_name}")
        self._setup_routes()

    def _check_auth(self, request: Request) -> None:
        """Validate bearer token if configured."""
        if not self._token:
            return
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {self._token}":
            raise HTTPException(status_code=401, detail="Unauthorized")

    def _setup_routes(self) -> None:
        """Register ACP REST endpoints."""
        app = self._app

        @app.get("/agents")
        async def list_agents(request: Request) -> JSONResponse:
            """List all available agents (ACP agent discovery)."""
            self._check_auth(request)
            agents = []
            for agent_info in self._agent_registry.list_all():
                agents.append(
                    {
                        "id": agent_info["name"],
                        "name": agent_info["name"],
                        "description": agent_info.get(
                            "description", f"Ganglion agent: {agent_info['name']}"
                        ),
                        "metadata": {
                            "server": self._server_name,
                            "class": agent_info.get("class", ""),
                            "module": agent_info.get("module", ""),
                        },
                    }
                )
            return JSONResponse(agents)

        @app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str, request: Request) -> JSONResponse:
            """Get details of a specific agent."""
            self._check_auth(request)
            if not self._agent_registry.has(agent_id):
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

            agent_cls = self._agent_registry.get(agent_id)
            # Instantiate briefly to get description
            try:
                instance = agent_cls()
                desc = instance.describe()
            except Exception:
                desc = {"class": agent_cls.__name__, "module": agent_cls.__module__}

            return JSONResponse(
                {
                    "id": agent_id,
                    "name": agent_id,
                    "description": f"Ganglion agent: {agent_id}",
                    "metadata": {
                        "server": self._server_name,
                        **desc,
                    },
                }
            )

        @app.post("/runs")
        async def create_run(request: Request) -> JSONResponse:
            """Create and execute an agent run (synchronous ACP run)."""
            self._check_auth(request)
            body = await request.json()

            agent_id = body.get("agent_id")
            if not agent_id:
                raise HTTPException(status_code=400, detail="agent_id is required")

            if not self._agent_registry.has(agent_id):
                raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

            # Extract input text from ACP message format
            input_text = _extract_input_text(body.get("input", []))

            run_id = str(uuid.uuid4())
            start_time = time.monotonic()

            # Store as in-progress
            self._runs[run_id] = {
                "id": run_id,
                "agent_id": agent_id,
                "status": "running",
                "created_at": time.time(),
            }

            try:
                agent_cls = self._agent_registry.get(agent_id)
                # Create a lightweight task context with the input
                from ganglion.orchestration.task_context import SubnetConfig, TaskContext

                task = TaskContext(
                    subnet_config=SubnetConfig(name="acp-request", netuid=0),
                    initial={"input": input_text, "acp_run_id": run_id},
                )

                # Instantiate and run the agent
                # Note: agents that require an llm_client will need one configured.
                # For tool-based (deterministic) agents, llm_client is not needed.
                instance = agent_cls()
                result = await instance.run(task)

                elapsed_ms = (time.monotonic() - start_time) * 1000
                run_result = {
                    "id": run_id,
                    "agent_id": agent_id,
                    "status": "completed",
                    "output": [
                        {
                            "parts": [
                                {
                                    "content": (
                                result.text if hasattr(result, "text") else str(result)
                            ),
                                    "content_type": "text/plain",
                                }
                            ],
                        }
                    ],
                    "metadata": {
                        "elapsed_ms": round(elapsed_ms, 2),
                        "success": result.success if hasattr(result, "success") else True,
                    },
                }
                self._runs[run_id] = run_result
                return JSONResponse(run_result)

            except Exception as e:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                logger.error("ACP run '%s' for agent '%s' failed: %s", run_id, agent_id, e)
                run_result = {
                    "id": run_id,
                    "agent_id": agent_id,
                    "status": "failed",
                    "error": str(e),
                    "output": [
                        {
                            "parts": [
                                {
                                    "content": f"Error: {e}",
                                    "content_type": "text/plain",
                                }
                            ],
                        }
                    ],
                    "metadata": {"elapsed_ms": round(elapsed_ms, 2)},
                }
                self._runs[run_id] = run_result
                return JSONResponse(run_result, status_code=500)

        @app.get("/runs/{run_id}")
        async def get_run(run_id: str, request: Request) -> JSONResponse:
            """Get the status/result of a run."""
            self._check_auth(request)
            if run_id not in self._runs:
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
            return JSONResponse(self._runs[run_id])

        @app.get("/healthz")
        async def healthz() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        @app.get("/readyz")
        async def readyz() -> JSONResponse:
            return JSONResponse(
                {
                    "status": "ready",
                    "server": self._server_name,
                    "agents": len(self._agent_registry.list_all()),
                }
            )

    async def run(self, host: str = "127.0.0.1", port: int = 8950) -> None:
        """Start the ACP server."""
        import uvicorn

        config = uvicorn.Config(self._app, host=host, port=port)
        server = uvicorn.Server(config)
        logger.info(
            "ACP server '%s' starting on %s:%d (%d agents)",
            self._server_name,
            host,
            port,
            len(self._agent_registry.list_all()),
        )
        await server.serve()


def _extract_input_text(input_messages: list[dict[str, Any]]) -> str:
    """Extract text content from ACP input messages.

    ACP input format: [ { "parts": [ { "content": "...", "content_type": "text/plain" } ] } ]
    """
    text_parts: list[str] = []
    for message in input_messages:
        parts = message.get("parts", [])
        for part in parts:
            content = part.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            else:
                text_parts.append(json.dumps(content))
    return "\n".join(text_parts)
