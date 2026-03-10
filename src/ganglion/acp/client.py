"""ACPClientBridge — discovers remote ACP agents and wraps them as Ganglion ToolDefs.

ACP agents are invoked via REST (POST /runs) and their results are returned
as ToolOutput, making them usable from any Ganglion pipeline stage that
consumes tools.

Each discovered remote agent becomes a callable tool:
  - Name: "{prefix}_{agent_name}" (or just agent_name if no prefix)
  - Calling the tool sends a message to the remote agent via POST /runs
  - The response text is returned as ToolOutput.content
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ganglion.acp.config import ACPClientConfig
from ganglion.acp.errors import ACPConnectionError, ACPNotAvailableError, ACPRunError
from ganglion.composition.tool_registry import ToolDef
from ganglion.composition.tool_returns import ToolOutput

logger = logging.getLogger(__name__)

try:
    import aiohttp

    ACP_AVAILABLE = True
except ImportError:
    ACP_AVAILABLE = False


def _check_acp_available() -> None:
    if not ACP_AVAILABLE:
        raise ACPNotAvailableError()


class ACPClientBridge:
    """Connects to an external ACP server, discovers agents, and wraps them as ToolDefs."""

    def __init__(self, config: ACPClientConfig):
        _check_acp_available()
        self.config = config
        self._session: aiohttp.ClientSession | None = None  # type: ignore[name-defined]
        self._agents: dict[str, dict[str, Any]] = {}  # agent_id -> agent card
        self._tools: dict[str, ToolDef] = {}

    async def connect(self) -> list[ToolDef]:
        """Connect to the ACP server, discover agents, and return wrapped ToolDefs."""
        errors = self.config.validate()
        if errors:
            raise ACPConnectionError(
                f"Invalid config for '{self.config.name}': {'; '.join(errors)}"
            )

        headers: dict[str, str] = dict(self.config.headers)
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        headers.setdefault("Content-Type", "application/json")

        try:
            self._session = aiohttp.ClientSession(  # type: ignore[attr-defined]
                base_url=self.config.url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),  # type: ignore[attr-defined]
            )
            await self._discover_agents()
            return list(self._tools.values())
        except ACPConnectionError:
            raise
        except Exception as e:
            await self._cleanup()
            raise ACPConnectionError(
                f"Failed to connect to ACP server '{self.config.name}': {e}"
            ) from e

    async def _discover_agents(self) -> None:
        """List agents from the ACP server and create ToolDef wrappers."""
        if self._session is None:
            raise ACPConnectionError("No active session — call connect() first")

        try:
            async with self._session.get("/agents") as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as e:
            raise ACPConnectionError(
                f"Failed to discover agents on '{self.config.name}': {e}"
            ) from e

        # ACP returns a list of agent cards (or an object with an "agents" key)
        agents_list = data if isinstance(data, list) else data.get("agents", [])
        prefix = self.config.agent_prefix or self.config.name
        skipped = 0

        for agent_card in agents_list:
            try:
                agent_name = agent_card.get("name") or agent_card.get("id", "unknown")
                agent_id = agent_card.get("id", agent_name)
                tool_name = f"{prefix}_{agent_name}" if prefix else agent_name
                description = agent_card.get("description", f"ACP agent: {agent_name}")

                self._agents[agent_id] = agent_card
                handler = self._make_handler(agent_id, agent_name)

                tool_def = ToolDef(
                    name=tool_name,
                    description=description,
                    func=handler,
                    parameters_schema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message/input to send to the agent",
                            },
                        },
                        "required": ["message"],
                    },
                    category="acp",
                )
                self._tools[tool_name] = tool_def
            except Exception as e:
                skipped += 1
                logger.warning(
                    "ACP server '%s': skipping agent '%s' due to error: %s",
                    self.config.name,
                    agent_card.get("name", "<unknown>"),
                    e,
                )

        logger.info(
            "ACP server '%s': discovered %d agents%s",
            self.config.name,
            len(self._tools),
            f" (skipped {skipped})" if skipped else "",
        )

    def _make_handler(self, agent_id: str, agent_name: str) -> Any:
        """Create an async handler closure that invokes the ACP agent via POST /runs."""
        session = self._session
        timeout = self.config.timeout
        server_name = self.config.name

        async def handler(message: str = "", **kwargs: Any) -> ToolOutput:
            if session is None or session.closed:
                raise ACPRunError(f"ACP server '{server_name}' is not connected")

            # Build ACP run request body
            # ACP spec: POST /runs with { agent_id, input: [...messages] }
            run_body: dict[str, Any] = {
                "agent_id": agent_id,
                "input": [
                    {
                        "parts": [
                            {
                                "content": message or json.dumps(kwargs),
                                "content_type": "text/plain",
                            }
                        ],
                    }
                ],
            }

            try:
                async with asyncio.timeout(timeout):
                    async with session.post("/runs", json=run_body) as resp:
                        resp.raise_for_status()
                        result = await resp.json()
            except TimeoutError as e:
                raise ACPRunError(
                    f"ACP agent '{agent_name}' on '{server_name}' timed out after {timeout}s"
                ) from e
            except ACPRunError:
                raise
            except Exception as e:
                raise ACPRunError(
                    f"ACP agent '{agent_name}' on '{server_name}' failed: {e}"
                ) from e

            # Extract text from ACP response
            # ACP run response contains output messages with parts
            content = _extract_run_output(result)

            status = result.get("status", "unknown")
            if status in ("failed", "error"):
                return ToolOutput(content=f"ACP Error: {content}")
            return ToolOutput(content=content)

        return handler

    def get_tools(self) -> dict[str, ToolDef]:
        """Return discovered tools as ToolDefs."""
        return dict(self._tools)

    def get_agents(self) -> dict[str, dict[str, Any]]:
        """Return discovered agent cards."""
        return dict(self._agents)

    async def disconnect(self) -> None:
        """Cleanly disconnect from the ACP server."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as e:
                logger.warning("Error closing ACP connection to '%s': %s", self.config.name, e)
        self._session = None
        self._tools.clear()
        self._agents.clear()


def _extract_run_output(result: dict[str, Any]) -> str:
    """Extract text content from an ACP run response.

    ACP run responses have the shape:
        { "status": "completed", "output": [ { "parts": [ { "content": "..." } ] } ] }
    """
    output = result.get("output", [])
    if not output:
        return result.get("error", str(result))

    text_parts: list[str] = []
    for message in output:
        parts = message.get("parts", [])
        for part in parts:
            content = part.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            else:
                text_parts.append(json.dumps(content))

    return "\n".join(text_parts) or str(result)
