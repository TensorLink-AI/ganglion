"""RalphLoopAgent — continuous loop agent that operates the ganglion HTTP server.

Ralph is a long-running agent that:
  1. Starts the ganglion HTTP bridge server in a background thread
  2. Loads the project's subnet config and pipeline definition
  3. Runs a continuous loop: monitor -> decide -> execute -> learn
  4. Uses the knowledge store to improve across iterations

Usage:
    ganglion ralph ./my-subnet --port 8899 --interval 300
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from typing import Any

from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.composition.tool_registry import get_finish_tool_schema
from ganglion.runtime.types import AgentResult

logger = logging.getLogger(__name__)


def _build_ralph_tool_schemas() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build OpenAI-compatible tool schemas from ralph_tools module."""
    from ganglion.agents.ralph_tools import RALPH_TOOLS

    schemas: list[dict[str, Any]] = []
    handlers: dict[str, Any] = {}

    # Type mapping for schema inference
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    for name, func in RALPH_TOOLS.items():
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or f"Tool: {name}"

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            hint = func.__annotations__.get(param_name)
            json_type = type_map.get(hint, "string") if hint else "string"
            prop: dict[str, Any] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                prop["default"] = param.default
            properties[param_name] = prop

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    **({"required": required} if required else {}),
                },
            },
        }
        schemas.append(schema)
        handlers[name] = func

    # Always include finish
    schemas.append(get_finish_tool_schema())
    handlers["finish"] = lambda **kwargs: kwargs

    return schemas, handlers


class RalphLoopAgent(BaseAgentWrapper):
    """Continuous loop agent that operates the ganglion HTTP server.

    Ralph monitors the framework state, runs the mining pipeline on a
    configurable interval, and uses the knowledge store to improve
    strategy across iterations. The HTTP bridge server runs in a
    background thread so external clients (OpenClaw, dashboards) can
    observe and interact with the framework while Ralph drives it.
    """

    def __init__(
        self,
        state: Any,
        config: Any = None,
        loop_interval: int = 300,
        max_iterations: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._state = state
        self._config = config
        self._loop_interval = loop_interval
        self._max_iterations = max_iterations
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._iteration = 0

    def build_system_prompt(self, task: Any) -> str:
        """Build the system prompt using the subnet config."""
        subnet_section = self._state.subnet_config.to_prompt_section()
        pipeline = self._state.pipeline_def.to_dict()
        stage_names = [s["name"] for s in pipeline.get("stages", [])]

        return f"""\
You are Ralph, a continuous loop agent operating a Ganglion mining node.

Your job is to manage the ganglion HTTP bridge server and continuously
run the mining pipeline to optimize subnet performance. You operate
autonomously in a loop: monitor the system state, decide what to do,
execute pipeline runs, and learn from the results.

── Subnet Configuration ──────────────────────────────────────────
{subnet_section}

── Pipeline ──────────────────────────────────────────────────────
Pipeline: {pipeline.get('name', 'unknown')}
Stages: {', '.join(stage_names)}

── Your Loop ─────────────────────────────────────────────────────
Each iteration you should:
  1. Check server health and framework status
  2. Review knowledge from previous runs (patterns & antipatterns)
  3. Decide on a strategy for this iteration:
     - Which asset to target
     - What overrides or configuration changes to try
     - Whether to run the full pipeline or specific stages
  4. Execute the pipeline with your chosen strategy
  5. Review the results and call finish with a summary

── Guidelines ────────────────────────────────────────────────────
- Always check server health first
- Review knowledge store for insights from past runs before deciding
- Vary your strategy across iterations — try different assets,
  model configurations, and parameter sweeps
- Track which approaches improve metrics and which don't
- If a pipeline run fails, diagnose the issue before retrying
- Call finish() with success=true and a summary of what you did
  and what you learned this iteration

Iteration: {self._iteration + 1}{f' / {self._max_iterations}' if self._max_iterations else ' (continuous)'}
Loop interval: {self._loop_interval}s between iterations"""

    def build_tools(self, task: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Provide tools for interacting with the framework."""
        return _build_ralph_tool_schemas()

    def build_context(self, task: Any) -> list[dict[str, Any]]:
        """Inject context about the current iteration."""
        context_parts = [
            f"Starting iteration {self._iteration + 1}.",
            f"Server running: {self._server_thread is not None and self._server_thread.is_alive()}.",
        ]
        if self._iteration > 0:
            context_parts.append(
                "This is a continuation — check knowledge store for learnings from previous runs."
            )
        return [{"role": "user", "content": " ".join(context_parts)}]

    def post_process(self, result: AgentResult, task: Any) -> AgentResult:
        """Log iteration results."""
        logger.info(
            "Ralph iteration %d completed: success=%s, turns=%d",
            self._iteration + 1,
            result.success,
            result.turns_used,
        )
        return result

    def start_server(self, host: str = "127.0.0.1", port: int = 8899) -> None:
        """Start the HTTP bridge server in a background thread."""
        from ganglion.agents.ralph_tools import _server_handle

        def _run_server() -> None:
            import uvicorn

            from ganglion.bridge.server import app, configure, setup_cors

            configure(self._state, self._config)
            setup_cors(
                self._config.cors_allowed_origins if self._config else None
            )

            server_config = uvicorn.Config(
                app, host=host, port=port, log_level="info"
            )
            server = uvicorn.Server(server_config)

            _server_handle["running"] = True
            _server_handle["host"] = host
            _server_handle["port"] = port

            logger.info("Ralph: HTTP bridge server starting on %s:%d", host, port)
            server.run()
            _server_handle["running"] = False

        self._server_thread = threading.Thread(
            target=_run_server, daemon=True, name="ralph-http-server"
        )
        self._server_thread.start()
        # Give the server a moment to bind
        time.sleep(1.0)

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._stop_event.set()

    async def run_loop(self) -> list[AgentResult]:
        """Run the continuous agent loop.

        Each iteration:
          1. Runs the agent (which uses tools to monitor + execute)
          2. Waits for loop_interval seconds
          3. Repeats until max_iterations or stop() is called
        """
        from ganglion.agents.ralph_tools import _set_state

        _set_state(self._state)

        results: list[AgentResult] = []

        while not self._stop_event.is_set():
            if self._max_iterations and self._iteration >= self._max_iterations:
                logger.info(
                    "Ralph: reached max iterations (%d), stopping",
                    self._max_iterations,
                )
                break

            logger.info(
                "Ralph: starting iteration %d", self._iteration + 1
            )

            try:
                result = await self.run(task=None)
                results.append(result)
            except Exception as e:
                logger.error(
                    "Ralph: iteration %d failed: %s",
                    self._iteration + 1,
                    e,
                    exc_info=True,
                )
                results.append(
                    AgentResult(
                        success=False,
                        raw_text=f"Iteration failed: {e}",
                        turns_used=0,
                    )
                )

            self._iteration += 1

            if self._max_iterations and self._iteration >= self._max_iterations:
                break

            logger.info(
                "Ralph: sleeping %ds before next iteration", self._loop_interval
            )
            # Sleep in small increments so stop() is responsive
            for _ in range(self._loop_interval):
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(1)

        logger.info(
            "Ralph: loop finished after %d iterations", self._iteration
        )
        return results

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "loop_interval": self._loop_interval,
                "max_iterations": self._max_iterations,
                "current_iteration": self._iteration,
                "server_running": (
                    self._server_thread is not None
                    and self._server_thread.is_alive()
                ),
            }
        )
        return base
