"""Tools for the Ralph loop agent — direct FrameworkState operations.

These tools give the Ralph agent control over the ganglion HTTP server
and pipeline lifecycle without going through HTTP. The server runs in
a background thread for external clients; Ralph drives it from inside.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Shared state reference (set by RalphLoopAgent before the loop starts) ──

_framework_state: Any = None
_server_handle: dict[str, Any] = {"running": False, "host": None, "port": None}


def _set_state(state: Any) -> None:
    global _framework_state
    _framework_state = state


def _get_state() -> Any:
    if _framework_state is None:
        raise RuntimeError("Framework state not set — call _set_state() first")
    return _framework_state


# ── Result wrapper ──────────────────────────────────────────────


@dataclass
class RalphToolResult:
    content: str
    structured: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


# ── Tools ───────────────────────────────────────────────────────


def check_server_health() -> RalphToolResult:
    """Check if the ganglion HTTP bridge server is running and ready.

    Returns the server health status including whether it's alive and
    configured, plus the host/port it's bound to.
    """
    return RalphToolResult(
        content=(
            f"Server running: {_server_handle['running']}, "
            f"host: {_server_handle['host']}, "
            f"port: {_server_handle['port']}"
        ),
        structured={
            "running": _server_handle["running"],
            "host": _server_handle["host"],
            "port": _server_handle["port"],
            "configured": _framework_state is not None,
        },
    )


def get_status() -> RalphToolResult:
    """Get the full framework state snapshot.

    Returns subnet config, pipeline definition, registered tools and agents,
    knowledge store summary, MCP connections, and compute status. Use this
    to understand the current state before making decisions.
    """
    state = _get_state()
    loop = asyncio.new_event_loop()
    try:
        desc = loop.run_until_complete(state.describe())
    finally:
        loop.close()
    return RalphToolResult(
        content=json.dumps(desc, indent=2, default=str),
        structured=desc,
    )


def get_pipeline_info() -> RalphToolResult:
    """Get the current pipeline definition.

    Returns all stages, their dependencies, input/output keys, and retry
    policies. Use this to understand what the pipeline does before running it.
    """
    state = _get_state()
    pipeline = state.pipeline_def.to_dict()
    return RalphToolResult(
        content=json.dumps(pipeline, indent=2, default=str),
        structured=pipeline,
    )


def run_pipeline(overrides: str = "") -> RalphToolResult:
    """Execute the full mining pipeline.

    Runs all stages in dependency order. Optionally pass overrides as a JSON
    string to inject initial values into the TaskContext (e.g. target_asset,
    competition type).

    Args:
        overrides: JSON string of key-value overrides for TaskContext, or empty
                   string for defaults.
    """
    state = _get_state()
    override_dict = None
    if overrides:
        try:
            override_dict = json.loads(overrides)
        except json.JSONDecodeError as e:
            return RalphToolResult(
                content=f"Invalid overrides JSON: {e}",
                structured={"success": False, "error": str(e)},
            )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(state.run_pipeline(overrides=override_dict))
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=True)
        return RalphToolResult(
            content=f"Pipeline failed: {e}",
            structured={"success": False, "error": str(e)},
        )
    finally:
        loop.close()

    result_dict = result.to_dict()
    return RalphToolResult(
        content=f"Pipeline completed: success={result_dict.get('success', False)}",
        structured=result_dict,
    )


def run_stage(stage_name: str, context: str = "") -> RalphToolResult:
    """Execute a single pipeline stage in isolation.

    Useful for testing individual stages or re-running a failed stage
    with different context.

    Args:
        stage_name: Name of the stage to run (e.g. 'plan', 'calibrate').
        context: JSON string of context values to inject, or empty string.
    """
    state = _get_state()
    ctx = None
    if context:
        try:
            ctx = json.loads(context)
        except json.JSONDecodeError as e:
            return RalphToolResult(
                content=f"Invalid context JSON: {e}",
                structured={"success": False, "error": str(e)},
            )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(state.run_single_stage(stage_name, ctx))
    except Exception as e:
        logger.error("Stage '%s' failed: %s", stage_name, e, exc_info=True)
        return RalphToolResult(
            content=f"Stage '{stage_name}' failed: {e}",
            structured={"success": False, "error": str(e)},
        )
    finally:
        loop.close()

    result_dict = result.to_dict()
    return RalphToolResult(
        content=f"Stage '{stage_name}': success={result_dict.get('success', False)}",
        structured=result_dict,
    )


def list_tools(category: str = "") -> RalphToolResult:
    """List all registered tools, optionally filtered by category.

    Args:
        category: Filter by tool category (e.g. 'data', 'training'), or
                  empty string for all tools.
    """
    state = _get_state()
    tools = state.tool_registry.list_all(category=category or None)
    return RalphToolResult(
        content=f"Registered tools ({len(tools)}): "
        + ", ".join(t["name"] for t in tools if isinstance(t, dict)),
        structured={"tools": tools, "count": len(tools)},
    )


def list_agents() -> RalphToolResult:
    """List all registered pipeline agents."""
    state = _get_state()
    agents = state.agent_registry.list_all()
    return RalphToolResult(
        content=f"Registered agents ({len(agents)}): "
        + ", ".join(a["name"] for a in agents if isinstance(a, dict)),
        structured={"agents": agents, "count": len(agents)},
    )


def get_knowledge(capability: str = "", max_entries: int = 20) -> RalphToolResult:
    """Query the knowledge store for patterns and antipatterns.

    The knowledge store tracks what model configurations and strategies
    have worked well (patterns) or poorly (antipatterns) across runs.

    Args:
        capability: Filter by capability/domain, or empty for all.
        max_entries: Maximum number of entries to return.
    """
    state = _get_state()
    if not state.knowledge:
        return RalphToolResult(
            content="No knowledge store configured",
            structured={"patterns": [], "antipatterns": [], "summary": None},
        )

    from ganglion.knowledge.types import KnowledgeQuery

    query = KnowledgeQuery(capability=capability or None, max_entries=max_entries)

    loop = asyncio.new_event_loop()
    try:

        async def _gather() -> dict[str, Any]:
            return {
                "patterns": [
                    p.__dict__
                    for p in await state.knowledge.backend.query_patterns(query)
                ],
                "antipatterns": [
                    a.__dict__
                    for a in await state.knowledge.backend.query_antipatterns(query)
                ],
                "summary": await state.knowledge.summary(),
            }

        result = loop.run_until_complete(_gather())
    finally:
        loop.close()

    pattern_count = len(result["patterns"])
    antipattern_count = len(result["antipatterns"])
    return RalphToolResult(
        content=f"Knowledge: {pattern_count} patterns, {antipattern_count} antipatterns",
        structured=result,
    )


def get_run_history(n: int = 10) -> RalphToolResult:
    """Get the history of past pipeline runs.

    Returns the most recent N pipeline runs with their results and metrics.

    Args:
        n: Number of recent runs to return (default 10).
    """
    state = _get_state()
    if state.persistence is None:
        return RalphToolResult(
            content="No persistence backend configured — no run history available",
            structured={"runs": []},
        )

    loop = asyncio.new_event_loop()
    try:
        runs = loop.run_until_complete(state.persistence.load_run_history(n=n))
    finally:
        loop.close()

    return RalphToolResult(
        content=f"Run history: {len(runs)} runs",
        structured={"runs": runs, "count": len(runs)},
    )


def get_subnet_config() -> RalphToolResult:
    """Get the full subnet configuration.

    Returns the subnet netuid, name, metrics, tasks, output spec,
    constraints, and docker prefabs. This is the source of truth for
    what the mining pipeline should optimize for.
    """
    state = _get_state()
    config = state.subnet_config.to_dict()
    prompt_section = state.subnet_config.to_prompt_section()
    return RalphToolResult(
        content=prompt_section,
        structured=config,
    )


def get_compute_status() -> RalphToolResult:
    """Get the status of compute backends and active jobs.

    Returns which compute backends are connected (RunPod, local, etc.),
    their routing rules, and any active jobs.
    """
    state = _get_state()
    loop = asyncio.new_event_loop()
    try:
        status = loop.run_until_complete(state.compute_status())
    finally:
        loop.close()

    return RalphToolResult(
        content=json.dumps(status, indent=2, default=str),
        structured=status,
    )


# ── Tool registry for Ralph ────────────────────────────────────

RALPH_TOOLS: dict[str, Any] = {
    "check_server_health": check_server_health,
    "get_status": get_status,
    "get_pipeline_info": get_pipeline_info,
    "run_pipeline": run_pipeline,
    "run_stage": run_stage,
    "list_tools": list_tools,
    "list_agents": list_agents,
    "get_knowledge": get_knowledge,
    "get_run_history": get_run_history,
    "get_subnet_config": get_subnet_config,
    "get_compute_status": get_compute_status,
}
