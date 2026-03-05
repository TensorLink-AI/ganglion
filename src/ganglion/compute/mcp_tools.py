"""MCP tool wrappers for the compute subsystem.

These functions can be registered into the ToolRegistry to expose
compute capabilities through the MCP server or HTTP bridge.
"""

from __future__ import annotations

import json
from typing import Any

from ganglion.state.framework_state import FrameworkState


def register_compute_tools(state: FrameworkState) -> None:
    """Register compute observation tools into the tool registry."""
    if state.job_manager is None:
        return

    async def compute_status() -> str:
        """Show available compute backends and active jobs."""
        return json.dumps(await state.compute_status())

    async def compute_jobs() -> str:
        """List active compute jobs with their status."""
        if state.job_manager is None:
            return json.dumps({"active_jobs": [], "cached_results": 0})
        return json.dumps(state.job_manager.status())

    async def compute_job_detail(job_id: str) -> str:
        """Get detailed info for a specific compute job."""
        if state.job_manager is None:
            return json.dumps({"error": "Compute not configured"})

        for handle in state.job_manager.list_active():
            if handle.job_id == job_id:
                return json.dumps(
                    {
                        "job_id": handle.job_id,
                        "backend": handle.backend_name,
                        "status": handle.status.value,
                    }
                )

        result = state.job_manager.get_result(job_id)
        if result:
            return json.dumps(
                {
                    "job_id": result.job_id,
                    "status": result.status.value,
                    "exit_code": result.exit_code,
                    "duration_seconds": result.duration_seconds,
                    "metrics": result.metrics,
                }
            )
        return json.dumps({"error": f"Job '{job_id}' not found"})

    async def compute_routes() -> str:
        """Show current stage-to-backend routing table."""
        if state.compute_router is None:
            return json.dumps({"backends": [], "routes": []})
        return json.dumps(state.compute_router.to_dict())

    _tools: list[tuple[str, Any, str, dict[str, Any]]] = [
        (
            "compute_status",
            compute_status,
            "Show available compute backends and active jobs.",
            {"type": "object", "properties": {}},
        ),
        (
            "compute_jobs",
            compute_jobs,
            "List active compute jobs with their status.",
            {"type": "object", "properties": {}},
        ),
        (
            "compute_job_detail",
            compute_job_detail,
            "Get detailed info for a specific compute job.",
            {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        ),
        (
            "compute_routes",
            compute_routes,
            "Show current stage-to-backend routing table.",
            {"type": "object", "properties": {}},
        ),
    ]

    for name, func, description, schema in _tools:
        if not state.tool_registry.has(name):
            state.tool_registry.register(
                name=name,
                func=func,
                description=description,
                parameters_schema=schema,
                category="compute",
            )
