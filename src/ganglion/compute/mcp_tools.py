"""MCP tool wrappers for the compute subsystem.

These functions can be registered into the ToolRegistry to expose
compute capabilities through the MCP server or HTTP bridge.
"""

from __future__ import annotations

import json
from typing import Any

from ganglion.state.framework_state import FrameworkState


def _render_dockerfile(
    base_image: str,
    dependencies: list[str],
    entrypoint: str,
    workdir: str = "/app",
    env: dict[str, str] | None = None,
    copy_files: list[str] | None = None,
) -> str:
    """Render a Dockerfile from structured parameters."""
    lines = [f"FROM {base_image}", ""]

    if env:
        for key, value in env.items():
            lines.append(f"ENV {key}={value}")
        lines.append("")

    lines.append(f"WORKDIR {workdir}")
    lines.append("")

    if dependencies:
        deps = " ".join(dependencies)
        lines.append(f"RUN pip install --no-cache-dir {deps}")
        lines.append("")

    if copy_files:
        for src in copy_files:
            lines.append(f"COPY {src} {workdir}/")
        lines.append("")

    parts = ['"' + p + '"' for p in entrypoint.split()]
    lines.append("ENTRYPOINT [" + ", ".join(parts) + "]")

    return "\n".join(lines) + "\n"


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

    async def write_dockerfile(
        base_image: str,
        dependencies: str,
        entrypoint: str,
        tag: str,
    ) -> str:
        """Generate a Dockerfile for a training job.

        The bot declares what it needs; infrastructure validates and builds.
        Returns the generated Dockerfile content and validation result.
        """
        deps = [d.strip() for d in dependencies.split(",") if d.strip()]
        dockerfile = _render_dockerfile(base_image, deps, entrypoint)

        result: dict[str, Any] = {
            "dockerfile": dockerfile,
            "tag": tag,
        }

        # If a build backend is available, validate immediately
        if state.build_backend is not None:
            errors = await state.build_backend.validate(dockerfile)
            result["validation_errors"] = errors
            result["valid"] = len(errors) == 0
        else:
            result["valid"] = True
            result["validation_errors"] = []

        return json.dumps(result)

    async def build_image(dockerfile: str, tag: str) -> str:
        """Build and push a container image from a Dockerfile.

        The bot provides the Dockerfile text and a tag. Infrastructure
        validates, builds, and pushes. Credentials are server-side only.
        Returns the image reference on success.
        """
        if state.build_backend is None:
            return json.dumps({"success": False, "error": "No build backend configured"})

        build_result = await state.build_backend.build_and_push(dockerfile, tag)
        return json.dumps({
            "success": build_result.success,
            "image_ref": build_result.image_ref,
            "error": build_result.error,
            "duration_seconds": build_result.duration_seconds,
        })

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
        (
            "write_dockerfile",
            write_dockerfile,
            "Generate a Dockerfile for a training job. Bot declares base image, "
            "dependencies (comma-separated), entrypoint, and tag. Infrastructure "
            "validates against allowed base images.",
            {
                "type": "object",
                "properties": {
                    "base_image": {
                        "type": "string",
                        "description": "Base Docker image (e.g. 'nvidia/pytorch:24.01-devel')",
                    },
                    "dependencies": {
                        "type": "string",
                        "description": "Comma-separated pip packages "
                        "(e.g. 'torch,transformers,wandb')",
                    },
                    "entrypoint": {
                        "type": "string",
                        "description": "Command to run (e.g. 'python train.py')",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Image tag (e.g. 'training-exp-42')",
                    },
                },
                "required": ["base_image", "dependencies", "entrypoint", "tag"],
            },
        ),
        (
            "build_image",
            build_image,
            "Build and push a container image from a Dockerfile. Bot provides "
            "Dockerfile text and tag. Infrastructure validates, builds, and pushes. "
            "Credentials are server-side only.",
            {
                "type": "object",
                "properties": {
                    "dockerfile": {
                        "type": "string",
                        "description": "Full Dockerfile content",
                    },
                    "tag": {
                        "type": "string",
                        "description": "Image tag (e.g. 'training-exp-42')",
                    },
                },
                "required": ["dockerfile", "tag"],
            },
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
