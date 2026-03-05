"""ComputeRouter — declarative stage-to-backend mapping."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any

from ganglion.compute.protocol import ComputeBackend, JobSpec

logger = logging.getLogger(__name__)


@dataclass
class ComputeRoute:
    """Maps a stage (or capability) to a compute backend."""

    pattern: str
    backend: str
    overrides: dict[str, Any] = field(default_factory=dict)


class ComputeRouter:
    """Route pipeline stages to appropriate compute backends.

    Routes are evaluated in order. The first matching route wins.
    A route matches if its pattern equals the stage name, matches as a glob,
    or is "default".
    """

    def __init__(
        self,
        backends: dict[str, ComputeBackend],
        routes: list[ComputeRoute],
    ):
        self._backends = backends
        self._routes = routes

    @property
    def backends(self) -> dict[str, ComputeBackend]:
        return self._backends

    @property
    def routes(self) -> list[ComputeRoute]:
        return self._routes

    def resolve(self, stage_name: str) -> ComputeBackend:
        """Find the backend for a given stage."""
        for route in self._routes:
            if self._matches(route.pattern, stage_name):
                backend = self._backends.get(route.backend)
                if backend:
                    return backend
        # Fall back to first available backend
        if "local" in self._backends:
            return self._backends["local"]
        return next(iter(self._backends.values()))

    def resolve_with_overrides(
        self, stage_name: str, base_spec: JobSpec
    ) -> tuple[ComputeBackend, JobSpec]:
        """Resolve backend AND apply route-specific overrides."""
        for route in self._routes:
            if self._matches(route.pattern, stage_name):
                backend = self._backends.get(route.backend)
                if backend:
                    spec = self._apply_overrides(base_spec, route.overrides)
                    return backend, spec
        fallback = self._backends.get("local") or next(iter(self._backends.values()))
        return fallback, base_spec

    def add_backend(self, name: str, backend: ComputeBackend) -> None:
        """Register a new backend."""
        self._backends[name] = backend

    def remove_backend(self, name: str) -> ComputeBackend | None:
        """Remove and return a backend."""
        return self._backends.pop(name, None)

    def add_route(self, route: ComputeRoute, index: int | None = None) -> None:
        """Add a route at the given position (or end)."""
        if index is not None:
            self._routes.insert(index, route)
        else:
            self._routes.append(route)

    def set_routes(self, routes: list[ComputeRoute]) -> None:
        """Replace all routes."""
        self._routes = routes

    def _matches(self, pattern: str, stage_name: str) -> bool:
        """Check if a route pattern matches a stage name."""
        if pattern == "default":
            return True
        if pattern == stage_name:
            return True
        return fnmatch.fnmatch(stage_name, pattern)

    @staticmethod
    def _apply_overrides(spec: JobSpec, overrides: dict[str, Any]) -> JobSpec:
        """Create a new JobSpec with overrides applied."""
        if not overrides:
            return spec
        spec_dict = {
            "image": spec.image,
            "command": spec.command,
            "env": spec.env,
            "gpu_type": spec.gpu_type,
            "gpu_count": spec.gpu_count,
            "cpu_cores": spec.cpu_cores,
            "memory_gb": spec.memory_gb,
            "timeout_seconds": spec.timeout_seconds,
            "artifacts_dir": spec.artifacts_dir,
            "upload_paths": spec.upload_paths,
        }
        spec_dict.update(overrides)
        return JobSpec(**spec_dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for observation endpoints."""
        return {
            "backends": [
                {"name": b.name} for b in self._backends.values()
            ],
            "routes": [
                {"pattern": r.pattern, "backend": r.backend, "overrides": r.overrides}
                for r in self._routes
            ],
        }
