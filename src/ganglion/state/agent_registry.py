"""AgentRegistry — dynamic agent loading and registration."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

from ganglion.composition.base_agent import BaseAgentWrapper
from ganglion.orchestration.errors import AgentNotFoundError, AgentValidationError

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Registry for pipeline agent classes."""

    def __init__(self) -> None:
        self._agents: dict[str, type[BaseAgentWrapper]] = {}

    def register(self, name: str, cls: type[BaseAgentWrapper]) -> None:
        """Register an agent class by name."""
        self._agents[name] = cls

    def register_from_file(self, path: Path, class_name: str) -> None:
        """Import a Python file and register a specific agent class from it."""
        module = self._import_module_from_path(path)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AgentNotFoundError(f"Class '{class_name}' not found in {path}")
        if not (inspect.isclass(cls) and issubclass(cls, BaseAgentWrapper)):
            raise AgentValidationError(f"'{class_name}' does not subclass BaseAgentWrapper")
        self.register(class_name, cls)

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        del self._agents[name]

    def has(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    def get(self, name: str) -> type[BaseAgentWrapper] | None:
        """Get an agent class by name."""
        return self._agents.get(name)

    def as_dict(self) -> dict[str, type[BaseAgentWrapper]]:
        """Return all agents as a dict."""
        return dict(self._agents)

    def list_all(self) -> list[dict]:
        """List all registered agents with metadata."""
        return [{"name": name, "module": cls.__module__} for name, cls in self._agents.items()]

    def _import_module_from_path(self, path: Path) -> Any:
        """Dynamically import a Python module from a file path."""
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
