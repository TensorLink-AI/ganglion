"""ToolRegistry with runtime add/remove support."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ganglion.composition.tool_registry import ToolDef
from ganglion.orchestration.errors import ToolAlreadyRegisteredError, ToolNotFoundError

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Runtime tool registry supporting both static and dynamic registration."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters_schema: dict,
        category: str = "general",
    ) -> None:
        """Register a tool at runtime."""
        if name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool '{name}' already registered. Unregister first."
            )
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            func=func,
            parameters_schema=parameters_schema,
            category=category,
        )

    def register_from_file(self, path: Path) -> list[str]:
        """Import a Python file and register all @tool-decorated functions found in it."""
        module = self._import_module_from_path(path)
        registered: list[str] = []
        for attr_name in dir(module):
            func = getattr(module, attr_name)
            if callable(func) and hasattr(func, "_tool_metadata"):
                meta = func._tool_metadata
                # Unregister if exists (update scenario)
                if meta.name in self._tools:
                    del self._tools[meta.name]
                self._tools[meta.name] = ToolDef(
                    name=meta.name,
                    description=meta.description,
                    func=func,
                    parameters_schema=meta.parameters_schema,
                    category=meta.category,
                )
                registered.append(meta.name)
        return registered

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        del self._tools[name]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get(self, name: str) -> ToolDef | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_all(self, category: str | None = None) -> list[dict]:
        """List all registered tools."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return [t.to_dict() for t in tools]

    def build_toolset(self, *names: str) -> tuple[list[dict], dict[str, Callable]]:
        """Build a scoped subset of tools for an agent."""
        from ganglion.composition.tool_registry import get_finish_tool_schema

        schemas: list[dict] = []
        handlers: dict[str, Callable] = {}

        for name in names:
            if name == "finish":
                continue
            tool_def = self._tools.get(name)
            if tool_def is None:
                logger.warning("Tool '%s' not found in registry", name)
                continue
            schemas.append(tool_def.to_openai_schema())
            handlers[tool_def.name] = tool_def.func

        schemas.append(get_finish_tool_schema())
        handlers["finish"] = lambda **kwargs: kwargs

        return schemas, handlers

    def _import_module_from_path(self, path: Path) -> Any:
        """Dynamically import a Python module from a file path."""
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
