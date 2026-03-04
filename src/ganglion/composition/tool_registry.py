"""Tool registration, schema inference, and scoped toolset building."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)

# Python type -> JSON schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolMetadata:
    """Metadata attached to a tool function by the @tool decorator."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    category: str = "general"


@dataclass
class ToolDef:
    """A registered tool definition."""

    name: str
    description: str
    func: Callable
    parameters_schema: dict[str, Any]
    category: str = "general"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "category": self.category,
        }

    def to_openai_schema(self) -> dict:
        """Format as an OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# Global registry for statically decorated tools
_global_tools: dict[str, ToolDef] = {}


def _infer_schema(func: Callable) -> dict[str, Any]:
    """Infer JSON schema from function type hints."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        hint = hints.get(param_name)
        property_schema: dict[str, Any] = {}

        if hint is not None:
            origin = getattr(hint, "__origin__", None)
            if origin is not None:
                json_type = _TYPE_MAP.get(origin, "string")
            else:
                json_type = _TYPE_MAP.get(hint, "string")
            property_schema["type"] = json_type
        else:
            property_schema["type"] = "string"

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            property_schema["default"] = param.default

        properties[param_name] = property_schema

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def tool(name: str, category: str = "general") -> Callable:
    """Decorator to register a function as a tool.

    Usage:
        @tool("run_experiment")
        def run_experiment(config: dict, epochs: int = 10) -> ExperimentResult:
            \"\"\"Run a training experiment.\"\"\"
            ...
    """

    def decorator(func: Callable) -> Callable:
        schema = _infer_schema(func)
        description = inspect.getdoc(func) or f"Tool: {name}"

        metadata = ToolMetadata(
            name=name,
            description=description,
            parameters_schema=schema,
            category=category,
        )
        func._tool_metadata = metadata  # type: ignore[attr-defined]

        tool_def = ToolDef(
            name=name,
            description=description,
            func=func,
            parameters_schema=schema,
            category=category,
        )
        _global_tools[name] = tool_def

        return func

    return decorator


def get_finish_tool_schema() -> dict:
    """Return the schema for the universal finish() tool."""
    return {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Call this when you are done. Pass success=true/false, "
                "an optional result object, and a human-readable summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether the task succeeded."},
                    "result": {"type": "object", "description": "Structured result data."},
                    "summary": {"type": "string", "description": "Human-readable summary."},
                },
                "required": ["success"],
            },
        },
    }


def build_toolset(*names: str) -> tuple[list[dict], dict[str, Callable]]:
    """Build a scoped subset of tools.

    Returns (schemas_for_llm, handler_dict).
    Always includes the finish tool.
    """
    schemas: list[dict] = []
    handlers: dict[str, Callable] = {}

    for name in names:
        if name == "finish":
            continue
        tool_def = _global_tools.get(name)
        if tool_def is None:
            logger.warning("Tool '%s' not found in global registry", name)
            continue
        schemas.append(tool_def.to_openai_schema())
        handlers[tool_def.name] = tool_def.func

    schemas.append(get_finish_tool_schema())
    handlers["finish"] = lambda **kwargs: kwargs

    return schemas, handlers


def get_all_tools() -> dict[str, ToolDef]:
    """Return all globally registered tools."""
    return dict(_global_tools)


def clear_global_registry() -> None:
    """Clear the global tool registry (useful for testing)."""
    _global_tools.clear()
