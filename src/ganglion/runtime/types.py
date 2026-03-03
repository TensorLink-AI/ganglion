"""Core types for the runtime layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool call parsed from LLM output."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a single tool call."""

    tool_call_id: str
    name: str
    content: str
    is_finish: bool = False
    structured: Any | None = None


@dataclass
class AgentResult:
    """Final result of an agent run."""

    success: bool
    structured: Any = None
    raw_text: str = ""
    messages: list[dict] = field(default_factory=list)
    turns_used: int = 0
