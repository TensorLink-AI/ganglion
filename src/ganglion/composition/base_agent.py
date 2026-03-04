"""BaseAgentWrapper — the 4-hook agent contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ganglion.runtime.agent import SimpleAgent
from ganglion.runtime.coerce import CoercionPipeline
from ganglion.runtime.llm_client import LLMClient
from ganglion.runtime.types import AgentResult


class TaskContext:
    """Forward reference — the real TaskContext is in orchestration.task_context.

    This import is deferred to avoid circular imports. At runtime,
    BaseAgentWrapper.run() receives the real TaskContext from the orchestrator.
    """

    pass


class BaseAgentWrapper:
    """The 4-hook contract for building pipeline agents.

    Subclass this and implement:
      - build_system_prompt(task) -> str
      - build_tools(task) -> tuple[list[dict], dict[str, Callable]]
      - build_context(task) -> list[dict]   (optional)
      - post_process(result, task) -> AgentResult  (optional)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_turns: int = 50,
        temperature: float = 0.7,
        model: str | None = None,
        coercion: CoercionPipeline | None = None,
        extra_system_context: str | None = None,
        **kwargs: Any,
    ):
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.temperature = temperature
        self.model = model
        self.coercion = coercion
        self.extra_system_context = extra_system_context

    def build_system_prompt(self, task: Any) -> str:
        """Return the system prompt for this agent. Must be overridden."""
        raise NotImplementedError("Subclasses must implement build_system_prompt()")

    def build_tools(self, task: Any) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
        """Return (tool_schemas, tool_handlers) for this agent. Must be overridden."""
        raise NotImplementedError("Subclasses must implement build_tools()")

    def build_context(self, task: Any) -> list[dict[str, Any]]:
        """Return additional context messages to inject. Override if needed."""
        return []

    def post_process(self, result: AgentResult, task: Any) -> AgentResult:
        """Post-process the agent result. Override to extract data into TaskContext."""
        return result

    async def run(self, task: Any) -> AgentResult:
        """Wire the 4 hooks into a SimpleAgent and execute."""
        if self.llm_client is None:
            raise RuntimeError("llm_client must be set before running an agent")

        system_prompt = self.build_system_prompt(task)
        if self.extra_system_context:
            system_prompt += f"\n\n{self.extra_system_context}"

        tools_schema, tool_handlers = self.build_tools(task)
        context_messages = self.build_context(task)

        agent = SimpleAgent(
            llm_client=self.llm_client,
            system_prompt=system_prompt,
            tools_schema=tools_schema,
            tool_handlers=tool_handlers,
            context_messages=context_messages,
            max_turns=self.max_turns,
            temperature=self.temperature,
            model=self.model,
            coercion=self.coercion,
        )

        result = await agent.run()
        result = self.post_process(result, task)
        return result

    def describe(self) -> dict[str, Any]:
        """Return a description of this agent for observation tools."""
        return {
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "max_turns": self.max_turns,
            "temperature": self.temperature,
            "model": self.model,
        }
