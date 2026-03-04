"""SimpleAgent — the turn-by-turn execution kernel."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from ganglion.runtime.coerce import CoercionPipeline
from ganglion.runtime.llm_client import LLMClient
from ganglion.runtime.types import AgentResult, ToolCall, ToolResult

logger = logging.getLogger(__name__)


class SimpleAgent:
    """The ~200-line for-loop kernel.

    send messages -> parse tool calls -> execute -> append results -> repeat until finish
    """

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools_schema: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[..., Any]],
        context_messages: list[dict[str, Any]] | None = None,
        max_turns: int = 50,
        temperature: float = 0.7,
        model: str | None = None,
        coercion: CoercionPipeline | None = None,
        type_hints: dict[str, dict[str, type]] | None = None,
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.tools_schema = tools_schema
        self.tool_handlers = tool_handlers
        self.max_turns = max_turns
        self.temperature = temperature
        self.model = model
        self.coercion = coercion or CoercionPipeline()
        self.type_hints = type_hints or {}

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if context_messages:
            self.messages.extend(context_messages)

    async def run(self) -> AgentResult:
        """Execute the agent loop until finish is called or max_turns is reached."""
        for turn in range(self.max_turns):
            response = await self.llm_client.chat_completion(
                messages=self.messages,
                tools=self.tools_schema if self.tools_schema else None,
                temperature=self.temperature,
                model=self.model,
            )

            assistant_message = self._build_assistant_message(response)
            self.messages.append(assistant_message)

            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                return AgentResult(
                    success=False,
                    raw_text=response.get("content", ""),
                    messages=self.messages,
                    turns_used=turn + 1,
                )

            results = []
            for tc in tool_calls:
                result = await self._execute_tool(tc)
                results.append(result)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                )

                if result.is_finish:
                    return AgentResult(
                        success=result.structured.get("success", True)
                        if isinstance(result.structured, dict)
                        else bool(result.structured),
                        structured=result.structured,
                        raw_text=response.get("content", ""),
                        messages=self.messages,
                        turns_used=turn + 1,
                    )

        logger.warning("Agent reached max turns (%d) without finishing", self.max_turns)
        return AgentResult(
            success=False,
            raw_text="Max turns reached without calling finish()",
            messages=self.messages,
            turns_used=self.max_turns,
        )

    def _build_assistant_message(self, response: dict[str, Any]) -> dict[str, Any]:
        """Build the assistant message to append to the conversation."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": response.get("content", ""),
        }
        if "tool_calls" in response:
            message["tool_calls"] = response["tool_calls"]
        return message

    def _parse_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Extract tool calls from the LLM response."""
        raw_calls = response.get("tool_calls", [])
        calls = []
        for raw in raw_calls:
            func = raw.get("function", {})
            name = func.get("name", "")
            raw_args = func.get("arguments", "{}")

            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                logger.warning("Failed to parse arguments for tool '%s': %s", name, raw_args)
                arguments = {}

            hints = self.type_hints.get(name, {})
            arguments = self.coercion.apply(arguments, hints)

            calls.append(
                ToolCall(
                    id=raw.get("id", ""),
                    name=name,
                    arguments=arguments,
                )
            )
        return calls

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call and return the result."""
        handler = self.tool_handlers.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error: Unknown tool '{tool_call.name}'",
            )

        try:
            if tool_call.name == "finish":
                return self._handle_finish(tool_call)
            result = handler(**tool_call.arguments)
            if hasattr(result, "content"):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=str(result.content),
                    structured=result.structured if hasattr(result, "structured") else None,
                )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=str(result),
            )
        except Exception as e:
            logger.error("Tool '%s' raised: %s", tool_call.name, e, exc_info=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=f"Error executing {tool_call.name}: {e}",
            )

    def _handle_finish(self, tool_call: ToolCall) -> ToolResult:
        """Handle the special finish() tool call."""
        args = tool_call.arguments
        success = args.get("success", True)
        result = args.get("result")
        summary = args.get("summary", "")

        structured = {
            "success": success,
            "result": result,
            "summary": summary,
        }

        return ToolResult(
            tool_call_id=tool_call.id,
            name="finish",
            content=summary or json.dumps(structured),
            is_finish=True,
            structured=structured,
        )
