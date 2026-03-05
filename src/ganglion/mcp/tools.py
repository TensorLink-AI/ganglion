"""Register all HTTP bridge operations as MCP-callable tools.

This ensures MCP clients have feature parity with the HTTP bridge.
Each tool is categorized for role-based filtering:
  - observation: read-only state queries
  - mutation: write tools, agents, prompts, pipeline changes
  - execution: run pipelines, stages, experiments
  - admin: rollback, MCP server management
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ganglion.state.framework_state import FrameworkState
    from ganglion.state.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def _json_result(data: Any) -> str:
    """Serialize result to JSON string for MCP text content."""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return str(data)


def register_framework_tools(registry: ToolRegistry, state: FrameworkState) -> list[str]:
    """Register all HTTP bridge operations as MCP tools.

    Returns a list of registered tool names.
    """
    registered: list[str] = []

    def _register(
        name: str,
        func: Any,
        description: str,
        parameters: dict[str, Any],
        category: str,
    ) -> None:
        if registry.has(name):
            return
        registry.register(
            name=name,
            func=func,
            description=description,
            parameters_schema=parameters,
            category=category,
        )
        registered.append(name)

    _no_params: dict[str, Any] = {"type": "object", "properties": {}}

    # ── Observation tools ──────────────────────────────────

    async def ganglion_get_status() -> str:
        """Full framework state snapshot."""
        return _json_result(await state.describe())

    _register(
        "ganglion_get_status",
        ganglion_get_status,
        "Full framework state snapshot (subnet, pipeline, tools, agents, knowledge, MCP status)",
        _no_params,
        "observation",
    )

    async def ganglion_get_pipeline() -> str:
        """Current pipeline definition."""
        return _json_result(state.pipeline_def.to_dict())

    _register(
        "ganglion_get_pipeline",
        ganglion_get_pipeline,
        "Get the current pipeline definition (stages, agents, retry policies)",
        _no_params,
        "observation",
    )

    async def ganglion_get_tools(category: str | None = None) -> str:
        """List registered tools."""
        return _json_result(state.tool_registry.list_all(category=category))

    _register(
        "ganglion_get_tools",
        ganglion_get_tools,
        "List all registered tools, optionally filtered by category",
        {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category"},
            },
        },
        "observation",
    )

    async def ganglion_get_agents() -> str:
        """List registered agents."""
        return _json_result(state.agent_registry.list_all())

    _register(
        "ganglion_get_agents",
        ganglion_get_agents,
        "List all registered agents",
        _no_params,
        "observation",
    )

    async def ganglion_get_runs(n: int = 10) -> str:
        """Past pipeline runs."""
        if state.persistence is None:
            return _json_result([])
        return _json_result(await state.persistence.load_run_history(n=n))

    _register(
        "ganglion_get_runs",
        ganglion_get_runs,
        "Get past pipeline run history",
        {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of runs to return", "default": 10},
            },
        },
        "observation",
    )

    async def ganglion_get_metrics(experiment_id: str | None = None) -> str:
        """Experiment metrics."""
        if state.persistence is None:
            return _json_result([])
        return _json_result(await state.persistence.query_metrics(experiment_id=experiment_id))

    _register(
        "ganglion_get_metrics",
        ganglion_get_metrics,
        "Get experiment metrics, optionally filtered by experiment ID",
        {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string", "description": "Filter by experiment ID"},
            },
        },
        "observation",
    )

    async def ganglion_get_leaderboard() -> str:
        """Current Bittensor subnet leaderboard."""
        subnet_client = getattr(state, "subnet_client", None)
        if subnet_client and hasattr(subnet_client, "get_leaderboard"):
            return _json_result(await subnet_client.get_leaderboard())
        return _json_result([])

    _register(
        "ganglion_get_leaderboard",
        ganglion_get_leaderboard,
        "Get the current Bittensor subnet leaderboard",
        _no_params,
        "observation",
    )

    async def ganglion_get_knowledge(
        capability: str | None = None,
        max_entries: int = 20,
    ) -> str:
        """Knowledge store contents."""
        if not state.knowledge:
            return _json_result({"patterns": [], "antipatterns": [], "summary": None})
        from ganglion.knowledge.types import KnowledgeQuery

        query = KnowledgeQuery(capability=capability, max_entries=max_entries)
        patterns = await state.knowledge.backend.query_patterns(query)
        antipatterns = await state.knowledge.backend.query_antipatterns(query)
        return _json_result(
            {
                "patterns": [p.__dict__ for p in patterns],
                "antipatterns": [a.__dict__ for a in antipatterns],
                "summary": await state.knowledge.summary(),
            }
        )

    _register(
        "ganglion_get_knowledge",
        ganglion_get_knowledge,
        "Get knowledge store contents (patterns and antipatterns)",
        {
            "type": "object",
            "properties": {
                "capability": {"type": "string", "description": "Filter by capability"},
                "max_entries": {
                    "type": "integer",
                    "description": "Max entries to return",
                    "default": 20,
                },
            },
        },
        "observation",
    )

    async def ganglion_get_source(path: str) -> str:
        """Read source code of a file in the project."""
        if ".." in path or path.startswith("/"):
            return _json_result({"error": "Path must be relative and cannot contain '..' "})
        full_path = state.project_root / path
        try:
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(state.project_root.resolve())):
                return _json_result({"error": "Path escapes project root"})
        except (OSError, ValueError):
            return _json_result({"error": "Invalid path"})
        if not full_path.exists():
            return _json_result({"error": f"Not found: {path}"})
        try:
            content = full_path.read_text()
        except OSError as exc:
            return _json_result({"error": f"Failed to read file: {exc}"})
        return _json_result({"path": path, "content": content})

    _register(
        "ganglion_get_source",
        ganglion_get_source,
        "Read source code of any file in the project (path must be relative)",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path within the project"},
            },
            "required": ["path"],
        },
        "observation",
    )

    async def ganglion_get_components() -> str:
        """Available model components."""
        training_framework = getattr(state, "training_framework", None)
        if training_framework and hasattr(training_framework, "list_components"):
            return _json_result(training_framework.list_components())
        return _json_result([])

    _register(
        "ganglion_get_components",
        ganglion_get_components,
        "List available model components (backbone, head, loss, etc.)",
        _no_params,
        "observation",
    )

    async def ganglion_get_mcp_status() -> str:
        """MCP integration status."""
        return _json_result(state._describe_mcp())

    _register(
        "ganglion_get_mcp_status",
        ganglion_get_mcp_status,
        "Get MCP integration status (connected servers and their tools)",
        _no_params,
        "observation",
    )

    # ── Mutation tools ─────────────────────────────────────

    async def ganglion_write_tool(
        name: str,
        code: str,
        category: str = "general",
        test_code: str | None = None,
    ) -> str:
        """Write and register a new tool."""
        result = await state.write_and_register_tool(name, code, category, test_code)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True, "path": result.path})

    _register(
        "ganglion_write_tool",
        ganglion_write_tool,
        "Write and register a new tool (validated, tested, and persisted)",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tool name"},
                "code": {"type": "string", "description": "Python source code"},
                "category": {"type": "string", "description": "Tool category", "default": "general"},
                "test_code": {"type": "string", "description": "Optional test code"},
            },
            "required": ["name", "code"],
        },
        "mutation",
    )

    async def ganglion_write_agent(
        name: str,
        code: str,
        test_task: dict[str, Any] | None = None,
    ) -> str:
        """Write and register a new agent."""
        result = await state.write_and_register_agent(name, code, test_task)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True, "path": result.path})

    _register(
        "ganglion_write_agent",
        ganglion_write_agent,
        "Write and register a new agent (validated and persisted)",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Agent name"},
                "code": {"type": "string", "description": "Python source code"},
                "test_task": {"type": "object", "description": "Optional test task config"},
            },
            "required": ["name", "code"],
        },
        "mutation",
    )

    async def ganglion_write_component(
        name: str,
        code: str,
        component_type: str = "general",
    ) -> str:
        """Write a model component."""
        training_framework = getattr(state, "training_framework", None)
        if training_framework and hasattr(training_framework, "write_component"):
            result = training_framework.write_component(name, code, component_type)
            return _json_result({"success": True, "result": result})
        path = state.project_root / "components" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(code)
        except OSError as exc:
            return _json_result({"success": False, "error": f"Failed to write: {exc}"})
        return _json_result({"success": True, "path": str(path)})

    _register(
        "ganglion_write_component",
        ganglion_write_component,
        "Write a new model component (backbone, head, loss, etc.)",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Component name"},
                "code": {"type": "string", "description": "Python source code"},
                "component_type": {
                    "type": "string",
                    "description": "Component type",
                    "default": "general",
                },
            },
            "required": ["name", "code"],
        },
        "mutation",
    )

    async def ganglion_write_prompt(
        agent_name: str,
        prompt_section: str,
        content: str,
    ) -> str:
        """Write or replace a prompt section for an agent."""
        result = await state.update_prompt(agent_name, prompt_section, content)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True, "path": result.path})

    _register(
        "ganglion_write_prompt",
        ganglion_write_prompt,
        "Write or replace a prompt section for an existing agent",
        {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "Agent name"},
                "prompt_section": {"type": "string", "description": "Prompt section name"},
                "content": {"type": "string", "description": "Prompt content"},
            },
            "required": ["agent_name", "prompt_section", "content"],
        },
        "mutation",
    )

    async def ganglion_patch_pipeline(operations: list[dict[str, Any]]) -> str:
        """Apply pipeline modifications."""
        result = await state.apply_pipeline_patch(operations)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True, "pipeline": result.pipeline})

    _register(
        "ganglion_patch_pipeline",
        ganglion_patch_pipeline,
        "Apply atomic pipeline modifications (add/remove/reorder stages)",
        {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of pipeline operations",
                },
            },
            "required": ["operations"],
        },
        "mutation",
    )

    async def ganglion_swap_policy(
        stage_name: str,
        retry_policy: dict[str, Any],
    ) -> str:
        """Swap retry policy for a stage."""
        target = stage_name if stage_name != "default" else None
        result = await state.swap_policy(target, retry_policy)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True})

    _register(
        "ganglion_swap_policy",
        ganglion_swap_policy,
        "Swap the retry policy for a pipeline stage (use 'default' for pipeline default)",
        {
            "type": "object",
            "properties": {
                "stage_name": {
                    "type": "string",
                    "description": "Stage name or 'default' for pipeline default",
                },
                "retry_policy": {"type": "object", "description": "New retry policy config"},
            },
            "required": ["stage_name", "retry_policy"],
        },
        "mutation",
    )

    # ── Execution tools ────────────────────────────────────

    async def ganglion_run_pipeline(overrides: dict[str, Any] | None = None) -> str:
        """Execute full pipeline."""
        try:
            result = await state.run_pipeline(overrides=overrides)
            return _json_result(result.to_dict())
        except Exception as exc:
            logger.error("Pipeline execution failed via MCP", exc_info=True)
            return _json_result({"success": False, "error": str(exc)})

    _register(
        "ganglion_run_pipeline",
        ganglion_run_pipeline,
        "Execute the full pipeline (all stages in order)",
        {
            "type": "object",
            "properties": {
                "overrides": {"type": "object", "description": "Optional context overrides"},
            },
        },
        "execution",
    )

    async def ganglion_run_stage(
        stage_name: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a single pipeline stage."""
        try:
            result = await state.run_single_stage(stage_name, context)
            return _json_result(result.to_dict())
        except Exception as exc:
            logger.error("Stage execution failed via MCP: %s", stage_name, exc_info=True)
            return _json_result({"success": False, "error": str(exc)})

    _register(
        "ganglion_run_stage",
        ganglion_run_stage,
        "Execute a single pipeline stage in isolation",
        {
            "type": "object",
            "properties": {
                "stage_name": {"type": "string", "description": "Name of the stage to run"},
                "context": {"type": "object", "description": "Optional context for the stage"},
            },
            "required": ["stage_name"],
        },
        "execution",
    )

    async def ganglion_run_experiment(config: dict[str, Any]) -> str:
        """Run a single experiment directly."""
        try:
            result = await state.run_direct_experiment(config)
            return _json_result(result)
        except Exception as exc:
            logger.error("Experiment execution failed via MCP", exc_info=True)
            return _json_result({"success": False, "error": str(exc)})

    _register(
        "ganglion_run_experiment",
        ganglion_run_experiment,
        "Run a single experiment directly (bypasses pipeline)",
        {
            "type": "object",
            "properties": {
                "config": {"type": "object", "description": "Experiment configuration"},
            },
            "required": ["config"],
        },
        "execution",
    )

    # ── Admin tools (rollback + MCP management) ────────────

    async def ganglion_rollback_last() -> str:
        """Undo the most recent mutation."""
        result = await state.rollback_last()
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True})

    _register(
        "ganglion_rollback_last",
        ganglion_rollback_last,
        "Undo the most recent mutation (tool write, pipeline patch, etc.)",
        _no_params,
        "admin",
    )

    async def ganglion_rollback_to(index: int) -> str:
        """Undo all mutations back to the given index."""
        if index < 0:
            return _json_result({"success": False, "errors": ["Index must be >= 0"]})
        result = await state.rollback_to(index)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True})

    _register(
        "ganglion_rollback_to",
        ganglion_rollback_to,
        "Undo all mutations back to the given index",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Target mutation index (>= 0)"},
            },
            "required": ["index"],
        },
        "admin",
    )

    async def ganglion_connect_mcp(
        name: str,
        transport: str = "stdio",
        command: list[str] | None = None,
        url: str | None = None,
        tool_prefix: str = "",
        category: str = "mcp",
        timeout: float = 30.0,
    ) -> str:
        """Connect to an external MCP server."""
        from ganglion.mcp.config import MCPClientConfig

        config = MCPClientConfig(
            name=name,
            transport=transport,
            command=command,
            url=url,
            tool_prefix=tool_prefix or name,
            category=category,
            timeout=timeout,
        )
        result = await state.connect_mcp_server(config)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True, "mcp": state._describe_mcp()})

    _register(
        "ganglion_connect_mcp",
        ganglion_connect_mcp,
        "Dynamically connect to an external MCP server and register its tools",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Server name"},
                "transport": {"type": "string", "enum": ["stdio", "sse"], "default": "stdio"},
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command to start stdio server",
                },
                "url": {"type": "string", "description": "URL for SSE server"},
                "tool_prefix": {"type": "string", "description": "Prefix for tool names"},
                "category": {"type": "string", "description": "Category for registered tools"},
                "timeout": {"type": "number", "description": "Timeout in seconds", "default": 30.0},
            },
            "required": ["name"],
        },
        "admin",
    )

    async def ganglion_disconnect_mcp(name: str) -> str:
        """Disconnect from an MCP server."""
        result = await state.disconnect_mcp_server(name)
        if not result.success:
            return _json_result({"success": False, "errors": result.errors})
        return _json_result({"success": True, "mcp": state._describe_mcp()})

    _register(
        "ganglion_disconnect_mcp",
        ganglion_disconnect_mcp,
        "Disconnect from an MCP server and unregister its tools",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Server name to disconnect"},
            },
            "required": ["name"],
        },
        "admin",
    )

    async def ganglion_reconnect_mcp(name: str) -> str:
        """Reconnect to a failed MCP server."""
        if name not in state._mcp_bridges:
            return _json_result({"success": False, "errors": [f"MCP server '{name}' not connected"]})
        bridge = state._mcp_bridges[name]
        config = bridge.config
        disconnect_result = await state.disconnect_mcp_server(name)
        if not disconnect_result.success:
            return _json_result({"success": False, "errors": disconnect_result.errors})
        connect_result = await state.connect_mcp_server(config)
        if not connect_result.success:
            return _json_result({"success": False, "errors": connect_result.errors})
        return _json_result({"success": True, "mcp": state._describe_mcp()})

    _register(
        "ganglion_reconnect_mcp",
        ganglion_reconnect_mcp,
        "Reconnect to a failed MCP server (disconnect + reconnect)",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Server name to reconnect"},
            },
            "required": ["name"],
        },
        "admin",
    )

    logger.info("Registered %d framework tools for MCP", len(registered))
    return registered
