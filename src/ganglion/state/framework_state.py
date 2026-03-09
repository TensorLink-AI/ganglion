"""FrameworkState — the mutable container for all framework runtime state."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Any

from ganglion.compute.artifacts import ArtifactMeta, ArtifactStore, LocalArtifactStore
from ganglion.compute.job_manager import JobManager
from ganglion.compute.protocol import BuildBackend, ComputeBackend
from ganglion.compute.router import ComputeRouter
from ganglion.knowledge.store import KnowledgeStore
from ganglion.orchestration.errors import (
    ConcurrentMutationError,
    PipelineOperationError,
)
from ganglion.orchestration.orchestrator import (
    PersistenceBackend,
    PipelineOrchestrator,
    PipelineResult,
    StageResult,
)
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import SubnetConfig, TaskContext
from ganglion.state.agent_registry import AgentRegistry
from ganglion.state.mutation import Mutation, MutationResult
from ganglion.state.tool_registry import ToolRegistry
from ganglion.state.validator import MutationValidator

logger = logging.getLogger(__name__)


class FrameworkState:
    """The mutable container for all framework runtime state.

    Holds the current pipeline definition, registered tools, registered
    agents, policies, and run history — and supports safe mutation of all
    of these while pipelines may be running.
    """

    def __init__(
        self,
        subnet_config: SubnetConfig,
        pipeline_def: PipelineDef,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        persistence: PersistenceBackend | None = None,
        project_root: Path | None = None,
        knowledge: KnowledgeStore | None = None,
        validator: MutationValidator | None = None,
        mcp_configs: list[Any] | None = None,
        compute_router: ComputeRouter | None = None,
        build_backend: BuildBackend | None = None,
        artifact_store: ArtifactStore | None = None,
    ):
        self.subnet_config = subnet_config
        self.pipeline_def = pipeline_def
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.persistence = persistence
        self.project_root = project_root or Path(".")
        self.knowledge = knowledge
        self.validator = validator or MutationValidator()

        # Compute
        self.compute_router = compute_router
        self._job_manager = JobManager(compute_router) if compute_router else None

        # Image building
        self.build_backend: BuildBackend | None = build_backend

        # Artifacts — task-agnostic store for run/experiment outputs
        self.artifact_store: ArtifactStore | None = artifact_store

        # MCP client bridges (name -> bridge)
        self._mcp_configs: list[Any] = mcp_configs or []
        self._mcp_bridges: dict[str, Any] = {}

        # Concurrency control
        self._run_lock = asyncio.Lock()
        self._mutation_lock = asyncio.Lock()
        self._is_running: bool = False

        # Mutation audit log
        self.mutations: list[Mutation] = []

    @classmethod
    def create(
        cls,
        subnet_config: SubnetConfig,
        pipeline_def: PipelineDef,
        project_root: Path | None = None,
        persistence: PersistenceBackend | None = None,
        knowledge: KnowledgeStore | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> FrameworkState:
        """Create a new FrameworkState with empty registries."""
        return cls(
            subnet_config=subnet_config,
            pipeline_def=pipeline_def,
            tool_registry=ToolRegistry(),
            agent_registry=AgentRegistry(),
            persistence=persistence,
            project_root=project_root,
            knowledge=knowledge,
            artifact_store=artifact_store,
        )

    @classmethod
    def load(cls, config_path: str | Path, bot_id: str | None = None) -> FrameworkState:
        """Load state from a project directory.

        Reads config.py, discovers tools in tools/, agents in agents/,
        loads pipeline definition, and initializes persistence.

        The config.py module must define:
          - subnet_config: SubnetConfig
          - pipeline: PipelineDef
        And optionally:
          - persistence: PersistenceBackend
          - knowledge: KnowledgeStore

        Args:
            config_path: Path to project directory or config file.
            bot_id: Optional bot identifier for multi-bot shared knowledge.
                    When set, knowledge entries are tagged with this id and
                    foreign knowledge from other bots can be queried.
        """
        project_root = Path(config_path)
        if project_root.is_file():
            project_root = project_root.parent

        config_file = project_root / "config.py"
        if not config_file.exists():
            raise FileNotFoundError(f"No config.py found in {project_root}")

        # Import config module
        spec = importlib.util.spec_from_file_location("_project_config", str(config_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from {config_file}")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        subnet_config = getattr(config_module, "subnet_config", None)
        if subnet_config is None:
            raise ValueError("config.py must define 'subnet_config'")

        pipeline_def = getattr(config_module, "pipeline", None)
        if pipeline_def is None:
            raise ValueError("config.py must define 'pipeline'")

        persistence = getattr(config_module, "persistence", None)
        knowledge = getattr(config_module, "knowledge", None)
        mcp_clients = getattr(config_module, "mcp_clients", None)
        compute_router = getattr(config_module, "compute_router", None)
        build_backend = getattr(config_module, "build_backend", None)
        artifact_store = getattr(config_module, "artifact_store", None)

        # Default: local artifact store under project root
        if artifact_store is None:
            artifact_store = LocalArtifactStore(root=project_root / "artifacts")

        # Set bot_id on knowledge store for multi-bot shared knowledge
        if knowledge is not None and bot_id is not None:
            knowledge.bot_id = bot_id

        # Build registries by discovering files
        tool_registry = ToolRegistry()
        tools_dir = project_root / "tools"
        if tools_dir.is_dir():
            for py_file in sorted(tools_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    tool_registry.register_from_file(py_file)
                except (ImportError, SyntaxError, OSError, AttributeError) as e:
                    logger.warning("Failed to load tool from %s: %s", py_file, e)

        agent_registry = AgentRegistry()
        agents_dir = project_root / "agents"
        if agents_dir.is_dir():
            for py_file in sorted(agents_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    # Import and find BaseAgentWrapper subclasses
                    module_spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
                    if module_spec and module_spec.loader:
                        module = importlib.util.module_from_spec(module_spec)
                        module_spec.loader.exec_module(module)
                        import inspect as _inspect

                        from ganglion.composition.base_agent import BaseAgentWrapper

                        for name, obj in _inspect.getmembers(module, _inspect.isclass):
                            if issubclass(obj, BaseAgentWrapper) and obj is not BaseAgentWrapper:
                                agent_registry.register(name, obj)
                except (ImportError, SyntaxError, OSError, AttributeError) as e:
                    logger.warning("Failed to load agent from %s: %s", py_file, e)

        return cls(
            subnet_config=subnet_config,
            pipeline_def=pipeline_def,
            tool_registry=tool_registry,
            agent_registry=agent_registry,
            persistence=persistence,
            project_root=project_root,
            knowledge=knowledge,
            mcp_configs=mcp_clients,
            compute_router=compute_router,
            build_backend=build_backend,
            artifact_store=artifact_store,
        )

    # ── Observation methods ─────────────────────────────────

    async def describe(self) -> dict[str, Any]:
        """Full snapshot of current state for observation tools."""
        desc: dict[str, Any] = {
            "subnet": self.subnet_config.to_dict(),
            "pipeline": self.pipeline_def.to_dict(),
            "tools": self.tool_registry.list_all(),
            "agents": self.agent_registry.list_all(),
            "knowledge": (await self.knowledge.summary()) if self.knowledge else None,
            "mcp": self._describe_mcp(),
            "artifacts": self.artifact_store is not None,
            "mutations": len(self.mutations),
            "running": self._is_running,
        }
        if self.compute_router:
            desc["compute"] = self._describe_compute()
        return desc

    def _describe_compute(self) -> dict[str, Any]:
        """Compute subsystem snapshot — names and routes only, never credentials."""
        if not self.compute_router:
            return {}
        result: dict[str, Any] = {
            "backends": [{"name": b.name} for b in self.compute_router.backends.values()],
            "routes": [
                {"pattern": r.pattern, "backend": r.backend} for r in self.compute_router.routes
            ],
        }
        if self._job_manager:
            result["active_jobs"] = len(self._job_manager.list_active())
        return result

    def _describe_mcp(self) -> dict[str, Any]:
        """MCP connection status snapshot."""
        servers = []
        for name, bridge in self._mcp_bridges.items():
            servers.append(
                {
                    "name": name,
                    "connected": bridge.session is not None,
                    "tools": len(bridge.get_tools()),
                }
            )
        return {
            "connected_servers": servers,
            "total_tools": sum(len(b.get_tools()) for b in self._mcp_bridges.values()),
        }

    # ── Mutation methods (all go through validation + audit) ─

    async def write_and_register_tool(
        self,
        name: str,
        code: str,
        category: str,
        test_code: str | None = None,
    ) -> MutationResult:
        """Write a new tool, validate, test, and register it."""
        async with self._mutation_lock:
            self._check_not_running("Cannot mutate tools during a pipeline run")

            result = self.validator.validate_tool(code)
            if not result.is_passed:
                return MutationResult(success=False, errors=result.errors)

            path = self.project_root / "tools" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.read_text() if path.exists() else None
            path.write_text(code)

            if test_code:
                test_result = self._run_test(test_code)
                if not test_result.is_passed:
                    if previous:
                        path.write_text(previous)
                    else:
                        path.unlink(missing_ok=True)
                    return MutationResult(
                        success=False,
                        errors=[f"Test failed: {e}" for e in test_result.errors],
                    )

            self.tool_registry.register_from_file(path)

            self.mutations.append(
                Mutation(
                    mutation_type="write_tool",
                    target=name,
                    description=f"Registered tool '{name}' in category '{category}'",
                    diff=code,
                    rollback_data={"path": str(path), "previous": previous},
                )
            )

            return MutationResult(success=True, path=str(path))

    async def write_and_register_agent(
        self,
        name: str,
        code: str,
        test_task: dict[str, Any] | None = None,
    ) -> MutationResult:
        """Write a new agent, validate, and register it."""
        async with self._mutation_lock:
            self._check_not_running("Cannot mutate agents during a pipeline run")

            result = self.validator.validate_agent(code)
            if not result.is_passed:
                return MutationResult(success=False, errors=result.errors)

            path = self.project_root / "agents" / f"{name.lower()}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.read_text() if path.exists() else None
            path.write_text(code)

            self.agent_registry.register_from_file(path, name)

            self.mutations.append(
                Mutation(
                    mutation_type="write_agent",
                    target=name,
                    description=f"Registered agent '{name}'",
                    diff=code,
                    rollback_data={"path": str(path), "previous": previous},
                )
            )

            return MutationResult(success=True, path=str(path))

    async def apply_pipeline_patch(
        self,
        operations: list[dict[str, Any]],
    ) -> MutationResult:
        """Apply atomic pipeline modifications. Validates before committing."""
        async with self._mutation_lock:
            self._check_not_running("Cannot mutate pipeline during a run")

            previous = self.pipeline_def.to_dict()
            new_pipeline = self.pipeline_def.copy()

            for op in operations:
                try:
                    new_pipeline = new_pipeline.apply_operation(op)
                except PipelineOperationError as e:
                    return MutationResult(success=False, errors=[str(e)])

            errors = new_pipeline.validate()

            # Check that referenced agents exist (only for agent stages)
            for stage in new_pipeline.stages:
                if isinstance(stage, StageDef) and not self.agent_registry.has(stage.agent):
                    errors.append(
                        f"Stage '{stage.name}' references unregistered agent '{stage.agent}'"
                    )

            if errors:
                return MutationResult(success=False, errors=errors)

            self.pipeline_def = new_pipeline

            self.mutations.append(
                Mutation(
                    mutation_type="patch_pipeline",
                    target="pipeline",
                    description=f"Applied {len(operations)} pipeline operations",
                    diff=str(operations),
                    rollback_data={"previous": previous},
                )
            )

            return MutationResult(success=True, pipeline=new_pipeline.to_dict())

    async def swap_policy(
        self,
        stage_name: str | None,
        retry_policy: Any,
    ) -> MutationResult:
        """Swap the retry policy for a stage or the pipeline default."""
        async with self._mutation_lock:
            self._check_not_running("Cannot swap policies during a run")

            if stage_name:
                stage = self.pipeline_def.get_stage(stage_name)
                if not stage:
                    return MutationResult(success=False, errors=[f"Stage '{stage_name}' not found"])
                previous_policy = stage.retry
                stage.retry = retry_policy
            else:
                previous_policy = self.pipeline_def.default_retry
                self.pipeline_def.default_retry = retry_policy

            self.mutations.append(
                Mutation(
                    mutation_type="swap_policy",
                    target=stage_name or "default",
                    description=f"Swapped retry policy for {stage_name or 'pipeline default'}",
                    rollback_data={
                        "stage": stage_name,
                        "previous": previous_policy,
                    },
                )
            )

            return MutationResult(success=True)

    async def update_prompt(
        self,
        agent_name: str,
        prompt_section: str,
        content: str,
    ) -> MutationResult:
        """Write or replace a prompt section for an existing agent."""
        async with self._mutation_lock:
            self._check_not_running("Cannot update prompts during a run")

            path = self.project_root / "prompts" / f"{agent_name.lower()}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.read_text() if path.exists() else None

            # Store as a simple key=value Python module
            if previous:
                # Append or replace section in existing file
                section_marker = f"# section: {prompt_section}"
                lines = previous.splitlines()
                new_lines = []
                should_skip = False
                is_replaced = False
                for line in lines:
                    if line.strip() == section_marker:
                        should_skip = True
                        is_replaced = True
                        new_lines.append(section_marker)
                        new_lines.append(f'{prompt_section} = """{content}"""')
                        continue
                    if should_skip and line.startswith("# section:"):
                        should_skip = False
                    if not should_skip:
                        new_lines.append(line)
                if not is_replaced:
                    new_lines.append("")
                    new_lines.append(section_marker)
                    new_lines.append(f'{prompt_section} = """{content}"""')
                path.write_text("\n".join(new_lines))
            else:
                section_marker = f"# section: {prompt_section}"
                path.write_text(f'{section_marker}\n{prompt_section} = """{content}"""')

            self.mutations.append(
                Mutation(
                    mutation_type="write_prompt",
                    target=f"{agent_name}/{prompt_section}",
                    description=(
                        f"Updated prompt section '{prompt_section}' for agent '{agent_name}'"
                    ),
                    diff=content,
                    rollback_data={"path": str(path), "previous": previous},
                )
            )

            return MutationResult(success=True, path=str(path))

    # ── Compute methods ──────────────────────────────────────

    @property
    def job_manager(self) -> JobManager | None:
        return self._job_manager

    async def hot_add_backend(self, name: str, backend: ComputeBackend) -> MutationResult:
        """Add a compute backend at runtime."""
        async with self._mutation_lock:
            self._check_not_running("Cannot add backends during a run")

            if self.compute_router is None:
                return MutationResult(
                    success=False,
                    errors=["No compute_router configured"],
                )

            self.compute_router.add_backend(name, backend)

            if self._job_manager is None:
                self._job_manager = JobManager(self.compute_router)

            self.mutations.append(
                Mutation(
                    mutation_type="add_compute_backend",
                    target=name,
                    description=f"Added compute backend '{name}'",
                )
            )
            return MutationResult(success=True)

    async def remove_backend(self, name: str) -> MutationResult:
        """Remove a compute backend at runtime."""
        async with self._mutation_lock:
            self._check_not_running("Cannot remove backends during a run")

            if self.compute_router is None:
                return MutationResult(
                    success=False,
                    errors=["No compute_router configured"],
                )

            removed = self.compute_router.remove_backend(name)
            if removed is None:
                return MutationResult(
                    success=False,
                    errors=[f"Backend '{name}' not found"],
                )

            self.mutations.append(
                Mutation(
                    mutation_type="remove_compute_backend",
                    target=name,
                    description=f"Removed compute backend '{name}'",
                )
            )
            return MutationResult(success=True)

    async def compute_status(self) -> dict[str, Any]:
        """Return compute subsystem status for observation endpoints."""
        result = self._describe_compute()
        if self._job_manager:
            result["jobs"] = self._job_manager.status()
        return result

    # ── Artifact methods ────────────────────────────────────

    async def store_artifact(
        self,
        key: str,
        data: bytes,
        run_id: str = "",
        experiment_id: str = "",
        stage: str = "",
        content_type: str = "",
        source_bot: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Store an artifact with metadata. Key should be ``{run_id}/{filename}``."""
        if self.artifact_store is None:
            logger.warning("No artifact store configured — artifact '%s' dropped", key)
            return
        # Auto-fill source_bot from knowledge store if not provided
        if source_bot is None and self.knowledge and hasattr(self.knowledge, "bot_id"):
            source_bot = self.knowledge.bot_id
        meta = ArtifactMeta(
            key=key,
            run_id=run_id,
            experiment_id=experiment_id,
            stage=stage,
            content_type=content_type,
            source_bot=source_bot,
            labels=labels or {},
        )
        await self.artifact_store.put(key, data, meta)

    async def store_job_artifacts(
        self,
        job_result: Any,
        run_id: str,
        stage: str = "",
    ) -> list[str]:
        """Persist all artifacts from a JobResult into the artifact store.

        Returns list of stored artifact keys.
        """
        if self.artifact_store is None or not hasattr(job_result, "artifacts"):
            return []
        stored: list[str] = []
        for filename, data in job_result.artifacts.items():
            key = f"{run_id}/{filename}"
            await self.store_artifact(
                key=key,
                data=data,
                run_id=run_id,
                stage=stage,
            )
            stored.append(key)
        return stored

    # ── MCP methods ─────────────────────────────────────────

    async def initialize_mcp(self) -> None:
        """Connect to all statically-configured MCP servers from config.py."""
        for config in self._mcp_configs:
            try:
                await self._connect_mcp_bridge(config)
            except Exception as e:
                logger.error("Failed to connect MCP server '%s': %s", config.name, e)

    async def shutdown_mcp(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._mcp_bridges):
            try:
                await self._disconnect_mcp_bridge(name)
            except Exception as e:
                logger.warning("Error disconnecting MCP server '%s': %s", name, e)

    async def connect_mcp_server(self, config: Any) -> MutationResult:
        """Dynamically connect to a new MCP server and register its tools."""
        async with self._mutation_lock:
            self._check_not_running("Cannot add MCP connections during a pipeline run")

            if config.name in self._mcp_bridges:
                return MutationResult(
                    success=False,
                    errors=[f"MCP server '{config.name}' already connected"],
                )

            errors = config.validate()
            if errors:
                return MutationResult(success=False, errors=errors)

            try:
                tool_names = await self._connect_mcp_bridge(config)
            except Exception as e:
                return MutationResult(success=False, errors=[f"Connection failed: {e}"])

            self.mutations.append(
                Mutation(
                    mutation_type="connect_mcp",
                    target=config.name,
                    description=(
                        f"Connected MCP server '{config.name}', registered {len(tool_names)} tools"
                    ),
                    rollback_data={"config": config.to_dict(), "tools": tool_names},
                )
            )

            return MutationResult(success=True)

    async def disconnect_mcp_server(self, name: str) -> MutationResult:
        """Disconnect from an MCP server and unregister its tools."""
        async with self._mutation_lock:
            self._check_not_running("Cannot remove MCP connections during a pipeline run")

            if name not in self._mcp_bridges:
                return MutationResult(success=False, errors=[f"MCP server '{name}' not connected"])

            bridge = self._mcp_bridges[name]
            tool_names = list(bridge.get_tools().keys())

            try:
                await self._disconnect_mcp_bridge(name)
            except Exception as e:
                return MutationResult(success=False, errors=[f"Disconnect failed: {e}"])

            self.mutations.append(
                Mutation(
                    mutation_type="disconnect_mcp",
                    target=name,
                    description=(
                        f"Disconnected MCP server '{name}', unregistered {len(tool_names)} tools"
                    ),
                    rollback_data={"tools": tool_names},
                )
            )

            return MutationResult(success=True)

    async def _connect_mcp_bridge(self, config: Any) -> list[str]:
        """Connect to a single MCP server and register its tools."""
        from ganglion.mcp.client import MCPClientBridge

        bridge = MCPClientBridge(config)
        tool_defs = await bridge.connect()

        tool_names = []
        for td in tool_defs:
            if not self.tool_registry.has(td.name):
                self.tool_registry.register(
                    name=td.name,
                    func=td.func,
                    description=td.description,
                    parameters_schema=td.parameters_schema,
                    category=td.category,
                )
                tool_names.append(td.name)

        self._mcp_bridges[config.name] = bridge
        logger.info(
            "MCP server '%s': connected, registered %d tools",
            config.name,
            len(tool_names),
        )
        return tool_names

    async def _disconnect_mcp_bridge(self, name: str) -> None:
        """Disconnect from a single MCP server and unregister its tools."""
        bridge = self._mcp_bridges.pop(name, None)
        if bridge is None:
            return

        for tool_name in bridge.get_tools():
            if self.tool_registry.has(tool_name):
                self.tool_registry.unregister(tool_name)

        await bridge.disconnect()

    # ── Execution methods ───────────────────────────────────

    async def run_pipeline(self, overrides: dict[str, Any] | None = None) -> PipelineResult:
        """Execute the current pipeline. Blocks mutations during execution."""
        async with self._run_lock:
            self._is_running = True
            try:
                task = TaskContext(
                    subnet_config=self.subnet_config,
                    initial=overrides,
                )
                orchestrator = PipelineOrchestrator(
                    pipeline=self.pipeline_def,
                    agents=self.agent_registry.as_dict(),
                    persistence=self.persistence,
                    knowledge=self.knowledge,
                    artifact_store=self.artifact_store,
                )
                result = await orchestrator.run(task)
                if self.persistence:
                    await self.persistence.save_run(result)
                if self.knowledge:
                    await self.knowledge.trim()
                return result
            finally:
                self._is_running = False

    async def run_single_stage(
        self,
        stage_name: str,
        context: dict[str, Any] | None = None,
    ) -> StageResult:
        """Run a single stage in isolation."""
        async with self._run_lock:
            self._is_running = True
            try:
                stage_def = self.pipeline_def.get_stage(stage_name)
                if not stage_def:
                    return StageResult(success=False, error=f"Stage '{stage_name}' not found")

                task = TaskContext(
                    subnet_config=self.subnet_config,
                    initial=context,
                )
                orchestrator = PipelineOrchestrator(
                    pipeline=self.pipeline_def,
                    agents=self.agent_registry.as_dict(),
                    persistence=self.persistence,
                    knowledge=self.knowledge,
                    artifact_store=self.artifact_store,
                )
                return await orchestrator._execute_stage(stage_def, task)
            finally:
                self._is_running = False

    async def run_direct_experiment(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run a single experiment directly, bypassing the pipeline.

        This is a thin passthrough — the actual experiment logic lives in
        subnet-specific @tool functions. This method looks for a registered
        tool named 'run_experiment' and calls it with the provided config.
        """
        tool_def = self.tool_registry.get("run_experiment")
        if tool_def is None:
            return {"success": False, "error": "No 'run_experiment' tool registered"}
        try:
            result = tool_def.func(**config)
            if hasattr(result, "content"):
                return {
                    "success": True,
                    "content": result.content,
                    "structured": result.structured if hasattr(result, "structured") else None,
                    "metrics": result.metrics if hasattr(result, "metrics") else None,
                }
            return {"success": True, "content": str(result)}
        except (TypeError, ValueError, KeyError) as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("Unexpected error in run_direct_experiment: %s", e, exc_info=True)
            return {"success": False, "error": str(e)}

    # ── Rollback ────────────────────────────────────────────

    async def rollback_last(self) -> MutationResult:
        """Undo the most recent mutation."""
        if not self.mutations:
            return MutationResult(success=False, errors=["No mutations to rollback"])

        async with self._mutation_lock:
            self._check_not_running("Cannot rollback during a run")
            mutation = self.mutations.pop()
            return await self._apply_rollback(mutation)

    async def rollback_to(self, index: int) -> MutationResult:
        """Undo all mutations back to the given index."""
        async with self._mutation_lock:
            self._check_not_running("Cannot rollback during a run")
            while len(self.mutations) > index:
                mutation = self.mutations.pop()
                result = await self._apply_rollback(mutation)
                if not result.success:
                    return result
            return MutationResult(success=True)

    # ── Internal ────────────────────────────────────────────

    def _check_not_running(self, message: str) -> None:
        if self._is_running:
            raise ConcurrentMutationError(message)

    async def _apply_rollback(self, mutation: Mutation) -> MutationResult:
        """Apply a rollback for a single mutation."""
        try:
            if mutation.mutation_type in ("write_tool", "write_agent"):
                path = Path(mutation.rollback_data["path"])
                previous = mutation.rollback_data.get("previous")
                if previous:
                    path.write_text(previous)
                else:
                    path.unlink(missing_ok=True)

                if mutation.mutation_type == "write_tool":
                    if self.tool_registry.has(mutation.target):
                        self.tool_registry.unregister(mutation.target)
                    if previous:
                        self.tool_registry.register_from_file(path)
                else:
                    if self.agent_registry.has(mutation.target):
                        self.agent_registry.unregister(mutation.target)

            elif mutation.mutation_type == "patch_pipeline":
                previous = mutation.rollback_data["previous"]
                self.pipeline_def = PipelineDef(
                    name=previous["name"],
                    stages=[
                        StageDef(**{k: v for k, v in s.items() if k != "retry"})
                        for s in previous["stages"]
                    ],
                )

            elif mutation.mutation_type in ("connect_mcp", "disconnect_mcp"):
                # MCP rollbacks: connect_mcp -> disconnect, disconnect_mcp -> no-op
                # (reconnecting would require the original config which may not be available)
                if mutation.mutation_type == "connect_mcp":
                    name = mutation.target
                    if name in self._mcp_bridges:
                        await self._disconnect_mcp_bridge(name)
                # disconnect_mcp rollback is a no-op — we can't reconnect without config

            elif mutation.mutation_type == "swap_policy":
                stage_name = mutation.rollback_data.get("stage")
                previous_policy = mutation.rollback_data.get("previous")
                if stage_name:
                    stage = self.pipeline_def.get_stage(stage_name)
                    if stage:
                        stage.retry = previous_policy
                else:
                    self.pipeline_def.default_retry = previous_policy

            return MutationResult(success=True)
        except (OSError, KeyError, TypeError, ValueError) as e:
            logger.error("Rollback failed for %s: %s", mutation.mutation_type, e)
            return MutationResult(success=False, errors=[str(e)])

    def _run_test(self, test_code: str) -> Any:
        """Run test code and return a ValidationResult-like object.

        Warning: executes arbitrary code. The MutationValidator should be
        called first to screen for blocked imports.
        """
        from ganglion.state.validator import ValidationResult

        try:
            exec(test_code, {"__builtins__": __builtins__})  # noqa: S102
            return ValidationResult(is_passed=True)
        except (AssertionError, TypeError, ValueError, KeyError, AttributeError) as e:
            return ValidationResult(is_passed=False, errors=[f"Test failed: {e}"])
        except Exception as e:
            logger.error("Unexpected error running test code: %s", e)
            return ValidationResult(is_passed=False, errors=[f"Test error: {e}"])
