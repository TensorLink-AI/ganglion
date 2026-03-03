"""FrameworkState — the mutable container for all framework runtime state."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

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
from ganglion.orchestration.pipeline import PipelineDef
from ganglion.orchestration.task_context import SubnetConfig, TaskContext
from ganglion.knowledge.store import KnowledgeStore
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
    ):
        self.subnet_config = subnet_config
        self.pipeline_def = pipeline_def
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.persistence = persistence
        self.project_root = project_root or Path(".")
        self.knowledge = knowledge
        self.validator = validator or MutationValidator()

        # Concurrency control
        self._run_lock = asyncio.Lock()
        self._mutation_lock = asyncio.Lock()
        self._running: bool = False

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
                except Exception as e:
                    logger.warning("Failed to load tool from %s: %s", py_file, e)

        agent_registry = AgentRegistry()
        agents_dir = project_root / "agents"
        if agents_dir.is_dir():
            for py_file in sorted(agents_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    # Import and find BaseAgentWrapper subclasses
                    mod_spec = importlib.util.spec_from_file_location(
                        py_file.stem, str(py_file)
                    )
                    if mod_spec and mod_spec.loader:
                        mod = importlib.util.module_from_spec(mod_spec)
                        mod_spec.loader.exec_module(mod)
                        from ganglion.composition.base_agent import BaseAgentWrapper
                        import inspect as _inspect

                        for name, obj in _inspect.getmembers(mod, _inspect.isclass):
                            if (
                                issubclass(obj, BaseAgentWrapper)
                                and obj is not BaseAgentWrapper
                            ):
                                agent_registry.register(name, obj)
                except Exception as e:
                    logger.warning("Failed to load agent from %s: %s", py_file, e)

        return cls(
            subnet_config=subnet_config,
            pipeline_def=pipeline_def,
            tool_registry=tool_registry,
            agent_registry=agent_registry,
            persistence=persistence,
            project_root=project_root,
            knowledge=knowledge,
        )

    # ── Observation methods ─────────────────────────────────

    async def describe(self) -> dict:
        """Full snapshot of current state for observation tools."""
        return {
            "subnet": self.subnet_config.to_dict(),
            "pipeline": self.pipeline_def.to_dict(),
            "tools": self.tool_registry.list_all(),
            "agents": self.agent_registry.list_all(),
            "knowledge": (await self.knowledge.summary()) if self.knowledge else None,
            "mutations": len(self.mutations),
            "running": self._running,
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
            if not result.passed:
                return MutationResult(success=False, errors=result.errors)

            path = self.project_root / "tools" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = path.read_text() if path.exists() else None
            path.write_text(code)

            if test_code:
                test_result = self._run_test(test_code)
                if not test_result.passed:
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
        test_task: dict | None = None,
    ) -> MutationResult:
        """Write a new agent, validate, and register it."""
        async with self._mutation_lock:
            self._check_not_running("Cannot mutate agents during a pipeline run")

            result = self.validator.validate_agent(code)
            if not result.passed:
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
        operations: list[dict],
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

            # Check that referenced agents exist
            for stage in new_pipeline.stages:
                if not self.agent_registry.has(stage.agent):
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
                    return MutationResult(
                        success=False, errors=[f"Stage '{stage_name}' not found"]
                    )
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
                skip = False
                replaced = False
                for line in lines:
                    if line.strip() == section_marker:
                        skip = True
                        replaced = True
                        new_lines.append(section_marker)
                        new_lines.append(f'{prompt_section} = """{content}"""')
                        continue
                    if skip and line.startswith("# section:"):
                        skip = False
                    if not skip:
                        new_lines.append(line)
                if not replaced:
                    new_lines.append("")
                    new_lines.append(section_marker)
                    new_lines.append(f'{prompt_section} = """{content}"""')
                path.write_text("\n".join(new_lines))
            else:
                section_marker = f"# section: {prompt_section}"
                path.write_text(f"{section_marker}\n{prompt_section} = \"\"\"{content}\"\"\"")

            self.mutations.append(
                Mutation(
                    mutation_type="write_prompt",
                    target=f"{agent_name}/{prompt_section}",
                    description=f"Updated prompt section '{prompt_section}' for agent '{agent_name}'",
                    diff=content,
                    rollback_data={"path": str(path), "previous": previous},
                )
            )

            return MutationResult(success=True, path=str(path))

    # ── Execution methods ───────────────────────────────────

    async def run_pipeline(
        self, overrides: dict | None = None
    ) -> PipelineResult:
        """Execute the current pipeline. Blocks mutations during execution."""
        async with self._run_lock:
            self._running = True
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
                )
                result = await orchestrator.run(task)
                if self.persistence:
                    await self.persistence.save_run(result)
                if self.knowledge:
                    await self.knowledge.trim()
                return result
            finally:
                self._running = False

    async def run_single_stage(
        self,
        stage_name: str,
        context: dict | None = None,
    ) -> StageResult:
        """Run a single stage in isolation."""
        async with self._run_lock:
            self._running = True
            try:
                stage_def = self.pipeline_def.get_stage(stage_name)
                if not stage_def:
                    return StageResult(
                        success=False, error=f"Stage '{stage_name}' not found"
                    )

                task = TaskContext(
                    subnet_config=self.subnet_config,
                    initial=context,
                )
                orchestrator = PipelineOrchestrator(
                    pipeline=self.pipeline_def,
                    agents=self.agent_registry.as_dict(),
                    persistence=self.persistence,
                    knowledge=self.knowledge,
                )
                return await orchestrator._execute_stage(stage_def, task)
            finally:
                self._running = False

    async def run_direct_experiment(self, config: dict) -> dict:
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
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Rollback ────────────────────────────────────────────

    async def rollback_last(self) -> MutationResult:
        """Undo the most recent mutation."""
        if not self.mutations:
            return MutationResult(
                success=False, errors=["No mutations to rollback"]
            )

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
        if self._running:
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
                from ganglion.orchestration.pipeline import StageDef

                self.pipeline_def = PipelineDef(
                    name=previous["name"],
                    stages=[
                        StageDef(**{k: v for k, v in s.items() if k != "retry"})
                        for s in previous["stages"]
                    ],
                )

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
        except Exception as e:
            return MutationResult(success=False, errors=[str(e)])

    def _run_test(self, test_code: str) -> Any:
        """Run test code and return a ValidationResult-like object."""
        from ganglion.state.validator import ValidationResult

        try:
            exec(test_code, {"__builtins__": __builtins__})
            return ValidationResult(passed=True)
        except Exception as e:
            return ValidationResult(passed=False, errors=[str(e)])
