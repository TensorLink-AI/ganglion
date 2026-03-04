"""PipelineOrchestrator — thin sequencer for multi-stage pipelines."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ganglion.orchestration.errors import PipelineValidationError
from ganglion.orchestration.events import (
    PipelineCompleted,
    PipelineEvent,
    PipelineStarted,
    StageCompleted,
    StageRetry,
    StageSkipped,
    StageStarted,
)
from ganglion.orchestration.pipeline import PipelineDef, StageDef
from ganglion.orchestration.task_context import TaskContext
from ganglion.runtime.types import AgentResult

logger = logging.getLogger(__name__)


class PersistenceBackend(Protocol):
    """Pluggable persistence interface."""

    async def save_checkpoint(
        self,
        stage: str,
        context_snapshot: dict[str, Any],
        result: Any,
    ) -> None: ...
    async def load_checkpoint(self, stage: str) -> tuple[dict[str, Any], Any] | None: ...
    async def save_run(self, pipeline_result: Any) -> None: ...
    async def load_run_history(
        self,
        n: int = 10,
        since: datetime | None = None,
        stage_filter: str | None = None,
        success_only: bool = False,
    ) -> list[Any]: ...
    async def query_metrics(
        self,
        experiment_id: str | None = None,
        metric_name: str | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]: ...
    async def save_mutation_log(self, mutations: list[Any]) -> None: ...
    async def load_mutation_log(self) -> list[Any]: ...


@dataclass
class StageResult:
    """Result of executing a single pipeline stage."""

    success: bool = False
    result: AgentResult | None = None
    attempts: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "attempts": self.attempts,
            "error": self.error,
            "structured": self.result.structured if self.result else None,
        }


@dataclass
class PipelineResult:
    """Result of executing the full pipeline."""

    success: bool
    failed_stage: str | None = None
    reason: str | None = None
    results: dict[str, StageResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed_stage": self.failed_stage,
            "reason": self.reason,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


class PipelineOrchestrator:
    """Thin pipeline sequencer.

    Responsibilities:
    1. Validate the pipeline definition
    2. Execute stages in dependency order
    3. Pass TaskContext between stages
    4. Delegate retry decisions to the RetryPolicy
    5. Emit events for observability
    6. Call the PersistenceBackend at checkpoints
    """

    def __init__(
        self,
        pipeline: PipelineDef,
        agents: dict[str, Any],
        persistence: PersistenceBackend | None = None,
        knowledge: Any | None = None,
        event_handler: Callable[[PipelineEvent], None] | None = None,
    ):
        self.pipeline = pipeline
        self.agents = agents
        self.persistence = persistence
        self.knowledge = knowledge
        self.emit = event_handler or (lambda e: None)

    async def run(self, task: TaskContext) -> PipelineResult:
        """Execute all stages in dependency order."""
        errors = self.pipeline.validate()
        if errors:
            raise PipelineValidationError(f"Pipeline validation failed: {'; '.join(errors)}")

        self.emit(PipelineStarted(pipeline_name=self.pipeline.name))

        results: dict[str, StageResult] = {}
        execution_order = self.pipeline._topological_order()

        for stage_def in execution_order:
            self.emit(StageStarted(stage=stage_def.name))

            # Check dependencies
            failed_deps = [
                d for d in stage_def.depends_on if d in results and not results[d].success
            ]
            if failed_deps:
                if stage_def.is_optional:
                    self.emit(
                        StageSkipped(
                            stage=stage_def.name,
                            reason=f"Failed deps: {failed_deps}",
                        )
                    )
                    results[stage_def.name] = StageResult(
                        success=False,
                        error=f"Skipped: deps failed {failed_deps}",
                    )
                    continue
                else:
                    result = PipelineResult(
                        success=False,
                        failed_stage=stage_def.name,
                        reason=f"Dependencies failed: {failed_deps}",
                        results=results,
                    )
                    self.emit(
                        PipelineCompleted(
                            pipeline_name=self.pipeline.name,
                            success=False,
                        )
                    )
                    return result

            # Execute with retry
            stage_result = await self._execute_stage(stage_def, task)
            results[stage_def.name] = stage_result

            # Persist checkpoint
            if self.persistence:
                await self.persistence.save_checkpoint(
                    stage_def.name, task.snapshot(), stage_result
                )

            self.emit(StageCompleted(stage=stage_def.name, result=stage_result))

            if not stage_result.success and not stage_def.is_optional:
                result = PipelineResult(
                    success=False,
                    failed_stage=stage_def.name,
                    reason=stage_result.error,
                    results=results,
                )
                self.emit(
                    PipelineCompleted(
                        pipeline_name=self.pipeline.name,
                        success=False,
                    )
                )
                return result

        result = PipelineResult(success=True, results=results)
        self.emit(PipelineCompleted(pipeline_name=self.pipeline.name, success=True))
        return result

    async def _execute_stage(self, stage_def: StageDef, task: TaskContext) -> StageResult:
        """Run a single stage, delegating retry to the RetryPolicy."""
        agent_cls = self._resolve_agent(stage_def.agent)
        if agent_cls is None:
            return StageResult(
                success=False,
                error=f"Agent '{stage_def.agent}' not found in registry",
            )

        policy = stage_def.retry or self.pipeline.default_retry
        attempt = 0
        last_result: AgentResult | None = None

        while True:
            # Get configuration for this attempt from the policy
            if policy is not None:
                attempt_config = policy.configure_attempt(attempt, last_result)
                if attempt_config is None:
                    break
                if hasattr(attempt_config, "agent_kwargs"):
                    agent_kwargs = attempt_config.agent_kwargs.copy()
                else:
                    agent_kwargs = {}
                if hasattr(attempt_config, "temperature"):
                    agent_kwargs["temperature"] = attempt_config.temperature
                if hasattr(attempt_config, "model") and attempt_config.model:
                    agent_kwargs["model"] = attempt_config.model
                extra = getattr(attempt_config, "extra_system_context", None)
                if extra:
                    agent_kwargs["extra_system_context"] = attempt_config.extra_system_context
            else:
                # No policy — single attempt
                if attempt > 0:
                    break
                agent_kwargs = {}

            # Inject knowledge context into agent prompts
            if self.knowledge:
                knowledge_ctx = await self.knowledge.to_prompt_context(stage_def.name)
                foreign_ctx = await self.knowledge.to_foreign_prompt_context(stage_def.name)
                combined = knowledge_ctx
                if foreign_ctx:
                    combined = f"{combined}\n\n{foreign_ctx}" if combined else foreign_ctx
                if combined:
                    existing = agent_kwargs.get("extra_system_context", "")
                    if existing:
                        agent_kwargs["extra_system_context"] = f"{existing}\n\n{combined}"
                    else:
                        agent_kwargs["extra_system_context"] = combined

            try:
                agent = agent_cls(**agent_kwargs)
                result = await agent.run(task)
                last_result = result

                if result.success:
                    # Record success to knowledge store
                    await self._record_knowledge(stage_def, result, task, success=True)
                    return StageResult(success=True, result=result, attempts=attempt + 1)
            except Exception as e:
                logger.error(
                    "Stage '%s' attempt %d raised: %s",
                    stage_def.name,
                    attempt + 1,
                    e,
                    exc_info=True,
                )
                last_result = AgentResult(success=False, raw_text=str(e), turns_used=0)

            attempt += 1
            if policy is not None:
                self.emit(
                    StageRetry(
                        stage=stage_def.name,
                        attempt=attempt,
                    )
                )

        # All retries exhausted
        await self._record_knowledge(stage_def, last_result, task, success=False)
        return StageResult(
            success=False,
            result=last_result,
            attempts=attempt,
            error=last_result.raw_text if last_result else "No result",
        )

    def _resolve_agent(self, agent_ref: str) -> Any | None:
        """Resolve an agent class from the registry."""
        return self.agents.get(agent_ref)

    async def _record_knowledge(
        self,
        stage_def: StageDef,
        result: AgentResult | None,
        task: TaskContext,
        success: bool,
    ) -> None:
        """Record stage outcome to knowledge store if available."""
        if not self.knowledge or not result:
            return

        try:
            structured = result.structured
            config = structured.get("config") if isinstance(structured, dict) else None
            if success:
                metrics = task.subnet_config.metrics
                await self.knowledge.record_success(
                    capability=stage_def.name,
                    description=(result.raw_text or "")[:200],
                    config=config,
                    metric_value=self._extract_metric(result, task),
                    metric_name=metrics[0].name if metrics else None,
                    stage=stage_def.name,
                )
            else:
                await self.knowledge.record_failure(
                    capability=stage_def.name,
                    error_summary=result.raw_text or "Unknown error",
                    config=config,
                    stage=stage_def.name,
                )
        except Exception as e:
            logger.warning("Failed to record knowledge: %s", e)

    def _extract_metric(self, result: AgentResult, task: TaskContext) -> float | None:
        """Try to extract the primary metric from the agent result."""
        if not isinstance(result.structured, dict):
            return None
        metrics = result.structured.get("metrics")
        if isinstance(metrics, dict) and task.subnet_config.metrics:
            primary = task.subnet_config.metrics[0].name
            return metrics.get(primary)
        return None
