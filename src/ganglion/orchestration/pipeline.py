"""PipelineDef and StageDef — declarative pipeline definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ganglion.orchestration.errors import PipelineOperationError


@dataclass
class StageDef:
    """Definition of a single pipeline stage."""

    name: str
    agent: str
    retry: Any | None = None
    is_optional: bool = False
    depends_on: list[str] = field(default_factory=list)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "agent": self.agent,
            "retry": str(self.retry) if self.retry else None,
            "is_optional": self.is_optional,
            "depends_on": self.depends_on,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
        }


@dataclass
class PipelineDef:
    """Declarative definition of a multi-stage pipeline."""

    name: str
    stages: list[StageDef]
    default_retry: Any | None = None

    def validate(self) -> list[str]:
        """Returns list of validation errors (empty = valid).

        Checks:
        - No duplicate stage names
        - Dependency graph is a DAG
        - All depends_on references exist
        - All input_keys are produced by an upstream stage's output_keys
        """
        errors: list[str] = []
        stage_names = [stage.name for stage in self.stages]

        # Check for duplicates
        seen: set[str] = set()
        for name in stage_names:
            if name in seen:
                errors.append(f"Duplicate stage name: '{name}'")
            seen.add(name)

        # Check all depends_on references exist
        name_set = set(stage_names)
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in name_set:
                    errors.append(f"Stage '{stage.name}' depends on '{dep}' which does not exist")

        # Check for cycles (topological sort)
        if not errors and self._has_cycle():
            errors.append("Dependency graph contains a cycle")

        # Check input_keys are produced by upstream output_keys
        if not errors:
            produced: dict[str, str] = {}  # key -> stage that produces it
            for stage in self._topological_order():
                for key in stage.input_keys:
                    if key not in produced:
                        errors.append(
                            f"Stage '{stage.name}' requires input_key '{key}' "
                            f"but no upstream stage produces it in output_keys"
                        )
                for key in stage.output_keys:
                    produced[key] = stage.name

        return errors

    def copy(self) -> PipelineDef:
        """Deep copy for safe mutation."""
        return PipelineDef(
            name=self.name,
            stages=[
                StageDef(
                    name=stage.name,
                    agent=stage.agent,
                    retry=stage.retry,
                    is_optional=stage.is_optional,
                    depends_on=list(stage.depends_on),
                    input_keys=list(stage.input_keys),
                    output_keys=list(stage.output_keys),
                )
                for stage in self.stages
            ],
            default_retry=self.default_retry,
        )

    def apply_operation(self, op: dict[str, Any]) -> PipelineDef:
        """Apply a single pipeline mutation. Returns new PipelineDef.

        Operations:
        - {"op": "add_stage", "stage": {...}}
        - {"op": "remove_stage", "stage_name": "..."}
        - {"op": "update_stage", "stage_name": "...", "updates": {...}}
        """
        mutated = self.copy()

        if op["op"] == "add_stage":
            stage_dict = op["stage"]
            new_stage = StageDef(**stage_dict)
            if any(stage.name == new_stage.name for stage in mutated.stages):
                raise PipelineOperationError(f"Stage '{new_stage.name}' already exists")
            mutated.stages.append(new_stage)

        elif op["op"] == "remove_stage":
            name = op["stage_name"]
            if not any(stage.name == name for stage in mutated.stages):
                raise PipelineOperationError(f"Stage '{name}' not found")
            mutated.stages = [stage for stage in mutated.stages if stage.name != name]
            # Clean up dependency references
            for stage in mutated.stages:
                stage.depends_on = [dep for dep in stage.depends_on if dep != name]

        elif op["op"] == "update_stage":
            name = op["stage_name"]
            updates = op["updates"]
            is_found = False
            for stage in mutated.stages:
                if stage.name == name:
                    is_found = True
                    for key, value in updates.items():
                        if not hasattr(stage, key):
                            raise PipelineOperationError(f"StageDef has no field '{key}'")
                        setattr(stage, key, value)
                    break
            if not is_found:
                raise PipelineOperationError(f"Stage '{name}' not found")

        else:
            raise PipelineOperationError(f"Unknown operation: {op['op']}")

        return mutated

    def get_stage(self, name: str) -> StageDef | None:
        """Find a stage by name."""
        return next((stage for stage in self.stages if stage.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation."""
        return {
            "name": self.name,
            "stages": [stage.to_dict() for stage in self.stages],
            "default_retry": str(self.default_retry) if self.default_retry else None,
        }

    def _has_cycle(self) -> bool:
        """Detect cycles using DFS."""
        adjacency: dict[str, list[str]] = {
            stage.name: list(stage.depends_on) for stage in self.stages
        }
        unvisited, in_progress, visited = 0, 1, 2
        visit_state: dict[str, int] = {name: unvisited for name in adjacency}

        def dfs(node: str) -> bool:
            visit_state[node] = in_progress
            for dep in adjacency.get(node, []):
                if dep not in visit_state:
                    continue
                if visit_state[dep] == in_progress:
                    return True
                if visit_state[dep] == unvisited and dfs(dep):
                    return True
            visit_state[node] = visited
            return False

        return any(visit_state[node] == unvisited and dfs(node) for node in adjacency)

    def _topological_order(self) -> list[StageDef]:
        """Return stages in topological (dependency) order."""
        name_to_stage = {stage.name: stage for stage in self.stages}
        in_degree: dict[str, int] = {stage.name: 0 for stage in self.stages}
        dependents: dict[str, list[str]] = {stage.name: [] for stage in self.stages}

        for stage in self.stages:
            for dep in stage.depends_on:
                if dep in dependents:
                    dependents[dep].append(stage.name)
                    in_degree[stage.name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        ordered: list[StageDef] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            name = queue.pop(0)
            ordered.append(name_to_stage[name])
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return ordered
