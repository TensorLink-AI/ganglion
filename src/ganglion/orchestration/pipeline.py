"""PipelineDef and StageDef — declarative pipeline definitions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ganglion.orchestration.errors import PipelineOperationError


@dataclass
class StageDef:
    """Definition of a single pipeline stage."""

    name: str
    agent: str
    retry: Any | None = None
    optional: bool = False
    depends_on: list[str] = field(default_factory=list)
    input_keys: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "agent": self.agent,
            "retry": str(self.retry) if self.retry else None,
            "optional": self.optional,
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
        stage_names = [s.name for s in self.stages]

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
                    errors.append(
                        f"Stage '{stage.name}' depends on '{dep}' which does not exist"
                    )

        # Check for cycles (topological sort)
        if not errors:
            if self._has_cycle():
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
                    name=s.name,
                    agent=s.agent,
                    retry=s.retry,
                    optional=s.optional,
                    depends_on=list(s.depends_on),
                    input_keys=list(s.input_keys),
                    output_keys=list(s.output_keys),
                )
                for s in self.stages
            ],
            default_retry=self.default_retry,
        )

    def apply_operation(self, op: dict) -> PipelineDef:
        """Apply a single pipeline mutation. Returns new PipelineDef.

        Operations:
        - {"op": "add_stage", "stage": {...}}
        - {"op": "remove_stage", "stage_name": "..."}
        - {"op": "update_stage", "stage_name": "...", "updates": {...}}
        """
        result = self.copy()

        if op["op"] == "add_stage":
            stage_dict = op["stage"]
            new_stage = StageDef(**stage_dict)
            if any(s.name == new_stage.name for s in result.stages):
                raise PipelineOperationError(f"Stage '{new_stage.name}' already exists")
            result.stages.append(new_stage)

        elif op["op"] == "remove_stage":
            name = op["stage_name"]
            if not any(s.name == name for s in result.stages):
                raise PipelineOperationError(f"Stage '{name}' not found")
            result.stages = [s for s in result.stages if s.name != name]
            # Clean up dependency references
            for s in result.stages:
                s.depends_on = [d for d in s.depends_on if d != name]

        elif op["op"] == "update_stage":
            name = op["stage_name"]
            updates = op["updates"]
            found = False
            for s in result.stages:
                if s.name == name:
                    found = True
                    for key, value in updates.items():
                        if not hasattr(s, key):
                            raise PipelineOperationError(f"StageDef has no field '{key}'")
                        setattr(s, key, value)
                    break
            if not found:
                raise PipelineOperationError(f"Stage '{name}' not found")

        else:
            raise PipelineOperationError(f"Unknown operation: {op['op']}")

        return result

    def get_stage(self, name: str) -> StageDef | None:
        """Find a stage by name."""
        return next((s for s in self.stages if s.name == name), None)

    def to_dict(self) -> dict:
        """Serializable representation."""
        return {
            "name": self.name,
            "stages": [s.to_dict() for s in self.stages],
            "default_retry": str(self.default_retry) if self.default_retry else None,
        }

    def _has_cycle(self) -> bool:
        """Detect cycles using DFS."""
        adj: dict[str, list[str]] = {s.name: list(s.depends_on) for s in self.stages}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in adj}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for dep in adj.get(node, []):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[node] = BLACK
            return False

        for node in adj:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def _topological_order(self) -> list[StageDef]:
        """Return stages in topological (dependency) order."""
        name_to_stage = {s.name: s for s in self.stages}
        in_degree: dict[str, int] = {s.name: 0 for s in self.stages}
        dependents: dict[str, list[str]] = {s.name: [] for s in self.stages}

        for s in self.stages:
            for dep in s.depends_on:
                if dep in dependents:
                    dependents[dep].append(s.name)
                    in_degree[s.name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[StageDef] = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            name = queue.pop(0)
            result.append(name_to_stage[name])
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result
