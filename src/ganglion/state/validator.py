"""MutationValidator — validates externally-written code before registration."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of validating code."""

    is_passed: bool
    errors: list[str] = field(default_factory=list)


class MutationValidator:
    """Validates externally-written code before it's registered."""

    def __init__(self, blocked_imports: list[str] | None = None):
        self.blocked_imports = blocked_imports or [
            "subprocess",
            "os.system",
            "shutil.rmtree",
            "socket",
            "http.server",
        ]

    def validate_tool(self, code: str) -> ValidationResult:
        """Checks: valid syntax, @tool decorator, type hints on all params,
        no blocked imports, docstring present."""
        errors: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(is_passed=False, errors=[f"Syntax error: {e}"])

        has_tool_decorator = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    decorator_name = None
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorator_name = decorator.func.id
                        elif isinstance(decorator.func, ast.Attribute):
                            decorator_name = decorator.func.attr
                    elif isinstance(decorator, ast.Name):
                        decorator_name = decorator.id

                    if decorator_name == "tool":
                        has_tool_decorator = True
                        for arg in node.args.args:
                            if arg.annotation is None and arg.arg != "self":
                                errors.append(
                                    f"Parameter '{arg.arg}' missing type hint"
                                )

                if not ast.get_docstring(node):
                    errors.append(f"Function '{node.name}' missing docstring")

        if not has_tool_decorator:
            errors.append("No @tool decorator found")

        errors.extend(self._check_blocked_imports(tree))

        return ValidationResult(is_passed=len(errors) == 0, errors=errors)

    def validate_agent(self, code: str) -> ValidationResult:
        """Checks: valid syntax, class inheriting BaseAgentWrapper,
        implements build_system_prompt and build_tools, no blocked imports."""
        errors: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(is_passed=False, errors=[f"Syntax error: {e}"])

        has_agent_class = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                base_names = [
                    getattr(b, "id", getattr(b, "attr", "")) for b in node.bases
                ]
                if "BaseAgentWrapper" in base_names:
                    has_agent_class = True
                    methods = {
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    }
                    if "build_system_prompt" not in methods:
                        errors.append("Missing build_system_prompt method")
                    if "build_tools" not in methods:
                        errors.append("Missing build_tools method")

        if not has_agent_class:
            errors.append("No class inheriting from BaseAgentWrapper found")

        errors.extend(self._check_blocked_imports(tree))

        return ValidationResult(is_passed=len(errors) == 0, errors=errors)

    def validate_pipeline(self, pipeline_def: object) -> ValidationResult:
        """Validate a PipelineDef instance."""
        if hasattr(pipeline_def, "validate"):
            errors = pipeline_def.validate()  # type: ignore[union-attr]
            return ValidationResult(is_passed=len(errors) == 0, errors=errors)
        return ValidationResult(is_passed=False, errors=["Not a valid PipelineDef"])

    def _check_blocked_imports(self, tree: ast.AST) -> list[str]:
        """Check for blocked imports in an AST."""
        errors: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.blocked_imports:
                        errors.append(f"Blocked import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    for blocked in self.blocked_imports:
                        if full_name.startswith(blocked) or module.startswith(
                            blocked
                        ):
                            errors.append(f"Blocked import: {full_name}")
                            break
        return errors
