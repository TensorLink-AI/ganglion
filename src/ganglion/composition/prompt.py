"""Composable prompt fragment mechanism."""

from __future__ import annotations


class PromptBuilder:
    """Builds system prompts from reusable, named sections.

    Usage:
        prompt = (
            PromptBuilder()
            .section("role", "You are an experiment planner.")
            .section("context", subnet_config.to_prompt_section())
            .section("constraints", "Max 10 experiments per run.")
            .build()
        )
    """

    def __init__(self) -> None:
        self._sections: list[tuple[str, str]] = []

    def section(self, name: str, content: str) -> PromptBuilder:
        """Add a named section to the prompt."""
        self._sections.append((name, content))
        return self

    def build(self, separator: str = "\n\n") -> str:
        """Assemble all sections into a single prompt string."""
        parts = []
        for name, content in self._sections:
            if content.strip():
                parts.append(f"## {name}\n{content}")
        return separator.join(parts)

    def remove(self, name: str) -> PromptBuilder:
        """Remove all sections with the given name."""
        self._sections = [(n, c) for n, c in self._sections if n != name]
        return self

    def replace(self, name: str, content: str) -> PromptBuilder:
        """Replace the first section with the given name, or append if not found."""
        for i, (n, _) in enumerate(self._sections):
            if n == name:
                self._sections[i] = (name, content)
                return self
        return self.section(name, content)

    def has_section(self, name: str) -> bool:
        """Check if a section with the given name exists."""
        return any(n == name for n, _ in self._sections)

    def section_names(self) -> list[str]:
        """Return the names of all sections in order."""
        return [n for n, _ in self._sections]
