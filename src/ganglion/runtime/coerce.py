"""Modular coercion pipeline for sanitizing LLM tool-call arguments."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def coerce_json_strings(
    arg_name: str,
    value: Any,
    expected_type: type | None,
) -> tuple[Any, bool]:
    """Parse JSON-encoded strings into their native types."""
    if not isinstance(value, str):
        return value, False
    stripped = value.strip()
    if not stripped:
        return value, False
    if stripped[0] in ('{', '[', '"') or stripped in ('true', 'false', 'null'):
        try:
            parsed = json.loads(stripped)
            logger.debug("Coerced JSON string for arg '%s': %r -> %r", arg_name, value, parsed)
            return parsed, True
        except (json.JSONDecodeError, ValueError):
            pass
    return value, False


def coerce_empty_to_list(
    arg_name: str,
    value: Any,
    expected_type: type | None,
) -> tuple[Any, bool]:
    """Coerce empty strings or None to empty list when list is expected."""
    if expected_type is list and (value is None or value == ""):
        logger.debug("Coerced empty to list for arg '%s'", arg_name)
        return [], True
    return value, False


def coerce_string_bools(
    arg_name: str,
    value: Any,
    expected_type: type | None,
) -> tuple[Any, bool]:
    """Coerce string booleans like 'true'/'false' to actual bools."""
    if expected_type is bool and isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("true", "1", "yes"):
            return True, True
        if lower in ("false", "0", "no"):
            return False, True
    return value, False


def coerce_string_numbers(
    arg_name: str,
    value: Any,
    expected_type: type | None,
) -> tuple[Any, bool]:
    """Coerce numeric strings to int or float when expected."""
    if isinstance(value, str):
        if expected_type is int:
            try:
                return int(value), True
            except ValueError:
                pass
        elif expected_type is float:
            try:
                return float(value), True
            except ValueError:
                pass
    return value, False


class CoercionPipeline:
    """Runs a sequence of coercion functions over tool-call arguments."""

    def __init__(self, coercions: list | None = None):
        self.coercions = coercions or [
            coerce_json_strings,
            coerce_empty_to_list,
            coerce_string_bools,
        ]

    def apply(
        self,
        arguments: dict[str, Any],
        type_hints: dict[str, type] | None = None,
    ) -> dict[str, Any]:
        """Apply all coercions to a dict of tool-call arguments."""
        type_hints = type_hints or {}
        result = {}
        for arg_name, value in arguments.items():
            expected = type_hints.get(arg_name)
            current = value
            for coerce_fn in self.coercions:
                current, modified = coerce_fn(arg_name, current, expected)
                if modified:
                    logger.info(
                        "Coercion '%s' modified arg '%s': %r -> %r",
                        coerce_fn.__name__,
                        arg_name,
                        value,
                        current,
                    )
            result[arg_name] = current
        return result
