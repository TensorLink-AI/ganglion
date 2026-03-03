"""Tests for Layer 1: Runtime."""

import json
import pytest

from ganglion.runtime.types import AgentResult, ToolCall, ToolResult
from ganglion.runtime.coerce import (
    CoercionPipeline,
    coerce_json_strings,
    coerce_empty_to_list,
    coerce_string_bools,
    coerce_string_numbers,
)


class TestAgentResult:
    def test_defaults(self):
        r = AgentResult(success=True)
        assert r.success is True
        assert r.structured is None
        assert r.raw_text == ""
        assert r.messages == []
        assert r.turns_used == 0

    def test_with_data(self):
        r = AgentResult(
            success=False,
            structured={"error": "oops"},
            raw_text="Something failed",
            turns_used=3,
        )
        assert r.success is False
        assert r.structured["error"] == "oops"
        assert r.turns_used == 3


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(id="tc_1", name="my_tool", arguments={"x": 1})
        assert tc.id == "tc_1"
        assert tc.name == "my_tool"
        assert tc.arguments == {"x": 1}


class TestToolResult:
    def test_basic(self):
        tr = ToolResult(tool_call_id="tc_1", name="my_tool", content="result")
        assert tr.is_finish is False
        assert tr.structured is None

    def test_finish(self):
        tr = ToolResult(
            tool_call_id="tc_1",
            name="finish",
            content="done",
            is_finish=True,
            structured={"success": True},
        )
        assert tr.is_finish is True


class TestCoercionFunctions:
    def test_coerce_json_strings_dict(self):
        value, modified = coerce_json_strings("arg", '{"key": "value"}', None)
        assert modified is True
        assert value == {"key": "value"}

    def test_coerce_json_strings_list(self):
        value, modified = coerce_json_strings("arg", '[1, 2, 3]', None)
        assert modified is True
        assert value == [1, 2, 3]

    def test_coerce_json_strings_non_json(self):
        value, modified = coerce_json_strings("arg", "hello world", None)
        assert modified is False
        assert value == "hello world"

    def test_coerce_json_strings_non_string(self):
        value, modified = coerce_json_strings("arg", 42, None)
        assert modified is False
        assert value == 42

    def test_coerce_empty_to_list(self):
        value, modified = coerce_empty_to_list("arg", "", list)
        assert modified is True
        assert value == []

    def test_coerce_empty_to_list_none(self):
        value, modified = coerce_empty_to_list("arg", None, list)
        assert modified is True
        assert value == []

    def test_coerce_empty_to_list_no_match(self):
        value, modified = coerce_empty_to_list("arg", "hello", list)
        assert modified is False

    def test_coerce_string_bools_true(self):
        value, modified = coerce_string_bools("arg", "true", bool)
        assert modified is True
        assert value is True

    def test_coerce_string_bools_false(self):
        value, modified = coerce_string_bools("arg", "False", bool)
        assert modified is True
        assert value is False

    def test_coerce_string_bools_no_match(self):
        value, modified = coerce_string_bools("arg", "maybe", bool)
        assert modified is False

    def test_coerce_string_numbers_int(self):
        value, modified = coerce_string_numbers("arg", "42", int)
        assert modified is True
        assert value == 42

    def test_coerce_string_numbers_float(self):
        value, modified = coerce_string_numbers("arg", "3.14", float)
        assert modified is True
        assert abs(value - 3.14) < 0.001


class TestCoercionPipeline:
    def test_default_pipeline(self):
        pipeline = CoercionPipeline()
        result = pipeline.apply(
            {"data": '{"x": 1}', "flag": "true", "items": ""},
            {"data": dict, "flag": bool, "items": list},
        )
        assert result["data"] == {"x": 1}
        assert result["flag"] is True
        assert result["items"] == []

    def test_custom_pipeline(self):
        pipeline = CoercionPipeline([coerce_json_strings])
        result = pipeline.apply({"data": '{"x": 1}', "flag": "maybe"}, {"flag": bool})
        assert result["data"] == {"x": 1}
        # String bools NOT coerced because we only have json coercion
        assert result["flag"] == "maybe"

    def test_no_modification(self):
        pipeline = CoercionPipeline()
        result = pipeline.apply({"x": 42, "y": "hello"})
        assert result == {"x": 42, "y": "hello"}
