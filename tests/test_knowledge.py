"""Tests for the Knowledge Store."""

import pytest
import tempfile
from pathlib import Path

from ganglion.knowledge.types import Pattern, Antipattern, KnowledgeQuery
from ganglion.knowledge.store import KnowledgeStore
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def json_backend(tmp_dir):
    return JsonKnowledgeBackend(tmp_dir / "knowledge")


@pytest.fixture
def store(json_backend):
    return KnowledgeStore(backend=json_backend)


class TestPattern:
    def test_to_dict_and_back(self):
        p = Pattern(
            capability="train",
            description="Used transformer backbone",
            config={"arch": "transformer"},
            metric_value=0.95,
            metric_name="accuracy",
        )
        d = p.to_dict()
        p2 = Pattern.from_dict(d)
        assert p2.capability == "train"
        assert p2.description == "Used transformer backbone"
        assert p2.config == {"arch": "transformer"}
        assert p2.metric_value == 0.95


class TestAntipattern:
    def test_to_dict_and_back(self):
        a = Antipattern(
            capability="train",
            error_summary="NaN in gradients",
            failure_mode="numerical_instability",
        )
        d = a.to_dict()
        a2 = Antipattern.from_dict(d)
        assert a2.capability == "train"
        assert a2.error_summary == "NaN in gradients"
        assert a2.failure_mode == "numerical_instability"


class TestJsonKnowledgeBackend:
    def test_save_and_query_patterns(self, json_backend):
        json_backend.save_pattern(
            Pattern(capability="train", description="Good approach", metric_value=0.9)
        )
        json_backend.save_pattern(
            Pattern(capability="eval", description="Eval approach", metric_value=0.8)
        )

        results = json_backend.query_patterns(KnowledgeQuery(capability="train"))
        assert len(results) == 1
        assert results[0].description == "Good approach"

    def test_save_and_query_antipatterns(self, json_backend):
        json_backend.save_antipattern(
            Antipattern(capability="train", error_summary="OOM error")
        )

        results = json_backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert len(results) == 1
        assert results[0].error_summary == "OOM error"

    def test_count(self, json_backend):
        json_backend.save_pattern(Pattern(capability="a", description="p1"))
        json_backend.save_pattern(Pattern(capability="b", description="p2"))
        json_backend.save_antipattern(Antipattern(capability="a", error_summary="e1"))

        counts = json_backend.count()
        assert counts["patterns"] == 2
        assert counts["antipatterns"] == 1

    def test_trim(self, json_backend):
        for i in range(10):
            json_backend.save_pattern(Pattern(capability="a", description=f"p{i}"))

        json_backend.trim(max_patterns=5, max_antipatterns=5)
        assert json_backend.count()["patterns"] == 5

    def test_max_entries(self, json_backend):
        for i in range(10):
            json_backend.save_pattern(Pattern(capability="a", description=f"p{i}"))

        results = json_backend.query_patterns(KnowledgeQuery(max_entries=3))
        assert len(results) == 3


class TestKnowledgeStore:
    def test_record_success(self, store):
        store.record_success(
            capability="train",
            description="Transformer worked well",
            metric_value=0.95,
            metric_name="accuracy",
        )
        assert store.summary()["patterns"] == 1

    def test_record_failure(self, store):
        store.record_failure(
            capability="train",
            error_summary="Model collapsed",
            failure_mode="mode_collapse",
        )
        assert store.summary()["antipatterns"] == 1

    def test_error_summary_truncation(self, store):
        long_error = "x" * 1000
        store.record_failure(capability="train", error_summary=long_error)
        results = store.backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert len(results[0].error_summary) <= 500

    def test_to_prompt_context(self, store):
        store.record_success(
            capability="train",
            description="Conv+Gaussian head",
            metric_value=0.85,
            metric_name="crps",
        )
        store.record_failure(
            capability="train",
            error_summary="LSTM too slow",
            failure_mode="performance",
        )

        ctx = store.to_prompt_context("train")
        assert "Accumulated Knowledge" in ctx
        assert "Known Good Approaches" in ctx
        assert "Conv+Gaussian head" in ctx
        assert "Known Failures" in ctx
        assert "LSTM too slow" in ctx

    def test_to_prompt_context_empty(self, store):
        ctx = store.to_prompt_context("nonexistent")
        assert ctx == ""

    def test_trim(self, store):
        store.max_patterns = 3
        for i in range(10):
            store.record_success(capability="train", description=f"approach {i}")
        store.trim()
        assert store.summary()["patterns"] == 3
