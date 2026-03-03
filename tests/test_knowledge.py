"""Tests for the Knowledge Store."""

import pytest
import tempfile
from pathlib import Path

from ganglion.knowledge.types import Pattern, Antipattern, KnowledgeQuery
from ganglion.knowledge.store import KnowledgeStore
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend
from ganglion.knowledge.backends.sqlite_backend import SqliteKnowledgeBackend


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


class TestMultiBotSharedKnowledge:
    """Tests for multi-bot shared knowledge via source_bot tagging and exclude_source filtering."""

    def test_source_bot_tagging(self, tmp_dir):
        """KnowledgeStore with bot_id tags all writes with source_bot."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend, bot_id="alpha")

        store.record_success(capability="train", description="Good approach")
        store.record_failure(capability="train", error_summary="Bad approach")

        patterns = backend.query_patterns(KnowledgeQuery(capability="train"))
        assert len(patterns) == 1
        assert patterns[0].source_bot == "alpha"

        antipatterns = backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert len(antipatterns) == 1
        assert antipatterns[0].source_bot == "alpha"

    def test_source_bot_none(self, tmp_dir):
        """KnowledgeStore without bot_id leaves source_bot as None (backward compat)."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend)

        store.record_success(capability="train", description="Good approach")
        store.record_failure(capability="train", error_summary="Bad approach")

        patterns = backend.query_patterns(KnowledgeQuery(capability="train"))
        assert patterns[0].source_bot is None

        antipatterns = backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert antipatterns[0].source_bot is None

    def test_exclude_source_filtering(self, tmp_dir):
        """query_patterns with exclude_source filters correctly."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")

        backend.save_pattern(Pattern(capability="train", description="From alpha", source_bot="alpha"))
        backend.save_pattern(Pattern(capability="train", description="From beta", source_bot="beta"))
        backend.save_pattern(Pattern(capability="train", description="From nobody", source_bot=None))

        # Exclude alpha — should get beta's and untagged entries
        results = backend.query_patterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        descriptions = {r.description for r in results}
        assert "From beta" in descriptions
        assert "From nobody" in descriptions
        assert "From alpha" not in descriptions

        # Same for antipatterns
        backend.save_antipattern(Antipattern(capability="train", error_summary="Alpha fail", source_bot="alpha"))
        backend.save_antipattern(Antipattern(capability="train", error_summary="Beta fail", source_bot="beta"))

        results = backend.query_antipatterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        summaries = {r.error_summary for r in results}
        assert "Beta fail" in summaries
        assert "Alpha fail" not in summaries

    def test_exclude_source_none(self, tmp_dir):
        """query_patterns without exclude_source returns everything."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")

        backend.save_pattern(Pattern(capability="train", description="From alpha", source_bot="alpha"))
        backend.save_pattern(Pattern(capability="train", description="From beta", source_bot="beta"))

        results = backend.query_patterns(KnowledgeQuery(capability="train"))
        assert len(results) == 2

    def test_foreign_prompt_context(self, tmp_dir):
        """to_foreign_prompt_context returns only other bots' entries."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend, bot_id="alpha")

        # Alpha records its own knowledge
        store.record_success(capability="train", description="Alpha's approach", metric_value=0.9, metric_name="acc")
        store.record_failure(capability="train", error_summary="Alpha's mistake")

        # Manually add entries from beta
        backend.save_pattern(Pattern(capability="train", description="Beta's approach", source_bot="beta", metric_value=0.8, metric_name="acc"))
        backend.save_antipattern(Antipattern(capability="train", error_summary="Beta's mistake", source_bot="beta", failure_mode="oom"))

        ctx = store.to_foreign_prompt_context("train")
        assert "Discoveries from other bots" in ctx
        assert "Approaches that worked for others" in ctx
        assert "Beta's approach" in ctx
        assert "Dead ends found by others" in ctx
        assert "Beta's mistake" in ctx
        # Should NOT contain alpha's own entries
        assert "Alpha's approach" not in ctx
        assert "Alpha's mistake" not in ctx

    def test_foreign_prompt_context_no_bot_id(self, tmp_dir):
        """to_foreign_prompt_context returns '' when bot_id not set."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend)

        backend.save_pattern(Pattern(capability="train", description="Some approach", source_bot="beta"))

        ctx = store.to_foreign_prompt_context("train")
        assert ctx == ""

    def test_shared_backend_two_stores(self, tmp_dir):
        """Two KnowledgeStore instances with different bot_ids on same backend
        can read each other's entries via to_foreign_prompt_context but not via to_prompt_context."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store_alpha = KnowledgeStore(backend=backend, bot_id="alpha")
        store_beta = KnowledgeStore(backend=backend, bot_id="beta")

        # Alpha records a success
        store_alpha.record_success(
            capability="train", description="Alpha discovered X", metric_value=0.9, metric_name="acc",
        )
        # Beta records a failure
        store_beta.record_failure(
            capability="train", error_summary="Beta hit dead end Y", failure_mode="timeout",
        )

        # Alpha's own prompt context should contain its own pattern but NOT beta's antipattern
        alpha_own = store_alpha.to_prompt_context("train")
        assert "Alpha discovered X" in alpha_own
        # to_prompt_context doesn't filter by source_bot, so it shows everything
        # The private vs foreign distinction is in to_foreign_prompt_context

        # Alpha's foreign context should show beta's entries but NOT alpha's own
        alpha_foreign = store_alpha.to_foreign_prompt_context("train")
        assert "Beta hit dead end Y" in alpha_foreign
        assert "Alpha discovered X" not in alpha_foreign

        # Beta's foreign context should show alpha's entries but NOT beta's own
        beta_foreign = store_beta.to_foreign_prompt_context("train")
        assert "Alpha discovered X" in beta_foreign
        assert "Beta hit dead end Y" not in beta_foreign

    def test_exclude_source_sqlite(self, tmp_dir):
        """SQLite backend also respects exclude_source filtering."""
        backend = SqliteKnowledgeBackend(tmp_dir / "knowledge.db")

        backend.save_pattern(Pattern(capability="train", description="From alpha", source_bot="alpha"))
        backend.save_pattern(Pattern(capability="train", description="From beta", source_bot="beta"))
        backend.save_pattern(Pattern(capability="train", description="From nobody", source_bot=None))

        # Exclude alpha
        results = backend.query_patterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        descriptions = {r.description for r in results}
        assert "From beta" in descriptions
        assert "From nobody" in descriptions
        assert "From alpha" not in descriptions

        # Antipatterns too
        backend.save_antipattern(Antipattern(capability="train", error_summary="Alpha fail", source_bot="alpha"))
        backend.save_antipattern(Antipattern(capability="train", error_summary="Beta fail", source_bot="beta"))

        results = backend.query_antipatterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        summaries = {r.error_summary for r in results}
        assert "Beta fail" in summaries
        assert "Alpha fail" not in summaries

    def test_source_bot_serialization_roundtrip(self):
        """source_bot survives to_dict/from_dict roundtrip."""
        p = Pattern(capability="train", description="Test", source_bot="gamma")
        d = p.to_dict()
        assert d["source_bot"] == "gamma"
        p2 = Pattern.from_dict(d)
        assert p2.source_bot == "gamma"

        a = Antipattern(capability="train", error_summary="Err", source_bot="delta")
        d = a.to_dict()
        assert d["source_bot"] == "delta"
        a2 = Antipattern.from_dict(d)
        assert a2.source_bot == "delta"
