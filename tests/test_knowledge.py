"""Tests for the Knowledge Store."""

import tempfile
from pathlib import Path

import pytest

from ganglion.knowledge.backends.federated import (
    FederatedKnowledgeBackend,
    FilesystemPeerDiscovery,
)
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend
from ganglion.knowledge.backends.sqlite_backend import SqliteKnowledgeBackend
from ganglion.knowledge.store import KnowledgeStore
from ganglion.knowledge.types import Antipattern, KnowledgeQuery, Pattern


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


@pytest.mark.asyncio
class TestJsonKnowledgeBackend:
    async def test_save_and_query_patterns(self, json_backend):
        await json_backend.save_pattern(
            Pattern(capability="train", description="Good approach", metric_value=0.9)
        )
        await json_backend.save_pattern(
            Pattern(capability="eval", description="Eval approach", metric_value=0.8)
        )

        results = await json_backend.query_patterns(KnowledgeQuery(capability="train"))
        assert len(results) == 1
        assert results[0].description == "Good approach"

    async def test_save_and_query_antipatterns(self, json_backend):
        await json_backend.save_antipattern(
            Antipattern(capability="train", error_summary="OOM error")
        )

        results = await json_backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert len(results) == 1
        assert results[0].error_summary == "OOM error"

    async def test_count(self, json_backend):
        await json_backend.save_pattern(Pattern(capability="a", description="p1"))
        await json_backend.save_pattern(Pattern(capability="b", description="p2"))
        await json_backend.save_antipattern(Antipattern(capability="a", error_summary="e1"))

        counts = await json_backend.count()
        assert counts["patterns"] == 2
        assert counts["antipatterns"] == 1

    async def test_trim(self, json_backend):
        for i in range(10):
            await json_backend.save_pattern(Pattern(capability="a", description=f"p{i}"))

        await json_backend.trim(max_patterns=5, max_antipatterns=5)
        assert (await json_backend.count())["patterns"] == 5

    async def test_max_entries(self, json_backend):
        for i in range(10):
            await json_backend.save_pattern(Pattern(capability="a", description=f"p{i}"))

        results = await json_backend.query_patterns(KnowledgeQuery(max_entries=3))
        assert len(results) == 3


@pytest.mark.asyncio
class TestKnowledgeStore:
    async def test_record_success(self, store):
        await store.record_success(
            capability="train",
            description="Transformer worked well",
            metric_value=0.95,
            metric_name="accuracy",
        )
        assert (await store.summary())["patterns"] == 1

    async def test_record_failure(self, store):
        await store.record_failure(
            capability="train",
            error_summary="Model collapsed",
            failure_mode="mode_collapse",
        )
        assert (await store.summary())["antipatterns"] == 1

    async def test_error_summary_truncation(self, store):
        long_error = "x" * 1000
        await store.record_failure(capability="train", error_summary=long_error)
        results = await store.backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert len(results[0].error_summary) <= 500

    async def test_to_prompt_context(self, store):
        await store.record_success(
            capability="train",
            description="Conv+Gaussian head",
            metric_value=0.85,
            metric_name="crps",
        )
        await store.record_failure(
            capability="train",
            error_summary="LSTM too slow",
            failure_mode="performance",
        )

        ctx = await store.to_prompt_context("train")
        assert "Accumulated Knowledge" in ctx
        assert "Known Good Approaches" in ctx
        assert "Conv+Gaussian head" in ctx
        assert "Known Failures" in ctx
        assert "LSTM too slow" in ctx

    async def test_to_prompt_context_empty(self, store):
        ctx = await store.to_prompt_context("nonexistent")
        assert ctx == ""

    async def test_trim(self, store):
        store.max_patterns = 3
        for i in range(10):
            await store.record_success(capability="train", description=f"approach {i}")
        await store.trim()
        assert (await store.summary())["patterns"] == 3


@pytest.mark.asyncio
class TestMultiBotSharedKnowledge:
    """Tests for multi-bot shared knowledge via source_bot tagging and exclude_source filtering."""

    async def test_source_bot_tagging(self, tmp_dir):
        """KnowledgeStore with bot_id tags all writes with source_bot."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend, bot_id="alpha")

        await store.record_success(capability="train", description="Good approach")
        await store.record_failure(capability="train", error_summary="Bad approach")

        patterns = await backend.query_patterns(KnowledgeQuery(capability="train"))
        assert len(patterns) == 1
        assert patterns[0].source_bot == "alpha"

        antipatterns = await backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert len(antipatterns) == 1
        assert antipatterns[0].source_bot == "alpha"

    async def test_source_bot_none(self, tmp_dir):
        """KnowledgeStore without bot_id leaves source_bot as None (backward compat)."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend)

        await store.record_success(capability="train", description="Good approach")
        await store.record_failure(capability="train", error_summary="Bad approach")

        patterns = await backend.query_patterns(KnowledgeQuery(capability="train"))
        assert patterns[0].source_bot is None

        antipatterns = await backend.query_antipatterns(KnowledgeQuery(capability="train"))
        assert antipatterns[0].source_bot is None

    async def test_exclude_source_filtering(self, tmp_dir):
        """query_patterns with exclude_source filters correctly."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")

        await backend.save_pattern(Pattern(capability="train", description="From alpha", source_bot="alpha"))
        await backend.save_pattern(Pattern(capability="train", description="From beta", source_bot="beta"))
        await backend.save_pattern(Pattern(capability="train", description="From nobody", source_bot=None))

        # Exclude alpha — should get beta's and untagged entries
        results = await backend.query_patterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        descriptions = {r.description for r in results}
        assert "From beta" in descriptions
        assert "From nobody" in descriptions
        assert "From alpha" not in descriptions

        # Same for antipatterns
        await backend.save_antipattern(Antipattern(capability="train", error_summary="Alpha fail", source_bot="alpha"))
        await backend.save_antipattern(Antipattern(capability="train", error_summary="Beta fail", source_bot="beta"))

        results = await backend.query_antipatterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        summaries = {r.error_summary for r in results}
        assert "Beta fail" in summaries
        assert "Alpha fail" not in summaries

    async def test_exclude_source_none(self, tmp_dir):
        """query_patterns without exclude_source returns everything."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")

        await backend.save_pattern(Pattern(capability="train", description="From alpha", source_bot="alpha"))
        await backend.save_pattern(Pattern(capability="train", description="From beta", source_bot="beta"))

        results = await backend.query_patterns(KnowledgeQuery(capability="train"))
        assert len(results) == 2

    async def test_foreign_prompt_context(self, tmp_dir):
        """to_foreign_prompt_context returns only other bots' entries."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend, bot_id="alpha")

        # Alpha records its own knowledge
        await store.record_success(capability="train", description="Alpha's approach", metric_value=0.9, metric_name="acc")
        await store.record_failure(capability="train", error_summary="Alpha's mistake")

        # Manually add entries from beta
        await backend.save_pattern(Pattern(capability="train", description="Beta's approach", source_bot="beta", metric_value=0.8, metric_name="acc"))
        await backend.save_antipattern(Antipattern(capability="train", error_summary="Beta's mistake", source_bot="beta", failure_mode="oom"))

        ctx = await store.to_foreign_prompt_context("train")
        assert "Discoveries from other bots" in ctx
        assert "Approaches that worked for others" in ctx
        assert "Beta's approach" in ctx
        assert "Dead ends found by others" in ctx
        assert "Beta's mistake" in ctx
        # Should NOT contain alpha's own entries
        assert "Alpha's approach" not in ctx
        assert "Alpha's mistake" not in ctx

    async def test_foreign_prompt_context_no_bot_id(self, tmp_dir):
        """to_foreign_prompt_context returns '' when bot_id not set."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store = KnowledgeStore(backend=backend)

        await backend.save_pattern(Pattern(capability="train", description="Some approach", source_bot="beta"))

        ctx = await store.to_foreign_prompt_context("train")
        assert ctx == ""

    async def test_shared_backend_two_stores(self, tmp_dir):
        """Two KnowledgeStore instances with different bot_ids on same backend
        can read each other's entries via to_foreign_prompt_context but not via to_prompt_context."""
        backend = JsonKnowledgeBackend(tmp_dir / "knowledge")
        store_alpha = KnowledgeStore(backend=backend, bot_id="alpha")
        store_beta = KnowledgeStore(backend=backend, bot_id="beta")

        # Alpha records a success
        await store_alpha.record_success(
            capability="train", description="Alpha discovered X", metric_value=0.9, metric_name="acc",
        )
        # Beta records a failure
        await store_beta.record_failure(
            capability="train", error_summary="Beta hit dead end Y", failure_mode="timeout",
        )

        # Alpha's own prompt context should contain its own pattern
        alpha_own = await store_alpha.to_prompt_context("train")
        assert "Alpha discovered X" in alpha_own

        # Alpha's foreign context should show beta's entries but NOT alpha's own
        alpha_foreign = await store_alpha.to_foreign_prompt_context("train")
        assert "Beta hit dead end Y" in alpha_foreign
        assert "Alpha discovered X" not in alpha_foreign

        # Beta's foreign context should show alpha's entries but NOT beta's own
        beta_foreign = await store_beta.to_foreign_prompt_context("train")
        assert "Alpha discovered X" in beta_foreign
        assert "Beta hit dead end Y" not in beta_foreign

    async def test_exclude_source_sqlite(self, tmp_dir):
        """SQLite backend also respects exclude_source filtering."""
        backend = SqliteKnowledgeBackend(tmp_dir / "knowledge.db")

        await backend.save_pattern(Pattern(capability="train", description="From alpha", source_bot="alpha"))
        await backend.save_pattern(Pattern(capability="train", description="From beta", source_bot="beta"))
        await backend.save_pattern(Pattern(capability="train", description="From nobody", source_bot=None))

        # Exclude alpha
        results = await backend.query_patterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
        descriptions = {r.description for r in results}
        assert "From beta" in descriptions
        assert "From nobody" in descriptions
        assert "From alpha" not in descriptions

        # Antipatterns too
        await backend.save_antipattern(Antipattern(capability="train", error_summary="Alpha fail", source_bot="alpha"))
        await backend.save_antipattern(Antipattern(capability="train", error_summary="Beta fail", source_bot="beta"))

        results = await backend.query_antipatterns(KnowledgeQuery(capability="train", exclude_source="alpha"))
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


@pytest.mark.asyncio
class TestFederatedKnowledgeBackend:
    """Tests for the federated backend with filesystem peer discovery."""

    async def test_federated_write_local_read_peers(self, tmp_dir):
        """FederatedKnowledgeBackend writes locally and reads from peers."""
        base_dir = tmp_dir / "shared"
        base_dir.mkdir()

        # Set up alpha's local backend + federated backend
        alpha_local = JsonKnowledgeBackend(base_dir / "alpha")
        alpha_peers = FilesystemPeerDiscovery(base_dir, "alpha")
        alpha_fed = FederatedKnowledgeBackend(alpha_local, alpha_peers)

        # Set up beta's local backend + federated backend
        beta_local = JsonKnowledgeBackend(base_dir / "beta")
        beta_peers = FilesystemPeerDiscovery(base_dir, "beta")
        beta_fed = FederatedKnowledgeBackend(beta_local, beta_peers)

        # Alpha writes a pattern
        await alpha_fed.save_pattern(Pattern(
            capability="train", description="Alpha approach", source_bot="alpha",
        ))

        # Beta writes a pattern
        await beta_fed.save_pattern(Pattern(
            capability="train", description="Beta approach", source_bot="beta",
        ))

        # Alpha queries — should see both (own + peer)
        results = await alpha_fed.query_patterns(KnowledgeQuery(capability="train"))
        descriptions = {r.description for r in results}
        assert "Alpha approach" in descriptions
        assert "Beta approach" in descriptions

        # Alpha queries with exclude_source=alpha — should only see beta
        results = await alpha_fed.query_patterns(
            KnowledgeQuery(capability="train", exclude_source="alpha")
        )
        descriptions = {r.description for r in results}
        assert "Beta approach" in descriptions
        assert "Alpha approach" not in descriptions

    async def test_federated_with_knowledge_store(self, tmp_dir):
        """KnowledgeStore on top of FederatedKnowledgeBackend."""
        base_dir = tmp_dir / "shared"
        base_dir.mkdir()

        alpha_local = JsonKnowledgeBackend(base_dir / "alpha")
        alpha_fed = FederatedKnowledgeBackend(
            alpha_local, FilesystemPeerDiscovery(base_dir, "alpha"),
        )
        store_alpha = KnowledgeStore(backend=alpha_fed, bot_id="alpha")

        beta_local = JsonKnowledgeBackend(base_dir / "beta")
        beta_fed = FederatedKnowledgeBackend(
            beta_local, FilesystemPeerDiscovery(base_dir, "beta"),
        )
        store_beta = KnowledgeStore(backend=beta_fed, bot_id="beta")

        # Alpha records success
        await store_alpha.record_success(
            capability="train", description="Alpha found X", metric_value=0.9, metric_name="acc",
        )

        # Beta records failure
        await store_beta.record_failure(
            capability="train", error_summary="Beta dead end Y", failure_mode="timeout",
        )

        # Alpha's foreign context should show beta's entries
        alpha_foreign = await store_alpha.to_foreign_prompt_context("train")
        assert "Beta dead end Y" in alpha_foreign
        assert "Alpha found X" not in alpha_foreign

        # Beta's foreign context should show alpha's entries
        beta_foreign = await store_beta.to_foreign_prompt_context("train")
        assert "Alpha found X" in beta_foreign
        assert "Beta dead end Y" not in beta_foreign

    async def test_federated_no_peers(self, tmp_dir):
        """Federated backend works fine with no peers."""
        base_dir = tmp_dir / "shared"
        base_dir.mkdir()

        local = JsonKnowledgeBackend(base_dir / "solo")
        fed = FederatedKnowledgeBackend(
            local, FilesystemPeerDiscovery(base_dir, "solo"),
        )

        await fed.save_pattern(Pattern(capability="train", description="Solo approach"))
        results = await fed.query_patterns(KnowledgeQuery(capability="train"))
        assert len(results) == 1
        assert results[0].description == "Solo approach"

    async def test_filesystem_peer_discovery_ignores_own(self, tmp_dir):
        """FilesystemPeerDiscovery doesn't include own directory."""
        base_dir = tmp_dir / "shared"
        base_dir.mkdir()

        # Create directories for alpha and beta
        alpha_backend = JsonKnowledgeBackend(base_dir / "alpha")
        beta_backend = JsonKnowledgeBackend(base_dir / "beta")

        await alpha_backend.save_pattern(Pattern(capability="train", description="Alpha", source_bot="alpha"))
        await beta_backend.save_pattern(Pattern(capability="train", description="Beta", source_bot="beta"))

        # Alpha's peer discovery should only find beta
        discovery = FilesystemPeerDiscovery(base_dir, "alpha")
        peer_patterns = await discovery.query_all_patterns(KnowledgeQuery(capability="train"))
        descriptions = {p.description for p in peer_patterns}
        assert "Beta" in descriptions
        assert "Alpha" not in descriptions

    async def test_federated_antipatterns(self, tmp_dir):
        """FederatedKnowledgeBackend merges antipatterns from peers."""
        base_dir = tmp_dir / "shared"
        base_dir.mkdir()

        alpha_local = JsonKnowledgeBackend(base_dir / "alpha")
        alpha_fed = FederatedKnowledgeBackend(
            alpha_local, FilesystemPeerDiscovery(base_dir, "alpha"),
        )

        beta_local = JsonKnowledgeBackend(base_dir / "beta")
        beta_fed = FederatedKnowledgeBackend(
            beta_local, FilesystemPeerDiscovery(base_dir, "beta"),
        )

        await alpha_fed.save_antipattern(Antipattern(
            capability="train", error_summary="Alpha error", source_bot="alpha",
        ))
        await beta_fed.save_antipattern(Antipattern(
            capability="train", error_summary="Beta error", source_bot="beta",
        ))

        # Alpha should see both
        results = await alpha_fed.query_antipatterns(KnowledgeQuery(capability="train"))
        summaries = {r.error_summary for r in results}
        assert "Alpha error" in summaries
        assert "Beta error" in summaries

        # Excluding alpha should only show beta
        results = await alpha_fed.query_antipatterns(
            KnowledgeQuery(capability="train", exclude_source="alpha")
        )
        summaries = {r.error_summary for r in results}
        assert "Beta error" in summaries
        assert "Alpha error" not in summaries
