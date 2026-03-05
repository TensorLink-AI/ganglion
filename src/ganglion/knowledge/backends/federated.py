"""Federated knowledge backend — write local, read from all peers."""

from __future__ import annotations

import logging
from pathlib import Path

from ganglion.knowledge.backends.base import KnowledgeBackend, PeerDiscovery
from ganglion.knowledge.backends.json_backend import JsonKnowledgeBackend
from ganglion.knowledge.types import AgentDesignPattern, Antipattern, KnowledgeQuery, Pattern

logger = logging.getLogger(__name__)


class FederatedKnowledgeBackend:
    """Write locally, read from all peers.

    Each bot owns a local backend for its own writes.
    Reads merge results from local + all discovered peer backends.
    The discovery and transport mechanism is pluggable.
    """

    def __init__(
        self,
        local: KnowledgeBackend,
        peers: PeerDiscovery,
    ):
        self.local = local
        self.peers = peers

    async def save_pattern(self, pattern: Pattern) -> None:
        await self.local.save_pattern(pattern)

    async def save_antipattern(self, antipattern: Antipattern) -> None:
        await self.local.save_antipattern(antipattern)

    async def save_agent_design(self, design: AgentDesignPattern) -> None:
        await self.local.save_agent_design(design)

    async def query_agent_designs(self, query: KnowledgeQuery) -> list[AgentDesignPattern]:
        # Agent designs query local only for now; peer discovery for designs
        # can be added when PeerDiscovery protocol is extended.
        return await self.local.query_agent_designs(query)

    async def query_patterns(self, query: KnowledgeQuery) -> list[Pattern]:
        local_results = await self.local.query_patterns(query)
        try:
            peer_results = await self.peers.query_all_patterns(query)
        except Exception as e:
            logger.warning("Failed to query peer patterns: %s", e)
            peer_results = []

        merged = local_results + peer_results

        if query.exclude_source is not None:
            merged = [p for p in merged if p.source_bot != query.exclude_source]

        merged.sort(key=lambda p: p.timestamp, reverse=True)
        return merged[: query.max_entries]

    async def query_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]:
        local_results = await self.local.query_antipatterns(query)
        try:
            peer_results = await self.peers.query_all_antipatterns(query)
        except Exception as e:
            logger.warning("Failed to query peer antipatterns: %s", e)
            peer_results = []

        merged = local_results + peer_results

        if query.exclude_source is not None:
            merged = [a for a in merged if a.source_bot != query.exclude_source]

        merged.sort(key=lambda a: a.timestamp, reverse=True)
        return merged[: query.max_entries]

    async def count(self) -> dict[str, int]:
        return await self.local.count()

    async def trim(self, max_patterns: int = 500, max_antipatterns: int = 500) -> None:
        await self.local.trim(max_patterns, max_antipatterns)


class FilesystemPeerDiscovery:
    """Discover peers by scanning sibling directories on the same filesystem.

    Layout:
        base_dir/
            alpha/          <-- bot "alpha" writes here
                patterns.json
                antipatterns.json
            beta/           <-- bot "beta" writes here
                patterns.json
                antipatterns.json

    Each bot writes to base_dir/{bot_id}/. Discovery scans all sibling
    directories (excluding our own) and reads their JSON files.
    """

    def __init__(self, base_dir: str | Path, local_bot_id: str):
        self.base_dir = Path(base_dir)
        self.local_bot_id = local_bot_id

    def _discover_peer_backends(self) -> list[JsonKnowledgeBackend]:
        """Find all peer directories (excluding our own)."""
        if not self.base_dir.is_dir():
            return []
        peers = []
        for child in sorted(self.base_dir.iterdir()):
            if child.is_dir() and child.name != self.local_bot_id:
                peers.append(JsonKnowledgeBackend(child))
        return peers

    async def query_all_patterns(self, query: KnowledgeQuery) -> list[Pattern]:
        results: list[Pattern] = []
        for peer in self._discover_peer_backends():
            try:
                results.extend(await peer.query_patterns(query))
            except Exception as e:
                logger.warning("Failed to read patterns from peer %s: %s", peer.directory, e)
        return results

    async def query_all_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]:
        results: list[Antipattern] = []
        for peer in self._discover_peer_backends():
            try:
                results.extend(await peer.query_antipatterns(query))
            except Exception as e:
                logger.warning("Failed to read antipatterns from peer %s: %s", peer.directory, e)
        return results
