"""KnowledgeBackend protocol — pluggable storage for knowledge."""

from __future__ import annotations

from typing import Protocol

from ganglion.knowledge.types import AgentDesignPattern, Antipattern, KnowledgeQuery, Pattern


class KnowledgeBackend(Protocol):
    """Protocol for knowledge store backends.

    All methods are async to support networked backends (S3, HTTP, etc.)
    without blocking. Local backends simply return immediately.
    """

    async def save_pattern(self, pattern: Pattern) -> None: ...
    async def save_antipattern(self, antipattern: Antipattern) -> None: ...
    async def save_agent_design(self, design: AgentDesignPattern) -> None: ...
    async def query_patterns(self, query: KnowledgeQuery) -> list[Pattern]: ...
    async def query_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]: ...
    async def query_agent_designs(self, query: KnowledgeQuery) -> list[AgentDesignPattern]: ...
    async def count(self) -> dict[str, int]: ...
    async def trim(self, max_patterns: int = 500, max_antipatterns: int = 500) -> None: ...


class PeerDiscovery(Protocol):
    """Find and read from peer knowledge stores.

    The discovery and transport mechanism is pluggable:
    - FilesystemPeerDiscovery: scan sibling directories on same machine
    - S3PeerDiscovery: list prefixes in a shared bucket
    - HttpPeerDiscovery: fan out to peer /knowledge endpoints
    """

    async def query_all_patterns(self, query: KnowledgeQuery) -> list[Pattern]: ...
    async def query_all_antipatterns(self, query: KnowledgeQuery) -> list[Antipattern]: ...
