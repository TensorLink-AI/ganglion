"""ArtifactStore protocol — pluggable storage for job artifacts."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class ArtifactStore(Protocol):
    """Protocol for artifact storage backends.

    Jobs produce files (models, checkpoints, logs) that need to:
    - Outlive the compute process
    - Be shared between pipeline stages
    - Potentially be shared between bots
    """

    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes | None: ...

    async def list(self, prefix: str = "") -> list[str]: ...

    async def delete(self, key: str) -> bool: ...


class LocalArtifactStore:
    """Store artifacts on the local filesystem."""

    def __init__(self, root: Path | None = None):
        self._root = root or Path("./artifacts")
        self._root.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def get(self, key: str) -> bytes | None:
        path = self._root / key
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    async def list(self, prefix: str = "") -> list[str]:
        search_dir = self._root / prefix if prefix else self._root
        if not search_dir.is_dir():
            return []
        return [str(p.relative_to(self._root)) for p in search_dir.rglob("*") if p.is_file()]

    async def delete(self, key: str) -> bool:
        path = self._root / key
        if path.is_file():
            path.unlink()
            return True
        if path.is_dir():
            shutil.rmtree(path)
            return True
        return False
