"""ArtifactStore protocol — pluggable storage for job artifacts.

Artifacts are task-agnostic: model weights, code files, configs, logs,
checkpoints — anything a pipeline run or experiment produces. They are
keyed by ``{run_id}/{filename}`` so external bots can request all
artifacts associated with a specific run or experiment.

Metadata is stored as a JSON sidecar (``{key}.__meta__``) alongside each
artifact, recording the run/experiment context without changing the core
put/get protocol.

Backends:
  - ``LocalArtifactStore`` — filesystem (default)
  - ``S3ArtifactStore`` — S3-compatible (R2, Minio, Hippius, AWS S3)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMeta:
    """Metadata sidecar for a stored artifact."""

    key: str
    run_id: str = ""
    experiment_id: str = ""
    stage: str = ""
    content_type: str = ""
    size_bytes: int = 0
    created_at: float = 0.0
    source_bot: str | None = None
    url: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMeta:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ArtifactStore(Protocol):
    """Protocol for artifact storage backends.

    Jobs produce files (models, checkpoints, logs) that need to:
    - Outlive the compute process
    - Be shared between pipeline stages
    - Potentially be shared between bots

    Remote backends (S3, R2, Hippius) return URLs via ``get_url()``
    so bots can retrieve artifacts directly without proxying through
    the framework.
    """

    async def put(self, key: str, data: bytes, meta: ArtifactMeta | None = None) -> None: ...

    async def get(self, key: str) -> bytes | None: ...

    async def get_meta(self, key: str) -> ArtifactMeta | None: ...

    async def get_url(self, key: str) -> str | None: ...

    async def list(self, prefix: str = "") -> list[str]: ...

    async def list_meta(self, prefix: str = "") -> list[ArtifactMeta]: ...

    async def delete(self, key: str) -> bool: ...


class LocalArtifactStore:
    """Store artifacts on the local filesystem.

    Keys use ``{run_id}/{filename}`` convention. Metadata is stored as
    a JSON sidecar next to each artifact file.
    """

    def __init__(self, root: Path | None = None):
        self._root = root or Path("./artifacts")
        self._root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, key: str) -> Path:
        return self._root / f"{key}.__meta__"

    async def put(self, key: str, data: bytes, meta: ArtifactMeta | None = None) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        # Write metadata sidecar
        m = meta or ArtifactMeta(key=key)
        m.key = key
        m.size_bytes = len(data)
        if not m.created_at:
            m.created_at = time.time()
        meta_path = self._meta_path(key)
        meta_path.write_text(json.dumps(m.to_dict()))

    async def get(self, key: str) -> bytes | None:
        path = self._root / key
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    async def get_url(self, key: str) -> str | None:
        """Local store has no URLs — returns None."""
        return None

    async def get_meta(self, key: str) -> ArtifactMeta | None:
        meta_path = self._meta_path(key)
        if not meta_path.is_file():
            # Artifact exists but no sidecar — synthesize minimal metadata
            path = self._root / key
            if path.is_file():
                return ArtifactMeta(
                    key=key,
                    size_bytes=path.stat().st_size,
                    created_at=path.stat().st_mtime,
                )
            return None
        try:
            data = json.loads(meta_path.read_text())
            return ArtifactMeta.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

    async def list(self, prefix: str = "") -> list[str]:
        search_dir = self._root / prefix if prefix else self._root
        if not search_dir.is_dir():
            return []
        return [
            str(p.relative_to(self._root))
            for p in search_dir.rglob("*")
            if p.is_file() and not p.name.endswith(".__meta__")
        ]

    async def list_meta(self, prefix: str = "") -> list[ArtifactMeta]:
        """List all artifact metadata under a prefix."""
        keys = await self.list(prefix)
        metas = []
        for key in keys:
            meta = await self.get_meta(key)
            if meta:
                metas.append(meta)
        return metas

    async def delete(self, key: str) -> bool:
        path = self._root / key
        deleted = False
        if path.is_file():
            path.unlink()
            deleted = True
        elif path.is_dir():
            shutil.rmtree(path)
            deleted = True
        # Clean up sidecar
        meta_path = self._meta_path(key)
        if meta_path.is_file():
            meta_path.unlink()
        return deleted
