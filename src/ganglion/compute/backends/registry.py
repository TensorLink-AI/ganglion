"""Backend registry — lazy discovery and instantiation of compute providers.

The key design principle: provider-specific dependencies (aiohttp, asyncssh, etc.)
are only imported when a backend is actually instantiated. This means you can
``pip install ganglion`` with zero extra deps and still use LocalBackend, then
``pip install ganglion[runpod]`` only when you need RunPod.

Third-party packages can register backends via the ``ganglion.compute.backends``
entry-point group (see pyproject.toml for the built-in examples).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from ganglion.compute.protocol import ComputeBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LazyBackendEntry:
    """Metadata for a backend that hasn't been imported yet."""

    module: str  # e.g. "ganglion.compute.backends.runpod"
    class_name: str  # e.g. "RunPodBackend"
    config_class_name: str | None = None  # e.g. "RunPodConfig"


# Built-in backends — add new first-party providers here.
_BUILTIN_BACKENDS: dict[str, _LazyBackendEntry] = {
    "local": _LazyBackendEntry(
        module="ganglion.compute.backends.local",
        class_name="LocalBackend",
    ),
    "runpod": _LazyBackendEntry(
        module="ganglion.compute.backends.runpod",
        class_name="RunPodBackend",
        config_class_name="RunPodConfig",
    ),
    "ssh": _LazyBackendEntry(
        module="ganglion.compute.backends.ssh",
        class_name="SSHBackend",
        config_class_name="SSHConfig",
    ),
}


class BackendRegistry:
    """Discovers, validates, and lazily instantiates compute backends.

    Usage::

        registry = BackendRegistry()

        # See what's available (no imports yet)
        registry.available()          # -> ["local", "runpod", "ssh", ...]
        registry.check("runpod")      # -> (True, None)  or  (False, "pip install ganglion[runpod]")

        # Instantiate only when needed
        backend = registry.create("local")
        backend = registry.create("runpod", api_key="...", preferred_gpu="A100")
    """

    def __init__(self) -> None:
        self._entries: dict[str, _LazyBackendEntry] = dict(_BUILTIN_BACKENDS)
        self._extras: dict[str, type[ComputeBackend]] = {}  # runtime-registered classes
        self._discover_entrypoints()

    # ── Public API ───────────────────────────────────────────

    def available(self) -> list[str]:
        """Return names of all known backends (built-in + plugins)."""
        return sorted(set(self._entries) | set(self._extras))

    def check(self, name: str) -> tuple[bool, str | None]:
        """Check whether *name* can be instantiated right now.

        Returns ``(True, None)`` if all dependencies are importable, or
        ``(False, hint)`` with a human-readable install hint.
        """
        if name in self._extras:
            return True, None

        entry = self._entries.get(name)
        if entry is None:
            return False, f"Unknown backend '{name}'. Known: {', '.join(self.available())}"

        try:
            importlib.import_module(entry.module)
            return True, None
        except ImportError as exc:
            hint = f"pip install ganglion[{name}]" if name != "local" else str(exc)
            return False, hint

    def create(self, backend_name: str, **kwargs: Any) -> ComputeBackend:
        """Instantiate a backend by name, passing *kwargs* to its constructor.

        For backends with a config dataclass (RunPodConfig, SSHConfig, …) you
        can either pass a pre-built config object as ``config=…`` or pass the
        config fields directly::

            registry.create("runpod", api_key="sk-...")
            # is equivalent to
            registry.create("runpod", config=RunPodConfig(api_key="sk-..."))
        """
        name = backend_name
        # Runtime-registered class takes priority
        if name in self._extras:
            return self._extras[name](**kwargs)

        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(f"Unknown backend '{name}'. Known: {', '.join(self.available())}")

        mod = self._import_module(entry.module, name)
        cls = getattr(mod, entry.class_name)

        # Auto-wrap kwargs into a config dataclass when the backend expects one
        if entry.config_class_name and "config" not in kwargs:
            config_cls = getattr(mod, entry.config_class_name)
            # Split kwargs into config fields and backend-level fields
            import dataclasses

            config_fields = {f.name for f in dataclasses.fields(config_cls)}
            config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
            rest_kwargs = {k: v for k, v in kwargs.items() if k not in config_fields}
            if config_kwargs:
                rest_kwargs["config"] = config_cls(**config_kwargs)
            return cls(**rest_kwargs)

        return cls(**kwargs)

    def register(self, name: str, cls: type[ComputeBackend]) -> None:
        """Register a backend class at runtime (e.g. from a plugin or test)."""
        self._extras[name] = cls

    def unregister(self, name: str) -> None:
        """Remove a runtime-registered backend."""
        self._extras.pop(name, None)

    # ── Internals ────────────────────────────────────────────

    def _discover_entrypoints(self) -> None:
        """Load third-party backends from ``ganglion.compute.backends`` entry-points."""
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            # Python 3.12+ returns a SelectableGroups; 3.11 returns a dict
            group = (
                eps.select(group="ganglion.compute.backends")
                if hasattr(eps, "select")
                else eps.get("ganglion.compute.backends", [])
            )
            for ep in group:
                # Don't overwrite builtins — they carry extra metadata
                if ep.name not in self._entries:
                    self._entries[ep.name] = _LazyBackendEntry(
                        module=ep.value.rsplit(":", 1)[0],
                        class_name=ep.value.rsplit(":", 1)[1],
                    )
        except Exception:
            # Entry-point discovery is best-effort
            logger.debug("Entry-point discovery failed", exc_info=True)

    @staticmethod
    def _import_module(module_path: str, backend_name: str) -> Any:
        """Import a module, raising a clear error on missing dependencies."""
        try:
            return importlib.import_module(module_path)
        except ImportError as exc:
            install_hint = (
                f"pip install ganglion[{backend_name}]" if backend_name != "local" else str(exc)
            )
            raise ImportError(
                f"Cannot load backend '{backend_name}': missing dependency. "
                f"Install with: {install_hint}"
            ) from exc


# Module-level singleton for convenience
_default_registry: BackendRegistry | None = None


def get_backend_registry() -> BackendRegistry:
    """Return the module-level BackendRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = BackendRegistry()
    return _default_registry
