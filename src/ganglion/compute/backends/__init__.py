"""Built-in compute backend implementations.

Use the registry for lazy, dependency-free discovery::

    from ganglion.compute.backends import get_backend_registry

    registry = get_backend_registry()
    registry.available()        # ["local", "runpod", "ssh", ...]
    registry.check("runpod")    # (True, None)  or  (False, "pip install ganglion[runpod]")
    backend = registry.create("local")
"""

from ganglion.compute.backends.registry import BackendRegistry, get_backend_registry

__all__ = ["BackendRegistry", "get_backend_registry"]
