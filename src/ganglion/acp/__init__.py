"""ACP (Agent Communication Protocol) integration for Ganglion.

Provides two bridge directions:
  - ACPClientBridge: discover and invoke remote ACP agents as Ganglion agents
  - ACPServerBridge: expose Ganglion's registered agents as ACP-compatible agents

The Agent Communication Protocol is an open REST-based standard for
agent-to-agent discovery and communication, originally created by IBM's
BeeAI project and now maintained under the Linux Foundation.

Install with: pip install ganglion[acp]
"""

from ganglion.acp.config import ACPClientConfig, ACPServerConfig
from ganglion.acp.errors import (
    ACPConnectionError,
    ACPError,
    ACPNotAvailableError,
    ACPRunError,
)

__all__ = [
    "ACPClientConfig",
    "ACPConnectionError",
    "ACPError",
    "ACPNotAvailableError",
    "ACPRunError",
    "ACPServerConfig",
]
