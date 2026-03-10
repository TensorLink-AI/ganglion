"""ACP-specific error types."""

from __future__ import annotations


class ACPError(Exception):
    """Base class for ACP errors."""

    pass


class ACPConnectionError(ACPError):
    """Failed to connect to an ACP server."""

    pass


class ACPRunError(ACPError):
    """ACP run returned an error or timed out."""

    pass


class ACPNotAvailableError(ACPError):
    """The aiohttp package (used for ACP REST calls) is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "ACP support requires the 'aiohttp' package. "
            "Install it with: pip install ganglion[acp]"
        )
