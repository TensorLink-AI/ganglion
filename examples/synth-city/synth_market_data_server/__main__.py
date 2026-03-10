"""Entry point for ``python -m synth_market_data_server``.

Starts an MCP server (stdio transport) that exposes the Synth City
market-data tools: fetch_price and fetch_historical_prices.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

# Ensure the synth-city example root is importable so that the tools/
# modules can resolve ganglion imports when the project is installed.
_example_root = Path(__file__).resolve().parent.parent
if str(_example_root) not in sys.path:
    sys.path.insert(0, str(_example_root))


def main() -> None:
    from ganglion.mcp.server import MCPServerBridge
    from ganglion.state.tool_registry import ToolRegistry

    registry = ToolRegistry()

    # Register the market-data tools from the tools/ directory.
    tools_dir = _example_root / "tools"
    registry.register_from_file(tools_dir / "fetch_price.py")
    registry.register_from_file(tools_dir / "fetch_historical_prices.py")

    bridge = MCPServerBridge(
        tool_registry=registry,
        server_name="synth-market-data",
        categories=["data"],
    )
    asyncio.run(bridge.run_stdio())


if __name__ == "__main__":
    main()
