"""Fetch live asset prices from the Pyth Hermes oracle.

SN50 miners need current spot prices to anchor their simulations.
The upstream repo uses the Pyth Network oracle for BTC, ETH, SOL, XAU,
and tokenised equities.

Reference:
  https://github.com/mode-network/synth-subnet/blob/main/synth/miner/price_simulation.py
"""

import json
import urllib.request
from typing import Any

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ToolOutput

# Pyth price-feed IDs (mainnet)
PYTH_FEED_IDS: dict[str, str] = {
    "BTC": "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    "XAU": "765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63c52ece24afb0e1582",
}

HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest"


@tool("fetch_price", category="data")
def fetch_price(asset: str) -> ToolOutput:
    """Fetch the current spot price for an asset from Pyth Hermes.

    Supported assets: BTC, ETH, SOL, XAU.
    Returns the parsed price and a confidence interval.
    """
    feed_id = PYTH_FEED_IDS.get(asset.upper())
    if not feed_id:
        return ToolOutput(
            content=f"Unknown asset '{asset}'.  Supported: {', '.join(PYTH_FEED_IDS)}"
        )

    url = f"{HERMES_URL}?ids[]={feed_id}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data: dict[str, Any] = json.loads(resp.read())

        parsed = data.get("parsed", [{}])[0]
        price_data = parsed.get("price", {})
        price = int(price_data.get("price", 0)) * (10 ** int(price_data.get("expo", 0)))
        conf = int(price_data.get("conf", 0)) * (10 ** int(price_data.get("expo", 0)))

        return ToolOutput(
            content=(
                f"{asset.upper()} spot price: ${price:,.2f} "
                f"(± ${conf:,.2f} confidence).  "
                f"Source: Pyth Hermes oracle."
            )
        )
    except Exception as exc:
        return ToolOutput(content=f"Failed to fetch {asset} price: {exc}")
