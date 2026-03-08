"""Fetch historical price data from the Pyth Benchmarks API.

The SN50 validator uses the Pyth Benchmarks TradingView shim to fetch
realised prices at 1-minute resolution.  This tool exposes the same API
so miners can pull historical data for volatility calibration, model
training, and local backtesting.

Upstream reference:
  https://github.com/mode-network/synth-subnet/blob/main/synth/validator/price_data_provider.py

API endpoint:
  https://benchmarks.pyth.network/v1/shims/tradingview/history
  Query params: symbol, resolution, from (unix), to (unix)
  Response:     {"t": [timestamps], "c": [close_prices], "s": "ok"}
"""

import json
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from ganglion.composition.tool_registry import tool
from ganglion.composition.tool_returns import ExperimentResult

# Same token map the SN50 validator uses (Pyth Benchmarks symbols)
PYTH_BENCHMARKS_TOKEN_MAP: dict[str, str] = {
    "BTC": "Crypto.BTC/USD",
    "ETH": "Crypto.ETH/USD",
    "SOL": "Crypto.SOL/USD",
    "XAU": "Crypto.XAUT/USD",
    "SPYX": "Crypto.SPYX/USD",
    "NVDAX": "Crypto.NVDAX/USD",
    "TSLAX": "Crypto.TSLAX/USD",
    "AAPLX": "Crypto.AAPLX/USD",
    "GOOGLX": "Crypto.GOOGLX/USD",
}

BENCHMARKS_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"


def _fetch_raw(symbol: str, from_ts: int, to_ts: int, resolution: int = 1) -> dict[str, Any]:
    """Hit the Pyth Benchmarks TradingView history endpoint."""
    params = (
        f"symbol={symbol}&resolution={resolution}"
        f"&from={from_ts}&to={to_ts}"
    )
    url = f"{BENCHMARKS_URL}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _transform_to_series(
    raw: dict[str, Any],
    from_ts: int,
    to_ts: int,
    time_increment: int,
) -> tuple[list[int], list[float | None]]:
    """Align raw API data to a regular grid, filling gaps with None.

    This mirrors the validator's _transform_data() which fills gaps with
    np.nan.  Returns (timestamps, prices) where prices[i] is None if the
    API had no data for that slot.
    """
    raw_timestamps = raw.get("t", [])
    raw_closes = raw.get("c", [])

    # Build lookup: timestamp -> close price
    price_lookup: dict[int, float] = {}
    for ts, price in zip(raw_timestamps, raw_closes):
        price_lookup[int(ts)] = float(price)

    # Generate expected grid
    timestamps = []
    prices: list[float | None] = []
    t = from_ts
    while t <= to_ts:
        timestamps.append(t)
        # Try exact match first, then ±30 s window (API timestamps can drift)
        p = price_lookup.get(t)
        if p is None:
            for offset in range(-30, 31):
                p = price_lookup.get(t + offset)
                if p is not None:
                    break
        prices.append(p)
        t += time_increment

    return timestamps, prices


@tool("fetch_historical_prices", category="data")
def fetch_historical_prices(config: dict) -> ExperimentResult:
    """Fetch historical price data from the Pyth Benchmarks API.

    Uses the same API and symbols the SN50 validator uses to fetch
    realised prices for scoring.  Returns a time-aligned price series
    suitable for backtesting, volatility estimation, and model training.

    Expected config keys:
        asset: str              — asset symbol (default "BTC")
                                  Supported: BTC ETH SOL XAU SPYX NVDAX
                                  TSLAX AAPLX GOOGLX
        hours_back: int         — how many hours of history (default 24)
        time_increment: int     — seconds between points (default 300)
                                  Use 300 for low-freq, 60 for high-freq
        end_time: int | None    — unix timestamp for end (default: now)

    Returns:
        metrics.timestamps  — list of unix timestamps
        metrics.prices      — list of prices (null for gaps)
        metrics.returns_bps — list of pct returns in basis points
                              (same scale the validator uses for CRPS)
        metrics.gap_count   — number of missing data points
        metrics.asset       — asset queried
    """
    asset = config.get("asset", "BTC")
    hours_back = config.get("hours_back", 24)
    time_increment = config.get("time_increment", 300)
    end_time = config.get("end_time")

    symbol = PYTH_BENCHMARKS_TOKEN_MAP.get(asset.upper())
    if not symbol:
        supported = ", ".join(sorted(PYTH_BENCHMARKS_TOKEN_MAP))
        return ExperimentResult(
            content=f"Unknown asset '{asset}'.  Supported: {supported}",
            metrics={},
        )

    to_ts = int(end_time) if end_time else int(time.time())
    from_ts = to_ts - (hours_back * 3600)

    try:
        raw = _fetch_raw(symbol, from_ts, to_ts)
    except Exception as exc:
        return ExperimentResult(
            content=f"Failed to fetch history for {asset}: {exc}",
            metrics={},
        )

    status = raw.get("s", "")
    if status != "ok":
        return ExperimentResult(
            content=f"API returned status='{status}' for {asset}.  No data available.",
            metrics={},
        )

    timestamps, prices = _transform_to_series(raw, from_ts, to_ts, time_increment)

    # Compute returns in basis points (matching validator scoring)
    import numpy as np

    valid_prices = [p for p in prices if p is not None]
    returns_bps: list[float] = []
    if len(valid_prices) >= 2:
        arr = np.array(valid_prices)
        raw_returns = np.diff(arr) / arr[:-1]
        returns_bps = (raw_returns * 10_000).tolist()

    gap_count = sum(1 for p in prices if p is None)
    total_points = len(prices)

    start_dt = datetime.fromtimestamp(from_ts, tz=timezone.utc).isoformat()
    end_dt = datetime.fromtimestamp(to_ts, tz=timezone.utc).isoformat()

    return ExperimentResult(
        content=(
            f"Fetched {total_points} price points for {asset} "
            f"({hours_back}h, {time_increment}s increments).  "
            f"Range: {start_dt} → {end_dt}.  "
            f"Gaps: {gap_count}/{total_points}.  "
            f"Valid prices: {len(valid_prices)}.  "
            f"Returns (bps): {len(returns_bps)} observations."
        ),
        experiment_id=f"hist-{asset.lower()}-{hours_back}h",
        metrics={
            "timestamps": timestamps,
            "prices": prices,
            "returns_bps": returns_bps,
            "gap_count": gap_count,
            "total_points": total_points,
            "valid_points": len(valid_prices),
            "asset": asset.upper(),
            "time_increment": time_increment,
            "from_ts": from_ts,
            "to_ts": to_ts,
        },
    )
