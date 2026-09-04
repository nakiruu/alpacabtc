"""Volatility features.

  * `atr_from_influx` — Wilder ATR-14 from Phase 0 trade tape (bracket supervisor)
  * `realized_vol_annualized` — rolling stdev of daily returns × sqrt(365)
    (Phase 4 vol-target overlay)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from influxdb_client import InfluxDBClient


BARS_PER_YEAR_DAILY = 365   # crypto is 24/7


def realized_vol_annualized(returns: list[float], lookback: int) -> list[float]:
    """Rolling annualized realized vol at each bar via stdev × sqrt(365).

    Length matches `returns`. First `lookback - 1` entries are 0.0 (no window yet).
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    out = [0.0] * len(returns)
    factor = math.sqrt(BARS_PER_YEAR_DAILY)
    for i in range(lookback - 1, len(returns)):
        window = returns[i - lookback + 1:i + 1]
        sd = statistics.stdev(window)
        out[i] = sd * factor
    return out


@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float


def _bucket_trades_to_bars(
    trades: list[tuple[datetime, float]], bar_minutes: int
) -> list[Bar]:
    """Aggregate (ts, price) trades into fixed-window OHLC bars.

    Bars are keyed by the START of their minute window; an empty minute produces
    no bar (we don't forward-fill — the bracket supervisor treats missing bars
    as "insufficient data" and abstains).
    """
    if not trades:
        return []
    bucket: dict[datetime, list[float]] = {}
    for ts, px in trades:
        # Truncate to the start of the bar window
        floored = ts.replace(
            minute=(ts.minute // bar_minutes) * bar_minutes,
            second=0,
            microsecond=0,
        )
        bucket.setdefault(floored, []).append(px)
    out: list[Bar] = []
    for ts in sorted(bucket):
        prices = bucket[ts]
        out.append(Bar(ts=ts, open=prices[0], high=max(prices), low=min(prices), close=prices[-1]))
    return out


def _true_range(bar: Bar, prev_close: float | None) -> float:
    if prev_close is None:
        return bar.high - bar.low
    return max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))


def wilder_atr(bars: list[Bar], n: int = 14) -> float | None:
    """Wilder-smoothed ATR over `n` periods. Needs at least n+1 bars."""
    if len(bars) < n + 1:
        return None
    trs = [_true_range(bars[i], bars[i - 1].close if i > 0 else None) for i in range(len(bars))]
    # First ATR = simple mean of first n TRs
    atr = sum(trs[:n]) / n
    # Wilder smoothing for the rest
    for tr in trs[n:]:
        atr = (atr * (n - 1) + tr) / n
    return atr


async def atr_from_influx(
    client: "InfluxDBClient",
    bucket: str,
    org: str,
    symbol: str,
    n_periods: int = 14,
    bar_minutes: int = 1,
) -> float | None:
    """Query recent trades, bucket into `bar_minutes` bars, return Wilder ATR-`n_periods`.

    Returns None if not enough data is present in Influx to compute the metric.
    """
    lookback_min = (n_periods + 5) * bar_minutes  # a few extra bars for safety
    flux = f'''
    from(bucket: "{bucket}")
      |> range(start: -{lookback_min}m)
      |> filter(fn: (r) => r._measurement == "crypto_trade" and r.symbol == "{symbol}" and r._field == "price")
      |> keep(columns: ["_time", "_value"])
      |> sort(columns: ["_time"])
    '''
    import asyncio
    tables = await asyncio.to_thread(client.query_api().query, flux, org=org)
    trades: list[tuple[datetime, float]] = []
    for tbl in tables:
        for rec in tbl.records:
            trades.append((rec.get_time(), float(rec.get_value())))
    bars = _bucket_trades_to_bars(trades, bar_minutes)
    return wilder_atr(bars, n_periods)
