"""Binance historical klines — extended BTC history for the Phase 3 harness.

Alpaca's crypto data starts in 2022. Binance goes back to 2017 for BTC/USDT
(the pair we treat as a USD proxy for backtest purposes). This gives us
enough history for the plan §5 "~27 folds" target.

USDT vs USD caveat: BTC/USDT is what Binance actually trades. USDT holds its
$1 peg within ~30bps most of the time; for daily-bar backtests over years,
this is a negligible approximation. When Phase 5 does per-fill accounting,
we'll switch execution to the appropriate stable-USD pair on Alpaca.

Public endpoint — no API key needed for OHLCV. Rate limit is generous
(1200 req/min IP-weighted); we make at most 4-5 requests for a full history
fetch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from ..alpaca_crypto.history import Bar   # reuse the Bar dataclass


_BINANCE_SYMBOL_MAP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "SOL/USD": "SOLUSDT",
    "ETH/BTC": "ETHBTC",
    "BTC/USDT": "BTCUSDT",   # explicit forms accepted too
    "ETH/USDT": "ETHUSDT",
}

_TIMEFRAME_MAP = {
    "1Day": "1d",
    "1Hour": "1h",
    "1Min": "1m",
}


def binance_symbol(alpaca_pair: str) -> str:
    """Translate an Alpaca-style symbol to Binance's compact form."""
    key = alpaca_pair.upper()
    if key in _BINANCE_SYMBOL_MAP:
        return _BINANCE_SYMBOL_MAP[key]
    # Fallback: strip slash. USD → USDT.
    stripped = key.replace("/", "")
    if stripped.endswith("USD") and not stripped.endswith("USDT"):
        stripped = stripped + "T"
    return stripped


class BinanceHistoryClient:
    BASE_URL = "https://api.binance.com"

    def __init__(self, base_url: str | None = None, timeout_s: float = 30.0) -> None:
        self._base_url = base_url or self.BASE_URL
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BinanceHistoryClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Accept": "application/json"},
            timeout=self._timeout_s,
        )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
        limit_per_page: int = 1000,
    ) -> list[Bar]:
        """Fetch bars in [start, end], following Binance's start/end pagination."""
        if self._client is None:
            raise RuntimeError("BinanceHistoryClient not opened; use `async with`")
        bn_symbol = binance_symbol(symbol)
        bn_interval = _TIMEFRAME_MAP.get(timeframe)
        if bn_interval is None:
            raise ValueError(
                f"unsupported timeframe {timeframe!r} for Binance; "
                f"got {sorted(_TIMEFRAME_MAP)}"
            )
        out: list[Bar] = []
        current_start = start
        while current_start < end:
            params = {
                "symbol": bn_symbol,
                "interval": bn_interval,
                "startTime": int(current_start.timestamp() * 1000),
                "endTime": int(end.timestamp() * 1000),
                "limit": limit_per_page,
            }
            r = await self._client.get("/api/v3/klines", params=params)
            r.raise_for_status()
            raw = r.json()
            if not raw:
                break
            out.extend(_parse_kline(row) for row in raw)
            last_open_ms = int(raw[-1][0])
            last_open = datetime.fromtimestamp(last_open_ms / 1000, tz=timezone.utc)
            # Advance past the last returned bar to avoid re-fetching it
            current_start = last_open + _timeframe_delta(timeframe)
            if len(raw) < limit_per_page:
                break
        return out


def _parse_kline(row: list) -> Bar:
    """Binance kline row layout — see Binance docs."""
    return Bar(
        ts=datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _timeframe_delta(tf: str) -> timedelta:
    return {
        "1Day": timedelta(days=1),
        "1Hour": timedelta(hours=1),
        "1Min": timedelta(minutes=1),
    }[tf]


def _cache_path(cache_dir: Path, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    sym = symbol.replace("/", "-")
    key = f"binance|{sym}|{timeframe}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"binance_{sym}_{timeframe}_{start.date()}_{end.date()}_{digest}.json"


def _atomic_write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


async def load_bars_cached(
    client: BinanceHistoryClient,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    timeframe: str = "1Day",
    force_refresh: bool = False,
) -> list[Bar]:
    """Read cached bars or fetch and cache. Same shape as Alpaca's loader."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, timeframe, start, end)
    if path.exists() and not force_refresh:
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return [Bar(ts=_parse_ts(r["ts"]), open=r["open"], high=r["high"],
                        low=r["low"], close=r["close"], volume=r["volume"]) for r in raw]
        except (json.JSONDecodeError, KeyError, ValueError):
            path.unlink(missing_ok=True)
    bars = await client.bars(symbol, start, end, timeframe)
    serializable = [{**asdict(b), "ts": b.ts.isoformat()} for b in bars]
    _atomic_write_json(path, serializable)
    return bars
