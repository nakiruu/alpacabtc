"""Bitstamp historical OHLC — the last standing no-auth free-tier long-history source.

The landscape as of 2026:
    * Binance.com     — geo-blocked in US
    * Binance.US      — no BTC before 2019
    * Kraken          — free tier capped at 720 bars regardless of `since`
    * CryptoCompare   — free tier now requires API key
    * CoinGecko       — free tier now requires API key
    * Bitstamp        — public endpoints STILL free, no auth (verified 2026)

Bitstamp has been trading BTC/USD since 2011, so we get ~15 years of daily
history. Pagination via `start`/`end`/`limit` is real (not a filter), and
`limit=1000` per request means ~5 calls for a decade of data.

    GET https://www.bitstamp.net/api/v2/ohlc/btcusd/
        ?step=86400&limit=1000&start=<unix_s>&end=<unix_s>

Bitstamp's symbol notation is compact lowercase without separator:
BTC/USD → btcusd, ETH/USD → ethusd, ETH/BTC → ethbtc.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..alpaca_crypto.history import Bar


_BITSTAMP_SYMBOL_MAP = {
    "BTC/USD": "btcusd",
    "ETH/USD": "ethusd",
    "ETH/BTC": "ethbtc",
    "LTC/USD": "ltcusd",
    "XRP/USD": "xrpusd",
    "BCH/USD": "bchusd",
    "SOL/USD": "solusd",
    "ADA/USD": "adausd",
}

_STEP_SECONDS = {
    "1Day": 86400,
    "1Hour": 3600,
    "1Min": 60,
}


def bitstamp_symbol(pair: str) -> str:
    """Translate an Alpaca-style pair to Bitstamp's compact lowercase form."""
    key = pair.upper()
    if key in _BITSTAMP_SYMBOL_MAP:
        return _BITSTAMP_SYMBOL_MAP[key]
    # Fallback: strip slash + lowercase
    return key.replace("/", "").lower()


class BitstampHistoryClient:
    BASE_URL = "https://www.bitstamp.net"

    def __init__(self, base_url: str | None = None, timeout_s: float = 30.0) -> None:
        self._base_url = base_url or self.BASE_URL
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BitstampHistoryClient":
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
    ) -> list[Bar]:
        """Fetch OHLC bars in [start, end], paginating forward via `start`."""
        if self._client is None:
            raise RuntimeError("BitstampHistoryClient not opened; use `async with`")
        bs_symbol = bitstamp_symbol(symbol)
        step = _STEP_SECONDS.get(timeframe)
        if step is None:
            raise ValueError(
                f"unsupported timeframe {timeframe!r} for Bitstamp; "
                f"expected one of {sorted(_STEP_SECONDS)}"
            )

        out: list[Bar] = []
        current_start = int(start.timestamp())
        end_ts = int(end.timestamp())

        # PAGINATION NOTE: passing BOTH `start` AND `end` to Bitstamp returns
        # the LAST `limit` bars ending at `end` — effectively ignoring `start`.
        # We pass only `start` per call and filter end locally to get real
        # forward pagination.
        for _ in range(50):
            if current_start >= end_ts:
                break
            params = {
                "step": step,
                "limit": 1000,
                "start": current_start,
            }
            r = await self._client.get(f"/api/v2/ohlc/{bs_symbol}/", params=params)
            r.raise_for_status()
            body = r.json()
            page_bars = ((body.get("data") or {}).get("ohlc")) or []
            if not page_bars:
                break
            for row in page_bars:
                ts = int(row["timestamp"])
                if ts >= end_ts:
                    continue
                out.append(Bar(
                    ts=datetime.fromtimestamp(ts, tz=timezone.utc),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                ))
            last_ts = int(page_bars[-1]["timestamp"])
            if last_ts <= current_start:
                break
            current_start = last_ts + step
            if len(page_bars) < 1000:
                break

        # Dedupe by ts, sort ascending
        seen: set[datetime] = set()
        deduped: list[Bar] = []
        for b in sorted(out, key=lambda x: x.ts):
            if b.ts in seen:
                continue
            seen.add(b.ts)
            deduped.append(b)
        return deduped


def _cache_path(cache_dir: Path, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    sym = symbol.replace("/", "-")
    key = f"bitstamp|{sym}|{timeframe}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"bs_{sym}_{timeframe}_{start.date()}_{end.date()}_{digest}.json"


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
    client: BitstampHistoryClient,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    timeframe: str = "1Day",
    force_refresh: bool = False,
) -> list[Bar]:
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
