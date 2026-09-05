"""Bitstamp historical OHLC — the last standing no-auth free-tier long-history source.

The landscape as of 2026:
    * Binance.com     — geo-blocked in US
    * Binance.US      — no BTC before 2019
    * Kraken          — free tier capped at 720 bars regardless of `since`
    * CryptoCompare   — free tier now requires API key
    * CoinGecko       — free tier now requires API key
    * Bitstamp        — public endpoints STILL free, no auth (verified 2026)

Timeframes and typical fetch sizes:
    1Day   → ~15 years of history = ~5,500 bars = ~6 API pages
    1Hour  → 2 years of history   = ~17,500 bars = ~18 pages
    1Min   → 2 years of history   = ~1.05M bars = ~1050 pages (~2-5 min at
             100ms courtesy delay). Cache once, then reads are instant.

    GET https://www.bitstamp.net/api/v2/ohlc/btcusd/
        ?step=<seconds>&limit=1000&start=<unix_s>

Bitstamp's symbol notation is compact lowercase without separator:
BTC/USD → btcusd, ETH/USD → ethusd, ETH/BTC → ethbtc.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..alpaca_crypto.history import Bar
from ...logs import get_logger

log = get_logger(__name__)


# Courtesy delay between paginated requests. Bitstamp doesn't document a
# strict rate limit for public endpoints, but ~100 ms between calls keeps
# us well under any sane threshold and safe from IP throttling if their
# policy changes.
_PAGE_THROTTLE_S = 0.10

# Per-page bar limit that Bitstamp enforces
_PAGE_LIMIT = 1000

# Max pages we'll walk before bailing. At 1000 bars/page:
#   * 1Day  → 3000 pages = ~8000 years (defensive infinity)
#   * 1Hour → 3000 pages = ~340 years   (defensive infinity)
#   * 1Min  → 3000 pages = ~5.7 years   (right-sized)
# The loop exits earlier when either `end` is reached or a partial page
# indicates end-of-data. 3000 is the pathological-runaway backstop.
_MAX_PAGES = 3000

# Log progress every N pages during long fetches so the user knows the
# process is alive
_PROGRESS_EVERY_PAGES = 20


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
        pages_fetched = 0
        for page_i in range(_MAX_PAGES):
            if current_start >= end_ts:
                break
            params = {
                "step": step,
                "limit": _PAGE_LIMIT,
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
            pages_fetched += 1

            # Progress logging for long intraday fetches
            if pages_fetched % _PROGRESS_EVERY_PAGES == 0:
                progress_pct = min(
                    100.0,
                    (current_start - int(start.timestamp()))
                    / max(1, end_ts - int(start.timestamp())) * 100.0,
                )
                log.info(
                    "bitstamp_pagination_progress",
                    symbol=symbol, timeframe=timeframe,
                    pages=pages_fetched, bars_so_far=len(out),
                    approx_progress_pct=round(progress_pct, 1),
                )

            if len(page_bars) < _PAGE_LIMIT:
                break

            # Courtesy throttle — do NOT delay on the FIRST page (fast path
            # for small ranges) or when we're about to exit anyway.
            await asyncio.sleep(_PAGE_THROTTLE_S)

        if pages_fetched >= _MAX_PAGES:
            log.warning(
                "bitstamp_pagination_hit_max",
                symbol=symbol, timeframe=timeframe, max_pages=_MAX_PAGES,
                note="range may be truncated; consider narrowing start/end",
            )

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
