"""CryptoCompare historical OHLC — the long-history free-tier source.

Why this exists: Kraken's free OHLC endpoint returns only the last 720 bars
regardless of `since` (a hard API limit — pagination requires their paid Trade
History endpoint). CryptoCompare's `histoday` endpoint returns 2000 bars per
call with real pagination via `toTs`, giving us BTC/USD from ~2013 to today
without an API key.

    GET https://min-api.cryptocompare.com/data/v2/histoday
        ?fsym=BTC&tsym=USD&limit=2000&toTs=<unix_end_seconds>

Response returns bars ascending in time. To go further back, use the FIRST
returned bar's timestamp minus one day as the next `toTs`. Iterate until
we've covered the requested start.

Rate limits: ~100k requests/month on the free tier — we make ~3 requests for
a decade of daily bars, well within budget.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from ..alpaca_crypto.history import Bar


_ALLOWED_QUOTES = {"USD", "USDT", "USDC", "EUR", "GBP", "JPY", "BTC", "ETH"}


def _split_symbol(pair: str) -> tuple[str, str]:
    """Split 'BTC/USD' into ('BTC', 'USD'). Accepts either slashed or unslashed forms."""
    key = pair.upper()
    if "/" in key:
        base, quote = key.split("/", 1)
        return base, quote
    for q in _ALLOWED_QUOTES:
        if key.endswith(q) and len(key) > len(q):
            return key[:-len(q)], q
    raise ValueError(f"cannot split symbol {pair!r} into base/quote")


class CryptoCompareHistoryClient:
    BASE_URL = "https://min-api.cryptocompare.com"

    def __init__(self, base_url: str | None = None, timeout_s: float = 30.0) -> None:
        self._base_url = base_url or self.BASE_URL
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CryptoCompareHistoryClient":
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
        """Fetch daily bars in [start, end], paginating BACKWARD in time via toTs."""
        if self._client is None:
            raise RuntimeError("CryptoCompareHistoryClient not opened; use `async with`")
        if timeframe != "1Day":
            raise ValueError(
                f"CryptoCompare adapter only supports 1Day (got {timeframe!r}); "
                "extend with /histohour or /histominute if needed"
            )
        fsym, tsym = _split_symbol(symbol)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        to_ts = end_ts

        collected: list[Bar] = []
        # Bounded loop — hard cap to prevent an unbounded fetch if the API misbehaves
        for _ in range(20):
            if to_ts <= start_ts:
                break
            params = {
                "fsym": fsym, "tsym": tsym,
                "limit": 2000, "toTs": to_ts, "aggregate": 1,
            }
            r = await self._client.get("/data/v2/histoday", params=params)
            r.raise_for_status()
            body = r.json()
            if body.get("Response") != "Success":
                raise RuntimeError(
                    f"CryptoCompare error: {body.get('Message', 'unknown')}"
                )
            page_bars = (body.get("Data") or {}).get("Data") or []
            if not page_bars:
                break
            for row in page_bars:
                bar_ts = int(row["time"])
                if bar_ts < start_ts or bar_ts >= end_ts:
                    continue
                # CryptoCompare returns zero-filled placeholder bars for gaps.
                # Skip those so downstream doesn't see fake flat prices.
                if row.get("close", 0) == 0 and row.get("high", 0) == 0:
                    continue
                collected.append(Bar(
                    ts=datetime.fromtimestamp(bar_ts, tz=timezone.utc),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volumefrom", 0.0)),
                ))
            first_ts = int(page_bars[0]["time"])
            if first_ts <= start_ts:
                break
            to_ts = first_ts - 86400   # step back 1 day for the next page

        # Dedupe + sort ascending
        seen: set[datetime] = set()
        deduped: list[Bar] = []
        for b in sorted(collected, key=lambda x: x.ts):
            if b.ts in seen:
                continue
            seen.add(b.ts)
            deduped.append(b)
        return deduped


def _cache_path(cache_dir: Path, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    sym = symbol.replace("/", "-")
    key = f"cryptocompare|{sym}|{timeframe}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"cc_{sym}_{timeframe}_{start.date()}_{end.date()}_{digest}.json"


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
    client: CryptoCompareHistoryClient,
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
