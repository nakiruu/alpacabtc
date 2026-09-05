"""Kraken historical OHLC — recent-bars source only.

**IMPORTANT LIMITATION**: Kraken's public OHLC endpoint returns at most 720
bars per call and pagination does NOT actually walk backward — the `since`
parameter is a filter, not a cursor. In practice you'll get the last ~720
bars regardless of the `since` value you send. For daily bars that's ~2 years
of history, not the decade you might want for Phase 4 gate evaluation.

Going further back requires Kraken's paid Trade History endpoint (needs an
API key and account). For free-tier extended history, use
`singularity.adapters.cryptocompare` instead.

    GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1440&since=<unix_s>

Symbol naming is quirky — Kraken has legacy prefixes (XXBTZUSD) and modern
short forms (XBTUSD). We use the short form on request; response key may still
come back in either form, so we grab whichever key isn't "last".
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from ..alpaca_crypto.history import Bar


_KRAKEN_SYMBOL_MAP = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "ETH/BTC": "ETHXBT",
    "SOL/USD": "SOLUSD",
    "XBT/USD": "XBTUSD",
}

_INTERVAL_MINUTES = {
    "1Day": 1440,
    "1Hour": 60,
    "1Min": 1,
}


def kraken_symbol(pair: str) -> str:
    """Translate an Alpaca-style pair to Kraken's compact form."""
    key = pair.upper()
    if key in _KRAKEN_SYMBOL_MAP:
        return _KRAKEN_SYMBOL_MAP[key]
    # Fallback: strip slash; Kraken uses XBT for BTC
    return key.replace("/", "").replace("BTC", "XBT")


class KrakenHistoryClient:
    BASE_URL = "https://api.kraken.com"

    def __init__(self, base_url: str | None = None, timeout_s: float = 30.0) -> None:
        self._base_url = base_url or self.BASE_URL
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "KrakenHistoryClient":
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
        """Fetch OHLC bars in [start, end], paginating via `last`."""
        if self._client is None:
            raise RuntimeError("KrakenHistoryClient not opened; use `async with`")
        kraken_pair = kraken_symbol(symbol)
        interval = _INTERVAL_MINUTES.get(timeframe)
        if interval is None:
            raise ValueError(
                f"unsupported timeframe {timeframe!r} for Kraken; "
                f"expected one of {sorted(_INTERVAL_MINUTES)}"
            )

        out: list[Bar] = []
        end_ts = int(end.timestamp())
        since = int(start.timestamp())
        # Kraken's `since` is exclusive-of-`since`, so subtract one interval
        # to include the first bar at or after `start`.
        since -= interval * 60

        while since < end_ts:
            params = {"pair": kraken_pair, "interval": interval, "since": since}
            r = await self._client.get("/0/public/OHLC", params=params)
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                raise RuntimeError(f"Kraken API error: {body['error']}")
            result = body.get("result") or {}
            # Response dict has one key for the pair (which may differ from what
            # we sent) plus a "last" key. Pick whichever key isn't "last".
            pair_data = None
            for key, val in result.items():
                if key != "last":
                    pair_data = val
                    break
            if not pair_data:
                break

            for row in pair_data:
                ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
                if ts >= end:
                    continue
                out.append(Bar(
                    ts=ts,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[6]),   # index 5 is VWAP, 6 is volume
                ))

            last_ts = result.get("last")
            if last_ts is None or int(last_ts) <= since:
                break
            since = int(last_ts)

        # Kraken can return duplicates across pages; dedupe by ts
        seen: set[datetime] = set()
        deduped: list[Bar] = []
        for b in out:
            if b.ts in seen:
                continue
            seen.add(b.ts)
            deduped.append(b)
        deduped.sort(key=lambda b: b.ts)

        # If the returned range is far shorter than requested, warn loudly —
        # Kraken's 720-bar cap silently drops earlier data.
        if deduped:
            requested_days = (end - start).days
            got_days = (deduped[-1].ts - deduped[0].ts).days
            if requested_days > got_days * 1.5 and requested_days > 30:
                warnings.warn(
                    f"Kraken returned {len(deduped)} bars covering "
                    f"{deduped[0].ts.date()} → {deduped[-1].ts.date()} "
                    f"({got_days}d), but you requested {start.date()} → {end.date()} "
                    f"({requested_days}d). Kraken's free OHLC endpoint caps at ~720 "
                    "bars. Use --data-source cryptocompare for full history.",
                    stacklevel=2,
                )
        return deduped


def _cache_path(cache_dir: Path, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    sym = symbol.replace("/", "-")
    key = f"kraken|{sym}|{timeframe}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"kraken_{sym}_{timeframe}_{start.date()}_{end.date()}_{digest}.json"


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
    client: KrakenHistoryClient,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    timeframe: str = "1Day",
    force_refresh: bool = False,
) -> list[Bar]:
    """Read cached bars or fetch and cache. Same shape as Alpaca/Binance loaders."""
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
