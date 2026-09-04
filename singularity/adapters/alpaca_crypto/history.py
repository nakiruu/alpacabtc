"""Alpaca crypto historical bars — for Phase 3 backtests.

    GET https://data.alpaca.markets/v1beta3/crypto/us/bars
        ?symbols=BTC/USD&timeframe=1Day&start=2023-01-01&end=2026-01-01

Response is paginated via `next_page_token`. We follow the token to completion.

Local cache: bars are written to `state/bars/<SYMBOL_slugified>_<TIMEFRAME>.json`
so a repeat backtest doesn't re-hammer the API. Cache is byte-identical between
runs — safe to inspect or gitignore.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoryClient:
    BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        # Store config only; open the httpx client in __aenter__ so a construct-
        # without-context-manager path can't leak an open connection pool.
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url or self.BASE_URL
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "HistoryClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
                "Accept": "application/json",
            },
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
        limit_per_page: int = 10_000,
    ) -> list[Bar]:
        """Fetch all bars in [start, end], following pagination. Returns sorted by ts."""
        if self._client is None:
            raise RuntimeError("HistoryClient not opened; use `async with HistoryClient(...) as c`")
        out: list[Bar] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": _rfc3339(start),
                "end": _rfc3339(end),
                "limit": limit_per_page,
            }
            if page_token:
                params["page_token"] = page_token
            r = await self._client.get("/v1beta3/crypto/us/bars", params=params)
            r.raise_for_status()
            body = r.json()
            raw_bars = (body.get("bars") or {}).get(symbol) or []
            out.extend(_parse_bar(b) for b in raw_bars)
            page_token = body.get("next_page_token")
            if not page_token:
                break
        out.sort(key=lambda x: x.ts)
        return out


def _rfc3339(dt: datetime) -> str:
    """RFC3339 with microseconds, always UTC. Preserves precision Alpaca can use."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_bar(b: dict) -> Bar:
    return Bar(
        ts=_parse_ts(b["t"]),
        open=float(b["o"]),
        high=float(b["h"]),
        low=float(b["l"]),
        close=float(b["c"]),
        volume=float(b["v"]),
    )


def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _cache_path(cache_dir: Path, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    """Cache key includes full ISO timestamps hashed short so intraday-varying inputs
    don't collide onto a single file, but the human-readable prefix stays useful."""
    sym = symbol.replace("/", "-")
    key = f"{sym}|{timeframe}|{_rfc3339(start)}|{_rfc3339(end)}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"{sym}_{timeframe}_{start.date()}_{end.date()}_{digest}.json"


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write to a sibling tempfile then rename. Prevents corrupt cache from a crash mid-write."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


async def load_bars_cached(
    client: HistoryClient,
    symbol: str,
    start: datetime,
    end: datetime,
    cache_dir: Path,
    timeframe: str = "1Day",
    force_refresh: bool = False,
) -> list[Bar]:
    """Load bars from local cache if present, else fetch and cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, symbol, timeframe, start, end)
    if path.exists() and not force_refresh:
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            return [Bar(ts=_parse_ts(r["ts"]), open=r["open"], high=r["high"],
                        low=r["low"], close=r["close"], volume=r["volume"]) for r in raw]
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupt or incompatible cache — fall through to refetch.
            path.unlink(missing_ok=True)
    bars = await client.bars(symbol, start, end, timeframe)
    serializable = [{**asdict(b), "ts": b.ts.isoformat()} for b in bars]
    _atomic_write_json(path, serializable)
    return bars
