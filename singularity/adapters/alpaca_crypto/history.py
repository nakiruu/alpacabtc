"""Alpaca crypto historical bars — for Phase 3 backtests.

    GET https://data.alpaca.markets/v1beta3/crypto/us/bars
        ?symbols=BTC/USD&timeframe=1Day&start=2023-01-01&end=2026-01-01

Response is paginated via `next_page_token`. We follow the token to completion.

Local cache: bars are written to `state/bars/<SYMBOL_slugified>_<TIMEFRAME>.json`
so a repeat backtest doesn't re-hammer the API. Cache is byte-identical between
runs — safe to inspect or gitignore.
"""

from __future__ import annotations

import asyncio
import json
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
        self._client = httpx.AsyncClient(
            base_url=base_url or self.BASE_URL,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Accept": "application/json",
            },
            timeout=timeout_s,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HistoryClient":
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
        out: list[Bar] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": limit_per_page,
            }
            if page_token:
                params["page_token"] = page_token
            r = await self._client.get("/v1beta3/crypto/us/bars", params=params)
            r.raise_for_status()
            body = r.json()
            raw_bars = (body.get("bars") or {}).get(symbol) or []
            for b in raw_bars:
                out.append(_parse_bar(b))
            page_token = body.get("next_page_token")
            if not page_token:
                break
        out.sort(key=lambda x: x.ts)
        return out


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
    sym = symbol.replace("/", "-")
    return cache_dir / f"{sym}_{timeframe}_{start.date()}_{end.date()}.json"


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
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Bar(ts=_parse_ts(r["ts"]), open=r["open"], high=r["high"],
                    low=r["low"], close=r["close"], volume=r["volume"]) for r in raw]
    bars = await client.bars(symbol, start, end, timeframe)
    serializable = [{**asdict(b), "ts": b.ts.isoformat()} for b in bars]
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f)
    return bars
