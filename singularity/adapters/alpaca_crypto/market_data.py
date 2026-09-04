"""Alpaca market data REST client.

Different base URL from the trading API (data.alpaca.markets, not paper-api).
For Phase 2 we need only the latest-quote endpoint; broader coverage (bars,
snapshots, historicals) belongs in Phase 3 harness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx


class MarketDataClient:
    BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str | None = None,
        timeout_s: float = 5.0,
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

    async def __aenter__(self) -> "MarketDataClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def latest_quote(self, symbol: str) -> dict[str, Any] | None:
        """Return {'bid_px', 'bid_sz', 'ask_px', 'ask_sz', 'ts'} for `symbol`, or None."""
        r = await self._client.get(
            "/v1beta3/crypto/us/latest/quotes", params={"symbols": symbol}
        )
        r.raise_for_status()
        quotes = r.json().get("quotes") or {}
        q = quotes.get(symbol)
        if q is None:
            return None
        ts_str = q.get("t")
        ts = _parse_ts(ts_str) if ts_str else datetime.now(timezone.utc)
        return {
            "bid_px": float(q["bp"]),
            "bid_sz": float(q["bs"]),
            "ask_px": float(q["ap"]),
            "ask_sz": float(q["as"]),
            "ts": ts,
        }

    async def latest_trade(self, symbol: str) -> dict[str, Any] | None:
        r = await self._client.get(
            "/v1beta3/crypto/us/latest/trades", params={"symbols": symbol}
        )
        r.raise_for_status()
        trades = r.json().get("trades") or {}
        t = trades.get(symbol)
        if t is None:
            return None
        return {
            "price": float(t["p"]),
            "size": float(t["s"]),
            "ts": _parse_ts(t["t"]),
        }


def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)
