"""CoinGecko historical prices — no-auth free-tier long-history source.

CryptoCompare's `/data/v2/histoday` now returns 401 without an API key (their
free tier tightened). CoinGecko's `/coins/{id}/market_chart/range` still works
anonymously and covers BTC daily prices back to 2013.

**Caveat**: this endpoint returns close prices only, no OHLC. We synthesize
Bar objects with open=high=low=close and volume=0. The backtest harness only
uses `close` for its return math (TSMOM signal, vol overlay), so this is
functionally lossless for our purposes. If you later need real OHLC on
long history, either sign up for a CryptoCompare API key or use Alpaca /
Binance-US where the range is short enough to not need extended history.

    GET https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range
        ?vs_currency=usd&from=<unix_s>&to=<unix_s>

Response: {"prices": [[ts_ms, close], ...], "market_caps": [...], "total_volumes": [...]}
Granularity: ranges > 7 days return daily; ranges <= 7 days return hourly.
Rate limit: ~10-30 calls/minute on the free tier. We make ONE call per fetch.
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


# CoinGecko coin IDs for common pairs
_COIN_ID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "XRP": "ripple",
    "LTC": "litecoin",
    "AVAX": "avalanche-2",
    "MATIC": "matic-network",
    "DOT": "polkadot",
}


def _resolve_pair(symbol: str) -> tuple[str, str]:
    """Split a pair like 'BTC/USD' into (coin_id, vs_currency)."""
    if "/" not in symbol:
        raise ValueError(f"expected 'BASE/QUOTE' form, got {symbol!r}")
    base, quote = symbol.upper().split("/", 1)
    if base not in _COIN_ID_MAP:
        raise ValueError(
            f"no CoinGecko coin_id mapping for base {base!r}; "
            f"add it to _COIN_ID_MAP or use --data-source alpaca"
        )
    coin_id = _COIN_ID_MAP[base]
    # CoinGecko accepts 'usd', 'eur', 'btc', etc as lowercase currency codes
    vs = quote.lower()
    if vs == "usdt":
        vs = "usd"   # CoinGecko treats USDT prices via USD peg
    return coin_id, vs


class CoinGeckoHistoryClient:
    BASE_URL = "https://api.coingecko.com"

    def __init__(self, base_url: str | None = None, timeout_s: float = 30.0) -> None:
        self._base_url = base_url or self.BASE_URL
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CoinGeckoHistoryClient":
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
        """Fetch daily close prices in [start, end]. Single request per fetch."""
        if self._client is None:
            raise RuntimeError("CoinGeckoHistoryClient not opened; use `async with`")
        if timeframe != "1Day":
            raise ValueError(
                f"CoinGecko adapter only supports 1Day (got {timeframe!r})"
            )
        coin_id, vs = _resolve_pair(symbol)
        params = {
            "vs_currency": vs,
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        }
        r = await self._client.get(f"/api/v3/coins/{coin_id}/market_chart/range", params=params)
        r.raise_for_status()
        body = r.json()
        raw_prices = body.get("prices") or []

        bars: list[Bar] = []
        for entry in raw_prices:
            if len(entry) < 2:
                continue
            ts_ms, close = entry
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            # Normalize to midnight UTC — CoinGecko returns bars at 00:00 for daily
            close_f = float(close)
            if close_f <= 0:
                continue
            bars.append(Bar(
                ts=ts, open=close_f, high=close_f, low=close_f, close=close_f,
                volume=0.0,
            ))
        bars.sort(key=lambda b: b.ts)
        # Dedupe by day (in case of any duplicates)
        seen: set[str] = set()
        deduped: list[Bar] = []
        for b in bars:
            key = b.ts.date().isoformat()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(b)
        return deduped


def _cache_path(cache_dir: Path, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    sym = symbol.replace("/", "-")
    key = f"coingecko|{sym}|{timeframe}|{start.isoformat()}|{end.isoformat()}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return cache_dir / f"cg_{sym}_{timeframe}_{start.date()}_{end.date()}_{digest}.json"


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
    client: CoinGeckoHistoryClient,
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
