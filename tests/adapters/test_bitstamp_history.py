"""Bitstamp adapter — symbol mapping + cache round-trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.bitstamp.history import (
    BitstampHistoryClient,
    bitstamp_symbol,
    load_bars_cached,
)


def test_symbol_known_pairs():
    assert bitstamp_symbol("BTC/USD") == "btcusd"
    assert bitstamp_symbol("ETH/USD") == "ethusd"
    assert bitstamp_symbol("ETH/BTC") == "ethbtc"


def test_symbol_case_insensitive():
    assert bitstamp_symbol("btc/usd") == "btcusd"


def test_symbol_fallback_lowercases_and_strips():
    assert bitstamp_symbol("XYZ/USD") == "xyzusd"


@pytest.mark.asyncio
async def test_load_bars_cached_writes_then_reads(tmp_path):
    class _Stub(BitstampHistoryClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def bars(self, symbol, start, end, timeframe="1Day"):
            from singularity.adapters.alpaca_crypto.history import Bar
            return [
                Bar(ts=start, open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0),
                Bar(ts=start + timedelta(days=1), open=100.5, high=102.0, low=100.0, close=101.5, volume=12.0),
            ]

    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 3, tzinfo=timezone.utc)
    stub = _Stub()

    bars1 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars1) == 2
    cache_files = list(tmp_path.glob("bs_*.json"))
    assert len(cache_files) == 1

    async def failing(*a, **kw):
        raise RuntimeError("cache hit expected")
    stub.bars = failing
    bars2 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars2) == 2
