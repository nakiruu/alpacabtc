"""Binance kline parsing + symbol mapping + cache round-trip."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from singularity.adapters.binance.history import (
    BinanceHistoryClient,
    _parse_kline,
    binance_symbol,
    load_bars_cached,
)


def test_binance_symbol_maps_alpaca_pairs():
    assert binance_symbol("BTC/USD") == "BTCUSDT"
    assert binance_symbol("ETH/USD") == "ETHUSDT"
    assert binance_symbol("ETH/BTC") == "ETHBTC"


def test_binance_symbol_passthrough_for_usdt_pairs():
    assert binance_symbol("BTC/USDT") == "BTCUSDT"
    assert binance_symbol("ETH/USDT") == "ETHUSDT"


def test_binance_symbol_case_insensitive():
    assert binance_symbol("btc/usd") == "BTCUSDT"


def test_binance_symbol_falls_back_to_stripped_form():
    # Unknown pair — try naive strip + USD→USDT translation
    assert binance_symbol("SOL/USD") == "SOLUSDT"
    assert binance_symbol("XYZ/BTC") == "XYZBTC"


def test_parse_kline_extracts_all_fields():
    row = [
        1499040000000, "0.01634790", "0.80000000", "0.01575800",
        "0.01577100", "148976.11427815", 1499644799999,
        "2434.19055334", 308, "1756.87402397",
        "28.46694368", "17928899.62484339",
    ]
    bar = _parse_kline(row)
    assert bar.ts == datetime(2017, 7, 3, tzinfo=timezone.utc)
    assert bar.open == pytest.approx(0.01634790)
    assert bar.high == pytest.approx(0.80000000)
    assert bar.low == pytest.approx(0.01575800)
    assert bar.close == pytest.approx(0.01577100)
    assert bar.volume == pytest.approx(148976.11427815)


@pytest.mark.asyncio
async def test_cache_round_trip(tmp_path, monkeypatch):
    """load_bars_cached: miss writes, hit reads. Uses stub client for offline test."""

    class _StubClient(BinanceHistoryClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def bars(self, symbol, start, end, timeframe="1Day", limit_per_page=1000):
            from singularity.adapters.alpaca_crypto.history import Bar
            return [
                Bar(ts=start, open=100.0, high=101.0, low=99.0, close=100.5, volume=10.0),
                Bar(ts=start + timedelta(days=1), open=100.5, high=102.0, low=100.0, close=101.5, volume=12.0),
            ]

    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 3, tzinfo=timezone.utc)
    stub = _StubClient()

    # First call: miss → fetch → cache written
    bars1 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars1) == 2
    cache_files = list(tmp_path.glob("binance_*.json"))
    assert len(cache_files) == 1

    # Corrupt the stub so a second fetch would fail — proves cache hit
    async def failing(*a, **kw):
        raise RuntimeError("would refetch")
    stub.bars = failing
    bars2 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars2) == 2
    assert bars2[0].close == bars1[0].close


@pytest.mark.asyncio
async def test_cache_recovers_from_corrupted_json(tmp_path):
    class _StubClient(BinanceHistoryClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def bars(self, symbol, start, end, timeframe="1Day", limit_per_page=1000):
            from singularity.adapters.alpaca_crypto.history import Bar
            return [Bar(ts=start, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)]

    # Write corrupt JSON at the cache path
    from singularity.adapters.binance.history import _cache_path
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 2, tzinfo=timezone.utc)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = _cache_path(tmp_path, "BTC/USD", "1Day", start, end)
    path.write_text("{not valid json")
    # Loader should detect and refetch, not blow up
    bars = await load_bars_cached(_StubClient(), "BTC/USD", start, end, tmp_path)
    assert len(bars) == 1
