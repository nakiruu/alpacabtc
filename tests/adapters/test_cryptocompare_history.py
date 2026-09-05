"""CryptoCompare parsing + symbol splitting + cache round-trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.cryptocompare.history import (
    CryptoCompareHistoryClient,
    _split_symbol,
    load_bars_cached,
)


def test_split_symbol_slashed_form():
    assert _split_symbol("BTC/USD") == ("BTC", "USD")
    assert _split_symbol("ETH/BTC") == ("ETH", "BTC")


def test_split_symbol_unslashed_form():
    assert _split_symbol("BTCUSD") == ("BTC", "USD")
    assert _split_symbol("ETHBTC") == ("ETH", "BTC")


def test_split_symbol_lowercase_normalized():
    assert _split_symbol("btc/usd") == ("BTC", "USD")


def test_split_symbol_rejects_unknown_quote():
    with pytest.raises(ValueError, match="cannot split"):
        _split_symbol("BTCXYZ")


@pytest.mark.asyncio
async def test_load_bars_cached_writes_then_reads(tmp_path):
    class _Stub(CryptoCompareHistoryClient):
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
    cache_files = list(tmp_path.glob("cc_*.json"))
    assert len(cache_files) == 1

    async def failing(*a, **kw):
        raise RuntimeError("cache hit expected")
    stub.bars = failing
    bars2 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars2) == 2
    assert bars2[0].close == bars1[0].close
