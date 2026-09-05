"""CoinGecko adapter — pair resolution + cache round-trip."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.coingecko.history import (
    CoinGeckoHistoryClient,
    _resolve_pair,
    load_bars_cached,
)


def test_resolve_pair_btc_usd():
    assert _resolve_pair("BTC/USD") == ("bitcoin", "usd")


def test_resolve_pair_eth_usd():
    assert _resolve_pair("ETH/USD") == ("ethereum", "usd")


def test_resolve_pair_case_insensitive():
    assert _resolve_pair("btc/usd") == ("bitcoin", "usd")


def test_resolve_pair_usdt_maps_to_usd_peg():
    """CoinGecko doesn't quote in USDT; we route USDT requests to the USD peg."""
    assert _resolve_pair("BTC/USDT") == ("bitcoin", "usd")


def test_resolve_pair_rejects_unknown_base():
    with pytest.raises(ValueError, match="no CoinGecko coin_id mapping"):
        _resolve_pair("XYZ/USD")


def test_resolve_pair_rejects_slashless():
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        _resolve_pair("BTCUSD")


@pytest.mark.asyncio
async def test_load_bars_cached_writes_then_reads(tmp_path):
    class _Stub(CoinGeckoHistoryClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def bars(self, symbol, start, end, timeframe="1Day"):
            from singularity.adapters.alpaca_crypto.history import Bar
            return [
                Bar(ts=start, open=100.5, high=100.5, low=100.5, close=100.5, volume=0.0),
                Bar(ts=start + timedelta(days=1), open=101.5, high=101.5, low=101.5, close=101.5, volume=0.0),
            ]

    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 3, tzinfo=timezone.utc)
    stub = _Stub()

    bars1 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars1) == 2
    cache_files = list(tmp_path.glob("cg_*.json"))
    assert len(cache_files) == 1

    async def failing(*a, **kw):
        raise RuntimeError("cache hit expected")
    stub.bars = failing
    bars2 = await load_bars_cached(stub, "BTC/USD", start, end, tmp_path)
    assert len(bars2) == 2
    assert bars2[0].close == 100.5
