"""Bitstamp adapter — symbol mapping + cache round-trip + intraday timeframes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.bitstamp.history import (
    _MAX_PAGES,
    _PAGE_LIMIT,
    _STEP_SECONDS,
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


def test_intraday_timeframes_supported():
    """The adapter accepts 1Day / 1Hour / 1Min — verify step values are correct."""
    assert _STEP_SECONDS["1Day"] == 86400
    assert _STEP_SECONDS["1Hour"] == 3600
    assert _STEP_SECONDS["1Min"] == 60


def test_pagination_budget_covers_multi_year_minute_bars():
    """_MAX_PAGES × _PAGE_LIMIT for 1Min must cover at least 5 years of data."""
    max_minute_bars = _MAX_PAGES * _PAGE_LIMIT
    min_bars_per_year = 60 * 24 * 365   # ~525,600
    assert max_minute_bars >= 5 * min_bars_per_year, (
        f"pagination budget {max_minute_bars} < 5 years of minute bars"
    )


@pytest.mark.asyncio
async def test_bars_dispatches_to_correct_step_for_minute_timeframe(monkeypatch):
    """Passing timeframe='1Min' should send step=60 to Bitstamp, not step=86400."""
    captured_params: dict[str, object] = {}

    class _CaptureClient(BitstampHistoryClient):
        async def __aenter__(self):
            class _StubHttp:
                async def get(_self, url, params=None):
                    captured_params.update(params or {})
                    class _Resp:
                        status_code = 200
                        def raise_for_status(_self): pass
                        def json(_self):
                            return {"data": {"ohlc": []}}
                    return _Resp()
                async def aclose(self): pass
            self._client = _StubHttp()
            return self

        async def __aexit__(self, *exc):
            pass

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    async with _CaptureClient() as c:
        await c.bars("BTC/USD", start, end, timeframe="1Min")
    assert captured_params["step"] == 60


@pytest.mark.asyncio
async def test_bars_rejects_unknown_timeframe():
    class _StubClient(BitstampHistoryClient):
        async def __aenter__(self):
            self._client = object()   # any truthy sentinel
            return self
        async def __aexit__(self, *exc):
            pass

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    async with _StubClient() as c:
        with pytest.raises(ValueError, match="unsupported timeframe"):
            await c.bars("BTC/USD", start, end, timeframe="15Min")
