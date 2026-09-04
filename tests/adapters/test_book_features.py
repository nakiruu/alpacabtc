"""Book feature primitives — OFI (Cont-Kukanov-Stoikov) and compute_book_features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.alpaca_crypto.book_features import (
    BookFeatureEngine,
    _ofi_side_delta,
    compute_book_features,
)
from singularity.adapters.alpaca_crypto.orderbook import OrderBook


# ---------- _ofi_side_delta ----------

def test_ofi_bid_price_improved_returns_new_size():
    d = _ofi_side_delta(new_px=101, new_sz=5, prev_px=100, prev_sz=3, is_bid=True)
    assert d == 5.0


def test_ofi_bid_price_worsened_returns_negative_prev_size():
    d = _ofi_side_delta(new_px=99, new_sz=5, prev_px=100, prev_sz=3, is_bid=True)
    assert d == -3.0


def test_ofi_bid_same_price_returns_size_delta():
    d = _ofi_side_delta(new_px=100, new_sz=5, prev_px=100, prev_sz=3, is_bid=True)
    assert d == 2.0


def test_ofi_ask_price_improved_lower_returns_new_size():
    # For asks, "improvement" = lower price
    d = _ofi_side_delta(new_px=101, new_sz=5, prev_px=102, prev_sz=3, is_bid=False)
    assert d == 5.0


def test_ofi_ask_price_worsened_higher_returns_negative_prev():
    d = _ofi_side_delta(new_px=103, new_sz=5, prev_px=102, prev_sz=3, is_bid=False)
    assert d == -3.0


def test_ofi_first_observation_is_zero():
    d = _ofi_side_delta(new_px=100, new_sz=5, prev_px=None, prev_sz=None, is_bid=True)
    assert d == 0.0


# ---------- BookFeatureEngine end-to-end ----------

def _apply_snapshot(ob: OrderBook, bid_px, bid_sz, ask_px, ask_sz):
    ob.apply({
        "r": True,
        "b": [{"p": bid_px, "s": bid_sz}],
        "a": [{"p": ask_px, "s": ask_sz}],
    })


def test_engine_accumulates_ofi_with_positive_buy_pressure():
    ob = OrderBook("BTC/USD")
    eng = BookFeatureEngine("BTC/USD", ofi_window_s=60)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_snapshot(ob, 100, 1, 101, 1)
    eng.update(ob, t0)

    # Bid improves → +new_size buy pressure
    _apply_snapshot(ob, 100.5, 2, 101, 1)
    eng.update(ob, t0 + timedelta(seconds=1))

    # Ask worsens (higher) → sellers left → -prev_ask_size means sell-side decrease
    # dbid - dask = +2 - (-1) = +3 net buy pressure
    _apply_snapshot(ob, 100.5, 2, 101.5, 1)
    eng.update(ob, t0 + timedelta(seconds=2))

    assert eng._ofi.value() == 3.0


def test_engine_ofi_windowed_expires_old_events():
    ob = OrderBook("BTC/USD")
    eng = BookFeatureEngine("BTC/USD", ofi_window_s=1)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    _apply_snapshot(ob, 100, 1, 101, 1)
    eng.update(ob, t0)
    _apply_snapshot(ob, 100.5, 2, 101, 1)
    eng.update(ob, t0 + timedelta(seconds=0.5))
    # Advance past window → the old event drops out
    _apply_snapshot(ob, 100.5, 2, 101, 1)
    eng.update(ob, t0 + timedelta(seconds=10))
    # Only the last update (same-price, size 2→2) contributes: 0
    assert eng._ofi.value() == 0.0


# ---------- compute_book_features basic sanity ----------

def test_compute_book_features_spread_and_imbalance():
    now = datetime.now(timezone.utc)
    bids = [(99.0, 2.0), (98.0, 1.0)]
    asks = [(101.0, 1.0), (102.0, 2.0)]
    f = compute_book_features(symbol="BTC/USD", ts=now, bids=bids, asks=asks, ofi_1m=0.0)
    # spread = 2 / 100 * 1e4 = 200 bps
    assert f.spread_bps == pytest.approx(200.0)
    # imb_1 = (2 - 1) / (2 + 1) = 0.333...
    assert f.imb_1 == pytest.approx(1 / 3)
    # depth_slope > 0 (cumulative size grows with offset)
    assert f.depth_slope > 0
