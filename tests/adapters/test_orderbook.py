"""Orderbook state invariants — snapshot atomicity and gap semantics."""

from __future__ import annotations

from singularity.adapters.alpaca_crypto.orderbook import OrderBook


def test_snapshot_replaces_state():
    ob = OrderBook("BTC/USD")
    ob.apply({"r": True, "b": [{"p": 100, "s": 1}], "a": [{"p": 101, "s": 1}]})
    ob.apply({"r": True, "b": [{"p": 200, "s": 2}], "a": [{"p": 201, "s": 2}]})
    assert list(ob.bids.items()) == [(200.0, 2.0)]
    assert list(ob.asks.items()) == [(201.0, 2.0)]


def test_delta_before_snapshot_is_rejected():
    ob = OrderBook("BTC/USD")
    ok = ob.apply({"b": [{"p": 100, "s": 1}], "a": [{"p": 101, "s": 1}]})
    assert ok is False
    assert not ob.initialized
    assert not ob.bids
    assert not ob.asks


def test_delta_removes_level_on_zero_size():
    ob = OrderBook("BTC/USD")
    ob.apply({"r": True, "b": [{"p": 100, "s": 1}, {"p": 99, "s": 2}], "a": []})
    ob.apply({"b": [{"p": 100, "s": 0}]})
    assert list(ob.bids.items()) == [(99.0, 2.0)]


def test_snapshot_atomicity_on_malformed_payload():
    ob = OrderBook("BTC/USD")
    ob.apply({"r": True, "b": [{"p": 100, "s": 1}], "a": [{"p": 101, "s": 1}]})
    good_bids = dict(ob.bids.items())
    good_asks = dict(ob.asks.items())
    try:
        # missing "p" field on second level → KeyError mid-parse
        ob.apply({"r": True, "b": [{"p": 200, "s": 2}, {"s": 3}], "a": [{"p": 201, "s": 2}]})
    except (KeyError, ValueError):
        pass
    # Book must still hold the previous good state OR be cleanly re-initialized;
    # what it must NOT be is a half-populated new snapshot with initialized=True.
    # Atomic swap: pre-swap state is preserved on failure.
    assert dict(ob.bids.items()) == good_bids
    assert dict(ob.asks.items()) == good_asks


def test_top_returns_best_prices():
    ob = OrderBook("BTC/USD")
    ob.apply({
        "r": True,
        "b": [{"p": 100, "s": 1}, {"p": 99, "s": 2}, {"p": 98, "s": 3}],
        "a": [{"p": 101, "s": 1}, {"p": 102, "s": 2}],
    })
    bid, ask = ob.top()
    assert bid == (100.0, 1.0)
    assert ask == (101.0, 1.0)


def test_bids_desc_sorted_highest_first():
    ob = OrderBook("BTC/USD")
    ob.apply({
        "r": True,
        "b": [{"p": 98, "s": 3}, {"p": 100, "s": 1}, {"p": 99, "s": 2}],
        "a": [],
    })
    assert ob.bids_desc(5) == [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
