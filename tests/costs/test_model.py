"""Deterministic tests for the cost model primitives.

Precision matters here — plan §3 gate is "within 3 bps". Regressions in this
module invalidate every downstream backtest.
"""

from __future__ import annotations

import math

import pytest

from singularity.costs import fees
from singularity.costs.model import (
    ADVERSE_K,
    BookSnapshot,
    _vwap_walk,
    adverse_selection_cost,
    fill_prob,
    one_way_cost,
    round_trip_cost,
)
from singularity.costs.types import Cost, Fill, Side
from datetime import datetime, timezone


# ---------- Fees ----------

def test_fee_tier_lookup():
    # Only tier0 is verified; all volumes map to tier0 until Phase 2 confirms.
    assert fees.lookup(0).name == "tier0"
    assert fees.lookup(1_000_000_000).name == "tier0"


def test_fee_bps_maker_vs_taker():
    assert fees.fee_bps(is_maker=True, volume_30d_usd=0) == 15.0
    assert fees.fee_bps(is_maker=False, volume_30d_usd=0) == 25.0
    assert fees.fee_bps(is_maker=True, volume_30d_usd=100_000_001) == 15.0


# ---------- VWAP walk ----------

def test_vwap_walk_within_top_level():
    levels = [(100.0, 5.0), (100.5, 5.0)]
    # take 2 units at 100.0
    assert _vwap_walk(2.0, levels) == pytest.approx(100.0)


def test_vwap_walk_spans_levels():
    levels = [(100.0, 2.0), (100.5, 3.0)]
    # 2 at 100.0 + 1 at 100.5 = 300.5 / 3 = 100.1666...
    assert _vwap_walk(3.0, levels) == pytest.approx(100.5 / 3 + 200.0 / 3)


def test_vwap_walk_insufficient_depth():
    levels = [(100.0, 1.0)]
    assert math.isinf(_vwap_walk(2.0, levels))


# ---------- Cost dataclass ----------

def test_cost_total_and_sum():
    a = Cost(fee_bps=5, spread_bps=2, impact_bps=1)
    b = Cost(fee_bps=3, spread_bps=-1, impact_bps=4)
    total = a + b
    assert total.fee_bps == 8
    assert total.spread_bps == 1
    assert total.impact_bps == 5
    assert total.total_bps == 14


# ---------- Book fixture ----------

def _book(spread_bps: float = 2.0, mid: float = 50_000.0) -> BookSnapshot:
    half = mid * spread_bps / 2 / 1e4
    bid = mid - half
    ask = mid + half
    return BookSnapshot(
        best_bid=bid,
        best_ask=ask,
        bid_levels=[(bid, 10.0), (bid - 1, 20.0)],
        ask_levels=[(ask, 10.0), (ask + 1, 20.0)],
    )


# ---------- One-way cost: maker ----------

def test_maker_gets_spread_rebate():
    book = _book(spread_bps=2.0)
    c = one_way_cost(qty=1.0, side=Side.BUY, book=book, is_maker=True)
    assert c.fee_bps == 15.0
    assert c.spread_bps == pytest.approx(-1.0)  # -half spread
    assert c.impact_bps == 0.0
    assert c.total_bps == pytest.approx(14.0)


def test_maker_symmetric_across_side():
    book = _book()
    buy = one_way_cost(qty=1.0, side=Side.BUY, book=book, is_maker=True)
    sell = one_way_cost(qty=1.0, side=Side.SELL, book=book, is_maker=True)
    assert buy == sell


# ---------- One-way cost: taker ----------

def test_taker_pays_half_spread_and_no_impact_at_touch():
    book = _book(spread_bps=2.0)
    c = one_way_cost(qty=1.0, side=Side.BUY, book=book, is_maker=False)
    assert c.fee_bps == 25.0
    assert c.spread_bps == pytest.approx(1.0)   # +half spread
    assert c.impact_bps == pytest.approx(0.0, abs=1e-9)  # 1 unit fits at touch


def test_taker_pays_impact_when_walking_book():
    book = _book(spread_bps=2.0)  # top ask has size 10, next has size 20
    c = one_way_cost(qty=15.0, side=Side.BUY, book=book, is_maker=False)
    assert c.impact_bps > 0.0


def test_taker_returns_infinite_impact_on_thin_book():
    thin_book = BookSnapshot(
        best_bid=49_999.0,
        best_ask=50_001.0,
        bid_levels=[(49_999.0, 1.0)],
        ask_levels=[(50_001.0, 1.0)],  # only 1 unit visible
    )
    c = one_way_cost(qty=5.0, side=Side.BUY, book=thin_book, is_maker=False)
    assert math.isinf(c.impact_bps)


# ---------- Round trip ----------

def test_round_trip_is_two_one_ways():
    book = _book()
    one = one_way_cost(qty=1.0, side=Side.BUY, book=book, is_maker=True)
    rt = round_trip_cost(qty=1.0, book=book, is_maker=True)
    assert rt.total_bps == pytest.approx(2 * one.total_bps)


# ---------- Fill probability ----------

def test_fill_prob_zero_wait():
    assert fill_prob(offset_bps=0, wait_seconds=0) == 0.0


def test_fill_prob_monotonic_in_wait():
    p1 = fill_prob(offset_bps=0, wait_seconds=30)
    p2 = fill_prob(offset_bps=0, wait_seconds=60)
    p3 = fill_prob(offset_bps=0, wait_seconds=300)
    assert p1 < p2 < p3


def test_fill_prob_offset_penalty():
    close = fill_prob(offset_bps=1, wait_seconds=120)
    far = fill_prob(offset_bps=20, wait_seconds=120)
    assert close > far


def test_fill_prob_bounded():
    p = fill_prob(offset_bps=0, wait_seconds=100_000)
    assert 0 <= p <= 1


def test_fill_prob_at_tau_is_one_minus_exp():
    # By construction: at offset=0 and wait=FILL_TAU_S, P = 1 - 1/e ≈ 0.632
    from singularity.costs.model import FILL_TAU_S
    p = fill_prob(offset_bps=0, wait_seconds=FILL_TAU_S)
    assert p == pytest.approx(1 - 1 / math.e, rel=1e-6)


def test_fill_prob_offset_sign_symmetric():
    # Model uses |offset|; +5 and -5 must return the same probability
    assert fill_prob(offset_bps=5, wait_seconds=60) == fill_prob(
        offset_bps=-5, wait_seconds=60
    )


# ---------- Adverse selection ----------

def test_adverse_selection_zero_wait_is_zero():
    c = adverse_selection_cost(
        side=Side.BUY, offset_bps=1.0, wait_seconds=0.0, realized_vol_bps_per_sqrt_s=1.0
    )
    assert c == 0.0


def test_adverse_selection_zero_vol_is_zero():
    c = adverse_selection_cost(
        side=Side.BUY, offset_bps=1.0, wait_seconds=60.0, realized_vol_bps_per_sqrt_s=0.0
    )
    assert c == 0.0


def test_adverse_selection_scales_as_sqrt_wait():
    a = adverse_selection_cost(
        side=Side.BUY, offset_bps=1.0, wait_seconds=60.0, realized_vol_bps_per_sqrt_s=1.0
    )
    b = adverse_selection_cost(
        side=Side.BUY, offset_bps=1.0, wait_seconds=240.0, realized_vol_bps_per_sqrt_s=1.0
    )
    # 4x wait → 2x cost (sqrt scaling)
    assert b == pytest.approx(2 * a)


def test_adverse_selection_scales_linearly_in_vol():
    a = adverse_selection_cost(
        side=Side.BUY, offset_bps=0.0, wait_seconds=60.0, realized_vol_bps_per_sqrt_s=1.0
    )
    b = adverse_selection_cost(
        side=Side.BUY, offset_bps=0.0, wait_seconds=60.0, realized_vol_bps_per_sqrt_s=3.0
    )
    assert b == pytest.approx(3 * a)


def test_adverse_selection_side_symmetric():
    kwargs = dict(offset_bps=1.0, wait_seconds=60.0, realized_vol_bps_per_sqrt_s=1.0)
    assert adverse_selection_cost(side=Side.BUY, **kwargs) == adverse_selection_cost(
        side=Side.SELL, **kwargs
    )


def test_adverse_selection_reference_value():
    # k=0.7, vol=1bps/√s, wait=60s → 0.7 * 1 * √60 ≈ 5.42 bps
    c = adverse_selection_cost(
        side=Side.BUY, offset_bps=0.0, wait_seconds=60.0, realized_vol_bps_per_sqrt_s=1.0
    )
    assert c == pytest.approx(ADVERSE_K * math.sqrt(60), rel=1e-6)


# ---------- Fill.realized_cost_bps — fee currency conversion ----------

def _fill(*, side: Side, price: float, fee_amount: float, fee_asset: str, qty: float = 1.0):
    return Fill(
        order_id="x",
        symbol="BTC/USD",
        side=side,
        qty=qty,
        price=price,
        filled_at=datetime.now(timezone.utc),
        fee_asset=fee_asset,
        fee_amount=fee_amount,
        is_maker=False,
    )


def test_realized_fee_bps_when_fee_in_base_asset():
    # BUY 1 BTC @ 50_000; fee = 0.0015 BTC (15 bps of received asset)
    # Expected fee in USD = 0.0015 * 50_000 = 75; notional = 50_000 → 15 bps.
    fill = _fill(side=Side.BUY, price=50_000.0, fee_amount=0.0015, fee_asset="BTC")
    cost = fill.realized_cost_bps(mid_at_submit=50_000.0)
    assert cost.fee_bps == pytest.approx(15.0, rel=1e-6)


def test_realized_fee_bps_when_fee_in_quote_asset():
    # SELL 1 BTC @ 50_000; fee = 75 USD (already in quote)
    fill = _fill(side=Side.SELL, price=50_000.0, fee_amount=75.0, fee_asset="USD")
    cost = fill.realized_cost_bps(mid_at_submit=50_000.0)
    assert cost.fee_bps == pytest.approx(15.0, rel=1e-6)


def test_realized_spread_bps_sign_by_side():
    # BUY above mid → positive cost; SELL below mid → also positive cost
    buy = _fill(side=Side.BUY, price=50_100.0, fee_amount=0.0, fee_asset="USD")
    sell = _fill(side=Side.SELL, price=49_900.0, fee_amount=0.0, fee_asset="USD")
    assert buy.realized_cost_bps(50_000).spread_bps > 0
    assert sell.realized_cost_bps(50_000).spread_bps > 0
