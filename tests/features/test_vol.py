"""ATR helper — Wilder smoothing math on synthetic bars."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.features.vol import Bar, _bucket_trades_to_bars, _true_range, wilder_atr


def _bar(open_p, high, low, close, minute=0):
    ts = datetime(2026, 1, 1, 0, minute, tzinfo=timezone.utc)
    return Bar(ts=ts, open=open_p, high=high, low=low, close=close)


def test_true_range_no_prev_close():
    b = _bar(100, 102, 99, 101)
    assert _true_range(b, prev_close=None) == pytest.approx(3.0)


def test_true_range_gap_up():
    b = _bar(105, 106, 104, 105)  # prev close was 100 → |high - prev_close| = 6 dominates
    assert _true_range(b, prev_close=100.0) == pytest.approx(6.0)


def test_true_range_gap_down():
    b = _bar(90, 91, 88, 89)      # prev close 100 → |low - prev_close| = 12 dominates
    assert _true_range(b, prev_close=100.0) == pytest.approx(12.0)


def test_wilder_atr_needs_at_least_n_plus_one_bars():
    # 14 bars is insufficient; need 15 to have an initial ATR AND at least one smoothing step
    bars = [_bar(100, 101, 99, 100, minute=i) for i in range(14)]
    assert wilder_atr(bars, n=14) is None


def test_wilder_atr_constant_bars_gives_constant_range():
    # 20 identical bars (H-L = 2) → ATR should equal 2
    bars = [_bar(100, 101, 99, 100, minute=i) for i in range(20)]
    atr = wilder_atr(bars, n=14)
    assert atr == pytest.approx(2.0)


def test_wilder_atr_smoothing_reduces_shock_over_time():
    # 14 quiet bars (TR=2) then one large TR spike (TR=20)
    bars = [_bar(100, 101, 99, 100, minute=i) for i in range(14)]
    bars.append(_bar(100, 120, 100, 110, minute=14))   # spike bar
    atr_after_spike = wilder_atr(bars, n=14)
    # ATR_1 = 2 (mean of first 14 TRs), then smoothed with TR=20:
    # ATR = (2 * 13 + 20) / 14 = 46/14 ≈ 3.286
    assert atr_after_spike == pytest.approx((2 * 13 + 20) / 14)


def test_bucket_trades_to_bars_groups_by_minute():
    t0 = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    trades = [
        (t0, 100.0),
        (t0 + timedelta(seconds=10), 102.0),
        (t0 + timedelta(seconds=20), 99.0),
        (t0 + timedelta(seconds=90), 101.0),  # next minute
    ]
    bars = _bucket_trades_to_bars(trades, bar_minutes=1)
    assert len(bars) == 2
    # First bar: opens at 100, high 102, low 99, closes at 99
    assert bars[0].open == 100 and bars[0].high == 102
    assert bars[0].low == 99 and bars[0].close == 99
    # Second bar: single trade
    assert bars[1].open == bars[1].close == 101


def test_bucket_trades_to_bars_empty_input():
    assert _bucket_trades_to_bars([], bar_minutes=1) == []
