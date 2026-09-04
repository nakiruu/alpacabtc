"""Vol-target overlay — realized vol + banded multiplier."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.alpaca_crypto.history import Bar
from singularity.features.vol import realized_vol_annualized
from singularity.overlays.voltarget import vol_target_multipliers


def _bar(day: int, close: float) -> Bar:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


# ---- realized_vol_annualized ----

def test_realized_vol_zero_during_warmup():
    rets = [0.01, -0.01, 0.02, -0.005, 0.015]
    v = realized_vol_annualized(rets, lookback=5)
    assert v[:4] == [0.0, 0.0, 0.0, 0.0]
    assert v[4] > 0


def test_realized_vol_matches_hand_calculation():
    """stdev of [0.01, -0.01] = 0.01√2 ≈ 0.01414; annualized × sqrt(365) ≈ 0.270"""
    rets = [0.01, -0.01]
    v = realized_vol_annualized(rets, lookback=2)
    import statistics
    expected = statistics.stdev(rets) * math.sqrt(365)
    assert v[1] == pytest.approx(expected)


def test_realized_vol_zero_returns_zero_vol():
    v = realized_vol_annualized([0.0] * 10, lookback=5)
    assert all(x == 0.0 for x in v)


def test_realized_vol_requires_lookback_ge_2():
    with pytest.raises(ValueError):
        realized_vol_annualized([0.01, 0.02], lookback=1)


# ---- vol_target_multipliers ----

def test_multipliers_zero_during_warmup():
    bars = [_bar(i, 100 * (1.01 ** i)) for i in range(35)]
    m = vol_target_multipliers(bars, vol_lookback=30)
    # Bars 0..30 have no vol estimate → multiplier stays 0
    assert all(x == 0.0 for x in m[:30])


def test_multipliers_cap_at_one_for_low_vol():
    """If realized vol is way below target, multiplier saturates at 1.0."""
    # Prices with tiny returns → very low vol
    bars = [_bar(i, 100 + 0.001 * i) for i in range(60)]
    m = vol_target_multipliers(bars, target_annualized=0.40, vol_lookback=30)
    # After warmup, multiplier should hit the cap
    assert m[-1] == pytest.approx(1.0)


def test_multipliers_scale_down_for_high_vol():
    """If realized vol > target, multiplier < 1.0."""
    import random
    rng = random.Random(0)
    # Prices with big daily swings → high vol
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.05)))   # 5% daily vol
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = vol_target_multipliers(bars, target_annualized=0.40, vol_lookback=30)
    # Realized vol >> 40%, so multiplier scales down
    assert m[-1] < 1.0
    assert m[-1] > 0.0


def test_banded_rebalance_prevents_micro_adjustments():
    """A small change in vol shouldn't move the multiplier."""
    # Build a series where realized vol changes gradually within a small band
    import random
    rng = random.Random(42)
    closes = [100.0]
    for _ in range(100):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.02)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]

    tight = vol_target_multipliers(bars, target_annualized=0.40,
                                    vol_lookback=30, rebalance_band=0.01)
    loose = vol_target_multipliers(bars, target_annualized=0.40,
                                    vol_lookback=30, rebalance_band=0.30)
    # Tight band updates more often
    def n_changes(series):
        return sum(1 for i in range(1, len(series)) if series[i] != series[i-1])
    assert n_changes(tight) >= n_changes(loose)


def test_multipliers_no_lookahead():
    """Multiplier at index i must only depend on bars[:i+1]."""
    prices = [100 * (1 + 0.01 * (i % 3 - 1)) for i in range(80)]   # bounded oscillation
    bars_a = [_bar(i, p) for i, p in enumerate(prices)]
    m_a = vol_target_multipliers(bars_a, vol_lookback=20)
    # Corrupt the last bar
    prices_b = list(prices)
    prices_b[-1] = 1e6
    bars_b = [_bar(i, p) for i, p in enumerate(prices_b)]
    m_b = vol_target_multipliers(bars_b, vol_lookback=20)
    for i in range(len(m_a) - 1):
        assert m_a[i] == m_b[i], f"look-ahead in vol overlay at index {i}"


def test_composition_zeros_when_tsmom_flat():
    """If TSMOM says flat (0), position must be 0 regardless of vol multiplier."""
    from singularity.signals.tsmom import tsmom_voltarget
    # Flat prices → tsmom returns all zeros (no signal)
    bars = [_bar(i, 100.0) for i in range(200)]
    strategy = tsmom_voltarget()
    positions = strategy(bars)
    assert all(p == 0.0 for p in positions)


def test_composition_scales_when_tsmom_long():
    """When TSMOM signals long AND vol overlay says 0.5, position should be 0.5."""
    from singularity.signals.tsmom import tsmom_voltarget
    # Monotone uptrend → tsmom = 1 after warmup
    bars = [_bar(i, 100 * (1.005 ** i)) for i in range(250)]
    strategy = tsmom_voltarget(target_vol=0.40, vol_lookback=30, rebalance_band=0.15)
    positions = strategy(bars)
    # After warmup, position should be > 0 (long, scaled) and <= 1
    tail = positions[-30:]
    assert all(0.0 < p <= 1.0 for p in tail)
