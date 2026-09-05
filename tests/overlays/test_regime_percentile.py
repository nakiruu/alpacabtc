"""Percentile-based regime gate (batch 4.4)."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.alpaca_crypto.history import Bar
from singularity.overlays.regime import (
    _rolling_percentile_threshold,
    regime_gate_percentile_multipliers,
)


def _bar(day: int, close: float) -> Bar:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


# ---- _rolling_percentile_threshold ----

def test_percentile_zero_when_below_min_samples():
    series = [1.0, 2.0, 3.0, 0.0, 4.0]
    out = _rolling_percentile_threshold(series, max_window=10, percentile=0.5, min_samples=6)
    assert all(x == 0.0 for x in out)


def test_percentile_uses_available_when_below_max_window():
    series = [10.0, 20.0, 30.0, 40.0, 50.0]
    out = _rolling_percentile_threshold(series, max_window=100, percentile=0.8, min_samples=3)
    # At index 2 (3 samples): 80th percentile of [10,20,30] = index 1.6 → int(1.6)=1 → 20
    assert out[0] == 0.0  # 1 sample < min_samples
    assert out[1] == 0.0  # 2 samples < min_samples
    assert out[2] == 20.0
    assert out[4] == 40.0  # 80th of [10,20,30,40,50] = index int(0.8*4)=3 → 40


def test_percentile_skips_zeros():
    """Zero vols (warmup) are excluded from the distribution."""
    series = [0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    out = _rolling_percentile_threshold(series, max_window=100, percentile=0.5, min_samples=3)
    # At index 5 (3 non-zero samples: 10,20,30): 50th percentile = index 1 → 20
    assert out[5] == 20.0


def test_percentile_50_matches_median():
    """50th percentile should agree with statistics.median for odd count."""
    import statistics
    series = [1.0, 5.0, 2.0, 8.0, 3.0]
    out = _rolling_percentile_threshold(series, max_window=10, percentile=0.5, min_samples=3)
    # At the end: median of sorted [1,2,3,5,8] = 3; our impl uses int(0.5*4)=2 → 3
    assert out[-1] == 3.0
    assert out[-1] == statistics.median(series)


# ---- regime_gate_percentile_multipliers ----

def test_percentile_regime_defaults_risk_on_during_warmup():
    bars = [_bar(i, 100.0 + 0.1 * i) for i in range(50)]
    m = regime_gate_percentile_multipliers(bars, vol_lookback=30, baseline_lookback=500)
    # Insufficient samples for percentile → all risk-on
    assert all(x == 1.0 for x in m)


def test_percentile_regime_triggers_on_extreme_vol():
    """Long quiet period then chaos — the chaos period lands in top 20% and fires."""
    rng = random.Random(0)
    closes = [100.0]
    # 200 bars of quiet
    for _ in range(200):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    # 60 bars of chaos
    for _ in range(60):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.10)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = regime_gate_percentile_multipliers(
        bars, vol_lookback=30, baseline_lookback=200,
        vol_percentile=0.80, risk_off_multiplier=0.5,
        min_baseline_samples=60, sticky_bars=20,
    )
    # Chaos period bars should mostly be risk-off
    chaos_off_count = sum(1 for x in m[220:] if x == 0.5)
    assert chaos_off_count > 20, f"expected many risk-off bars in chaos, got {chaos_off_count}"


def test_percentile_regime_no_lookahead():
    """Corrupting future bars doesn't change past multipliers."""
    rng = random.Random(42)
    closes = [100.0]
    for _ in range(250):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.02)))
    bars_a = [_bar(i, c) for i, c in enumerate(closes)]
    m_a = regime_gate_percentile_multipliers(bars_a, min_baseline_samples=30)

    closes_b = list(closes)
    closes_b[-1] = 1e6
    bars_b = [_bar(i, c) for i, c in enumerate(closes_b)]
    m_b = regime_gate_percentile_multipliers(bars_b, min_baseline_samples=30)

    for i in range(len(m_a) - 1):
        assert m_a[i] == m_b[i], f"look-ahead at index {i}"


def test_percentile_regime_sticky_exit():
    """Once risk-off, must stay for sticky_bars regardless of vol drop."""
    rng = random.Random(1)
    closes = [100.0]
    for _ in range(200):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    # One large spike bar
    closes.append(closes[-1] * 1.5)
    for _ in range(100):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = regime_gate_percentile_multipliers(
        bars, vol_lookback=30, baseline_lookback=180,
        vol_percentile=0.75, risk_off_multiplier=0.5,
        min_baseline_samples=60, sticky_bars=25,
    )
    first_off = next((i for i, x in enumerate(m) if x == 0.5), None)
    if first_off is not None:
        window = m[first_off:first_off + 25]
        assert all(x == 0.5 for x in window), \
            f"regime broke sticky rule within first 25 bars: {window}"


def test_percentile_regime_values_bounded():
    """Values must be in {risk_off, 1.0}."""
    rng = random.Random(7)
    closes = [100.0]
    for _ in range(300):
        v = 0.02 if rng.random() > 0.05 else 0.08
        closes.append(closes[-1] * (1 + rng.gauss(0, v)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = regime_gate_percentile_multipliers(bars, min_baseline_samples=30, risk_off_multiplier=0.3)
    assert set(m) <= {0.3, 1.0}


# ---- Composition ----

def test_tsmom_full_v2_composition():
    from singularity.signals.tsmom import tsmom_full_v2
    # Monotone uptrend → tsmom long after warmup
    bars = [_bar(i, 100 * (1.003 ** i)) for i in range(300)]
    strategy = tsmom_full_v2(
        regime_baseline_lookback=200,
        regime_min_baseline_samples=30,
    )
    positions = strategy(bars)
    tail = positions[-20:]
    assert all(0.0 < p <= 1.0 for p in tail), f"expected fractional longs, got {tail[:5]}"
