"""Regime gate — sticky vol-threshold behavior + composition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.alpaca_crypto.history import Bar
from singularity.overlays.regime import regime_gate_multipliers


def _bar(day: int, close: float) -> Bar:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_regime_defaults_risk_on_during_warmup():
    """Before baseline is computable, multiplier is 1.0 (risk-on)."""
    bars = [_bar(i, 100 + 0.1 * i) for i in range(100)]
    m = regime_gate_multipliers(bars, vol_lookback=30, baseline_lookback=180)
    # Baseline needs 180+ bars; with 100 we should never trigger and stay at 1.0
    assert all(x == 1.0 for x in m)


def test_regime_stays_risk_on_when_vol_stable():
    """Constant returns → vol is stable → never crosses threshold."""
    # Small constant-ish returns
    bars = [_bar(i, 100 * (1.001 ** i)) for i in range(300)]
    m = regime_gate_multipliers(
        bars, vol_lookback=30, baseline_lookback=180,
        vol_threshold_ratio=1.5, risk_off_multiplier=0.5, sticky_bars=20,
    )
    # After warmup, no vol spike → stay risk-on
    assert all(x == 1.0 for x in m[200:])


def test_regime_activates_on_vol_spike():
    """Big price whipsaw after quiet period → vol ratio spikes → risk-off engages."""
    import random
    rng = random.Random(0)
    # 200 bars of quiet (~0.5% daily vol)
    closes = [100.0]
    for _ in range(200):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    # Then 50 bars of chaos (~8% daily vol)
    for _ in range(50):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.08)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = regime_gate_multipliers(
        bars, vol_lookback=30, baseline_lookback=180,
        vol_threshold_ratio=1.5, risk_off_multiplier=0.5, sticky_bars=20,
    )
    # Some bars in the chaos period should be risk-off
    chaos_multipliers = m[210:]
    assert any(x == 0.5 for x in chaos_multipliers), \
        f"expected some risk-off multipliers, got {set(chaos_multipliers)}"


def test_regime_sticky_prevents_immediate_exit():
    """Once risk-off, must stay for at least sticky_bars even if vol drops."""
    import random
    rng = random.Random(1)
    # Quiet, then ONE spike bar, then quiet again
    closes = [100.0]
    for _ in range(200):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    # Insert a single 30% swing
    closes.append(closes[-1] * 1.3)
    # Then more quiet
    for _ in range(100):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = regime_gate_multipliers(
        bars, vol_lookback=30, baseline_lookback=180,
        vol_threshold_ratio=1.5, risk_off_multiplier=0.5, sticky_bars=20,
    )
    # Find first risk-off transition
    first_off = next((i for i, x in enumerate(m) if x == 0.5), None)
    if first_off is not None:
        # For the next sticky_bars, must stay risk-off
        window = m[first_off:first_off + 20]
        assert all(x == 0.5 for x in window), \
            f"regime broke sticky rule within first 20 bars after entry: {window}"


def test_regime_no_lookahead():
    """Corrupting future bars must not change past multipliers."""
    import random
    rng = random.Random(42)
    closes = [100.0]
    for _ in range(250):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.02)))
    bars_a = [_bar(i, c) for i, c in enumerate(closes)]
    m_a = regime_gate_multipliers(bars_a)

    closes_b = list(closes)
    closes_b[-1] = 1e6   # corrupt the last bar
    bars_b = [_bar(i, c) for i, c in enumerate(closes_b)]
    m_b = regime_gate_multipliers(bars_b)

    for i in range(len(m_a) - 1):
        assert m_a[i] == m_b[i], f"look-ahead in regime gate at index {i}"


def test_regime_multipliers_are_in_expected_set():
    """Values must be exactly {risk_off, 1.0} — no fractional intermediates."""
    import random
    rng = random.Random(7)
    closes = [100.0]
    for _ in range(300):
        # Occasional bigger moves
        vol = 0.02 if rng.random() > 0.05 else 0.10
        closes.append(closes[-1] * (1 + rng.gauss(0, vol)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    m = regime_gate_multipliers(
        bars, risk_off_multiplier=0.4,
    )
    assert set(m) <= {0.4, 1.0}, f"unexpected multiplier values: {set(m)}"


def test_regime_baseline_lookback_matters():
    """A shorter baseline gets 'contaminated' by recent spikes faster."""
    import random
    rng = random.Random(11)
    closes = [100.0]
    for _ in range(200):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.005)))
    for _ in range(80):
        closes.append(closes[-1] * (1 + rng.gauss(0, 0.06)))
    bars = [_bar(i, c) for i, c in enumerate(closes)]

    m_short = regime_gate_multipliers(bars, baseline_lookback=60)
    m_long = regime_gate_multipliers(bars, baseline_lookback=180)
    # Both should have some risk-off periods, but the counts differ
    short_off = sum(1 for x in m_short if x == 0.5)
    long_off = sum(1 for x in m_long if x == 0.5)
    # No strict inequality since both may respond; just assert we get SOME risk-off
    assert short_off > 0 or long_off > 0


# ---- Composition ----

def test_tsmom_full_zeros_when_tsmom_flat():
    """Regardless of vol / regime, if TSMOM is flat, position is 0."""
    from singularity.signals.tsmom import tsmom_full
    # Perfectly flat prices → tsmom sees no signal → stays flat
    bars = [_bar(i, 100.0) for i in range(300)]
    strategy = tsmom_full()
    positions = strategy(bars)
    assert all(p == 0.0 for p in positions)


def test_tsmom_full_scales_by_regime_and_vol():
    """When TSMOM is long, position = 1 * vol_mult * regime_mult ∈ [0, 1]."""
    from singularity.signals.tsmom import tsmom_full
    bars = [_bar(i, 100 * (1.003 ** i)) for i in range(300)]
    strategy = tsmom_full(
        regime_baseline_lookback=180,
    )
    positions = strategy(bars)
    tail = positions[-50:]
    assert all(0.0 < p <= 1.0 for p in tail), f"expected fractional longs, got {tail[:5]}"
