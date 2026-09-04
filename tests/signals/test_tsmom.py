"""TSMOM signal math + hysteresis + look-ahead safety."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.alpaca_crypto.history import Bar
from singularity.signals.tsmom import (
    hysteresis_positions,
    tsmom,
    tsmom_signal,
)


def _bar(day: int, close: float) -> Bar:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


# ---- Signal computation ----

def test_signal_zero_during_warmup():
    """Before max(lookbacks) bars, signal is 0 (insufficient history)."""
    bars = [_bar(i, 100 + i) for i in range(180)]  # max lookback default = 180
    raws = tsmom_signal(bars)
    assert all(r == 0.0 for r in raws)


def test_signal_positive_on_monotone_uptrend():
    """Prices trending steadily up → sign(ret/vol) = +1 across all lookbacks → raw = +1."""
    bars = [_bar(i, 100 * (1.005 ** i)) for i in range(200)]   # ~0.5%/day up
    raws = tsmom_signal(bars, lookbacks=(30, 60, 90, 180))
    # Beyond warmup, signal should be strongly positive
    last = raws[-1]
    assert last == pytest.approx(1.0), f"expected raw=+1 on monotone uptrend, got {last}"


def test_signal_negative_on_monotone_downtrend():
    bars = [_bar(i, 100 * (0.995 ** i)) for i in range(200)]   # ~0.5%/day down
    raws = tsmom_signal(bars, lookbacks=(30, 60, 90, 180))
    assert raws[-1] == pytest.approx(-1.0)


def test_signal_zero_on_perfectly_flat_prices():
    """Zero returns → zero std → each lookback abstains → raw = 0."""
    bars = [_bar(i, 100.0) for i in range(200)]
    raws = tsmom_signal(bars, lookbacks=(30, 60, 90, 180))
    assert raws[-1] == 0.0


def test_signal_uses_only_history_no_lookahead():
    """The signal at index t depends ONLY on bars[:t+1]. Editing a future bar
    must not change past signals."""
    prices = [100 * (1.002 ** i) for i in range(200)]
    bars_a = [_bar(i, p) for i, p in enumerate(prices)]
    raws_a = tsmom_signal(bars_a)
    # Corrupt the LAST bar's price dramatically
    prices_b = list(prices)
    prices_b[-1] = 1e6
    bars_b = [_bar(i, p) for i, p in enumerate(prices_b)]
    raws_b = tsmom_signal(bars_b)
    # All signals BEFORE the corrupted bar should be identical
    for i in range(len(raws_a) - 1):
        assert raws_a[i] == raws_b[i], f"look-ahead detected at index {i}"


def test_signal_reproduces_expected_ratio_on_symmetric_walk():
    """A price series that goes up 10% then back down 10% over the window
    should net near-zero ret_L and give a near-zero signal."""
    n = 100
    prices = ([100 * (1.001 ** i) for i in range(n // 2)] +
              [100 * (1.001 ** (n // 2 - 1)) * (0.999 ** (i - n // 2 + 1))
               for i in range(n // 2, n)])
    bars = [_bar(i, p) for i, p in enumerate(prices)]
    raws = tsmom_signal(bars, lookbacks=(30,))
    # At the end, the 30-bar return spans the down leg → negative signal
    assert raws[-1] == -1.0


# ---- Hysteresis ----

def test_hysteresis_starts_flat_and_stays_flat_below_enter():
    positions = hysteresis_positions(
        [0.0, 0.1, 0.2, 0.24, 0.0, -0.5], enter=0.25, exit_=-0.10,
    )
    assert positions == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_hysteresis_enters_long_when_signal_exceeds_enter():
    positions = hysteresis_positions(
        [0.0, 0.26, 0.3, 0.1, 0.0], enter=0.25, exit_=-0.10,
    )
    # Bar 0: flat. Bar 1: 0.26 > enter → long. Bar 2..4: stay long (never crosses exit_)
    assert positions == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_hysteresis_exits_only_when_signal_drops_below_exit_threshold():
    positions = hysteresis_positions(
        [0.30, 0.20, 0.10, 0.00, -0.05, -0.11], enter=0.25, exit_=-0.10,
    )
    # Bar 0: enters long. Bars 1-4: stay long (raw > exit_=-0.10). Bar 5: -0.11 < exit_ → flat
    assert positions == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_hysteresis_asymmetric_band_prevents_whipsaw():
    """Oscillating around 0 shouldn't cause repeated in/out transitions."""
    # Signal oscillates in (-0.10, +0.25) — no crossings should fire
    raws = [0.05, -0.05, 0.10, -0.05, 0.15, -0.09, 0.20]
    positions = hysteresis_positions(raws, enter=0.25, exit_=-0.10)
    assert positions == [0.0] * 7


def test_hysteresis_full_cycle_enter_hold_exit():
    raws = [0.0, 0.30, 0.10, -0.15, 0.30, -0.20]
    positions = hysteresis_positions(raws, enter=0.25, exit_=-0.10)
    # flat, long (>enter), long, flat (<exit_), long (>enter), flat (<exit_)
    assert positions == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]


# ---- Factory ----

def test_tsmom_factory_produces_positions_of_right_length():
    bars = [_bar(i, 100 * (1.005 ** i)) for i in range(200)]
    strategy = tsmom()
    positions = strategy(bars)
    assert len(positions) == len(bars)
    # After warmup, monotone uptrend should have us long
    assert positions[-1] == 1.0


def test_tsmom_factory_stays_flat_during_warmup():
    bars = [_bar(i, 100 * (1.005 ** i)) for i in range(200)]
    strategy = tsmom(lookbacks=(30, 60, 90, 180))
    positions = strategy(bars)
    # First 180 bars have no signal → hysteresis stays flat
    assert all(p == 0.0 for p in positions[:180])


def test_tsmom_no_lookahead_end_to_end():
    """Full pipeline check: strategy output at index t depends only on bars[:t+1]."""
    prices = [100 * (1.002 ** i) for i in range(250)]
    bars_a = [_bar(i, p) for i, p in enumerate(prices)]
    strategy = tsmom()
    pos_a = strategy(bars_a)

    prices_b = list(prices)
    prices_b[-1] = 1e6
    bars_b = [_bar(i, p) for i, p in enumerate(prices_b)]
    pos_b = strategy(bars_b)

    for i in range(len(pos_a) - 1):
        assert pos_a[i] == pos_b[i], f"look-ahead in tsmom() at index {i}"


# ---- Harness integration: warmup ----

def test_run_fold_passes_warmup_bars_to_strategy():
    """Verify strategy sees pre-test history via warmup_bars parameter."""
    from singularity.harness.backtest import run_fold
    from singularity.harness.walkforward import Fold

    bars = [_bar(i, 100 + i) for i in range(50)]

    seen_lens: list[int] = []
    def spy(bars_slice):
        seen_lens.append(len(bars_slice))
        return [0.0] * len(bars_slice)

    fold = Fold(index=0, train_start_idx=0, train_end_idx=20,
                val_start_idx=20, val_end_idx=25,
                test_start_idx=25, test_end_idx=35)
    res = run_fold(spy, bars, fold, "1Day", warmup_bars=10)
    # Strategy should have seen bars[15:35] = 20 bars (10 warmup + 10 test)
    assert seen_lens == [20]
    # Position count on the FoldResult reflects only the test window
    assert res.n_test_bars == 10


def test_run_fold_clamps_warmup_at_start_of_bars():
    """When warmup would go before bar 0, we clamp — no negative slicing."""
    from singularity.harness.backtest import run_fold
    from singularity.harness.walkforward import Fold

    bars = [_bar(i, 100 + i) for i in range(30)]

    def spy(bars_slice):
        return [0.0] * len(bars_slice)

    fold = Fold(index=0, train_start_idx=0, train_end_idx=5,
                val_start_idx=5, val_end_idx=10,
                test_start_idx=10, test_end_idx=20)
    # Request warmup=100 but test_start=10 → clamped to 0
    res = run_fold(spy, bars, fold, "1Day", warmup_bars=100)
    assert res.n_test_bars == 10
