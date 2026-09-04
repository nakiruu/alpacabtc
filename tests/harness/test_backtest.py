"""Backtest orchestrator — fold slicing, position lag, strategy contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from singularity.adapters.alpaca_crypto.history import Bar
from singularity.harness.backtest import (
    _bar_returns,
    buy_and_hold,
    flat,
    run_backtest,
    run_fold,
)
from singularity.harness.walkforward import WalkForwardSpec, WalkForwardSplitter


def _bar(day: int, close: float) -> Bar:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return Bar(ts=ts, open=close, high=close, low=close, close=close, volume=1.0)


def test_bar_returns_produces_n_minus_1():
    bars = [_bar(i, 100 * (1.01 ** i)) for i in range(5)]
    r = _bar_returns(bars)
    assert len(r) == 4
    for x in r:
        assert x == pytest.approx(0.01)


def test_buy_and_hold_full_position_every_bar():
    bars = [_bar(i, 100.0) for i in range(10)]
    assert buy_and_hold(bars) == [1.0] * 10


def test_flat_strategy_produces_zero_returns():
    bars = [_bar(i, 100 * (1.01 ** i)) for i in range(20)]
    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=2, advance_bars=2)
    result = run_backtest(
        strategy=flat, strategy_name="flat",
        bars=bars, splitter=WalkForwardSplitter(spec),
        symbol="BTC/USD",
    )
    for f in result.per_fold:
        assert f.metrics.total_return == pytest.approx(0.0)


def test_buy_and_hold_captures_test_window_returns():
    # Prices double over 20 days; each fold's test window is 2 days
    bars = [_bar(i, 100 * (2 ** (i / 20))) for i in range(20)]
    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=2, advance_bars=2)
    splitter = WalkForwardSplitter(spec)
    result = run_backtest(
        strategy=buy_and_hold, strategy_name="buy_and_hold",
        bars=bars, splitter=splitter, symbol="BTC/USD",
    )
    assert len(result.per_fold) > 0
    for f in result.per_fold:
        # Each 2-bar test window has 1 return; positive since prices are rising
        assert f.metrics.total_return > 0
        assert f.metrics.n == 1


def test_strategy_returning_wrong_length_raises():
    bars = [_bar(i, 100.0) for i in range(10)]
    bad_strategy = lambda bs: [1.0] * (len(bs) - 1)   # off by one
    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=2, advance_bars=2)
    splitter = WalkForwardSplitter(spec)
    folds = splitter.folds(len(bars))
    with pytest.raises(ValueError, match="positions"):
        run_fold(bad_strategy, bars, folds[0], "1Day")


def test_run_backtest_aggregate_helpers():
    bars = [_bar(i, 100 + i) for i in range(20)]
    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=2, advance_bars=2)
    result = run_backtest(
        strategy=buy_and_hold, strategy_name="buy_and_hold",
        bars=bars, splitter=WalkForwardSplitter(spec),
        symbol="BTC/USD",
    )
    assert result.n_folds > 0
    assert isinstance(result.mean_sharpe, float)
    assert 0 <= result.n_negative_folds <= result.n_folds


def test_empty_folds_when_insufficient_bars():
    """If nothing fits, aggregates are safe defaults, not divide-by-zero."""
    bars = [_bar(i, 100.0) for i in range(3)]
    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=2, advance_bars=2)
    result = run_backtest(
        strategy=buy_and_hold, strategy_name="buy_and_hold",
        bars=bars, splitter=WalkForwardSplitter(spec),
        symbol="BTC/USD",
    )
    assert result.n_folds == 0
    assert result.mean_sharpe == 0.0
    assert result.n_negative_folds == 0


def test_bar_returns_rejects_non_monotonic_timestamps():
    """Data-quality guard: a duplicate or out-of-order bar must fail loudly."""
    from singularity.harness.backtest import _bar_returns
    bars = [_bar(0, 100), _bar(0, 101)]   # same day twice
    with pytest.raises(ValueError, match="non-monotonic"):
        _bar_returns(bars)


def test_bar_returns_rejects_zero_close():
    from singularity.harness.backtest import _bar_returns
    bars = [_bar(0, 0.0), _bar(1, 100.0)]
    with pytest.raises(ValueError, match="non-positive close"):
        _bar_returns(bars)


def test_position_lag_flip_captures_correct_return():
    """A position that flips from 0 to 1 mid-test-window must earn only the
    returns AFTER the flip. Guards against off-by-one in position/return alignment
    that constant strategies (buy_and_hold, flat) can't detect."""
    # Prices: 100, 110, 121, 133.1 (10% each step)
    bars = [_bar(i, 100 * (1.1 ** i)) for i in range(15)]

    # Strategy: position 0 for first half of test window, 1 for second half.
    # For a 5-bar test window: positions = [0, 0, 1, 1, 1]
    # positions[i] earns return from bars[i]→bars[i+1]:
    #   0 * r0, 0 * r1, 1 * r2, 1 * r3, (positions[4] dropped)
    # So we capture returns r2 and r3, each 0.1
    # Cumulative = 1.1 * 1.1 - 1 = 0.21
    def flip_strategy(bs):
        return [0.0, 0.0, 1.0, 1.0, 1.0][:len(bs)]

    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=5, advance_bars=5)
    splitter = WalkForwardSplitter(spec)
    folds = splitter.folds(len(bars))
    assert folds  # at least one fold fits
    from singularity.harness.backtest import run_fold
    res = run_fold(flip_strategy, bars, folds[0], "1Day")
    # Gross return isolates position-lag math from cost drag (which is now on by default).
    assert res.gross_metrics.total_return == pytest.approx(0.21, rel=1e-6)


def test_fold_result_has_both_bar_counts():
    """After the audit rename, both n_test_bars and n_return_bars must be present."""
    bars = [_bar(i, 100 + i) for i in range(15)]
    spec = WalkForwardSpec(train_bars=5, val_bars=2, test_bars=3, advance_bars=3)
    result = run_backtest(
        strategy=buy_and_hold, strategy_name="buy_and_hold",
        bars=bars, splitter=WalkForwardSplitter(spec), symbol="BTC/USD",
    )
    for f in result.per_fold:
        assert f.n_test_bars == 3
        assert f.n_return_bars == 2
