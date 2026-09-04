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
