"""Metrics primitives — Sharpe, drawdown, cumulative return, hit rate."""

from __future__ import annotations

import math

import pytest

from singularity.harness.metrics import (
    annualized_sharpe,
    cumulative_return,
    hit_rate,
    max_drawdown,
    summary,
)


def test_annualized_sharpe_constant_positive_return():
    # 1% every day, no volatility → sharpe would be infinity; we return 0 on sd=0
    assert annualized_sharpe([0.01] * 30) == 0.0


def test_annualized_sharpe_matches_hand_calculation():
    # Simple pattern: alternating +2% / -1%, mean=0.5%, sd=1.5%
    r = [0.02, -0.01, 0.02, -0.01, 0.02, -0.01]
    sr = annualized_sharpe(r, timeframe="1Day")
    # Expected daily SR = 0.005 / stdev; annualized × sqrt(365)
    import statistics
    expected = statistics.fmean(r) / statistics.stdev(r) * math.sqrt(365)
    assert sr == pytest.approx(expected)


def test_annualized_sharpe_empty():
    assert annualized_sharpe([]) == 0.0
    assert annualized_sharpe([0.01]) == 0.0


def test_cumulative_return_compounds():
    assert cumulative_return([0.1, 0.1]) == pytest.approx(0.21)
    assert cumulative_return([-0.5, 1.0]) == pytest.approx(0.0)   # halved then doubled


def test_cumulative_return_empty():
    assert cumulative_return([]) == 0.0


def test_max_drawdown_monotone_up_is_zero():
    assert max_drawdown([0.01, 0.02, 0.01, 0.03]) == 0.0


def test_max_drawdown_captures_peak_to_trough():
    # +100% then -50%: peak equity 2.0, trough 1.0 → -50% dd
    dd = max_drawdown([1.0, -0.5])
    assert dd == pytest.approx(-0.5)


def test_max_drawdown_deep_dip_recovered():
    # +50%, -60%, +100%: peak 1.5, trough 0.6, recover to 1.2 → dd = (0.6 - 1.5) / 1.5 = -0.6
    dd = max_drawdown([0.5, -0.6, 1.0])
    assert dd == pytest.approx(-0.6)


def test_hit_rate():
    assert hit_rate([0.1, -0.1, 0.1, 0.1]) == 0.75
    assert hit_rate([0.0, 0.0]) == 0.0
    assert hit_rate([]) == 0.0


def test_summary_populates_all_fields():
    s = summary([0.01, -0.005, 0.02])
    assert s.n == 3
    assert s.mean_return > 0
    assert s.stdev_return > 0
    assert s.annualized_sharpe != 0
    assert s.max_drawdown <= 0
    assert 0 <= s.hit_rate <= 1


def test_unknown_timeframe_raises():
    """Silent fallback to 365 would mislabel every intraday Sharpe."""
    with pytest.raises(ValueError, match="unknown timeframe"):
        annualized_sharpe([0.01, 0.02, 0.01], timeframe="15Min")
