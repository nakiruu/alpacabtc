"""Baseline per-fold metrics — plan §5.3 subset (DSR/PBO/bootstrap in batch 3.2).

Convention: `returns` is a sequence of arithmetic per-bar returns
    r_t = (P_t - P_{t-1}) / P_{t-1}
Weight the returns by strategy positions in [0, 1] before passing them here.

Crypto trades 24/7 so annualization factor for daily bars is sqrt(365), not sqrt(252).
For hourly bars: sqrt(365*24). For 1m: sqrt(365*24*60).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


BARS_PER_YEAR = {
    "1Min": 365 * 24 * 60,
    "1Hour": 365 * 24,
    "1Day": 365,
}


@dataclass(frozen=True)
class Metrics:
    n: int
    total_return: float          # cumulative product minus 1
    annualized_sharpe: float
    max_drawdown: float          # fraction, negative (e.g. -0.20 = -20%)
    hit_rate: float              # fraction of positive-return bars
    mean_return: float           # per-bar
    stdev_return: float          # per-bar


def _annualization_factor(timeframe: str) -> float:
    return math.sqrt(BARS_PER_YEAR.get(timeframe, 365))


def annualized_sharpe(returns: list[float], timeframe: str = "1Day") -> float:
    if len(returns) < 2:
        return 0.0
    mu = statistics.fmean(returns)
    sd = statistics.stdev(returns)
    if sd == 0.0:
        return 0.0
    return mu / sd * _annualization_factor(timeframe)


def cumulative_return(returns: list[float]) -> float:
    """Product of (1+r) - 1."""
    cum = 1.0
    for r in returns:
        cum *= 1.0 + r
    return cum - 1.0


def max_drawdown(returns: list[float]) -> float:
    """Max peak-to-trough drawdown on cumulative equity, returned as a negative fraction."""
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        dd = (equity - peak) / peak
        if dd < worst:
            worst = dd
    return worst


def hit_rate(returns: list[float]) -> float:
    if not returns:
        return 0.0
    positives = sum(1 for r in returns if r > 0)
    return positives / len(returns)


def summary(returns: list[float], timeframe: str = "1Day") -> Metrics:
    if not returns:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mu = statistics.fmean(returns)
    sd = statistics.stdev(returns) if len(returns) > 1 else 0.0
    return Metrics(
        n=len(returns),
        total_return=cumulative_return(returns),
        annualized_sharpe=annualized_sharpe(returns, timeframe),
        max_drawdown=max_drawdown(returns),
        hit_rate=hit_rate(returns),
        mean_return=mu,
        stdev_return=sd,
    )
