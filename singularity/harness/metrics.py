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
    if timeframe not in BARS_PER_YEAR:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; expected one of {sorted(BARS_PER_YEAR)}"
        )
    return math.sqrt(BARS_PER_YEAR[timeframe])


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
    return math.prod(1.0 + r for r in returns) - 1.0


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
    sharpe = 0.0 if sd == 0.0 else mu / sd * _annualization_factor(timeframe)
    return Metrics(
        n=len(returns),
        total_return=cumulative_return(returns),
        annualized_sharpe=sharpe,
        max_drawdown=max_drawdown(returns),
        hit_rate=hit_rate(returns),
        mean_return=mu,
        stdev_return=sd,
    )


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
#
# Two corrections applied:
#   1. Small-sample bias — variance of the Sharpe estimator accounts for
#      skew and (excess) kurtosis of returns. Fat-tailed BTC returns inflate
#      raw Sharpe estimates on short windows; this term deflates them.
#   2. Multiple-testing bias — when N strategies are tried, the max Sharpe
#      observed under a true-zero null is > 0 by chance. E[SR_max*] estimates
#      that chance-maximum given n_trials, and we subtract it.
#
# Returns the DEFLATED Sharpe: observed SR minus the noise floor. A value > 0
# means the observed SR beats what the null distribution + selection bias
# would produce; that's the plan §5.3 gate.
# ---------------------------------------------------------------------------

_EULER = 0.577215664901532


def _skew(returns: list[float], mu: float, sd: float) -> float:
    if sd == 0.0 or len(returns) < 3:
        return 0.0
    return statistics.fmean(((r - mu) / sd) ** 3 for r in returns)


def _excess_kurt(returns: list[float], mu: float, sd: float) -> float:
    if sd == 0.0 or len(returns) < 4:
        return 0.0
    return statistics.fmean(((r - mu) / sd) ** 4 for r in returns) - 3.0


def deflated_sharpe(
    sharpe: float,
    returns: list[float],
    n_trials: int = 1,
    timeframe: str = "1Day",
) -> float:
    """Deflated Sharpe: SR minus its chance-driven noise floor.

    Args:
        sharpe: annualized Sharpe already computed on `returns`.
        returns: per-bar returns the Sharpe was derived from.
        n_trials: how many independent strategies/configs were tried
            before selecting this one. For a single strategy: 1 (no
            multiple-testing deflation, but small-sample bias still applied).
        timeframe: annualization convention consistent with `sharpe`.

    Returns:
        Deflated annualized Sharpe. > 0 means the observation beats noise.
    """
    n = len(returns)
    if n < 4 or sharpe == 0.0:
        return sharpe
    mu = statistics.fmean(returns)
    sd = statistics.stdev(returns)
    if sd == 0.0:
        return sharpe

    # Convert sharpe from annualized back to per-bar SR for the variance calc.
    factor = _annualization_factor(timeframe)
    sr_per_bar = sharpe / factor

    skew = _skew(returns, mu, sd)
    kurt = _excess_kurt(returns, mu, sd)  # excess (normal = 0)
    # Variance of the per-bar SR estimator, per Mertens (2002) / Bailey-LdP:
    sr_var_per_bar = (1.0 - skew * sr_per_bar + (kurt / 4.0) * sr_per_bar ** 2) / (n - 1)
    if sr_var_per_bar <= 0.0:
        return sharpe
    sr_std_per_bar = math.sqrt(sr_var_per_bar)

    # Chance-driven maximum SR expected across n_trials, in per-bar units
    if n_trials <= 1:
        e_max_per_bar = 0.0
    else:
        nd = statistics.NormalDist()
        z_a = nd.inv_cdf(1.0 - 1.0 / n_trials)
        z_b = nd.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        e_max_per_bar = sr_std_per_bar * ((1.0 - _EULER) * z_a + _EULER * z_b)

    deflated_per_bar = sr_per_bar - e_max_per_bar
    return deflated_per_bar * factor
