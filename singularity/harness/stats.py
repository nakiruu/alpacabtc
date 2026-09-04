"""Plan §5.3 statistics — Sharpe-diff testing via paired circular block bootstrap.

Why bootstrap and not a t-test on the Sharpe difference:
  * Sharpe is a ratio of estimators, not a mean → its sampling distribution
    isn't normally distributed even for large n.
  * Financial returns are serially dependent (autocorrelation, vol clustering).
    Naive iid tests overstate significance.
  * Block bootstrap preserves the local autocorrelation structure. Circular
    (wrap-around) removes the endpoint truncation bias of plain block.

Paired: we sample the SAME block indices from both series so their
cross-correlation is preserved. Otherwise a strategy and its benchmark would
be tested as if independent, which they clearly aren't (both consume the same
market).

For a walk-forward run: concatenate all fold net_returns and bootstrap the
concatenation. Doesn't preserve fold-boundary structure but doesn't need to —
the test is "is the strategy's Sharpe over the whole test period different
from the benchmark's Sharpe over the same period."
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from .metrics import _annualization_factor


@dataclass(frozen=True)
class BootstrapResult:
    n_bootstrap: int
    block_size: int
    observed_diff: float          # observed Sharpe_a - Sharpe_b (annualized)
    ci_low: float                 # 2.5th percentile of bootstrap distribution
    ci_high: float                # 97.5th percentile
    p_value: float                # two-sided: P(true diff = 0)


def _sharpe_from_returns(rs: list[float], factor: float) -> float:
    if len(rs) < 2:
        return 0.0
    sd = statistics.stdev(rs)
    if sd == 0.0:
        return 0.0
    return statistics.fmean(rs) / sd * factor


def _default_block_size(n: int) -> int:
    """n^(1/3) heuristic (Politis-White-ish). Bounded at [1, n]."""
    if n <= 1:
        return 1
    return max(1, min(n, int(round(n ** (1.0 / 3.0)))))


def _resample_indices(n: int, block_size: int, rng: random.Random) -> list[int]:
    """Circular block bootstrap indices: sample ceil(n/b) starting points,
    take b consecutive (wrapping at n), truncate to n."""
    n_blocks = math.ceil(n / block_size)
    out: list[int] = []
    for _ in range(n_blocks):
        start = rng.randrange(n)
        for k in range(block_size):
            out.append((start + k) % n)
            if len(out) >= n:
                return out
    return out[:n]


def paired_block_bootstrap(
    returns_a: list[float],
    returns_b: list[float],
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    timeframe: str = "1Day",
    seed: int = 0,
) -> BootstrapResult:
    """Return the bootstrap distribution of SR(a) - SR(b).

    Both series MUST be the same length and time-aligned. The `paired`
    property comes from sampling one set of block indices per iteration
    and applying it to both — this preserves cross-correlation between
    the strategies.
    """
    if len(returns_a) != len(returns_b):
        raise ValueError(
            f"paired bootstrap requires equal-length series: "
            f"a={len(returns_a)} b={len(returns_b)}"
        )
    n = len(returns_a)
    if n < 4:
        return BootstrapResult(
            n_bootstrap=0, block_size=1,
            observed_diff=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0,
        )

    factor = _annualization_factor(timeframe)
    b = block_size or _default_block_size(n)
    rng = random.Random(seed)

    observed = _sharpe_from_returns(returns_a, factor) - _sharpe_from_returns(returns_b, factor)

    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(n, b, rng)
        sample_a = [returns_a[i] for i in idx]
        sample_b = [returns_b[i] for i in idx]
        diffs.append(
            _sharpe_from_returns(sample_a, factor) - _sharpe_from_returns(sample_b, factor)
        )

    diffs.sort()
    ci_low = diffs[max(0, int(round(0.025 * n_bootstrap)) - 1)]
    ci_high = diffs[min(n_bootstrap - 1, int(round(0.975 * n_bootstrap)) - 1)]

    # Two-sided percentile p-value: fraction of the bootstrap distribution on
    # the SIGN OPPOSITE to the observed effect (Efron & Tibshirani). Doubled
    # for two-sidedness. This is the standard "achieved significance level."
    n_lt_zero = sum(1 for d in diffs if d < 0.0)
    n_gt_zero = sum(1 for d in diffs if d > 0.0)
    tail = min(n_lt_zero, n_gt_zero) / n_bootstrap
    p_value = min(1.0, 2.0 * tail)

    return BootstrapResult(
        n_bootstrap=n_bootstrap,
        block_size=b,
        observed_diff=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
    )


def concat_fold_returns(per_fold_returns: list[list[float]]) -> list[float]:
    """Flatten per-fold returns into one time-ordered series for bootstrap input."""
    out: list[float] = []
    for f in per_fold_returns:
        out.extend(f)
    return out
