"""Backtest orchestrator.

Runs a strategy across the walk-forward folds and aggregates per-fold metrics.

## Strategy contract

    def positions(bars) -> list[float]:
        # Return desired position in [0, 1] for each bar.
        #
        # SEMANTICS: positions[i] is the target position for the period between
        # bars[i].close and bars[i+1].close. It MUST be decided using only
        # information available at bars[i].close — i.e. bars[0..i] inclusive.
        # Peeking at bars[i+1..] is look-ahead bias.
        #
        # Return length must equal len(bars); the last element (positions[-1])
        # is dropped because there is no bar after it to earn a return.

The orchestrator enforces the length equality but not the "no peeking" rule —
strategy authors are responsible for that self-discipline. buy_and_hold and
flat are trivially compliant (positions are constant). Phase 4's TSMOM and
XGBoost strategies must be careful to reference only history.

Cost simulation (spread/impact/fill_prob/adverse_selection) lands in batch 3.2
and will multiply against the raw weighted returns produced here.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from ..adapters.alpaca_crypto.history import Bar
from . import metrics as m
from .walkforward import Fold, WalkForwardSplitter


Strategy = Callable[[list[Bar]], list[float]]


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    n_test_bars: int          # bars in the test window
    n_return_bars: int        # bars we earned returns on (n_test_bars - 1)
    metrics: m.Metrics


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    per_fold: list[FoldResult]

    @property
    def n_folds(self) -> int:
        return len(self.per_fold)

    @property
    def fold_sharpes(self) -> list[float]:
        return [f.metrics.annualized_sharpe for f in self.per_fold]

    @property
    def mean_sharpe(self) -> float:
        return statistics.fmean(self.fold_sharpes) if self.per_fold else 0.0

    @property
    def n_negative_folds(self) -> int:
        return sum(1 for x in self.fold_sharpes if x < 0)


def _bar_returns(bars: list[Bar]) -> list[float]:
    """Close-to-close arithmetic returns. len == len(bars) - 1.

    Guards: strictly-increasing timestamps and positive prior close. A gap or
    a zero/negative close silently blowing up returns is worse than a loud
    ValueError — that's exactly the kind of "backtest lies" the plan §5 warns
    about.
    """
    out: list[float] = []
    for prev, curr in zip(bars, bars[1:]):
        if prev.close <= 0:
            raise ValueError(f"non-positive close at {prev.ts.isoformat()}: {prev.close}")
        if curr.ts <= prev.ts:
            raise ValueError(f"non-monotonic bars: {prev.ts.isoformat()} → {curr.ts.isoformat()}")
        out.append((curr.close - prev.close) / prev.close)
    return out


def run_fold(strategy: Strategy, bars: list[Bar], fold: Fold, timeframe: str) -> FoldResult:
    """Evaluate `strategy` on the TEST window of `fold`.

    positions[i] applies to the return earned from bars[i].close → bars[i+1].close.
    positions[-1] is dropped (no bar after it to earn a return on).
    """
    test_bars = bars[fold.test_start_idx:fold.test_end_idx]
    positions = strategy(test_bars)
    if len(positions) != len(test_bars):
        raise ValueError(
            f"strategy returned {len(positions)} positions for {len(test_bars)} bars"
        )
    raw_returns = _bar_returns(test_bars)
    weighted = [p * r for p, r in zip(positions, raw_returns)]
    metrics = m.summary(weighted, timeframe=timeframe)
    return FoldResult(
        fold=fold,
        n_test_bars=len(test_bars),
        n_return_bars=len(weighted),
        metrics=metrics,
    )


def run_backtest(
    *,
    strategy: Strategy,
    strategy_name: str,
    bars: list[Bar],
    splitter: WalkForwardSplitter,
    symbol: str,
    timeframe: str = "1Day",
) -> BacktestResult:
    folds = splitter.folds(len(bars))
    per_fold = [run_fold(strategy, bars, f, timeframe) for f in folds]
    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        per_fold=per_fold,
    )


# ---- Strategies ----

def buy_and_hold(bars: list[Bar]) -> list[float]:
    """Trivially long the whole time. Baseline the plan §0 calls the gate."""
    return [1.0] * len(bars)


def flat(bars: list[Bar]) -> list[float]:
    """Always zero position. Used to sanity-check the runner (should return zero returns)."""
    return [0.0] * len(bars)


def random_binary(bars: list[Bar], seed: int = 0) -> list[float]:
    """Random 0/1 positions from a seeded RNG.

    Coarse "known-null" scaffolding for plan §5.3's harness self-check: after
    cost sim lands in batch 3.2, running this strategy through the harness
    should produce Sharpes statistically indistinguishable from zero. The
    matched-turnover variant used for the formal null-gate test lands in batch
    3.3 alongside the block bootstrap.
    """
    rng = random.Random(seed)
    return [float(rng.randint(0, 1)) for _ in bars]
