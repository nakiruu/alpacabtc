"""Backtest orchestrator.

Runs a strategy across the walk-forward folds and aggregates per-fold metrics.

Strategy contract (Batch 3.1 — cost-free evaluation):

    def positions(bars) -> list[float]:
        # Return desired position in [0, 1] for each bar, aligned to `bars`.
        # Buy-and-hold: always 1.0. Random: random 0/1. TSMOM (Phase 4): computed.

The fold runner slices bars to the TEST window, invokes the strategy to get
positions, computes weighted per-bar returns, and hands them to metrics.

Cost simulation (spread/impact/fill_prob/adverse_selection) lands in batch 3.2
and will multiply against these raw returns to produce net-of-cost figures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..adapters.alpaca_crypto.history import Bar
from . import metrics as m
from .walkforward import Fold, WalkForwardSplitter


Strategy = Callable[[list[Bar]], list[float]]


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    n_bars: int
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
        s = self.fold_sharpes
        return sum(s) / len(s) if s else 0.0

    @property
    def n_negative_folds(self) -> int:
        return sum(1 for x in self.fold_sharpes if x < 0)


def _bar_returns(bars: list[Bar]) -> list[float]:
    """Close-to-close arithmetic returns. len == len(bars) - 1."""
    return [
        (bars[i].close - bars[i - 1].close) / bars[i - 1].close
        for i in range(1, len(bars))
    ]


def run_fold(strategy: Strategy, bars: list[Bar], fold: Fold, timeframe: str) -> FoldResult:
    """Evaluate `strategy` on the TEST window of `fold`.

    The strategy is called with the test-window bars; the position at bar t
    is applied to the return from t → t+1 (positions are lagged by one bar
    to avoid look-ahead).
    """
    test_bars = bars[fold.test_start_idx:fold.test_end_idx]
    positions = strategy(test_bars)
    if len(positions) != len(test_bars):
        raise ValueError(
            f"strategy returned {len(positions)} positions for {len(test_bars)} bars"
        )
    raw_returns = _bar_returns(test_bars)
    # Lag positions by 1 bar (decision at bar t applies to return t→t+1)
    weighted = [positions[i] * raw_returns[i] for i in range(len(raw_returns))]
    metrics = m.summary(weighted, timeframe=timeframe)
    return FoldResult(fold=fold, n_bars=len(test_bars), metrics=metrics)


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
