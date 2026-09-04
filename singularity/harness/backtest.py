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
from .simulate import CostBreakdown, CostConfig, apply_costs
from .walkforward import Fold, WalkForwardSplitter


Strategy = Callable[[list[Bar]], list[float]]


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    n_test_bars: int          # bars in the test window
    n_return_bars: int        # bars we earned returns on (n_test_bars - 1)
    gross_metrics: m.Metrics
    net_metrics: m.Metrics
    cost_breakdown: CostBreakdown
    net_returns: list[float]  # kept so downstream can compute DSR / bootstrap

    @property
    def metrics(self) -> m.Metrics:
        """Back-compat: unqualified `metrics` refers to net (cost-honest)."""
        return self.net_metrics


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    per_fold: list[FoldResult]
    cost_config: CostConfig

    @property
    def n_folds(self) -> int:
        return len(self.per_fold)

    @property
    def fold_sharpes_gross(self) -> list[float]:
        return [f.gross_metrics.annualized_sharpe for f in self.per_fold]

    @property
    def fold_sharpes_net(self) -> list[float]:
        return [f.net_metrics.annualized_sharpe for f in self.per_fold]

    @property
    def fold_sharpes(self) -> list[float]:
        """Back-compat: net Sharpes."""
        return self.fold_sharpes_net

    @property
    def mean_sharpe_gross(self) -> float:
        return statistics.fmean(self.fold_sharpes_gross) if self.per_fold else 0.0

    @property
    def mean_sharpe_net(self) -> float:
        return statistics.fmean(self.fold_sharpes_net) if self.per_fold else 0.0

    @property
    def mean_sharpe(self) -> float:
        return self.mean_sharpe_net

    @property
    def n_negative_folds(self) -> int:
        return sum(1 for x in self.fold_sharpes_net if x < 0)

    @property
    def total_turnover(self) -> float:
        return sum(f.cost_breakdown.turnover for f in self.per_fold)


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


def run_fold(
    strategy: Strategy,
    bars: list[Bar],
    fold: Fold,
    timeframe: str,
    cost_config: CostConfig | None = None,
) -> FoldResult:
    """Evaluate `strategy` on the TEST window of `fold`.

    positions[i] applies to the return earned from bars[i].close → bars[i+1].close.
    positions[-1] is dropped (no bar after it to earn a return on). Cost is
    charged at each position change and once more on forced-exit-to-flat at the
    end of the fold.
    """
    test_bars = bars[fold.test_start_idx:fold.test_end_idx]
    positions = strategy(test_bars)
    if len(positions) != len(test_bars):
        raise ValueError(
            f"strategy returned {len(positions)} positions for {len(test_bars)} bars"
        )
    raw_returns = _bar_returns(test_bars)
    gross = [p * r for p, r in zip(positions, raw_returns)]
    prices = [b.close for b in test_bars]
    net, breakdown = apply_costs(
        positions=positions, returns=raw_returns, prices=prices, config=cost_config,
    )
    return FoldResult(
        fold=fold,
        n_test_bars=len(test_bars),
        n_return_bars=len(gross),
        gross_metrics=m.summary(gross, timeframe=timeframe),
        net_metrics=m.summary(net, timeframe=timeframe),
        cost_breakdown=breakdown,
        net_returns=net,
    )


def run_backtest(
    *,
    strategy: Strategy,
    strategy_name: str,
    bars: list[Bar],
    splitter: WalkForwardSplitter,
    symbol: str,
    timeframe: str = "1Day",
    cost_config: CostConfig | None = None,
) -> BacktestResult:
    folds = splitter.folds(len(bars))
    cfg = cost_config or CostConfig()
    per_fold = [run_fold(strategy, bars, f, timeframe, cfg) for f in folds]
    return BacktestResult(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        per_fold=per_fold,
        cost_config=cfg,
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
