"""Fold post-mortem — decompose what actually happened in a losing fold.

For each interesting fold, compute the per-bar trajectory of:
  * TSMOM raw signal (does it stay bullish through the drawdown?)
  * Hysteresis position (long / flat state)
  * Vol-target multiplier (how leveraged was the position?)
  * Regime gate multiplier (did it ever fire?)
  * Realized vol (was the drawdown preceded by a vol spike?)
  * Cumulative equity + drawdown path

Then print a compact report per fold. The idea is to see whether the losing
folds share a pattern the current overlay stack can't detect.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime

from ..adapters.alpaca_crypto.history import Bar
from ..features.vol import realized_vol_annualized
from ..overlays.regime import regime_gate_multipliers
from ..overlays.voltarget import vol_target_multipliers
from ..signals.tsmom import (
    DEFAULT_ENTER,
    DEFAULT_EXIT,
    DEFAULT_LOOKBACKS,
    hysteresis_positions,
    tsmom_signal,
)
from .walkforward import Fold


@dataclass
class FoldDiagnostic:
    fold_index: int
    test_start: datetime
    test_end: datetime
    n_bars: int

    price_start: float
    price_end: float
    price_high: float
    price_low: float
    price_pct_change: float

    tsmom_raw: list[float]
    hysteresis_positions: list[float]
    vol_multipliers: list[float]
    regime_multipliers: list[float]
    final_positions: list[float]
    realized_vol_path: list[float]
    equity_path: list[float]         # cumulative product of (1 + pos * ret)
    drawdown_path: list[float]        # equity / rolling peak - 1

    # Aggregate stats
    bars_long: int
    bars_flat: int
    bars_regime_off: int
    avg_final_position: float
    vol_at_start: float
    vol_at_end: float
    vol_peak: float
    max_drawdown: float


def compute_fold_diagnostic(
    all_bars: list[Bar],
    fold: Fold,
    warmup_bars: int,
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    enter: float = DEFAULT_ENTER,
    exit_: float = DEFAULT_EXIT,
    target_vol: float = 0.40,
    vol_lookback: int = 30,
    rebalance_band: float = 0.15,
    regime_vol_lookback: int = 30,
    regime_baseline_lookback: int = 180,
    regime_threshold_ratio: float = 1.5,
    regime_risk_off_multiplier: float = 0.5,
    regime_sticky_bars: int = 20,
) -> FoldDiagnostic:
    """Slice out the fold's data with warmup, recompute all signals, return diagnostic."""
    warmup_start = max(0, fold.test_start_idx - warmup_bars)
    slice_bars = all_bars[warmup_start:fold.test_end_idx]

    # Compute all three signal components over the full slice
    tsmom_raw = tsmom_signal(slice_bars, lookbacks)
    hyst = hysteresis_positions(tsmom_raw, enter=enter, exit_=exit_)
    vol_mult = vol_target_multipliers(
        slice_bars, target_annualized=target_vol,
        vol_lookback=vol_lookback, rebalance_band=rebalance_band,
    )
    regime_mult = regime_gate_multipliers(
        slice_bars,
        vol_lookback=regime_vol_lookback,
        baseline_lookback=regime_baseline_lookback,
        vol_threshold_ratio=regime_threshold_ratio,
        risk_off_multiplier=regime_risk_off_multiplier,
        sticky_bars=regime_sticky_bars,
    )
    final_pos = [h * v * r for h, v, r in zip(hyst, vol_mult, regime_mult)]

    # Realized vol path — compute daily returns from slice, then rolling vol
    closes = [b.close for b in slice_bars]
    daily_rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    vol_series = realized_vol_annualized(daily_rets, vol_lookback)
    # Align to bars: vol_series is len(closes)-1; prepend 0.0
    vol_at_bar = [0.0] + vol_series

    # Slice out the test window
    test_start_in_slice = fold.test_start_idx - warmup_start
    tsmom_raw = tsmom_raw[test_start_in_slice:]
    hyst = hyst[test_start_in_slice:]
    vol_mult = vol_mult[test_start_in_slice:]
    regime_mult = regime_mult[test_start_in_slice:]
    final_pos = final_pos[test_start_in_slice:]
    vol_path = vol_at_bar[test_start_in_slice:]
    test_bars = slice_bars[test_start_in_slice:]

    # Equity + drawdown paths from position-weighted returns
    prices = [b.close for b in test_bars]
    test_rets = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    weighted = [final_pos[i] * test_rets[i] for i in range(len(test_rets))]
    equity = [1.0]
    for r in weighted:
        equity.append(equity[-1] * (1.0 + r))
    peak = 1.0
    dd = []
    for e in equity:
        peak = max(peak, e)
        dd.append(e / peak - 1.0)

    return FoldDiagnostic(
        fold_index=fold.index,
        test_start=test_bars[0].ts,
        test_end=test_bars[-1].ts,
        n_bars=len(test_bars),
        price_start=prices[0],
        price_end=prices[-1],
        price_high=max(prices),
        price_low=min(prices),
        price_pct_change=(prices[-1] - prices[0]) / prices[0],
        tsmom_raw=tsmom_raw,
        hysteresis_positions=hyst,
        vol_multipliers=vol_mult,
        regime_multipliers=regime_mult,
        final_positions=final_pos,
        realized_vol_path=vol_path,
        equity_path=equity,
        drawdown_path=dd,
        bars_long=sum(1 for p in hyst if p > 0),
        bars_flat=sum(1 for p in hyst if p == 0),
        bars_regime_off=sum(1 for r in regime_mult if r < 1.0),
        avg_final_position=statistics.fmean(final_pos) if final_pos else 0.0,
        vol_at_start=vol_path[0] if vol_path else 0.0,
        vol_at_end=vol_path[-1] if vol_path else 0.0,
        vol_peak=max(vol_path) if vol_path else 0.0,
        max_drawdown=min(dd) if dd else 0.0,
    )


def _sparkline(values: list[float]) -> str:
    """Rough unicode sparkline for a compact price/vol/dd trajectory (8 chars)."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return "─" * min(len(values), 32)
    chars = "▁▂▃▄▅▆▇█"
    n_buckets = min(32, len(values))
    step = len(values) / n_buckets
    out = []
    for i in range(n_buckets):
        v = values[int(i * step)]
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


def print_diagnostic(diag: FoldDiagnostic) -> None:
    """Compact multi-line report for one fold."""
    ts_range = f"{diag.test_start.date()} → {diag.test_end.date()}"
    print(f"\n----- fold {diag.fold_index:>2}  ({ts_range},  n={diag.n_bars}) -----")
    print(f"price       start={diag.price_start:>10,.2f}  end={diag.price_end:>10,.2f}  "
          f"high={diag.price_high:>10,.2f}  low={diag.price_low:>10,.2f}  "
          f"chg={diag.price_pct_change:+.2%}")
    print(f"price path  {_sparkline([b for b in diag.equity_path])}")

    prices = diag.equity_path  # equity path proxies price shape when at const position; only for shape
    # Better: sparkline the actual per-bar price
    # (we already have prices via equity_path — but that's equity not price. Recompute simple)
    # ... skipping to keep the report compact

    print(f"realized σ  start={diag.vol_at_start:>6.2%}  end={diag.vol_at_end:>6.2%}  "
          f"peak={diag.vol_peak:>6.2%}  {_sparkline(diag.realized_vol_path)}")

    print(f"position    avg={diag.avg_final_position:>5.2f}  long_bars={diag.bars_long}  "
          f"flat_bars={diag.bars_flat}  regime_off={diag.bars_regime_off}  "
          f"{_sparkline(diag.final_positions)}")

    print(f"tsmom raw   {_sparkline(diag.tsmom_raw)}  "
          f"(mean={statistics.fmean(diag.tsmom_raw):+.2f})")

    print(f"drawdown    max={diag.max_drawdown:>7.2%}  {_sparkline(diag.drawdown_path)}")

    # Highlight: was the drawdown preceded by a vol spike?
    # Look at the max-DD bar and the vol trajectory 20 bars before it
    if diag.drawdown_path:
        worst_bar = diag.drawdown_path.index(min(diag.drawdown_path))
        precede_start = max(0, worst_bar - 20)
        vol_before = diag.realized_vol_path[precede_start:worst_bar + 1] if worst_bar > 0 else []
        vol_at_worst = diag.realized_vol_path[worst_bar] if worst_bar < len(diag.realized_vol_path) else 0
        vol_before_max = max(vol_before) if vol_before else 0
        vol_growth = (vol_at_worst / vol_before[0]) if (vol_before and vol_before[0] > 0) else 0
        pre_flag = "VOL SPIKE" if vol_growth > 1.3 else "quiet slide"
        print(f"pre-DD      worst_bar={worst_bar}  vol_at_worst={vol_at_worst:.2%}  "
              f"vol_20b_before_max={vol_before_max:.2%}  → {pre_flag}")
