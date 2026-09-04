"""Plan §6.1 — blended-lookback TSMOM with asymmetric hysteresis.

    signals = [sign(ret(L) / vol(L)) for L in (30, 60, 90, 180)]  # days
    raw     = mean(signals)
    weight  = hysteresis(raw, enter=+0.25, exit=-0.10)

Design points from the plan:

  * BLENDED lookbacks, never a single one. A single lookback picks a specific
    regime; a blend degrades gracefully across regimes.
  * The hysteresis band IS the cost-aware filter — it prevents whipsawing
    around zero which destroys TSMOM in chop. Tune enter/exit against the
    cost model, not against raw returns.
  * Long-only spot: raw ∈ [-1, +1] but we map to {0, 1} via the hysteresis.
    Fractional positions (0 < w < 1) come later with the vol-target overlay
    in Phase 4.2.
  * The signal at bar t uses close[t] and prior returns only. Position at
    bar t applies to the return earned from t → t+1 (harness convention).
    No look-ahead by construction.

For walk-forward, the caller (harness) MUST pass `warmup_bars >= max(lookbacks)`
so the signal has enough history at the start of each fold's test window.
Otherwise TSMOM emits 0 (stay flat) for the warm-up bars.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable

from ..adapters.alpaca_crypto.history import Bar


DEFAULT_LOOKBACKS: tuple[int, ...] = (30, 60, 90, 180)
DEFAULT_ENTER = 0.25
DEFAULT_EXIT = -0.10


def _daily_returns(closes: list[float]) -> list[float]:
    """Arithmetic close-to-close returns; len == len(closes)-1, index 0 = ret bar 0→1."""
    return [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]


def tsmom_signal(
    bars: list[Bar],
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
) -> list[float]:
    """Compute raw TSMOM signal at each bar.

    Returns list of length len(bars). Value at index t is:
        mean over L in lookbacks of sign(ret_L(t) / vol_L(t))

    Where ret_L(t) = (close[t] - close[t-L]) / close[t-L] and
    vol_L(t) = stdev(daily returns over [t-L+1, t]) × sqrt(L).

    For bars with insufficient history (t < max(lookbacks)), returns 0.0.
    """
    closes = [b.close for b in bars]
    n = len(closes)
    if n < 2:
        return [0.0] * n
    daily_rets = _daily_returns(closes)  # len n-1; index i = return bar i → i+1
    max_L = max(lookbacks)
    raws = [0.0] * n
    for t in range(max_L, n):
        signs: list[float] = []
        for L in lookbacks:
            ret_L = (closes[t] - closes[t - L]) / closes[t - L]
            # daily returns spanning [t-L+1..t] i.e. daily_rets indices [t-L..t-1]
            window = daily_rets[t - L:t]
            if len(window) < 2:
                signs.append(0.0)
                continue
            sd = statistics.stdev(window)
            if sd == 0.0:
                signs.append(0.0)
                continue
            vol_L = sd * math.sqrt(L)
            ratio = ret_L / vol_L
            signs.append(1.0 if ratio > 0.0 else (-1.0 if ratio < 0.0 else 0.0))
        raws[t] = statistics.fmean(signs) if signs else 0.0
    return raws


def hysteresis_positions(
    raw_signal: list[float],
    enter: float = DEFAULT_ENTER,
    exit_: float = DEFAULT_EXIT,
    initial_state: str = "flat",
) -> list[float]:
    """Apply asymmetric hysteresis: enter long when raw > enter, exit when raw < exit_.

    initial_state: "flat" or "long" — the state carried in from before this series.
    The harness slices out the test-window portion after computation, so the
    warmup section naturally initializes state before the test window starts.
    """
    state = initial_state
    positions: list[float] = []
    for r in raw_signal:
        if state == "flat" and r > enter:
            state = "long"
        elif state == "long" and r < exit_:
            state = "flat"
        positions.append(1.0 if state == "long" else 0.0)
    return positions


Strategy = Callable[[list[Bar]], list[float]]


def tsmom(
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    enter: float = DEFAULT_ENTER,
    exit_: float = DEFAULT_EXIT,
) -> Strategy:
    """Strategy factory: blended-lookback TSMOM with asymmetric hysteresis."""
    def strategy(bars: list[Bar]) -> list[float]:
        raws = tsmom_signal(bars, lookbacks)
        return hysteresis_positions(raws, enter=enter, exit_=exit_)
    return strategy


def tsmom_voltarget(
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
    enter: float = DEFAULT_ENTER,
    exit_: float = DEFAULT_EXIT,
    target_vol: float = 0.40,
    vol_lookback: int = 30,
    rebalance_band: float = 0.15,
) -> Strategy:
    """Composition: TSMOM signal (0/1) multiplied by the vol-target overlay.

    Position at bar i = tsmom_signal_i × vol_multiplier_i. Produces fractional
    positions in [0, 1]. When TSMOM says flat, position is 0 regardless of vol
    (multiplication zeros it). When TSMOM says long, position sizing is scaled
    to target vol.
    """
    from ..overlays.voltarget import vol_target_multipliers
    tsmom_strat = tsmom(lookbacks, enter, exit_)

    def strategy(bars: list[Bar]) -> list[float]:
        tsmom_pos = tsmom_strat(bars)
        mult = vol_target_multipliers(
            bars, target_annualized=target_vol,
            vol_lookback=vol_lookback, rebalance_band=rebalance_band,
        )
        return [t * m for t, m in zip(tsmom_pos, mult)]

    return strategy
