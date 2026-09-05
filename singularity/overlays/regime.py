"""Plan §6.3 — sticky vol-threshold regime gate.

    Sticky HDP-HMM or BOCPD on (return, realized vol, volume). Output
    multiplies gross exposure: final_weight = tsmom * voltarget * regime_gate.
    Risk gate only. The reliable content is "volatility is exploding / my
    training distribution no longer resembles today."

We ship the *simplest* implementation of that idea: detect when current
realized vol exceeds a rolling-baseline vol by a threshold ratio, and
multiply exposure by a risk-off factor (default 0.5) while in that regime.
Sticky exit prevents whipsawing back to full leverage on the first day vol
subsides — you have to stay below threshold for at least `sticky_bars` bars
after entry before we allow the regime to flip back to risk-on.

Explicitly *not* modeled:
  * Return-regime forecasts — plan §6.3 says these compound two error sources
  * HDP-HMM / BOCPD — deferred; the vol-threshold captures the same
    "training distribution no longer looks like today" signal with 20 lines
    instead of 200, and passes the null-gate test empirically before adding
    complexity
  * Volume — Bitstamp OHLC has volume but its scale drifts a lot over years;
    would need per-symbol normalization we haven't built yet

Composition: `regime_gate_multipliers(bars) * vol_target_multipliers(bars)
* tsmom_positions(bars)`. All three are element-wise multiplied — signal
still picks direction, vol-target scales to target vol, regime clamps to
risk-off cap during dangerous windows.
"""

from __future__ import annotations

import statistics

from ..adapters.alpaca_crypto.history import Bar
from ..features.vol import realized_vol_annualized


def _rolling_median(series: list[float], window: int) -> list[float]:
    """Rolling median across `window` bars. First (window-1) entries are 0.0."""
    out = [0.0] * len(series)
    for i in range(window - 1, len(series)):
        window_slice = [x for x in series[i - window + 1:i + 1] if x > 0.0]
        if window_slice:
            out[i] = statistics.median(window_slice)
    return out


def regime_gate_multipliers(
    bars: list[Bar],
    vol_lookback: int = 30,
    baseline_lookback: int = 180,
    vol_threshold_ratio: float = 1.5,
    risk_off_multiplier: float = 0.5,
    sticky_bars: int = 20,
) -> list[float]:
    """Per-bar exposure multiplier from the vol-regime gate.

    Args:
        bars: input bars; multiplier at index i uses only bars[:i+1] (no look-ahead).
        vol_lookback: window for current realized vol estimate.
        baseline_lookback: window for rolling-median baseline vol (typically 4-6×
            vol_lookback so recent spikes don't inflate the baseline).
        vol_threshold_ratio: trigger risk-off when current / baseline > this.
        risk_off_multiplier: exposure while in risk-off (e.g. 0.5 = half position).
        sticky_bars: minimum bars to stay in risk-off after entry; prevents
            single-day vol dips from re-enabling full exposure prematurely.

    Returns list of length len(bars). Values are in {risk_off_multiplier, 1.0}.
    During warmup (before both current and baseline vol are computable) the
    multiplier is 1.0 — we default to *risk-on* rather than risk-off so the
    strategy isn't crippled at fold boundaries with insufficient history.
    """
    n = len(bars)
    if n < 2:
        return [1.0] * n

    closes = [b.close for b in bars]
    daily_rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, n)]

    # realized_vol_annualized returns len == len(daily_rets) == n - 1.
    # Values at index i estimate σ AT bar i+1 (i.e. using daily_rets[..i]).
    current_vol_series = realized_vol_annualized(daily_rets, vol_lookback)
    baseline_series = _rolling_median(current_vol_series, baseline_lookback)

    # Align to bars: prepend 0.0 for bar 0 (no vol estimate).
    current_at_bar = [0.0] + current_vol_series
    baseline_at_bar = [0.0] + baseline_series

    out = [1.0] * n
    risk_off_since: int | None = None

    for i in range(n):
        curr = current_at_bar[i]
        base = baseline_at_bar[i]
        if base <= 0.0:
            # Warmup — no baseline yet; default risk-on.
            out[i] = 1.0
            continue

        ratio = curr / base if curr > 0 else 0.0

        if risk_off_since is None:
            # Currently risk-on — check for entry
            if ratio > vol_threshold_ratio:
                risk_off_since = i
                out[i] = risk_off_multiplier
            else:
                out[i] = 1.0
        else:
            # Currently risk-off — check for exit
            bars_in_regime = i - risk_off_since
            if bars_in_regime >= sticky_bars and ratio < vol_threshold_ratio:
                risk_off_since = None
                out[i] = 1.0
            else:
                out[i] = risk_off_multiplier
    return out
