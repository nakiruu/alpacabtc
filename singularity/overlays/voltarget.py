"""Plan §6.2 — banded volatility target overlay.

    w_target = min(1.0, sigma_target / sigma_hat)
    if abs(w_target - w_current) > 0.15:    # banded — do NOT rebalance continuously
        rebalance()

The band matters. Continuous rebalancing on a vol-target overlay generates
significant turnover on tiny vol wiggles, which pays cost with no signal.
The band says: only move exposure when the target shifts meaningfully.

Long-only spot: multiplier is capped at 1.0 (we never leverage). If realized
vol is below target, we're fully invested; the overlay never scales UP.

For BTC/USD with historical realized vol of 60-100% annualized and a target
of 40%, the multiplier typically sits between 0.4 and 0.7. That's a
substantial exposure cut — exactly what the plan §6.2 design wants for a
strategy that entering full-size during high-vol regimes gets whipsawed.
"""

from __future__ import annotations

from ..adapters.alpaca_crypto.history import Bar
from ..features.vol import realized_vol_annualized


def vol_target_multipliers(
    bars: list[Bar],
    target_annualized: float = 0.40,
    vol_lookback: int = 30,
    rebalance_band: float = 0.15,
) -> list[float]:
    """Per-bar multiplier in [0.0, 1.0] that scales strategy exposure to target vol.

    Args:
        bars: input bars; the multiplier at index i is decided from close[i]
            (using bars 0..i to estimate realized vol, no look-ahead).
        target_annualized: desired annualized vol (0.40 = 40%).
        vol_lookback: bars of history used for the rolling stdev.
        rebalance_band: only change the multiplier when the new target differs
            from the current by more than this. Keeps the overlay's own
            turnover low.

    Returns list of length len(bars). During warmup (first vol_lookback bars),
    multiplier is 0.0 (no vol estimate → conservative default of "flat").
    """
    n = len(bars)
    if n < 2:
        return [0.0] * n

    closes = [b.close for b in bars]
    daily_rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, n)]
    # daily_rets has len n-1; index i is the return from bar i to bar i+1
    # (i.e. rolling vol at index i in daily_rets estimates σ AT bar i+1's close).
    vol_series = realized_vol_annualized(daily_rets, vol_lookback)
    # Align to bars: vol at bar i+1 = vol_series[i]; bar 0 has no vol → 0.0
    vol_at_bar = [0.0] + vol_series  # len == n

    out = [0.0] * n
    prev = 0.0
    for i, sigma_hat in enumerate(vol_at_bar):
        target = 0.0 if sigma_hat <= 0.0 else min(1.0, target_annualized / sigma_hat)
        new_mult = target if abs(target - prev) > rebalance_band else prev
        out[i] = new_mult
        prev = new_mult
    return out
