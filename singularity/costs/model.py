"""Phase 1 — cost model. Fee + spread + impact + fill probability + adverse selection.

All costs are in bps of notional at mid. Signs are consistent:

    fee_bps      always positive
    spread_bps   negative for filled maker (rebate), positive for taker (crossing)
    impact_bps   always positive for taker (walking the book), zero for maker

Composition:
    one_way_cost   → cost of one leg (entry OR exit)
    round_trip_cost → cost of entry + exit under symmetric-book assumption

Higher-level composition (with fill probability, adverse selection, holding-period
alpha) is a caller responsibility — see execution/ and signals/ for that logic.

Plan §3 gate: modeled vs realized within 3 bps on rolling 100-trade window.
Every constant in this module is a knob for calibration.py to turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import fees
from .types import Cost, Side


@dataclass(frozen=True)
class BookSnapshot:
    """Just enough of the book to price a trade. Assemble from orderbook.OrderBook.top() + depth."""

    best_bid: float
    best_ask: float
    bid_levels: list[tuple[float, float]]  # highest-first
    ask_levels: list[tuple[float, float]]  # lowest-first

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread_bps(self) -> float:
        return (self.best_ask - self.best_bid) / self.mid * 1e4


def _vwap_walk(qty: float, levels: list[tuple[float, float]]) -> float:
    """VWAP for a marketable order of size `qty` walking `levels`.

    Returns math.inf if the visible book cannot fill qty. Callers must treat
    infinity as "cannot price from visible data — split the order or wait."
    """
    filled = 0.0
    notional = 0.0
    for px, sz in levels:
        take = min(sz, qty - filled)
        notional += take * px
        filled += take
        if filled >= qty - 1e-12:
            break
    if filled < qty - 1e-12:
        return math.inf
    return notional / qty


def one_way_cost(
    *,
    qty: float,
    side: Side,
    book: BookSnapshot,
    is_maker: bool,
    volume_30d_usd: float = 0.0,
) -> Cost:
    """Cost of a single execution leg."""
    fee_bps = fees.fee_bps(is_maker, volume_30d_usd)
    half_spread_bps = 0.5 * book.spread_bps

    if is_maker:
        # Rest at own touch, get filled at own price → save half spread
        return Cost(fee_bps=fee_bps, spread_bps=-half_spread_bps, impact_bps=0.0)

    # Taker: pay half spread + walk-the-book impact past the touch
    if side is Side.BUY:
        touch = book.best_ask
        levels = book.ask_levels
    else:
        touch = book.best_bid
        levels = book.bid_levels
    vwap = _vwap_walk(qty, levels)
    if math.isinf(vwap):
        impact_bps = math.inf
    else:
        impact_bps = abs(vwap - touch) / book.mid * 1e4
    return Cost(fee_bps=fee_bps, spread_bps=half_spread_bps, impact_bps=impact_bps)


def round_trip_cost(
    *,
    qty: float,
    book: BookSnapshot,
    is_maker: bool,
    volume_30d_usd: float = 0.0,
) -> Cost:
    """Round trip = one BUY + one SELL. Symmetric book assumption.

    For asymmetric fills (e.g. maker entry, taker exit) compose two one_way_cost
    calls yourself.
    """
    buy = one_way_cost(qty=qty, side=Side.BUY, book=book, is_maker=is_maker, volume_30d_usd=volume_30d_usd)
    sell = one_way_cost(qty=qty, side=Side.SELL, book=book, is_maker=is_maker, volume_30d_usd=volume_30d_usd)
    return buy + sell


# ---------------------------------------------------------------------------
# Fill probability — plan §3.2 placeholder until Phase 2 gives us empirical fits
# ---------------------------------------------------------------------------

def fill_prob(offset_bps: float, wait_seconds: float, vol_regime: str | None = None) -> float:
    """Probability a resting order at `offset_bps` inside the touch fills within `wait_seconds`.

    Placeholder shape (plan §3.2 literal): ~60% at the touch by 60s, saturating
    to ~90% at 5 minutes, with an exponential offset penalty. Replace once Phase 2
    logs enough resting orders to fit an empirical curve.
    """
    del vol_regime  # placeholder — vol-regime conditioning waits for real data
    if wait_seconds <= 0:
        return 0.0
    if wait_seconds < 60.0:
        base = 0.6 * (wait_seconds / 60.0)
    else:
        base = 0.6 + 0.3 * min(1.0, (wait_seconds - 60.0) / 240.0)
    offset_penalty = math.exp(-abs(offset_bps) / 5.0)  # ~5 bps half-life
    return max(0.0, min(1.0, base * offset_penalty))


# ---------------------------------------------------------------------------
# Adverse selection cost.
#
# Shape (a) from plan §3.2 — volatility-scaled diffusion:
#     miss_bps = ADVERSE_K * σ_1s * √wait_seconds
#
# Rationale: prices diffuse as σ√t under a martingale null. A missed passive
# order experiences that same diffusion against it on average, because the
# reason it missed is that the market moved through where it was resting.
# Empirically the drift is closer to 0.5–1× the σ√t baseline; we start with
# ADVERSE_K = 0.7 as a defensible mid-range prior.
#
# Alternatives (see git history for options b and c in the original TODO):
#   (b) offset-scaled — captures "picked-off maker" but ignores time-in-book
#   (c) empirical replay — best; unlocked once Phase 2 logs missed orders
#
# calibration.py should compare model output against realized subsequent moves
# on actually-missed orders (Phase 2 wiring) and re-fit ADVERSE_K per symbol
# and possibly per vol regime.
# ---------------------------------------------------------------------------

ADVERSE_K = 0.7


def adverse_selection_cost(
    *,
    side: Side,
    offset_bps: float,
    wait_seconds: float,
    realized_vol_bps_per_sqrt_s: float,
) -> float:
    """Expected cost in bps when a resting passive order fails to fill.

    Args:
        side: intended side of the resting order (unused — model is side-symmetric)
        offset_bps: how far inside the touch the order rested (unused in shape (a))
        wait_seconds: how long it waited before we gave up
        realized_vol_bps_per_sqrt_s: 1-second realized vol scale in bps

    Returns:
        Expected bps of cost from the miss. Positive = adverse.
    """
    del side, offset_bps  # shape (a) doesn't use these
    if wait_seconds <= 0 or realized_vol_bps_per_sqrt_s <= 0:
        return 0.0
    return ADVERSE_K * realized_vol_bps_per_sqrt_s * math.sqrt(wait_seconds)
