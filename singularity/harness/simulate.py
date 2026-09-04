"""Plan §5.2 fill simulator — cost-adjust returns via Phase 1's cost model.

We don't have real book snapshots for historical bars — daily OHLCV is what we
have. So we build a *stylized book* per transition: mid = close price, spread
= configured fixed bps, effectively-infinite depth at top level (retail sizes
don't walk the book). This gives us fee + half-spread accounting; impact is
zero for retail on BTC/USD by construction, which matches what live capture
shows on Alpaca crypto.

Cost accounting model:

  * A transition at bar `i` (going from prev_pos to positions[i]) pays a
    one-way cost proportional to |Δpos|. Cost applies to the same bar's
    return-space so `net_returns[i] = pos[i] * raw_returns[i] - drag`.
  * At end-of-fold we assume forced exit to flat, adding one final one-way
    cost. Walk-forward folds are self-contained trades; this makes fold-level
    P&L honest.
  * `is_maker=False` (default) is the conservative assumption — every trade
    pays taker (fee 25 + half spread). Set `is_maker=True` to model the
    passive fill loop's rebate-earning path.

Cost breakdown is accumulated per-component (fee, spread, impact) so callers
can inspect where the drag came from — fee-dominated in low-turnover
strategies, spread-dominated in high-turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..costs.model import BookSnapshot, one_way_cost
from ..costs.types import Cost, Side


@dataclass
class CostConfig:
    """Cost model parameters used by apply_costs.

    Defaults are conservative (taker execution, 3bps stylized BTC spread,
    tier 0 fees). Tune down assumed_spread_bps to match empirical live
    capture once Phase 0 has enough tape. Set enabled=False to short-circuit
    every trade to zero cost (cost-free baseline for sanity checks).
    """
    is_maker: bool = False
    assumed_spread_bps: float = 3.0
    volume_30d_usd: float = 0.0
    enabled: bool = True


@dataclass
class CostBreakdown:
    """Sum of all costs paid across a fold, decomposed by component (bps of notional)."""
    n_trades: int = 0
    total_fee_bps: float = 0.0
    total_spread_bps: float = 0.0
    total_impact_bps: float = 0.0
    turnover: float = 0.0  # sum of |Δpos| across all transitions

    @property
    def total_cost_bps(self) -> float:
        return self.total_fee_bps + self.total_spread_bps + self.total_impact_bps

    def add(self, cost: Cost, delta: float) -> None:
        """Accumulate a single trade's cost, scaled by |Δpos|."""
        self.n_trades += 1
        self.total_fee_bps += cost.fee_bps * delta
        self.total_spread_bps += cost.spread_bps * delta
        self.total_impact_bps += cost.impact_bps * delta
        self.turnover += delta


def stylized_book(mid: float, spread_bps: float) -> BookSnapshot:
    """Synthetic top-of-book: single level each side, effectively infinite depth."""
    half = mid * spread_bps / 2.0 / 1e4
    return BookSnapshot(
        best_bid=mid - half,
        best_ask=mid + half,
        # A very large size acts as infinite depth for retail size checks.
        bid_levels=[(mid - half, 1e9)],
        ask_levels=[(mid + half, 1e9)],
    )


def apply_costs(
    *,
    positions: list[float],
    returns: list[float],
    prices: list[float],
    config: CostConfig | None = None,
) -> tuple[list[float], CostBreakdown]:
    """Return (net_returns, breakdown).

    Args:
        positions: length N — target position at each bar. positions[i] is
            entered at prices[i] (bar i's close) and held until prices[i+1].
        returns: length N-1 — arithmetic returns from bars[i] → bars[i+1].
        prices: length N — reference price at each bar (close). Used to
            build the stylized book at each transition.
        config: cost params. Defaults to conservative taker.

    Semantics: entry into positions[0] pays cost at prices[0]; exit at end
    of series unwinds positions[N-2] back to flat at prices[N-1], paying a
    final one-way cost. positions[N-1] is dropped (no return earned on it).
    """
    if len(prices) != len(positions):
        raise ValueError(
            f"prices ({len(prices)}) and positions ({len(positions)}) length mismatch"
        )
    if len(returns) != len(positions) - 1:
        raise ValueError(
            f"returns ({len(returns)}) must be len(positions)-1 ({len(positions) - 1})"
        )

    cfg = config or CostConfig()
    breakdown = CostBreakdown()
    net: list[float] = []
    prev_pos = 0.0

    for i in range(len(returns)):
        pos_now = positions[i]
        delta = pos_now - prev_pos
        drag = 0.0
        if delta != 0.0 and cfg.enabled:
            side = Side.BUY if delta > 0 else Side.SELL
            book = stylized_book(prices[i], cfg.assumed_spread_bps)
            cost = one_way_cost(
                qty=abs(delta), side=side, book=book,
                is_maker=cfg.is_maker, volume_30d_usd=cfg.volume_30d_usd,
            )
            drag = abs(delta) * cost.total_bps / 1e4
            breakdown.add(cost, abs(delta))
        elif delta != 0.0:
            # Still record turnover for reporting even when cost is disabled
            breakdown.turnover += abs(delta)
            breakdown.n_trades += 1
        net.append(pos_now * returns[i] - drag)
        prev_pos = pos_now

    # Exit at end — return to flat, one final one-way cost
    if prev_pos != 0.0 and net:
        delta = -prev_pos
        if cfg.enabled:
            side = Side.SELL if prev_pos > 0 else Side.BUY
            book = stylized_book(prices[-1], cfg.assumed_spread_bps)
            cost = one_way_cost(
                qty=abs(delta), side=side, book=book,
                is_maker=cfg.is_maker, volume_30d_usd=cfg.volume_30d_usd,
            )
            exit_drag = abs(delta) * cost.total_bps / 1e4
            net[-1] -= exit_drag
            breakdown.add(cost, abs(delta))
        else:
            breakdown.turnover += abs(delta)
            breakdown.n_trades += 1

    return net, breakdown
