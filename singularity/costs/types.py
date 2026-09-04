"""Typed primitives for the cost layer.

All costs are in basis points relative to trade notional. Positive = money out.
Signs are consistent so total = fee + spread + impact for any side.

Notional convention: qty × price at the reference (mid) — not at the fill price.
This keeps spread and impact orthogonal to price level.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"


@dataclass(frozen=True)
class Cost:
    """Per-round-trip cost decomposition, all in bps of notional at mid."""

    fee_bps: float
    spread_bps: float
    impact_bps: float

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.spread_bps + self.impact_bps

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            fee_bps=self.fee_bps + other.fee_bps,
            spread_bps=self.spread_bps + other.spread_bps,
            impact_bps=self.impact_bps + other.impact_bps,
        )


@dataclass(frozen=True)
class OrderIntent:
    """What we asked for. Logged before submission."""

    id: str
    symbol: str
    side: Side
    qty: float
    order_type: OrderType
    tif: TimeInForce
    limit_price: float | None
    submitted_at: datetime
    mid_at_submit: float
    modeled_cost: Cost


@dataclass(frozen=True)
class Fill:
    """What we got. Reconciled from Alpaca fills + CFEE activities."""

    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    filled_at: datetime
    fee_asset: str            # e.g. "BTC" for BTC/USD buy — plan §3: per-asset accounting
    fee_amount: float         # in fee_asset units
    is_maker: bool

    def realized_cost_bps(self, mid_at_submit: float) -> Cost:
        """Reconstruct realized cost from a fill, in bps of notional at submission mid."""
        notional_at_mid = self.qty * mid_at_submit
        # signed price improvement/dis-improvement vs mid, in bps
        if self.side is Side.BUY:
            spread_and_impact_bps = (self.price - mid_at_submit) / mid_at_submit * 1e4
        else:
            spread_and_impact_bps = (mid_at_submit - self.price) / mid_at_submit * 1e4
        # fee in bps of notional-at-mid; per-asset conversion handled by caller if needed
        # (for same-quote pairs like BTC/USD, fee_amount / notional_at_mid works directly
        # when fee_asset happens to be the quote currency — otherwise caller converts)
        fee_bps = self.fee_amount / notional_at_mid * 1e4 if notional_at_mid > 0 else 0.0
        # We can't separate spread from impact without the book at fill time;
        # calibration lumps them and compares to modeled(spread + impact).
        return Cost(fee_bps=fee_bps, spread_bps=spread_and_impact_bps, impact_bps=0.0)
