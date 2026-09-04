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


class OrderStatus(str, Enum):
    """Lifecycle states we track locally. Alpaca has richer status vocabulary
    (accepted, partially_filled, done_for_day, etc.) — the reconcile loop
    maps those onto ours."""

    PENDING = "pending"        # persisted intent, not yet submitted to Alpaca
    SUBMITTED = "submitted"    # Alpaca acknowledged
    PARTIAL = "partial"        # some fills received, order still live
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


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
        """Reconstruct realized cost from a fill, in bps of notional at submission mid.

        Note: we cannot split spread vs impact after the fact without the book at
        fill time, so the combined figure lands in ``spread_bps`` and ``impact_bps``
        is zero. Calibration compares totals only.
        """
        notional_at_mid = self.qty * mid_at_submit
        if self.side is Side.BUY:
            spread_and_impact_bps = (self.price - mid_at_submit) / mid_at_submit * 1e4
        else:
            spread_and_impact_bps = (mid_at_submit - self.price) / mid_at_submit * 1e4

        # Convert fee to quote units. On BTC/USD, a buy is charged in BTC (the received
        # asset); a sell is charged in USD. Dividing raw fee_amount by USD notional
        # would be dimensionally wrong when fee_asset is the base — off by a factor
        # of price.
        base_asset = self.symbol.split("/")[0]
        fee_in_quote = self.fee_amount * self.price if self.fee_asset == base_asset else self.fee_amount
        fee_bps = fee_in_quote / notional_at_mid * 1e4 if notional_at_mid > 0 else 0.0

        return Cost(fee_bps=fee_bps, spread_bps=spread_and_impact_bps, impact_bps=0.0)
