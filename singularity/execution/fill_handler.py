"""Handles trade_updates events — persists fills, updates order lifecycle,
derives position state, adopts unknown orders inline.

The FK constraint on `fills.order_id → orders.id` means we cannot save a fill
whose order isn't in the state store. If the fill event references an unknown
`client_order_id`, we synthesize a placeholder OrderIntent from the event
payload and adopt it (same shape as `reconcile._adopt_alien_order`). Otherwise
we'd drop fills, which would break both position tracking and the calibration
loop's rolling gate.

Position derivation: signed sum of fills per symbol. Rebuild-on-write keeps the
positions table consistent with the fills log without needing correctness proofs
on incremental math.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ..costs.types import (
    Cost,
    Fill,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


# Alpaca event → our OrderStatus mapping
_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.SUBMITTED,
    "partial_fill": OrderStatus.PARTIAL,
    "fill": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "done_for_day": OrderStatus.CANCELED,
    "replaced": OrderStatus.CANCELED,
}


def _parse_ts(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


class FillHandler:
    """Applies trade_updates events to the state store."""

    def __init__(self, store: StateStore) -> None:
        self._store = store

    async def handle(self, event: dict) -> None:
        """Entry point wired to TradeUpdatesClient.on_event."""
        e = event.get("event")
        order = event.get("order") or {}
        client_id = order.get("client_order_id")
        alpaca_id = order.get("id")
        if not alpaca_id:
            log.warning("trade_update_no_order_id", ev_type=e)
            return

        # Ensure the order exists locally before touching fills
        await asyncio.to_thread(self._ensure_order, order)

        # Persist fill for fill / partial_fill
        if e in ("fill", "partial_fill"):
            await asyncio.to_thread(self._persist_fill, event, order)

        # Update lifecycle status regardless of event type
        new_status = _STATUS_MAP.get(e)
        if new_status is not None and client_id:
            await asyncio.to_thread(self._store.update_status, client_id, new_status)

        # After any status change that could move position, rebuild it
        if e in ("fill", "partial_fill", "canceled", "rejected"):
            symbol = order.get("symbol")
            if symbol:
                await asyncio.to_thread(self._rebuild_position, symbol)

        log.info(
            "trade_update_applied",
            ev_type=e,
            intent_id=client_id,
            alpaca_order_id=alpaca_id,
            symbol=order.get("symbol"),
        )

    # ---- internals ----

    def _ensure_order(self, order: dict) -> None:
        """If order isn't in state, adopt it as a placeholder."""
        client_id = order.get("client_order_id")
        if not client_id:
            log.warning("trade_update_no_client_order_id", order_id=order.get("id"))
            return
        existing = self._store.get_order(client_id)
        if existing is not None:
            # Make sure alpaca_order_id is bound
            if not existing["alpaca_order_id"] and order.get("id"):
                self._store.mark_submitted(client_id, order["id"])
            return
        # Unknown — synthesize a placeholder OrderIntent so the FK holds.
        try:
            intent = OrderIntent(
                id=client_id,
                symbol=order["symbol"],
                side=Side(order["side"]),
                qty=float(order["qty"]),
                order_type=OrderType(order.get("type", "limit")),
                tif=TimeInForce(order.get("time_in_force", "gtc").lower()),
                limit_price=float(order["limit_price"]) if order.get("limit_price") else None,
                submitted_at=_parse_ts(order.get("submitted_at", order.get("created_at", ""))),
                mid_at_submit=0.0,  # unknown for adopted orders
                modeled_cost=Cost(0.0, 0.0, 0.0),
            )
        except (KeyError, ValueError, TypeError) as ex:
            log.error("adopt_alien_order_failed", client_id=client_id, error=str(ex))
            return
        self._store.save_intent(intent)
        if order.get("id"):
            self._store.mark_submitted(client_id, order["id"])
        log.warning("trade_update_adopted_alien_order", client_id=client_id)

    def _persist_fill(self, event: dict, order: dict) -> None:
        client_id = order.get("client_order_id")
        if not client_id:
            return
        try:
            fill = Fill(
                order_id=client_id,
                symbol=order["symbol"],
                side=Side(order["side"]),
                qty=float(event["qty"]),
                price=float(event["price"]),
                filled_at=_parse_ts(event["timestamp"]),
                # fees arrive T+1 via CFEE — leave zero for now; update_fill_fee
                # in calibration.py fills these in when Activities API is polled.
                fee_asset=order["symbol"].split("/")[0],  # placeholder guess
                fee_amount=0.0,
                is_maker=_infer_is_maker(order, event),
            )
        except (KeyError, ValueError, TypeError) as ex:
            log.error("fill_parse_failed", client_id=client_id, error=str(ex))
            return
        self._store.save_fill(fill)

    def _rebuild_position(self, symbol: str) -> None:
        """Sum signed fills for `symbol` and upsert the position row.

        avg_entry_price = sum(qty*price for BUY fills that contribute to current
        long) / sum(qty). Simplified: use volume-weighted average of buys minus
        volume of sells. For a long-only spot strategy the arithmetic is:
            net_qty = Σ(buy_qty) - Σ(sell_qty)
            if net_qty > 0: avg_entry = Σ(buy_qty * buy_price) / Σ(buy_qty)
            else: position closed, avg_entry irrelevant (persist as 0)
        """
        with self._store._connect() as c:
            rows = c.execute(
                "SELECT side, qty, price FROM fills WHERE symbol=?", (symbol,)
            ).fetchall()
        buy_qty = 0.0
        buy_notional = 0.0
        sell_qty = 0.0
        for r in rows:
            if r["side"] == Side.BUY.value:
                buy_qty += r["qty"]
                buy_notional += r["qty"] * r["price"]
            else:
                sell_qty += r["qty"]
        net_qty = buy_qty - sell_qty
        avg_entry = (buy_notional / buy_qty) if (buy_qty > 0 and net_qty > 0) else 0.0
        self._store.upsert_position(symbol, net_qty, avg_entry)


def _infer_is_maker(order: dict, event: dict) -> bool:
    """Alpaca doesn't cleanly label maker/taker on the event.

    Heuristic: a limit order that fills at its own limit price (not crossed
    through) is a maker fill. Market/IOC orders are always takers.
    """
    otype = order.get("type", "").lower()
    if otype in ("market", "stop"):
        return False
    tif = order.get("time_in_force", "").lower()
    if tif == "ioc":
        return False
    # For limit orders: if the fill price matches the limit exactly, we rested → maker.
    lp = order.get("limit_price")
    fp = event.get("price")
    if lp is not None and fp is not None:
        try:
            return abs(float(lp) - float(fp)) < 1e-9
        except (ValueError, TypeError):
            return False
    return True  # default optimistic for GTC limits without price detail
