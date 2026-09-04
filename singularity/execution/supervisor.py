"""Bracket supervisor — plan §4.2.

Per-symbol stop/target monitor. On startup and periodically, reads the
`brackets` table, fetches latest trade price via MarketDataClient, and
submits a marketable IOC to close the position if a threshold is breached.

Design decisions:

  * One bracket per symbol (not per entry). Multiple entries on the same
    symbol update the bracket to reflect current avg_entry_price.
  * Poll cadence 2s (config). Plan §4.2 says "consumes the trade stream";
    for a long-only spot strategy at retail size, 2s reaction beats the
    complexity of maintaining a WS-subscribed trade cache. Upgrade to
    trade WS if we ever run intraday-fast strategies.
  * Uses market IOC for the exit — plan §4.2 explicitly forbids resting
    stop_limit as the primary exit ("won't fill in a gap-through, and
    crypto gaps through routinely").
  * On trigger: cancel any open orders for the symbol first (avoids
    fighting our own resting orders), then close the position, then
    delete the bracket row.

Wired into the executor as a background task alongside heartbeat +
trade_updates.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from ..adapters.alpaca_crypto.market_data import MarketDataClient
from ..adapters.alpaca_crypto.orders import OrderAdapter
from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..costs.model import BookSnapshot, one_way_cost
from ..costs.types import Cost, OrderIntent, OrderType, Side, TimeInForce
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


class BracketSupervisor:
    def __init__(
        self,
        rest: AlpacaRestClient,
        adapter: OrderAdapter,
        market: MarketDataClient,
        store: StateStore,
        poll_interval_s: float = 2.0,
    ) -> None:
        self._rest = rest
        self._adapter = adapter
        self._market = market
        self._store = store
        self._interval = poll_interval_s
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        log.info("bracket_supervisor_starting", interval_s=self._interval)
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                log.exception("bracket_supervisor_tick_error")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue

    async def _tick(self) -> None:
        brackets = await asyncio.to_thread(self._store.all_brackets)
        for br in brackets:
            symbol = br["symbol"]
            # Confirm we still hold the position — bracket without position is stale
            pos = await asyncio.to_thread(self._store.get_position, symbol)
            if pos is None or abs(pos["qty"]) < 1e-9:
                log.info("bracket_stale_removed", symbol=symbol)
                await asyncio.to_thread(self._store.delete_bracket, symbol)
                continue

            trade = await self._market.latest_trade(symbol)
            if trade is None:
                continue
            price = trade["price"]
            trigger = _decide_trigger(price, br["stop_price"], br["target_price"], pos["qty"])
            if trigger is None:
                continue
            log.critical(
                "bracket_triggered",
                symbol=symbol, trigger=trigger, price=price,
                stop=br["stop_price"], target=br["target_price"], qty=pos["qty"],
            )
            await self._close_position(symbol, pos["qty"])

    async def _close_position(self, symbol: str, qty: float) -> None:
        """Cancel open orders on the symbol, then IOC-close the position via the
        exit-side of the current best.

        Uses the position sign to decide side: long → SELL, short → BUY.
        For long-only spot on Alpaca, `qty > 0` → SELL close.
        """
        # 1. Cancel any resting orders on this symbol so we're not fighting them
        try:
            open_orders = await self._rest.get_orders(status="open")
        except Exception:
            log.exception("bracket_close_get_orders_failed", symbol=symbol)
            open_orders = []
        for o in open_orders:
            if o.get("symbol") == symbol and o.get("id"):
                try:
                    await self._rest.cancel_order(o["id"])
                except Exception:
                    log.exception("bracket_close_cancel_failed", order_id=o.get("id"))

        # 2. Submit IOC close
        side = Side.SELL if qty > 0 else Side.BUY
        q = await self._market.latest_quote(symbol)
        if q is None:
            log.error("bracket_close_no_quote", symbol=symbol)
            return
        book = BookSnapshot(
            best_bid=q["bid_px"], best_ask=q["ask_px"],
            bid_levels=[(q["bid_px"], q["bid_sz"])],
            ask_levels=[(q["ask_px"], q["ask_sz"])],
        )
        cost = one_way_cost(qty=abs(qty), side=side, book=book, is_maker=False)
        # Cross to the far touch to guarantee marketability
        limit = book.best_ask if side is Side.BUY else book.best_bid
        intent = OrderIntent(
            id=f"bracket-exit-{uuid.uuid4().hex[:12]}",
            symbol=symbol, side=side, qty=abs(qty),
            order_type=OrderType.LIMIT, tif=TimeInForce.IOC,
            limit_price=round(limit, 2),
            submitted_at=datetime.now(timezone.utc),
            mid_at_submit=book.mid,
            modeled_cost=cost,
        )
        try:
            await self._adapter.submit(intent)
        except Exception:
            log.exception("bracket_close_submit_failed", intent_id=intent.id)
            return

        # 3. Drop the bracket row — even if the IOC fails to fully fill, we don't
        #    want to re-trigger every 2s. Reconciliation and manual review handle
        #    the residual case.
        await asyncio.to_thread(self._store.delete_bracket, symbol)
        log.info("bracket_close_submitted", intent_id=intent.id, symbol=symbol, qty=qty)


def _decide_trigger(
    price: float, stop: float, target: float, position_qty: float
) -> str | None:
    """Return 'stop', 'target', or None. Sign of position_qty determines direction.

    Long (qty > 0):  stop is BELOW entry, target ABOVE.
                     stop hit if price <= stop; target hit if price >= target.
    Short (qty < 0): stop is ABOVE entry, target BELOW.
                     stop hit if price >= stop; target hit if price <= target.
    """
    if position_qty > 0:
        if price <= stop:
            return "stop"
        if price >= target:
            return "target"
    elif position_qty < 0:
        if price >= stop:
            return "stop"
        if price <= target:
            return "target"
    return None
