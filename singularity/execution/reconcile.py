"""Reconciliation — plan §4.3.

    positions = GET /v2/positions
    orders    = GET /v2/orders?status=open
    intent    = state_store.load()
    diff      = reconcile(positions, orders, intent)
    if diff: alert + repair, do not trade until clean

Categorization of diffs:

  alien_orders     Alpaca has an open order; we have no record of it.
                   Repair: adopt into local orders as SUBMITTED (with placeholder
                   modeled cost) so subsequent lifecycle transitions have a home.
                   Alert: WARN — someone/something submitted outside our path.

  ghost_orders     We think an order is PENDING or SUBMITTED; Alpaca doesn't
                   list it as open. It probably terminal-ed while we were down.
                   Repair: query the full order by client_order_id; update local
                   status to whatever Alpaca reports (filled / canceled / rejected).

  alien_positions  Alpaca reports a position; we have no local record.
                   Repair: NONE. Never auto-flatten a real position from a
                   startup diff — that's the failure mode the watchdog exists to
                   handle. Log CRITICAL and mark reconcile has_critical → executor
                   halts.

  ghost_positions  We think we hold; Alpaca says zero.
                   Repair: zero the local position with a WARN. Almost certainly
                   the position closed while we were down (e.g., liquidation).

CLI: `reconcile` (registered in pyproject.toml). Exits:
    0  clean
    1  diffs found and repaired
    2  critical diff (alien position); manual intervention required
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..costs.types import Cost, OrderIntent, OrderStatus, OrderType, Side, TimeInForce
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


def _normalize_symbol(s: str) -> str:
    """Alpaca sometimes returns crypto symbols with a slash, sometimes without.
    Normalize to slash form used throughout the codebase."""
    if "/" in s:
        return s
    # Split off known quote assets (USD, USDT, USDC, BTC, ETH)
    for quote in ("USDT", "USDC", "USD", "BTC", "ETH"):
        if s.endswith(quote) and len(s) > len(quote):
            return f"{s[:-len(quote)]}/{quote}"
    return s


@dataclass
class ReconcileDiff:
    alien_orders: list[dict] = field(default_factory=list)
    ghost_orders: list[dict] = field(default_factory=list)
    alien_positions: list[dict] = field(default_factory=list)
    ghost_positions: list[dict] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.alien_orders
            or self.ghost_orders
            or self.alien_positions
            or self.ghost_positions
        )

    @property
    def has_critical(self) -> bool:
        return bool(self.alien_positions)

    def print(self) -> None:
        print("\n=== reconcile report ===\n")
        print(f"alien_orders    : {len(self.alien_orders)}")
        print(f"ghost_orders    : {len(self.ghost_orders)}")
        print(f"alien_positions : {len(self.alien_positions)}  {'[CRITICAL]' if self.alien_positions else ''}")
        print(f"ghost_positions : {len(self.ghost_positions)}")
        if self.repairs:
            print("\nrepairs:")
            for r in self.repairs:
                print(f"  - {r}")
        if self.is_clean:
            print("\nstate: CLEAN")
        elif self.has_critical:
            print("\nstate: CRITICAL — manual intervention required")
        else:
            print("\nstate: REPAIRED")


def _adopt_alien_order(store: StateStore, o: dict) -> str:
    """Build a placeholder OrderIntent for an unknown Alpaca order and persist it."""
    intent = OrderIntent(
        id=o.get("client_order_id") or f"adopted:{o['id']}",
        symbol=_normalize_symbol(o["symbol"]),
        side=Side(o["side"]),
        qty=float(o["qty"]),
        order_type=OrderType(o["type"]),
        tif=TimeInForce(o.get("time_in_force", "gtc").lower()),
        limit_price=float(o["limit_price"]) if o.get("limit_price") else None,
        submitted_at=datetime.fromisoformat(o["submitted_at"].replace("Z", "+00:00")),
        mid_at_submit=0.0,  # unknown — modeled cost is meaningless for adopted orders
        modeled_cost=Cost(0.0, 0.0, 0.0),
    )
    store.save_intent(intent)
    store.mark_submitted(intent.id, o["id"])
    return intent.id


async def _resolve_ghost_order(rest: AlpacaRestClient, store: StateStore, local_row: dict) -> str | None:
    """Query Alpaca for the true state of an order we think is live but isn't in the open list."""
    if not local_row["alpaca_order_id"]:
        # Never made it to Alpaca — mark as rejected/canceled locally.
        store.update_status(local_row["id"], OrderStatus.REJECTED)
        return "rejected"
    try:
        orders = await rest.get_orders(status="all")
    except httpx.HTTPStatusError:
        return None
    for o in orders:
        if o["id"] == local_row["alpaca_order_id"]:
            status_map = {
                "filled": OrderStatus.FILLED,
                "partially_filled": OrderStatus.PARTIAL,
                "canceled": OrderStatus.CANCELED,
                "expired": OrderStatus.CANCELED,
                "rejected": OrderStatus.REJECTED,
                "done_for_day": OrderStatus.CANCELED,
                "replaced": OrderStatus.CANCELED,
            }
            new_status = status_map.get(o["status"], OrderStatus.CANCELED)
            store.update_status(local_row["id"], new_status)
            return new_status.value
    # Alpaca has never heard of it — mark rejected
    store.update_status(local_row["id"], OrderStatus.REJECTED)
    return "rejected"


async def reconcile_once(rest: AlpacaRestClient, store: StateStore) -> ReconcileDiff:
    """Query Alpaca, compare against local state, repair what's safe, return diff."""
    alpaca_orders = await rest.get_orders(status="open")
    alpaca_positions = await rest.get_positions()
    local_orders = await asyncio.to_thread(store.open_orders)
    local_positions = await asyncio.to_thread(store.all_positions)

    diff = ReconcileDiff()

    # ---- Orders ----
    local_alpaca_ids = {row["alpaca_order_id"] for row in local_orders if row["alpaca_order_id"]}
    local_client_ids = {row["id"] for row in local_orders}
    alpaca_ids = {o["id"] for o in alpaca_orders}
    alpaca_client_ids = {o.get("client_order_id") for o in alpaca_orders if o.get("client_order_id")}

    # Alien: at Alpaca, not in local (neither by alpaca_id nor client_id)
    for o in alpaca_orders:
        if o["id"] not in local_alpaca_ids and o.get("client_order_id") not in local_client_ids:
            diff.alien_orders.append(o)
            adopted_id = await asyncio.to_thread(_adopt_alien_order, store, o)
            diff.repairs.append(f"adopted alien order {o['id']} as {adopted_id}")
            log.warning("reconcile_alien_order_adopted", alpaca_id=o["id"], client_order_id=o.get("client_order_id"))

    # Ghost: in local PENDING/SUBMITTED/PARTIAL, not in Alpaca open list
    for row in local_orders:
        if row["alpaca_order_id"] and row["alpaca_order_id"] not in alpaca_ids:
            diff.ghost_orders.append(dict(row))
            resolved = await _resolve_ghost_order(rest, store, dict(row))
            diff.repairs.append(f"resolved ghost order {row['id']} → {resolved}")
            log.warning("reconcile_ghost_order_resolved", intent_id=row["id"], resolved=resolved)
        elif not row["alpaca_order_id"] and row["status"] == OrderStatus.PENDING.value:
            # Never made it to Alpaca — mark rejected
            diff.ghost_orders.append(dict(row))
            await asyncio.to_thread(store.update_status, row["id"], OrderStatus.REJECTED)
            diff.repairs.append(f"marked pending-never-submitted {row['id']} as rejected")
            log.warning("reconcile_ghost_pending_rejected", intent_id=row["id"])

    # ---- Positions ----
    alpaca_pos_by_symbol = {_normalize_symbol(p["symbol"]): p for p in alpaca_positions}
    local_pos_by_symbol = {row["symbol"]: dict(row) for row in local_positions}

    for sym, p in alpaca_pos_by_symbol.items():
        if sym not in local_pos_by_symbol:
            diff.alien_positions.append(p)
            log.critical(
                "reconcile_alien_position",
                symbol=sym,
                qty=p.get("qty"),
                avg_entry_price=p.get("avg_entry_price"),
                note="refusing to auto-flatten from startup diff; manual intervention required",
            )

    for sym, row in local_pos_by_symbol.items():
        if sym not in alpaca_pos_by_symbol:
            diff.ghost_positions.append(row)
            await asyncio.to_thread(store.upsert_position, sym, 0.0, row["avg_entry_price"])
            diff.repairs.append(f"zeroed ghost position {sym}")
            log.warning("reconcile_ghost_position_zeroed", symbol=sym, prev_qty=row["qty"])

    return diff


async def _run() -> int:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)
    store = StateStore(Path(settings.state_db_path))
    async with AlpacaRestClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_trading_url,
    ) as rest:
        diff = await reconcile_once(rest, store)
    diff.print()
    if diff.has_critical:
        return 2
    if not diff.is_clean:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile local state against Alpaca")
    parser.parse_args()
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
