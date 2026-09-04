"""Order submission — persists intent first, then submits to Alpaca.

The "persist-then-submit" ordering matters: if the process crashes between
persist and submit, reconciliation on restart sees a PENDING order and can
either resubmit (idempotent via client_order_id) or cancel out to a clean state.
If we submitted first and crashed before persisting, we'd have a live order
Alpaca-side and no local record — a stranded order, which is exactly the
Phase 2 gate failure mode.
"""

from __future__ import annotations

import asyncio

import httpx

from ...costs.types import OrderIntent, OrderStatus
from ...logs import get_logger
from ...ops.state import StateStore
from .rest import AlpacaRestClient

log = get_logger(__name__)


class OrderAdapter:
    def __init__(self, rest: AlpacaRestClient, store: StateStore) -> None:
        self._rest = rest
        self._store = store

    async def submit(self, intent: OrderIntent) -> str:
        """Persist intent, submit to Alpaca, mark submitted. Returns Alpaca order id.

        Idempotent per intent.id — the same intent can be safely retried; Alpaca
        rejects duplicate client_order_ids with 422 which we treat as "already live."
        """
        await asyncio.to_thread(self._store.save_intent, intent)
        try:
            resp = await self._rest.submit_order(
                symbol=intent.symbol,
                side=intent.side.value,
                qty=intent.qty,
                order_type=intent.order_type.value,
                tif=intent.tif.value,
                client_order_id=intent.id,
                limit_price=intent.limit_price,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422 and "already exists" in e.response.text.lower():
                log.info("submit_duplicate_client_order_id", intent_id=intent.id)
                # Query orders to find the existing one and pick up its alpaca id
                existing = await self._find_by_client_id(intent.id)
                if existing:
                    await asyncio.to_thread(
                        self._store.mark_submitted, intent.id, existing["id"]
                    )
                    return existing["id"]
                raise
            log.error(
                "submit_failed",
                intent_id=intent.id,
                status=e.response.status_code,
                body=e.response.text[:500],
            )
            await asyncio.to_thread(
                self._store.update_status, intent.id, OrderStatus.REJECTED
            )
            raise

        await asyncio.to_thread(self._store.mark_submitted, intent.id, resp["id"])
        log.info(
            "submit_ok",
            intent_id=intent.id,
            alpaca_order_id=resp["id"],
            symbol=intent.symbol,
            side=intent.side.value,
            qty=intent.qty,
        )
        return resp["id"]

    async def cancel(self, intent_id: str) -> None:
        row = await asyncio.to_thread(self._store.get_order, intent_id)
        if row is None:
            log.warning("cancel_unknown_intent", intent_id=intent_id)
            return
        alpaca_id = row["alpaca_order_id"]
        if alpaca_id:
            try:
                await self._rest.cancel_order(alpaca_id)
            except httpx.HTTPStatusError as e:
                # 422 typically means "already terminal" — safe to mark canceled locally.
                if e.response.status_code == 422:
                    log.info("cancel_already_terminal", intent_id=intent_id)
                else:
                    log.error(
                        "cancel_failed",
                        intent_id=intent_id,
                        status=e.response.status_code,
                    )
                    raise
        await asyncio.to_thread(
            self._store.update_status, intent_id, OrderStatus.CANCELED
        )

    async def _find_by_client_id(self, client_order_id: str) -> dict | None:
        # Alpaca's /v2/orders?client_order_id={id} accepts this filter
        # and returns matching orders regardless of status.
        try:
            orders = await self._rest.get_orders(status="all")
        except httpx.HTTPStatusError:
            return None
        for o in orders:
            if o.get("client_order_id") == client_order_id:
                return o
        return None
