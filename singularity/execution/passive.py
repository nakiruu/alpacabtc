"""Passive fill loop — plan §4.1.

    submit limit @ bid (gtc)
      ├─ filled           → done, maker fee
      ├─ t > T1 → cancel, resubmit @ touch (may have moved)
      ├─ t > T2 → cancel, resubmit @ mid
      └─ t > T3 → cross with ioc, accept taker

A one-shot function: given an intent (side, qty, symbol), drive it through
the ladder and return the collected fills. Callers are responsible for the
higher-level decision "should I enter" — this only executes.

Fill notifications come via the shared StateStore: fill_handler in the
executor process writes to state.fills; here we poll state.fills_for(id)
at 500ms cadence. Cross-process because SQLite is happy with WAL + concurrent
readers, and the poll cost is negligible.

Cost model is called on each leg with the current top-of-book so the modeled
cost persisted in the order row reflects reality at submission time.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import httpx

from ..adapters.alpaca_crypto.market_data import MarketDataClient
from ..adapters.alpaca_crypto.orders import OrderAdapter
from ..costs.model import BookSnapshot, one_way_cost
from ..costs.types import Cost, OrderIntent, OrderType, Side, TimeInForce
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


class LadderPhase(str, Enum):
    TOUCH = "touch"          # rest at own touch
    TOUCH_2 = "touch_2"      # replace at (new) touch after T1
    MID = "mid"              # replace at mid after T2
    CROSS_IOC = "cross_ioc"  # cross with IOC after T3


@dataclass
class LadderConfig:
    t1_s: float = 10.0
    t2_s: float = 60.0
    t3_s: float = 180.0
    poll_interval_s: float = 0.5
    ioc_wait_s: float = 5.0
    volume_30d_usd: float = 0.0


@dataclass
class PassiveResult:
    intents: list[OrderIntent] = field(default_factory=list)
    filled_qty: float = 0.0
    avg_price: float = 0.0
    final_phase: LadderPhase | None = None
    timed_out: bool = False
    rejected: bool = False           # 4xx from Alpaca — order never became live
    rejection_reason: str | None = None

    @property
    def is_filled(self) -> bool:
        return self.filled_qty > 0.0

    def add_fill(self, qty: float, price: float) -> None:
        new_qty = self.filled_qty + qty
        if new_qty <= 0:
            return
        self.avg_price = (self.avg_price * self.filled_qty + price * qty) / new_qty
        self.filled_qty = new_qty


class PassiveEntry:
    def __init__(
        self,
        adapter: OrderAdapter,
        market: MarketDataClient,
        store: StateStore,
        config: LadderConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._market = market
        self._store = store
        self._cfg = config or LadderConfig()

    async def execute(self, *, side: Side, symbol: str, qty: float) -> PassiveResult:
        log.info("passive_start", side=side.value, symbol=symbol, qty=qty)
        result = PassiveResult()
        remaining = qty

        try:
            # ----- Phase A: rest at own touch, wait T1 -----
            touch_intent = await self._submit_leg(
                side=side, symbol=symbol, qty=remaining, phase=LadderPhase.TOUCH,
                price_selector=lambda book: book.best_bid if side is Side.BUY else book.best_ask,
                tif=TimeInForce.GTC, is_maker=True,
            )
            if touch_intent is not None:
                result.intents.append(touch_intent)
                phase_fills = await self._collect_fills(touch_intent.id, remaining, self._cfg.t1_s)
                self._accumulate(result, phase_fills)
                remaining = qty - result.filled_qty
            if remaining <= 1e-9:
                result.final_phase = LadderPhase.TOUCH
                log.info("passive_done_at_touch", filled_qty=result.filled_qty)
                return result

            # ----- Phase B: cancel remainder, resubmit at new touch, wait T2-T1 -----
            if touch_intent is not None:
                await self._cancel_quietly(touch_intent.id)
            touch2_intent = await self._submit_leg(
                side=side, symbol=symbol, qty=remaining, phase=LadderPhase.TOUCH_2,
                price_selector=lambda book: book.best_bid if side is Side.BUY else book.best_ask,
                tif=TimeInForce.GTC, is_maker=True,
            )
            if touch2_intent is not None:
                result.intents.append(touch2_intent)
                wait_s = max(0.0, self._cfg.t2_s - self._cfg.t1_s)
                phase_fills = await self._collect_fills(touch2_intent.id, remaining, wait_s)
                self._accumulate(result, phase_fills)
                remaining = qty - result.filled_qty
            if remaining <= 1e-9:
                result.final_phase = LadderPhase.TOUCH_2
                log.info("passive_done_at_touch_2", filled_qty=result.filled_qty)
                return result

            # ----- Phase C: cancel, resubmit at mid, wait T3-T2 -----
            if touch2_intent is not None:
                await self._cancel_quietly(touch2_intent.id)
            mid_intent = await self._submit_leg(
                side=side, symbol=symbol, qty=remaining, phase=LadderPhase.MID,
                price_selector=lambda book: book.mid,
                tif=TimeInForce.GTC, is_maker=True,  # resting at mid still counts as maker if it fills there
            )
            if mid_intent is not None:
                result.intents.append(mid_intent)
                wait_s = max(0.0, self._cfg.t3_s - self._cfg.t2_s)
                phase_fills = await self._collect_fills(mid_intent.id, remaining, wait_s)
                self._accumulate(result, phase_fills)
                remaining = qty - result.filled_qty
            if remaining <= 1e-9:
                result.final_phase = LadderPhase.MID
                log.info("passive_done_at_mid", filled_qty=result.filled_qty)
                return result

            # ----- Phase D: cross with IOC at far touch -----
            if mid_intent is not None:
                await self._cancel_quietly(mid_intent.id)
            ioc_intent = await self._submit_leg(
                side=side, symbol=symbol, qty=remaining, phase=LadderPhase.CROSS_IOC,
                price_selector=lambda book: book.best_ask if side is Side.BUY else book.best_bid,
                tif=TimeInForce.IOC, is_maker=False,
            )
            if ioc_intent is not None:
                result.intents.append(ioc_intent)
                phase_fills = await self._collect_fills(ioc_intent.id, remaining, self._cfg.ioc_wait_s)
                self._accumulate(result, phase_fills)

            result.final_phase = LadderPhase.CROSS_IOC
            result.timed_out = result.filled_qty < qty - 1e-9
            log.info(
                "passive_done_cross_ioc",
                filled_qty=result.filled_qty,
                requested_qty=qty,
                timed_out=result.timed_out,
            )
            return result
        except httpx.HTTPStatusError as e:
            # 4xx = the order was rejected outright (bad qty, insufficient balance,
            # symbol restriction, etc). Escalating through the ladder gains nothing.
            body = e.response.text[:200]
            result.rejected = True
            result.rejection_reason = f"HTTP {e.response.status_code}: {body}"
            log.error(
                "passive_aborted_on_rejection",
                status=e.response.status_code,
                body=body,
                phase=result.final_phase.value if result.final_phase else "startup",
            )
            return result

    async def _submit_leg(
        self,
        *,
        side: Side,
        symbol: str,
        qty: float,
        phase: LadderPhase,
        price_selector,
        tif: TimeInForce,
        is_maker: bool,
    ) -> OrderIntent | None:
        try:
            book = await self._fetch_book(symbol)
        except Exception:
            log.exception("passive_book_fetch_failed", symbol=symbol, phase=phase.value)
            return None
        limit_price = round(price_selector(book), 2)
        modeled = one_way_cost(
            qty=qty, side=side, book=book, is_maker=is_maker,
            volume_30d_usd=self._cfg.volume_30d_usd,
        )
        intent = OrderIntent(
            id=f"passive-{phase.value}-{uuid.uuid4().hex[:12]}",
            symbol=symbol, side=side, qty=qty,
            order_type=OrderType.LIMIT, tif=tif,
            limit_price=limit_price,
            submitted_at=datetime.now(timezone.utc),
            mid_at_submit=book.mid,
            modeled_cost=modeled,
        )
        try:
            await self._adapter.submit(intent)
        except httpx.HTTPStatusError as e:
            # 4xx = validation/auth/permission rejection. Escalating gains nothing —
            # let execute() see it and abort the whole ladder.
            if 400 <= e.response.status_code < 500:
                raise
            log.exception("passive_submit_transient_error",
                          intent_id=intent.id, phase=phase.value,
                          status=e.response.status_code)
            return None
        except Exception:
            log.exception("passive_submit_failed", intent_id=intent.id, phase=phase.value)
            return None
        log.info(
            "passive_submitted",
            intent_id=intent.id, phase=phase.value,
            limit_price=limit_price, mid=book.mid, spread_bps=book.spread_bps,
        )
        return intent

    async def _fetch_book(self, symbol: str) -> BookSnapshot:
        q = await self._market.latest_quote(symbol)
        if q is None:
            raise RuntimeError(f"no latest quote for {symbol}")
        return BookSnapshot(
            best_bid=q["bid_px"], best_ask=q["ask_px"],
            bid_levels=[(q["bid_px"], q["bid_sz"])],
            ask_levels=[(q["ask_px"], q["ask_sz"])],
        )

    async def _collect_fills(
        self, intent_id: str, target_qty: float, timeout_s: float
    ) -> list[dict]:
        """Poll state.fills_for until target_qty filled or timeout. Returns fills seen this phase."""
        deadline = time.monotonic() + timeout_s
        seen: set[str] = set()
        collected: list[dict] = []
        while time.monotonic() < deadline:
            rows = await asyncio.to_thread(self._store.fills_for, intent_id)
            for r in rows:
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                collected.append(dict(r))
            filled = sum(r["qty"] for r in rows)
            if filled >= target_qty - 1e-9:
                return collected
            await asyncio.sleep(self._cfg.poll_interval_s)
        return collected

    def _accumulate(self, result: PassiveResult, fills: list[dict]) -> None:
        for f in fills:
            result.add_fill(f["qty"], f["price"])

    async def _cancel_quietly(self, intent_id: str) -> None:
        try:
            await self._adapter.cancel(intent_id)
        except Exception:
            log.warning("passive_cancel_failed", intent_id=intent_id)
