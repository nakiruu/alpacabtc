"""FillHandler applies trade_updates events to state — inline order adoption,
fill persistence, position derivation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from singularity.costs.types import (
    Cost,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from singularity.execution.fill_handler import FillHandler
from singularity.ops.state import StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


@pytest.fixture
def handler(store):
    return FillHandler(store)


def _order_payload(client_id="ord-1", alpaca_id="alp-1", symbol="BTC/USD",
                    side="buy", qty="0.01", limit_price="50000",
                    order_type="limit", tif="gtc"):
    return {
        "id": alpaca_id,
        "client_order_id": client_id,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "filled_qty": "0",
        "limit_price": limit_price,
        "type": order_type,
        "time_in_force": tif,
        "submitted_at": "2026-01-01T00:00:00Z",
        "status": "new",
    }


def _fill_event(event="fill", price="50000", qty="0.01", **order_overrides):
    return {
        "event": event,
        "order": _order_payload(**order_overrides),
        "price": price,
        "qty": qty,
        "timestamp": "2026-01-01T00:00:05Z",
    }


# ---- Known-order path ----

@pytest.mark.asyncio
async def test_fill_for_known_order_persists(store, handler):
    intent = OrderIntent(
        id="ord-1", symbol="BTC/USD", side=Side.BUY, qty=0.01,
        order_type=OrderType.LIMIT, tif=TimeInForce.GTC, limit_price=50_000.0,
        submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        mid_at_submit=50_100.0,
        modeled_cost=Cost(15.0, -1.0, 0.0),
    )
    store.save_intent(intent)
    store.mark_submitted("ord-1", "alp-1")

    await handler.handle(_fill_event())

    fills = store.fills_for("ord-1")
    assert len(fills) == 1
    assert fills[0]["price"] == 50_000.0
    assert store.get_order("ord-1")["status"] == OrderStatus.FILLED.value


# ---- Unknown-order path: inline adoption ----

@pytest.mark.asyncio
async def test_fill_for_unknown_order_adopts_and_persists(store, handler):
    event = _fill_event(client_id="ext-1", alpaca_id="alp-999")
    await handler.handle(event)

    row = store.get_order("ext-1")
    assert row is not None
    assert row["alpaca_order_id"] == "alp-999"
    assert store.fills_for("ext-1")[0]["price"] == 50_000.0


# ---- Position derivation ----

@pytest.mark.asyncio
async def test_buy_fill_creates_long_position(store, handler):
    await handler.handle(_fill_event(client_id="ord-1", price="50000", qty="0.01"))
    pos = store.get_position("BTC/USD")
    assert pos is not None
    assert pos["qty"] == pytest.approx(0.01)
    assert pos["avg_entry_price"] == pytest.approx(50_000.0)


@pytest.mark.asyncio
async def test_buy_then_partial_sell_reduces_position(store, handler):
    await handler.handle(_fill_event(client_id="ord-buy", price="50000", qty="0.01"))
    await handler.handle(
        _fill_event(client_id="ord-sell", price="51000", qty="0.004", side="sell")
    )
    pos = store.get_position("BTC/USD")
    assert pos["qty"] == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_flatten_by_sell_zeroes_position(store, handler):
    await handler.handle(_fill_event(client_id="ord-buy", price="50000", qty="0.01"))
    await handler.handle(
        _fill_event(client_id="ord-sell", price="51000", qty="0.01", side="sell")
    )
    pos = store.get_position("BTC/USD")
    assert pos["qty"] == pytest.approx(0.0)
    assert pos["avg_entry_price"] == 0.0


# ---- Lifecycle events without fill ----

@pytest.mark.asyncio
async def test_new_event_marks_submitted(store, handler):
    event = {
        "event": "new",
        "order": _order_payload(),
        "timestamp": "2026-01-01T00:00:00Z",
    }
    await handler.handle(event)
    row = store.get_order("ord-1")
    assert row["status"] == OrderStatus.SUBMITTED.value


@pytest.mark.asyncio
async def test_canceled_event_marks_canceled(store, handler):
    event = {
        "event": "canceled",
        "order": _order_payload(),
        "timestamp": "2026-01-01T00:00:00Z",
    }
    await handler.handle(event)
    row = store.get_order("ord-1")
    assert row["status"] == OrderStatus.CANCELED.value


# ---- Maker/taker inference ----

@pytest.mark.asyncio
async def test_ioc_order_marked_taker(store, handler):
    event = _fill_event(tif="ioc")
    await handler.handle(event)
    fills = store.fills_for("ord-1")
    assert fills[0]["is_maker"] == 0


@pytest.mark.asyncio
async def test_limit_at_own_price_marked_maker(store, handler):
    # limit_price 50000, fill price 50000 → rested → maker
    event = _fill_event(limit_price="50000", price="50000")
    await handler.handle(event)
    fills = store.fills_for("ord-1")
    assert fills[0]["is_maker"] == 1


@pytest.mark.asyncio
async def test_limit_crossed_marked_taker(store, handler):
    # limit at 50100 but filled at 50000 → crossed → taker
    event = _fill_event(limit_price="50100", price="50000")
    await handler.handle(event)
    fills = store.fills_for("ord-1")
    assert fills[0]["is_maker"] == 0


# ---- update_fill_fee ----

def test_update_fill_fee_populates_columns(store):
    intent = OrderIntent(
        id="ord-1", symbol="BTC/USD", side=Side.BUY, qty=0.01,
        order_type=OrderType.LIMIT, tif=TimeInForce.GTC, limit_price=50_000.0,
        submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        mid_at_submit=50_100.0,
        modeled_cost=Cost(15.0, -1.0, 0.0),
    )
    store.save_intent(intent)
    from singularity.costs.types import Fill
    filled_at = datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc)
    store.save_fill(Fill(
        order_id="ord-1", symbol="BTC/USD", side=Side.BUY, qty=0.01, price=50_000.0,
        filled_at=filled_at, fee_asset="BTC", fee_amount=0.0, is_maker=True,
    ))
    store.update_fill_fee("ord-1", filled_at, "BTC", 0.0000015)
    fills = store.fills_for("ord-1")
    assert fills[0]["fee_amount"] == pytest.approx(0.0000015)
