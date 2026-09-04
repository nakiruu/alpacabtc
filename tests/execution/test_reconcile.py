"""Reconciliation semantics — the Phase 2 gate depends on this being right."""

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
from singularity.execution.reconcile import _normalize_symbol, reconcile_once
from singularity.ops.state import StateStore

from ._fake_rest import FakeRest


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


def _local_intent(id="ord-1", alpaca_id="alp-1", status=OrderStatus.SUBMITTED, symbol="BTC/USD"):
    return OrderIntent(
        id=id, symbol=symbol, side=Side.BUY, qty=0.01,
        order_type=OrderType.LIMIT, tif=TimeInForce.GTC, limit_price=50_000.0,
        submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        mid_at_submit=50_100.0,
        modeled_cost=Cost(15.0, -1.0, 0.0),
    )


def _alpaca_order(id="alp-1", client_order_id="ord-1", symbol="BTC/USD",
                  status="new", side="buy", qty="0.01"):
    return {
        "id": id, "client_order_id": client_order_id, "symbol": symbol,
        "status": status, "side": side, "qty": qty, "type": "limit",
        "time_in_force": "gtc", "limit_price": "50000.00",
        "submitted_at": "2026-01-01T00:00:00Z",
    }


# ---- Symbol normalization ----

def test_normalize_symbol_slash_form_unchanged():
    assert _normalize_symbol("BTC/USD") == "BTC/USD"


def test_normalize_symbol_adds_slash():
    assert _normalize_symbol("BTCUSD") == "BTC/USD"
    assert _normalize_symbol("ETHBTC") == "ETH/BTC"
    assert _normalize_symbol("BTCUSDT") == "BTC/USDT"


# ---- Reconcile: clean ----

@pytest.mark.asyncio
async def test_clean_when_everything_matches(store):
    intent = _local_intent()
    store.save_intent(intent)
    store.mark_submitted(intent.id, "alp-1")

    rest = FakeRest(open_orders=[_alpaca_order()])
    diff = await reconcile_once(rest, store)
    assert diff.is_clean
    assert not diff.has_critical


# ---- Reconcile: alien order ----

@pytest.mark.asyncio
async def test_alien_order_is_adopted(store):
    rest = FakeRest(
        open_orders=[_alpaca_order(id="alp-999", client_order_id="ext-1")]
    )
    diff = await reconcile_once(rest, store)
    assert len(diff.alien_orders) == 1
    assert not diff.is_clean
    # After adoption, order exists in state
    row = store.get_order("ext-1")
    assert row is not None
    assert row["alpaca_order_id"] == "alp-999"
    assert row["status"] == OrderStatus.SUBMITTED.value


@pytest.mark.asyncio
async def test_alien_order_without_client_id_gets_synthesized_id(store):
    rest = FakeRest(open_orders=[_alpaca_order(id="alp-x", client_order_id=None)])
    diff = await reconcile_once(rest, store)
    assert len(diff.alien_orders) == 1
    row = store.get_order("adopted:alp-x")
    assert row is not None


# ---- Reconcile: ghost order ----

@pytest.mark.asyncio
async def test_ghost_order_resolved_to_filled(store):
    intent = _local_intent()
    store.save_intent(intent)
    store.mark_submitted(intent.id, "alp-1")

    # Alpaca open list empty; all-orders list shows it filled
    rest = FakeRest(
        open_orders=[],
        all_orders=[_alpaca_order(status="filled")],
    )
    diff = await reconcile_once(rest, store)
    assert len(diff.ghost_orders) == 1
    assert store.get_order("ord-1")["status"] == OrderStatus.FILLED.value


@pytest.mark.asyncio
async def test_ghost_pending_never_submitted_becomes_rejected(store):
    intent = _local_intent()
    store.save_intent(intent)  # PENDING, no alpaca_order_id
    rest = FakeRest()
    diff = await reconcile_once(rest, store)
    assert len(diff.ghost_orders) == 1
    assert store.get_order("ord-1")["status"] == OrderStatus.REJECTED.value


# ---- Reconcile: positions ----

@pytest.mark.asyncio
async def test_alien_position_is_critical_and_not_auto_flattened(store):
    rest = FakeRest(
        positions=[{"symbol": "BTC/USD", "qty": "0.5", "avg_entry_price": "50000"}]
    )
    diff = await reconcile_once(rest, store)
    assert diff.has_critical
    assert not diff.is_clean
    # Critical: MUST NOT close the position automatically
    assert rest.closed_positions == []
    assert rest.cancel_all_calls == 0


@pytest.mark.asyncio
async def test_ghost_position_is_zeroed_locally(store):
    store.upsert_position("BTC/USD", 0.01, 50_000.0)
    rest = FakeRest(positions=[])  # Alpaca says flat
    diff = await reconcile_once(rest, store)
    assert len(diff.ghost_positions) == 1
    row = store.get_position("BTC/USD")
    assert row["qty"] == 0.0
    # Local zeroing MUST NOT touch Alpaca
    assert rest.closed_positions == []


@pytest.mark.asyncio
async def test_symbol_without_slash_matches_local_slash_form(store):
    store.upsert_position("BTC/USD", 0.01, 50_000.0)
    rest = FakeRest(
        positions=[{"symbol": "BTCUSD", "qty": "0.01", "avg_entry_price": "50000"}]
    )
    diff = await reconcile_once(rest, store)
    # Should match by normalized symbol → no alien, no ghost
    assert not diff.alien_positions
    assert not diff.ghost_positions
