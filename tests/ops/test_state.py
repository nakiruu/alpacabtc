"""State store correctness — the durable record backing every reconciliation
and gate check. Errors here directly break the plan §4.3 gate."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from singularity.costs.types import (
    Cost,
    Fill,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)
from singularity.ops.state import StateStore, row_to_intent


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


def _intent(**overrides) -> OrderIntent:
    base = dict(
        id="ord-1",
        symbol="BTC/USD",
        side=Side.BUY,
        qty=0.01,
        order_type=OrderType.LIMIT,
        tif=TimeInForce.GTC,
        limit_price=50_000.0,
        submitted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        mid_at_submit=50_100.0,
        modeled_cost=Cost(fee_bps=15.0, spread_bps=-1.0, impact_bps=0.0),
    )
    base.update(overrides)
    return OrderIntent(**base)


def _fill(order_id="ord-1", **overrides) -> Fill:
    base = dict(
        order_id=order_id,
        symbol="BTC/USD",
        side=Side.BUY,
        qty=0.01,
        price=50_050.0,
        filled_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        fee_asset="BTC",
        fee_amount=0.0000015,
        is_maker=True,
    )
    base.update(overrides)
    return Fill(**base)


# ---- Orders ----

def test_intent_saved_starts_pending(store):
    store.save_intent(_intent())
    row = store.get_order("ord-1")
    assert row is not None
    assert row["status"] == OrderStatus.PENDING.value
    assert row["alpaca_order_id"] is None


def test_mark_submitted_transitions_pending_to_submitted(store):
    store.save_intent(_intent())
    store.mark_submitted("ord-1", alpaca_order_id="abc-123")
    row = store.get_order("ord-1")
    assert row["status"] == OrderStatus.SUBMITTED.value
    assert row["alpaca_order_id"] == "abc-123"


def test_update_status_terminal_transitions(store):
    store.save_intent(_intent())
    store.mark_submitted("ord-1", "abc")
    store.update_status("ord-1", OrderStatus.FILLED)
    assert store.get_order("ord-1")["status"] == OrderStatus.FILLED.value


def test_open_orders_returns_only_non_terminal(store):
    store.save_intent(_intent(id="a"))
    store.save_intent(_intent(id="b"))
    store.save_intent(_intent(id="c"))
    store.mark_submitted("a", "x")
    store.update_status("b", OrderStatus.FILLED)      # terminal
    store.update_status("c", OrderStatus.CANCELED)    # terminal
    open_ids = {row["id"] for row in store.open_orders()}
    assert open_ids == {"a"}


def test_round_trip_intent_reconstruction(store):
    store.save_intent(_intent())
    row = store.get_order("ord-1")
    intent = row_to_intent(row)
    assert intent.symbol == "BTC/USD"
    assert intent.side is Side.BUY
    assert intent.modeled_cost.fee_bps == 15.0
    assert intent.modeled_cost.spread_bps == -1.0


# ---- Fills ----

def test_fill_saved_and_retrievable(store):
    store.save_intent(_intent())
    store.save_fill(_fill())
    fills = store.fills_for("ord-1")
    assert len(fills) == 1
    assert fills[0]["price"] == 50_050.0
    assert fills[0]["is_maker"] == 1


def test_fill_save_is_idempotent(store):
    store.save_intent(_intent())
    f = _fill()
    store.save_fill(f)
    store.save_fill(f)  # same order_id + filled_at → same primary key
    assert len(store.fills_for("ord-1")) == 1


def test_fills_between_time_filter(store):
    store.save_intent(_intent())
    early = _fill(filled_at=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc))
    late = _fill(filled_at=datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc))
    store.save_fill(early)
    store.save_fill(late)
    got = store.fills_between(
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        datetime(2026, 1, 3, tzinfo=timezone.utc),
    )
    assert len(got) == 1
    assert got[0]["price"] == 50_050.0


# ---- Positions ----

def test_upsert_position_creates_then_updates(store):
    store.upsert_position("BTC/USD", 0.01, 50_000.0)
    row = store.get_position("BTC/USD")
    assert row["qty"] == 0.01
    store.upsert_position("BTC/USD", 0.02, 50_100.0)
    row = store.get_position("BTC/USD")
    assert row["qty"] == 0.02
    assert row["avg_entry_price"] == 50_100.0


def test_all_positions_excludes_flat(store):
    store.upsert_position("BTC/USD", 0.01, 50_000.0)
    store.upsert_position("ETH/USD", 0.0, 3_000.0)
    positions = store.all_positions()
    assert {p["symbol"] for p in positions} == {"BTC/USD"}


# ---- Heartbeats ----

def test_heartbeat_upserts(store):
    store.heartbeat("executor")
    first = store.last_heartbeat("executor")
    time.sleep(0.01)
    store.heartbeat("executor")
    second = store.last_heartbeat("executor")
    assert second > first


def test_heartbeat_age_returns_seconds(store):
    store.heartbeat("executor")
    age = store.heartbeat_age_s("executor")
    assert age is not None and age < 1.0


def test_heartbeat_age_none_when_missing(store):
    assert store.heartbeat_age_s("nonexistent") is None


# ---- Foreign key integrity ----

def test_fill_without_order_is_rejected(store):
    # foreign_keys=ON; saving a fill for a non-existent order should raise
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.save_fill(_fill(order_id="does-not-exist"))
