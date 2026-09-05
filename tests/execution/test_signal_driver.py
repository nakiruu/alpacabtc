"""Signal driver — guard rails and idempotency."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from singularity.costs.types import Cost, OrderIntent, OrderType, Side, TimeInForce
from singularity.execution.signal_driver import (
    ALPACA_MIN_NOTIONAL_USD,
    _already_ticked_today,
    _signal_intent_id,
    _pre_stamp_intent,
)
from singularity.ops.state import StateStore


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


def test_signal_intent_id_encodes_date_and_symbol():
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    intent_id = _signal_intent_id("BTC/USD", Side.BUY, now)
    assert intent_id.startswith("signal-tick-BTC-USD-2026-03-15-")
    # Random suffix means two calls same day produce different ids
    other = _signal_intent_id("BTC/USD", Side.BUY, now)
    assert intent_id != other


def test_already_ticked_today_false_on_empty_store(store):
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    assert _already_ticked_today(store, "BTC/USD", now) is False


def test_already_ticked_today_true_after_prestamp(store):
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    _pre_stamp_intent(store, "BTC/USD", Side.BUY, now)
    assert _already_ticked_today(store, "BTC/USD", now) is True


def test_idempotency_respects_symbol_boundary(store):
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    _pre_stamp_intent(store, "BTC/USD", Side.BUY, now)
    # A different symbol on the same day should NOT be blocked
    assert _already_ticked_today(store, "ETH/USD", now) is False


def test_idempotency_respects_date_boundary(store):
    yesterday = datetime(2026, 3, 14, tzinfo=timezone.utc)
    today = datetime(2026, 3, 15, tzinfo=timezone.utc)
    _pre_stamp_intent(store, "BTC/USD", Side.BUY, yesterday)
    # Same symbol on a different day should NOT be blocked
    assert _already_ticked_today(store, "BTC/USD", today) is False


def test_min_notional_constant_matches_alpaca_documented():
    """If Alpaca changes their minimum, this test alerts us to update guard."""
    assert ALPACA_MIN_NOTIONAL_USD == 10.0


def test_prestamp_persists_as_pending(store):
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    _pre_stamp_intent(store, "BTC/USD", Side.BUY, now)
    # Should now show up in open_orders since it was persisted as PENDING
    open_orders = store.open_orders()
    assert len(open_orders) == 1
    assert open_orders[0]["symbol"] == "BTC/USD"


def test_prestamp_idempotent_via_random_suffix(store):
    """Two prestamps in the same second produce distinct rows (random suffix)."""
    now = datetime(2026, 3, 15, tzinfo=timezone.utc)
    _pre_stamp_intent(store, "BTC/USD", Side.BUY, now)
    _pre_stamp_intent(store, "BTC/USD", Side.BUY, now)
    open_orders = store.open_orders()
    assert len(open_orders) == 2   # two rows, distinct ids
