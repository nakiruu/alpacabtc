"""Bracket supervisor — stop/target trigger semantics + IOC close side."""

from __future__ import annotations

import pytest

from singularity.adapters.alpaca_crypto.orders import OrderAdapter
from singularity.execution.supervisor import BracketSupervisor, _decide_trigger
from singularity.ops.state import StateStore

from ._fake_rest import FakeMarket, FakeRest


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


# ---- Trigger logic ----

def test_long_stop_hit_when_price_at_or_below():
    assert _decide_trigger(price=49_900, stop=50_000, target=51_000, position_qty=0.01) == "stop"
    assert _decide_trigger(price=50_000, stop=50_000, target=51_000, position_qty=0.01) == "stop"


def test_long_target_hit_when_price_at_or_above():
    assert _decide_trigger(price=51_100, stop=50_000, target=51_000, position_qty=0.01) == "target"
    assert _decide_trigger(price=51_000, stop=50_000, target=51_000, position_qty=0.01) == "target"


def test_long_no_trigger_between_stop_and_target():
    assert _decide_trigger(price=50_500, stop=50_000, target=51_000, position_qty=0.01) is None


def test_short_trigger_signs_reversed():
    # short: stop ABOVE, target BELOW
    assert _decide_trigger(price=51_100, stop=51_000, target=49_000, position_qty=-0.01) == "stop"
    assert _decide_trigger(price=48_900, stop=51_000, target=49_000, position_qty=-0.01) == "target"


def test_flat_position_never_triggers():
    assert _decide_trigger(price=0, stop=50_000, target=51_000, position_qty=0) is None


# ---- Supervisor.run ----

@pytest.mark.asyncio
async def test_tick_ignores_bracket_when_position_flat(store):
    store.upsert_bracket("BTC/USD", stop_price=49_000, target_price=51_000,
                          atr_used=500, k_stop=2, m_target=4, entry_price=50_000)
    # No matching position row → supervisor should delete stale bracket
    rest = FakeRest()
    market = FakeMarket()
    sup = BracketSupervisor(rest, OrderAdapter(rest, store), market, store, poll_interval_s=0.1)
    await sup._tick()
    assert store.get_bracket("BTC/USD") is None
    assert rest.submitted == []


@pytest.mark.asyncio
async def test_tick_no_trigger_within_range(store):
    store.upsert_bracket("BTC/USD", stop_price=49_000, target_price=51_000,
                          atr_used=500, k_stop=2, m_target=4, entry_price=50_000)
    store.upsert_position("BTC/USD", 0.01, 50_000)
    rest = FakeRest()
    market = FakeMarket(trade_price=50_500)
    sup = BracketSupervisor(rest, OrderAdapter(rest, store), market, store, poll_interval_s=0.1)
    await sup._tick()
    assert store.get_bracket("BTC/USD") is not None
    assert rest.submitted == []


@pytest.mark.asyncio
async def test_tick_stop_hit_closes_position_and_removes_bracket(store):
    store.upsert_bracket("BTC/USD", stop_price=49_000, target_price=51_000,
                          atr_used=500, k_stop=2, m_target=4, entry_price=50_000)
    store.upsert_position("BTC/USD", 0.01, 50_000)
    rest = FakeRest()
    market = FakeMarket(trade_price=48_500)  # below stop
    sup = BracketSupervisor(rest, OrderAdapter(rest, store), market, store, poll_interval_s=0.1)
    await sup._tick()
    # IOC close should have been submitted
    assert len(rest.submitted) == 1
    submitted = rest.submitted[0]
    assert submitted["side"] == "sell"      # long → sell to close
    assert submitted["tif"] == "ioc"
    # Bracket removed so we don't retrigger
    assert store.get_bracket("BTC/USD") is None


@pytest.mark.asyncio
async def test_tick_target_hit_closes_position(store):
    store.upsert_bracket("BTC/USD", stop_price=49_000, target_price=51_000,
                          atr_used=500, k_stop=2, m_target=4, entry_price=50_000)
    store.upsert_position("BTC/USD", 0.01, 50_000)
    rest = FakeRest()
    market = FakeMarket(trade_price=51_500)
    sup = BracketSupervisor(rest, OrderAdapter(rest, store), market, store, poll_interval_s=0.1)
    await sup._tick()
    assert len(rest.submitted) == 1
    assert rest.submitted[0]["side"] == "sell"


@pytest.mark.asyncio
async def test_tick_cancels_open_orders_before_closing(store):
    store.upsert_bracket("BTC/USD", stop_price=49_000, target_price=51_000,
                          atr_used=500, k_stop=2, m_target=4, entry_price=50_000)
    store.upsert_position("BTC/USD", 0.01, 50_000)
    rest = FakeRest(
        open_orders=[{"id": "alp-open-1", "symbol": "BTC/USD"}],
    )
    # Force trigger via low market trade price
    market = FakeMarket(trade_price=48_000)
    sup = BracketSupervisor(rest, OrderAdapter(rest, store), market, store, poll_interval_s=0.1)
    await sup._tick()
    assert "alp-open-1" in rest.canceled
