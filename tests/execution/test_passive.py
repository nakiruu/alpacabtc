"""Passive fill loop — T1/T2/T3 ladder progression."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from singularity.adapters.alpaca_crypto.orders import OrderAdapter
from singularity.costs.types import Fill, Side
from singularity.execution.passive import LadderConfig, LadderPhase, PassiveEntry
from singularity.ops.state import StateStore

from ._fake_rest import FakeMarket, FakeRest


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


@pytest.fixture
def cfg():
    # tight timings for fast tests
    return LadderConfig(t1_s=0.3, t2_s=0.6, t3_s=1.0, poll_interval_s=0.05, ioc_wait_s=0.3)


def _plant_fill(store, intent_id: str, qty: float, price: float, is_maker: bool = True):
    """Simulate fill_handler having received a fill for this intent."""
    from singularity.costs.types import Side as S
    store.save_fill(Fill(
        order_id=intent_id, symbol="BTC/USD", side=S.BUY, qty=qty, price=price,
        filled_at=datetime.now(timezone.utc),
        fee_asset="BTC", fee_amount=0.0, is_maker=is_maker,
    ))


@pytest.mark.asyncio
async def test_full_fill_at_touch_stops_ladder(store, cfg):
    rest = FakeRest()
    adapter = OrderAdapter(rest, store)
    market = FakeMarket()
    ladder = PassiveEntry(adapter, market, store, config=cfg)

    async def plant_when_submitted():
        # wait for the first intent to appear in state, then plant a fill
        while True:
            open_orders = store.open_orders()
            if open_orders:
                _plant_fill(store, open_orders[0]["id"], qty=0.01, price=49_999.0)
                return
            await asyncio.sleep(0.02)

    plant_task = asyncio.create_task(plant_when_submitted())
    result = await ladder.execute(side=Side.BUY, symbol="BTC/USD", qty=0.01)
    await plant_task

    assert result.filled_qty == pytest.approx(0.01)
    assert result.final_phase is LadderPhase.TOUCH
    assert len(result.intents) == 1


@pytest.mark.asyncio
async def test_no_fill_progresses_to_ioc(store, cfg):
    rest = FakeRest()
    adapter = OrderAdapter(rest, store)
    market = FakeMarket()
    ladder = PassiveEntry(adapter, market, store, config=cfg)

    # No fills planted at all — ladder should walk all four phases
    result = await ladder.execute(side=Side.BUY, symbol="BTC/USD", qty=0.01)

    assert result.final_phase is LadderPhase.CROSS_IOC
    assert result.timed_out is True
    assert result.filled_qty == 0.0
    assert len(result.intents) == 4    # touch, touch_2, mid, cross_ioc


@pytest.mark.asyncio
async def test_partial_fill_at_touch_carries_remainder_forward(store, cfg):
    rest = FakeRest()
    adapter = OrderAdapter(rest, store)
    market = FakeMarket()
    ladder = PassiveEntry(adapter, market, store, config=cfg)

    async def plant_partial_then_stop():
        while True:
            open_orders = store.open_orders()
            if open_orders and open_orders[0]["id"].startswith("passive-touch-"):
                _plant_fill(store, open_orders[0]["id"], qty=0.004, price=49_999.0)
                return
            await asyncio.sleep(0.02)

    plant_task = asyncio.create_task(plant_partial_then_stop())
    result = await ladder.execute(side=Side.BUY, symbol="BTC/USD", qty=0.01)
    await plant_task

    assert result.filled_qty == pytest.approx(0.004)
    # Progressed past TOUCH
    assert result.final_phase is not LadderPhase.TOUCH
    assert len(result.intents) >= 2


@pytest.mark.asyncio
async def test_4xx_rejection_aborts_ladder_immediately(store, cfg):
    """A validation-error rejection should stop the ladder; escalating won't help."""
    rest = FakeRest(submit_status_code=403)  # simulates Alpaca min-notional rejection
    adapter = OrderAdapter(rest, store)
    market = FakeMarket()
    ladder = PassiveEntry(adapter, market, store, config=cfg)

    result = await ladder.execute(side=Side.BUY, symbol="BTC/USD", qty=0.0001)

    assert result.rejected is True
    assert result.rejection_reason is not None and "403" in result.rejection_reason
    # Only ONE submit attempt — no ladder escalation
    assert len(rest.submitted) == 1
    assert result.filled_qty == 0.0


@pytest.mark.asyncio
async def test_prices_derived_from_book(store, cfg):
    rest = FakeRest()
    adapter = OrderAdapter(rest, store)
    market = FakeMarket(bid_px=49_990.0, ask_px=50_010.0)
    ladder = PassiveEntry(adapter, market, store, config=cfg)

    await ladder.execute(side=Side.BUY, symbol="BTC/USD", qty=0.001)

    # Inspect submitted orders to confirm price selection
    intents = store.open_orders()  # all still open — no fills
    # Sort by phase-in-id to check ordering
    by_phase = {row["id"].split("-")[1]: row for row in intents}
    # touch and touch_2 for a BUY should be at bid=49990
    if "touch" in by_phase:
        assert by_phase["touch"]["limit_price"] == pytest.approx(49_990.0)
    # mid = (bid + ask) / 2 = 50000
    if "mid" in by_phase:
        assert by_phase["mid"]["limit_price"] == pytest.approx(50_000.0)
    # cross_ioc for a BUY = ask
    if "cross" in by_phase:  # id template contains "cross_ioc" — split gives ["passive","cross","ioc","<hex>"]
        assert by_phase["cross"]["limit_price"] == pytest.approx(50_010.0)
