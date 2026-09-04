"""Dead-man's-switch semantics — the last line of defense in a 24/7 market."""

from __future__ import annotations

import time

import pytest

from singularity.execution.watchdog import DeadMansSwitch
from singularity.ops.state import StateStore

from ._fake_rest import FakeRest


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "test.db")


@pytest.mark.asyncio
async def test_no_trigger_when_heartbeat_fresh(store):
    store.heartbeat("executor")
    rest = FakeRest(positions=[{"symbol": "BTC/USD", "qty": "0.01"}])
    switch = DeadMansSwitch(rest, store, max_age_s=60.0)
    fired = await switch.check_once()
    assert fired is False
    assert rest.cancel_all_calls == 0
    assert rest.closed_positions == []


@pytest.mark.asyncio
async def test_no_trigger_when_stale_but_flat(store):
    # Simulate a stale heartbeat by using a very tight threshold
    store.heartbeat("executor")
    time.sleep(0.05)
    rest = FakeRest(positions=[])  # flat
    switch = DeadMansSwitch(rest, store, max_age_s=0.01)
    fired = await switch.check_once()
    assert fired is False
    assert rest.cancel_all_calls == 0


@pytest.mark.asyncio
async def test_trigger_when_stale_and_holding(store):
    store.heartbeat("executor")
    time.sleep(0.05)
    rest = FakeRest(positions=[
        {"symbol": "BTC/USD", "qty": "0.01"},
        {"symbol": "ETH/USD", "qty": "0.1"},
    ])
    switch = DeadMansSwitch(rest, store, max_age_s=0.01)
    fired = await switch.check_once()
    assert fired is True
    assert switch.triggered
    assert rest.cancel_all_calls == 1
    assert set(rest.closed_positions) == {"BTC/USD", "ETH/USD"}


@pytest.mark.asyncio
async def test_no_trigger_when_no_heartbeat_yet(store):
    # Executor never wrote a heartbeat — don't fire, just log
    rest = FakeRest(positions=[{"symbol": "BTC/USD", "qty": "0.01"}])
    switch = DeadMansSwitch(rest, store, max_age_s=60.0)
    fired = await switch.check_once()
    assert fired is False
    assert rest.cancel_all_calls == 0
