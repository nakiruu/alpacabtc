"""Fill simulator — cost math on constructed positions/returns."""

from __future__ import annotations

import pytest

from singularity.harness.simulate import (
    CostBreakdown,
    CostConfig,
    apply_costs,
    stylized_book,
)


def test_stylized_book_symmetry():
    b = stylized_book(mid=50_000.0, spread_bps=2.0)
    assert b.best_bid == pytest.approx(50_000.0 - 5.0)
    assert b.best_ask == pytest.approx(50_000.0 + 5.0)
    assert b.spread_bps == pytest.approx(2.0)


def test_no_position_changes_no_cost():
    """Flat position throughout → zero trades, zero cost drag."""
    positions = [0.0, 0.0, 0.0, 0.0]
    returns = [0.01, -0.02, 0.03]
    prices = [100, 100, 100, 100]
    net, br = apply_costs(positions=positions, returns=returns, prices=prices)
    assert net == [0.0, 0.0, 0.0]
    assert br.n_trades == 0
    assert br.turnover == 0.0
    assert br.total_cost_bps == 0.0


def test_buy_and_hold_pays_exactly_two_one_way_costs():
    """[1, 1, 1] with flat before/after = 1 entry + 1 exit = 2 one-way trades."""
    positions = [1.0, 1.0, 1.0]
    returns = [0.01, 0.02]
    prices = [100, 101, 103]
    net, br = apply_costs(positions=positions, returns=returns, prices=prices)
    assert br.n_trades == 2
    assert br.turnover == pytest.approx(2.0)


def test_taker_cost_matches_fee_plus_half_spread():
    """Default is_maker=False → fee 25 + half spread 1.5 (at 3bps) = 26.5 bps per trade."""
    positions = [1.0]  # single entry, then exit at end
    returns: list[float] = []
    prices = [100.0]
    # With no returns to earn on, net = [] but breakdown still records
    # (need at least one return for the entry trade to appear)
    positions2 = [1.0, 1.0]
    returns2 = [0.0]
    prices2 = [100.0, 100.0]
    _, br = apply_costs(positions=positions2, returns=returns2, prices=prices2)
    # 2 trades × 26.5 bps × delta=1.0 = 53 bps total
    assert br.total_cost_bps == pytest.approx(53.0, rel=1e-3)
    assert br.total_fee_bps == pytest.approx(50.0)  # 25 * 2
    assert br.total_spread_bps == pytest.approx(3.0)  # 1.5 * 2


def test_maker_rebate_produces_negative_spread_bps():
    """Maker rebate = -half_spread; fee is still positive."""
    positions = [1.0, 1.0]
    returns = [0.0]
    prices = [100.0, 100.0]
    cfg = CostConfig(is_maker=True, assumed_spread_bps=3.0)
    _, br = apply_costs(positions=positions, returns=returns, prices=prices, config=cfg)
    # fee 15 * 2 = 30; spread -1.5 * 2 = -3; total = 27 bps
    assert br.total_fee_bps == pytest.approx(30.0)
    assert br.total_spread_bps == pytest.approx(-3.0)
    assert br.total_cost_bps == pytest.approx(27.0)


def test_partial_position_scales_cost_by_delta():
    """Going from 0 to 0.5 pays half the cost of going 0 to 1.0."""
    p_full = [1.0, 1.0]
    p_half = [0.5, 0.5]
    returns = [0.0]
    prices = [100.0, 100.0]
    _, br_full = apply_costs(positions=p_full, returns=returns, prices=prices)
    _, br_half = apply_costs(positions=p_half, returns=returns, prices=prices)
    # Half turnover → half cost
    assert br_half.total_cost_bps == pytest.approx(br_full.total_cost_bps * 0.5, rel=1e-3)


def test_high_turnover_strategy_bleeds_as_expected():
    """Alternating positions 0/1 pays cost on every bar transition."""
    positions = [1.0, 0.0, 1.0, 0.0, 1.0]  # 4 transitions in-window + 0 exit (already flat)
    returns = [0.0, 0.0, 0.0, 0.0]
    prices = [100.0, 100.0, 100.0, 100.0, 100.0]
    _, br = apply_costs(positions=positions, returns=returns, prices=prices)
    # Transitions: 0→1, 1→0, 0→1, 1→0 = 4 trades, each |Δ|=1, plus no exit (ended flat)
    assert br.n_trades == 4
    assert br.turnover == pytest.approx(4.0)
    # 4 × 26.5 = 106 bps
    assert br.total_cost_bps == pytest.approx(106.0, rel=1e-3)


def test_zero_spread_maker_gives_just_fee():
    positions = [1.0, 1.0]
    returns = [0.0]
    prices = [100.0, 100.0]
    cfg = CostConfig(is_maker=True, assumed_spread_bps=0.0)
    _, br = apply_costs(positions=positions, returns=returns, prices=prices, config=cfg)
    assert br.total_cost_bps == pytest.approx(30.0)  # fee 15 * 2 trades


def test_apply_costs_length_validation():
    with pytest.raises(ValueError, match="length mismatch"):
        apply_costs(positions=[1.0, 1.0], returns=[0.01], prices=[100.0])  # prices too short
    with pytest.raises(ValueError, match="len\\(positions\\)-1"):
        apply_costs(positions=[1.0, 1.0, 1.0], returns=[0.01], prices=[100.0, 101.0, 102.0])


def test_cost_drag_matches_hand_calculation():
    """A known scenario end-to-end: buy at 100, price stays flat, exit at 100.
    With default taker cost 26.5 bps × 2 trades = 53 bps drag.
    Net return should be -0.0053."""
    positions = [1.0, 1.0]
    returns = [0.0]
    prices = [100.0, 100.0]
    net, br = apply_costs(positions=positions, returns=returns, prices=prices)
    # Entry at bar 0 drags -26.5bps → -0.00265 from bar 0's net
    # Exit at end drags another -26.5bps applied to net[-1]
    # Total drag = -0.0053
    assert sum(net) == pytest.approx(-0.0053, rel=1e-3)


def test_disabled_cost_produces_zero_drag_but_records_turnover():
    """When enabled=False, cost bps sum to zero but turnover still counted."""
    positions = [1.0, 0.0, 1.0, 1.0]   # 3 transitions in-window + 1 exit
    returns = [0.0, 0.0, 0.0]
    prices = [100.0, 100.0, 100.0, 100.0]
    cfg = CostConfig(enabled=False)
    net, br = apply_costs(positions=positions, returns=returns, prices=prices, config=cfg)
    assert br.total_cost_bps == 0.0
    assert sum(net) == 0.0
    assert br.turnover == pytest.approx(4.0)   # 3 in-window + 1 exit
    assert br.n_trades == 4


def test_random_style_drag_scales_with_number_of_trades():
    """A strategy that flips every bar should show N × per-trade drag.

    positions[N-1] is dropped by the orchestrator's zip semantics, so
    positions[0..N-2] are the ones we actually enter. Here N=6 → we use
    [1,0,1,0,1] which is 5 in-window transitions, and since the last used
    position is 1 (not 0), we also pay one exit trade. Total = 6."""
    positions = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    returns = [0.0, 0.0, 0.0, 0.0, 0.0]
    prices = [100.0] * 6
    _, br = apply_costs(positions=positions, returns=returns, prices=prices)
    assert br.n_trades == 6
    # 6 trades × 26.5 bps = 159 bps aggregate cost
    assert br.total_cost_bps == pytest.approx(159.0, rel=1e-3)
