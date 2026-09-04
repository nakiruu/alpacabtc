"""Alpaca crypto fee schedule.

Tiers below reflect the published schedule as of 2026Q3. If Alpaca changes their
tiers, update ``TIERS`` — every downstream cost estimate reads from here.

Convention: bps of notional. 15.0 means 0.15 % = 15 bps.

Plan anti-pattern (§12): "Trading to reach a fee tier (costs $250/mo to save 3 bps)."
Do not size or route trades to chase tiers. This module is a lookup, not a policy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeTier:
    name: str
    min_30d_volume_usd: float
    maker_bps: float
    taker_bps: float


# Ordered highest-volume → lowest so first match wins on downward scan.
TIERS: tuple[FeeTier, ...] = (
    FeeTier("tier5", 100_000_000, 2.0, 10.0),
    FeeTier("tier4",  25_000_000, 4.0, 12.0),
    FeeTier("tier3",  10_000_000, 6.0, 15.0),
    FeeTier("tier2",   1_000_000, 8.0, 18.0),
    FeeTier("tier1",     100_000, 12.0, 22.0),
    FeeTier("tier0",           0, 15.0, 25.0),
)


def lookup(volume_30d_usd: float = 0.0) -> FeeTier:
    """Return the applicable fee tier for a given 30d rolling notional (USD).

    Defaults to tier 0 — Phase 2 will feed real volume from CFEE reconciliation.
    """
    v = max(0.0, volume_30d_usd)
    for tier in TIERS:
        if v >= tier.min_30d_volume_usd:
            return tier
    return TIERS[-1]  # unreachable given tier0 min=0, but keeps mypy happy


def fee_bps(is_maker: bool, volume_30d_usd: float = 0.0) -> float:
    tier = lookup(volume_30d_usd)
    return tier.maker_bps if is_maker else tier.taker_bps
