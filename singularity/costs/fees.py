"""Alpaca crypto fee schedule.

Convention: bps of notional. 15.0 means 0.15 % = 15 bps.

Plan anti-pattern (§12): "Trading to reach a fee tier (costs $250/mo to save 3 bps)."
Do not size or route trades to chase tiers. This module is a lookup, not a policy.

Only Tier 0 is verified against Alpaca's current published crypto fee schedule.
Alpaca publishes higher volume bands (100k / 1M / 10M / 25M / 100M USD 30d) with
better maker/taker rates, but the bps values move often and encoding them from
memory produced a stale table on the first pass of this file. When Phase 2 CFEE
reconciliation confirms a tier crossing, append the verified row here rather
than pre-populating from a guess.
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
    FeeTier("tier0", 0, 15.0, 25.0),
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
