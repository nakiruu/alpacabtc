"""Nightly calibration of the cost model against realized fills.

Plan §3.3:
    "Every order logs: intent price, submitted price, fill price(s), fill/no-fill,
     timestamp, book snapshot at submission. Nightly, join against CFEE/FEE
     records from the Activities API and fit:
       - realized maker ratio vs. modeled
       - implementation shortfall vs. modeled
       - fee accrual vs. actual"

Plan §3.3 gate:
    "modeled vs. realized cost within 3 bps on a rolling 100-trade window.
     Until this gate passes, treat every backtest number as unverified."

This module is scaffolded now (Phase 1) and wired to real data in Phase 2 when
the execution layer starts logging intents and Alpaca posts CFEE activities.

CLI entry point: `calibrate` (registered in pyproject.toml). Exits:
    0  gate passing OR insufficient data (< N trades)
    1  gate failing (rolling divergence > 3 bps)
    2  cannot reach Influx / operational error
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from influxdb_client import InfluxDBClient

from ..config import get_settings
from ..logs import configure as configure_logging
from ..logs import get_logger
from .types import Cost, Fill, OrderIntent

log = get_logger(__name__)

# Plan §3.3 gate
DIVERGENCE_BUDGET_BPS = 3.0
ROLLING_WINDOW_TRADES = 100
MIN_TRADES_FOR_GATE = 20   # below this, report "insufficient data" rather than pass/fail


@dataclass
class Divergence:
    """Per-fill difference between modeled and realized cost, in bps."""

    order_id: str
    filled_at: datetime
    modeled_total_bps: float
    realized_total_bps: float
    fee_diff_bps: float
    spread_plus_impact_diff_bps: float

    @property
    def total_diff_bps(self) -> float:
        return self.realized_total_bps - self.modeled_total_bps


@dataclass
class CalibrationReport:
    window_start: datetime
    window_end: datetime
    n_fills: int
    n_makers: int
    n_takers: int
    modeled_maker_ratio: float | None
    realized_maker_ratio: float | None
    rolling_median_diff_bps: float | None    # rolling 100
    rolling_p95_diff_bps: float | None
    gate_status: str                          # PASS / FAIL / INSUFFICIENT

    def print(self) -> None:
        print(f"\n=== cost calibration — {self.window_start.date()} → {self.window_end.date()} ===\n")
        print(f"fills          : {self.n_fills}  (maker {self.n_makers}, taker {self.n_takers})")
        if self.realized_maker_ratio is not None:
            print(f"maker ratio    : realized {self.realized_maker_ratio:.2%}   modeled {self.modeled_maker_ratio:.2%}")
        if self.rolling_median_diff_bps is None:
            print(f"rolling {ROLLING_WINDOW_TRADES}-trade divergence: n/a")
        else:
            print(f"rolling {ROLLING_WINDOW_TRADES}-trade divergence:")
            print(f"  median |realized − modeled| : {self.rolling_median_diff_bps:+.2f} bps")
            print(f"  p95    |realized − modeled| : {self.rolling_p95_diff_bps:+.2f} bps")
            print(f"  budget                       : ±{DIVERGENCE_BUDGET_BPS:.1f} bps")
        print(f"gate           : [{self.gate_status}]\n")


# ---------------------------------------------------------------------------
# Data loaders — thin wrappers, wired in Phase 2 when intents & fills exist.
# For now they return empty lists so the CLI can run end-to-end and report
# "insufficient data" instead of crashing.
# ---------------------------------------------------------------------------

def load_intents(
    client: InfluxDBClient, bucket: str, org: str, start: datetime, end: datetime
) -> list[OrderIntent]:
    """Load logged OrderIntent records from `order_intent` measurement.

    Phase 2 writes to this measurement from execution/. Until then, empty.
    """
    return []


def load_fills(
    client: InfluxDBClient, bucket: str, org: str, start: datetime, end: datetime
) -> list[Fill]:
    """Load reconciled Fill records from `order_fill` measurement.

    Phase 2 will populate this from Alpaca fills + CFEE Activities API.
    """
    return []


# ---------------------------------------------------------------------------
# Core reconcile
# ---------------------------------------------------------------------------

def divergences(intents: list[OrderIntent], fills: list[Fill]) -> list[Divergence]:
    """Join intents ↔ fills on order_id, compute per-fill divergence."""
    by_id = {i.id: i for i in intents}
    out: list[Divergence] = []
    for f in fills:
        intent = by_id.get(f.order_id)
        if intent is None:
            log.warning("orphan_fill", order_id=f.order_id)
            continue
        realized = f.realized_cost_bps(intent.mid_at_submit)
        modeled = intent.modeled_cost
        out.append(
            Divergence(
                order_id=f.order_id,
                filled_at=f.filled_at,
                modeled_total_bps=modeled.total_bps,
                realized_total_bps=realized.total_bps,
                fee_diff_bps=realized.fee_bps - modeled.fee_bps,
                spread_plus_impact_diff_bps=(realized.spread_bps + realized.impact_bps)
                - (modeled.spread_bps + modeled.impact_bps),
            )
        )
    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def build_report(
    fills: list[Fill],
    intents: list[OrderIntent],
    window_start: datetime,
    window_end: datetime,
) -> CalibrationReport:
    diffs = divergences(intents, fills)
    n_makers = sum(1 for f in fills if f.is_maker)
    n_takers = len(fills) - n_makers
    realized_maker = n_makers / len(fills) if fills else None
    modeled_maker = None  # Phase 2 sets this once intents carry an is_maker prediction

    if len(diffs) < MIN_TRADES_FOR_GATE:
        return CalibrationReport(
            window_start=window_start,
            window_end=window_end,
            n_fills=len(fills),
            n_makers=n_makers,
            n_takers=n_takers,
            modeled_maker_ratio=modeled_maker,
            realized_maker_ratio=realized_maker,
            rolling_median_diff_bps=None,
            rolling_p95_diff_bps=None,
            gate_status="INSUFFICIENT",
        )

    # rolling 100 = last N by fill time
    diffs.sort(key=lambda d: d.filled_at)
    tail = diffs[-ROLLING_WINDOW_TRADES:]
    abs_diffs = sorted(abs(d.total_diff_bps) for d in tail)
    median_abs = statistics.median(abs_diffs)
    p95_abs = _percentile(abs_diffs, 0.95)
    gate = "PASS" if median_abs <= DIVERGENCE_BUDGET_BPS else "FAIL"

    return CalibrationReport(
        window_start=window_start,
        window_end=window_end,
        n_fills=len(fills),
        n_makers=n_makers,
        n_takers=n_takers,
        modeled_maker_ratio=modeled_maker,
        realized_maker_ratio=realized_maker,
        rolling_median_diff_bps=median_abs,
        rolling_p95_diff_bps=p95_abs,
        gate_status=gate,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cost model calibration")
    parser.add_argument("--days", type=int, default=1, help="lookback window in days (default 1)")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)

    try:
        client = InfluxDBClient(
            url=settings.influx_url, token=settings.influx_token, org=settings.influx_org
        )
    except Exception as e:
        print(f"[fatal] cannot connect to Influx: {e}", file=sys.stderr)
        sys.exit(2)

    with client:
        try:
            intents = load_intents(client, settings.influx_bucket_raw, settings.influx_org, start, now)
            fills = load_fills(client, settings.influx_bucket_raw, settings.influx_org, start, now)
        except Exception as e:
            print(f"[fatal] flux query failed: {e}", file=sys.stderr)
            sys.exit(2)

    report = build_report(fills, intents, start, now)
    report.print()

    if report.gate_status == "FAIL":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
