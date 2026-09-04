"""CLI operational report.

Text-only monitoring — no dashboards. Run on-demand or from cron:

    uv run report                 # last 24h
    uv run report --window 7d     # last 7 days (the plan §2 gate window)

Exit codes:
    0  everything within gates
    1  one or more gates failing (safe to alert on)
    2  cannot reach Influx / usage error
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient

from ..config import get_settings
from ..logs import configure as configure_logging
from ..logs import get_logger

log = get_logger(__name__)

# Phase 0 gate — plan §2
GAP_BUDGET_FRACTION = 0.005  # <0.5% gap time
STALENESS_ALERT_S = 300.0    # 5min without any message on a subscribed symbol


@dataclass
class SymbolThroughput:
    symbol: str
    trades: int
    quotes: int
    book_updates: int


@dataclass
class GapStats:
    total_gap_s: float
    n_events: int
    worst_gap_s: float


def _parse_window(s: str) -> str:
    # Trust Flux syntax: "24h", "7d", "1h30m" etc. Just validate loosely.
    if not s or s[-1] not in {"s", "m", "h", "d", "w"}:
        raise argparse.ArgumentTypeError(f"window must end in s/m/h/d/w, got {s!r}")
    return s


def _window_seconds(s: str) -> float:
    unit = s[-1]
    val = float(s[:-1])
    return val * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


def query_throughput(client: InfluxDBClient, bucket: str, org: str, window: str) -> list[SymbolThroughput]:
    flux = f"""
    from(bucket: "{bucket}")
      |> range(start: -{window})
      |> filter(fn: (r) => r._measurement == "crypto_trade" or r._measurement == "crypto_quote" or r._measurement == "book_features")
      |> filter(fn: (r) => r._field == "price" or r._field == "bid_px" or r._field == "spread_bps")
      |> group(columns: ["symbol", "_measurement"])
      |> count()
    """
    tables = client.query_api().query(flux, org=org)
    counts: dict[str, dict[str, int]] = {}
    for tbl in tables:
        for rec in tbl.records:
            sym = rec.values.get("symbol", "?")
            meas = rec.get_measurement()
            counts.setdefault(sym, {})[meas] = int(rec.get_value())
    return [
        SymbolThroughput(
            symbol=sym,
            trades=v.get("crypto_trade", 0),
            quotes=v.get("crypto_quote", 0),
            book_updates=v.get("book_features", 0),
        )
        for sym, v in sorted(counts.items())
    ]


def query_gaps(client: InfluxDBClient, bucket: str, org: str, window: str) -> GapStats:
    flux = f"""
    from(bucket: "{bucket}")
      |> range(start: -{window})
      |> filter(fn: (r) => r._measurement == "stream_gap" and r._field == "gap_s")
    """
    tables = client.query_api().query(flux, org=org)
    vals = [float(rec.get_value()) for tbl in tables for rec in tbl.records]
    return GapStats(
        total_gap_s=sum(vals),
        n_events=len(vals),
        worst_gap_s=max(vals, default=0.0),
    )


def query_last_seen(client: InfluxDBClient, bucket: str, org: str) -> dict[str, datetime]:
    flux = f"""
    from(bucket: "{bucket}")
      |> range(start: -1h)
      |> filter(fn: (r) => r._measurement == "crypto_quote" and r._field == "bid_px")
      |> group(columns: ["symbol"])
      |> last()
    """
    tables = client.query_api().query(flux, org=org)
    out: dict[str, datetime] = {}
    for tbl in tables:
        for rec in tbl.records:
            sym = rec.values.get("symbol", "?")
            out[sym] = rec.get_time()
    return out


def _fmt_int(n: int) -> str:
    return f"{n:>12,}"


def _fmt_rate(n: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{n / seconds * 60:.1f}/min"


def report(window: str = "24h") -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    now = datetime.now(timezone.utc)
    window_s = _window_seconds(window)
    gate_budget_s = GAP_BUDGET_FRACTION * window_s

    print(f"\n=== singularity Phase 0 report — window {window} ({now.isoformat()}) ===\n")

    try:
        client = InfluxDBClient(
            url=settings.influx_url, token=settings.influx_token, org=settings.influx_org
        )
    except Exception as e:
        print(f"[fatal] cannot connect to Influx: {e}", file=sys.stderr)
        return 2

    failing = 0

    with client:
        try:
            throughput = query_throughput(client, settings.influx_bucket_raw, settings.influx_org, window)
            gaps = query_gaps(client, settings.influx_bucket_raw, settings.influx_org, window)
            last_seen = query_last_seen(client, settings.influx_bucket_raw, settings.influx_org)
        except Exception as e:
            print(f"[fatal] flux query failed: {e}", file=sys.stderr)
            return 2

    # --- Throughput ---
    print("throughput")
    print(f"  {'symbol':<12} {'trades':>12} {'quotes':>12} {'book_feats':>12}   rate(quotes)")
    for t in throughput:
        print(
            f"  {t.symbol:<12} {_fmt_int(t.trades)} {_fmt_int(t.quotes)} {_fmt_int(t.book_updates)}   {_fmt_rate(t.quotes, window_s):>10}"
        )
    if not throughput:
        print("  (no data — is capture running and pointing at this bucket?)")
        failing += 1

    # --- Staleness ---
    print("\nstaleness (seconds since last quote)")
    subscribed = set(settings.quotes_symbols())
    for sym in sorted(subscribed):
        ts = last_seen.get(sym)
        if ts is None:
            age = float("inf")
            marker = "  [FAIL] no quotes in the last hour"
            failing += 1
        else:
            age = (now - ts).total_seconds()
            if age > STALENESS_ALERT_S:
                failing += 1
                marker = "  [FAIL]"
            else:
                marker = ""
        print(f"  {sym:<12} {age:>10.1f}s{marker}")

    # --- Gap gate ---
    print("\nstream gaps")
    pct = gaps.total_gap_s / window_s * 100 if window_s > 0 else 0.0
    budget_pct = GAP_BUDGET_FRACTION * 100
    gate_status = "PASS" if gaps.total_gap_s <= gate_budget_s else "FAIL"
    if gate_status == "FAIL":
        failing += 1
    print(f"  events        : {gaps.n_events}")
    print(f"  total gap     : {gaps.total_gap_s:.1f}s ({pct:.3f}% of window)")
    print(f"  worst single  : {gaps.worst_gap_s:.1f}s")
    print(f"  budget        : {gate_budget_s:.1f}s ({budget_pct:.2f}% of window)")
    print(f"  gate          : [{gate_status}]")

    print()
    if failing:
        print(f"result: FAIL ({failing} gate breach{'es' if failing != 1 else ''})")
        return 1
    print("result: OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 operational report")
    parser.add_argument("--window", type=_parse_window, default="24h",
                        help="Flux duration, e.g. 24h, 7d, 1h30m (default 24h)")
    args = parser.parse_args()
    sys.exit(report(args.window))


if __name__ == "__main__":
    main()
