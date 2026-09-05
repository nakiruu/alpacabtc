"""Daily signal-driver — the one-shot that turns backtested strategy into paper orders.

Runs once per invocation (cron-driven, once per day at 00:15 UTC recommended):

    1. HEALTH CHECK. Verify Alpaca account is not trading_blocked.
    2. RECONCILE. Halt on critical diffs (alien positions).
    3. HEARTBEAT. Verify executor container's watchdog signal is fresh.
    4. IDEMPOTENCY. Refuse to trade if we already submitted a signal-driven
       order today (prevents double-trading if cron fires twice).
    5. FETCH BARS. Pull ~2 years of daily bars from Alpaca (same venue we
       execute on → no data source drift).
    6. COMPUTE SIGNAL. Run tsmom_full strategy on the bars. Take positions[-1]
       as today's target weight in [0, 1].
    7. SIZE. target_btc = account_equity × weight / current_mid.
       delta = target_btc − current_position_btc.
    8. GATE. Skip trade if |delta × mid| < $10 (Alpaca crypto min notional).
    9. EXECUTE via PassiveEntry ladder (batch 3b passive fill loop).
   10. LOG the decision to Influx for post-hoc audit.

Exit codes:
    0 — clean tick (traded or intentionally skipped)
    1 — trade attempted, PassiveEntry did not fully fill
    2 — refused to trade (reconcile critical, heartbeat stale, trading_blocked)
    3 — operational error (couldn't fetch bars, network, etc.)

Guard rails philosophy: fail closed. Any ambiguity → refuse to trade and log
loudly. Better to skip a day than send a bad order.
"""

from __future__ import annotations

import argparse
import asyncio
import signal as sig_mod
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from ..adapters.alpaca_crypto.history import HistoryClient, load_bars_cached
from ..adapters.alpaca_crypto.market_data import MarketDataClient
from ..adapters.alpaca_crypto.orders import OrderAdapter
from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..costs.model import BookSnapshot, one_way_cost
from ..costs.types import OrderIntent, OrderType, Side, TimeInForce
from ..logs import get_logger
from ..ops.state import StateStore
from ..signals.tsmom import tsmom_full
from .passive import LadderConfig, PassiveEntry
from .reconcile import _normalize_symbol, reconcile_once

log = get_logger(__name__)


# Minimum notional Alpaca requires per crypto order
ALPACA_MIN_NOTIONAL_USD = 10.0

# Signal driver marks its orders with this client_order_id prefix so we can
# detect "was there a signal-driven trade today" without a separate table.
SIGNAL_ORDER_ID_PREFIX = "signal-tick"


def _today_utc_date_str(now: datetime) -> str:
    return now.strftime("%Y-%m-%d")


def _signal_intent_id(symbol: str, side: Side, when: datetime) -> str:
    """Deterministic-per-day client_order_id. Two ticks on the same UTC day
    trying to trade the same side will collide via Alpaca's client_order_id
    uniqueness check → second call becomes a no-op idempotently."""
    date_part = _today_utc_date_str(when)
    sym_part = symbol.replace("/", "-")
    # Add small random suffix per call — otherwise a legitimate rebalance
    # within the same day (e.g. from an emergency re-tick) would be blocked
    # by our own idempotency. Rely on the state-DB check for daily idempotency
    # instead.
    return f"{SIGNAL_ORDER_ID_PREFIX}-{sym_part}-{date_part}-{uuid.uuid4().hex[:8]}"


def _already_ticked_today(store: StateStore, symbol: str, now: datetime) -> bool:
    """Check if a signal-driven order for this symbol was already submitted today."""
    date_str = _today_utc_date_str(now)
    sym_part = symbol.replace("/", "-")
    marker = f"{SIGNAL_ORDER_ID_PREFIX}-{sym_part}-{date_str}-"
    # Look at all orders (open + terminal) whose id starts with today's marker
    # Use the raw connection for a LIKE query since our helpers don't support it
    with store._connect() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM orders WHERE id LIKE ? || '%'", (marker,),
        ).fetchone()
    return (row["n"] if row else 0) > 0


async def paper_tick(
    *,
    symbol: str = "BTC/USD",
    bars_lookback_days: int = 730,
    dry_run: bool = False,
    heartbeat_max_age_s: float = 90.0,
    ladder_config: LadderConfig | None = None,
    cache_dir: Path = Path("./state/bars"),
    idempotency_mode: str = "daily",  # "daily" | "drift"
    min_notional_delta: float | None = None,
) -> int:
    """Execute one signal-driven tick. Returns exit code.

    idempotency_mode:
      "daily" — once per (symbol, UTC-day). Right for cron-once-daily.
      "drift" — no daily block; rely on min_notional_delta to prevent churn.
                Right for continuous-daemon operation.

    min_notional_delta — override the $10 Alpaca minimum. In daemon mode,
    raise this to (say) $50 so intraday micro-drifts don't churn.
    """
    min_delta = min_notional_delta if min_notional_delta is not None else ALPACA_MIN_NOTIONAL_USD
    from ..config import get_settings
    settings = get_settings()
    store = StateStore(Path(settings.state_db_path))
    now = datetime.now(timezone.utc)

    async with AlpacaRestClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_trading_url,
    ) as rest:
        # 1. Account health check
        try:
            account = await rest.get_account()
        except httpx.HTTPStatusError as e:
            log.error("paper_tick_account_error", status=e.response.status_code)
            return 3
        if account.get("trading_blocked"):
            log.critical("paper_tick_trading_blocked", account_status=account.get("status"))
            return 2

        # 2. Reconcile — halt on critical (alien positions)
        try:
            diff = await reconcile_once(rest, store)
        except Exception:
            log.exception("paper_tick_reconcile_error")
            return 3
        if diff.has_critical:
            log.critical("paper_tick_reconcile_critical_halt",
                         alien_positions=len(diff.alien_positions))
            return 2

        # 3. Heartbeat check — refuse to trade if executor is dead
        age = store.heartbeat_age_s("executor")
        if age is None:
            log.error("paper_tick_no_executor_heartbeat_yet")
            return 2
        if age > heartbeat_max_age_s:
            log.error("paper_tick_executor_heartbeat_stale",
                      age_s=age, threshold_s=heartbeat_max_age_s)
            return 2

        # 4. Idempotency (daily mode only)
        if idempotency_mode == "daily" and _already_ticked_today(store, symbol, now):
            log.info("paper_tick_already_ticked_today", symbol=symbol,
                     date=_today_utc_date_str(now))
            return 0

        # 5. Fetch bars — use Alpaca (same venue we trade on)
        # Round `end` to midnight so cache key stays stable within a UTC day.
        # Daily bars only get a new value once per UTC midnight, so intraday
        # re-fetching returns the same data anyway.
        start = now - timedelta(days=bars_lookback_days)
        end = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        try:
            async with HistoryClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            ) as hc:
                bars = await load_bars_cached(
                    hc, symbol, start, end, cache_dir,
                    timeframe="1Day",
                    force_refresh=False,  # cache hit within same UTC day
                )
        except Exception:
            log.exception("paper_tick_bar_fetch_failed", symbol=symbol)
            return 3
        if len(bars) < 200:
            log.error("paper_tick_insufficient_bars", n=len(bars),
                      required=200)
            return 3

        # 6. Compute strategy signal
        strategy = tsmom_full()  # defaults from Phase 4.3 that passed the null-gate
        positions = strategy(bars)
        target_weight = positions[-1] if positions else 0.0

        # 7. Size — get equity + current position + current mid
        equity = float(account.get("equity", 0.0))
        if equity <= 0.0:
            log.error("paper_tick_zero_equity")
            return 3

        # current position from Alpaca (source of truth vs local state)
        try:
            alpaca_positions = await rest.get_positions()
        except Exception:
            log.exception("paper_tick_get_positions_failed")
            return 3
        current_btc = 0.0
        target_symbol_norm = _normalize_symbol(symbol)
        for p in alpaca_positions:
            if _normalize_symbol(p.get("symbol") or "") == target_symbol_norm:
                current_btc = float(p.get("qty") or 0.0)
                break

        async with MarketDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        ) as market:
            quote = await market.latest_quote(symbol)
            if quote is None:
                log.error("paper_tick_no_quote", symbol=symbol)
                return 3
            mid = 0.5 * (quote["bid_px"] + quote["ask_px"])
            if mid <= 0:
                log.error("paper_tick_bad_mid", mid=mid)
                return 3

            # 8. Compute delta and gate on min notional
            target_btc = (equity * target_weight) / mid
            delta_btc = target_btc - current_btc
            notional_delta = abs(delta_btc) * mid

            decision = {
                "symbol": symbol,
                "date": _today_utc_date_str(now),
                "target_weight": target_weight,
                "equity_usd": equity,
                "mid": mid,
                "current_btc": current_btc,
                "target_btc": target_btc,
                "delta_btc": delta_btc,
                "notional_delta_usd": notional_delta,
                "dry_run": dry_run,
            }
            log.info("paper_tick_decision", **decision)

            if notional_delta < min_delta:
                log.info("paper_tick_below_min_notional",
                         notional=notional_delta,
                         min_required=min_delta)
                return 0

            if delta_btc == 0.0:
                log.info("paper_tick_no_change")
                return 0

            side = Side.BUY if delta_btc > 0 else Side.SELL
            qty = abs(delta_btc)

            if dry_run:
                log.info("paper_tick_dry_run", side=side.value, qty=qty)
                return 0

            # 9. Execute via passive ladder — mark intent id with today's stamp
            #    so future idempotency checks find it.
            adapter = OrderAdapter(rest, store)
            ladder = PassiveEntry(
                adapter=adapter, market=market, store=store,
                config=ladder_config or LadderConfig(
                    t1_s=settings.passive_t1_s,
                    t2_s=settings.passive_t2_s,
                    t3_s=settings.passive_t3_s,
                ),
            )
            # Pre-seed a stamped id (daily-mode idempotency); no-op if drift-mode
            if idempotency_mode == "daily":
                _pre_stamp_intent(store, symbol, side, now)

            result = await ladder.execute(side=side, symbol=symbol, qty=qty)

        if result.rejected:
            log.error("paper_tick_ladder_rejected", reason=result.rejection_reason)
            return 2
        if result.timed_out:
            log.warning("paper_tick_ladder_timed_out",
                        filled=result.filled_qty, requested=qty)
            return 1
        log.info("paper_tick_filled",
                 filled=result.filled_qty, avg_price=result.avg_price,
                 final_phase=result.final_phase.value if result.final_phase else "n/a")
        return 0


def _pre_stamp_intent(store: StateStore, symbol: str, side: Side, now: datetime) -> None:
    """Insert a placeholder pending order row whose id carries today's date
    marker. The subsequent PassiveEntry.execute creates real orders with
    different (random-suffix) ids; the placeholder is what the next paper_tick
    invocation's idempotency query finds via LIKE match.

    We use PENDING status; if the strategy tick errors out before executing
    the ladder, this row stays PENDING and reconciliation on next executor
    startup marks it REJECTED. Either way, today's paper_tick is done.
    """
    from ..costs.types import Cost, OrderType, TimeInForce
    intent_id = _signal_intent_id(symbol, side, now)
    intent = OrderIntent(
        id=intent_id,
        symbol=symbol,
        side=side,
        qty=0.0,   # placeholder — real orders come from PassiveEntry
        order_type=OrderType.LIMIT,
        tif=TimeInForce.GTC,
        limit_price=None,
        submitted_at=now,
        mid_at_submit=0.0,
        modeled_cost=Cost(0.0, 0.0, 0.0),
    )
    try:
        store.save_intent(intent)
    except Exception:
        log.warning("paper_tick_prestamp_failed", intent_id=intent_id)


async def paper_loop(
    *,
    symbol: str,
    interval_s: float,
    min_notional_delta: float,
    bars_lookback_days: int,
    heartbeat_max_age_s: float,
    cache_dir: Path,
) -> None:
    """Long-running daemon: re-evaluate signal every interval_s and act on drift.

    Uses `drift` idempotency mode — no daily block, but the min_notional_delta
    gate prevents churn when the strategy weight × current price is close to
    the actual position. Signal itself only changes when a new daily bar
    rolls (00:00 UTC); intraday ticks re-check whether current position drifted
    from target (e.g. because equity moved during a big price change).
    """
    stop = asyncio.Event()

    def _handle_stop() -> None:
        log.info("paper_loop_stop_requested")
        stop.set()

    loop = asyncio.get_running_loop()
    for s in (sig_mod.SIGINT, sig_mod.SIGTERM):
        try:
            loop.add_signal_handler(s, _handle_stop)
        except NotImplementedError:
            sig_mod.signal(s, lambda *_: _handle_stop())

    log.info("paper_loop_starting", symbol=symbol, interval_s=interval_s,
             min_notional_delta=min_notional_delta)
    tick_n = 0
    while not stop.is_set():
        tick_n += 1
        log.info("paper_loop_tick_begin", tick=tick_n)
        try:
            await paper_tick(
                symbol=symbol,
                bars_lookback_days=bars_lookback_days,
                dry_run=False,
                heartbeat_max_age_s=heartbeat_max_age_s,
                cache_dir=cache_dir,
                idempotency_mode="drift",
                min_notional_delta=min_notional_delta,
            )
        except Exception:
            log.exception("paper_loop_tick_error", tick=tick_n)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    log.info("paper_loop_stopped", ticks=tick_n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal-driven paper trading — one-shot (cron) or --daemon"
    )
    parser.add_argument("--symbol", default="BTC/USD",
                        help="Symbol to trade (default BTC/USD)")
    parser.add_argument("--bars-lookback-days", type=int, default=730,
                        help="Days of Alpaca history to fetch for signal (default 730 = 2y)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute decision and log it; do not submit orders")
    parser.add_argument("--heartbeat-max-age-s", type=float, default=90.0,
                        help="Refuse to trade if executor heartbeat older than this (default 90s)")
    parser.add_argument("--cache-dir", default="./state/bars")
    parser.add_argument("--daemon", action="store_true",
                        help="Continuous mode: loop every --interval-s using drift idempotency")
    parser.add_argument("--interval-s", type=float, default=900.0,
                        help="Daemon-mode tick interval in seconds (default 900 = 15 min)")
    parser.add_argument("--min-notional-delta", type=float, default=None,
                        help="Override min notional for a rebalance (default $10; suggest $50+ in daemon)")
    parser.add_argument("--idempotency-mode", choices=["daily", "drift"], default=None,
                        help="'daily' = once per UTC day (cron), 'drift' = every tick (daemon). "
                             "Defaults to 'drift' when --daemon is set, else 'daily'.")
    args = parser.parse_args()

    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.daemon:
        # Sensible default for daemon: raise min notional to prevent churn
        min_delta = args.min_notional_delta if args.min_notional_delta is not None else 50.0
        asyncio.run(paper_loop(
            symbol=args.symbol,
            interval_s=args.interval_s,
            min_notional_delta=min_delta,
            bars_lookback_days=args.bars_lookback_days,
            heartbeat_max_age_s=args.heartbeat_max_age_s,
            cache_dir=Path(args.cache_dir),
        ))
        sys.exit(0)

    idempotency = args.idempotency_mode or "daily"
    exit_code = asyncio.run(paper_tick(
        symbol=args.symbol,
        bars_lookback_days=args.bars_lookback_days,
        dry_run=args.dry_run,
        heartbeat_max_age_s=args.heartbeat_max_age_s,
        cache_dir=Path(args.cache_dir),
        idempotency_mode=idempotency,
        min_notional_delta=args.min_notional_delta,
    ))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
