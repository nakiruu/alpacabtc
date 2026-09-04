"""place-test-order CLI — manual trigger for the passive fill loop.

Real orders leave the box. Sanity-check twice.

    uv run place-test-order --side buy --symbol BTC/USD --qty 0.0001
    uv run place-test-order --side buy --symbol BTC/USD --qty 0.0001 \
        --with-bracket --k-stop 2 --m-target 4

Design choices:
  * qty is in base asset units. 0.0001 BTC ≈ $6 at current prices — safe test size.
  * bracket sizing: stop = entry - k*ATR (long), target = entry + m*ATR.
    ATR pulled from Influx trade tape via features/vol.py.
  * On successful entry, if --with-bracket, upserts the brackets row so the
    executor's supervisor picks it up on its next tick.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from influxdb_client import InfluxDBClient

from ..adapters.alpaca_crypto.market_data import MarketDataClient
from ..adapters.alpaca_crypto.orders import OrderAdapter
from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..costs.types import Side
from ..features.vol import atr_from_influx
from ..logs import get_logger
from ..ops.state import StateStore
from .passive import LadderConfig, PassiveEntry

log = get_logger(__name__)

# Alpaca crypto minimum notional (USD-equivalent). Rejection code 40310000.
ALPACA_CRYPTO_MIN_NOTIONAL = 10.0


async def _run(args) -> int:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)
    store = StateStore(Path(settings.state_db_path))

    if not args.yes:
        print(
            f"About to submit a real {args.side.upper()} {args.qty} {args.symbol} on\n"
            f"  {settings.alpaca_trading_url}\n"
            "Type 'yes' to confirm, anything else to abort: ",
            end="",
        )
        if input().strip().lower() != "yes":
            print("aborted")
            return 1

    async with AlpacaRestClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        base_url=settings.alpaca_trading_url,
    ) as rest, MarketDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    ) as market:
        # Pre-flight: reject before touching REST if the order can't clear Alpaca's minimum.
        q = await market.latest_quote(args.symbol)
        if q is None:
            print(f"[fatal] no quote available for {args.symbol}")
            return 2
        estimated = args.qty * q["ask_px"]
        if estimated < ALPACA_CRYPTO_MIN_NOTIONAL:
            suggested = ALPACA_CRYPTO_MIN_NOTIONAL * 1.1 / q["ask_px"]
            print(
                f"[fatal] estimated notional ${estimated:.2f} < Alpaca minimum "
                f"${ALPACA_CRYPTO_MIN_NOTIONAL:.2f} for crypto orders.\n"
                f"        try --qty {suggested:.6f} or higher (~${suggested * q['ask_px']:.2f})"
            )
            return 2

        adapter = OrderAdapter(rest, store)
        ladder = PassiveEntry(
            adapter=adapter, market=market, store=store,
            config=LadderConfig(
                t1_s=settings.passive_t1_s,
                t2_s=settings.passive_t2_s,
                t3_s=settings.passive_t3_s,
            ),
        )
        result = await ladder.execute(
            side=Side(args.side), symbol=args.symbol, qty=args.qty
        )

    if result.rejected:
        print(f"\npassive result: REJECTED — {result.rejection_reason}")
        return 2
    print(
        f"\npassive result: filled {result.filled_qty}/{args.qty} @ {result.avg_price:.2f} "
        f"(final phase: {result.final_phase.value if result.final_phase else 'n/a'}, "
        f"timed_out={result.timed_out})"
    )

    if args.with_bracket and result.is_filled:
        await _create_bracket(
            settings=settings,
            store=store,
            symbol=args.symbol,
            entry_price=result.avg_price,
            k_stop=args.k_stop,
            m_target=args.m_target,
        )

    return 0 if result.is_filled else 1


async def _create_bracket(
    *,
    settings,
    store: StateStore,
    symbol: str,
    entry_price: float,
    k_stop: float,
    m_target: float,
) -> None:
    with InfluxDBClient(
        url=settings.influx_url, token=settings.influx_token, org=settings.influx_org
    ) as client:
        atr = await atr_from_influx(
            client, settings.influx_bucket_raw, settings.influx_org, symbol
        )
    if atr is None:
        print("[warn] not enough trade history in Influx to compute ATR; bracket not created")
        return
    stop = entry_price - k_stop * atr
    target = entry_price + m_target * atr
    await asyncio.to_thread(
        store.upsert_bracket,
        symbol, stop, target, atr, k_stop, m_target, entry_price,
    )
    print(
        f"bracket set: symbol={symbol} entry={entry_price:.2f} atr={atr:.2f} "
        f"stop={stop:.2f} target={target:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a test order via the passive fill loop")
    parser.add_argument("--side", choices=["buy", "sell"], required=True)
    parser.add_argument("--symbol", required=True, help="e.g. BTC/USD")
    parser.add_argument("--qty", type=float, required=True, help="Base asset units")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--with-bracket", action="store_true",
                        help="Create ATR-based bracket on successful entry")
    parser.add_argument("--k-stop", type=float, default=2.0)
    parser.add_argument("--m-target", type=float, default=4.0)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
