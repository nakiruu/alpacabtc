"""close-position CLI — immediate market close via Alpaca REST.

Uses DELETE /v2/positions/{symbol}. Alpaca handles the market order internally.
Also removes any bracket row so the supervisor doesn't try to close a second
time on its next tick.

    uv run close-position --symbol BTC/USD --yes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


async def _run(args) -> int:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)
    store = StateStore(Path(settings.state_db_path))

    if not args.yes:
        print(
            f"About to market-close position on {args.symbol} at {settings.alpaca_trading_url}\n"
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
    ) as rest:
        # Cancel any resting orders first so we're not fighting them
        try:
            open_orders = await rest.get_orders(status="open")
        except Exception:
            log.exception("close_get_orders_failed")
            open_orders = []
        for o in open_orders:
            if o.get("symbol") == args.symbol:
                try:
                    await rest.cancel_order(o["id"])
                    print(f"canceled resting order {o['id']}")
                except Exception:
                    log.warning("close_cancel_failed", order_id=o.get("id"))

        # Close position
        try:
            resp = await rest.close_position(args.symbol)
            print(f"close submitted: {resp}")
        except Exception as e:
            print(f"[fatal] close_position failed: {e}")
            return 2

    # Drop bracket row so supervisor doesn't try to close again on its tick
    await asyncio.to_thread(store.delete_bracket, args.symbol)
    print(f"bracket row for {args.symbol} removed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-close a position via Alpaca REST")
    parser.add_argument("--symbol", required=True, help="e.g. BTC/USD")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
