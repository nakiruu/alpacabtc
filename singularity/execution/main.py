"""Executor process entry point.

Runs three concurrent responsibilities:
  1. heartbeat  → durable proof of life for the watchdog
  2. reconcile  → on startup, halt on critical diffs
  3. trade updates WS → persist fills, update order lifecycle, derive positions

The passive fill loop and bracket supervisor (batch 3b) will hook into the
fill_handler side so their submissions get lifecycle tracking automatically.

Run:
    uv run executor
    # or docker compose up -d executor
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..adapters.alpaca_crypto.trade_updates import TradeUpdatesClient
from ..config import get_settings
from ..logs import configure as configure_logging
from ..logs import get_logger
from ..ops.state import StateStore
from .fill_handler import FillHandler
from .reconcile import reconcile_once

log = get_logger(__name__)


def _trade_updates_url(trading_url: str) -> str:
    """Derive wss://.../stream from the https:// trading URL."""
    return trading_url.replace("https://", "wss://").rstrip("/") + "/stream"


class Executor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = StateStore(Path(self.settings.state_db_path))
        self.rest: AlpacaRestClient | None = None
        self._stop = asyncio.Event()
        self._fill_handler = FillHandler(self.store)
        self._trade_updates: TradeUpdatesClient | None = None

    async def run(self) -> None:
        self.rest = AlpacaRestClient(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            base_url=self.settings.alpaca_trading_url,
        )
        self._trade_updates = TradeUpdatesClient(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            stream_url=_trade_updates_url(self.settings.alpaca_trading_url),
            on_event=self._fill_handler.handle,
        )
        try:
            await self._verify_connectivity()
            await self._reconcile_startup()
            # Run heartbeat + trade updates concurrently. Either failing is fatal;
            # container restart-policy brings us back and reconcile picks up the pieces.
            await asyncio.gather(
                self._heartbeat_loop(),
                self._trade_updates.run(),
            )
        finally:
            await self.rest.close()

    async def _reconcile_startup(self) -> None:
        """Plan §4.3: reconcile before any trading logic runs; halt on critical."""
        assert self.rest is not None
        log.info("executor_reconciling")
        diff = await reconcile_once(self.rest, self.store)
        if diff.has_critical:
            log.critical(
                "executor_reconcile_critical_halt",
                alien_positions=len(diff.alien_positions),
                note="Alpaca reports positions the state store doesn't know about. "
                     "Refusing to start. Inspect via `docker compose run --rm executor reconcile`, "
                     "then either adopt manually or flatten the account.",
            )
            raise SystemExit(2)
        if not diff.is_clean:
            log.warning(
                "executor_reconcile_repaired",
                alien_orders=len(diff.alien_orders),
                ghost_orders=len(diff.ghost_orders),
                ghost_positions=len(diff.ghost_positions),
                repairs=diff.repairs,
            )
        else:
            log.info("executor_reconcile_clean")

    async def _verify_connectivity(self) -> None:
        """Fail loud on startup if credentials or endpoint are misconfigured."""
        assert self.rest is not None
        account = await self.rest.get_account()
        log.info(
            "executor_connected",
            account_status=account.get("status"),
            trading_blocked=account.get("trading_blocked"),
            cash=account.get("cash"),
        )
        if account.get("trading_blocked"):
            raise SystemExit("Alpaca account is trading_blocked — refusing to start executor")

    async def _heartbeat_loop(self) -> None:
        interval = self.settings.executor_heartbeat_s
        while not self._stop.is_set():
            await asyncio.to_thread(self.store.heartbeat, "executor")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    def request_stop(self) -> None:
        self._stop.set()
        if self._trade_updates is not None:
            self._trade_updates.request_stop()


def _install_signal_handlers(exe: Executor) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, exe.request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: exe.request_stop())


async def _amain() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY missing — copy .env.example to .env and fill in."
        )
    exe = Executor()
    _install_signal_handlers(exe)
    log.info(
        "executor_starting",
        trading_url=settings.alpaca_trading_url,
        state_db=settings.state_db_path,
        note="fills received via trade_updates WS; passive fill loop pending (batch 3b)",
    )
    await exe.run()
    log.info("executor_stopped")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
