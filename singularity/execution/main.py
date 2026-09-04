"""Executor process entry point.

Phase 2 batch 1 (this file): starts the state store, opens the REST client,
verifies Alpaca connectivity, and runs the heartbeat loop. It does NOT
submit orders. That comes in batch 3 with the passive fill loop.

The heartbeat is what the watchdog (batch 2) will read to decide whether
the executor is alive; the dead-man's-switch triggers if it goes stale.

Run:
    uv run executor

or via docker compose (see compose service `executor`).
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..config import get_settings
from ..logs import configure as configure_logging
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


class Executor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = StateStore(Path(self.settings.state_db_path))
        self.rest: AlpacaRestClient | None = None
        self._stop = asyncio.Event()

    async def run(self) -> None:
        self.rest = AlpacaRestClient(
            api_key=self.settings.alpaca_api_key,
            secret_key=self.settings.alpaca_secret_key,
            base_url=self.settings.alpaca_trading_url,
        )
        try:
            await self._verify_connectivity()
            await self._heartbeat_loop()
        finally:
            await self.rest.close()

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
        note="no orders will be submitted until the passive loop is wired (batch 3)",
    )
    await exe.run()
    log.info("executor_stopped")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
