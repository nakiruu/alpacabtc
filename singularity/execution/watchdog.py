"""Watchdog / dead-man's-switch — plan §4.3.

Runs as a separate process. Reads the executor's heartbeat from the state store;
if it goes stale beyond ``HEARTBEAT_MAX_AGE_S`` AND we're holding positions,
flattens everything via Alpaca REST directly (cancel-all-orders + close-position).

Why separate process: if the executor crashes, an in-process monitor crashes
with it. The watchdog is intentionally minimal — it only reads state and calls
REST endpoints — so its failure surface is small and uncorrelated with the
executor's.

Threshold defaults (config knob):
    executor heartbeat every 15s → threshold 90s = 6x tolerance
    watchdog polls every 30s

If you're intentionally restarting the executor, a fresh heartbeat lands
within ~30s of container start. The 90s threshold survives that.

Run:
    uv run watchdog

or the compose service `watchdog`.
"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

from ..adapters.alpaca_crypto.rest import AlpacaRestClient
from ..logs import get_logger
from ..ops.state import StateStore

log = get_logger(__name__)


class DeadMansSwitch:
    def __init__(
        self,
        rest: AlpacaRestClient,
        store: StateStore,
        max_age_s: float,
    ) -> None:
        self._rest = rest
        self._store = store
        self._max_age = max_age_s
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    async def check_once(self) -> bool:
        """Poll once. Returns True if the switch fired on this call."""
        age = await asyncio.to_thread(self._store.heartbeat_age_s, "executor")
        if age is None:
            log.info("watchdog_no_executor_heartbeat_yet")
            return False
        if age <= self._max_age:
            return False

        # Executor heartbeat stale — check if we're holding anything worth flattening
        try:
            positions = await self._rest.get_positions()
        except Exception:
            log.exception("watchdog_get_positions_failed", age_s=age)
            return False

        if not positions:
            log.warning(
                "watchdog_executor_stale_but_flat",
                age_s=age,
                threshold_s=self._max_age,
            )
            return False

        log.critical(
            "watchdog_triggered",
            age_s=age,
            threshold_s=self._max_age,
            n_positions=len(positions),
            symbols=[p.get("symbol") for p in positions],
        )
        try:
            await self._rest.cancel_all_orders()
        except Exception:
            log.exception("watchdog_cancel_all_failed")
        for p in positions:
            sym = p.get("symbol")
            if not sym:
                continue
            try:
                await self._rest.close_position(sym)
                log.critical("watchdog_position_closed", symbol=sym)
            except Exception:
                log.exception("watchdog_close_position_failed", symbol=sym)

        self._triggered = True
        return True


class Watchdog:
    def __init__(self) -> None:
        from ..config import get_settings
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
            switch = DeadMansSwitch(
                self.rest, self.store, self.settings.heartbeat_max_age_s
            )
            interval = self.settings.watchdog_check_s
            log.info(
                "watchdog_starting",
                check_s=interval,
                max_age_s=self.settings.heartbeat_max_age_s,
            )
            while not self._stop.is_set():
                try:
                    await switch.check_once()
                except Exception:
                    log.exception("watchdog_check_error")
                await asyncio.to_thread(self.store.heartbeat, "watchdog")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except TimeoutError:
                    continue
        finally:
            await self.rest.close()

    def request_stop(self) -> None:
        self._stop.set()


def _install_signal_handlers(w: Watchdog) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, w.request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: w.request_stop())


async def _amain() -> None:
    from ..config import get_settings
    from ..logs import configure as configure_logging
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY missing — the watchdog needs REST access to flatten."
        )
    w = Watchdog()
    _install_signal_handlers(w)
    await w.run()
    log.info("watchdog_stopped")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
