"""Phase 0 — Alpaca crypto WS capture → InfluxDB.

Subscribes to trades / quotes / orderbooks per config, maintains L2 book state,
computes book features on a fixed cadence, writes everything to Influx.

Reconnect: exponential backoff via tenacity. Every reconnect logs a gap event
so we can measure the <0.5% gap-time gate from plan §2.

Run:
    uv run capture
    # or
    python -m singularity.adapters.alpaca_crypto.stream
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from datetime import datetime, timezone

import orjson
import websockets
from tenacity import AsyncRetrying, stop_never, wait_exponential

from ...config import get_settings
from ...logs import configure as configure_logging
from ...logs import get_logger
from .book_features import BookFeatureEngine
from .influx_writer import InfluxWriter
from .orderbook import OrderBook, _parse_ts

log = get_logger(__name__)


class StreamCapture:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.writer = InfluxWriter(
            url=self.settings.influx_url,
            token=self.settings.influx_token,
            org=self.settings.influx_org,
            bucket=self.settings.influx_bucket_raw,
        )
        self.books: dict[str, OrderBook] = {
            sym: OrderBook(sym) for sym in self.settings.orderbook_symbols()
        }
        self.engines: dict[str, BookFeatureEngine] = {
            sym: BookFeatureEngine(sym) for sym in self.settings.orderbook_symbols()
        }
        self.last_msg_at: datetime | None = None
        self._stop = asyncio.Event()
        self._features_disabled = False

    async def run(self) -> None:
        await self.writer.start()
        feature_task = asyncio.create_task(self._feature_loop(), name="feature-loop")
        try:
            async for attempt in AsyncRetrying(
                stop=stop_never,
                wait=wait_exponential(multiplier=1, min=1, max=30),
                reraise=False,
            ):
                with attempt:
                    await self._connect_and_stream()
                if self._stop.is_set():
                    break
        finally:
            feature_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await feature_task
            await self.writer.stop()

    async def _connect_and_stream(self) -> None:
        url = self.settings.alpaca_crypto_stream_url
        log.info("ws_connecting", url=url)
        gap_started_at = self.last_msg_at
        async with websockets.connect(url, ping_interval=15, ping_timeout=10) as ws:
            await self._auth(ws)
            await self._subscribe(ws)
            if gap_started_at is not None:
                now = datetime.now(timezone.utc)
                gap = (now - gap_started_at).total_seconds()
                await self.writer.write(
                    self.writer.gap_event_point("*", gap, "reconnect", now)
                )
                log.warning("stream_gap_recovered", gap_s=gap)
            await self._recv_loop(ws)

    async def _auth(self, ws) -> None:
        await ws.send(
            orjson.dumps(
                {
                    "action": "auth",
                    "key": self.settings.alpaca_api_key,
                    "secret": self.settings.alpaca_secret_key,
                }
            )
        )
        raw = await ws.recv()
        msgs = orjson.loads(raw)
        for m in msgs:
            if m.get("T") == "error":
                raise RuntimeError(f"auth failed: {m}")
        log.info("ws_authenticated")

    async def _subscribe(self, ws) -> None:
        sub = {
            "action": "subscribe",
            "trades": self.settings.trades_symbols(),
            "quotes": self.settings.quotes_symbols(),
            "orderbooks": self.settings.orderbook_symbols(),
        }
        await ws.send(orjson.dumps(sub))
        log.info("ws_subscribed", **{k: v for k, v in sub.items() if k != "action"})

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            try:
                msgs = orjson.loads(raw)
            except orjson.JSONDecodeError:
                log.warning("bad_json", raw=raw[:200])
                continue
            for m in msgs:
                await self._dispatch(m)
            self.last_msg_at = datetime.now(timezone.utc)

    async def _dispatch(self, m: dict) -> None:
        t = m.get("T")
        try:
            if t == "t":
                await self._on_trade(m)
            elif t == "q":
                await self._on_quote(m)
            elif t == "o":
                await self._on_book(m)
            elif t in ("success", "subscription"):
                log.info("ws_control", **m)
            elif t == "error":
                log.error("ws_error", **m)
            else:
                # Unknown message type — logged so a silent Alpaca protocol change
                # (new "T" values) surfaces instead of being dropped invisibly.
                log.warning("ws_unknown_type", T=t)
        except (KeyError, ValueError, TypeError) as e:
            # A single malformed message must not kill the WS loop.
            log.warning("dispatch_error", T=t, error=str(e), msg=str(m)[:200])

    async def _on_trade(self, m: dict) -> None:
        ts = _parse_ts(m["t"])
        line = self.writer.trade_point(
            symbol=m["S"],
            price=float(m["p"]),
            size=float(m["s"]),
            side=m.get("tks", ""),  # taker side if present
            ts=ts,
        )
        await self.writer.write(line)

    async def _on_quote(self, m: dict) -> None:
        ts = _parse_ts(m["t"])
        line = self.writer.quote_point(
            symbol=m["S"],
            bid_px=float(m["bp"]),
            bid_sz=float(m["bs"]),
            ask_px=float(m["ap"]),
            ask_sz=float(m["as"]),
            ts=ts,
        )
        await self.writer.write(line)

    async def _on_book(self, m: dict) -> None:
        sym = m["S"]
        book = self.books.get(sym)
        if book is None:
            return
        ok = book.apply(m)
        if ok:
            now = book.last_update or datetime.now(timezone.utc)
            self.engines[sym].update(book, now)

    async def _feature_loop(self) -> None:
        cadence = self.settings.book_feature_cadence_s
        while not self._stop.is_set():
            await asyncio.sleep(cadence)
            now = datetime.now(timezone.utc)
            for sym, book in self.books.items():
                if self._features_disabled:
                    break
                try:
                    feats = self.engines[sym].snapshot(book, now)
                except NotImplementedError:
                    if not self._features_disabled:
                        log.warning(
                            "book_features_not_implemented",
                            note="fill in compute_book_features in book_features.py",
                        )
                        self._features_disabled = True
                    break
                if feats is None:
                    continue
                line = self.writer.book_features_point(
                    symbol=feats.symbol,
                    imb_1=feats.imb_1,
                    imb_5=feats.imb_5,
                    imb_10=feats.imb_10,
                    ofi_1m=feats.ofi_1m,
                    depth_slope=feats.depth_slope,
                    spread_bps=feats.spread_bps,
                    ts=feats.ts,
                )
                await self.writer.write(line)

    def request_stop(self) -> None:
        self._stop.set()


def _install_signal_handlers(capture: StreamCapture) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, capture.request_stop)
        except NotImplementedError:
            # Windows: signal handlers via add_signal_handler unsupported
            signal.signal(sig, lambda *_: capture.request_stop())


async def _amain() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        raise SystemExit(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY missing — copy .env.example to .env and fill in."
        )
    capture = StreamCapture()
    _install_signal_handlers(capture)
    log.info("phase0_capture_starting")
    await capture.run()
    log.info("phase0_capture_stopped")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
