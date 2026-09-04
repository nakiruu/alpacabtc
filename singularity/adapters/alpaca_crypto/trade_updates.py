"""Trade updates WebSocket — Alpaca account-level order/fill events.

Endpoint (paper): wss://paper-api.alpaca.markets/stream
Payload shape (event names lowercased by Alpaca):

    {"stream": "trade_updates",
     "data": {
        "event": "fill" | "partial_fill" | "new" | "canceled" | "rejected" | ...,
        "order": {
            "id": "...",
            "client_order_id": "...",
            "symbol": "BTC/USD",
            "side": "buy",
            "qty": "0.001",
            "filled_qty": "0.001",
            "filled_avg_price": "50000.00",
            "status": "filled",
            "order_class": "",
            "order_type": "limit",
            ...
        },
        "price": "50000.00",     # per-fill for fill/partial_fill
        "qty": "0.001",          # per-fill delta qty
        "timestamp": "2026-01-01T00:00:00.123Z"
     }}

Fees are NOT on the fill event for crypto — they arrive T+1 via the CFEE
activity. Callers should persist the fill with fee_amount=0 and reconcile
fees via the Activities API (batch 3b).

Reconnect: exponential backoff via tenacity. Emits a stream_gap event on
recovery so the report can see it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import orjson
import websockets
from tenacity import AsyncRetrying, stop_never, wait_exponential

from ...logs import get_logger

log = get_logger(__name__)


TradeUpdate = dict  # kept as raw dict; parsing into typed events happens in fill_handler


class TradeUpdatesClient:
    """Subscribes to account trade updates and hands each event to a handler."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        stream_url: str,
        on_event: Callable[[TradeUpdate], Awaitable[None]],
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._stream_url = stream_url
        self._on_event = on_event
        self._stop = asyncio.Event()
        self._last_msg_at: datetime | None = None

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        async for attempt in AsyncRetrying(
            stop=stop_never,
            wait=wait_exponential(multiplier=1, min=1, max=30),
            reraise=False,
        ):
            with attempt:
                await self._connect_and_stream()
            if self._stop.is_set():
                break

    async def _connect_and_stream(self) -> None:
        log.info("trade_updates_connecting", url=self._stream_url)
        gap_started = self._last_msg_at
        async with websockets.connect(
            self._stream_url, ping_interval=15, ping_timeout=10
        ) as ws:
            await self._auth(ws)
            await self._subscribe(ws)
            if gap_started is not None:
                now = datetime.now(timezone.utc)
                gap = (now - gap_started).total_seconds()
                log.warning("trade_updates_gap_recovered", gap_s=gap)
            await self._recv_loop(ws)

    async def _auth(self, ws) -> None:
        await ws.send(
            orjson.dumps(
                {
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._secret_key,
                }
            )
        )
        raw = await ws.recv()
        msg = orjson.loads(raw)
        # Alpaca returns a message like {"stream":"authorization","data":{"status":"authorized"}}
        data = msg.get("data", {})
        if data.get("status") != "authorized":
            raise RuntimeError(f"trade_updates auth failed: {msg}")
        log.info("trade_updates_authenticated")

    async def _subscribe(self, ws) -> None:
        await ws.send(
            orjson.dumps(
                {"action": "listen", "data": {"streams": ["trade_updates"]}}
            )
        )
        # Alpaca replies with a listening ack; log and continue
        raw = await ws.recv()
        msg = orjson.loads(raw)
        log.info("trade_updates_subscribed", ack=msg)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            try:
                msg = orjson.loads(raw)
            except orjson.JSONDecodeError:
                log.warning("trade_updates_bad_json", raw=raw[:200])
                continue
            self._last_msg_at = datetime.now(timezone.utc)
            if msg.get("stream") != "trade_updates":
                # Alpaca also multiplexes 'listening' / 'authorization' acks
                continue
            try:
                await self._on_event(msg["data"])
            except Exception:
                log.exception("trade_updates_handler_error", ev_type=msg.get("data", {}).get("event"))
