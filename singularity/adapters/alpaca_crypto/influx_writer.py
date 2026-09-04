"""Buffered async writer to InfluxDB using line protocol.

We build line-protocol strings in a bounded queue and flush every ``flush_interval_s``
seconds, or when the buffer hits ``max_batch``. The synchronous influxdb-client is
called from a worker task; it releases the GIL on network I/O so this is fine for
Phase 0 throughput.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from ...logs import get_logger

log = get_logger(__name__)


def _to_ns(ts: datetime) -> int:
    return int(ts.timestamp() * 1_000_000_000)


def _escape_tag(v: str) -> str:
    return v.replace(",", "\\,").replace(" ", "\\ ").replace("=", "\\=")


class InfluxWriter:
    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        flush_interval_s: float = 1.0,
        max_batch: int = 5000,
    ) -> None:
        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
        self._bucket = bucket
        self._flush_interval_s = flush_interval_s
        self._max_batch = max_batch
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop(), name="influx-flush")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            await self._task
        await self._flush_now()
        self._client.close()

    async def write(self, line: str) -> None:
        async with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._max_batch:
                await self._flush_now_locked()

    async def _flush_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self._flush_interval_s)
            await self._flush_now()

    async def _flush_now(self) -> None:
        async with self._lock:
            await self._flush_now_locked()

    async def _flush_now_locked(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer
        self._buffer = []
        try:
            await asyncio.to_thread(
                self._write_api.write,
                bucket=self._bucket,
                record=batch,
            )
        except Exception:
            log.exception("influx_write_failed", batch_size=len(batch))
            # Restore batch at the head so we retry on next flush. Cap buffer at
            # max_batch * 3 to bound growth when Influx is durably down.
            cap = self._max_batch * 3
            combined = batch + self._buffer
            if len(combined) > cap:
                log.warning("influx_buffer_overflow_drop", dropped=len(combined) - cap)
                combined = combined[-cap:]
            self._buffer = combined

    # ---- convenience builders (line protocol) ----

    def trade_point(self, symbol: str, price: float, size: float, side: str, ts: datetime) -> str:
        return (
            f"crypto_trade,symbol={_escape_tag(symbol)} "
            f"price={price},size={size},side=\"{side}\" "
            f"{_to_ns(ts)}"
        )

    def quote_point(
        self,
        symbol: str,
        bid_px: float,
        bid_sz: float,
        ask_px: float,
        ask_sz: float,
        ts: datetime,
    ) -> str:
        return (
            f"crypto_quote,symbol={_escape_tag(symbol)} "
            f"bid_px={bid_px},bid_sz={bid_sz},ask_px={ask_px},ask_sz={ask_sz} "
            f"{_to_ns(ts)}"
        )

    def book_features_point(
        self,
        symbol: str,
        imb_1: float,
        imb_5: float,
        imb_10: float,
        ofi_1m: float,
        depth_slope: float,
        spread_bps: float,
        ts: datetime,
    ) -> str:
        return (
            f"book_features,symbol={_escape_tag(symbol)} "
            f"imb_1={imb_1},imb_5={imb_5},imb_10={imb_10},"
            f"ofi_1m={ofi_1m},depth_slope={depth_slope},spread_bps={spread_bps} "
            f"{_to_ns(ts)}"
        )

    def gap_event_point(self, symbol: str, gap_s: float, reason: str, ts: datetime) -> str:
        return (
            f"stream_gap,symbol={_escape_tag(symbol)},reason={_escape_tag(reason)} "
            f"gap_s={gap_s} {_to_ns(ts)}"
        )
