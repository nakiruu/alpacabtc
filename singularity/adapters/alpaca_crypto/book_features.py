"""Derived L2 microstructure features computed on ingest.

Ingest cadence is set by ``BOOK_FEATURE_CADENCE_S`` (default 1s per plan §2).
We keep only these derived series long-term; raw books age out fast because
the *features* are what downstream signals actually consume.

Feature contract (plan §2 measurement `book_features`):

    imb_1        top-of-book size imbalance
    imb_5        depth-5 aggregated size imbalance
    imb_10       depth-10 aggregated size imbalance
    ofi_1m       order-flow imbalance over the last 60s
    depth_slope  price impact of adding one bps of size at the top
    spread_bps   (ask - bid) / mid in basis points

The exact formulas are a decision that lives in ``compute_book_features``
below — see the TODO block.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .orderbook import OrderBook


@dataclass
class BookFeatures:
    symbol: str
    ts: datetime
    imb_1: float
    imb_5: float
    imb_10: float
    ofi_1m: float
    depth_slope: float
    spread_bps: float


@dataclass
class _OFIState:
    """Running order-flow-imbalance accumulator over a rolling window."""

    window: timedelta
    events: deque = None  # deque of (ts, delta_bid_size, delta_ask_size)

    def __post_init__(self) -> None:
        if self.events is None:
            self.events = deque()

    def push(self, ts: datetime, dbid: float, dask: float) -> None:
        self.events.append((ts, dbid, dask))
        cutoff = ts - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def value(self) -> float:
        # signed order-flow imbalance: positive → buy pressure
        return sum(dbid - dask for _, dbid, dask in self.events)


class BookFeatureEngine:
    """Stateful per-symbol feature computer.

    Call ``update(book)`` after every incoming book update; call ``snapshot(now)``
    on your cadence timer to emit a ``BookFeatures`` sample (or None if the book
    isn't ready).
    """

    def __init__(self, symbol: str, ofi_window_s: float = 60.0) -> None:
        self.symbol = symbol
        self._ofi = _OFIState(window=timedelta(seconds=ofi_window_s))
        self._prev_best_bid_size: float | None = None
        self._prev_best_ask_size: float | None = None

    def update(self, book: OrderBook, now: datetime) -> None:
        bid, ask = book.top()
        if bid is None or ask is None:
            return
        dbid = 0.0 if self._prev_best_bid_size is None else bid[1] - self._prev_best_bid_size
        dask = 0.0 if self._prev_best_ask_size is None else ask[1] - self._prev_best_ask_size
        self._ofi.push(now, dbid, dask)
        self._prev_best_bid_size = bid[1]
        self._prev_best_ask_size = ask[1]

    def snapshot(self, book: OrderBook, now: datetime) -> BookFeatures | None:
        if not book.initialized:
            return None
        bid, ask = book.top()
        if bid is None or ask is None:
            return None
        bids10 = book.bids_desc(10)
        asks10 = book.asks_asc(10)
        return compute_book_features(
            symbol=self.symbol,
            ts=now,
            bids=bids10,
            asks=asks10,
            ofi_1m=self._ofi.value(),
        )


# ---------------------------------------------------------------------------
# Implementation notes — semantics locked here so Phase 1 cost calibration can
# attribute correctly.
# ---------------------------------------------------------------------------
#   imb_N       Bounded size ratio  (Σ bid_size − Σ ask_size) / (Σ bid_size + Σ ask_size)
#               over the top-N levels each side, ∈ [−1, +1].  Chosen over log-ratio
#               because it stays finite when either side thins out.
#   mid         Arithmetic  (best_bid + best_ask) / 2.  NOT microprice — keeps
#               spread_bps and imb_1 as independent signals rather than double-
#               counting book pressure. Microprice belongs in a separate feature
#               if we want it downstream.
#   spread_bps  (ask − bid) / mid × 1e4.
#   depth_slope OLS slope of  cumulative_ask_size  regressed on  price_offset_bps
#               across the top-10 asks. Units: base-asset size per bps.
#               Higher = book stacks up faster = deeper = lower market-impact cost.
#               Ask side only: we run long-only spot, so buy-side impact matters.
#   ofi_1m     Passed through from the engine. Naive form:  Σ(Δbid_top_sz − Δask_top_sz)
#              over the trailing 60s. Signed; positive = net buy pressure at the touch.
# ---------------------------------------------------------------------------


def compute_book_features(
    symbol: str,
    ts: datetime,
    bids: list[tuple[float, float]],  # (price, size), highest-first
    asks: list[tuple[float, float]],  # (price, size), lowest-first
    ofi_1m: float,
) -> BookFeatures:
    if not bids or not asks:
        # Book empty on one side — shouldn't happen after a snapshot but be defensive
        return BookFeatures(symbol, ts, 0.0, 0.0, 0.0, ofi_1m, 0.0, 0.0)

    best_bid_px = bids[0][0]
    best_ask_px = asks[0][0]
    mid = 0.5 * (best_bid_px + best_ask_px)
    spread_bps = (best_ask_px - best_bid_px) / mid * 1e4

    imb_1 = _bounded_imbalance(bids[:1], asks[:1])
    imb_5 = _bounded_imbalance(bids[:5], asks[:5])
    imb_10 = _bounded_imbalance(bids[:10], asks[:10])

    depth_slope = _ask_depth_slope(asks, mid)

    return BookFeatures(
        symbol=symbol,
        ts=ts,
        imb_1=imb_1,
        imb_5=imb_5,
        imb_10=imb_10,
        ofi_1m=ofi_1m,
        depth_slope=depth_slope,
        spread_bps=spread_bps,
    )


def _bounded_imbalance(
    bids: list[tuple[float, float]], asks: list[tuple[float, float]]
) -> float:
    b = sum(sz for _, sz in bids)
    a = sum(sz for _, sz in asks)
    denom = b + a
    if denom <= 0.0:
        return 0.0
    return (b - a) / denom


def _ask_depth_slope(asks: list[tuple[float, float]], mid: float) -> float:
    """OLS slope of cumulative ask size on price offset in bps."""
    if len(asks) < 2 or mid <= 0.0:
        return 0.0
    xs: list[float] = []
    ys: list[float] = []
    cum = 0.0
    for px, sz in asks:
        cum += sz
        xs.append((px - mid) / mid * 1e4)
        ys.append(cum)
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = xs[i] - mx
        num += dx * (ys[i] - my)
        den += dx * dx
    if den < 1e-12:
        return 0.0
    return num / den
