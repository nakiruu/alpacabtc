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

import itertools
import statistics
from collections import deque
from dataclasses import dataclass, field
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
    events: deque = field(default_factory=deque)  # (ts, dbid, dask)

    def push(self, ts: datetime, dbid: float, dask: float) -> None:
        self.events.append((ts, dbid, dask))
        cutoff = ts - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def value(self) -> float:
        # signed order-flow imbalance: positive → buy pressure
        return sum(dbid - dask for _, dbid, dask in self.events)


def _ofi_side_delta(
    *, new_px: float, new_sz: float, prev_px: float | None, prev_sz: float | None, is_bid: bool
) -> float:
    """Cont-Kukanov-Stoikov OFI contribution for one side of the touch.

    For bids, an improved (higher) best price = fresh buy liquidity → +new_sz.
    A worsened (lower) best price = old bid was consumed/pulled → -prev_sz.
    Same-price change = arrivals/cancels at the level → new_sz - prev_sz.
    Ask side is mirrored (improvement = lower price).
    """
    if prev_px is None or prev_sz is None:
        return 0.0
    if is_bid:
        if new_px > prev_px:
            return float(new_sz)
        if new_px < prev_px:
            return -float(prev_sz)
        return float(new_sz - prev_sz)
    # ask side
    if new_px < prev_px:
        return float(new_sz)
    if new_px > prev_px:
        return -float(prev_sz)
    return float(new_sz - prev_sz)


class BookFeatureEngine:
    """Stateful per-symbol feature computer.

    Call ``update(book)`` after every incoming book update; call ``snapshot(now)``
    on your cadence timer to emit a ``BookFeatures`` sample (or None if the book
    isn't ready).
    """

    def __init__(self, symbol: str, ofi_window_s: float = 60.0) -> None:
        self.symbol = symbol
        self._ofi = _OFIState(window=timedelta(seconds=ofi_window_s))
        self._prev_bid_px: float | None = None
        self._prev_bid_sz: float | None = None
        self._prev_ask_px: float | None = None
        self._prev_ask_sz: float | None = None

    def update(self, book: OrderBook, now: datetime) -> None:
        bid, ask = book.top()
        if bid is None or ask is None:
            return
        bid_px, bid_sz = bid
        ask_px, ask_sz = ask
        dbid = _ofi_side_delta(
            new_px=bid_px, new_sz=bid_sz,
            prev_px=self._prev_bid_px, prev_sz=self._prev_bid_sz,
            is_bid=True,
        )
        dask = _ofi_side_delta(
            new_px=ask_px, new_sz=ask_sz,
            prev_px=self._prev_ask_px, prev_sz=self._prev_ask_sz,
            is_bid=False,
        )
        self._ofi.push(now, dbid, dask)
        self._prev_bid_px, self._prev_bid_sz = bid_px, bid_sz
        self._prev_ask_px, self._prev_ask_sz = ask_px, ask_sz

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
    xs = [(px - mid) / mid * 1e4 for px, _ in asks]
    ys = list(itertools.accumulate(sz for _, sz in asks))
    try:
        return statistics.linear_regression(xs, ys).slope
    except statistics.StatisticsError:
        return 0.0  # degenerate x-range
