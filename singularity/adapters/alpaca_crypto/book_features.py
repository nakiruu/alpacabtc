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
# TODO — user contribution
# ---------------------------------------------------------------------------
# Implement the six microstructure formulas below. The engine already:
#   * hands you the top-10 levels each side (bids highest-first, asks lowest-first)
#   * hands you the running OFI over the last 60s (net Δbid_size − Δask_size)
#   * calls this each second per BOOK_FEATURE_CADENCE_S
#
# Design choices to consider (there is no one right answer):
#   1. Imbalance normalization. Classic:  (Σ bid_size − Σ ask_size) / (Σ bid_size + Σ ask_size)
#      Bounded [-1, +1] and scale-invariant. Alternative: log-ratio ln(Σbid / Σask),
#      which is symmetric and more sensitive at the tails.
#   2. Depth slope. Two common shapes:
#        (a) linear regression of cumulative size on price offset in bps
#        (b) reciprocal:  1 / (bps needed to buy 1 BTC at market)
#      (a) is more informative but (b) is more directly a "cost" number.
#   3. Spread. Trivially (ask − bid) / mid * 1e4. But do you want mid = arithmetic mean,
#      or microprice = (ask*bid_size + bid*ask_size) / (bid_size + ask_size)? Microprice
#      is a better forecast of the next trade price when the book is imbalanced.
#
# Whatever you pick, write it down in a comment — the calibration in Phase 1 needs
# to know exactly what "spread_bps" means to attribute cost correctly.
# ---------------------------------------------------------------------------


def compute_book_features(
    symbol: str,
    ts: datetime,
    bids: list[tuple[float, float]],  # (price, size), highest-first
    asks: list[tuple[float, float]],  # (price, size), lowest-first
    ofi_1m: float,
) -> BookFeatures:
    raise NotImplementedError(
        "Implement microstructure features — see TODO block above. "
        "Return BookFeatures(symbol, ts, imb_1, imb_5, imb_10, ofi_1m, depth_slope, spread_bps)."
    )
