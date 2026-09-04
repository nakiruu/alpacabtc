"""L2 orderbook state maintained from Alpaca crypto WS deltas.

Alpaca sends periodic full snapshots marked ``r=True`` and delta updates otherwise.
On a snapshot we replace state. On a delta we upsert (or remove when size == 0).
Until a snapshot is received we ignore deltas — a book rebuilt from deltas alone is
unsound.

Gap detection: we record the last message time per symbol and let callers query
staleness. A feature computed across a silent gap is a landmine (plan §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sortedcontainers import SortedDict

Level = tuple[float, float]  # (price, size)


@dataclass
class OrderBook:
    symbol: str
    # bids: price → size, iterated in descending price order via .keys()[::-1]
    bids: SortedDict = field(default_factory=SortedDict)
    # asks: price → size, iterated in ascending price order
    asks: SortedDict = field(default_factory=SortedDict)
    last_update: datetime | None = None
    initialized: bool = False  # True once a snapshot has been applied

    def apply(self, msg: dict) -> bool:
        """Apply an Alpaca orderbook message. Returns True if state now valid."""
        is_snapshot = bool(msg.get("r", False))
        if is_snapshot:
            # Build the new book locally and swap in only after successful parse.
            # A mid-parse failure otherwise leaves us with initialized=True over
            # a half-cleared book, which then applies deltas onto garbage.
            new_bids: SortedDict = SortedDict()
            new_asks: SortedDict = SortedDict()
            for lvl in msg.get("b", []):
                _upsert(new_bids, float(lvl["p"]), float(lvl["s"]))
            for lvl in msg.get("a", []):
                _upsert(new_asks, float(lvl["p"]), float(lvl["s"]))
            self.bids = new_bids
            self.asks = new_asks
            self.initialized = True
        else:
            if not self.initialized:
                # Delta before snapshot — cannot trust state, drop
                return False
            for lvl in msg.get("b", []):
                _upsert(self.bids, float(lvl["p"]), float(lvl["s"]))
            for lvl in msg.get("a", []):
                _upsert(self.asks, float(lvl["p"]), float(lvl["s"]))

        ts = msg.get("t")
        if ts:
            self.last_update = _parse_ts(ts)
        return self.initialized

    def top(self) -> tuple[Level | None, Level | None]:
        best_bid = self._best_bid()
        best_ask = self._best_ask()
        return best_bid, best_ask

    def _best_bid(self) -> Level | None:
        if not self.bids:
            return None
        px = self.bids.keys()[-1]  # highest bid
        return (px, self.bids[px])

    def _best_ask(self) -> Level | None:
        if not self.asks:
            return None
        px = self.asks.keys()[0]  # lowest ask
        return (px, self.asks[px])

    def bids_desc(self, depth: int) -> list[Level]:
        return [(px, self.bids[px]) for px in reversed(self.bids.keys()[-depth:])]

    def asks_asc(self, depth: int) -> list[Level]:
        return [(px, self.asks[px]) for px in self.asks.keys()[:depth]]

    def staleness_s(self, now: datetime) -> float | None:
        if self.last_update is None:
            return None
        return (now - self.last_update).total_seconds()


def _upsert(side: SortedDict, price: float, size: float) -> None:
    if size <= 0:
        side.pop(price, None)
    else:
        side[price] = size


def _parse_ts(ts: str) -> datetime:
    # Alpaca uses RFC3339 with 'Z' suffix; python 3.11+ fromisoformat handles it
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)
