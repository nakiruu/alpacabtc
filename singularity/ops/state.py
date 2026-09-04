"""Durable state store for orders, fills, positions, heartbeats.

Backed by SQLite. WAL mode so the executor can write while another process
(reconcile CLI, report, ad-hoc `sqlite3` inspection) reads without blocking.

Synchronous API on purpose. SQLite releases the GIL on IO; async callers wrap
individual methods in `asyncio.to_thread` at the call site. Adding an async
layer here would obscure that everything runs in one process and one thread
of execution as far as the store is concerned.

Plan §4.3 gate: "30 days of paper trading with zero unreconciled state events
and zero stranded positions." Every write to this store is a step toward proving
that gate, so consistency > convenience.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..costs.types import Fill, OrderIntent, OrderStatus, OrderType, Side, TimeInForce

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id                  TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,
    qty                 REAL NOT NULL,
    order_type          TEXT NOT NULL,
    tif                 TEXT NOT NULL,
    limit_price         REAL,
    submitted_at        TEXT NOT NULL,
    mid_at_submit       REAL NOT NULL,
    modeled_fee_bps     REAL NOT NULL,
    modeled_spread_bps  REAL NOT NULL,
    modeled_impact_bps  REAL NOT NULL,
    alpaca_order_id     TEXT,
    status              TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id            TEXT PRIMARY KEY,
    order_id      TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    qty           REAL NOT NULL,
    price         REAL NOT NULL,
    filled_at     TEXT NOT NULL,
    fee_asset     TEXT NOT NULL,
    fee_amount    REAL NOT NULL,
    is_maker      INTEGER NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(id)
);

CREATE TABLE IF NOT EXISTS positions (
    symbol           TEXT PRIMARY KEY,
    qty              REAL NOT NULL,
    avg_entry_price  REAL NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heartbeats (
    process     TEXT PRIMARY KEY,
    last_beat   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS ix_fills_order_id ON fills(order_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ---- Orders ----

    def save_intent(self, intent: OrderIntent) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT INTO orders (
                    id, symbol, side, qty, order_type, tif, limit_price,
                    submitted_at, mid_at_submit,
                    modeled_fee_bps, modeled_spread_bps, modeled_impact_bps,
                    alpaca_order_id, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    intent.id, intent.symbol, intent.side.value, intent.qty,
                    intent.order_type.value, intent.tif.value, intent.limit_price,
                    _iso(intent.submitted_at), intent.mid_at_submit,
                    intent.modeled_cost.fee_bps,
                    intent.modeled_cost.spread_bps,
                    intent.modeled_cost.impact_bps,
                    OrderStatus.PENDING.value, _now_iso(),
                ),
            )

    def mark_submitted(self, order_id: str, alpaca_order_id: str) -> None:
        with self._connect() as c:
            c.execute(
                "UPDATE orders SET alpaca_order_id=?, status=?, updated_at=? WHERE id=?",
                (alpaca_order_id, OrderStatus.SUBMITTED.value, _now_iso(), order_id),
            )

    def update_status(self, order_id: str, status: OrderStatus) -> None:
        with self._connect() as c:
            c.execute(
                "UPDATE orders SET status=?, updated_at=? WHERE id=?",
                (status.value, _now_iso(), order_id),
            )

    def get_order(self, order_id: str) -> sqlite3.Row | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM orders WHERE id=?", (order_id,)).fetchone()
        return row

    def open_orders(self) -> list[sqlite3.Row]:
        with self._connect() as c:
            return c.execute(
                "SELECT * FROM orders WHERE status IN (?, ?, ?) ORDER BY submitted_at",
                (OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value, OrderStatus.PARTIAL.value),
            ).fetchall()

    # ---- Fills ----

    def save_fill(self, fill: Fill) -> None:
        # Use ROWID-derived id if the caller doesn't have one — Alpaca fill events
        # carry their own unique id which is what we want in practice.
        fill_id = f"{fill.order_id}:{_iso(fill.filled_at)}"
        with self._connect() as c:
            c.execute(
                """
                INSERT OR IGNORE INTO fills (
                    id, order_id, symbol, side, qty, price, filled_at,
                    fee_asset, fee_amount, is_maker
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id, fill.order_id, fill.symbol, fill.side.value,
                    fill.qty, fill.price, _iso(fill.filled_at),
                    fill.fee_asset, fill.fee_amount, int(fill.is_maker),
                ),
            )

    def fills_for(self, order_id: str) -> list[sqlite3.Row]:
        with self._connect() as c:
            return c.execute(
                "SELECT * FROM fills WHERE order_id=? ORDER BY filled_at",
                (order_id,),
            ).fetchall()

    def update_fill_fee(
        self, order_id: str, filled_at: datetime, fee_asset: str, fee_amount: float
    ) -> None:
        """Update fee_asset/fee_amount on a specific fill. Used by CFEE reconciliation
        (T+1) when Alpaca posts fee activities that weren't on the original fill event."""
        fill_id = f"{order_id}:{_iso(filled_at)}"
        with self._connect() as c:
            c.execute(
                "UPDATE fills SET fee_asset=?, fee_amount=? WHERE id=?",
                (fee_asset, fee_amount, fill_id),
            )

    def all_fills(self, limit: int = 1000) -> list[sqlite3.Row]:
        with self._connect() as c:
            return c.execute(
                "SELECT * FROM fills ORDER BY filled_at DESC LIMIT ?", (limit,)
            ).fetchall()

    def fills_between(self, start: datetime, end: datetime) -> list[sqlite3.Row]:
        with self._connect() as c:
            return c.execute(
                "SELECT * FROM fills WHERE filled_at BETWEEN ? AND ? ORDER BY filled_at",
                (_iso(start), _iso(end)),
            ).fetchall()

    # ---- Positions ----

    def upsert_position(self, symbol: str, qty: float, avg_entry_price: float) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT INTO positions (symbol, qty, avg_entry_price, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    qty=excluded.qty,
                    avg_entry_price=excluded.avg_entry_price,
                    updated_at=excluded.updated_at
                """,
                (symbol, qty, avg_entry_price, _now_iso()),
            )

    def get_position(self, symbol: str) -> sqlite3.Row | None:
        with self._connect() as c:
            return c.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()

    def all_positions(self) -> list[sqlite3.Row]:
        with self._connect() as c:
            return c.execute("SELECT * FROM positions WHERE qty != 0").fetchall()

    # ---- Heartbeats ----

    def heartbeat(self, process: str) -> None:
        with self._connect() as c:
            c.execute(
                """
                INSERT INTO heartbeats (process, last_beat) VALUES (?, ?)
                ON CONFLICT(process) DO UPDATE SET last_beat=excluded.last_beat
                """,
                (process, _now_iso()),
            )

    def last_heartbeat(self, process: str) -> datetime | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT last_beat FROM heartbeats WHERE process=?", (process,)
            ).fetchone()
        return _from_iso(row["last_beat"]) if row else None

    def heartbeat_age_s(self, process: str) -> float | None:
        last = self.last_heartbeat(process)
        if last is None:
            return None
        return (datetime.now(timezone.utc) - last).total_seconds()


def row_to_intent(row: sqlite3.Row) -> OrderIntent:
    """Rebuild an OrderIntent from a stored row. Modeled cost gets reconstructed too."""
    from ..costs.types import Cost
    return OrderIntent(
        id=row["id"],
        symbol=row["symbol"],
        side=Side(row["side"]),
        qty=row["qty"],
        order_type=OrderType(row["order_type"]),
        tif=TimeInForce(row["tif"]),
        limit_price=row["limit_price"],
        submitted_at=_from_iso(row["submitted_at"]),
        mid_at_submit=row["mid_at_submit"],
        modeled_cost=Cost(
            fee_bps=row["modeled_fee_bps"],
            spread_bps=row["modeled_spread_bps"],
            impact_bps=row["modeled_impact_bps"],
        ),
    )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row) if row is not None else {}
