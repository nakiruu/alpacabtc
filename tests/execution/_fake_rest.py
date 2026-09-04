"""Test double for AlpacaRestClient. Records calls, returns configured payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeRest:
    open_orders: list[dict] = field(default_factory=list)
    all_orders: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    canceled: list[str] = field(default_factory=list)
    cancel_all_calls: int = 0
    closed_positions: list[str] = field(default_factory=list)

    async def get_orders(self, status: str = "open", limit: int = 500) -> list[dict[str, Any]]:
        return list(self.all_orders if status == "all" else self.open_orders)

    async def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    async def cancel_order(self, alpaca_order_id: str) -> None:
        self.canceled.append(alpaca_order_id)

    async def cancel_all_orders(self) -> list[dict[str, Any]]:
        self.cancel_all_calls += 1
        return []

    async def close_position(self, symbol: str) -> dict[str, Any]:
        self.closed_positions.append(symbol)
        return {"symbol": symbol, "status": "closed"}

    async def close(self) -> None:
        pass
