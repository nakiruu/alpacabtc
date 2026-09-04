"""Test doubles for AlpacaRestClient + MarketDataClient. Records calls, returns configured payloads."""

from __future__ import annotations

from datetime import datetime, timezone
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
    submitted: list[dict] = field(default_factory=list)
    submit_response_id: str = "alp-generated"

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

    async def submit_order(self, **kwargs) -> dict[str, Any]:
        self.submitted.append(kwargs)
        return {"id": self.submit_response_id, "status": "new", **kwargs}

    async def get_activities(self, activity_types="FILL,CFEE", after=None, until=None, page_size=100):
        return []

    async def close(self) -> None:
        pass


@dataclass
class FakeMarket:
    """Test double for MarketDataClient — returns a fixed quote/trade."""

    bid_px: float = 49_999.0
    bid_sz: float = 0.5
    ask_px: float = 50_001.0
    ask_sz: float = 0.5
    trade_price: float = 50_000.0

    async def latest_quote(self, symbol: str) -> dict[str, Any] | None:
        return {
            "bid_px": self.bid_px, "bid_sz": self.bid_sz,
            "ask_px": self.ask_px, "ask_sz": self.ask_sz,
            "ts": datetime.now(timezone.utc),
        }

    async def latest_trade(self, symbol: str) -> dict[str, Any] | None:
        return {
            "price": self.trade_price, "size": 0.01,
            "ts": datetime.now(timezone.utc),
        }

    async def close(self) -> None:
        pass
