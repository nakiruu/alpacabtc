"""Alpaca REST client — account, positions, orders, activities.

Thin async wrapper over httpx. Raises for HTTP errors so callers see failures
loudly rather than silent None returns. Uses `client_order_id` for idempotent
POSTs — retrying the same intent gets rejected as duplicate, which is what we
want during reconciliation.

Base URL comes from ``ALPACA_TRADING_URL`` in .env. Paper default:
    https://paper-api.alpaca.markets

Docs: https://docs.alpaca.markets/reference (trading API v2)
"""

from __future__ import annotations

from typing import Any

import httpx


class AlpacaRestClient:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        timeout_s: float = 10.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=timeout_s,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AlpacaRestClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # ---- Account ----

    async def get_account(self) -> dict[str, Any]:
        r = await self._client.get("/v2/account")
        r.raise_for_status()
        return r.json()

    # ---- Positions ----

    async def get_positions(self) -> list[dict[str, Any]]:
        r = await self._client.get("/v2/positions")
        r.raise_for_status()
        return r.json()

    async def close_position(self, symbol: str) -> dict[str, Any]:
        # symbol has a slash for crypto — needs URL encoding
        r = await self._client.delete(f"/v2/positions/{symbol}")
        r.raise_for_status()
        return r.json()

    # ---- Orders ----

    async def get_orders(
        self, status: str = "open", limit: int = 500
    ) -> list[dict[str, Any]]:
        r = await self._client.get(
            "/v2/orders",
            params={"status": status, "limit": limit, "direction": "desc"},
        )
        r.raise_for_status()
        return r.json()

    async def submit_order(
        self,
        *,
        symbol: str,
        side: str,          # "buy" | "sell"
        qty: float,
        order_type: str,    # "market" | "limit" | "stop_limit"
        tif: str,           # "gtc" | "ioc"
        client_order_id: str,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": tif,
            "client_order_id": client_order_id,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
        r = await self._client.post("/v2/orders", json=payload)
        r.raise_for_status()
        return r.json()

    async def cancel_order(self, alpaca_order_id: str) -> None:
        r = await self._client.delete(f"/v2/orders/{alpaca_order_id}")
        # Alpaca returns 204 No Content on success
        if r.status_code not in (200, 204):
            r.raise_for_status()

    async def cancel_all_orders(self) -> list[dict[str, Any]]:
        r = await self._client.delete("/v2/orders")
        r.raise_for_status()
        return r.json() if r.content else []

    # ---- Activities (for CFEE reconciliation) ----

    async def get_activities(
        self,
        activity_types: str = "FILL,CFEE",
        after: str | None = None,
        until: str | None = None,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"activity_types": activity_types, "page_size": page_size}
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        r = await self._client.get("/v2/account/activities", params=params)
        r.raise_for_status()
        return r.json()
