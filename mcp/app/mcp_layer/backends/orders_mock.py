"""MOCK order backend: deterministic local fixtures (data/sample_orders.json)."""

import json
from pathlib import Path
from typing import Any

from app.mcp_layer.backends.base import BackendError

DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "sample_orders.json"


class MockOrderBackend:
    source = "mock"

    def __init__(self, data_file: Path = DATA_FILE):
        self._data_file = data_file

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        try:
            with self._data_file.open(encoding="utf-8") as f:
                orders = json.load(f)["orders"]
        except (OSError, ValueError, KeyError) as exc:
            raise BackendError("BACKEND_ERROR", "Order service is unavailable.",
                               detail=type(exc).__name__) from exc

        order = orders.get(order_id)
        if order is None:
            raise BackendError("ORDER_NOT_FOUND", f"No order found with ID {order_id}.")

        # Only fields a support answer needs: no addresses, emails or payment details.
        return {
            "order_id": order_id,
            "status": order["status"],
            "financial_status": order.get("financial_status"),
            "fulfillment_status": order.get("fulfillment_status"),
            "carrier": order["carrier"],
            "tracking_number": order["tracking_number"],
            "tracking_url": order.get("tracking_url"),
            "estimated_delivery": order["estimated_delivery"],
            "shopify_order_name": None,
            "simulated": True,
            "source": "mock",
        }
