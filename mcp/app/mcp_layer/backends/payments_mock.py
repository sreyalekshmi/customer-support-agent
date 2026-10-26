"""MOCK payment backend: deterministic local fixtures (data/sample_payments.json)."""

import json
from pathlib import Path
from typing import Any

from app.mcp_layer.backends.base import BackendError

DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "sample_payments.json"


class MockPaymentBackend:
    source = "mock"

    def __init__(self, data_file: Path = DATA_FILE):
        self._data_file = data_file

    def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        try:
            with self._data_file.open(encoding="utf-8") as f:
                payments = json.load(f)["payments"]
        except (OSError, ValueError, KeyError) as exc:
            raise BackendError("BACKEND_ERROR", "Payment service is unavailable.",
                               detail=type(exc).__name__) from exc

        p = payments.get(payment_id)
        if p is None:
            raise BackendError("PAYMENT_NOT_FOUND", f"No payment found with ID {payment_id}.")
        return {
            "payment_id": payment_id,
            "status": p["status"],
            "status_summary": p["status_summary"],
            "amount_minor": p["amount_minor"],
            "currency": p["currency"],
            "amount_received_minor": p["amount_received_minor"],
            "amount_refunded_minor": p.get("amount_refunded_minor"),
            "created": p["created"],
            "failure_code": p.get("failure_code"),
            "livemode": False,
            "simulated": True,
            "source": "mock",
        }
