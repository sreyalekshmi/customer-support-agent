"""get_payment_status tool logic (read-only).

The input is a Stripe PaymentIntent ID (pi_...). It is NOT an order ID and no
Automatic order-to-payment lookup is not implemented here.
"""

import re
from typing import Any

from app.mcp_layer import services
from app.mcp_layer.backends.base import PaymentBackend
from app.mcp_layer.tools.common import error_result, new_request_id, run_backend

TOOL_NAME = "get_payment_status"
_PAYMENT_ID = re.compile(r"^pi_[A-Za-z0-9]{6,64}$")  # case-sensitive, as Stripe IDs are


def lookup_payment_status(
    payment_id: str,
    request_id: str | None = None,
    backend: PaymentBackend | None = None,
) -> dict[str, Any]:
    request_id = new_request_id(request_id)
    payment_id = payment_id.strip()
    if not _PAYMENT_ID.match(payment_id):
        return error_result(
            TOOL_NAME, request_id, "INVALID_PAYMENT_ID",
            "Payment IDs are Stripe PaymentIntent IDs that start with pi_ (not order IDs).",
        )
    return run_backend(
        TOOL_NAME, request_id,
        lambda: (backend or services.get_payment_backend()).get_payment_status(payment_id),
    )
