"""Shared backend types.

A backend turns a validated ID into a plain dict of customer-safe fields, or raises
BackendError. Tool functions convert BackendError into the shared ToolResult shape.
Backends must never return data from a different source than the one configured.
"""

from typing import Any, Protocol


class BackendError(Exception):
    """A failure with a stable machine-readable code.

    `message` is safe to show to an agent/customer. `detail` is for stderr logs only
    and must never contain secrets or customer personal data.
    """

    def __init__(self, code: str, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


# Error codes used across backends.
CODES = (
    "INVALID_ORDER_ID", "INVALID_PAYMENT_ID",
    "ORDER_NOT_FOUND", "PAYMENT_NOT_FOUND",
    "AUTH_FAILED", "PERMISSION_DENIED", "RATE_LIMITED",
    "UPSTREAM_ERROR", "UPSTREAM_TIMEOUT", "MALFORMED_RESPONSE",
    "CONFIG_ERROR", "BACKEND_ERROR", "INTERNAL_ERROR",
)

GENERIC_UNAVAILABLE = "The service is temporarily unavailable."


class OrderBackend(Protocol):
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """order_id is canonical (ORD-1234 or ORD-D10). Returns data or raises BackendError."""


class PaymentBackend(Protocol):
    def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        """payment_id is a Stripe PaymentIntent ID. Returns the `data` dict or raises BackendError."""
