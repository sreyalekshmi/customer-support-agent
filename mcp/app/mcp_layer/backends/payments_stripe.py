"""Stripe backend: read-only PaymentIntent lookup, TEST MODE ONLY.

- Only GET /v1/payment_intents/{id} is ever called. There is no refund, capture or cancel code.
- Config refuses live keys (see app.config.load_stripe_config) and this backend additionally
  refuses any response with livemode=true.
- A Shopify order ID is NOT a Stripe ID.
  associated with its PaymentIntent.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from app.config import StripeConfig
from app.mcp_layer.backends.base import GENERIC_UNAVAILABLE, BackendError

log = logging.getLogger("mcp_layer.stripe")

API_BASE = "https://api.stripe.com/v1"
MAX_ATTEMPTS = 3

STATUS_SUMMARY = {
    "succeeded": "paid",
    "processing": "processing",
    "requires_payment_method": "payment not completed",
    "requires_confirmation": "awaiting confirmation",
    "requires_action": "awaiting customer action",
    "requires_capture": "authorized, not yet captured",
    "canceled": "canceled",
}


class StripePaymentBackend:
    source = "stripe_test"

    def __init__(
        self,
        cfg: StripeConfig,
        client: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._cfg = cfg
        self._client = client or httpx.Client(timeout=cfg.timeout_seconds)
        self._sleep = sleep

    def get_payment_status(self, payment_id: str) -> dict[str, Any]:
        body = self._get(payment_id)
        try:
            if body.get("object") != "payment_intent" or body.get("id") != payment_id:
                raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="unexpected object/id")
            if body.get("livemode") is not False:
                # Defence in depth: never surface live-mode data from this prototype.
                raise BackendError("CONFIG_ERROR", "Live-mode payment data is not permitted here.",
                                   detail="livemode != false")
            status = body["status"]
            amount, currency = body["amount"], body["currency"]
            created = body["created"]
            if not (isinstance(status, str) and isinstance(amount, int)
                    and isinstance(currency, str) and isinstance(created, int)):
                raise ValueError("field types")
            received = body.get("amount_received", 0)
            if not isinstance(received, int):
                raise ValueError("amount_received")
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="payment_intent shape") from exc

        charge = body.get("latest_charge")  # expanded object, or an ID string / null if not expanded
        refunded = charge.get("amount_refunded") if isinstance(charge, dict) else None
        error = body.get("last_payment_error")
        failure_code = error.get("code") if isinstance(error, dict) else None

        return {
            "payment_id": payment_id,
            "status": status,
            "status_summary": STATUS_SUMMARY.get(status, status.replace("_", " ")),
            "amount_minor": amount,
            "currency": currency.upper(),
            "amount_received_minor": received,
            "amount_refunded_minor": refunded if isinstance(refunded, int) else None,
            "created": datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "failure_code": failure_code if isinstance(failure_code, str) else None,
            "livemode": False,
            "simulated": False,
            "source": "stripe_test",
        }

    def _get(self, payment_id: str) -> dict[str, Any]:
        url = f"{API_BASE}/payment_intents/{payment_id}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self._client.get(
                    url,
                    params={"expand[]": "latest_charge"},
                    headers={"Authorization": f"Bearer {self._cfg.secret_key}"},
                )
            except httpx.TimeoutException as exc:
                raise BackendError("UPSTREAM_TIMEOUT", GENERIC_UNAVAILABLE, detail="stripe timeout") from exc
            except httpx.HTTPError as exc:
                raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE,
                                   detail=f"stripe {type(exc).__name__}") from exc

            status = resp.status_code
            if status == 200:
                try:
                    body = resp.json()
                except ValueError as exc:
                    raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="non-JSON") from exc
                if not isinstance(body, dict):
                    raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="body not object")
                return body
            if status == 401:
                raise BackendError("AUTH_FAILED", "The payment service rejected our credentials.",
                                   detail="stripe HTTP 401")
            if status == 403:
                raise BackendError("PERMISSION_DENIED", "Our integration lacks permission to read payments.",
                                   detail="stripe HTTP 403")
            if status == 404:
                raise BackendError("PAYMENT_NOT_FOUND", f"No payment found with ID {payment_id}.")
            if status == 429:
                if attempt >= MAX_ATTEMPTS:
                    raise BackendError("RATE_LIMITED", "The payment service is busy. Please try again shortly.",
                                       detail="stripe throttled")
                log.warning("Stripe rate limited; retrying (attempt %d)", attempt)
                self._sleep(float(attempt))
                continue
            raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE, detail=f"stripe HTTP {status}")
        raise BackendError("RATE_LIMITED", "The payment service is busy. Please try again shortly.")
