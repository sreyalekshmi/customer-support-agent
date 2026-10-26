"""Test helpers: fake Shopify/Stripe HTTP servers built on httpx.MockTransport (no network)."""

import json
from typing import Any, Callable

import httpx

from app.config import ShopifyConfig, StripeConfig
from app.mcp_layer.backends.orders_shopify import ShopifyOrderBackend
from app.mcp_layer.backends.payments_stripe import StripePaymentBackend

SHOP_CFG = ShopifyConfig(
    store_domain="demo-store.myshopify.com",
    api_version="2026-07",
    access_token="shpat_TEST_TOKEN",
)
CC_CFG = ShopifyConfig(
    store_domain="demo-store.myshopify.com",
    api_version="2026-07",
    client_id="cid_test",
    client_secret="csecret_test",
)
STRIPE_CFG = StripeConfig(secret_key="sk_test_FAKEKEY")

NO_SLEEP = lambda seconds: None  # noqa: E731


def shopify_backend(handler: Callable[[httpx.Request], httpx.Response], cfg: ShopifyConfig = SHOP_CFG,
                    sleep: Callable[[float], None] = NO_SLEEP, clock: Callable[[], float] | None = None):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs = {"clock": clock} if clock else {}
    return ShopifyOrderBackend(cfg, client, sleep=sleep, **kwargs)


def stripe_backend(handler: Callable[[httpx.Request], httpx.Response], sleep: Callable[[float], None] = NO_SLEEP):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return StripePaymentBackend(STRIPE_CFG, client, sleep=sleep)


def order_node(name: str = "#1001", *, cancelled: str | None = None, financial: str | None = "PAID",
               fulfillment: str = "FULFILLED", fulfillments: list[dict[str, Any]] | None = None,
               **extra: Any) -> dict[str, Any]:
    if fulfillments is None:
        fulfillments = [{
            "displayStatus": "IN_TRANSIT",
            "estimatedDeliveryAt": "2026-10-25T00:00:00Z",
            "trackingInfo": [{"company": "BlueDart", "number": "TRK123", "url": "https://t.example/TRK123"}],
        }]
    node = {
        "name": name,
        "cancelledAt": cancelled,
        "displayFinancialStatus": financial,
        "displayFulfillmentStatus": fulfillment,
        "fulfillments": fulfillments,
    }
    node.update(extra)
    return node


def fake_shopify_store(nodes: list[dict[str, Any]], seen: list[httpx.Request] | None = None):
    """Handler that behaves like a store: returns nodes whose name matches the search term."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        term = body["variables"]["q"].removeprefix("name:").lstrip("#")
        matches = [n for n in nodes if n["name"].lstrip("#") == term]
        return httpx.Response(200, json={"data": {"orders": {"nodes": matches}}})

    return handler


def payment_intent(pid: str = "pi_TEST123456", **overrides: Any) -> dict[str, Any]:
    body = {
        "id": pid,
        "object": "payment_intent",
        "status": "succeeded",
        "amount": 89900,
        "amount_received": 89900,
        "currency": "inr",
        "created": 1760000000,
        "livemode": False,
        "last_payment_error": None,
        "latest_charge": {"id": "ch_1", "object": "charge", "amount_refunded": 0},
        # PII-ish fields Stripe returns that must never reach the tool result:
        "receipt_email": "customer@example.com",
        "customer": "cus_123",
        "shipping": {"name": "Jane", "address": {"line1": "1 Main St"}},
    }
    body.update(overrides)
    return body
