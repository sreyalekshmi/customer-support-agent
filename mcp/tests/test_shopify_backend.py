"""Shopify backend tests using MOCKED HTTP responses.

These prove our request building, response parsing and error handling. They do NOT prove
the request and response handling without contacting a real store.
"""

import json

import httpx
import pytest

from app.mcp_layer.backends.orders_shopify import ORDER_QUERY, to_shopify_order_name
from app.mcp_layer.tools.orders import lookup_order_status
from tests.helpers import (CC_CFG, SHOP_CFG, fake_shopify_store, order_node, shopify_backend)


def lookup(backend, order_id="ORD-1001"):
    return lookup_order_status(order_id, "req-1", backend=backend)


# ------------------------------------------------------------------ mapping
def test_order_id_maps_to_shopify_name():
    assert to_shopify_order_name("ORD-1001") == "#1001"
    assert to_shopify_order_name("ORD-1001", prefix="SH-", suffix="-A") == "SH-1001-A"


# ------------------------------------------------------------------ success
def test_shipped_order_with_tracking():
    r = lookup(shopify_backend(fake_shopify_store([order_node()])))
    assert r["success"] and r["error"] is None and r["request_id"] == "req-1"
    d = r["data"]
    assert d["order_id"] == "ORD-1001" and d["shopify_order_name"] == "#1001"
    assert d["status"] == "shipped"
    assert d["financial_status"] == "paid" and d["fulfillment_status"] == "fulfilled"
    assert (d["carrier"], d["tracking_number"]) == ("BlueDart", "TRK123")
    assert d["tracking_url"] == "https://t.example/TRK123"
    assert d["estimated_delivery"] == "2026-10-25"
    assert d["source"] == "shopify" and d["simulated"] is False


def test_hash_prefixed_input_is_accepted_and_canonicalised():
    r = lookup(shopify_backend(fake_shopify_store([order_node()])), "#1001")
    assert r["success"] and r["data"]["order_id"] == "ORD-1001"


@pytest.mark.parametrize("node,expected", [
    (order_node(fulfillment="UNFULFILLED", fulfillments=[]), "processing"),
    (order_node(fulfillment="ON_HOLD", fulfillments=[]), "on_hold"),
    (order_node(fulfillment="PARTIALLY_FULFILLED"), "partially_shipped"),
    (order_node(fulfillments=[{"displayStatus": "DELIVERED", "trackingInfo": []}]), "delivered"),
    (order_node(cancelled="2026-10-20T10:00:00Z", fulfillment="UNFULFILLED", fulfillments=[]), "cancelled"),
    (order_node(fulfillment="RESTOCKED", fulfillments=[]), "restocked"),
])
def test_status_mapping(node, expected):
    r = lookup(shopify_backend(fake_shopify_store([node])))
    assert r["success"] and r["data"]["status"] == expected


def test_unfulfilled_order_has_no_tracking_and_null_financial_ok():
    node = order_node(fulfillment="UNFULFILLED", financial=None, fulfillments=[])
    d = lookup(shopify_backend(fake_shopify_store([node])))["data"]
    assert d["carrier"] is None and d["tracking_number"] is None and d["financial_status"] is None


def test_voided_fulfillment_is_ignored_for_tracking():
    node = order_node(fulfillments=[
        {"displayStatus": "LABEL_VOIDED", "trackingInfo": [{"company": "X", "number": "OLD", "url": None}]},
        {"displayStatus": "IN_TRANSIT", "trackingInfo": [{"company": "Y", "number": "NEW", "url": None}]},
    ])
    assert lookup(shopify_backend(fake_shopify_store([node])))["data"]["tracking_number"] == "NEW"


# ------------------------------------------------------------------ request shape
def test_request_is_authenticated_graphql_to_the_configured_store():
    seen: list[httpx.Request] = []
    lookup(shopify_backend(fake_shopify_store([order_node()], seen)))
    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://demo-store.myshopify.com/admin/api/2026-07/graphql.json"
    assert req.headers["X-Shopify-Access-Token"] == "shpat_TEST_TOKEN"
    body = json.loads(req.content)
    assert body["variables"] == {"q": "name:#1001"}  # value travels as a variable, not spliced into GraphQL


def test_query_requests_no_customer_personal_data():
    q = ORDER_QUERY.lower()
    for forbidden in ("email", "phone", "address", "customer", "lineitems", "billing", "shipping",
                      "note", "transactions", "paymentgateway"):
        assert forbidden not in q


def test_extra_fields_in_response_never_leak():
    node = order_node(email="jane@example.com", customer={"firstName": "Jane"},
                      shippingAddress={"address1": "1 Main St"}, totalPriceSet={"amount": "10"})
    r = lookup(shopify_backend(fake_shopify_store([node])))
    text = json.dumps(r)
    assert "jane@example.com" not in text and "Main St" not in text and "Jane" not in text
    assert set(r["data"]) == {"order_id", "status", "financial_status", "fulfillment_status", "carrier",
                              "tracking_number", "tracking_url", "estimated_delivery",
                              "shopify_order_name", "simulated", "source"}


# ------------------------------------------------------------------ not found / invalid
def test_unknown_order_is_not_found_not_mock_data():
    r = lookup(shopify_backend(fake_shopify_store([])), "ORD-1001")  # ORD-1001 exists in MOCK data only
    assert not r["success"] and r["data"] is None and r["error"]["code"] == "ORDER_NOT_FOUND"


def test_fuzzy_search_hit_is_rejected_unless_name_matches_exactly():
    """Shopify search is fuzzy: a '#1001-A' hit must never be reported as '#1001'."""

    def handler(request):
        return httpx.Response(200, json={"data": {"orders": {"nodes": [order_node("#1001-A"), order_node("#10011")]}}})

    r = lookup(shopify_backend(handler))
    assert not r["success"] and r["error"]["code"] == "ORDER_NOT_FOUND"


def test_bare_number_search_fallback_is_used_when_hash_form_finds_nothing():
    searches = []

    def handler(request):
        q = json.loads(request.content)["variables"]["q"]
        searches.append(q)
        nodes = [order_node()] if q == "name:1001" else []
        return httpx.Response(200, json={"data": {"orders": {"nodes": nodes}}})

    assert lookup(shopify_backend(handler))["success"]
    assert searches == ["name:#1001", "name:1001"]


@pytest.mark.parametrize("bad", ["", "1001", "ORD-1", "ORD-123456789", "ORD-10 01", "1; DROP TABLE", "ORD-1001\nname:*"])
def test_invalid_input_never_reaches_shopify(bad):
    def handler(request):
        raise AssertionError("HTTP call made for invalid input")

    r = lookup(shopify_backend(handler), bad)
    assert not r["success"] and r["error"]["code"] == "INVALID_ORDER_ID"


# ------------------------------------------------------------------ failures
def _resp(status, **kw):
    return lambda request: httpx.Response(status, **kw)


def _raises(exc):
    def handler(request):
        raise exc(" boom", request=request)
    return handler


FAILURES = [
    ("auth_401", _resp(401, json={"errors": "[API] Invalid API key or access token"}), "AUTH_FAILED"),
    ("forbidden_403", _resp(403, json={"errors": "forbidden"}), "PERMISSION_DENIED"),
    ("graphql_access_denied", _resp(200, json={"errors": [{"message": "Access denied for orders field",
                                                           "extensions": {"code": "ACCESS_DENIED"}}]}), "PERMISSION_DENIED"),
    ("graphql_validation_error", _resp(200, json={"errors": [{"message": "Field 'x' doesn't exist"}]}), "UPSTREAM_ERROR"),
    ("server_500", _resp(500, text="oops"), "UPSTREAM_ERROR"),
    ("bad_gateway_502", _resp(502, text="bad gateway"), "UPSTREAM_ERROR"),
    ("store_404", _resp(404, text="not found"), "UPSTREAM_ERROR"),
    ("timeout", _raises(httpx.ReadTimeout), "UPSTREAM_TIMEOUT"),
    ("connect_error", _raises(httpx.ConnectError), "UPSTREAM_ERROR"),
    ("rate_limited_forever_429", _resp(429, headers={"Retry-After": "2"}, text="slow down"), "RATE_LIMITED"),
    ("throttled_forever_graphql", _resp(200, json={"errors": [{"message": "Throttled", "extensions": {"code": "THROTTLED"}}]}),
     "RATE_LIMITED"),
    ("malformed_not_json", _resp(200, text="<html>maintenance</html>"), "MALFORMED_RESPONSE"),
    ("malformed_json_array", _resp(200, json=[1, 2]), "MALFORMED_RESPONSE"),
    ("malformed_no_data", _resp(200, json={}), "MALFORMED_RESPONSE"),
    ("malformed_nodes_not_list", _resp(200, json={"data": {"orders": {"nodes": "x"}}}), "MALFORMED_RESPONSE"),
    ("malformed_node_without_name", _resp(200, json={"data": {"orders": {"nodes": [{"cancelledAt": None}]}}}),
     "MALFORMED_RESPONSE"),
    ("malformed_fulfillments_not_list",
     _resp(200, json={"data": {"orders": {"nodes": [order_node(fulfillments="oops")]}}}), "MALFORMED_RESPONSE"),
    ("malformed_tracking_not_list",
     _resp(200, json={"data": {"orders": {"nodes": [order_node(fulfillments=[{"displayStatus": "IN_TRANSIT",
                                                                                "trackingInfo": "x"}])]}}}),
     "MALFORMED_RESPONSE"),
]


@pytest.mark.parametrize("handler,code", [(h, c) for _, h, c in FAILURES], ids=[n for n, _, _ in FAILURES])
def test_failure_modes_return_error_and_never_fall_back_to_mock(handler, code):
    # ORD-1001 exists in the MOCK fixtures. A silent fallback would make this succeed.
    r = lookup(shopify_backend(handler), "ORD-1001")
    assert r["success"] is False
    assert r["data"] is None
    assert r["error"]["code"] == code
    text = json.dumps(r)
    assert '"simulated": true' not in text and '"source": "mock"' not in text and "MOCK-TRK" not in text


def test_error_messages_do_not_leak_secrets_or_internals():
    for _, handler, _ in FAILURES:
        text = json.dumps(lookup(shopify_backend(handler)))
        assert "shpat_TEST_TOKEN" not in text and "myshopify.com" not in text


# ------------------------------------------------------------------ rate limit retry
def test_http_429_is_retried_honouring_retry_after_then_succeeds():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"data": {"orders": {"nodes": [order_node()]}}})

    r = lookup(shopify_backend(handler, sleep=sleeps.append))
    assert r["success"] and sleeps == [2.0] and len(calls) == 2


def test_graphql_throttled_error_is_retried_then_succeeds():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(200, json={"errors": [{"message": "Throttled", "extensions": {"code": "THROTTLED"}}]})
        return httpx.Response(200, json={"data": {"orders": {"nodes": [order_node()]}}})

    assert lookup(shopify_backend(handler, sleep=sleeps.append))["success"]
    assert sleeps == [1.0, 2.0]


def test_retry_wait_is_capped():
    sleeps = []
    handler = _resp(429, headers={"Retry-After": "9999"})
    lookup(shopify_backend(handler, sleep=sleeps.append))
    assert max(sleeps) <= 5.0


def test_persistent_throttle_stops_after_three_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    r = lookup(shopify_backend(handler))
    assert r["error"]["code"] == "RATE_LIMITED" and len(calls) == 3


# ------------------------------------------------------------------ client credentials auth
def _cc_handler(order_nodes, token_status=200, token_body=None, graphql_401_first=False):
    log = {"token_calls": 0, "graphql_calls": 0, "tokens_used": []}

    def handler(request):
        if request.url.path == "/admin/oauth/access_token":
            log["token_calls"] += 1
            assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
            form = dict(x.split("=") for x in request.content.decode().split("&"))
            assert form == {"grant_type": "client_credentials", "client_id": "cid_test",
                            "client_secret": "csecret_test"}
            body = token_body if token_body is not None else {
                "access_token": f"tok_{log['token_calls']}", "scope": "read_orders", "expires_in": 86399}
            return httpx.Response(token_status, json=body)
        log["graphql_calls"] += 1
        log["tokens_used"].append(request.headers["X-Shopify-Access-Token"])
        if graphql_401_first and log["graphql_calls"] == 1:
            return httpx.Response(401, json={"errors": "expired"})
        return fake_shopify_store(order_nodes)(request)

    return handler, log


def test_client_credentials_token_is_fetched_once_and_cached():
    handler, log = _cc_handler([order_node()])
    backend = shopify_backend(handler, cfg=CC_CFG)
    assert lookup(backend)["success"] and lookup(backend)["success"]
    assert log["token_calls"] == 1 and log["tokens_used"] == ["tok_1", "tok_1"]


def test_client_credentials_token_is_refreshed_after_expiry():
    now = [0.0]
    handler, log = _cc_handler([order_node()])
    backend = shopify_backend(handler, cfg=CC_CFG, clock=lambda: now[0])
    lookup(backend)
    now[0] = 86399 - 299  # inside the 5 minute early-refresh window
    lookup(backend)
    assert log["token_calls"] == 2 and log["tokens_used"] == ["tok_1", "tok_2"]


def test_revoked_client_credentials_token_is_refreshed_once_on_401():
    handler, log = _cc_handler([order_node()], graphql_401_first=True)
    r = lookup(shopify_backend(handler, cfg=CC_CFG))
    assert r["success"] and log["token_calls"] == 2


def test_static_token_401_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    assert lookup(shopify_backend(handler))["error"]["code"] == "AUTH_FAILED" and len(calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403])
def test_bad_client_credentials_are_auth_failed(status):
    handler, _ = _cc_handler([order_node()], token_status=status, token_body={"error": "invalid_client"})
    r = lookup(shopify_backend(handler, cfg=CC_CFG))
    assert not r["success"] and r["error"]["code"] == "AUTH_FAILED"


@pytest.mark.parametrize("body", [{}, {"access_token": ""}, {"access_token": 123}, ["x"]])
def test_malformed_token_response_is_reported(body):
    handler, _ = _cc_handler([order_node()], token_body=body)
    r = lookup(shopify_backend(handler, cfg=CC_CFG))
    assert not r["success"] and r["error"]["code"] == "MALFORMED_RESPONSE"


def test_token_endpoint_outage_is_upstream_error():
    handler, _ = _cc_handler([order_node()], token_status=503, token_body={})
    assert lookup(shopify_backend(handler, cfg=CC_CFG))["error"]["code"] == "UPSTREAM_ERROR"
