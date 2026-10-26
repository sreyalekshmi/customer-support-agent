"""Stripe backend tests using MOCKED HTTP responses (no network, no real Stripe account)."""

import json

import httpx
import pytest

from app.mcp_layer.tools.payments import lookup_payment_status
from tests.helpers import payment_intent, stripe_backend

PID = "pi_TEST123456"
MOCK_PID = "pi_MOCK1001"  # exists in MOCK fixtures; used to detect silent fallback


def lookup(backend, payment_id=PID):
    return lookup_payment_status(payment_id, "req-9", backend=backend)


def ok(body, status=200):
    return lambda request: httpx.Response(status, json=body)


def test_succeeded_payment():
    r = lookup(stripe_backend(ok(payment_intent(latest_charge={"id": "ch_1", "amount_refunded": 25000}))))
    assert r["success"] and r["request_id"] == "req-9"
    d = r["data"]
    assert d == {
        "payment_id": PID, "status": "succeeded", "status_summary": "paid",
        "amount_minor": 89900, "currency": "INR", "amount_received_minor": 89900,
        "amount_refunded_minor": 25000, "created": "2025-10-09T08:53:20Z", "failure_code": None,
        "livemode": False, "simulated": False, "source": "stripe_test",
    }


def test_failed_payment_exposes_only_failure_code():
    body = payment_intent(status="requires_payment_method", amount_received=0, latest_charge=None,
                          last_payment_error={"code": "card_declined", "decline_code": "insufficient_funds",
                                              "message": "Your card was declined.",
                                              "payment_method": {"card": {"last4": "4242"}}})
    d = lookup(stripe_backend(ok(body)))["data"]
    assert d["status"] == "requires_payment_method" and d["status_summary"] == "payment not completed"
    assert d["failure_code"] == "card_declined" and d["amount_refunded_minor"] is None
    assert "4242" not in json.dumps(d)


def test_unexpanded_charge_means_refund_amount_unknown_not_zero():
    assert lookup(stripe_backend(ok(payment_intent(latest_charge="ch_1"))))["data"]["amount_refunded_minor"] is None


def test_unknown_status_is_passed_through_readably():
    d = lookup(stripe_backend(ok(payment_intent(status="requires_something_new"))))["data"]
    assert d["status_summary"] == "requires something new"


def test_customer_personal_data_never_leaks():
    text = json.dumps(lookup(stripe_backend(ok(payment_intent()))))
    for leaked in ("customer@example.com", "cus_123", "Main St", "Jane", "receipt_email", "shipping"):
        assert leaked not in text


def test_request_is_get_with_bearer_key_and_expand():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payment_intent())

    lookup(stripe_backend(handler))
    req = seen[0]
    assert req.method == "GET"
    assert req.url.host == "api.stripe.com" and req.url.path == f"/v1/payment_intents/{PID}"
    assert req.headers["Authorization"] == "Bearer sk_test_FAKEKEY"
    assert req.url.params["expand[]"] == "latest_charge"


@pytest.mark.parametrize("bad", ["", "ORD-1001", "1001", "pi_", "pi_short", "PI_TEST123456", "pi_x/../y", "pi_abc def123"])
def test_invalid_ids_never_reach_stripe(bad):
    def handler(request):
        raise AssertionError("HTTP call made for invalid input")

    r = lookup(stripe_backend(handler), bad)
    assert not r["success"] and r["error"]["code"] == "INVALID_PAYMENT_ID"


def test_order_id_is_not_accepted_as_payment_id():
    r = lookup_payment_status("ORD-1001")  # default (mock) backend
    assert r["error"]["code"] == "INVALID_PAYMENT_ID"


def _resp(status, **kw):
    return lambda request: httpx.Response(status, **kw)


def _raises(exc):
    def handler(request):
        raise exc("boom", request=request)
    return handler


FAILURES = [
    ("missing_404", _resp(404, json={"error": {"code": "resource_missing", "type": "invalid_request_error"}}),
     "PAYMENT_NOT_FOUND"),
    ("bad_key_401", _resp(401, json={"error": {"type": "invalid_request_error", "message": "Invalid API Key"}}),
     "AUTH_FAILED"),
    ("restricted_key_403", _resp(403, json={"error": {"type": "invalid_request_error"}}), "PERMISSION_DENIED"),
    ("server_500", _resp(500, text="oops"), "UPSTREAM_ERROR"),
    ("timeout", _raises(httpx.ReadTimeout), "UPSTREAM_TIMEOUT"),
    ("connect_error", _raises(httpx.ConnectError), "UPSTREAM_ERROR"),
    ("rate_limited_429", _resp(429, json={"error": {"type": "rate_limit_error"}}), "RATE_LIMITED"),
    ("not_json", _resp(200, text="<html>"), "MALFORMED_RESPONSE"),
    ("json_array", _resp(200, json=[1]), "MALFORMED_RESPONSE"),
    ("wrong_object", ok(payment_intent(MOCK_PID, object="charge")), "MALFORMED_RESPONSE"),
    ("id_mismatch", ok(payment_intent("pi_OTHER123456")), "MALFORMED_RESPONSE"),
    ("missing_status", ok({k: v for k, v in payment_intent(MOCK_PID).items() if k != "status"}), "MALFORMED_RESPONSE"),
    ("amount_wrong_type", ok(payment_intent(MOCK_PID, amount="899.00")), "MALFORMED_RESPONSE"),
    ("created_wrong_type", ok(payment_intent(MOCK_PID, created="yesterday")), "MALFORMED_RESPONSE"),
    ("received_wrong_type", ok(payment_intent(MOCK_PID, amount_received="x")), "MALFORMED_RESPONSE"),
    ("livemode_true_refused", ok(payment_intent(MOCK_PID, livemode=True)), "CONFIG_ERROR"),
    ("livemode_missing_refused", ok({k: v for k, v in payment_intent(MOCK_PID).items() if k != "livemode"}), "CONFIG_ERROR"),
]


@pytest.mark.parametrize("handler,code", [(h, c) for _, h, c in FAILURES], ids=[n for n, _, _ in FAILURES])
def test_failures_return_error_and_never_fall_back_to_mock(handler, code):
    # pi_MOCK1001 exists in MOCK data; a silent fallback would make this succeed.
    r = lookup(stripe_backend(handler), MOCK_PID)
    assert r["success"] is False and r["data"] is None and r["error"]["code"] == code
    text = json.dumps(r)
    assert '"simulated": true' not in text and '"source": "mock"' not in text and "89900" not in text
    assert "sk_test_FAKEKEY" not in json.dumps(r)


def test_429_is_retried_then_succeeds():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json=payment_intent())

    assert lookup(stripe_backend(handler, sleep=sleeps.append))["success"] and sleeps == [1.0]


def test_persistent_429_stops_after_three_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    assert lookup(stripe_backend(handler))["error"]["code"] == "RATE_LIMITED" and len(calls) == 3


def test_backend_only_ever_issues_get_requests():
    methods = set()

    def handler(request):
        methods.add(request.method)
        return httpx.Response(200, json=payment_intent())

    lookup(stripe_backend(handler))
    assert methods == {"GET"}  # no refund / capture / cancel capability
