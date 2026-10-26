"""Shopify Admin GraphQL backend: read-only order lookup (scope needed: read_orders).

Identifier mapping
------------------
Customers/agents use canonical IDs like ``ORD-1001``. Shopify has its own identifiers:
a global ID (gid://shopify/Order/123...) and a human-visible *order name* ("#1001").
We map ``ORD-<n>`` -> order name ``<prefix><n><suffix>`` (default ``#1001``) and look the
order up by name. The returned name is compared for an EXACT match, because Shopify's
search is fuzzy and a near-miss must never be reported as someone else's order.

Privacy
-------
The GraphQL query requests no customer, email, address, line-item or payment fields, and
the result is built from a whitelist. Extra fields in a response are ignored.

Failure policy
--------------
Every failure raises BackendError with a stable code. Nothing here returns mock data.
"""

import logging
import threading
import time
from typing import Any, Callable

import httpx

from app.config import ShopifyConfig
from app.mcp_layer.backends.base import GENERIC_UNAVAILABLE, BackendError

log = logging.getLogger("mcp_layer.shopify")

ORDER_QUERY = """
query OrderByName($q: String!) {
  orders(first: 5, query: $q) {
    nodes {
      name
      cancelledAt
      displayFinancialStatus
      displayFulfillmentStatus
      fulfillments(first: 5) {
        displayStatus
        estimatedDeliveryAt
        trackingInfo(first: 5) { company number url }
      }
    }
  }
}
"""

MAX_ATTEMPTS = 3  # initial try + 2 retries on throttling
MAX_RETRY_WAIT = 5.0
_INACTIVE_FULFILLMENT = {"CANCELED", "LABEL_VOIDED", "FAILURE"}
_DELIVERED = {"DELIVERED", "PICKED_UP"}


def to_shopify_order_name(order_id: str, prefix: str = "#", suffix: str = "") -> str:
    """Convert canonical ``ORD-1001`` or ``ORD-D10`` to a Shopify order name."""
    return f"{prefix}{order_id.removeprefix('ORD-')}{suffix}"


class _TokenProvider:
    """Static Admin API token, or a cached 24h client-credentials token (Dev Dashboard apps)."""

    def __init__(self, cfg: ShopifyConfig, client: httpx.Client, clock: Callable[[], float]):
        self._cfg, self._client, self._clock = cfg, client, clock
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        if self._cfg.access_token:
            return self._cfg.access_token
        with self._lock:
            if self._token and self._clock() < self._expires_at:
                return self._token
            self._token, self._expires_at = self._fetch()
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token = None

    def _fetch(self) -> tuple[str, float]:
        url = f"https://{self._cfg.store_domain}/admin/oauth/access_token"
        try:
            resp = self._client.post(url, data={
                "grant_type": "client_credentials",
                "client_id": self._cfg.client_id,
                "client_secret": self._cfg.client_secret,
            })
        except httpx.TimeoutException as exc:
            raise BackendError("UPSTREAM_TIMEOUT", GENERIC_UNAVAILABLE, detail="token timeout") from exc
        except httpx.HTTPError as exc:
            raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE,
                               detail=f"token {type(exc).__name__}") from exc

        if resp.status_code in (400, 401, 403):
            raise BackendError("AUTH_FAILED", "The order service rejected our credentials.",
                               detail=f"token endpoint HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE,
                               detail=f"token endpoint HTTP {resp.status_code}")
        try:
            body = resp.json()
            token = body["access_token"]
            expires_in = float(body.get("expires_in", 3600))
        except (ValueError, KeyError, TypeError) as exc:
            raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE,
                               detail="token response") from exc
        if not isinstance(token, str) or not token:
            raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="empty token")
        # Refresh 5 minutes early so a token never expires mid-request.
        return token, self._clock() + max(expires_in - 300, 0)


class ShopifyOrderBackend:
    source = "shopify"

    def __init__(
        self,
        cfg: ShopifyConfig,
        client: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._cfg = cfg
        # follow_redirects=False (httpx default): never forward the token to another host.
        self._client = client or httpx.Client(timeout=cfg.timeout_seconds)
        self._sleep = sleep
        self._tokens = _TokenProvider(cfg, self._client, clock)
        self._url = f"https://{cfg.store_domain}/admin/api/{cfg.api_version}/graphql.json"

    # ------------------------------------------------------------------ public
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        name = to_shopify_order_name(order_id, self._cfg.order_name_prefix, self._cfg.order_name_suffix)
        digits = order_id.removeprefix("ORD-")
        # The exact search syntax for names is not pinned down in Shopify's docs, so try the
        # '#'-prefixed form first and the bare number second. Safe either way: results are
        # only accepted on an exact name match.
        for search in (f"name:{name}", f"name:{digits}"):
            node = self._find_exact(search, name)
            if node is not None:
                return self._to_result(order_id, name, node)
        raise BackendError("ORDER_NOT_FOUND", f"No order found with ID {order_id}.")

    # ----------------------------------------------------------------- private
    def _find_exact(self, search: str, expected_name: str) -> dict[str, Any] | None:
        payload = self._graphql(ORDER_QUERY, {"q": search})
        nodes = _dig(payload, "data", "orders", "nodes")
        if not isinstance(nodes, list):
            raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="orders.nodes missing")
        for node in nodes:
            if not isinstance(node, dict) or not isinstance(node.get("name"), str):
                raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="order node shape")
            if node["name"] == expected_name:
                return node
        return None

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        refreshed_token = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            token = self._tokens.get()
            try:
                resp = self._client.post(
                    self._url,
                    json={"query": query, "variables": variables},
                    headers={"X-Shopify-Access-Token": token, "Accept": "application/json"},
                )
            except httpx.TimeoutException as exc:
                raise BackendError("UPSTREAM_TIMEOUT", GENERIC_UNAVAILABLE, detail="graphql timeout") from exc
            except httpx.HTTPError as exc:
                raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE,
                                   detail=f"graphql {type(exc).__name__}") from exc

            status = resp.status_code
            if status == 401:
                # A client-credentials token may have been revoked: refresh once, then give up.
                if self._cfg.uses_client_credentials and not refreshed_token:
                    refreshed_token = True
                    self._tokens.invalidate()
                    continue
                raise BackendError("AUTH_FAILED", "The order service rejected our credentials.",
                                   detail="graphql HTTP 401")
            if status == 403:
                raise BackendError("PERMISSION_DENIED", "Our integration lacks permission to read orders.",
                                   detail="graphql HTTP 403")
            if status == 429:
                if self._wait_or_raise(attempt, resp.headers.get("Retry-After")):
                    continue
            if status >= 400:
                raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE, detail=f"graphql HTTP {status}")

            try:
                body = resp.json()
            except ValueError as exc:
                raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="non-JSON body") from exc
            if not isinstance(body, dict):
                raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="body not an object")

            errors = body.get("errors")
            if errors:
                kind = _classify_graphql_errors(errors)
                if kind == "THROTTLED":
                    if self._wait_or_raise(attempt, None):
                        continue
                if kind == "ACCESS_DENIED":
                    raise BackendError("PERMISSION_DENIED", "Our integration lacks permission to read orders.",
                                       detail="graphql ACCESS_DENIED")
                raise BackendError("UPSTREAM_ERROR", GENERIC_UNAVAILABLE, detail="graphql errors")
            return body

        # Unreachable: _wait_or_raise raises on the final attempt.
        raise BackendError("RATE_LIMITED", "The order service is busy. Please try again shortly.")

    def _wait_or_raise(self, attempt: int, retry_after: str | None) -> bool:
        """Back off and return True to retry, or raise RATE_LIMITED when out of attempts."""
        if attempt >= MAX_ATTEMPTS:
            raise BackendError("RATE_LIMITED", "The order service is busy. Please try again shortly.",
                               detail=f"throttled after {attempt} attempts")
        try:
            wait = float(retry_after) if retry_after else 1.0 * attempt  # Shopify suggests ~1s backoff
        except ValueError:
            wait = 1.0 * attempt
        log.warning("Shopify throttled; retrying in %.1fs (attempt %d)", min(wait, MAX_RETRY_WAIT), attempt)
        self._sleep(min(wait, MAX_RETRY_WAIT))
        return True

    def _to_result(self, order_id: str, name: str, node: dict[str, Any]) -> dict[str, Any]:
        fulfillment_status = _upper_or_none(node.get("displayFulfillmentStatus"))
        financial_status = _upper_or_none(node.get("displayFinancialStatus"))
        cancelled = node.get("cancelledAt") is not None

        raw_fulfillments = node.get("fulfillments")
        if raw_fulfillments is None:
            raw_fulfillments = []
        if not isinstance(raw_fulfillments, list) or not all(isinstance(f, dict) for f in raw_fulfillments):
            raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="fulfillments shape")
        active = [f for f in raw_fulfillments if f.get("displayStatus") not in _INACTIVE_FULFILLMENT]

        company = number = url = estimated = None
        for f in active:
            tracking = f.get("trackingInfo") or []
            if not isinstance(tracking, list) or not all(isinstance(t, dict) for t in tracking):
                raise BackendError("MALFORMED_RESPONSE", GENERIC_UNAVAILABLE, detail="trackingInfo shape")
            if estimated is None and isinstance(f.get("estimatedDeliveryAt"), str):
                estimated = f["estimatedDeliveryAt"][:10]  # YYYY-MM-DD
            if number is None and tracking:
                first = tracking[0]
                company, number, url = first.get("company"), first.get("number"), first.get("url")

        return {
            "order_id": order_id,
            "status": _unified_status(cancelled, fulfillment_status, active),
            "financial_status": financial_status.lower() if financial_status else None,
            "fulfillment_status": fulfillment_status.lower() if fulfillment_status else None,
            "carrier": company,
            "tracking_number": number,
            "tracking_url": url,
            "estimated_delivery": estimated,
            "shopify_order_name": name,
            "simulated": False,
            "source": "shopify",
        }


# --------------------------------------------------------------------- helpers
def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _upper_or_none(value: Any) -> str | None:
    return value.upper() if isinstance(value, str) and value else None


def _classify_graphql_errors(errors: Any) -> str:
    """Return 'THROTTLED', 'ACCESS_DENIED' or 'OTHER' from a GraphQL `errors` array."""
    if not isinstance(errors, list):
        return "OTHER"
    for err in errors:
        if not isinstance(err, dict):
            continue
        code = _dig(err, "extensions", "code")
        message = err.get("message")
        if code == "THROTTLED" or (isinstance(message, str) and "throttled" in message.lower()):
            return "THROTTLED"
        if code == "ACCESS_DENIED":
            return "ACCESS_DENIED"
    return "OTHER"


def _unified_status(cancelled: bool, fulfillment_status: str | None, active: list[dict[str, Any]]) -> str:
    """Collapse Shopify's statuses into the vocabulary the agent already understands.

    cancelled | delivered | shipped | partially_shipped | processing | on_hold,
    or the lower-cased Shopify fulfillment status if it is none of those.
    """
    if cancelled:
        return "cancelled"
    if active and all(f.get("displayStatus") in _DELIVERED for f in active):
        return "delivered"
    if fulfillment_status == "FULFILLED":
        return "shipped"
    if fulfillment_status in ("PARTIALLY_FULFILLED", "IN_PROGRESS"):
        return "partially_shipped"
    if fulfillment_status == "ON_HOLD":
        return "on_hold"
    if fulfillment_status in ("UNFULFILLED", "SCHEDULED", "PENDING_FULFILLMENT", "OPEN", None):
        return "processing"
    return fulfillment_status.lower()
