"""get_order_status tool logic.

Validates and normalises the ID, then delegates to the configured backend
(MOCK or SHOPIFY; see app/mcp_layer/services.py). Kept separate from server.py so
it is unit-testable without MCP.

Accepted inputs: ``ORD-1001`` (canonical, case-insensitive), ``#1001``, or a
Shopify order name such as ``#D10``. The result's ``order_id`` is canonicalized
to ``ORD-...`` so the backend interface remains stable.
"""

import re
from typing import Any

from app.mcp_layer import services
from app.mcp_layer.backends.base import OrderBackend
from app.mcp_layer.tools.common import error_result, new_request_id, run_backend

TOOL_NAME = "get_order_status"
_ORDER_ID = re.compile(r"^(?:ORD-([0-9]{4,8})|#([A-Z0-9][A-Z0-9-]*))$", re.IGNORECASE)


def normalize_order_id(raw: str) -> str | None:
    """Normalize ``ORD-1001``, ``#1001``, and Shopify names such as ``#D10``."""
    match = _ORDER_ID.match(raw.strip().upper())
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return f"ORD-{value}"


def lookup_order_status(
    order_id: str,
    request_id: str | None = None,
    backend: OrderBackend | None = None,
) -> dict[str, Any]:
    request_id = new_request_id(request_id)
    canonical = normalize_order_id(order_id)
    if canonical is None:
        return error_result(
            TOOL_NAME, request_id, "INVALID_ORDER_ID",
            "Order IDs look like ORD-1234 or a Shopify order name such as #D10.",
        )
    return run_backend(
        TOOL_NAME, request_id,
        lambda: (backend or services.get_order_backend()).get_order_status(canonical),
    )
