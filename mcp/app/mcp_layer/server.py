"""MCP server exposing customer-support tools.

Run directly:  python -m app.mcp_layer.server
Transport: stdio (the client launches this file as a subprocess).

Backends are chosen by configuration (ORDER_BACKEND=mock|shopify,
PAYMENT_BACKEND=mock|stripe; see .env.example). Every result says where it came from
in data["source"].

IMPORTANT for stdio: never print() to stdout in this process. stdout carries the
JSON-RPC messages, so stray prints corrupt the protocol. Logs go to stderr.
"""

import logging
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from app.mcp_layer.tools.orders import lookup_order_status
from app.mcp_layer.tools.payments import lookup_payment_status

mcp = MCPServer(
    "support-tools",
    instructions=(
        "Customer support tools. Check data['source'] and data['simulated'] in each result: "
        "'mock' data is simulated and must not be presented as real."
    ),
)


@mcp.tool()
def get_order_status(order_id: str, request_id: str | None = None) -> dict[str, Any]:
    """Look up the current status of a customer's order.

    Use when the customer asks where their order is, whether it has shipped, or
    when it will arrive. Takes an order ID such as ORD-1001 (or #1001). Returns status
    (processing, shipped, delivered, cancelled, ...), financial status, fulfillment
    status, carrier, tracking number/URL and estimated delivery when available.
    Read-only. On failure returns success=false with an error code, e.g. ORDER_NOT_FOUND.
    """
    return lookup_order_status(order_id, request_id)


@mcp.tool()
def get_payment_status(payment_id: str, request_id: str | None = None) -> dict[str, Any]:
    """Look up the status of a payment by its Stripe PaymentIntent ID (starts with pi_).

    This is NOT an order ID. Use when the customer asks whether a payment went through
    and a payment ID is known. Returns status, a plain-language summary, amount (in minor
    units such as paise or cents), currency and amount refunded when available.
    Read-only: this tool cannot create, capture or refund payments.
    """
    return lookup_payment_status(payment_id, request_id)


def main() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
