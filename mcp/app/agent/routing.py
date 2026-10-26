"""Rule-based router: (request, intent) -> RoutingDecision.

This is where the *decision* happens. MCP only carries out the chosen tool call.
Deterministic rules were chosen over LLM tool-selection for the deadline: they are
testable, fast and cannot hallucinate a tool. An LLM selector can replace this later.
"""

import re

from app.schemas import IntentResult, RoutingDecision, SupportRequest

MIN_CONFIDENCE = 0.5
# Accept the app's canonical IDs and Shopify order names such as "#D10".
# The latter are common when a store uses a custom order-number prefix.
_ORDER_ID = re.compile(r"(?:\bORD-\d{4,8}\b|#[A-Z0-9][A-Z0-9-]*)", re.IGNORECASE)
_PAYMENT_ID = re.compile(r"\bpi_[A-Za-z0-9]{6,64}\b")


def route(request: SupportRequest, intent: IntentResult) -> RoutingDecision:
    base = dict(intent=intent.intent, confidence=intent.confidence)

    if intent.confidence < MIN_CONFIDENCE or intent.intent in ("unknown", "human_escalation"):
        return RoutingDecision(**base, destination="escalate",
                               reason="Low confidence, unknown, or customer asked for a human.")

    if intent.intent == "order_status":
        match = _ORDER_ID.search(request.user_message)
        if not match:
            return RoutingDecision(**base, destination="direct", requires_clarification=True,
                                   reason="Order status needs an order ID.")
        return RoutingDecision(**base, destination="mcp_tool", tool_name="get_order_status",
                               tool_arguments={"order_id": match.group(0).upper(),
                                               "request_id": request.request_id},
                               reason="Order status maps to get_order_status.")

    if intent.intent == "payment_status":
        match = _PAYMENT_ID.search(request.user_message)
        if not match:
            return RoutingDecision(**base, destination="direct", requires_clarification=True,
                                   reason="Payment status needs a Stripe PaymentIntent ID.")
        return RoutingDecision(**base, destination="mcp_tool", tool_name="get_payment_status",
                               tool_arguments={"payment_id": match.group(0),
                                               "request_id": request.request_id},
                               reason="Payment status maps to get_payment_status.")

    if intent.intent in ("policy_faq", "product_info"):
        return RoutingDecision(**base, destination="rag", reason="Knowledge question.")

    if intent.intent == "greeting_general":
        return RoutingDecision(**base, destination="direct", reason="Greeting.")

    if intent.intent == "refund_request":
        return RoutingDecision(**base, destination="escalate", requires_confirmation=True,
                               reason="Refunds are financial mutations; not automated in the prototype.")

    # complaint_or_ticket: tool not built yet
    return RoutingDecision(**base, destination="escalate",
                           reason=f"No tool implemented yet for '{intent.intent}'.")
