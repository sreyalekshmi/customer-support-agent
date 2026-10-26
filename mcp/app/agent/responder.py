"""Deterministic responder for tool results, retrieved context, and escalations."""

from app.schemas import RetrievalResult, RoutingDecision, ToolResult

_SIM = " (Simulated demo data.)"


def greeting() -> str:
    return "Hello! I can check an order's status or answer questions about shipping and returns."


def ask_for_order_id() -> str:
    return "Happy to check that. What is your order ID? It can look like ORD-1234 or #D10."


def ask_for_payment_id() -> str:
    return "Happy to check that. What is your Stripe PaymentIntent ID? It starts with pi_."


def from_tool_result(result: ToolResult) -> tuple[str, bool]:
    """Returns (reply, escalated)."""
    if result.success and result.data:
        d = result.data
        status = d["status"]
        eta = d.get("estimated_delivery")
        eta_text = f" Estimated delivery: {eta}." if eta else ""
        if status in ("shipped", "partially_shipped"):
            via = f" via {d['carrier']}" if d.get("carrier") else ""
            track = f" Tracking number: {d['tracking_number']}." if d.get("tracking_number") else ""
            what = "has shipped" if status == "shipped" else "has partly shipped"
            text = f"Order {d['order_id']} {what}{via}.{track}{eta_text}"
        elif status == "processing":
            text = f"Order {d['order_id']} is being processed and hasn't shipped yet.{eta_text}"
        elif status == "on_hold":
            text = f"Order {d['order_id']} is currently on hold."
        elif status == "delivered":
            text = f"Order {d['order_id']} was delivered."
        elif status == "cancelled":
            text = f"Order {d['order_id']} has been cancelled."
        else:
            text = f"Order {d['order_id']} status: {status.replace('_', ' ')}."
        return text + (_SIM if d.get("simulated") else ""), False

    code = result.error.code if result.error else "UNKNOWN"
    if code == "ORDER_NOT_FOUND":
        return "I couldn't find that order. Please double-check the ID.", False
    if code == "INVALID_ORDER_ID":
        return "That doesn't look like a valid order ID. IDs look like ORD-1234 or #D10.", False
    return ("I'm having trouble reaching the order system right now. "
            "I'll pass this to a human agent."), True


def from_retrieval(retrieval: RetrievalResult) -> tuple[str, list[str], bool]:
    """Returns (reply, sources, escalated). Quotes retrieved text; does not invent."""
    if not retrieval.chunks:
        return ("I don't have information about that in my knowledge base. "
                "I'll pass this to a human agent."), [], True
    best = retrieval.chunks[0]
    # Cite only the chunk actually quoted in the reply.
    return f"{best.text} (Source: {best.source}; sample policy text.)", [best.source], False


def escalation(decision: RoutingDecision) -> str:
    if decision.intent == "refund_request":
        return ("I can't process refunds automatically. A human support agent needs to review "
                "this request.")
    return "I can't resolve this automatically, so I'll pass it to a human support agent."
