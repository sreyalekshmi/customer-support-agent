"""Integration tests for intent recognition, routing, RAG, and MCP."""

import asyncio

import pytest

from app.agent.orchestrator import handle
from app.agent.routing import route
from app.intent.classifier import predict_intent
from app.schemas import SupportRequest


def req(msg: str) -> SupportRequest:
    return SupportRequest(request_id="t-1", session_id="s-1", user_message=msg)


def run(msg: str):
    return asyncio.run(handle(req(msg)))


# (message, expected destination, expected tool or None)
ROUTING_CASES = [
    ("Where is my order ORD-1001?", "mcp_tool", "get_order_status"),
    ("track ord-1002 please", "mcp_tool", "get_order_status"),
    ("Where is my order?", "direct", None),  # clarification: missing ID
    ("How long does shipping take?", "rag", None),
    ("What is your return policy?", "rag", None),
    ("How long do refunds take?", "rag", None),
    ("I want a refund for ORD-1003", "escalate", None),
    ("hello", "direct", None),
    ("I want to talk to a human", "escalate", None),
    ("purple elephants dance", "escalate", None),
    ("I was charged twice", "direct", None),  # payment lookup needs a payment ID
    ("Is payment pi_TEST123456 complete?", "mcp_tool", "get_payment_status"),
]


@pytest.mark.parametrize("msg,dest,tool", ROUTING_CASES)
def test_routing(msg, dest, tool):
    decision = route(req(msg), predict_intent(msg))
    assert decision.destination == dest
    assert decision.tool_name == tool


def test_missing_order_id_requests_clarification():
    d = route(req("Where is my order?"), predict_intent("Where is my order?"))
    assert d.requires_clarification


def test_missing_payment_id_requests_clarification():
    d = route(req("Did my payment go through?"), predict_intent("Did my payment go through?"))
    assert d.destination == "direct" and d.requires_clarification


def test_payment_intent_id_is_routed_to_stripe_mcp_tool():
    d = route(req("Is payment pi_TEST123456 complete?"),
              predict_intent("Is payment pi_TEST123456 complete?"))
    assert d.destination == "mcp_tool"
    assert d.tool_name == "get_payment_status"
    assert d.tool_arguments["payment_id"] == "pi_TEST123456"

def test_shopify_order_name_is_routed_to_mcp():
    decision = route(req("My order number is #D10"), predict_intent("Where is my order #D10?"))
    assert decision.destination == "mcp_tool"
    assert decision.tool_name == "get_order_status"
    assert decision.tool_arguments["order_id"] == "#D10"


def test_refund_requires_confirmation_and_is_not_executed():
    d = route(req("I want a refund for ORD-1003"), predict_intent("I want a refund for ORD-1003"))
    assert d.requires_confirmation and d.tool_name is None


# ---- end to end (spawns the MCP server subprocess for tool paths) ----------
def test_e2e_order_status_uses_mcp_tool():
    r = run("Where is my order ORD-1001?")
    assert r.tool_used == "get_order_status" and "shipped" in r.reply and not r.escalated
    assert r.responder == "mock"


def test_e2e_unknown_order():
    r = run("status of ORD-9999")
    assert "couldn't find" in r.reply and not r.escalated


def test_e2e_rag_answer_has_source():
    r = run("How long does shipping take?")
    assert r.sources == ["return_and_shipping_policy.md"] and not r.escalated


def test_e2e_rag_no_answer_escalates():
    r = run("What is the warranty on laptops? Policy please")
    assert r.escalated and r.sources == []


def test_e2e_refund_is_escalated():
    r = run("I want a refund for ORD-1003")
    assert r.escalated and r.tool_used is None


# ---- responder must cope with real (Shopify) data: missing carrier/tracking/ETA ----
def _tool_result(**data):
    from app.schemas import ToolResult
    base = {"order_id": "ORD-1001", "status": "shipped", "carrier": None, "tracking_number": None,
            "estimated_delivery": None, "simulated": False, "source": "shopify"}
    base.update(data)
    return ToolResult(success=True, tool_name="get_order_status", data=base)


def test_responder_handles_shipped_order_without_tracking():
    from app.agent.responder import from_tool_result
    text, escalated = from_tool_result(_tool_result())
    assert "None" not in text and "shipped" in text and not escalated
    assert "Simulated" not in text  # real data is not labelled as simulated


@pytest.mark.parametrize("status,word", [("partially_shipped", "partly shipped"), ("on_hold", "on hold"),
                                          ("restocked", "restocked")])
def test_responder_handles_new_statuses(status, word):
    from app.agent.responder import from_tool_result
    text, _ = from_tool_result(_tool_result(status=status))
    assert word in text and "None" not in text
