"""Tests for the MCP order tool: logic-level and through a real MCP client/server."""

import asyncio

from app.mcp_layer.client import call_tool, list_tool_specs
from app.mcp_layer.tools.orders import lookup_order_status


# ---- logic level (no MCP) -------------------------------------------------
def test_valid_order():
    r = lookup_order_status("ORD-1001")
    assert r["success"] and r["data"]["status"] == "shipped"
    assert r["data"]["simulated"] is True


def test_normalizes_case_and_whitespace():
    assert lookup_order_status("  ord-1002 ")["data"]["order_id"] == "ORD-1002"


def test_accepts_shopify_order_name():
    result = lookup_order_status("#D10")
    assert not result["success"]
    assert result["error"]["code"] == "ORDER_NOT_FOUND"


def test_unknown_order():
    r = lookup_order_status("ORD-9999")
    assert not r["success"] and r["error"]["code"] == "ORDER_NOT_FOUND"


def test_malformed_id():
    r = lookup_order_status("1; DROP TABLE")
    assert not r["success"] and r["error"]["code"] == "INVALID_ORDER_ID"


def test_request_id_is_echoed():
    assert lookup_order_status("ORD-1001", "req-42")["request_id"] == "req-42"


def test_no_sensitive_fields_leak():
    data = lookup_order_status("ORD-1001")["data"]
    assert not {"items", "total", "email", "address"} & set(data)


# ---- through the real MCP protocol (spawns the server subprocess) ----------
def test_discovery_lists_get_order_status():
    specs = asyncio.run(list_tool_specs())
    names = [s["name"] for s in specs]
    assert "get_order_status" in names
    spec = next(s for s in specs if s["name"] == "get_order_status")
    assert spec["description"] and "order_id" in spec["input_schema"]["required"]


def test_invocation_over_mcp():
    r = asyncio.run(call_tool("get_order_status", {"order_id": "ORD-1003"}))
    assert r["success"] and r["data"]["status"] == "delivered"


def test_schema_rejection_is_normalized():
    r = asyncio.run(call_tool("get_order_status", {"order_id": 1001}))
    assert not r["success"] and r["error"]["code"] == "TOOL_PROTOCOL_ERROR"
