from app import ui


def test_empty_message_does_not_call_agent():
    message, history, status, session_id = ui.respond("", [], None)
    assert message == ""
    assert history == []
    assert status == "Enter a question to begin."
    assert session_id


def test_respond_exposes_mcp_result(monkeypatch):
    async def fake_handle(request):
        return type(
            "Response",
            (),
            {
                "reply": "Order ORD-1001 shipped.",
                "destination": "mcp_tool",
                "tool_used": "get_order_status",
                "sources": [],
            },
        )()

    monkeypatch.setattr(ui, "handle", fake_handle)
    _, history, status, session_id = ui.respond("Where is my order?", [], "session-1")
    assert history[-1]["content"] == "Order ORD-1001 shipped."
    assert status == "mcp_tool · get_order_status"
    assert session_id == "session-1"


def test_respond_exposes_rag_source(monkeypatch):
    async def fake_handle(request):
        return type(
            "Response",
            (),
            {
                "reply": "Shipping takes 3 to 5 business days.",
                "destination": "rag",
                "tool_used": None,
                "sources": ["return_and_shipping_policy.md"],
            },
        )()

    monkeypatch.setattr(ui, "handle", fake_handle)
    _, history, status, _ = ui.respond("How long does shipping take?", [], "session-1")
    assert history[-1]["content"].startswith("Shipping takes")
    assert status == "rag · source: return_and_shipping_policy.md"
