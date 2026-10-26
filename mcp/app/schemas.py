"""Shared data contracts for the integrated support agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field

Destination = Literal["rag", "mcp_tool", "direct", "escalate"]
RetrievalMode = Literal["hybrid", "semantic", "bm25"]

# App-level intent taxonomy; the classifier adapter maps model labels to these.
Intent = Literal[
    "order_status",
    "payment_status",
    "refund_request",
    "policy_faq",
    "product_info",
    "complaint_or_ticket",
    "greeting_general",
    "human_escalation",
    "unknown",
]


class SupportRequest(BaseModel):
    request_id: str
    session_id: str
    user_message: str
    customer_id: str | None = None
    conversation_context: list[dict[str, str]] = Field(default_factory=list)  # [{"role","content"}]
    retrieval_mode: RetrievalMode | None = None


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    raw_label: str | None = None  # original Bitext label, for debugging
    is_mock: bool = False


class RetrievedChunk(BaseModel):
    text: str
    source: str  # e.g. "refund_policy.md"
    score: float


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    is_mock: bool = False
    retrieval_mode: RetrievalMode = "hybrid"
    retrieval_fallback: bool = False


class RoutingDecision(BaseModel):
    intent: Intent
    confidence: float
    destination: Destination
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    requires_clarification: bool = False
    requires_confirmation: bool = False
    reason: str = ""


class ToolError(BaseModel):
    code: str
    message: str


class ToolResult(BaseModel):
    success: bool
    tool_name: str
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    request_id: str | None = None


class AgentResponse(BaseModel):
    request_id: str
    reply: str
    destination: Destination
    sources: list[str] = Field(default_factory=list)
    tool_used: str | None = None
    escalated: bool = False
    responder: Literal["mock", "llm"] = "mock"
    retrieval_mode: RetrievalMode | None = None
    retrieval_fallback: bool = False
