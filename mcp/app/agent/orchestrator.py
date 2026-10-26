"""Orchestrator: request -> intent -> route -> (MCP tool | RAG | direct | escalate) -> reply.

Selection (route) happens here in the agent; execution happens in the MCP server.
Logs request_id, intent and destination only: never message text or tool data.
"""

import logging

from app.agent import responder
from app.agent.routing import route
from app.config import load_env
from app.intent.classifier import predict_intent
from app.mcp_layer.client import call_tool
from app.rag.retriever import generate_rag_answer, retrieve_context
from app.schemas import AgentResponse, SupportRequest, ToolResult

log = logging.getLogger("agent")


async def handle(request: SupportRequest) -> AgentResponse:
    load_env()
    intent = predict_intent(request.user_message)
    decision = route(request, intent)
    log.info("request=%s intent=%s dest=%s", request.request_id, decision.intent, decision.destination)

    def reply(text: str, **kw) -> AgentResponse:
        return AgentResponse(request_id=request.request_id, reply=text,
                             destination=decision.destination, **kw)

    if decision.destination == "mcp_tool":
        raw = await call_tool(decision.tool_name, decision.tool_arguments)
        result = ToolResult.model_validate(raw)
        text, escalated = responder.from_tool_result(result)
        return reply(text, tool_used=decision.tool_name, escalated=escalated)

    if decision.destination == "rag":
        retrieval = retrieve_context(
            request.user_message, mode=request.retrieval_mode
        )
        generated = generate_rag_answer(retrieval)
        if generated:
            return reply(
                generated,
                sources=sorted({chunk.source for chunk in retrieval.chunks}),
                responder="llm",
                retrieval_mode=retrieval.retrieval_mode,
                retrieval_fallback=retrieval.retrieval_fallback,
            )
        text, sources, escalated = responder.from_retrieval(retrieval)
        return reply(
            text,
            sources=sources,
            escalated=escalated,
            retrieval_mode=retrieval.retrieval_mode,
            retrieval_fallback=retrieval.retrieval_fallback,
        )

    if decision.destination == "direct":
        if decision.requires_clarification and decision.intent == "payment_status":
            text = responder.ask_for_payment_id()
        elif decision.requires_clarification:
            text = responder.ask_for_order_id()
        else:
            text = responder.greeting()
        return reply(text)

    return reply(responder.escalation(decision), escalated=True)
