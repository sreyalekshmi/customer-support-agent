"""Framework-free message adapter retained for small scripts and API callers."""

import asyncio
import logging
import uuid

from app.agent.orchestrator import handle
from app.schemas import SupportRequest

log = logging.getLogger("support_ui")


def respond(
    message: str,
    history: list[dict[str, str]] | None = None,
    session_id: str | None = None,
    retrieval_mode: str | None = None,
) -> tuple[str, list[dict[str, str]], str, str]:
    """Process one message and return the existing chat history/status tuple."""
    history = history or []
    session_id = session_id or str(uuid.uuid4())
    if not message or not message.strip():
        return "", history, "Enter a question to begin.", session_id

    text = message.strip()
    context = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]
    request = SupportRequest(
        request_id=str(uuid.uuid4()),
        session_id=session_id,
        user_message=text,
        conversation_context=context,
        retrieval_mode=retrieval_mode,
    )
    try:
        response = asyncio.run(handle(request))
    except (ValueError, RuntimeError, OSError) as exc:
        log.exception("Support request failed: %s", type(exc).__name__)
        error = "I couldn't process that request. Please try again."
        return "", history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": error},
        ], "Request failed", session_id

    details = response.destination
    if response.tool_used:
        details += f" · {response.tool_used}"
    if response.sources:
        details += f" · source: {', '.join(response.sources)}"
    retrieval_mode = getattr(response, "retrieval_mode", None)
    if retrieval_mode:
        details += f" · retrieval: {retrieval_mode}"
        if getattr(response, "retrieval_fallback", False):
            details += " (semantic unavailable; BM25 fallback)"
    return "", history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": response.reply},
    ], details, session_id
