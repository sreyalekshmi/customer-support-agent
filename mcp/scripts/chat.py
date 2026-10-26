"""Terminal chat for the agent. Run from repo root: python -m scripts.chat"""

import asyncio
import uuid

from app.agent.orchestrator import handle
from app.schemas import SupportRequest


async def main() -> None:
    session_id = str(uuid.uuid4())
    print("Support agent (trained intent model, RAG, and MCP tools). Ctrl+C to quit. Try ORD-1001.")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        req = SupportRequest(request_id=str(uuid.uuid4()), session_id=session_id, user_message=msg)
        resp = await handle(req)
        extra = f"  [{resp.destination}, tool={resp.tool_used}, escalated={resp.escalated}]"
        print(f"Agent: {resp.reply}\n{extra}")


if __name__ == "__main__":
    asyncio.run(main())
