"""MCP client wrapper used by the agent.

The agent never talks JSON-RPC itself; this wrapper launches the server over
stdio and exposes two things: discover tools, call a tool.

Environment: the MCP SDK passes only HOME/PATH/SHELL/TERM to a stdio server by default,
so we explicitly forward this project's settings (ORDER_BACKEND, SHOPIFY_*, STRIPE_* ...).
The server also reads REPO_ROOT/.env itself; variables already set take precedence.
"""

import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.client.stdio import get_default_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARDED_PREFIXES = ("ORDER_", "PAYMENT_", "SHOPIFY_", "STRIPE_", "TICKET_", "ZENDESK_")


def server_params(env_overrides: dict[str, str] | None = None) -> StdioServerParameters:
    env = get_default_environment()
    env.update({k: v for k, v in os.environ.items() if k.startswith(FORWARDED_PREFIXES)})
    env.update(env_overrides or {})
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_layer.server"],
        cwd=REPO_ROOT,
        env=env,
    )


def _specs(result) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in result.tools]


def _tool_result(name: str, arguments: dict[str, Any], result) -> dict[str, Any]:
    """Protocol-level failures (schema rejection, server error) come back in the ToolResult shape too."""
    if result.is_error or result.structured_content is None:
        text = " ".join(getattr(c, "text", "") for c in result.content)
        return {
            "success": False,
            "tool_name": name,
            "data": None,
            "error": {"code": "TOOL_PROTOCOL_ERROR", "message": text or "Tool call failed."},
            "request_id": arguments.get("request_id"),
        }
    return result.structured_content


class MCPToolClient:
    """One long-lived connection to the MCP server. RECOMMENDED for the running agent.

    Reuses a single server subprocess, so there is no per-call start-up cost and the
    server keeps its in-memory Shopify access token between calls.

        async with MCPToolClient() as tools:
            specs = await tools.list_tools()
            result = await tools.call_tool("get_order_status", {"order_id": "ORD-1001"})
    """

    def __init__(self, env_overrides: dict[str, str] | None = None):
        self._env = env_overrides
        self._cm: Client | None = None
        self._client: Client | None = None

    async def __aenter__(self) -> "MCPToolClient":
        self._cm = Client(server_params(self._env))
        self._client = await self._cm.__aenter__()
        return self

    async def __aexit__(self, *exc_info) -> None:
        assert self._cm is not None
        await self._cm.__aexit__(*exc_info)
        self._cm = self._client = None

    async def list_tools(self) -> list[dict[str, Any]]:
        assert self._client is not None, "use 'async with MCPToolClient()'"
        return _specs(await self._client.list_tools())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None, "use 'async with MCPToolClient()'"
        return _tool_result(name, arguments, await self._client.call_tool(name, arguments))


async def list_tool_specs(env_overrides: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """One-shot tool discovery (starts and stops a server process). Fine for scripts and tests."""
    async with MCPToolClient(env_overrides) as tools:
        return await tools.list_tools()


async def call_tool(
    name: str,
    arguments: dict[str, Any],
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One-shot tool call (starts and stops a server process per call; see MCPToolClient)."""
    async with MCPToolClient(env_overrides) as tools:
        return await tools.call_tool(name, arguments)
