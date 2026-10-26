"""Shared ToolResult construction and error handling for all tools.

Every tool returns {success, tool_name, data, error, request_id}. Business failures
(unknown ID, bad format, upstream outage, bad credentials) are returned as
success=False with a stable error code; they are never raised and never replaced by
mock data.
"""

import logging
import uuid
from typing import Any, Callable

from app.config import ConfigError
from app.mcp_layer.backends.base import BackendError

log = logging.getLogger("mcp_layer.tools")


def new_request_id(request_id: str | None) -> str:
    return request_id or str(uuid.uuid4())


def build_result(
    tool_name: str,
    request_id: str,
    *,
    data: dict[str, Any] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "success": error is None,
        "tool_name": tool_name,
        "data": data,
        "error": error,
        "request_id": request_id,
    }


def error_result(tool_name: str, request_id: str, code: str, message: str) -> dict[str, Any]:
    return build_result(tool_name, request_id, error={"code": code, "message": message})


def run_backend(tool_name: str, request_id: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run a backend call and convert every failure into a ToolResult. Logs no PII or secrets."""
    try:
        return build_result(tool_name, request_id, data=call())
    except BackendError as exc:
        log.warning("tool=%s request=%s code=%s detail=%s", tool_name, request_id, exc.code, exc.detail)
        return error_result(tool_name, request_id, exc.code, exc.message)
    except ConfigError as exc:
        log.error("tool=%s request=%s CONFIG_ERROR: %s", tool_name, request_id, exc)
        return error_result(tool_name, request_id, "CONFIG_ERROR",
                            "This service is not configured correctly. Please contact support.")
    except Exception as exc:  # last resort: never leak a traceback or crash the server
        log.exception("tool=%s request=%s unexpected %s", tool_name, request_id, type(exc).__name__)
        return error_result(tool_name, request_id, "INTERNAL_ERROR", "An unexpected error occurred.")
