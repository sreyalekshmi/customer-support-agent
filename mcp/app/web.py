"""Small stdlib HTTP API and production static-file server for the npm UI."""

import asyncio
import json
import logging
import mimetypes
import os
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from app.agent.orchestrator import handle
from app.config import load_env
from app.schemas import SupportRequest

log = logging.getLogger("support_web")
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
MAX_BODY_BYTES = 128_000


class SupportRequestHandler(BaseHTTPRequestHandler):
    server_version = "SupportDesk/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(200, {"status": "ok"})
            return

        if not FRONTEND_DIST.exists():
            self._send(
                503,
                b"Frontend is not built yet. Run npm install and npm run build in frontend/.",
                "text/plain; charset=utf-8",
            )
            return

        relative = unquote(path).lstrip("/")
        candidate = (FRONTEND_DIST / relative).resolve()
        try:
            candidate.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            self._json(404, {"error": "Not found"})
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.is_file():
            candidate = FRONTEND_DIST / "index.html"
        if not candidate.is_file():
            self._json(404, {"error": "Not found"})
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send(200, candidate.read_bytes(), f"{content_type}; charset=utf-8")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self._json(404, {"error": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"error": "Invalid request length"})
            return
        if content_length < 1 or content_length > MAX_BODY_BYTES:
            self._json(413, {"error": "Request body is empty or too large"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            message = payload.get("message", "")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("Message must be a non-empty string.")
            history = payload.get("history", [])
            if not isinstance(history, list):
                raise ValueError("History must be a list.")
            context = [
                {"role": item["role"], "content": item["content"]}
                for item in history[-20:]
                if isinstance(item, dict)
                and item.get("role") in {"user", "assistant"}
                and isinstance(item.get("content"), str)
            ]
            request = SupportRequest(
                request_id=str(uuid.uuid4()),
                session_id=payload.get("session_id") or str(uuid.uuid4()),
                user_message=message.strip(),
                conversation_context=context,
                retrieval_mode=payload.get("retrieval_mode"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError, ValidationError) as exc:
            self._json(400, {"error": str(exc) or "Invalid request"})
            return

        try:
            response = asyncio.run(handle(request))
        except Exception as exc:
            log.exception("Support request failed (%s)", type(exc).__name__)
            self._json(500, {"error": "The support service could not process that message."})
            return
        self._json(200, response.model_dump(mode="json"))

    def log_message(self, format: str, *args) -> None:
        log.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    load_env()
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("SUPPORT_HOST", "127.0.0.1")
    port = int(os.getenv("SUPPORT_PORT", "8000"))
    server = HTTPServer((host, port), SupportRequestHandler)
    log.info("SupportDesk server listening at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
