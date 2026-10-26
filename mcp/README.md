# AI Customer Support Agent: MCP and external-service layer

MCP server + client exposing customer-support tools to the agent, with two interchangeable backends per
service: **mock** (offline, deterministic) and **real** (Shopify Admin API, Stripe test mode).

## Verification status (read this first)

| Capability | How it is verified | Verified against the real service? |
|---|---|---|
| MCP server, tool discovery and invocation over stdio | Automated tests spawn the real server and call it with a real MCP client | n/a (no external service) |
| `get_order_status`, MOCK backend | Automated tests | n/a |
| `get_order_status`, SHOPIFY backend | Automated tests with **mocked HTTP responses** (`httpx.MockTransport`) | Not tested against a live Shopify store |
| `get_payment_status`, MOCK backend | Automated tests | n/a |
| `get_payment_status`, STRIPE test mode | Automated tests with **mocked HTTP responses** | **No. Not yet tested against a live Stripe account** |
| Refunds, Zendesk tickets | Not implemented | n/a |

Mocked-HTTP tests prove our request building, parsing and error handling against the response shapes
Shopify/Stripe documentation describes. They cannot prove the live API behaves that way. Two Shopify details are
unconfirmed until a live run: the order-name search syntax and the exact throttling error format (the code handles several variants).

## Quick start (mock mode, no credentials)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env   (ORDER_BACKEND=mock by default)
python -m scripts.chat               # terminal chat
cd frontend && npm install && npm run build && cd ..
python -m app.web                     # launch the built npm web app
```

## npm web app

The browser UI is a React + TypeScript app built with Vite. For development, use
two terminals from mcp/:

~~~~bash
# Terminal 1: Python API
source .venv/bin/activate
python -m app.web
~~~~

~~~~bash
# Terminal 2: npm frontend
cd frontend
npm install
npm run dev
~~~~

Open the local URL printed by Vite. Its development server forwards API requests
to the Python service. To serve a production build from Python:

~~~~bash
cd frontend
npm install
npm run build
cd ..
python -m app.web
~~~~

## Switching to real Shopify / Stripe

Edit `.env` (never commit it):

```
ORDER_BACKEND=shopify               # or mock
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_CLIENT_ID=...               # Dev Dashboard app, scope: read_orders only
SHOPIFY_CLIENT_SECRET=...
PAYMENT_BACKEND=stripe              # or mock
STRIPE_SECRET_KEY=sk_test_...       # test keys only; live keys are refused
```

The mode is explicit: an unknown value is an error, and a real mode with bad config returns `CONFIG_ERROR`. It never silently
returns mock data. Every result says where it came from in `data.source` (`mock`, `shopify`, `stripe_test`).

## Web interface

The web interface uses the same orchestrator as the terminal client: order and
payment questions call MCP tools, while policy questions use RAG. Use the
**Policy search strategy** selector to compare **Hybrid**, **Semantic**, and
**BM25** retrieval per question. Hybrid applies query expansion, local BM25 and
local Chroma semantic search, then reciprocal-rank fusion. Semantic uses Chroma
alone; BM25 is the keyword-only baseline. Chunks retain their Markdown section
headings and are shared between the search methods. Chroma uses
`HuggingFaceEmbeddings` and stores its index in `data/chroma_index/`; no hosted
vector database is used. If local embedding dependencies or model weights are
unavailable, semantic and hybrid modes report a BM25 fallback in the status.
When `GROQ_API_KEY` (or
`GROQ_API`) is configured, `ChatGroq` produces an answer grounded in retrieved
policy text; otherwise the deterministic source-quoting responder is used. The
hybrid retrieval design follows the multi-strategy, query optimization, and rank
fusion approach from [production-rag](https://github.com/KazKozDev/production-rag),
adapted to the libraries and local storage already used by this app.

Policy documents live under `data/knowledge_base/`. The local Chroma collection
is built on first use and reused across restarts. To rebuild it after changing
policy documents, remove `data/chroma_index/` and restart the app.

### Optional LangChain tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to send
RAG generation runs to LangSmith; `LANGCHAIN_PROJECT` sets the project name.
Current `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`
variables are also supported. Tracing is off by default and needs no extra
vector database. The Groq call is tagged as `customer-support`, `rag`, and
`groq`.

## Running the MCP server and client

The client starts the server for you over stdio, so you rarely run the server by hand:

```bash
python -m app.mcp_layer.server       # server alone (waits on stdin; logs to stderr). Ctrl+C to stop
```

You can also point any MCP client at: `command = <path to .venv python>`, `args = ["-m","app.mcp_layer.server"]`, `cwd = repo root`.
The server reads `.env` from the repo root itself, and variables already in the process environment win over `.env`.

## Tools

| Tool | Input | Notes |
|---|---|---|
| `get_order_status` | `order_id` (`ORD-1001`, `#1001`, or Shopify names such as `#D10`), optional `request_id` | Read-only. Interface unchanged; result fields added (below) |
| `get_payment_status` | `payment_id` (Stripe `pi_…`, **not** an order ID), optional `request_id` | Read-only. Cannot create, capture or refund |

Result shape for every tool: `{success, tool_name, data, error, request_id}`.

`get_order_status` result fields (the original keys are all still present; added: `financial_status`, `fulfillment_status`, `tracking_url`, `shopify_order_name`; `status` gained `partially_shipped` and `on_hold`):

```jsonc
// SHOPIFY backend. Output of the code against MOCKED HTTP responses, not a live store
{ "success": true, "tool_name": "get_order_status",
  "data": { "order_id": "ORD-1001", "status": "shipped", "financial_status": "paid",
            "fulfillment_status": "fulfilled", "carrier": "BlueDart", "tracking_number": "TRK123",
            "tracking_url": "https://t.example/TRK123", "estimated_delivery": "2026-10-25",
            "shopify_order_name": "#1001", "simulated": false, "source": "shopify" },
  "error": null, "request_id": "req-123" }

// Failure: never mock data
{ "success": false, "tool_name": "get_order_status", "data": null,
  "error": { "code": "AUTH_FAILED", "message": "The order service rejected our credentials." },
  "request_id": "req-124" }
```

```jsonc
// get_payment_status (STRIPE test mode). Output against MOCKED HTTP, amounts in minor units
{ "success": true, "tool_name": "get_payment_status",
  "data": { "payment_id": "pi_TEST123456", "status": "succeeded", "status_summary": "paid",
            "amount_minor": 89900, "currency": "INR", "amount_received_minor": 89900,
            "amount_refunded_minor": 0, "created": "2025-10-09T08:53:20Z", "failure_code": null,
            "livemode": false, "simulated": false, "source": "stripe_test" },
  "error": null, "request_id": "req-125" }
```


## Tests

```bash
python -m pytest -q                                    # all offline tests (mock + mocked HTTP)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q   # use this if a global pytest plugin (e.g. ROS 2) breaks collection
```

Last run in the development sandbox: **162 passed, 5 skipped** (the 5 are the live tests: no credentials there).
Tests never touch the network and ignore your `.env` (`tests/conftest.py` forces mock mode); live tests are opt-in.
Test files: `test_mcp_tools.py` (original MCP tests, unchanged), `test_shopify_backend.py`, `test_stripe_backend.py`,
`test_config_and_modes.py` (config validation, mock mode, no-fallback through the real MCP protocol), `test_agent.py`, and
`test_intent_classifier.py`.

## Connecting the MCP tools to the main agent (for teammates)

Use **one persistent session** (it keeps one server process alive and its cached Shopify token):

```python
from app.mcp_layer.client import MCPToolClient

async with MCPToolClient() as tools:                       # start once, at app startup
    specs = await tools.list_tools()                       # [{name, description, input_schema}] -> give to the LLM as tools
    result = await tools.call_tool("get_order_status",     # execute whatever tool the agent/LLM selected
                                   {"order_id": "ORD-1001", "request_id": req.request_id})
    # result == {"success", "tool_name", "data", "error", "request_id"}: pass it back to the LLM to word the reply
```

- Selecting the tool is the agent's job (`app/agent/routing.py`); the server only executes it.
- Branch on `result["success"]` and `result["error"]["code"]`. On `CONFIG_ERROR`/`AUTH_FAILED`/`UPSTREAM_*`, escalate; don't invent an answer.
- If `data["simulated"]` is true, never present it as real customer data.
- One-shot helpers `call_tool()` / `list_tool_specs()` still exist (a new server process per call; fine for scripts/tests).
- Intent recognition uses the trained TF-IDF model from `../intent_classification/`.
- RAG reads policy documents from `data/knowledge_base/` and persists a local Chroma index in `data/chroma_index/`.

## Layout

```
app/config.py                     env/.env config + validation (no secrets in code)
app/mcp_layer/server.py           MCP server: tool definitions
app/mcp_layer/client.py           MCPToolClient (persistent) + one-shot helpers
app/mcp_layer/services.py         picks mock vs real backend from config (never falls back)
app/mcp_layer/tools/              validation + ToolResult building (orders, payments, common)
app/mcp_layer/backends/           orders_mock, orders_shopify, payments_mock, payments_stripe, base
app/agent/ app/intent/ app/rag/   integrated agent layer
data/                             mock order/payment fixtures and policy knowledge base
app/rag/                          query optimization, BM25, local Chroma, fusion, Groq generation
tests/  scripts/
```

## Security notes

Credentials only via environment/`.env` (git-ignored); shop domain must be `*.myshopify.com` (token can't be sent elsewhere); HTTPS only,
no redirects followed; least-privilege scope (`read_orders`); Stripe live keys refused; queries request no customer PII and results are
whitelisted; user input is regex-validated and sent to Shopify as a GraphQL *variable*; logs contain request IDs and error codes only
(no tokens, no customer data); retries are bounded (3 attempts, wait ≤ 5 s).

## Known limitations

- Not yet run against a live Shopify store or Stripe account (see the table above).
- Shopify returns only the last 60 days of orders unless `read_all_orders` is granted (older orders look "not found").
- No customer authentication: any caller who knows an order ID can look it up. Fine for a demo; production needs identity verification
  before returning order data, and ID enumeration limits.
- No automatic order → payment association; no refunds; no Zendesk yet.
