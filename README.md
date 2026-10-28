# Customer Support Agent

An AI-powered customer-support agent that combines intent classification, retrieval-augmented
policy answers, and MCP tools for order and payment lookups.

The runnable application lives in [`mcp/`](./mcp/). It runs offline by default with deterministic
mock order and payment data, so you can start it without external credentials.

## What is included

- **Agent orchestration**: routes customer messages to intent classification, RAG, or MCP tools.
- **Intent classification**: predicts one of the supported customer-support intents.
- **RAG policy search**: supports hybrid, semantic, and BM25 retrieval over local policy documents.
- **MCP service layer**: exposes order and payment tools with mock backends, plus optional Shopify
  and Stripe test-mode integrations.
- **Web UI**: React and TypeScript frontend served by a small Python HTTP API.
- **Terminal client**: interactive chat for quick local testing.

## Prerequisites

- Python 3.10 or newer
- Node.js and npm for the web frontend
- Git

## Quick start

From the repository root:

```bash
cd mcp
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the Python dependencies and create the local configuration:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows Command Prompt, use `copy .env.example .env` instead of `cp`.
The generated configuration uses `mock` backends and does not require credentials.

## Run the project

### Browser UI in development mode

Open two terminals, both starting in `mcp/`.

**Terminal 1 - Python API:**

```bash
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m app.web
```

**Terminal 2 - Vite frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://127.0.0.1:5173`.
The Vite development server proxies `/api` requests to the Python service at
`http://127.0.0.1:8000`.

### Browser UI as a production build

From `mcp/`:

```bash
cd frontend
npm install
npm run build
cd ..
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
python -m app.web
```

Then open `http://127.0.0.1:8000`. The Python service serves the compiled frontend and API.
You can change the bind address and port with `SUPPORT_HOST` and `SUPPORT_PORT`.

### Terminal chat

From `mcp/`, with the virtual environment activated:

```bash
python -m scripts.chat
```

Try a mock order query such as `Where is order ORD-1001?`. Press `Ctrl+C` to exit.

### MCP server by itself

The client normally starts the MCP server automatically. To run it directly:

```bash
python -m app.mcp_layer.server
```

The server communicates over standard input/output and logs to standard error.

## Configuration

Copy [`mcp/.env.example`](./mcp/.env.example) to `mcp/.env`. Never commit `.env` or put
credentials in source code.

The default offline configuration is:

```dotenv
ORDER_BACKEND=mock
PAYMENT_BACKEND=mock
```

To use external services, configure the corresponding variables in `mcp/.env`:

```dotenv
ORDER_BACKEND=shopify
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_CLIENT_ID=...
SHOPIFY_CLIENT_SECRET=...

PAYMENT_BACKEND=stripe
STRIPE_SECRET_KEY=sk_test_...
```

Only Shopify Admin API configuration and Stripe **test-mode** keys are supported. Invalid or
incomplete real-service configuration produces an explicit configuration error; it does not
silently fall back to mock data.

For optional Groq-generated RAG answers, set `GROQ_API_KEY` (or the legacy `GROQ_API` alias).
Without it, the application uses the deterministic source-quoting responder. LangSmith tracing
is also optional and disabled by default.

See the detailed service-layer documentation in [`mcp/README.md`](./mcp/README.md) for all
environment variables, tool schemas, security notes, and integration details.

## Tests

Run the offline test suite from `mcp/`:

```bash
python -m pytest -q
```

If a globally installed pytest plugin interferes with collection:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The tests use mock backends and mocked HTTP responses; they do not require live Shopify, Stripe,
or Groq credentials.

## Project layout

```text
intent_classification/     Intent data, models, training scripts, and evaluation results
mcp/app/                   Agent, RAG, MCP, configuration, and web API code
mcp/frontend/              React + TypeScript + Vite browser application
mcp/data/                  Mock fixtures, policy documents, and local Chroma index
mcp/scripts/               Terminal chat and RAG evaluation utilities
mcp/tests/                 Automated tests
rag/                       Standalone retrieval components
```

## Troubleshooting

- **`Frontend is not built yet`**: run `cd mcp/frontend && npm install && npm run build`, then
  restart `python -m app.web`.
- **Port 8000 is busy**: set `SUPPORT_PORT` to another port, for example
  `SUPPORT_PORT=8001 python -m app.web`.
- **Semantic retrieval is unavailable**: the app reports a BM25 fallback when local embedding
  dependencies or model weights cannot be loaded.
- **Policy changes are not reflected**: remove `mcp/data/chroma_index/` and restart the app so
  the local index is rebuilt.

## Further documentation

- [MCP and external-service layer](./mcp/README.md)
- [Intent classification module](./intent_classification/README_intent_module.md)
