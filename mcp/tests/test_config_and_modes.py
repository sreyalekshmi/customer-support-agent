"""Configuration validation, mock mode, and "real mode never silently becomes mock".

The last group goes through a real MCP client/server subprocess with real mode selected
but unusable configuration: the tool must report an error, never mock data.
"""

import asyncio
import json

import pytest

from app import config
from app.config import ConfigError
from app.mcp_layer import services
from app.mcp_layer.client import call_tool, list_tool_specs
from app.mcp_layer.tools.orders import lookup_order_status
from app.mcp_layer.tools.payments import lookup_payment_status

SHOPIFY_VARS = ["SHOPIFY_STORE_DOMAIN", "SHOPIFY_ACCESS_TOKEN", "SHOPIFY_CLIENT_ID", "SHOPIFY_CLIENT_SECRET",
                "SHOPIFY_API_VERSION", "SHOPIFY_ORDER_NAME_PREFIX", "SHOPIFY_ORDER_NAME_SUFFIX",
                "SHOPIFY_TIMEOUT_SECONDS", "STRIPE_SECRET_KEY", "STRIPE_TIMEOUT_SECONDS"]


@pytest.fixture()
def clean_env(monkeypatch):
    for name in SHOPIFY_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ------------------------------------------------------------------ mock mode
def test_mock_mode_returns_labelled_simulated_data():
    r = lookup_order_status("ORD-1001")
    assert r["success"] and r["data"]["source"] == "mock" and r["data"]["simulated"] is True
    assert r["data"]["financial_status"] == "paid" and r["data"]["fulfillment_status"] == "fulfilled"


def test_mock_payment_mode():
    d = lookup_payment_status("pi_MOCK1003")["data"]
    assert d["source"] == "mock" and d["simulated"] is True and d["amount_refunded_minor"] == 100000
    assert lookup_payment_status("pi_MOCK9999")["error"]["code"] == "PAYMENT_NOT_FOUND"


def test_unset_mode_defaults_to_mock(clean_env):
    clean_env.delenv("ORDER_BACKEND")
    assert config.order_backend_mode() == "mock"


def test_unknown_mode_is_rejected_not_defaulted(clean_env):
    clean_env.setenv("ORDER_BACKEND", "shopfiy")  # typo
    with pytest.raises(ConfigError):
        config.order_backend_mode()
    r = lookup_order_status("ORD-1001")
    assert not r["success"] and r["data"] is None and r["error"]["code"] == "CONFIG_ERROR"


# ------------------------------------------------------------------ shopify config
def test_shopify_config_requires_domain(clean_env):
    with pytest.raises(ConfigError, match="SHOPIFY_STORE_DOMAIN"):
        config.load_shopify_config()


@pytest.mark.parametrize("domain", [
    "https://demo.myshopify.com", "demo.myshopify.com/admin", "demo.example.com", "evil.com",
    "demo.myshopify.com.evil.com", "demo myshopify.com", "myshopify.com", "-x.myshopify.com",
])
def test_shopify_domain_must_be_a_myshopify_host(clean_env, domain):
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", domain)
    clean_env.setenv("SHOPIFY_ACCESS_TOKEN", "t")
    with pytest.raises(ConfigError):
        config.load_shopify_config()


def test_shopify_config_requires_some_credentials(clean_env):
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", "demo.myshopify.com")
    clean_env.setenv("SHOPIFY_CLIENT_ID", "only-id")
    with pytest.raises(ConfigError, match="credentials"):
        config.load_shopify_config()


def test_shopify_config_accepts_static_token_or_client_credentials(clean_env):
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", "Demo.MyShopify.com")
    clean_env.setenv("SHOPIFY_ACCESS_TOKEN", "t")
    cfg = config.load_shopify_config()
    assert cfg.store_domain == "demo.myshopify.com" and cfg.api_version == config.DEFAULT_SHOPIFY_API_VERSION
    assert not cfg.uses_client_credentials
    clean_env.delenv("SHOPIFY_ACCESS_TOKEN")
    clean_env.setenv("SHOPIFY_CLIENT_ID", "i")
    clean_env.setenv("SHOPIFY_CLIENT_SECRET", "s")
    assert config.load_shopify_config().uses_client_credentials


@pytest.mark.parametrize("value", ["abc", "0", "31", "-5"])
def test_timeout_is_validated(clean_env, value):
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", "demo.myshopify.com")
    clean_env.setenv("SHOPIFY_ACCESS_TOKEN", "t")
    clean_env.setenv("SHOPIFY_TIMEOUT_SECONDS", value)
    with pytest.raises(ConfigError):
        config.load_shopify_config()


def test_secrets_do_not_appear_in_repr(clean_env):
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", "demo.myshopify.com")
    clean_env.setenv("SHOPIFY_ACCESS_TOKEN", "shpat_SECRET")
    clean_env.setenv("STRIPE_SECRET_KEY", "sk_test_SECRET")
    assert "shpat_SECRET" not in repr(config.load_shopify_config())
    assert "sk_test_SECRET" not in repr(config.load_stripe_config())


# ------------------------------------------------------------------ stripe config
@pytest.mark.parametrize("key", ["sk_live_abc", "rk_live_abc", "pk_test_abc", "whatever"])
def test_stripe_live_or_wrong_keys_are_refused(clean_env, key):
    clean_env.setenv("STRIPE_SECRET_KEY", key)
    with pytest.raises(ConfigError, match="test-mode"):
        config.load_stripe_config()


def test_stripe_test_keys_are_accepted(clean_env):
    for key in ("sk_test_abc", "rk_test_abc"):
        clean_env.setenv("STRIPE_SECRET_KEY", key)
        assert config.load_stripe_config().secret_key == key


# ------------------------------------------------------------------ real mode: no silent fallback (in-process)
def test_shopify_mode_without_config_is_an_error_not_mock(clean_env):
    clean_env.setenv("ORDER_BACKEND", "shopify")
    services.reset()
    r = lookup_order_status("ORD-1001")  # exists in mock data
    assert r["success"] is False and r["data"] is None and r["error"]["code"] == "CONFIG_ERROR"
    assert "MOCK" not in json.dumps(r) and "mock" not in json.dumps(r["data"] or {})


def test_stripe_mode_with_live_key_is_an_error_not_mock(clean_env):
    clean_env.setenv("PAYMENT_BACKEND", "stripe")
    clean_env.setenv("STRIPE_SECRET_KEY", "sk_live_abc")
    services.reset()
    r = lookup_payment_status("pi_MOCK1001")
    assert r["success"] is False and r["data"] is None and r["error"]["code"] == "CONFIG_ERROR"


def test_config_error_message_for_agent_does_not_reveal_settings(clean_env):
    clean_env.setenv("ORDER_BACKEND", "shopify")
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", "demo.myshopify.com")
    services.reset()
    msg = lookup_order_status("ORD-1001")["error"]["message"]
    assert "SHOPIFY" not in msg and "myshopify" not in msg


# ------------------------------------------------------------------ real mode: no silent fallback (through MCP)
SHOPIFY_UNCONFIGURED = {"ORDER_BACKEND": "shopify", "SHOPIFY_STORE_DOMAIN": "", "SHOPIFY_ACCESS_TOKEN": "",
                        "SHOPIFY_CLIENT_ID": "", "SHOPIFY_CLIENT_SECRET": ""}


def test_mcp_tool_in_shopify_mode_unconfigured_returns_error_not_mock():
    r = asyncio.run(call_tool("get_order_status", {"order_id": "ORD-1001"}, env_overrides=SHOPIFY_UNCONFIGURED))
    assert r["success"] is False and r["data"] is None and r["error"]["code"] == "CONFIG_ERROR"


def test_mcp_tool_with_invalid_mode_returns_error_not_mock():
    r = asyncio.run(call_tool("get_order_status", {"order_id": "ORD-1001"}, env_overrides={"ORDER_BACKEND": "banana"}))
    assert r["success"] is False and r["error"]["code"] == "CONFIG_ERROR"


def test_mcp_payment_tool_with_live_key_returns_error():
    r = asyncio.run(call_tool("get_payment_status", {"payment_id": "pi_MOCK1001"},
                              env_overrides={"PAYMENT_BACKEND": "stripe", "STRIPE_SECRET_KEY": "sk_live_x"}))
    assert r["success"] is False and r["error"]["code"] == "CONFIG_ERROR"


# ------------------------------------------------------------------ MCP interface stability
def test_mcp_discovery_lists_both_tools_with_stable_order_status_schema():
    specs = {s["name"]: s for s in asyncio.run(list_tool_specs())}
    assert set(specs) == {"get_order_status", "get_payment_status"}
    schema = specs["get_order_status"]["input_schema"]
    assert set(schema["properties"]) == {"order_id", "request_id"} and schema["required"] == ["order_id"]
    assert specs["get_payment_status"]["input_schema"]["required"] == ["payment_id"]


def test_mcp_payment_tool_over_protocol_in_mock_mode():
    r = asyncio.run(call_tool("get_payment_status", {"payment_id": "pi_MOCK1001"}))
    assert r["success"] and r["data"]["status"] == "succeeded" and r["data"]["source"] == "mock"


def test_legacy_order_result_fields_are_preserved():
    """The original (milestone 1) data keys must still be present: additive changes only."""
    legacy = {"order_id", "status", "carrier", "tracking_number", "estimated_delivery", "simulated", "source"}
    assert legacy <= set(lookup_order_status("ORD-1001")["data"])


# ------------------------------------------------------------------ env -> backend wiring (no network)
def test_env_builds_real_shopify_backend(clean_env):
    from app.mcp_layer.backends.orders_shopify import ShopifyOrderBackend
    clean_env.setenv("ORDER_BACKEND", "shopify")
    clean_env.setenv("SHOPIFY_STORE_DOMAIN", "demo.myshopify.com")
    clean_env.setenv("SHOPIFY_CLIENT_ID", "i")
    clean_env.setenv("SHOPIFY_CLIENT_SECRET", "s")
    services.reset()
    assert isinstance(services.get_order_backend(), ShopifyOrderBackend)


def test_env_builds_real_stripe_backend(clean_env):
    from app.mcp_layer.backends.payments_stripe import StripePaymentBackend
    clean_env.setenv("PAYMENT_BACKEND", "stripe")
    clean_env.setenv("STRIPE_SECRET_KEY", "sk_test_abc")
    services.reset()
    assert isinstance(services.get_payment_backend(), StripePaymentBackend)


# ------------------------------------------------------------------ persistent client session
def test_persistent_client_reuses_one_server_for_many_calls():
    from app.mcp_layer.client import MCPToolClient

    async def scenario():
        async with MCPToolClient() as tools:
            names = [s["name"] for s in await tools.list_tools()]
            a = await tools.call_tool("get_order_status", {"order_id": "ORD-1001"})
            b = await tools.call_tool("get_payment_status", {"payment_id": "pi_MOCK1002"})
            c = await tools.call_tool("get_order_status", {"order_id": 5})  # schema rejection
            return names, a, b, c

    names, a, b, c = asyncio.run(scenario())
    assert "get_order_status" in names and a["success"] and b["data"]["failure_code"] == "card_declined"
    assert c["error"]["code"] == "TOOL_PROTOCOL_ERROR"
