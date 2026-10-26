"""Environment-driven configuration. Secrets come only from the environment or .env.

Backend selection is explicit and validated: an unknown mode is a ConfigError, and a
real mode with missing settings is a ConfigError. Nothing here ever falls back to mock.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

ORDER_MODES = ("mock", "shopify")
PAYMENT_MODES = ("mock", "stripe")
DEFAULT_SHOPIFY_API_VERSION = "2026-07"

# Only *.myshopify.com is accepted, so a typo or malicious value can never make us
# send an access token to an arbitrary host.
_SHOP_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]*\.myshopify\.com$")
_API_VERSION = re.compile(r"^\d{4}-\d{2}$")


class ConfigError(Exception):
    """Invalid or incomplete configuration. Message is safe to log (never contains secrets)."""


def load_env() -> None:
    """Load REPO_ROOT/.env without overriding variables already set in the process."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def _get(name: str) -> str:
    return os.environ.get(name, "").strip()


def _mode(var: str, allowed: tuple[str, ...]) -> str:
    value = (_get(var) or "mock").lower()
    if value not in allowed:
        raise ConfigError(f"{var} must be one of {', '.join(allowed)} (got {value!r}).")
    return value


def order_backend_mode() -> str:
    return _mode("ORDER_BACKEND", ORDER_MODES)


def payment_backend_mode() -> str:
    return _mode("PAYMENT_BACKEND", PAYMENT_MODES)


def _timeout(var: str) -> float:
    raw = _get(var) or "10"
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{var} must be a number of seconds.") from None
    if not 1 <= value <= 30:
        raise ConfigError(f"{var} must be between 1 and 30 seconds.")
    return value


@dataclass(frozen=True)
class ShopifyConfig:
    store_domain: str
    api_version: str
    # Auth: a static Admin API token, OR client credentials (Dev Dashboard apps).
    access_token: str = field(default="", repr=False)
    client_id: str = field(default="", repr=False)
    client_secret: str = field(default="", repr=False)
    # Mapping ORD-1001 -> Shopify order name "#1001" (stores can customise their format).
    order_name_prefix: str = "#"
    order_name_suffix: str = ""
    timeout_seconds: float = 10.0

    @property
    def uses_client_credentials(self) -> bool:
        return not self.access_token


def load_shopify_config() -> ShopifyConfig:
    domain = _get("SHOPIFY_STORE_DOMAIN").lower()
    if not domain:
        raise ConfigError("SHOPIFY_STORE_DOMAIN is not set.")
    if not _SHOP_DOMAIN.match(domain):
        raise ConfigError(
            "SHOPIFY_STORE_DOMAIN must look like your-store.myshopify.com "
            "(no https://, no path)."
        )

    api_version = _get("SHOPIFY_API_VERSION") or DEFAULT_SHOPIFY_API_VERSION
    if not _API_VERSION.match(api_version):
        raise ConfigError("SHOPIFY_API_VERSION must look like 2026-07.")

    token = _get("SHOPIFY_ACCESS_TOKEN")
    client_id = _get("SHOPIFY_CLIENT_ID")
    client_secret = _get("SHOPIFY_CLIENT_SECRET")
    if not token and not (client_id and client_secret):
        raise ConfigError(
            "Shopify credentials missing: set SHOPIFY_ACCESS_TOKEN, or both "
            "SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET."
        )

    return ShopifyConfig(
        store_domain=domain,
        api_version=api_version,
        access_token=token,
        client_id=client_id,
        client_secret=client_secret,
        order_name_prefix=os.environ.get("SHOPIFY_ORDER_NAME_PREFIX", "#"),
        order_name_suffix=os.environ.get("SHOPIFY_ORDER_NAME_SUFFIX", ""),
        timeout_seconds=_timeout("SHOPIFY_TIMEOUT_SECONDS"),
    )


@dataclass(frozen=True)
class StripeConfig:
    secret_key: str = field(repr=False)
    timeout_seconds: float = 10.0


def load_stripe_config() -> StripeConfig:
    key = _get("STRIPE_SECRET_KEY")
    if not key:
        raise ConfigError("STRIPE_SECRET_KEY is not set.")
    # Test mode only in this prototype. Live keys are refused outright.
    if not key.startswith(("sk_test_", "rk_test_")):
        raise ConfigError("STRIPE_SECRET_KEY must be a test-mode key (sk_test_... or rk_test_...).")
    return StripeConfig(secret_key=key, timeout_seconds=_timeout("STRIPE_TIMEOUT_SECONDS"))
