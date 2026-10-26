"""Backend selection. One place decides MOCK vs real, from explicit configuration.

A configuration problem is remembered and re-raised as ConfigError on every call; it is
never converted into mock data.
"""

import logging
import threading
from typing import Any

from app import config
from app.config import ConfigError
from app.mcp_layer.backends.base import OrderBackend, PaymentBackend

log = logging.getLogger("mcp_layer.services")

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _build_order_backend() -> OrderBackend:
    mode = config.order_backend_mode()
    if mode == "shopify":
        from app.mcp_layer.backends.orders_shopify import ShopifyOrderBackend
        cfg = config.load_shopify_config()
        log.info("Order backend: SHOPIFY (store=%s, api=%s, auth=%s)", cfg.store_domain, cfg.api_version,
                 "client_credentials" if cfg.uses_client_credentials else "static_token")
        return ShopifyOrderBackend(cfg)
    from app.mcp_layer.backends.orders_mock import MockOrderBackend
    log.info("Order backend: MOCK (simulated data)")
    return MockOrderBackend()


def _build_payment_backend() -> PaymentBackend:
    mode = config.payment_backend_mode()
    if mode == "stripe":
        from app.mcp_layer.backends.payments_stripe import StripePaymentBackend
        log.info("Payment backend: STRIPE TEST MODE")
        return StripePaymentBackend(config.load_stripe_config())
    from app.mcp_layer.backends.payments_mock import MockPaymentBackend
    log.info("Payment backend: MOCK (simulated data)")
    return MockPaymentBackend()


def _get(key: str, builder):
    with _lock:
        if key not in _cache:
            config.load_env()
            try:
                _cache[key] = builder()
            except ConfigError as exc:
                _cache[key] = exc
        value = _cache[key]
    if isinstance(value, ConfigError):
        raise value
    return value


def get_order_backend() -> OrderBackend:
    return _get("orders", _build_order_backend)


def get_payment_backend() -> PaymentBackend:
    return _get("payments", _build_payment_backend)


def reset() -> None:
    """Forget cached backends (tests only)."""
    with _lock:
        _cache.clear()
