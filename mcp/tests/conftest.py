"""Test isolation: default every test to MOCK backends and keep the developer's .env out.

Variables set here win over .env (the loader never overrides existing variables), so a
real .env with Shopify/Stripe credentials cannot make the normal test run touch a real API.
Live tests opt in explicitly (RUN_LIVE_TESTS=1) and build their own backends.
"""

import os

import pytest

os.environ["ORDER_BACKEND"] = "mock"
os.environ["PAYMENT_BACKEND"] = "mock"


@pytest.fixture(autouse=True)
def _fresh_backends():
    from app.mcp_layer import services
    services.reset()
    yield
    services.reset()
