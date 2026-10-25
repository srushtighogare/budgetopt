"""Unit tests: security primitives, cost math and budget-state rules."""

from decimal import Decimal

import jwt
import pytest

from app.core.security import (
    create_token,
    decode_token,
    decrypt_secret,
    encrypt_secret,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from app.db.models import ModelPricing
from app.services.budget import BudgetState, classify
from app.services.pricing import compute_cost


def test_password_hash_roundtrip():
    h = hash_password("s3cret-pass")
    assert h.startswith("$argon2id$")
    assert verify_password("s3cret-pass", h)
    assert not verify_password("wrong-pass", h)
    assert not verify_password("anything", "not-a-hash")


def test_token_type_is_enforced():
    refresh = create_token("42", "refresh")
    assert decode_token(refresh, "refresh")["sub"] == "42"
    with pytest.raises(jwt.InvalidTokenError):
        decode_token(refresh, "access")


def test_tampered_token_rejected():
    token = create_token("1", "access")
    with pytest.raises(jwt.InvalidTokenError):
        decode_token(token[:-2] + ("A" if token[-2] != "A" else "B") + token[-1], "access")


def test_api_key_hash_matches():
    key, prefix, key_hash = generate_api_key()
    assert key.startswith("bo_") and key.startswith(prefix)
    assert hash_api_key(key) == key_hash and key not in key_hash


def test_provider_secret_encryption():
    ct = encrypt_secret("gsk_provider_key")
    assert "gsk_provider_key" not in ct
    assert decrypt_secret(ct) == "gsk_provider_key"


def test_compute_cost_exact():
    p = ModelPricing(input_per_1m=Decimal("0.15"), output_per_1m=Decimal("0.60"))
    # 1,200 in * 0.15/1M + 800 out * 0.60/1M = 0.00018 + 0.00048
    assert compute_cost(p, 1200, 800) == Decimal("0.00066000")
    assert compute_cost(p, 0, 0) == Decimal("0E-8")
    with pytest.raises(ValueError):
        compute_cost(p, -1, 0)


@pytest.mark.parametrize(
    ("spend", "budget", "state"),
    [
        ("0", "10", BudgetState.OK),
        ("7.99", "10", BudgetState.OK),
        ("8", "10", BudgetState.SOFT_ALERT),
        ("9.99", "10", BudgetState.SOFT_ALERT),
        ("10", "10", BudgetState.HARD_BLOCK),
        ("12", "10", BudgetState.HARD_BLOCK),
    ],
)
def test_budget_classify(spend, budget, state):
    assert classify(Decimal(spend), Decimal(budget), 0.8)[1] == state
