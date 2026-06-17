"""V3 backend names and test-layer backend selection helpers."""

from __future__ import annotations

import os

V3_BACKEND_LL = "ll"
V3_BACKEND_NORMAL = "normal"
V3_BACKEND_AUTO = "auto"

BACKEND_ENV = "MEGAMOE_DCU_BACKEND"
NORMAL_LL_TOKEN_THRESHOLD_ENV = "MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD"
DEFAULT_NORMAL_LL_TOKEN_THRESHOLD = 256

_VALID_V3_BACKENDS = {V3_BACKEND_LL, V3_BACKEND_NORMAL}
_VALID_BACKEND_MODES = {V3_BACKEND_AUTO, V3_BACKEND_LL, V3_BACKEND_NORMAL}


def normalize_v3_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in _VALID_V3_BACKENDS:
        raise ValueError(
            f"V3 backend must be one of {sorted(_VALID_V3_BACKENDS)}, got {value!r}"
        )
    return backend


def normalize_backend_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in _VALID_BACKEND_MODES:
        raise ValueError(
            f"{BACKEND_ENV} must be one of {sorted(_VALID_BACKEND_MODES)}, got {value!r}"
        )
    return mode


def normal_ll_token_threshold(value: str | int | None = None) -> int:
    raw = os.getenv(
        NORMAL_LL_TOKEN_THRESHOLD_ENV,
        str(DEFAULT_NORMAL_LL_TOKEN_THRESHOLD),
    )
    if value is not None:
        raw = value
    threshold = int(raw)
    if threshold < 0:
        raise ValueError(f"{NORMAL_LL_TOKEN_THRESHOLD_ENV} must be non-negative")
    return threshold


def v3_backend_mode(value: str | None = None) -> str:
    raw = os.getenv(BACKEND_ENV, V3_BACKEND_AUTO) if value is None else value
    return normalize_backend_mode(raw)


def select_v3_backend(selector_tokens: int, backend_mode: str | None = None) -> str:
    """Return the test-layer V3 backend for a uniform EP selector token bucket."""

    tokens = int(selector_tokens)
    if tokens < 0:
        raise ValueError("selector_tokens must be non-negative")
    mode = v3_backend_mode(backend_mode)
    if mode != V3_BACKEND_AUTO:
        return normalize_v3_backend(mode)
    threshold = normal_ll_token_threshold()
    return V3_BACKEND_LL if tokens <= threshold else V3_BACKEND_NORMAL
