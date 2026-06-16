"""V3 gate and backend selection for DCU MegaMoE large-opt stages."""

from __future__ import annotations

import os


LARGE_OPT_3STAGE_ENV = "MEGAMOE_DCU_USE_LARGE_OPT_3STAGE"
V3_ENV = "USE_MEGAMOE_V3"
V3_BACKEND_ENV = "MEGAMOE_DCU_V3_BACKEND"
V3_BACKEND_LL = "ll"
V3_BACKEND_NORMAL = "normal"

_LARGE_OPT_FORCE_VALUES = {"1", "true", "yes", "on", "large_opt", "3stage"}
_V3_TRUE_VALUES = {"1", "true", "yes", "on"}
_VALID_V3_BACKENDS = {V3_BACKEND_LL, V3_BACKEND_NORMAL}


def _normalize_env_value(value: str | None) -> str:
    return "" if value is None else value.strip().lower()


def large_opt_3stage_forced(value: str | None = None) -> bool:
    raw = os.getenv(LARGE_OPT_3STAGE_ENV, "auto") if value is None else value
    return _normalize_env_value(raw) in _LARGE_OPT_FORCE_VALUES


def v3_requested(value: str | None = None) -> bool:
    raw = os.getenv(V3_ENV, "0") if value is None else value
    return _normalize_env_value(raw) in _V3_TRUE_VALUES


def v3_enabled(
    *,
    large_opt_3stage_value: str | None = None,
    use_v3_value: str | None = None,
) -> bool:
    return large_opt_3stage_forced(large_opt_3stage_value) and v3_requested(
        use_v3_value
    )


def normalize_v3_backend(value: str | None) -> str:
    backend = (value or V3_BACKEND_NORMAL).strip().lower()
    if backend not in _VALID_V3_BACKENDS:
        raise ValueError(
            f"{V3_BACKEND_ENV} must be one of {sorted(_VALID_V3_BACKENDS)}, "
            f"got {value!r}"
        )
    return backend


def get_v3_backend() -> str:
    return normalize_v3_backend(os.getenv(V3_BACKEND_ENV, V3_BACKEND_NORMAL))
