"""V3 backend names and test-layer backend selection helpers."""

from __future__ import annotations

import os

V3_BACKEND_LL = "ll"
V3_BACKEND_NORMAL = "normal"
V3_BACKEND_AUTO = "auto"

V3_QUANT_FP8 = "fp8"
V3_QUANT_INT8 = "int8"

BACKEND_ENV = "MEGAMOE_DCU_BACKEND"
NORMAL_LL_TOKEN_THRESHOLD_ENV = "MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD"
DEFAULT_NORMAL_LL_TOKEN_THRESHOLD = 512

_VALID_V3_BACKENDS = {V3_BACKEND_LL, V3_BACKEND_NORMAL}
_VALID_BACKEND_MODES = {V3_BACKEND_AUTO, V3_BACKEND_LL, V3_BACKEND_NORMAL}
_VALID_V3_QUANTS = {V3_QUANT_FP8, V3_QUANT_INT8}

SUPPORTED_STAGED_EP_RANKS = (8, 16, 32)

DEEPSEEK_V4_FLASH_SHAPE = (256, 6, 4096, 2048)
DEEPSEEK_V4_PRO_SHAPE = (384, 6, 7168, 3072)
YGZP_INT8_SHAPE = (288, 8, 4096, 2048)
STAGED_PACK5_MODEL_SHAPES = {
    "DeepSeek-V4-Flash": DEEPSEEK_V4_FLASH_SHAPE,
    "DeepSeek-V4-Pro": DEEPSEEK_V4_PRO_SHAPE,
}
STAGED_PACK5_LOCAL_EXPERTS = tuple(
    sorted(
        {
            num_experts // num_ranks
            for num_experts, _, _, _ in STAGED_PACK5_MODEL_SHAPES.values()
            for num_ranks in SUPPORTED_STAGED_EP_RANKS
        }
    )
)
STAGED_PACK5_SHAPE_CONTRACT = (
    "DCU MegaMoE staged LL/normal pack5 path supports "
    "DeepSeek-V4-Flash EP8/EP16/EP32 experts=256 topk=6 hidden=4096 "
    "intermediate=2048 and DeepSeek-V4-Pro EP8/EP16/EP32 experts=384 "
    "topk=6 hidden=7168 intermediate=3072"
)

# This registry is deliberately separate from STAGED_PACK5_MODEL_SHAPES.  The
# latter describes the legacy FP8 shape/layout whitelist, while this one is the
# exact execution capability gate used by quant-aware dispatch.
STAGED_V3_MODEL_CAPABILITIES = {
    "DeepSeek-V4-Flash": {
        "shape": DEEPSEEK_V4_FLASH_SHAPE,
        "quant": V3_QUANT_FP8,
        "backends": (V3_BACKEND_LL, V3_BACKEND_NORMAL),
        "ep_ranks": SUPPORTED_STAGED_EP_RANKS,
    },
    "DeepSeek-V4-Pro": {
        "shape": DEEPSEEK_V4_PRO_SHAPE,
        "quant": V3_QUANT_FP8,
        "backends": (V3_BACKEND_LL, V3_BACKEND_NORMAL),
        "ep_ranks": SUPPORTED_STAGED_EP_RANKS,
    },
    "YGZP-INT8": {
        "shape": YGZP_INT8_SHAPE,
        "quant": V3_QUANT_INT8,
        "backends": (V3_BACKEND_NORMAL,),
        "ep_ranks": (8,),
    },
}
STAGED_V3_CAPABILITY_CONTRACT = (
    "DCU MegaMoE staged V3 supports FP8 DeepSeek-V4-Flash/Pro LL or normal "
    "on EP8/EP16/EP32, and INT8 YGZP normal on EP8 with experts=288 topk=8 "
    "hidden=4096 intermediate=2048"
)


def normalize_v3_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in _VALID_V3_BACKENDS:
        raise ValueError(
            f"V3 backend must be one of {sorted(_VALID_V3_BACKENDS)}, got {value!r}"
        )
    return backend


def normalize_v3_quant(value: str) -> str:
    quant = str(value).strip().lower()
    if quant not in _VALID_V3_QUANTS:
        raise ValueError(
            f"V3 quant must be one of {sorted(_VALID_V3_QUANTS)}, got {value!r}"
        )
    return quant


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


def staged_pack5_model_shape_supported(
    *,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    shape = (
        int(num_experts),
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
    )
    return shape in STAGED_PACK5_MODEL_SHAPES.values()


def staged_pack5_shape_supported(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    ranks = int(num_ranks)
    experts = int(num_experts)
    return (
        ranks in SUPPORTED_STAGED_EP_RANKS
        and experts % ranks == 0
        and staged_pack5_model_shape_supported(
            num_experts=experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    )


def staged_v3_capability_supported(
    *,
    quant: str,
    backend: str,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    """Return whether an exact quant/backend/EP/model-shape combination exists."""

    quant_mode = str(quant).strip().lower()
    backend_mode = str(backend).strip().lower()
    ranks = int(num_ranks)
    experts = int(num_experts)
    if (
        quant_mode not in _VALID_V3_QUANTS
        or backend_mode not in _VALID_V3_BACKENDS
        or ranks <= 0
        or experts % ranks != 0
    ):
        return False

    shape = (
        experts,
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
    )
    return any(
        shape == capability["shape"]
        and quant_mode == capability["quant"]
        and backend_mode in capability["backends"]
        and ranks in capability["ep_ranks"]
        for capability in STAGED_V3_MODEL_CAPABILITIES.values()
    )


def staged_v3_capability_local_experts(
    *,
    quant: str,
    backend: str,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> int:
    """Return local experts only after the exact execution gate succeeds."""

    if not staged_v3_capability_supported(
        quant=quant,
        backend=backend,
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    ):
        raise ValueError(STAGED_V3_CAPABILITY_CONTRACT)
    return int(num_experts) // int(num_ranks)


def staged_pack5_local_experts(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> int:
    if not staged_pack5_shape_supported(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    ):
        raise ValueError(STAGED_PACK5_SHAPE_CONTRACT)
    return int(num_experts) // int(num_ranks)


def staged_pack5_local_experts_supported(local_experts: int) -> bool:
    return int(local_experts) in STAGED_PACK5_LOCAL_EXPERTS


def staged_pack5_k1_shape_supported(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    l1_rows: int,
) -> bool:
    intermediate_hidden = int(l1_rows) // 2
    return (
        int(l1_rows) % 2 == 0
        and staged_pack5_shape_supported(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    )


def staged_pack5_k3_dims_supported(*, hidden: int, intermediate_hidden: int) -> bool:
    return any(
        int(hidden) == shape_hidden and int(intermediate_hidden) == shape_intermediate
        for _, _, shape_hidden, shape_intermediate in STAGED_PACK5_MODEL_SHAPES.values()
    )


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
