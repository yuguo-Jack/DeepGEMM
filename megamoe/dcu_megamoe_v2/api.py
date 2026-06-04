"""Public Python API for the isolated DCU MegaMoE V2 path."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from .runtime import run_stages_fused_v2
from .runtime import get_v2_stage_plan

V2_BACKEND_ENV = "MEGAMOE_DCU_V2_BACKEND"
V2_BACKEND_LL = "ll"
V2_BACKEND_NORMAL = "normal"
_VALID_BACKENDS = {V2_BACKEND_LL, V2_BACKEND_NORMAL}


def normalize_v2_backend(value: str | None) -> str:
    backend = (value or V2_BACKEND_LL).strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{V2_BACKEND_ENV} must be one of {sorted(_VALID_BACKENDS)}, got {value!r}"
        )
    return backend


def get_v2_backend() -> str:
    return normalize_v2_backend(os.getenv(V2_BACKEND_ENV, V2_BACKEND_LL))


def fp8_w8a8_mega_moe_v2(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
    *,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    dispatch_num_tokens: Optional[int] = None,
    backend: Optional[str] = None,
    profile_stages: Optional[dict[str, float]] = None,
) -> None:
    """Run the isolated V2 staged-fused path.

    The implementation is intentionally separate from ``megamoe.fp8_w8a8_mega_moe``.
    ``backend`` selects the prototype family:
    - ``ll``: low-latency kernels for all sizes.
    - ``normal``: high-throughput C pack5 kernels for all sizes.
    ``dispatch_num_tokens=-1`` makes K1 read each rank's runtime token count
    from the sym-buffer header, which is the V2 eager entry for uneven tokens.
    """
    selected_backend = normalize_v2_backend(backend) if backend is not None else get_v2_backend()
    run_stages_fused_v2(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        dispatch_num_tokens=dispatch_num_tokens,
        backend=selected_backend,
        profile_stages=profile_stages,
    )
