"""V3 pack5 weight layout helpers for offline fixtures.

These helpers are intentionally not used by the large-opt runtime path. V3
execution expects callers/tests to provide weights that are already in this
pack5 layout.
"""

from __future__ import annotations

from typing import Tuple

import torch


PACK5_K64 = 64
PACK5_N256 = 256
PACK5_K16 = 16
PACK5_N16 = 16


def pack5_physical_to_logical_indices(device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(
        [(physical & 3) * 4 + (physical >> 2) for physical in range(PACK5_N16)],
        dtype=torch.long,
        device=device,
    )


def pack5_logical_to_physical_ni(logical_ni: int) -> int:
    if logical_ni < 0 or logical_ni >= PACK5_N16:
        raise ValueError("logical_ni must be in [0, 16)")
    return (logical_ni & 3) * 4 + (logical_ni >> 2)


def pack5_shape(experts: int, n: int, k: int) -> tuple[int, int, int, int, int, int, int]:
    if n % PACK5_N256 != 0 or k % PACK5_K64 != 0:
        raise ValueError("pack5 shape expects n divisible by 256 and k divisible by 64")
    return (
        experts,
        k // PACK5_K64,
        n // PACK5_N256,
        PACK5_N256 // PACK5_N16,
        PACK5_K64 // PACK5_K16,
        PACK5_N16,
        PACK5_K16,
    )


def pack5_flat_offset(*, expert: int, n: int, k: int, row: int, col: int) -> int:
    shape = pack5_shape(expert + 1, n, k)
    if row < 0 or row >= n or col < 0 or col >= k:
        raise ValueError("row/col out of range for pack5 shape")
    ko = col // PACK5_K64
    no = row // PACK5_N256
    ni16 = (row % PACK5_N256) // PACK5_N16
    logical_ni = row % PACK5_N16
    ks = (col % PACK5_K64) // PACK5_K16
    physical_ni = pack5_logical_to_physical_ni(logical_ni)
    ki = col % PACK5_K16
    _, k_outer, n_outer, n16_outer, k16_segment, n16, k_inner = shape
    return (
        ((((((expert * k_outer + ko) * n_outer + no) * n16_outer + ni16)
           * k16_segment + ks) * n16 + physical_ni) * k_inner + ki)
    )


def pack5_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError("pack5_weight expects [expert, n, k]")
    experts, n, k = weight.shape
    if n % PACK5_N256 != 0 or k % PACK5_K64 != 0:
        raise ValueError("pack5_weight expects n divisible by 256 and k divisible by 64")

    view = weight.reshape(
        experts,
        n // PACK5_N256,
        PACK5_N256 // PACK5_N16,
        PACK5_N16,
        k // PACK5_K64,
        PACK5_K64 // PACK5_K16,
        PACK5_K16,
    )
    physical_order = pack5_physical_to_logical_indices(weight.device)
    view = view.index_select(3, physical_order)
    return view.permute(0, 4, 1, 2, 5, 3, 6).contiguous()


def flatten_pack5_weight(weight: torch.Tensor) -> torch.Tensor:
    packed = pack5_weight(weight)
    return packed.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])


def pack5_weight_asm_normal(weight: torch.Tensor) -> torch.Tensor:
    """Pack L2 weights for the isolated normal K3 ASM-pack5 experiment.

    The original K3 ASM epilogue stores accumulator lanes in logical `ni`
    order. The default V3 C/LL pack5 layout uses a transposed physical `ni`
    order and compensates with an accumulator-lane shuffle before store. This
    helper keeps the same 5pack tile nesting but leaves `ni` in plain order so
    the isolated no-tail ASM path can reuse the original store schedule.
    """
    if weight.dim() != 3:
        raise ValueError("pack5_weight_asm_normal expects [expert, n, k]")
    experts, n, k = weight.shape
    if n % PACK5_N256 != 0 or k % PACK5_K64 != 0:
        raise ValueError(
            "pack5_weight_asm_normal expects n divisible by 256 and k divisible by 64"
        )
    view = weight.reshape(
        experts,
        n // PACK5_N256,
        PACK5_N256 // PACK5_N16,
        PACK5_N16,
        k // PACK5_K64,
        PACK5_K64 // PACK5_K16,
        PACK5_K16,
    )
    return view.permute(0, 4, 1, 2, 5, 3, 6).contiguous()


def flatten_pack5_weight_asm_normal(weight: torch.Tensor) -> torch.Tensor:
    packed = pack5_weight_asm_normal(weight)
    return packed.reshape(weight.shape[0], weight.shape[1] * weight.shape[2])


def unpack_pack5_weight(packed: torch.Tensor, *, n: int, k: int) -> torch.Tensor:
    if packed.dim() != 7:
        raise ValueError("unpack_pack5_weight expects the 7D pack5 tensor")
    experts = packed.shape[0]
    expected_shape = pack5_shape(experts, n, k)
    if tuple(packed.shape) != expected_shape:
        raise ValueError(f"invalid pack5 shape: got {tuple(packed.shape)}, expected {expected_shape}")

    view = packed.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    logical_to_physical = torch.empty(PACK5_N16, dtype=torch.long, device=packed.device)
    logical_to_physical[pack5_physical_to_logical_indices(packed.device)] = torch.arange(
        PACK5_N16,
        dtype=torch.long,
        device=packed.device,
    )
    view = view.index_select(3, logical_to_physical)
    return view.reshape(experts, n, k).contiguous()


def _cast_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.dim() != 3:
        raise ValueError("V3 weight transform expects [expert, n, k]")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for V3 FP8 weight transform")
    scale = weight.float().abs().amax(dim=2).clamp(min=1.0e-4) / 448.0
    fp8 = (weight.float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return fp8.contiguous(), scale.contiguous()


def _pack_fp8_weight_and_scale(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8, scale = _cast_weight_to_fp8(weight)
    return flatten_pack5_weight(fp8), scale


def transform_fp8_weights_for_mega_moe_v3_pack5(
    l1_bf16: torch.Tensor,
    l2_bf16: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Transform BF16 L1/L2 weights into V3-owned FP8 pack5 layout.

    This is an offline/test setup helper. Do not call it from the runtime or
    benchmark execution path.
    """
    return _pack_fp8_weight_and_scale(l1_bf16), _pack_fp8_weight_and_scale(l2_bf16)
