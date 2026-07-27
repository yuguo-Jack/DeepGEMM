"""V3 pack5 weight layout helpers for offline fixtures.

These helpers are intentionally not used by the opt runtime path. V3
execution expects callers/tests to provide weights that are already in this
pack5 layout. Callers select the runtime layout by passing either a single
"unified" weight dict entry or a backend-specific entry such as "normal".
"""

from __future__ import annotations

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


def _repack_grouped_weight_inplace_(
    weight: torch.Tensor,
    pack_chunk,
    *,
    chunk_experts: int,
    function_name: str,
) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError(f"{function_name} expects [expert, n, k]")
    if not isinstance(chunk_experts, int) or isinstance(chunk_experts, bool):
        raise TypeError(f"{function_name} expects chunk_experts to be an integer")
    if chunk_experts <= 0:
        raise ValueError(f"{function_name} expects chunk_experts to be positive")
    if not weight.is_contiguous():
        raise ValueError(f"{function_name} requires contiguous input storage")
    if weight.storage_offset() != 0:
        raise ValueError(f"{function_name} requires zero storage_offset")

    expected_nbytes = weight.numel() * weight.element_size()
    if weight.untyped_storage().nbytes() != expected_nbytes:
        raise ValueError(
            f"{function_name} requires independent storage covering exactly the input tensor"
        )

    experts = weight.shape[0]
    with torch.no_grad():
        for start in range(0, experts, chunk_experts):
            end = min(start + chunk_experts, experts)
            packed = pack_chunk(weight[start:end])
            weight[start:end].reshape(-1).copy_(packed.reshape(-1))
            del packed
    return weight.detach()


def flatten_pack5_weight_inplace_(
    weight: torch.Tensor,
    *,
    chunk_experts: int = 1,
) -> torch.Tensor:
    """Repack LL/unified pack5 weights into their existing storage.

    The returned ``[expert, n * k]`` tensor is a detached view of ``weight`` and
    has the same storage pointer. This destructive post-load helper is intended
    for inference weights that own independent contiguous storage.
    """
    if weight.dim() != 3:
        raise ValueError("flatten_pack5_weight_inplace_ expects [expert, n, k]")
    experts, n, k = weight.shape
    if n % PACK5_N256 != 0 or k % PACK5_K64 != 0:
        raise ValueError(
            "flatten_pack5_weight_inplace_ expects n divisible by 256 and k divisible by 64"
        )
    repacked = _repack_grouped_weight_inplace_(
        weight,
        pack5_weight,
        chunk_experts=chunk_experts,
        function_name="flatten_pack5_weight_inplace_",
    )
    return repacked.reshape(experts, n * k)


def pack5_weight_asm_normal(weight: torch.Tensor) -> torch.Tensor:
    """Pack weights in plain ni order for the isolated normal ASM path."""
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


def flatten_pack5_weight_asm_normal_inplace_(
    weight: torch.Tensor,
    *,
    chunk_experts: int = 1,
) -> torch.Tensor:
    """Repack normal-ASM pack5 weights into their existing storage."""
    if weight.dim() != 3:
        raise ValueError(
            "flatten_pack5_weight_asm_normal_inplace_ expects [expert, n, k]"
        )
    experts, n, k = weight.shape
    if n % PACK5_N256 != 0 or k % PACK5_K64 != 0:
        raise ValueError(
            "flatten_pack5_weight_asm_normal_inplace_ expects n divisible by 256 "
            "and k divisible by 64"
        )
    repacked = _repack_grouped_weight_inplace_(
        weight,
        pack5_weight_asm_normal,
        chunk_experts=chunk_experts,
        function_name="flatten_pack5_weight_asm_normal_inplace_",
    )
    return repacked.reshape(experts, n * k)
