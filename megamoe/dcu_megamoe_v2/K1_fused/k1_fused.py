"""Python boundary for V2 K1 fused kernels."""

from __future__ import annotations

from typing import Optional

import torch

_K1_EXT = None


def _load_k1_ext(verbose: bool = False):
    global _K1_EXT
    if _K1_EXT is not None:
        return _K1_EXT
    del verbose
    try:
        from . import k1_fused_ext

        _K1_EXT = k1_fused_ext
        return _K1_EXT
    except Exception as exc:
        raise RuntimeError(
            "V2 K1 extension is not built. Build the package extension first, "
            "for example: DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace"
        ) from exc


def k1_dispatch_pull_l1_fused_v2(
    l1_out: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
    *,
    route_scratch: torch.Tensor,
    staged_x: Optional[torch.Tensor] = None,
    staged_x_scale: Optional[torch.Tensor] = None,
    problem_size: Optional[torch.Tensor] = None,
    row_expert: Optional[torch.Tensor] = None,
    route_weights: Optional[torch.Tensor] = None,
    output_index: Optional[torch.Tensor] = None,
    row_combine_ptrs: Optional[torch.Tensor] = None,
    local_topk_mask: Optional[torch.Tensor] = None,
    tail_tokens: Optional[torch.Tensor] = None,
    grid_barrier: Optional[torch.Tensor] = None,
    route_scratch_i32: Optional[torch.Tensor] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    num_tokens: int,
    num_ranks: Optional[int] = None,
    num_global_experts: Optional[int] = None,
    num_max_tokens_per_rank: Optional[int] = None,
    num_topk: Optional[int] = None,
    rank_idx: Optional[int] = None,
    rows_aligned_per_expert: Optional[int] = None,
    valid_rows_per_expert: Optional[int] = None,
    epoch: int = 1,
    ll_block_m: int = 32,
    ll_cus: int = 64,
    backend: str,
    verbose_build: bool = False,
) -> torch.Tensor:
    """Run V2 K1 dispatch-pull plus L1 groupgemm.

    The production implementation will dispatch to the low-latency or normal C
    pack5 extension entry points split out of the prototype harness.
    """
    if backend not in {"ll", "normal"}:
        raise ValueError("V2 K1 backend must be 'll' or 'normal'")
    required_ints = {
        "num_ranks": num_ranks,
        "num_global_experts": num_global_experts,
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "num_topk": num_topk,
        "rank_idx": rank_idx,
        "rows_aligned_per_expert": rows_aligned_per_expert,
        "valid_rows_per_expert": valid_rows_per_expert,
    }
    missing = [name for name, value in required_ints.items() if value is None]
    if missing:
        raise ValueError(f"missing V2 K1 launch metadata: {', '.join(missing)}")
    if staged_x is None or staged_x_scale is None or grid_barrier is None:
        raise ValueError("staged_x, staged_x_scale, and grid_barrier are required")
    if not hasattr(sym_buffer, "buffer"):
        raise TypeError("sym_buffer must be a megamoe SymmBuffer-like object")

    del route_scratch
    l1_weight, l1_scale = l1_weights
    ext = _load_k1_ext(verbose=verbose_build)
    if backend == "ll":
        if problem_size is None or route_scratch_i32 is None:
            raise ValueError("problem_size and route_scratch_i32 are required for V2 K1 ll")
        if (
            route_weights is None
            or row_expert is None
            or output_index is None
            or row_combine_ptrs is None
            or local_topk_mask is None
            or tail_tokens is None
        ):
            raise ValueError(
                "route_weights, row_expert, output_index, row_combine_ptrs, "
                "local_topk_mask, and tail_tokens are required for V2 K1 ll metadata"
            )
        stats_tensor = (
            cumulative_local_expert_recv_stats
            if cumulative_local_expert_recv_stats is not None
            else route_scratch_i32[:0]
        )
        ext.launch_k1_ll_symm_stage(
            l1_out,
            staged_x,
            l1_weight.contiguous(),
            staged_x_scale,
            l1_scale.contiguous(),
            problem_size,
            sym_buffer.buffer,
            route_scratch_i32,
            grid_barrier,
            route_weights,
            row_expert,
            output_index,
            row_combine_ptrs,
            local_topk_mask,
            tail_tokens,
            stats_tensor,
            int(rank_idx),
            int(num_ranks),
            int(num_global_experts),
            int(num_max_tokens_per_rank),
            int(num_topk),
            int(num_tokens),
            int(rows_aligned_per_expert),
            int(valid_rows_per_expert),
            int(ll_block_m),
            int(ll_cus),
        )
    else:
        if (
            row_expert is None
            or route_scratch_i32 is None
            or route_weights is None
            or output_index is None
            or row_combine_ptrs is None
            or local_topk_mask is None
            or tail_tokens is None
        ):
            raise ValueError(
                "row_expert, route_scratch_i32, route_weights, output_index, "
                "row_combine_ptrs, local_topk_mask, and tail_tokens are "
                "required for V2 K1 normal metadata"
            )
        stats_tensor = (
            cumulative_local_expert_recv_stats
            if cumulative_local_expert_recv_stats is not None
            else route_scratch_i32[:0]
        )
        ext.launch_k1_normal_symm_stage(
            l1_out,
            staged_x,
            l1_weight.contiguous(),
            staged_x_scale,
            l1_scale.contiguous(),
            row_expert,
            sym_buffer.buffer,
            route_scratch_i32,
            grid_barrier,
            route_weights,
            output_index,
            row_combine_ptrs,
            local_topk_mask,
            tail_tokens,
            stats_tensor,
            int(rank_idx),
            int(num_ranks),
            int(num_global_experts),
            int(num_max_tokens_per_rank),
            int(num_topk),
            int(num_tokens),
            int(rows_aligned_per_expert),
            int(valid_rows_per_expert),
            int(epoch),
        )
    return l1_out
