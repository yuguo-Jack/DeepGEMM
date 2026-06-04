"""Python boundary for V2 K3 fused kernels."""

from __future__ import annotations

from typing import Optional

import torch

_K3_EXT = None


def _load_k3_ext(verbose: bool = False):
    global _K3_EXT
    if _K3_EXT is not None:
        return _K3_EXT
    del verbose
    try:
        from . import k3_fused_ext

        _K3_EXT = k3_fused_ext
        return _K3_EXT
    except Exception as exc:
        raise RuntimeError(
            "V2 K3 extension is not built. Build the package extension first, "
            "for example: DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace"
        ) from exc


def k3_l2_combine_fused_v2(
    y: torch.Tensor,
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    sym_buffer,
    *,
    route_scratch: torch.Tensor,
    l2_workspace: Optional[torch.Tensor] = None,
    problem_size: Optional[torch.Tensor] = None,
    row_expert: Optional[torch.Tensor] = None,
    grid_barrier: Optional[torch.Tensor] = None,
    row_output_ptrs: Optional[torch.Tensor] = None,
    local_topk_mask: Optional[torch.Tensor] = None,
    tail_tokens: Optional[torch.Tensor] = None,
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
    k3_copy_workers: int = 16,
    backend: str,
    activation_clamp: Optional[float],
    verbose_build: bool = False,
) -> None:
    """Run V2 K3 L2 groupgemm plus combine/reduce in one fused path."""
    if backend not in {"ll", "normal"}:
        raise ValueError("V2 K3 backend must be 'll' or 'normal'")
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
        raise ValueError(f"missing V2 K3 launch metadata: {', '.join(missing)}")
    if grid_barrier is None or row_output_ptrs is None:
        raise ValueError("grid_barrier and row_output_ptrs are required")
    if not hasattr(sym_buffer, "buffer"):
        raise TypeError("sym_buffer must be a megamoe SymmBuffer-like object")

    del route_scratch, activation_clamp
    l2_weight, l2_scale = l2_weights
    ext = _load_k3_ext(verbose=verbose_build)
    if backend == "ll":
        if problem_size is None:
            raise ValueError("problem_size is required for V2 K3 ll")
        ext.launch_k3_ll_rowptr_tail_reduce(
            y,
            act_fp8,
            l2_weight.contiguous(),
            act_scale,
            l2_scale.contiguous(),
            problem_size,
            sym_buffer.buffer,
            grid_barrier,
            row_output_ptrs,
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
        return

    if (
        l2_workspace is None
        or row_expert is None
        or local_topk_mask is None
        or tail_tokens is None
    ):
        raise ValueError(
            "l2_workspace, row_expert, local_topk_mask, and tail_tokens are "
            "required for V2 K3 normal"
        )
    ext.launch_k3_normal_copy_stage_tail_reduce(
        l2_workspace,
        act_fp8,
        l2_weight.contiguous(),
        act_scale,
        l2_scale.contiguous(),
        row_expert,
        sym_buffer.buffer,
        grid_barrier,
        row_output_ptrs,
        local_topk_mask,
        y,
        tail_tokens,
        int(rank_idx),
        int(num_ranks),
        int(num_global_experts),
        int(num_max_tokens_per_rank),
        int(num_topk),
        int(num_tokens),
        int(num_tokens),
        int(rows_aligned_per_expert),
        int(valid_rows_per_expert),
        int(k3_copy_workers),
        int(epoch),
    )
