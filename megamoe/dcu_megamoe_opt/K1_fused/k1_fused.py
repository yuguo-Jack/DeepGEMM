from __future__ import annotations

from pathlib import Path

import torch

from ..v3_config import normalize_v3_backend, unified_weight_layout_enabled
from . import k1_fused_ext as _ext


THIS_DIR = Path(__file__).resolve().parent

FUSED_L1_ASM_NAME = (
    "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_"
    "MEGAMOE_DISPATCH_PULL_L1"
)
FUSED_L1_ASM_PACK5_NAME = f"{FUSED_L1_ASM_NAME}_PACK5"
FUSED_L1_ASM_PACK5_CO = THIS_DIR / f"{FUSED_L1_ASM_PACK5_NAME}.co"
FUSED_L1_ASM_UNIFIED_PACK5_NAME = f"{FUSED_L1_ASM_NAME}_UNIFIED_PACK5"
FUSED_L1_ASM_UNIFIED_PACK5_CO = THIS_DIR / f"{FUSED_L1_ASM_UNIFIED_PACK5_NAME}.co"

K1_SUPPORTED_RANKS = 8
K1_SUPPORTED_EXPERTS = 256
K1_SUPPORTED_TOPK = 6
K1_SUPPORTED_HIDDEN = 4096
K1_SUPPORTED_ALIGNMENT = 256
K1_SHAPE_CONTRACT = (
    "K1_fused dispatch-pull L1 asm currently supports only ranks=8, "
    "experts=256, local_experts=32, topk=6, hidden=4096, alignment=256, "
    "and 0<=num_tokens_per_rank<=num_max_tokens_per_rank"
)


def _check_fused_l1_shape(
    *,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    alignment: int,
) -> None:
    if (
        int(num_ranks) != K1_SUPPORTED_RANKS
        or int(num_experts) != K1_SUPPORTED_EXPERTS
        or int(num_topk) != K1_SUPPORTED_TOPK
        or int(hidden) != K1_SUPPORTED_HIDDEN
        or int(alignment) != K1_SUPPORTED_ALIGNMENT
        or int(num_tokens) < 0
        or int(num_tokens) > int(num_max_tokens_per_rank)
    ):
        raise ValueError(K1_SHAPE_CONTRACT)


def ensure_fused_l1_asm_pack5_code_object() -> Path:
    co = (
        FUSED_L1_ASM_UNIFIED_PACK5_CO
        if unified_weight_layout_enabled()
        else FUSED_L1_ASM_PACK5_CO
    )
    if not co.exists():
        raise FileNotFoundError(
            f"prebuilt K1 V3 pack5 asm code object not found: {co}. "
            "Rebuild and reinstall the megamoe wheel."
        )
    return co


def load_extension(verbose: bool = False):
    return _ext


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dcu_workspace_offset(num_ranks: int) -> int:
    signal_ptrs_offset = _align(num_ranks * 8, 16)
    runtime_tokens_offset = _align(signal_ptrs_offset + num_ranks * 8, 16)
    return _align(runtime_tokens_offset + 4, 16)


def _symm_x_addr_range(sym_buffer, num_ranks: int, hidden: int) -> tuple[int, int]:
    handle = getattr(sym_buffer, "handle", None)
    ptrs = getattr(handle, "buffer_ptrs", None)
    if ptrs is None:
        raise RuntimeError("sym_buffer.handle.buffer_ptrs is required for graph-safe K1 address setup")
    ptrs = [int(ptr) for ptr in ptrs[: int(num_ranks)] if int(ptr) != 0]
    if not ptrs:
        raise RuntimeError("failed to read symm peer pointers")
    x_offset = _dcu_workspace_offset(int(num_ranks))
    base = min(ptrs) + x_offset
    end = max(ptrs) + x_offset + int(sym_buffer.num_max_tokens_per_rank) * int(hidden)
    return base, end - base


def _cached_symm_x_addr_range(sym_buffer, num_ranks: int, hidden: int) -> tuple[int, int]:
    key = (
        int(sym_buffer.buffer.data_ptr()),
        int(num_ranks),
        int(hidden),
        int(sym_buffer.num_max_tokens_per_rank),
    )
    cached = getattr(sym_buffer, "_k1_symm_x_addr_range_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = _symm_x_addr_range(sym_buffer, num_ranks, hidden)
    setattr(sym_buffer, "_k1_symm_x_addr_range_cache", (key, value))
    return value


def k1_graph_flag_reset_layout(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    l1_rows: int,
    alignment: int = 256,
) -> tuple[int, int, int, int, int, int]:
    ext = load_extension(verbose=False)
    return tuple(
        int(v)
        for v in ext.k1_graph_flag_reset_layout(
            int(num_ranks),
            int(num_experts),
            int(num_max_tokens_per_rank),
            int(num_tokens),
            int(num_topk),
            int(hidden),
            int(l1_rows),
            int(alignment),
        )
    )


def _v3_asm_pack5_weight_view(
    l1_weight_pack5: torch.Tensor,
    l1_scale: torch.Tensor,
    *,
    num_ranks: int,
    num_experts: int,
    hidden: int,
) -> torch.Tensor:
    return l1_weight_pack5.contiguous().view(
        int(num_experts) // int(num_ranks),
        int(l1_scale.size(1)) // 16,
        int(hidden) * 16,
    )


def k1_symm_fused_l1_v3_asm_pack5(
    sym_buffer,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    alignment: int = 256,
    l1_out_workspace: torch.Tensor | None = None,
    cumulative_local_expert_recv_stats: torch.Tensor | None = None,
    force_compact_prebuild: bool = False,
    capacity_num_tokens: int | None = None,
    verbose_build: bool = False,
):
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
    )
    l1_weight_pack5, l1_scale = l1_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_pack5_code_object()
    symm_base_addr, symm_x_span = _cached_symm_x_addr_range(
        sym_buffer, num_ranks, hidden
    )
    return ext.k1_symm_fused_l1_v3_asm_pack5(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        _v3_asm_pack5_weight_view(
            l1_weight_pack5,
            l1_scale,
            num_ranks=num_ranks,
            num_experts=num_experts,
            hidden=hidden,
        ),
        l1_scale.contiguous(),
        int(rank_idx),
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(num_tokens),
        int(num_topk),
        int(hidden),
        int(symm_base_addr),
        int(symm_x_span),
        int(alignment),
        str(code_object),
        l1_out_workspace,
        cumulative_local_expert_recv_stats,
        None,
        bool(force_compact_prebuild),
        -1 if capacity_num_tokens is None else int(capacity_num_tokens),
    )


def k1_symm_fused_l1_v3_asm_pack5_graph(
    sym_buffer,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    graph_max_tokens: int,
    num_topk: int,
    hidden: int,
    runtime_num_tokens: torch.Tensor,
    alignment: int = 256,
    l1_out_workspace: torch.Tensor | None = None,
    verbose_build: bool = False,
):
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=graph_max_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
    )
    l1_weight_pack5, l1_scale = l1_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_pack5_code_object()
    symm_base_addr, symm_x_span = _cached_symm_x_addr_range(
        sym_buffer, num_ranks, hidden
    )
    return ext.k1_symm_fused_l1_v3_asm_pack5(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        _v3_asm_pack5_weight_view(
            l1_weight_pack5,
            l1_scale,
            num_ranks=num_ranks,
            num_experts=num_experts,
            hidden=hidden,
        ),
        l1_scale.contiguous(),
        int(rank_idx),
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(graph_max_tokens),
        int(num_topk),
        int(hidden),
        int(symm_base_addr),
        int(symm_x_span),
        int(alignment),
        str(code_object),
        l1_out_workspace,
        None,
        runtime_num_tokens.contiguous(),
        True,
    )


def k1_symm_fused_l1_v3(
    sym_buffer,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    backend: str,
    alignment: int = 256,
    l1_out_workspace: torch.Tensor | None = None,
    cumulative_local_expert_recv_stats: torch.Tensor | None = None,
    force_compact_prebuild: bool = False,
    capacity_num_tokens: int | None = None,
    ll_block_m: int = 32,
    ll_asm_compatible_layout: bool = False,
    enable_start_rank_barrier: bool = False,
    tail_done_counter: torch.Tensor | None = None,
    graph_runtime_num_tokens_out: torch.Tensor | None = None,
    graph_tail_signal_generation_out: torch.Tensor | None = None,
    verbose_build: bool = False,
):
    backend = normalize_v3_backend(backend)
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
    )
    ext = load_extension(verbose=verbose_build)
    if backend == "normal":
        return k1_symm_fused_l1_v3_asm_pack5(
            sym_buffer,
            l1_weights,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_tokens=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            alignment=alignment,
            l1_out_workspace=l1_out_workspace,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            force_compact_prebuild=force_compact_prebuild,
            capacity_num_tokens=capacity_num_tokens,
            verbose_build=verbose_build,
        )
    l1_weight_pack5, l1_scale = l1_weights
    return ext.k1_symm_fused_l1_v3_pack5(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        l1_weight_pack5.contiguous(),
        l1_scale.contiguous(),
        int(rank_idx),
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(num_tokens),
        int(num_topk),
        int(hidden),
        int(alignment),
        backend,
        l1_out_workspace,
        cumulative_local_expert_recv_stats,
        None,
        int(ll_block_m),
        64,
        bool(ll_asm_compatible_layout),
        -1 if capacity_num_tokens is None else int(capacity_num_tokens),
        bool(enable_start_rank_barrier),
        tail_done_counter,
        graph_runtime_num_tokens_out,
        graph_tail_signal_generation_out,
    )


def k1_symm_fused_l1_v3_graph(
    sym_buffer,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    graph_max_tokens: int,
    num_topk: int,
    hidden: int,
    runtime_num_tokens: torch.Tensor,
    backend: str,
    alignment: int = 256,
    l1_out_workspace: torch.Tensor | None = None,
    ll_block_m: int = 32,
    ll_asm_compatible_layout: bool = False,
    enable_start_rank_barrier: bool = False,
    tail_done_counter: torch.Tensor | None = None,
    graph_runtime_num_tokens_out: torch.Tensor | None = None,
    graph_tail_signal_generation_out: torch.Tensor | None = None,
    verbose_build: bool = False,
):
    backend = normalize_v3_backend(backend)
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=graph_max_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
    )
    ext = load_extension(verbose=verbose_build)
    if backend == "normal":
        return k1_symm_fused_l1_v3_asm_pack5_graph(
            sym_buffer,
            l1_weights,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            num_experts=num_experts,
            graph_max_tokens=graph_max_tokens,
            num_topk=num_topk,
            hidden=hidden,
            runtime_num_tokens=runtime_num_tokens,
            alignment=alignment,
            l1_out_workspace=l1_out_workspace,
            verbose_build=verbose_build,
        )
    l1_weight_pack5, l1_scale = l1_weights
    return ext.k1_symm_fused_l1_v3_pack5(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        l1_weight_pack5.contiguous(),
        l1_scale.contiguous(),
        int(rank_idx),
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(graph_max_tokens),
        int(num_topk),
        int(hidden),
        int(alignment),
        backend,
        l1_out_workspace,
        None,
        runtime_num_tokens.contiguous(),
        int(ll_block_m),
        64,
        bool(ll_asm_compatible_layout),
        -1,
        bool(enable_start_rank_barrier),
        tail_done_counter,
        graph_runtime_num_tokens_out,
        graph_tail_signal_generation_out,
    )
