from __future__ import annotations

from pathlib import Path

import torch

from . import k1_fused_ext as _ext


THIS_DIR = Path(__file__).resolve().parent

FUSED_L1_ASM_NAME = (
    "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_"
    "MEGAMOE_DISPATCH_PULL_L1"
)
FUSED_L1_ASM_CO = THIS_DIR / f"{FUSED_L1_ASM_NAME}.co"

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


def ensure_fused_l1_asm_code_object() -> Path:
    if not FUSED_L1_ASM_CO.exists():
        raise FileNotFoundError(
            f"prebuilt K1 asm code object not found: {FUSED_L1_ASM_CO}. "
            "Rebuild and reinstall the megamoe wheel."
        )
    return FUSED_L1_ASM_CO


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


def k1_symm_fused_l1_asm(
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
    l1_weight, l1_scale = l1_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_code_object()
    symm_base_addr, symm_x_span = _cached_symm_x_addr_range(
        sym_buffer, num_ranks, hidden
    )
    return ext.k1_symm_fused_l1(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        l1_weight.contiguous(),
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
    )


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


def k1_symm_fused_l1_asm_graph(
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
    l1_weight, l1_scale = l1_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_code_object()
    symm_base_addr, symm_x_span = _cached_symm_x_addr_range(
        sym_buffer, num_ranks, hidden
    )
    return ext.k1_symm_fused_l1(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        l1_weight.contiguous(),
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
