from __future__ import annotations

from pathlib import Path

import torch

from ..v3_config import (
    STAGED_PACK5_SHAPE_CONTRACT,
    SUPPORTED_STAGED_EP_RANKS,
    V3_BACKEND_NORMAL,
    V3_QUANT_FP8,
    V3_QUANT_INT8,
    normalize_v3_backend,
    normalize_v3_quant,
    staged_pack5_k1_shape_supported,
    staged_v3_capability_supported,
)
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
FUSED_L1_INT8_ASM_NAME = (
    "DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_"
    "MEGAMOE_DISPATCH_PULL_L1"
)
FUSED_L1_INT8_ASM_PACK5_CO = THIS_DIR / f"{FUSED_L1_INT8_ASM_NAME}_PACK5.co"
PRO_LL_MASKED_L1_ASM_NAME = (
    "DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_"
    "MT256x64x128_WGM8_GROUPGEMM_MASKED"
)
PRO_LL_MASKED_L1_ASM_CO = (
    THIS_DIR / "deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co"
)

K1_SUPPORTED_RANKS = SUPPORTED_STAGED_EP_RANKS
K1_SUPPORTED_ALIGNMENT = 256
K1_SHAPE_CONTRACT = (
    f"{STAGED_PACK5_SHAPE_CONTRACT}, K1 L1 output features must equal "
    "2 * intermediate, alignment=256, and "
    "0<=num_tokens_per_rank<=num_max_tokens_per_rank"
)


def _check_fused_l1_shape(
    *,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    l1_rows: int,
    alignment: int,
    quant_mode: str = V3_QUANT_FP8,
) -> None:
    quant_mode = normalize_v3_quant(quant_mode)
    shape_supported = (
        staged_v3_capability_supported(
            quant=quant_mode,
            backend=V3_BACKEND_NORMAL,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=l1_rows // 2,
        )
        if quant_mode == V3_QUANT_INT8 and l1_rows % 2 == 0
        else staged_pack5_k1_shape_supported(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            l1_rows=l1_rows,
        )
    )
    if (
        not shape_supported
        or int(alignment) != K1_SUPPORTED_ALIGNMENT
        or int(num_tokens) < 0
        or int(num_tokens) > int(num_max_tokens_per_rank)
    ):
        raise ValueError(K1_SHAPE_CONTRACT)


def ensure_fused_l1_asm_pack5_code_object(
    use_unified_weight_layout: bool = False,
    int8_compute: bool = False,
) -> Path:
    if int8_compute and use_unified_weight_layout:
        raise ValueError("YGZP INT8 K1 supports normal non-unified PACK5 only")
    if int8_compute:
        co = FUSED_L1_INT8_ASM_PACK5_CO
    else:
        co = (
            FUSED_L1_ASM_UNIFIED_PACK5_CO
            if use_unified_weight_layout
            else FUSED_L1_ASM_PACK5_CO
        )
    if not co.exists():
        raise FileNotFoundError(
            f"prebuilt K1 V3 pack5 asm code object not found: {co}. "
            "Rebuild and reinstall the megamoe wheel."
        )
    return co


def ensure_pro_ll_masked_l1_asm_code_object() -> Path:
    if not PRO_LL_MASKED_L1_ASM_CO.exists():
        raise FileNotFoundError(
            f"prebuilt Pro LL masked K1 asm code object not found: {PRO_LL_MASKED_L1_ASM_CO}. "
            "Rebuild and reinstall the megamoe wheel."
        )
    return PRO_LL_MASKED_L1_ASM_CO


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
    use_unified_weight_layout: bool = False,
    quant_mode: str = V3_QUANT_FP8,
    verbose_build: bool = False,
    return_active_tiles_host_hint: bool = False,
):
    l1_weight_pack5, l1_scale = l1_weights
    quant_mode = normalize_v3_quant(quant_mode)
    if return_active_tiles_host_hint and quant_mode != V3_QUANT_INT8:
        raise ValueError("active_tiles host hint is available for INT8 K1 only")
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        l1_rows=int(l1_scale.size(1)),
        alignment=alignment,
        quant_mode=quant_mode,
    )
    expected_dtype = torch.int8 if quant_mode == V3_QUANT_INT8 else torch.float8_e4m3fn
    if l1_weight_pack5.dtype != expected_dtype:
        raise ValueError(f"K1 {quant_mode} weight must have {expected_dtype} dtype")
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_pack5_code_object(
        use_unified_weight_layout=use_unified_weight_layout,
        int8_compute=quant_mode == V3_QUANT_INT8,
    )
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
        bool(return_active_tiles_host_hint),
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
    use_unified_weight_layout: bool = False,
    verbose_build: bool = False,
):
    l1_weight_pack5, l1_scale = l1_weights
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=graph_max_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        l1_rows=int(l1_scale.size(1)),
        alignment=alignment,
    )
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_pack5_code_object(
        use_unified_weight_layout=use_unified_weight_layout
    )
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


def k1_ll_masked_groupgemm(
    input_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_masked: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    *,
    verbose_build: bool = False,
):
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_pro_ll_masked_l1_asm_code_object()
    return ext.k1_ll_masked_groupgemm_pack5(
        input_fp8.contiguous(),
        input_scale.contiguous(),
        weight_masked.contiguous(),
        weight_scale.contiguous(),
        output,
        masked_m.contiguous(),
        int(expected_m_per_group),
        str(code_object),
    )


def k1_ll_masked_prepare_compact_active(
    staged_x: torch.Tensor,
    staged_x_scale: torch.Tensor,
    route_weights: torch.Tensor,
    row_combine_ptrs: torch.Tensor,
    actual_m: torch.Tensor,
    compact_staged_x: torch.Tensor,
    compact_staged_x_scale: torch.Tensor,
    compact_route_weights: torch.Tensor,
    compact_row_combine_ptrs: torch.Tensor,
    compact_m: torch.Tensor,
    rows_per_expert: int,
    *,
    verbose_build: bool = False,
):
    ext = load_extension(verbose=verbose_build)
    return ext.k1_ll_masked_prepare_compact_active_pack5(
        staged_x.contiguous(),
        staged_x_scale.contiguous(),
        route_weights.contiguous(),
        row_combine_ptrs.contiguous(),
        actual_m.contiguous(),
        compact_staged_x,
        compact_staged_x_scale,
        compact_route_weights,
        compact_row_combine_ptrs,
        compact_m,
        int(rows_per_expert),
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
    use_unified_weight_layout: bool = False,
    ll_block_m: int = 32,
    enable_start_rank_barrier: bool = False,
    tail_done_counter: torch.Tensor | None = None,
    graph_runtime_num_tokens_out: torch.Tensor | None = None,
    graph_tail_signal_generation_out: torch.Tensor | None = None,
    ll_stage_only: bool = False,
    ll_return_staged: bool = False,
    quant_mode: str = V3_QUANT_FP8,
    verbose_build: bool = False,
    return_active_tiles_host_hint: bool = False,
):
    backend = normalize_v3_backend(backend)
    quant_mode = normalize_v3_quant(quant_mode)
    if quant_mode == V3_QUANT_INT8 and backend != V3_BACKEND_NORMAL:
        raise ValueError("YGZP INT8 K1 supports normal backend only")
    if return_active_tiles_host_hint and backend != V3_BACKEND_NORMAL:
        raise ValueError("active_tiles host hint is available on normal K1 only")
    if return_active_tiles_host_hint and quant_mode != V3_QUANT_INT8:
        raise ValueError("active_tiles host hint is available for INT8 K1 only")
    l1_weight_pack5, l1_scale = l1_weights
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        l1_rows=int(l1_scale.size(1)),
        alignment=alignment,
        quant_mode=quant_mode,
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
            use_unified_weight_layout=use_unified_weight_layout,
            quant_mode=quant_mode,
            verbose_build=verbose_build,
            return_active_tiles_host_hint=return_active_tiles_host_hint,
        )
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
        -1 if capacity_num_tokens is None else int(capacity_num_tokens),
        bool(enable_start_rank_barrier),
        tail_done_counter,
        graph_runtime_num_tokens_out,
        graph_tail_signal_generation_out,
        bool(ll_stage_only),
        bool(ll_return_staged),
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
    capacity_num_tokens: int | None = None,
    alignment: int = 256,
    l1_out_workspace: torch.Tensor | None = None,
    use_unified_weight_layout: bool = False,
    ll_block_m: int = 32,
    enable_start_rank_barrier: bool = False,
    tail_done_counter: torch.Tensor | None = None,
    graph_runtime_num_tokens_out: torch.Tensor | None = None,
    graph_tail_signal_generation_out: torch.Tensor | None = None,
    ll_stage_only: bool = False,
    ll_return_staged: bool = False,
    verbose_build: bool = False,
):
    backend = normalize_v3_backend(backend)
    l1_weight_pack5, l1_scale = l1_weights
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=graph_max_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        l1_rows=int(l1_scale.size(1)),
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
            use_unified_weight_layout=use_unified_weight_layout,
            verbose_build=verbose_build,
        )
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
        -1 if capacity_num_tokens is None else int(capacity_num_tokens),
        bool(enable_start_rank_barrier),
        tail_done_counter,
        graph_runtime_num_tokens_out,
        graph_tail_signal_generation_out,
        bool(ll_stage_only),
        bool(ll_return_staged),
    )
