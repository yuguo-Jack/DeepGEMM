from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .dcu_megamoe_large_opt.K1_fused.k1_fused import (
    k1_graph_flag_reset_layout,
    k1_symm_fused_l1_asm,
    k1_symm_fused_l1_asm_graph,
)
from .dcu_megamoe_large_opt.K2_fused.k2_fused import swiglu_quant_channelwise_out
from .dcu_megamoe_large_opt.K3_fused.k3_fused import (
    build_asm_tail_signal_addrs,
    k3_l2_fused_asm_to_combine,
    rank_barrier,
    reduce_local_combine,
    reduce_local_combine_graph,
)


K_DCU_ROUTE_TILE_M = 32
K_PROB_STORAGE_BYTES = 256
K_DTYPE_SIZES = {
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float8_e4m3fn: 1,
    torch.uint8: 1,
    torch.int32: 4,
    torch.int64: 8,
}


@dataclass
class _RouteScratchViews:
    k1_active_tiles: torch.Tensor
    l1_out: torch.Tensor
    act_fp8: torch.Tensor
    act_scale: torch.Tensor
    k3_prob_storage: torch.Tensor
    asm_tail_done_counter: torch.Tensor
    asm_tail_signal_addrs: torch.Tensor


@dataclass
class _LargeOptState:
    scratch: _RouteScratchViews
    empty_bf16: torch.Tensor
    asm_tail_signal_addrs_ready: bool = False


_LARGE_OPT_FORCE_VALUES = {"1", "true", "yes", "on", "large_opt", "3stage"}
_LARGE_OPT_AUTO_VALUES = {"auto", "threshold", "adaptive"}


def env_enabled() -> bool:
    value = os.getenv("MEGAMOE_DCU_USE_LARGE_OPT_3STAGE", "auto").strip().lower()
    return value in _LARGE_OPT_FORCE_VALUES or value in _LARGE_OPT_AUTO_VALUES


def k3_tail_reduce_enabled() -> bool:
    value = os.getenv("K3_USE_ASM_TAIL_REDUCE", "0").strip().lower()
    return value in {"1", "true", "yes", "on", "tail-reduce"}


def k2_skip_inactive_rows_enabled(num_tokens: int) -> bool:
    min_tokens = int(os.getenv("K2_SKIP_INACTIVE_ROWS_MIN_TOKENS", "1536"))
    return num_tokens >= min_tokens


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _route_task_workspace_bytes(*, num_ranks: int, num_experts: int, num_max_tokens: int) -> int:
    local_experts = num_experts // num_ranks
    task_capacity_per_expert = num_ranks * num_ranks * num_max_tokens
    bytes_ = local_experts * 4
    bytes_ = _align(bytes_, 16)
    bytes_ += local_experts * task_capacity_per_expert * 4
    return _align(bytes_, 16)


def _route_scratch_capacity_tiles(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
) -> int:
    local_experts = num_experts // num_ranks
    max_route_tasks = num_ranks * num_max_tokens * num_topk
    return local_experts + (max_route_tasks + K_DCU_ROUTE_TILE_M - 1) // K_DCU_ROUTE_TILE_M


def _route_tile_offsets(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> tuple[int, int, int, int, int]:
    max_route_tiles = _route_scratch_capacity_tiles(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
    )
    capacity_rows = max_route_tiles * K_DCU_ROUTE_TILE_M
    offset = 0
    offset += capacity_rows * hidden
    offset = _align(offset, 16)
    act_bf16_offset = offset
    offset += capacity_rows * intermediate_hidden * K_DTYPE_SIZES[torch.bfloat16]
    offset = _align(offset, 16)
    act_fp8_offset = offset
    offset += capacity_rows * intermediate_hidden * K_DTYPE_SIZES[torch.float8_e4m3fn]
    offset = _align(offset, 16)
    act_scale_offset = offset
    offset += capacity_rows * K_DTYPE_SIZES[torch.float32]
    offset = _align(offset, 16)
    act_chunk_amax_offset = offset
    return capacity_rows, act_bf16_offset, act_fp8_offset, act_scale_offset, act_chunk_amax_offset


def _route_scratch_tensor(
    route_scratch: torch.Tensor,
    *,
    byte_offset: int,
    byte_capacity: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    dtype_size = K_DTYPE_SIZES[dtype]
    numel = 1
    for dim in shape:
        numel *= dim
    required_bytes = numel * dtype_size
    if byte_offset % dtype_size != 0:
        raise RuntimeError(f"route_scratch offset {byte_offset} is not aligned for {dtype}")
    if required_bytes > byte_capacity:
        raise RuntimeError(f"route_scratch view {shape} exceeds reserved {dtype} region")
    if byte_offset + required_bytes > route_scratch.numel():
        raise RuntimeError("route_scratch is too small for large-opt scratch views")
    return route_scratch.narrow(0, byte_offset, required_bytes).view(dtype).view(shape)


def _route_scratch_views(
    sym_buffer,
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> _RouteScratchViews:
    route_scratch = sym_buffer.route_scratch
    if route_scratch.dtype != torch.int8 or not route_scratch.is_contiguous():
        raise RuntimeError("sym_buffer.route_scratch must be contiguous int8")

    num_max_tokens = int(sym_buffer.num_max_tokens_per_rank)
    route_base = _route_task_workspace_bytes(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
    )
    capacity_rows, act_bf16_offset, act_fp8_offset, act_scale_offset, act_chunk_amax_offset = (
        _route_tile_offsets(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_max_tokens=num_max_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )
    )
    l1_cols = intermediate_hidden * 2
    l1_capacity_bytes = act_fp8_offset - act_bf16_offset
    l1_capacity_rows = l1_capacity_bytes // (l1_cols * K_DTYPE_SIZES[torch.bfloat16])
    act_fp8_capacity_bytes = act_scale_offset - act_fp8_offset
    act_scale_capacity_bytes = act_chunk_amax_offset - act_scale_offset
    prob_offset = route_base + act_chunk_amax_offset
    tail_done_offset = _align(prob_offset + K_PROB_STORAGE_BYTES, K_DTYPE_SIZES[torch.int32])
    tail_signal_addrs_offset = _align(
        tail_done_offset + K_DTYPE_SIZES[torch.int32],
        K_DTYPE_SIZES[torch.int64],
    )

    return _RouteScratchViews(
        k1_active_tiles=_route_scratch_tensor(
            route_scratch,
            byte_offset=64 * K_DTYPE_SIZES[torch.int32],
            byte_capacity=K_DTYPE_SIZES[torch.int32],
            dtype=torch.int32,
            shape=(1,),
        ),
        l1_out=_route_scratch_tensor(
            route_scratch,
            byte_offset=route_base + act_bf16_offset,
            byte_capacity=l1_capacity_bytes,
            dtype=torch.bfloat16,
            shape=(l1_capacity_rows, l1_cols),
        ),
        act_fp8=_route_scratch_tensor(
            route_scratch,
            byte_offset=route_base + act_fp8_offset,
            byte_capacity=act_fp8_capacity_bytes,
            dtype=torch.float8_e4m3fn,
            shape=(capacity_rows, intermediate_hidden),
        ),
        act_scale=_route_scratch_tensor(
            route_scratch,
            byte_offset=route_base + act_scale_offset,
            byte_capacity=act_scale_capacity_bytes,
            dtype=torch.float32,
            shape=(capacity_rows,),
        ),
        k3_prob_storage=_route_scratch_tensor(
            route_scratch,
            byte_offset=prob_offset,
            byte_capacity=K_PROB_STORAGE_BYTES,
            dtype=torch.uint8,
            shape=(K_PROB_STORAGE_BYTES,),
        ),
        asm_tail_done_counter=_route_scratch_tensor(
            route_scratch,
            byte_offset=tail_done_offset,
            byte_capacity=K_DTYPE_SIZES[torch.int32],
            dtype=torch.int32,
            shape=(1,),
        ),
        asm_tail_signal_addrs=_route_scratch_tensor(
            route_scratch,
            byte_offset=tail_signal_addrs_offset,
            byte_capacity=16 * K_DTYPE_SIZES[torch.int64],
            dtype=torch.int64,
            shape=(16,),
        ),
    )


def _state(
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    init_tail_reduce: bool,
    verbose_build: bool,
) -> _LargeOptState:
    device = sym_buffer.buffer.device
    key = (
        int(device.index or 0),
        int(sym_buffer.route_scratch.data_ptr()),
        int(sym_buffer.num_max_tokens_per_rank),
        rank_idx,
        num_ranks,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    cached = getattr(sym_buffer, "_large_opt_3stage_state", None)
    if cached is not None and cached[0] == key:
        state = cached[1]
        if init_tail_reduce and not state.asm_tail_signal_addrs_ready:
            build_asm_tail_signal_addrs(
                sym_buffer,
                rank_idx=rank_idx,
                num_ranks=num_ranks,
                out=state.scratch.asm_tail_signal_addrs,
                verbose_build=verbose_build,
            )
            state.asm_tail_signal_addrs_ready = True
        return state

    state = _LargeOptState(
        scratch=_route_scratch_views(
            sym_buffer,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        ),
        empty_bf16=sym_buffer.buffer[:0].view(torch.bfloat16),
    )
    if init_tail_reduce:
        build_asm_tail_signal_addrs(
            sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            out=state.scratch.asm_tail_signal_addrs,
            verbose_build=verbose_build,
        )
        state.asm_tail_signal_addrs_ready = True
    setattr(sym_buffer, "_large_opt_3stage_state", (key, state))
    return state


def prepare_large_opt_3stage(sym_buffer, *, verbose_build: bool = False) -> None:
    if not env_enabled():
        return
    _check_shape(
        num_ranks=sym_buffer.group.size(),
        num_experts=sym_buffer.num_experts,
        num_topk=sym_buffer.num_topk,
        hidden=sym_buffer.hidden,
        intermediate_hidden=sym_buffer.intermediate_hidden,
        num_tokens=1,
        num_max_tokens_per_rank=int(sym_buffer.num_max_tokens_per_rank),
    )
    _state(
        sym_buffer,
        rank_idx=sym_buffer.group.rank(),
        num_ranks=sym_buffer.group.size(),
        num_experts=sym_buffer.num_experts,
        num_topk=sym_buffer.num_topk,
        hidden=sym_buffer.hidden,
        intermediate_hidden=sym_buffer.intermediate_hidden,
        init_tail_reduce=k3_tail_reduce_enabled(),
        verbose_build=verbose_build,
    )


def _check_shape(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_tokens: int,
    num_max_tokens_per_rank: int,
) -> None:
    if (
        num_ranks != 8
        or num_experts != 256
        or num_topk != 6
        or hidden != 4096
        or intermediate_hidden != 2048
        or num_tokens <= 0
        or num_tokens > num_max_tokens_per_rank
    ):
        raise ValueError(
            "MEGAMOE_DCU_USE_LARGE_OPT_3STAGE supports only DeepSeek-V4-Flash "
            "EP8 shape: experts=256, topk=6, hidden=4096, intermediate=2048, "
            "and 0<num_tokens_per_rank<=num_max_tokens_per_rank"
        )


def fp8_mega_moe_large_opt_3stage(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    activation_clamp: Optional[float],
    fast_math: bool,
) -> None:
    if not fast_math:
        raise RuntimeError("large-opt DCU MegaMoE path requires fast_math")

    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    num_tokens = int(y.size(0))
    hidden = int(y.size(1))
    intermediate_hidden = int(l1_scale.size(1) // 2)
    num_max_tokens_per_rank = int(sym_buffer.num_max_tokens_per_rank)
    _check_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_tokens=num_tokens,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
    )

    alignment = 256
    verbose_build = os.getenv("MEGAMOE_DCU_LARGE_OPT_VERBOSE_BUILD", "0") == "1"
    use_tail_reduce = k3_tail_reduce_enabled()
    state = _state(
        sym_buffer,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        init_tail_reduce=use_tail_reduce,
        verbose_build=verbose_build,
    )

    # Make symmetric-buffer input copies visible to the asm dispatch-pull stage.
    rank_barrier(
        sym_buffer,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        asm_done_counter=state.scratch.asm_tail_done_counter if use_tail_reduce else None,
        reset_tail_signal_slots=use_tail_reduce,
        verbose_build=verbose_build,
    )

    l1_out, route_weights, m_indices, output_index, row_combine_ptrs = k1_symm_fused_l1_asm(
        sym_buffer,
        (l1_weight, l1_scale),
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
        l1_out_workspace=state.scratch.l1_out,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        verbose_build=verbose_build,
    )
    rows = int(l1_out.size(0))

    act_fp8 = state.scratch.act_fp8[:rows]
    act_scale = state.scratch.act_scale[:rows]
    swiglu_quant_channelwise_out(
        l1_out,
        route_weights,
        act_fp8,
        act_scale,
        state.empty_bf16,
        num_per_channels=intermediate_hidden,
        output_bf16=False,
        clamp_value=activation_clamp,
        row_combine_ptrs=(
            row_combine_ptrs if k2_skip_inactive_rows_enabled(num_tokens) else None
        ),
        verbose_build=verbose_build,
    )

    if use_tail_reduce:
        total_wgs = ((rows + 255) // 256) * ((hidden + 255) // 256)
        k3_l2_fused_asm_to_combine(
            act_fp8,
            act_scale,
            m_indices,
            (l2_weight, l2_scale),
            row_combine_ptrs,
            asm_done_counter=state.scratch.asm_tail_done_counter,
            asm_signal_addrs=state.scratch.asm_tail_signal_addrs,
            asm_done_target=total_wgs,
            asm_signal_num_ranks=num_ranks,
            asm_signal_generation=1,
            asm_reduce_y=y,
            sym_buffer=sym_buffer,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_tokens=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            output_workspace=l1_out,
            prob_storage=state.scratch.k3_prob_storage,
            verbose_build=verbose_build,
        )
    else:
        k3_l2_fused_asm_to_combine(
            act_fp8,
            act_scale,
            m_indices,
            (l2_weight, l2_scale),
            row_combine_ptrs,
            output_workspace=l1_out,
            prob_storage=state.scratch.k3_prob_storage,
            verbose_build=verbose_build,
        )
        rank_barrier(sym_buffer, rank_idx=rank_idx, num_ranks=num_ranks, verbose_build=verbose_build)
        reduce_local_combine(
            y,
            sym_buffer,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_tokens=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            verbose_build=verbose_build,
        )


def fp8_mega_moe_large_opt_3stage_graph(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    activation_clamp: Optional[float],
    fast_math: bool,
    graph_max_tokens: Optional[int] = None,
) -> None:
    if not fast_math:
        raise RuntimeError("large-opt DCU MegaMoE graph path requires fast_math")
    if cumulative_local_expert_recv_stats is not None:
        raise ValueError("large-opt DCU MegaMoE graph path does not support stats")
    if k3_tail_reduce_enabled():
        raise RuntimeError(
            "large-opt DCU MegaMoE graph prototype supports the non-tail-reduce K3 path only"
        )

    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    hidden = int(y.size(1))
    intermediate_hidden = int(l1_scale.size(1) // 2)
    if graph_max_tokens is None:
        graph_max_tokens = int(getattr(sym_buffer, "cuda_graph_max_tokens_per_rank", y.size(0)))
    graph_max_tokens = int(graph_max_tokens)
    if graph_max_tokens <= 0:
        raise ValueError("graph_max_tokens must be positive")
    if int(y.size(0)) < graph_max_tokens:
        raise ValueError(
            "large-opt graph path requires y to have at least graph_max_tokens rows"
        )
    num_max_tokens_per_rank = int(sym_buffer.num_max_tokens_per_rank)
    _check_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_tokens=graph_max_tokens,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
    )

    alignment = 256
    verbose_build = os.getenv("MEGAMOE_DCU_LARGE_OPT_VERBOSE_BUILD", "0") == "1"
    state = _state(
        sym_buffer,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        init_tail_reduce=False,
        verbose_build=verbose_build,
    )

    flags_offset, flags_numel, meta_flags_offset, meta_flags_numel, _, _ = (
        k1_graph_flag_reset_layout(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_tokens=graph_max_tokens,
            num_topk=num_topk,
            hidden=hidden,
            l1_rows=int(l1_scale.size(1)),
            alignment=alignment,
        )
    )
    rank_barrier(
        sym_buffer,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        k1_graph_reset_layout=(
            flags_offset,
            flags_numel,
            meta_flags_offset,
            meta_flags_numel,
        ),
        verbose_build=verbose_build,
    )

    runtime_num_tokens = sym_buffer.cuda_graph_num_tokens
    l1_out, route_weights, m_indices, output_index, row_combine_ptrs = (
        k1_symm_fused_l1_asm_graph(
            sym_buffer,
            (l1_weight, l1_scale),
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            num_experts=num_experts,
            graph_max_tokens=graph_max_tokens,
            num_topk=num_topk,
            hidden=hidden,
            runtime_num_tokens=runtime_num_tokens,
            alignment=alignment,
            l1_out_workspace=state.scratch.l1_out,
            verbose_build=verbose_build,
        )
    )
    rows = int(l1_out.size(0))

    act_fp8 = state.scratch.act_fp8[:rows]
    act_scale = state.scratch.act_scale[:rows]
    swiglu_quant_channelwise_out(
        l1_out,
        route_weights,
        act_fp8,
        act_scale,
        state.empty_bf16,
        num_per_channels=intermediate_hidden,
        output_bf16=False,
        clamp_value=activation_clamp,
        row_combine_ptrs=row_combine_ptrs,
        verbose_build=verbose_build,
    )

    k3_l2_fused_asm_to_combine(
        act_fp8,
        act_scale,
        m_indices,
        (l2_weight, l2_scale),
        row_combine_ptrs,
        output_workspace=l1_out,
        prob_storage=state.scratch.k3_prob_storage,
        active_tiles=state.scratch.k1_active_tiles,
        verbose_build=verbose_build,
    )
    rank_barrier(sym_buffer, rank_idx=rank_idx, num_ranks=num_ranks, verbose_build=verbose_build)
    reduce_local_combine_graph(
        y[:graph_max_tokens],
        sym_buffer,
        num_ranks=num_ranks,
        num_experts=num_experts,
        graph_num_tokens=graph_max_tokens,
        runtime_num_tokens=runtime_num_tokens,
        num_topk=num_topk,
        hidden=hidden,
        verbose_build=verbose_build,
    )
