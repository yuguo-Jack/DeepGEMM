from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .dcu_megamoe_opt.K1_fused.k1_fused import (
    k1_graph_flag_reset_layout,
    k1_symm_fused_l1_v3,
    k1_symm_fused_l1_v3_graph,
)
from .dcu_megamoe_opt.K2_fused.k2_fused import swiglu_quant_channelwise_out
from .dcu_megamoe_opt.K3_fused.k3_fused import (
    build_asm_tail_signal_addrs,
    k3_l2_fused_v3_to_combine,
    rank_barrier,
    reduce_local_combine,
    reduce_local_combine_graph,
)
from .dcu_megamoe_opt.v3_config import V3_BACKEND_LL, normalize_v3_backend


K_K1_ROUTE_TILE_M = 256
K_K1_ALIGNMENT = 256
K_K1_ROUTE_CAPACITY_SLACK = 64
K_K1_LL_ROW_TILE = 64
K_K1_LL_HEADROOM_EXPECTED_ROWS_THRESHOLD = 48
K_K1_LL_HEADROOM_ROWS = 64
K_K1_ASM_LAUNCH_ARGS_BYTES = 256
K_K2_GRAPH_ROW_BLOCKS = 8192
K_PROB_STORAGE_BYTES = 256
K_TAIL_DONE_COUNTER_RING_SLOTS = 16
K_TAIL_DONE_COUNTER_INTS = 80
K_POST_K3_BARRIER_SIGNAL_SLOT_BASE = 20
K_DSV4_FLASH_SUPPORTED_EP_RANKS = (8, 16, 32)
V3_LL_BLOCK_M = 32
K_LL_SPLIT_TAIL_MAX_TOKENS = 512
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
    graph_runtime_num_tokens: torch.Tensor
    graph_tail_signal_generation: torch.Tensor
    asm_tail_done_counter: torch.Tensor
    asm_tail_signal_addrs: torch.Tensor


@dataclass
class _OptState:
    scratch: _RouteScratchViews
    empty_bf16: torch.Tensor
    asm_tail_signal_addrs_ready: bool = False
    asm_signal_generation: int = 0


def k3_tail_reduce_enabled() -> bool:
    value = os.getenv("K3_USE_ASM_TAIL_REDUCE", "1").strip().lower()
    return value in {"1", "true", "yes", "on", "tail-reduce"}


def _tail_reduce_enabled_for_backend(v3_backend: str) -> bool:
    if v3_backend == V3_BACKEND_LL:
        return True
    return k3_tail_reduce_enabled()


def ll_k3_split_tail_enabled() -> bool:
    value = os.getenv("MEGAMOE_DCU_LL_K3_SPLIT_TAIL", "1").strip().lower()
    return value in {"1", "true", "yes", "on", "split"}


def _ll_k3_split_tail_enabled_for_tokens(v3_backend: str, num_tokens: int) -> bool:
    return (
        v3_backend == V3_BACKEND_LL
        and ll_k3_split_tail_enabled()
        and int(num_tokens) <= K_LL_SPLIT_TAIL_MAX_TOKENS
    )


def k2_skip_inactive_rows_enabled(num_tokens: int) -> bool:
    min_tokens = int(os.getenv("K2_SKIP_INACTIVE_ROWS_MIN_TOKENS", "1536"))
    return num_tokens >= min_tokens


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _dcu_tail_signal_slot_base(num_ranks: int) -> int:
    return 8 if int(num_ranks) <= 8 else int(num_ranks)


def _dcu_start_barrier_signal_slot_base(num_ranks: int) -> int:
    if int(num_ranks) <= 8:
        return 18
    return _dcu_tail_signal_slot_base(num_ranks) + int(num_ranks)


def _dcu_post_k3_barrier_signal_slot_base(num_ranks: int) -> int:
    if int(num_ranks) <= 8:
        return K_POST_K3_BARRIER_SIGNAL_SLOT_BASE
    return _dcu_start_barrier_signal_slot_base(num_ranks) + 2


def _dcu_tail_signal_addrs_count(num_ranks: int) -> int:
    return 2 * int(num_ranks)


def _route_task_workspace_bytes(*, num_ranks: int, num_experts: int, num_max_tokens: int) -> int:
    local_experts = num_experts // num_ranks
    task_capacity_per_expert = num_ranks * num_ranks * num_max_tokens
    bytes_ = local_experts * 4
    bytes_ = _align(bytes_, 16)
    bytes_ += local_experts * task_capacity_per_expert * 4
    return _align(bytes_, 16)


def _v3_staged_capacity_rows(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
) -> int:
    local_experts = num_experts // num_ranks
    ll_expected_rows_per_expert = max(
        1,
        _ceil_div(num_max_tokens * num_topk, local_experts),
    )
    ll_rows_per_expert = _align(ll_expected_rows_per_expert, K_K1_LL_ROW_TILE)
    min_slack = (
        K_K1_LL_HEADROOM_ROWS
        if ll_expected_rows_per_expert >= K_K1_LL_HEADROOM_EXPECTED_ROWS_THRESHOLD
        else 0
    )
    if ll_rows_per_expert - ll_expected_rows_per_expert < min_slack:
        ll_rows_per_expert = _align(
            ll_expected_rows_per_expert + min_slack,
            K_K1_LL_ROW_TILE,
        )
    ll_capacity_rows = local_experts * ll_rows_per_expert
    total_tasks = num_ranks * num_max_tokens * num_topk
    expected_per_expert = _ceil_div(total_tasks, num_experts)
    rows_per_expert_target = max(
        K_K1_ALIGNMENT,
        expected_per_expert + K_K1_ROUTE_CAPACITY_SLACK,
    )
    fixed_capacity_tiles_per_expert = _ceil_div(
        rows_per_expert_target,
        K_K1_ROUTE_TILE_M,
    )
    normal_capacity_rows = (
        local_experts * fixed_capacity_tiles_per_expert * K_K1_ROUTE_TILE_M
    )
    return max(ll_capacity_rows, normal_capacity_rows)


def _route_tile_offsets(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> tuple[int, int, int, int, int]:
    capacity_rows = _v3_staged_capacity_rows(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
    )
    offset = 0
    offset += capacity_rows * hidden
    offset = _align(offset, 16)
    k1_row_tiles = _ceil_div(capacity_rows, K_K1_ROUTE_TILE_M)
    k1_l1_tiles = _ceil_div(intermediate_hidden * 2, K_K1_ROUTE_TILE_M)
    offset += k1_row_tiles * k1_l1_tiles * K_DTYPE_SIZES[torch.int32]
    offset = _align(offset, 16)
    offset += k1_row_tiles * K_DTYPE_SIZES[torch.int32]
    offset = _align(offset, 16)
    offset += K_K1_ASM_LAUNCH_ARGS_BYTES
    offset = _align(offset, 16)
    act_bf16_offset = offset
    offset += capacity_rows * intermediate_hidden * 2 * K_DTYPE_SIZES[torch.bfloat16]
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
        raise RuntimeError("route_scratch is too small for opt scratch views")
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
    tail_signal_addrs_count = _dcu_tail_signal_addrs_count(num_ranks)
    tail_signal_addrs_offset = _align(
        tail_done_offset + K_TAIL_DONE_COUNTER_INTS * K_DTYPE_SIZES[torch.int32],
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
        graph_runtime_num_tokens=_route_scratch_tensor(
            route_scratch,
            byte_offset=prob_offset + K_PROB_STORAGE_BYTES - 2 * K_DTYPE_SIZES[torch.int32],
            byte_capacity=K_DTYPE_SIZES[torch.int32],
            dtype=torch.int32,
            shape=(1,),
        ),
        graph_tail_signal_generation=_route_scratch_tensor(
            route_scratch,
            byte_offset=prob_offset + K_PROB_STORAGE_BYTES - K_DTYPE_SIZES[torch.int32],
            byte_capacity=K_DTYPE_SIZES[torch.int32],
            dtype=torch.int32,
            shape=(1,),
        ),
        asm_tail_done_counter=_route_scratch_tensor(
            route_scratch,
            byte_offset=tail_done_offset,
            byte_capacity=K_TAIL_DONE_COUNTER_INTS * K_DTYPE_SIZES[torch.int32],
            dtype=torch.int32,
            shape=(K_TAIL_DONE_COUNTER_INTS,),
        ),
        asm_tail_signal_addrs=_route_scratch_tensor(
            route_scratch,
            byte_offset=tail_signal_addrs_offset,
            byte_capacity=tail_signal_addrs_count * K_DTYPE_SIZES[torch.int64],
            dtype=torch.int64,
            shape=(tail_signal_addrs_count,),
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
) -> _OptState:
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
    cached = getattr(sym_buffer, "_opt_3stage_state", None)
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

    state = _OptState(
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
    setattr(sym_buffer, "_opt_3stage_state", (key, state))
    return state


def prepare_opt_3stage(sym_buffer, *, verbose_build: bool = False) -> None:
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
        num_ranks not in K_DSV4_FLASH_SUPPORTED_EP_RANKS
        or num_experts != 256
        or num_topk != 6
        or hidden != 4096
        or intermediate_hidden != 2048
        or num_tokens < 0
        or num_tokens > num_max_tokens_per_rank
    ):
        raise ValueError(
            "DCU MegaMoE V3 staged path supports only DeepSeek-V4-Flash "
            "EP8/EP16/EP32 shape: experts=256, topk=6, hidden=4096, intermediate=2048, "
            "and 0<=num_tokens_per_rank<=num_max_tokens_per_rank"
        )


def fp8_mega_moe_opt_3stage(
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
    v3_backend: str,
    capacity_num_tokens: Optional[int] = None,
    use_unified_weight_layout: bool = False,
) -> None:
    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    num_tokens = int(y.size(0))
    hidden = int(y.size(1))
    intermediate_hidden = int(l1_scale.size(1) // 2)
    num_max_tokens_per_rank = int(sym_buffer.num_max_tokens_per_rank)
    v3_backend = normalize_v3_backend(v3_backend)
    route_capacity_num_tokens = (
        num_tokens if capacity_num_tokens is None else int(capacity_num_tokens)
    )
    if (
        route_capacity_num_tokens < num_tokens
        or route_capacity_num_tokens > num_max_tokens_per_rank
    ):
        raise ValueError(
            "capacity_num_tokens must be in "
            f"[num_tokens, {num_max_tokens_per_rank}] for eager DCU MegaMoE"
        )
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
    verbose_build = os.getenv("MEGAMOE_DCU_OPT_VERBOSE_BUILD", "0") == "1"
    use_tail_reduce = _tail_reduce_enabled_for_backend(v3_backend)
    use_ll_split_tail = _ll_k3_split_tail_enabled_for_tokens(v3_backend, num_tokens)
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
    asm_signal_generation = 1
    if use_tail_reduce:
        state.asm_signal_generation = (state.asm_signal_generation % 0x3fffffff) + 1
        asm_signal_generation = state.asm_signal_generation

    inline_ll_start = v3_backend == V3_BACKEND_LL
    if not inline_ll_start:
        # Make symmetric-buffer input copies visible to the dispatch stage.
        rank_barrier(
            sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            asm_done_counter=(
                state.scratch.asm_tail_done_counter
                if use_tail_reduce
                else None
            ),
            reset_tail_signal_slots=use_tail_reduce,
            graph_max_tokens=num_tokens,
            barrier_signal_slot_base=_dcu_start_barrier_signal_slot_base(num_ranks),
            verbose_build=verbose_build,
        )

    force_safe_compact = num_tokens == 0 or route_capacity_num_tokens > num_tokens
    k1_launcher = k1_symm_fused_l1_v3
    k1_kwargs = dict(
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
        l1_out_workspace=state.scratch.l1_out,
        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
        force_compact_prebuild=force_safe_compact,
        capacity_num_tokens=route_capacity_num_tokens,
        use_unified_weight_layout=use_unified_weight_layout,
        verbose_build=verbose_build,
    )
    k1_kwargs["backend"] = v3_backend
    if v3_backend == V3_BACKEND_LL:
        k1_kwargs["ll_block_m"] = V3_LL_BLOCK_M
        if inline_ll_start:
            k1_kwargs["enable_start_rank_barrier"] = True
            k1_kwargs["tail_done_counter"] = state.scratch.asm_tail_done_counter
    l1_out, route_weights, m_indices, output_index, row_combine_ptrs = k1_launcher(
        sym_buffer,
        (l1_weight, l1_scale),
        **k1_kwargs,
    )
    rows = int(l1_out.size(0))

    act_fp8 = state.scratch.act_fp8[:rows]
    act_scale = state.scratch.act_scale[:rows]
    k2_actual_m = m_indices if v3_backend == V3_BACKEND_LL else None
    k2_m_per_expert = rows // (num_experts // num_ranks) if k2_actual_m is not None else 0
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
            row_combine_ptrs
            if (force_safe_compact or k2_skip_inactive_rows_enabled(num_tokens))
            else None
        ),
        actual_m=k2_actual_m,
        m_per_expert=k2_m_per_expert,
        fast_math=fast_math,
        verbose_build=verbose_build,
    )

    if use_tail_reduce:
        total_wgs = ((rows + 255) // 256) * ((hidden + 255) // 256)
        k3_launcher = k3_l2_fused_v3_to_combine
        k3_kwargs = dict(
            asm_done_counter=state.scratch.asm_tail_done_counter,
            asm_signal_addrs=state.scratch.asm_tail_signal_addrs,
            asm_done_target=total_wgs,
            asm_signal_num_ranks=num_ranks,
            asm_signal_generation=asm_signal_generation,
            asm_reduce_y=y,
            sym_buffer=sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_tokens=num_tokens,
            num_topk=num_topk,
            hidden=hidden,
            output_workspace=l1_out,
            prob_storage=state.scratch.k3_prob_storage,
            ll_split_tail=use_ll_split_tail,
            use_unified_weight_layout=use_unified_weight_layout,
            verbose_build=verbose_build,
        )
        k3_kwargs["backend"] = v3_backend
        if v3_backend == V3_BACKEND_LL:
            k3_kwargs["ll_block_m"] = V3_LL_BLOCK_M
        k3_launcher(
            act_fp8,
            act_scale,
            m_indices,
            (l2_weight, l2_scale),
            row_combine_ptrs,
            **k3_kwargs,
        )
    else:
        k3_launcher = k3_l2_fused_v3_to_combine
        k3_kwargs = dict(
            output_workspace=l1_out,
            prob_storage=state.scratch.k3_prob_storage,
            use_unified_weight_layout=use_unified_weight_layout,
            verbose_build=verbose_build,
        )
        k3_kwargs["backend"] = v3_backend
        if v3_backend == V3_BACKEND_LL:
            k3_kwargs["ll_block_m"] = V3_LL_BLOCK_M
        k3_launcher(
            act_fp8,
            act_scale,
            m_indices,
            (l2_weight, l2_scale),
            row_combine_ptrs,
            **k3_kwargs,
        )
        rank_barrier(
            sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            barrier_signal_slot_base=_dcu_post_k3_barrier_signal_slot_base(num_ranks),
            verbose_build=verbose_build,
        )
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


def _run_opt_3stage_graph(
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
    v3_backend: str,
    use_unified_weight_layout: bool = False,
) -> None:
    if cumulative_local_expert_recv_stats is not None:
        raise ValueError("opt DCU MegaMoE graph path does not support stats")
    l1_weight, l1_scale = l1_weights
    l2_weight, l2_scale = l2_weights
    hidden = int(y.size(1))
    intermediate_hidden = int(l1_scale.size(1) // 2)
    if graph_max_tokens is None:
        graph_max_tokens = int(sym_buffer.cuda_graph_max_tokens_per_rank)
    graph_max_tokens = int(graph_max_tokens)
    if graph_max_tokens <= 0:
        raise ValueError("graph_max_tokens must be positive")
    if int(y.size(0)) < graph_max_tokens:
        raise ValueError(
            "opt graph path requires y to have at least graph_max_tokens rows"
    )
    num_max_tokens_per_rank = int(sym_buffer.num_max_tokens_per_rank)
    v3_backend = normalize_v3_backend(v3_backend)
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
    verbose_build = os.getenv("MEGAMOE_DCU_OPT_VERBOSE_BUILD", "0") == "1"
    use_tail_reduce = _tail_reduce_enabled_for_backend(v3_backend)
    use_ll_split_tail = _ll_k3_split_tail_enabled_for_tokens(
        v3_backend, graph_max_tokens
    )
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

    k1_graph_reset_layout = None
    if v3_backend != V3_BACKEND_LL:
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
        k1_graph_reset_layout = (
            flags_offset,
            flags_numel,
            meta_flags_offset,
            meta_flags_numel,
        )
    inline_ll_start = v3_backend == V3_BACKEND_LL
    if not inline_ll_start:
        rank_barrier(
            sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            asm_done_counter=(
                state.scratch.asm_tail_done_counter
                if use_tail_reduce
                else None
            ),
            reset_tail_signal_slots=use_tail_reduce,
            k1_graph_reset_layout=k1_graph_reset_layout,
            graph_runtime_num_tokens=sym_buffer.cuda_graph_num_tokens,
            graph_runtime_num_tokens_out=state.scratch.graph_runtime_num_tokens,
            graph_tail_signal_generation_out=(
                state.scratch.graph_tail_signal_generation if use_tail_reduce else None
            ),
            graph_max_tokens=graph_max_tokens,
            barrier_signal_slot_base=_dcu_start_barrier_signal_slot_base(num_ranks),
            verbose_build=verbose_build,
        )

    runtime_num_tokens = sym_buffer.cuda_graph_num_tokens
    k1_graph_launcher = k1_symm_fused_l1_v3_graph
    k1_kwargs = dict(
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        num_experts=num_experts,
        graph_max_tokens=graph_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        runtime_num_tokens=runtime_num_tokens,
        alignment=alignment,
        l1_out_workspace=state.scratch.l1_out,
        use_unified_weight_layout=use_unified_weight_layout,
        verbose_build=verbose_build,
    )
    k1_kwargs["backend"] = v3_backend
    if v3_backend == V3_BACKEND_LL:
        k1_kwargs["ll_block_m"] = V3_LL_BLOCK_M
        if inline_ll_start:
            k1_kwargs["enable_start_rank_barrier"] = True
            k1_kwargs["tail_done_counter"] = state.scratch.asm_tail_done_counter
            k1_kwargs["graph_runtime_num_tokens_out"] = (
                state.scratch.graph_runtime_num_tokens
            )
            if use_tail_reduce:
                k1_kwargs["graph_tail_signal_generation_out"] = (
                    state.scratch.graph_tail_signal_generation
                )
    l1_out, route_weights, m_indices, output_index, row_combine_ptrs = (
        k1_graph_launcher(
            sym_buffer,
            (l1_weight, l1_scale),
            **k1_kwargs,
        )
    )
    rows = int(l1_out.size(0))

    act_fp8 = state.scratch.act_fp8[:rows]
    act_scale = state.scratch.act_scale[:rows]
    k2_actual_m = m_indices if v3_backend == V3_BACKEND_LL else None
    k2_m_per_expert = rows // (num_experts // num_ranks) if k2_actual_m is not None else 0
    k2_active_tiles = (
        state.scratch.k1_active_tiles if v3_backend != V3_BACKEND_LL else None
    )
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
        max_row_blocks=K_K2_GRAPH_ROW_BLOCKS,
        actual_m=k2_actual_m,
        m_per_expert=k2_m_per_expert,
        active_tiles=k2_active_tiles,
        active_tile_m=K_K1_ROUTE_TILE_M if k2_active_tiles is not None else 0,
        fast_math=fast_math,
        verbose_build=verbose_build,
    )

    if use_tail_reduce:
        total_wgs = ((rows + 255) // 256) * ((hidden + 255) // 256)
        graph_runtime_offset = (
            state.scratch.graph_runtime_num_tokens.data_ptr()
            - state.scratch.k1_active_tiles.data_ptr()
        )
        k3_launcher = k3_l2_fused_v3_to_combine
        k3_kwargs = dict(
            asm_done_counter=state.scratch.asm_tail_done_counter,
            asm_signal_addrs=state.scratch.asm_tail_signal_addrs,
            asm_done_target=total_wgs,
            asm_signal_num_ranks=num_ranks,
            asm_signal_generation=1,
            asm_signal_generation_tensor=state.scratch.graph_tail_signal_generation,
            asm_reduce_y=y[:graph_max_tokens],
            sym_buffer=sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_tokens=graph_max_tokens,
            num_topk=num_topk,
            hidden=hidden,
            output_workspace=l1_out,
            prob_storage=state.scratch.k3_prob_storage,
            active_tiles=state.scratch.k1_active_tiles,
            graph_runtime_offset_from_active_tiles=graph_runtime_offset,
            ll_split_tail=use_ll_split_tail,
            use_unified_weight_layout=use_unified_weight_layout,
            verbose_build=verbose_build,
        )
        k3_kwargs["backend"] = v3_backend
        if v3_backend == V3_BACKEND_LL:
            k3_kwargs["ll_block_m"] = V3_LL_BLOCK_M
            k3_kwargs["graph_runtime_num_tokens"] = state.scratch.graph_runtime_num_tokens
        k3_launcher(
            act_fp8,
            act_scale,
            m_indices,
            (l2_weight, l2_scale),
            row_combine_ptrs,
            **k3_kwargs,
        )
    else:
        graph_runtime_offset = (
            state.scratch.graph_runtime_num_tokens.data_ptr()
            - state.scratch.k1_active_tiles.data_ptr()
        )
        k3_launcher = k3_l2_fused_v3_to_combine
        k3_kwargs = dict(
            output_workspace=l1_out,
            prob_storage=state.scratch.k3_prob_storage,
            active_tiles=state.scratch.k1_active_tiles,
            graph_runtime_offset_from_active_tiles=graph_runtime_offset,
            use_unified_weight_layout=use_unified_weight_layout,
            verbose_build=verbose_build,
        )
        k3_kwargs["backend"] = v3_backend
        if v3_backend == V3_BACKEND_LL:
            k3_kwargs["ll_block_m"] = V3_LL_BLOCK_M
        k3_launcher(
            act_fp8,
            act_scale,
            m_indices,
            (l2_weight, l2_scale),
            row_combine_ptrs,
            **k3_kwargs,
        )
        rank_barrier(
            sym_buffer,
            rank_idx=rank_idx,
            num_ranks=num_ranks,
            barrier_signal_slot_base=_dcu_post_k3_barrier_signal_slot_base(num_ranks),
            verbose_build=verbose_build,
        )
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
