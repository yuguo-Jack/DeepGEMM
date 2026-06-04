"""Runtime orchestration for DCU MegaMoE V2 staged fused execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

K_DCU_ROUTE_TILE_M = 32
K_DCU_ROUTE_SCRATCH_MMAC_TILE_M = 16
K_DCU_ROUTE_SCRATCH_PIPELINE_COUNTER_COUNT = 5
K3_COPY_WORKERS = 16
K1K3_LOCAL_EXPERTS = 32
_K1_STAGE_LAUNCHER = None
_K2_STAGE_LAUNCHER = None
_K3_STAGE_LAUNCHER = None

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_DTYPE_SIZES = {
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
}
if _FP8_DTYPE is not None:
    _DTYPE_SIZES[_FP8_DTYPE] = 1


@dataclass(frozen=True)
class V2StagePlan:
    backend: str
    k1_kernel_family: str
    k1_problem_k: int
    k1_problem_n: int
    k1_uses_dispatch_pull: bool
    k3_kernel_family: str
    k3_problem_k: int
    k3_problem_n: int
    k3_uses_copy_stage: bool
    k3_uses_tail_reduce: bool


@dataclass(frozen=True)
class V2RouteTileLayout:
    max_route_tiles: int
    capacity_rows: int
    x_fp8_offset: int
    act_bf16_offset: int
    act_fp8_offset: int
    act_scale_offset: int
    act_chunk_amax_offset: int
    tile_x_row_ptrs_offset: int
    tile_combine_row_ptrs_offset: int
    tile_route_weight_offset: int
    tile_x_scale_offset: int
    tile_expert_offset: int
    tile_pool_base_offset: int
    tile_count_offset: int
    expert_l1_task_offset: int
    expert_quant_done_count_offset: int
    l2_group_done_count_offset: int
    tile_pull_done_offset: int
    l1_done_count_offset: int
    l2_queue_offset: int
    l2_queue_ready_offset: int
    pipeline_counter_offset: int
    total_tiles_offset: int
    bytes: int


@dataclass
class V2WorkspaceViews:
    staged_x: torch.Tensor
    staged_x_scale: torch.Tensor
    route_weights: torch.Tensor
    l1_out: torch.Tensor
    l2_workspace: torch.Tensor
    act_fp8: torch.Tensor
    act_scale: torch.Tensor
    problem_size: torch.Tensor
    route_scratch_i32: torch.Tensor
    grid_barrier: torch.Tensor
    row_expert: torch.Tensor
    m_indices: torch.Tensor
    row_output_ptrs: torch.Tensor
    row_combine_ptrs: torch.Tensor
    output_index: torch.Tensor
    local_topk_mask: torch.Tensor
    tail_tokens: torch.Tensor
    rows_aligned_per_expert: int
    valid_rows_per_expert: int
    launch_rows: int
    local_experts: int
    capacity_rows: int
    grid_barrier_ints: int
    max_route_tiles: int
    route_base_bytes: int
    route_scratch_bytes: int


@dataclass
class V2RuntimeState:
    scratch: V2WorkspaceViews
    empty_bf16: torch.Tensor
    epoch: int = 1


def get_v2_stage_plan(backend: str) -> V2StagePlan:
    if backend == "ll":
        return V2StagePlan(
            backend="ll",
            k1_kernel_family="low_latency_c_pack5",
            k1_problem_k=4096,
            k1_problem_n=4096,
            k1_uses_dispatch_pull=True,
            k3_kernel_family="low_latency_c_pack5",
            k3_problem_k=2048,
            k3_problem_n=4096,
            k3_uses_copy_stage=False,
            k3_uses_tail_reduce=True,
        )
    if backend == "normal":
        return V2StagePlan(
            backend="normal",
            k1_kernel_family="normal_c_pack5",
            k1_problem_k=4096,
            k1_problem_n=4096,
            k1_uses_dispatch_pull=True,
            k3_kernel_family="normal_c_pack5",
            k3_problem_k=2048,
            k3_problem_n=4096,
            k3_uses_copy_stage=True,
            k3_uses_tail_reduce=True,
        )
    raise ValueError("V2 backend must be 'll' or 'normal'")


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _dtype_size(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_SIZES[dtype]
    except KeyError as exc:
        raise RuntimeError(f"V2 workspace does not support dtype {dtype}") from exc


def _fp8_dtype() -> torch.dtype:
    if _FP8_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is required for V2 FP8 workspace views")
    return _FP8_DTYPE


def _route_task_workspace_bytes(*, num_ranks: int, num_experts: int, num_max_tokens: int) -> int:
    local_experts = num_experts // num_ranks
    task_capacity_per_expert = num_ranks * num_ranks * num_max_tokens
    bytes_ = local_experts * _dtype_size(torch.int32)
    bytes_ = _align(bytes_, 16)
    bytes_ += local_experts * task_capacity_per_expert * _dtype_size(torch.int32)
    return _align(bytes_, 16)


def _route_scratch_capacity_tiles(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    min_capacity_rows: int = 0,
) -> int:
    local_experts = num_experts // num_ranks
    max_route_tasks = num_ranks * num_max_tokens * num_topk
    route_tiles = local_experts + (
        max_route_tasks + K_DCU_ROUTE_TILE_M - 1
    ) // K_DCU_ROUTE_TILE_M
    min_tiles = (min_capacity_rows + K_DCU_ROUTE_TILE_M - 1) // K_DCU_ROUTE_TILE_M
    return max(route_tiles, min_tiles)


def _route_tile_layout(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    min_capacity_rows: int = 0,
) -> V2RouteTileLayout:
    max_route_tiles = _route_scratch_capacity_tiles(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        min_capacity_rows=min_capacity_rows,
    )
    capacity_rows = max_route_tiles * K_DCU_ROUTE_TILE_M
    offset = 0

    x_fp8_offset = offset
    offset += capacity_rows * hidden
    offset = _align(offset, 16)

    act_bf16_offset = offset
    offset += capacity_rows * intermediate_hidden * _dtype_size(torch.bfloat16)
    offset = _align(offset, 16)

    act_fp8_offset = offset
    offset += capacity_rows * intermediate_hidden
    offset = _align(offset, 16)

    act_scale_offset = offset
    offset += capacity_rows * _dtype_size(torch.float32)
    offset = _align(offset, 16)

    act_chunk_amax_offset = offset
    offset += (
        capacity_rows
        * ((intermediate_hidden + 255) // 256)
        * _dtype_size(torch.float32)
    )
    offset = _align(offset, 16)

    tile_x_row_ptrs_offset = offset
    offset += capacity_rows * _dtype_size(torch.int64)
    offset = _align(offset, 16)

    tile_combine_row_ptrs_offset = offset
    offset += capacity_rows * _dtype_size(torch.int64)
    offset = _align(offset, 16)

    tile_route_weight_offset = offset
    offset += capacity_rows * _dtype_size(torch.float32)
    offset = _align(offset, 16)

    tile_x_scale_offset = offset
    offset += capacity_rows * _dtype_size(torch.float32)
    offset = _align(offset, 16)

    tile_expert_offset = offset
    offset += max_route_tiles * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    tile_pool_base_offset = offset
    offset += max_route_tiles * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    tile_count_offset = offset
    offset += max_route_tiles * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    expert_l1_task_offset = offset
    offset += (max_route_tiles + 1) * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    expert_quant_done_count_offset = offset
    offset += (max_route_tiles + 1) * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    subtiles_per_tile = K_DCU_ROUTE_TILE_M // K_DCU_ROUTE_SCRATCH_MMAC_TILE_M
    max_subtiles = max_route_tiles * subtiles_per_tile

    l2_group_done_count_offset = offset
    offset += max_subtiles * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    tile_pull_done_offset = offset
    offset += max_route_tiles * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    l1_done_count_offset = offset
    offset += max_subtiles * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    max_l2_tasks = max_subtiles * ((hidden + 15) // 16)
    l2_queue_offset = offset
    offset += max_l2_tasks * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    l2_queue_ready_offset = offset
    offset += max_l2_tasks * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    pipeline_counter_offset = offset
    offset += K_DCU_ROUTE_SCRATCH_PIPELINE_COUNTER_COUNT * _dtype_size(torch.int32)
    offset = _align(offset, 16)

    total_tiles_offset = offset
    offset += _dtype_size(torch.int32)

    return V2RouteTileLayout(
        max_route_tiles=max_route_tiles,
        capacity_rows=capacity_rows,
        x_fp8_offset=x_fp8_offset,
        act_bf16_offset=act_bf16_offset,
        act_fp8_offset=act_fp8_offset,
        act_scale_offset=act_scale_offset,
        act_chunk_amax_offset=act_chunk_amax_offset,
        tile_x_row_ptrs_offset=tile_x_row_ptrs_offset,
        tile_combine_row_ptrs_offset=tile_combine_row_ptrs_offset,
        tile_route_weight_offset=tile_route_weight_offset,
        tile_x_scale_offset=tile_x_scale_offset,
        tile_expert_offset=tile_expert_offset,
        tile_pool_base_offset=tile_pool_base_offset,
        tile_count_offset=tile_count_offset,
        expert_l1_task_offset=expert_l1_task_offset,
        expert_quant_done_count_offset=expert_quant_done_count_offset,
        l2_group_done_count_offset=l2_group_done_count_offset,
        tile_pull_done_offset=tile_pull_done_offset,
        l1_done_count_offset=l1_done_count_offset,
        l2_queue_offset=l2_queue_offset,
        l2_queue_ready_offset=l2_queue_ready_offset,
        pipeline_counter_offset=pipeline_counter_offset,
        total_tiles_offset=total_tiles_offset,
        bytes=_align(offset, 16),
    )


def _v2_route_scratch_min_bytes(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    backend: str | None = None,
) -> int:
    local_experts = _check_v2_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_max_tokens=num_max_tokens,
    )
    backends = ("ll", "normal") if backend is None else (backend,)
    min_capacity_rows = 0
    for candidate in backends:
        rows_aligned_per_expert, _ = _v2_rows_for_backend(
            backend=candidate,
            num_max_tokens=num_max_tokens,
            num_topk=num_topk,
            local_experts=local_experts,
        )
        min_capacity_rows = max(
            min_capacity_rows,
            2 * local_experts * rows_aligned_per_expert,
        )
    layout = _route_tile_layout(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        min_capacity_rows=min_capacity_rows,
    )
    return _align(
        _route_task_workspace_bytes(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_max_tokens=num_max_tokens,
        )
        + layout.bytes,
        16,
    )


def _route_scratch_tensor(
    route_scratch: torch.Tensor,
    *,
    byte_offset: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    dtype_size = _dtype_size(dtype)
    required_bytes = _numel(shape) * dtype_size
    if byte_offset % dtype_size != 0:
        raise RuntimeError(f"route_scratch offset {byte_offset} is not aligned for {dtype}")
    if byte_offset + required_bytes > route_scratch.numel():
        raise RuntimeError("route_scratch is too small for V2 scratch views")
    return route_scratch.narrow(0, byte_offset, required_bytes).view(dtype).view(shape)


def _v2_rows_for_backend(
    *,
    backend: str,
    num_max_tokens: int,
    num_topk: int,
    local_experts: int,
) -> tuple[int, int]:
    valid_rows_per_expert = max(
        1,
        (num_max_tokens * num_topk + local_experts - 1) // local_experts,
    )
    row_tile = 64 if backend == "ll" else 256
    return _align(valid_rows_per_expert, row_tile), valid_rows_per_expert


def _v2_grid_barrier_ints(*, backend: str, launch_rows: int) -> int:
    if backend == "normal":
        c_grid_y = (launch_rows + 255) // 256
        return 16 * c_grid_y + 2
    return 2


def _check_v2_shape(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_max_tokens: int,
) -> int:
    if num_ranks <= 0 or num_experts <= 0 or num_experts % num_ranks != 0:
        raise ValueError("invalid V2 rank/expert shape")
    local_experts = num_experts // num_ranks
    if local_experts != K1K3_LOCAL_EXPERTS:
        raise ValueError("V2 C pack5 kernels are currently specialized for 32 local experts")
    if num_topk <= 0 or hidden <= 0 or intermediate_hidden <= 0 or num_max_tokens <= 0:
        raise ValueError("invalid V2 token/topk/hidden shape")
    return local_experts


def _v2_workspace_views(
    sym_buffer,
    *,
    backend: str,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> V2WorkspaceViews:
    get_v2_stage_plan(backend)
    route_scratch = getattr(sym_buffer, "route_scratch", None)
    if route_scratch is None:
        raise TypeError("sym_buffer must expose route_scratch for V2")
    if route_scratch.dtype != torch.int8 or not route_scratch.is_contiguous():
        raise RuntimeError("sym_buffer.route_scratch must be contiguous int8")

    num_max_tokens = int(sym_buffer.num_max_tokens_per_rank)
    local_experts = _check_v2_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_max_tokens=num_max_tokens,
    )
    route_base = _route_task_workspace_bytes(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
    )
    rows_aligned_per_expert, valid_rows_per_expert = _v2_rows_for_backend(
        backend=backend,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        local_experts=local_experts,
    )
    launch_rows = local_experts * rows_aligned_per_expert
    layout = _route_tile_layout(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        min_capacity_rows=2 * launch_rows,
    )
    min_bytes = _align(route_base + layout.bytes, 16)
    if route_scratch.numel() < min_bytes:
        raise RuntimeError(
            f"sym_buffer.route_scratch is too small for V2: got {route_scratch.numel()} "
            f"bytes, need at least {min_bytes}"
        )

    l1_cols = intermediate_hidden * 2
    l1_capacity_bytes = layout.act_fp8_offset - layout.act_bf16_offset
    l1_capacity_rows = l1_capacity_bytes // (l1_cols * _dtype_size(torch.bfloat16))
    if launch_rows > l1_capacity_rows:
        raise RuntimeError(
            "V2 route_scratch L1 BF16 region cannot cover the padded launch rows; "
            f"launch_rows={launch_rows}, l1_capacity_rows={l1_capacity_rows}"
        )

    grid_barrier_ints = _v2_grid_barrier_ints(backend=backend, launch_rows=launch_rows)
    route_scratch_i32_ints = local_experts + 2 * local_experts * rows_aligned_per_expert

    cursor = route_base + layout.l2_queue_offset
    metadata_end = route_base + layout.bytes

    def alloc(dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal cursor
        cursor = _align(cursor, _dtype_size(dtype))
        tensor = _route_scratch_tensor(
            route_scratch,
            byte_offset=cursor,
            dtype=dtype,
            shape=shape,
        )
        cursor += _numel(shape) * _dtype_size(dtype)
        if cursor > metadata_end:
            raise RuntimeError("V2 metadata views exceed route_scratch metadata region")
        return tensor

    staged_x = _route_scratch_tensor(
        route_scratch,
        byte_offset=route_base + layout.x_fp8_offset,
        dtype=_fp8_dtype(),
        shape=(layout.capacity_rows, hidden),
    )
    l1_out = _route_scratch_tensor(
        route_scratch,
        byte_offset=route_base + layout.act_bf16_offset,
        dtype=torch.bfloat16,
        shape=(l1_capacity_rows, l1_cols),
    )
    act_fp8 = _route_scratch_tensor(
        route_scratch,
        byte_offset=route_base + layout.act_fp8_offset,
        dtype=_fp8_dtype(),
        shape=(layout.capacity_rows, intermediate_hidden),
    )
    act_scale = _route_scratch_tensor(
        route_scratch,
        byte_offset=route_base + layout.act_scale_offset,
        dtype=torch.float32,
        shape=(layout.capacity_rows,),
    )
    route_weights = _route_scratch_tensor(
        route_scratch,
        byte_offset=route_base + layout.tile_route_weight_offset,
        dtype=torch.float32,
        shape=(layout.capacity_rows,),
    )
    staged_x_scale = _route_scratch_tensor(
        route_scratch,
        byte_offset=route_base + layout.tile_x_scale_offset,
        dtype=torch.float32,
        shape=(layout.capacity_rows,),
    )

    row_expert = alloc(torch.int32, (launch_rows,))
    row_output_ptrs = alloc(torch.int64, (launch_rows,))
    output_index = alloc(
        torch.int32,
        (num_ranks * num_max_tokens, num_topk),
    )

    return V2WorkspaceViews(
        staged_x=staged_x,
        staged_x_scale=staged_x_scale,
        route_weights=route_weights,
        l1_out=l1_out,
        l2_workspace=l1_out,
        act_fp8=act_fp8,
        act_scale=act_scale,
        problem_size=alloc(torch.int32, (local_experts,)),
        route_scratch_i32=alloc(torch.int32, (route_scratch_i32_ints,)),
        grid_barrier=alloc(torch.int32, (grid_barrier_ints,)),
        row_expert=row_expert,
        m_indices=row_expert,
        row_output_ptrs=row_output_ptrs,
        row_combine_ptrs=row_output_ptrs,
        output_index=output_index,
        local_topk_mask=alloc(torch.uint8, (num_max_tokens * num_topk,)),
        tail_tokens=alloc(torch.int32, (num_max_tokens,)),
        rows_aligned_per_expert=rows_aligned_per_expert,
        valid_rows_per_expert=valid_rows_per_expert,
        launch_rows=launch_rows,
        local_experts=local_experts,
        capacity_rows=layout.capacity_rows,
        grid_barrier_ints=grid_barrier_ints,
        max_route_tiles=layout.max_route_tiles,
        route_base_bytes=route_base,
        route_scratch_bytes=min_bytes,
    )


def _device_key(tensor: torch.Tensor) -> tuple[str, int]:
    device = tensor.device
    return device.type, int(device.index if device.index is not None else -1)


def _symm_group_size_rank(sym_buffer) -> tuple[int, int]:
    group = getattr(sym_buffer, "group", None)
    if group is None:
        raise TypeError("sym_buffer must expose a distributed group for V2")
    return int(group.size()), int(group.rank())


def _get_k1_stage_launcher():
    global _K1_STAGE_LAUNCHER
    if _K1_STAGE_LAUNCHER is None:
        from megamoe.dcu_megamoe_v2.K1_fused.k1_fused import (
            k1_dispatch_pull_l1_fused_v2,
        )

        _K1_STAGE_LAUNCHER = k1_dispatch_pull_l1_fused_v2
    return _K1_STAGE_LAUNCHER


def _get_k2_stage_launcher():
    global _K2_STAGE_LAUNCHER
    if _K2_STAGE_LAUNCHER is None:
        from megamoe.dcu_megamoe_v2.K2_fused.k2_fused import (
            swiglu_quant_channelwise_out_v2,
        )

        _K2_STAGE_LAUNCHER = swiglu_quant_channelwise_out_v2
    return _K2_STAGE_LAUNCHER


def _get_k3_stage_launcher():
    global _K3_STAGE_LAUNCHER
    if _K3_STAGE_LAUNCHER is None:
        from megamoe.dcu_megamoe_v2.K3_fused.k3_fused import (
            k3_l2_combine_fused_v2,
        )

        _K3_STAGE_LAUNCHER = k3_l2_combine_fused_v2
    return _K3_STAGE_LAUNCHER


def _run_profiled_stage(
    profile_stages: Optional[dict[str, float]],
    name: str,
    fn,
) -> None:
    if profile_stages is None:
        fn()
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    profile_stages[name] = float(start.elapsed_time(end))


def _v2_state(
    sym_buffer,
    *,
    backend: str,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> V2RuntimeState:
    route_scratch = getattr(sym_buffer, "route_scratch", None)
    buffer = getattr(sym_buffer, "buffer", None)
    if route_scratch is None or buffer is None:
        raise TypeError("sym_buffer must expose buffer and route_scratch for V2")
    key = (
        _device_key(route_scratch),
        int(route_scratch.data_ptr()),
        int(buffer.data_ptr()),
        int(sym_buffer.num_max_tokens_per_rank),
        backend,
        num_ranks,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    cached = getattr(sym_buffer, "_dcu_megamoe_v2_state", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    state = V2RuntimeState(
        scratch=_v2_workspace_views(
            sym_buffer,
            backend=backend,
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        ),
        empty_bf16=buffer[:0].view(torch.bfloat16),
    )
    state.scratch.grid_barrier.zero_()
    setattr(sym_buffer, "_dcu_megamoe_v2_state", (key, state))
    return state


def run_stages_fused_v2(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
    *,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    activation_clamp: Optional[float],
    fast_math: bool,
    dispatch_num_tokens: Optional[int],
    backend: str,
    profile_stages: Optional[dict[str, float]] = None,
) -> None:
    """Run V2 K1/K2/K3 staged fused kernels.

    This stub is kept narrow while K1/K3 pybind wrappers are split out from the
    prototype harness. It prevents accidental fallback to baseline or big-fused
    paths during early integration.
    """
    plan = get_v2_stage_plan(backend)
    num_ranks, rank_idx = _symm_group_size_rank(sym_buffer)
    state = _v2_state(
        sym_buffer,
        backend=plan.backend,
        num_ranks=num_ranks,
        num_experts=int(sym_buffer.num_experts),
        num_topk=int(sym_buffer.num_topk),
        hidden=int(y.size(1)),
        intermediate_hidden=int(l1_weights[1].size(1) // 2),
    )
    scratch = state.scratch
    launch_epoch = state.epoch
    state.epoch += 1
    if dispatch_num_tokens is None:
        num_tokens = int(y.size(0))
        k1_runtime_num_tokens = num_tokens
    else:
        requested_tokens = int(dispatch_num_tokens)
        if requested_tokens == -1:
            if not hasattr(sym_buffer, "cuda_graph_num_tokens"):
                raise TypeError(
                    "dispatch_num_tokens=-1 requires sym_buffer.cuda_graph_num_tokens "
                    "to contain the local runtime token count"
                )
            num_tokens = int(y.size(0))
            k1_runtime_num_tokens = -1
        else:
            num_tokens = requested_tokens
            k1_runtime_num_tokens = num_tokens
    if num_tokens < 0 or num_tokens > int(sym_buffer.num_max_tokens_per_rank):
        raise ValueError(
            "V2 dispatch_num_tokens must be in "
            f"0..{int(sym_buffer.num_max_tokens_per_rank)}, or -1 to use "
            "sym-buffer runtime token counts"
        )

    def launch_k1() -> None:
        _get_k1_stage_launcher()(
            scratch.l1_out,
            l1_weights,
            sym_buffer,
            route_scratch=sym_buffer.route_scratch,
            staged_x=scratch.staged_x,
            staged_x_scale=scratch.staged_x_scale,
            problem_size=scratch.problem_size,
            row_expert=scratch.row_expert,
            route_weights=scratch.route_weights,
            output_index=scratch.output_index,
            row_combine_ptrs=scratch.row_combine_ptrs,
            local_topk_mask=scratch.local_topk_mask,
            tail_tokens=scratch.tail_tokens,
            grid_barrier=scratch.grid_barrier,
            route_scratch_i32=scratch.route_scratch_i32,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
            num_tokens=k1_runtime_num_tokens,
            num_ranks=num_ranks,
            num_global_experts=int(sym_buffer.num_experts),
            num_max_tokens_per_rank=int(sym_buffer.num_max_tokens_per_rank),
            num_topk=int(sym_buffer.num_topk),
            rank_idx=rank_idx,
            rows_aligned_per_expert=scratch.rows_aligned_per_expert,
            valid_rows_per_expert=scratch.valid_rows_per_expert,
            epoch=launch_epoch,
            backend=plan.backend,
        )

    _run_profiled_stage(profile_stages, "k1_ms", launch_k1)
    rows = scratch.launch_rows

    def launch_k2() -> None:
        _get_k2_stage_launcher()(
            scratch.l1_out.narrow(0, 0, rows),
            scratch.route_weights.narrow(0, 0, rows),
            scratch.act_fp8.narrow(0, 0, rows),
            scratch.act_scale.narrow(0, 0, rows),
            state.empty_bf16,
            num_per_channels=int(scratch.act_fp8.size(1)),
            output_bf16=False,
            clamp_value=activation_clamp,
            row_combine_ptrs=scratch.row_combine_ptrs.narrow(0, 0, rows),
        )

    _run_profiled_stage(profile_stages, "k2_ms", launch_k2)
    k3_epoch = state.epoch
    state.epoch += 1

    def launch_k3() -> None:
        _get_k3_stage_launcher()(
            y,
            l2_weights,
            scratch.act_fp8.narrow(0, 0, rows),
            scratch.act_scale.narrow(0, 0, rows),
            sym_buffer,
            route_scratch=sym_buffer.route_scratch,
            l2_workspace=scratch.l2_workspace,
            problem_size=scratch.route_scratch_i32.narrow(0, 0, scratch.local_experts),
            row_expert=scratch.row_expert,
            grid_barrier=scratch.grid_barrier,
            row_output_ptrs=scratch.row_output_ptrs,
            local_topk_mask=scratch.local_topk_mask,
            tail_tokens=scratch.tail_tokens,
            num_tokens=k1_runtime_num_tokens,
            num_ranks=num_ranks,
            num_global_experts=int(sym_buffer.num_experts),
            num_max_tokens_per_rank=int(sym_buffer.num_max_tokens_per_rank),
            num_topk=int(sym_buffer.num_topk),
            rank_idx=rank_idx,
            rows_aligned_per_expert=scratch.rows_aligned_per_expert,
            valid_rows_per_expert=scratch.valid_rows_per_expert,
            epoch=k3_epoch,
            k3_copy_workers=K3_COPY_WORKERS,
            backend=plan.backend,
            activation_clamp=activation_clamp,
        )

    _run_profiled_stage(profile_stages, "k3_ms", launch_k3)
    del (
        sym_buffer,
        activation_clamp,
        fast_math,
    )
    return None
