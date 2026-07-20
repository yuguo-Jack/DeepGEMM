import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import megamoe


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third-party"))

DEEPEP_BUFFER_BYTES = int(2.0e9)
DEEPEP_CONFIG = (24, 8, 256)
DEEPEP_EXPERT_ALIGNMENT = 256
DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD = 750.0
DCU_REFERENCE_HBM_GBPS_PER_CARD = 1600.0
DCU_REFERENCE_XHCL_GBPS_PER_CARD = 448.0
TILELANG_BENCH_BACKEND = "event"

import deep_ep
import deepgemm
from tilelang.profiler.bench import do_bench as tilelang_bench
from triton_ops import (
    triton_ep_gather_channelwise,
    triton_ep_scatter_channelwise,
)
from megamoe.dcu_megamoe_opt.v3_config import (
    BACKEND_ENV,
    NORMAL_LL_TOKEN_THRESHOLD_ENV,
    V3_BACKEND_AUTO,
    select_v3_backend,
    v3_backend_mode,
)
from megamoe.dcu_megamoe_opt.K2_fused.k2_fused import swiglu_quant_channelwise_out


UNIFIED_WEIGHT_LAYOUT_ENV = "MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT"
BASELINE_AUTO = "auto"
BASELINE_NORMAL_CONTIGUOUS = "normal-contiguous"
BASELINE_LL_MASKED = "ll-masked"
ROUTE_PATTERN_RANDOM = "random"
ROUTE_PATTERN_SINGLE_LOCAL_RANK = "single-local-rank"
DEFAULT_FP8_ATOL = 0.0035
DEFAULT_INT8_ATOL = 0.01


def env_flag_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def dcu_peer_memory_fabric_enabled() -> bool:
    value = os.getenv("MEGAMOE_DCU_PEER_MEMORY", "ipc").strip().lower()
    return value in {"1", "true", "yes", "on", "rpc", "fabric", "mnvl"}


def print_once(rank: int, msg: str = ""):
    if rank == 0:
        print(msg, flush=True)


def apply_route_pattern(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    pattern: str,
    target_rank: int,
    num_ranks: int,
    num_experts_per_rank: int,
    num_topk: int,
) -> None:
    if pattern == ROUTE_PATTERN_RANDOM:
        return
    if pattern != ROUTE_PATTERN_SINGLE_LOCAL_RANK:
        raise ValueError(f"unsupported route pattern: {pattern}")
    if target_rank < 0 or target_rank >= num_ranks:
        raise ValueError(f"route target rank {target_rank} is outside 0..{num_ranks - 1}")
    if num_topk > num_experts_per_rank:
        raise ValueError(
            "single-local-rank route pattern requires topk <= local experts "
            f"({num_topk} > {num_experts_per_rank})"
        )
    first_expert = int(target_rank) * int(num_experts_per_rank)
    experts = torch.arange(
        first_expert,
        first_expert + int(num_topk),
        device=topk_idx.device,
        dtype=topk_idx.dtype,
    )
    topk_idx.copy_(experts.view(1, -1).expand_as(topk_idx))
    topk_weights.fill_(1.0 / float(num_topk))


def baseline_kind_description(kind: str, prepost_backend: str) -> str:
    if kind == BASELINE_LL_MASKED:
        return "deepep_ll_dispatch_deepgemm_masked_k2_swiglu_quant_deepep_ll_combine_channelwise"
    return (
        f"deepep_dispatch_{prepost_backend}_scatter_deepgemm_k2_swiglu_quant_"
        f"{prepost_backend}_gather_deepep_combine_channelwise"
    )


_DEEPGEMM_CONTIGUOUS_WEIGHT_LAYOUT: str | None = None


def _probe_deepgemm_contiguous_weight_layout(device: torch.device) -> str:
    global _DEEPGEMM_CONTIGUOUS_WEIGHT_LAYOUT
    if _DEEPGEMM_CONTIGUOUS_WEIGHT_LAYOUT is not None:
        return _DEEPGEMM_CONTIGUOUS_WEIGHT_LAYOUT

    a_fp8 = torch.zeros((64, 128), device=device, dtype=torch.float8_e4m3fn)
    a_scale = torch.ones((64,), device=device, dtype=torch.float32)
    b_fp8 = torch.zeros((1, 256, 128), device=device, dtype=torch.float8_e4m3fn)
    b_scale = torch.ones((1, 256), device=device, dtype=torch.float32)
    output = torch.empty((64, 256), device=device, dtype=torch.bfloat16)
    m_indices = torch.zeros((64,), device=device, dtype=torch.int32)
    candidates = (
        (
            "k64n16_7d",
            lambda: megamoe.weight8bit_nt_kpack2_marlin_contiguous_k64n16(b_fp8),
        ),
        (
            "legacy_n16_flat",
            lambda: megamoe.weight8bit_nt_kpack2_marlin(b_fp8),
        ),
    )
    errors = []
    for name, pack in candidates:
        try:
            deepgemm.m_grouped_fp8_gemm_nt_contiguous(
                (a_fp8, a_scale),
                (pack(), b_scale),
                output,
                m_indices,
            )
            torch.cuda.synchronize()
            _DEEPGEMM_CONTIGUOUS_WEIGHT_LAYOUT = name
            return name
        except Exception as exc:
            torch.cuda.synchronize()
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "unable to find a compatible DeepGEMM contiguous weight layout: "
        + " | ".join(errors)
    )


def pack_deepgemm_contiguous_weight(weight: torch.Tensor) -> tuple[torch.Tensor, str]:
    layout = _probe_deepgemm_contiguous_weight_layout(weight.device)
    if layout == "k64n16_7d":
        return megamoe.weight8bit_nt_kpack2_marlin_contiguous_k64n16(weight), layout
    if layout == "legacy_n16_flat":
        return megamoe.weight8bit_nt_kpack2_marlin(weight), layout
    raise RuntimeError(f"unknown DeepGEMM contiguous weight layout: {layout}")


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def init_dist(local_rank: int, num_local_ranks: int):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", "1"))
    node_rank = int(os.getenv("RANK", "0"))
    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, dist.new_group(list(range(world_size)))


def parse_dispatch_result(result):
    if len(result) == 6:
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, handle, event = result
        recv_counts_cuda = None
    elif len(result) == 7:
        recv_x, recv_topk_idx, recv_topk_weights, recv_counts, recv_counts_cuda, handle, event = result
    else:
        raise RuntimeError(f"unexpected DeepEP dispatch return arity: {len(result)}")
    if hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
        event.current_stream_wait()
    return recv_x, recv_topk_idx, recv_topk_weights, recv_counts, recv_counts_cuda, handle


def unpack_recv_x_fp8_channelwise(recv_x):
    if isinstance(recv_x, tuple):
        recv_x_fp8, recv_x_scale = recv_x
        if recv_x_scale.dim() == 2:
            recv_x_scale = recv_x_scale[:, 0].contiguous()
    else:
        recv_x_fp8, recv_x_scale = megamoe.cast_to_fp8_channelwise(recv_x)
    return recv_x_fp8.contiguous(), recv_x_scale.contiguous()

def counts_to_gpu_and_total(recv_counts, recv_counts_cuda, device, recv_topk_idx):
    if recv_counts_cuda is not None:
        counts_gpu = recv_counts_cuda.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    elif torch.is_tensor(recv_counts):
        counts_gpu = recv_counts.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    else:
        counts_gpu = torch.tensor(recv_counts, dtype=torch.int32, device=device)

    if not torch.is_tensor(recv_counts):
        all_tokens = int(sum(int(v) for v in recv_counts))
    elif recv_counts.device.type == "cpu":
        all_tokens = int(recv_counts.sum().item())
    else:
        # Avoid a GPU->CPU sync in the timed path. This upper bound leaves unused rows
        # with m_indices=-1 and does not affect DeepGEMM output rows gathered back.
        all_tokens = int(recv_topk_idx.numel())
    return counts_gpu, max(all_tokens, 1)


def dispatch_deepep_normal(
    ep_buffer,
    x_fp8,
    x_scale,
    topk_idx,
    topk_weights,
    num_tokens_per_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    ep_config,
    expert_alignment,
):
    kwargs = dict(
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=ep_config,
        async_finish=False,
    )
    payload = (x_fp8, x_scale.view(-1, 1))
    if expert_alignment > 0 and getattr(dispatch_deepep_normal, "supports_alignment", True):
        try:
            return ep_buffer.dispatch(payload, expert_alignment=expert_alignment, **kwargs)
        except TypeError:
            dispatch_deepep_normal.supports_alignment = False
    return ep_buffer.dispatch(payload, **kwargs)


def create_deepep_low_latency_buffer(
    group,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
):
    allow_mnnvl = dcu_peer_memory_fabric_enabled()
    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        num_max_tokens_per_rank,
        hidden,
        num_ranks,
        num_experts,
    )
    try:
        buffer = deep_ep.Buffer(
            group,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts_per_rank,
            allow_mnnvl=allow_mnnvl,
            explicitly_destroy=True,
        )
    except TypeError:
        buffer = deep_ep.Buffer(
            group,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_experts_per_rank,
            explicitly_destroy=True,
        )
    if hasattr(buffer, "clean_low_latency_buffer"):
        buffer.clean_low_latency_buffer(
            num_max_tokens_per_rank,
            hidden,
            num_experts,
        )
    return buffer


def wait_deepep_event(event, hook=None):
    if hook is not None:
        hook()
    elif hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
        event.current_stream_wait()


def as_masked_deepgemm_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 2:
        return scale.unsqueeze(-1).contiguous()
    return scale.contiguous()


def as_masked_deepgemm_weights(weights: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    weight, scale = weights
    return weight, as_masked_deepgemm_scale(scale)


def deepgemm_masked_fp8_gemm(
    a: tuple[torch.Tensor, torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
):
    if getattr(deepgemm_masked_fp8_gemm, "needs_overlap_arg", False):
        deepgemm.m_grouped_fp8_gemm_nt_masked(
            a,
            weights,
            out,
            masked_m,
            expected_m,
            False,
            signal=None,
        )
        return
    try:
        deepgemm.m_grouped_fp8_gemm_nt_masked(
            a,
            weights,
            out,
            masked_m,
            expected_m,
        )
    except TypeError:
        deepgemm_masked_fp8_gemm.needs_overlap_arg = True
        deepgemm.m_grouped_fp8_gemm_nt_masked(
            a,
            weights,
            out,
            masked_m,
            expected_m,
            False,
            signal=None,
        )


def deepep_deepgemm_preprocess_channelwise(
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    recv_counts,
    recv_counts_cuda,
    backend: str,
):
    recv_x_fp8, recv_x_scale = unpack_recv_x_fp8_channelwise(recv_x)
    if recv_topk_idx.dim() == 1:
        recv_topk_idx = recv_topk_idx.view(-1, 1)
        recv_topk_weights = recv_topk_weights.view(-1, 1)
    recv_topk_idx = recv_topk_idx.contiguous()
    recv_topk_weights = recv_topk_weights.contiguous()
    counts_gpu, all_tokens = counts_to_gpu_and_total(
        recv_counts,
        recv_counts_cuda,
        recv_x_fp8.device,
        recv_topk_idx,
    )
    if backend == "hip":
        grouped_x, grouped_x_scale, route_weights, m_indices, output_index = (
            megamoe.deepep_deepgemm_preprocess_channelwise(
                recv_x_fp8,
                recv_x_scale,
                recv_topk_idx,
                recv_topk_weights,
                counts_gpu,
                all_tokens,
            )
        )
    elif backend == "triton":
        if triton_ep_scatter_channelwise is None:
            raise RuntimeError("triton_ops.triton_ep_scatter_channelwise is unavailable")
        grouped_x, grouped_x_scale, route_weights, m_indices, output_index = triton_ep_scatter_channelwise(
            recv_x_fp8,
            recv_x_scale,
            recv_topk_idx,
            recv_topk_weights,
            counts_gpu,
            all_tokens,
        )
    else:
        raise ValueError(f"unknown pre/post backend: {backend}")
    return {
        "a": (grouped_x, grouped_x_scale),
        "route_weights": route_weights,
        "m_indices": m_indices,
        "output_index": output_index,
        "recv_topk_idx": recv_topk_idx,
        "recv_topk_weights": recv_topk_weights,
        "recv_rows": recv_x_fp8.shape[0],
    }


def run_deepgemm_megamoe_baseline(
    ep_buffer,
    ep_config,
    x_fp8,
    x_scale,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    num_experts: int,
    num_experts_per_rank: int,
    intermediate_hidden: int,
    hidden: int,
    activation_clamp: float | None,
    expert_alignment: int,
    prepost_backend: str,
    layout_cache=None,
    return_stats: bool = False,
):
    if layout_cache is None:
        layout_result = ep_buffer.get_dispatch_layout(topk_idx, num_experts)
        if len(layout_result) == 5:
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, event = layout_result
        else:
            raise RuntimeError(f"unexpected DeepEP layout return arity: {len(layout_result)}")
        if hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
            event.current_stream_wait()
    else:
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = layout_cache

    recv = dispatch_deepep_normal(
        ep_buffer,
        x_fp8,
        x_scale,
        topk_idx,
        topk_weights,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        ep_config,
        expert_alignment,
    )
    recv_x, recv_topk_idx, recv_topk_weights, recv_counts, recv_counts_cuda, handle = parse_dispatch_result(recv)
    grouped = deepep_deepgemm_preprocess_channelwise(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_counts,
        recv_counts_cuda,
        prepost_backend,
    )

    l1_out = torch.empty(
        (grouped["a"][0].shape[0], intermediate_hidden * 2),
        device=x_fp8.device,
        dtype=torch.bfloat16,
    )
    deepgemm.m_grouped_fp8_gemm_nt_contiguous(
        grouped["a"],
        l1_weights,
        l1_out,
        grouped["m_indices"],
    )

    l2_a_fp8 = torch.empty(
        (l1_out.shape[0], intermediate_hidden),
        device=x_fp8.device,
        dtype=torch.float8_e4m3fn,
    )
    l2_a_scale = torch.empty((l1_out.shape[0],), device=x_fp8.device, dtype=torch.float32)
    l2_out = torch.empty(
        (grouped["a"][0].shape[0], hidden),
        device=x_fp8.device,
        dtype=torch.bfloat16,
    )
    swiglu_quant_channelwise_out(
        l1_out,
        grouped["route_weights"],
        l2_a_fp8,
        l2_a_scale,
        l2_out,
        num_per_channels=intermediate_hidden,
        output_bf16=False,
        clamp_value=activation_clamp,
        row_combine_ptrs=None,
    )
    deepgemm.m_grouped_fp8_gemm_nt_contiguous(
        (l2_a_fp8, l2_a_scale),
        l2_weights,
        l2_out,
        grouped["m_indices"],
    )

    recv_y = torch.empty((grouped["recv_rows"], hidden), device=x_fp8.device, dtype=torch.bfloat16)
    if prepost_backend == "hip":
        megamoe.deepep_deepgemm_postprocess_channelwise(
            recv_y,
            l2_out,
            grouped["recv_topk_idx"],
            grouped["recv_topk_weights"],
            grouped["output_index"],
            apply_topk_weights=False,
        )
    elif prepost_backend == "triton":
        if triton_ep_gather_channelwise is None:
            raise RuntimeError("triton_ops.triton_ep_gather_channelwise is unavailable")
        triton_ep_gather_channelwise(
            l2_out,
            grouped["recv_topk_idx"],
            grouped["recv_topk_weights"],
            grouped["output_index"],
            recv_y,
            apply_topk_weights=False,
        )
    else:
        raise ValueError(f"unknown pre/post backend: {prepost_backend}")

    combined = ep_buffer.combine(recv_y, handle=handle, config=ep_config)
    y_baseline = combined[0]
    if hasattr(combined[-1], "current_stream_wait") and getattr(combined[-1], "event", None) is not None:
        combined[-1].current_stream_wait()

    if return_stats:
        valid_routes = grouped["output_index"] >= 0
        valid_experts = grouped["recv_topk_idx"][valid_routes]
        if valid_experts.numel():
            stats = torch.bincount(valid_experts.to(torch.int64), minlength=num_experts_per_rank).to(torch.int32)
        else:
            stats = torch.zeros((num_experts_per_rank,), device=x_fp8.device, dtype=torch.int32)
    else:
        stats = None
    return y_baseline, stats, {
        "valid_m": int((grouped["output_index"] >= 0).sum().item()) if return_stats else int(recv_topk_idx.numel()),
        "recv_rows": grouped["recv_rows"],
        "grouped_rows": grouped["a"][0].shape[0],
    }


def run_deepep_ll_deepgemm_masked_baseline(
    ep_buffer,
    x_bf16,
    topk_idx,
    topk_weights,
    l1_weights,
    l2_weights,
    num_max_tokens_per_rank: int,
    _expected_tokens_per_rank: int,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_topk: int,
    intermediate_hidden: int,
    hidden: int,
    activation_clamp: float | None,
    fast_math: bool,
    out: torch.Tensor | None = None,
    return_stats: bool = False,
):
    topk_idx_i64 = topk_idx.to(torch.int64).contiguous()
    topk_weights = topk_weights.contiguous()
    packed_recv_hidden, masked_m, handle, event, hook = ep_buffer.low_latency_dispatch(
        x_bf16.contiguous(),
        topk_idx_i64,
        num_max_tokens_per_rank,
        num_experts,
        quant_type=2,
        quant_group_size=0,
        fp8_round_scale=False,
        async_finish=False,
        return_recv_hook=False,
    )
    wait_deepep_event(event)
    if not isinstance(packed_recv_hidden, tuple):
        raise RuntimeError("DeepEP LL baseline expects FP8 low_latency_dispatch output")

    recv_x_fp8, recv_x_scale = packed_recv_hidden
    recv_x_fp8 = recv_x_fp8.contiguous()
    recv_x_scale = as_masked_deepgemm_scale(recv_x_scale)
    masked_m = masked_m.to(device=x_bf16.device, dtype=torch.int32, non_blocking=True).contiguous()

    num_groups, m_per_expert, _ = recv_x_fp8.shape
    runtime_tokens_for_expected = int(x_bf16.shape[0])
    expected_m = max(
        1,
        (runtime_tokens_for_expected * int(num_ranks) * int(num_topk) + int(num_experts))
        // int(num_experts),
    )

    l1_out = torch.empty(
        (num_groups, m_per_expert, intermediate_hidden * 2),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    deepgemm_masked_fp8_gemm(
        (recv_x_fp8, recv_x_scale),
        as_masked_deepgemm_weights(l1_weights),
        l1_out,
        masked_m,
        expected_m,
    )

    flat_l1 = l1_out.view(num_groups * m_per_expert, intermediate_hidden * 2)
    flat_l2_fp8 = torch.empty(
        (num_groups * m_per_expert, intermediate_hidden),
        device=x_bf16.device,
        dtype=torch.float8_e4m3fn,
    )
    flat_l2_scale = torch.empty(
        (num_groups * m_per_expert,),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    dummy_l2_bf16 = torch.empty(
        (num_groups * m_per_expert, intermediate_hidden),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    swiglu_quant_channelwise_out(
        flat_l1,
        None,
        flat_l2_fp8,
        flat_l2_scale,
        dummy_l2_bf16,
        num_per_channels=intermediate_hidden,
        output_bf16=False,
        clamp_value=activation_clamp,
        row_combine_ptrs=None,
        actual_m=masked_m,
        m_per_expert=m_per_expert,
        fast_math=fast_math,
    )

    l2_out = torch.empty(
        (num_groups, m_per_expert, hidden),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    deepgemm_masked_fp8_gemm(
        (
            flat_l2_fp8.view(num_groups, m_per_expert, intermediate_hidden),
            flat_l2_scale.view(num_groups, m_per_expert, 1).contiguous(),
        ),
        as_masked_deepgemm_weights(l2_weights),
        l2_out,
        masked_m,
        expected_m,
    )

    combine_kwargs = dict(
        x=l2_out,
        topk_idx=topk_idx_i64,
        topk_weights=topk_weights,
        handle=handle,
        zero_copy=False,
        async_finish=False,
        return_recv_hook=False,
    )
    if out is not None:
        combine_kwargs["out"] = out
    y_baseline, event, hook = ep_buffer.low_latency_combine(**combine_kwargs)
    wait_deepep_event(event)

    stats = masked_m[:num_experts_per_rank].contiguous() if return_stats else None
    return y_baseline, stats, {
        "valid_m": int(topk_idx.numel()),
        "recv_rows": int(num_groups * m_per_expert),
        "grouped_rows": int(num_groups * m_per_expert),
        "expected_m": expected_m,
    }


def bench_tilelang_ms(fn, warmup: int, repeat: int, backend: str = TILELANG_BENCH_BACKEND) -> dict[str, float]:
    median_ms = tilelang_bench(
        fn,
        _n_warmup=warmup,
        _n_repeat=repeat,
        backend=backend,
        return_mode="median",
    )
    min_ms = tilelang_bench(
        fn,
        _n_warmup=warmup,
        _n_repeat=repeat,
        backend=backend,
        return_mode="min",
    )
    return {"median_ms": median_ms, "min_ms": min_ms, "backend": backend}


def safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def json_finite_or_none(value: float) -> float | None:
    return value if math.isfinite(float(value)) else None


def nonfinite_count(tensor: torch.Tensor) -> int:
    return int((~torch.isfinite(tensor.float())).sum().item()) if tensor.numel() else 0


def pct(numerator: float, denominator: float) -> float:
    return safe_div(numerator, denominator) * 100.0


def dcu_w8a8_effective_hbm_bytes(
    num_recv_tokens: int,
    num_touched_experts: int,
    hidden: int,
    intermediate_hidden: int,
) -> int:
    weight_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden +
        num_touched_experts * hidden * intermediate_hidden
    )
    weight_scale_bytes = num_touched_experts * (intermediate_hidden * 2 + hidden) * 4
    input_bytes = num_recv_tokens * (hidden + 4)
    l2_act_bytes = num_recv_tokens * (intermediate_hidden + 4) * 2
    output_bytes = num_recv_tokens * hidden * 2
    return weight_bytes + weight_scale_bytes + input_bytes + l2_act_bytes + output_bytes


def dcu_w8a8_effective_xhcl_bytes(num_recv_tokens: int, hidden: int) -> int:
    return num_recv_tokens * (hidden + 4 + hidden * 2)


def gather_rank_metrics(values: list[float], device: torch.device, group: dist.ProcessGroup) -> torch.Tensor:
    local = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, local, group=group)
    return torch.stack(gathered, dim=0)


def uneven_all_gather_topk_idx(topk_idx: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group=group)
    local_rows = torch.tensor([topk_idx.size(0)], dtype=torch.int64, device=topk_idx.device)
    row_tensors = [torch.empty_like(local_rows) for _ in range(world_size)]
    dist.all_gather(row_tensors, local_rows, group=group)
    rows = [int(t.item()) for t in row_tensors]
    max_rows = max(rows)

    padded = torch.full(
        (max_rows, *topk_idx.shape[1:]),
        -1,
        dtype=topk_idx.dtype,
        device=topk_idx.device,
    )
    padded[:topk_idx.size(0)].copy_(topk_idx)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    return torch.cat([tensor[:rows[i]] for i, tensor in enumerate(gathered)], dim=0)


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    exact_num_tokens_per_rank = None
    if args.num_tokens_per_rank_list:
        exact_num_tokens_per_rank = parse_int_list(args.num_tokens_per_rank_list)
        if len(exact_num_tokens_per_rank) != num_ranks:
            raise ValueError(
                f"--num-tokens-per-rank-list needs {num_ranks} entries, "
                f"got {len(exact_num_tokens_per_rank)}"
            )
        num_tokens = exact_num_tokens_per_rank[rank]
    elif args.num_tokens == 0:
        remove = random.randint(0, args.num_max_removed_tokens)
        num_tokens = max(0, num_max_tokens_per_rank - remove)
    else:
        num_tokens = args.num_tokens
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_experts % num_ranks == 0
    assert 0 <= num_tokens <= num_max_tokens_per_rank
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0

    backend_selector_tokens = num_tokens
    if exact_num_tokens_per_rank is not None:
        backend_selector_tokens = max(exact_num_tokens_per_rank)
    elif args.num_tokens == 0 and args.num_max_removed_tokens > 0:
        backend_selector_tokens = num_max_tokens_per_rank

    backend_mode = v3_backend_mode(args.megamoe_backend)
    v3_backend = select_v3_backend(backend_selector_tokens, backend_mode)
    quant_mode = str(args.quant_mode).strip().lower()
    atol_source = "explicit"
    if args.atol is None:
        args.atol = DEFAULT_INT8_ATOL if quant_mode == "int8" else DEFAULT_FP8_ATOL
        atol_source = f"{quant_mode}-default"
    if quant_mode == "int8":
        if v3_backend != "normal":
            raise ValueError("--quant-mode int8 requires --megamoe-backend normal")
        if args.cuda_graph:
            raise ValueError("--quant-mode int8 currently supports eager execution only")
    baseline_kind = (
        BASELINE_LL_MASKED
        if args.baseline_kind == BASELINE_AUTO and v3_backend == "ll"
        else (
            BASELINE_NORMAL_CONTIGUOUS
            if args.baseline_kind == BASELINE_AUTO
            else args.baseline_kind
        )
    )
    pro_shape = (
        int(num_experts),
        int(num_topk),
        int(hidden),
        int(intermediate_hidden),
    ) == (384, 6, 7168, 3072)
    force_unified_layout = env_flag_enabled(UNIFIED_WEIGHT_LAYOUT_ENV)
    if v3_backend == "ll" and pro_shape:
        weight_layout = "unified" if force_unified_layout else "ll_pro_masked"
    else:
        weight_layout = "normal" if v3_backend == "normal" else "unified"
    fused_execution = f"v3_{v3_backend}_{quant_mode}_eager"
    graph_execution = f"v3_{v3_backend}_cuda_graph_replay" if args.cuda_graph else "disabled"
    print_once(rank, "DCU MegaMoE channelwise W8A8 test:")
    print_once(rank, f" > megamoe: {getattr(megamoe, '__file__', None)}")
    print_once(rank, f" > deep_ep: {getattr(deep_ep, '__file__', None)}")
    print_once(rank, f" > deepgemm: {getattr(deepgemm, '__file__', None)}")
    print_once(rank, f" > build config: {megamoe.get_mega_moe_hip_build_config()}")
    print_once(rank, f" > fused execution={fused_execution}")
    print_once(rank, f" > fused quant mode={quant_mode}")
    print_once(rank, f" > cuda graph execution={graph_execution}")

    sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        quant_mode=quant_mode,
    )
    ll_masked_baseline = baseline_kind == BASELINE_LL_MASKED
    baseline_execution = (
        "ll_masked_cuda_graph_replay"
        if ll_masked_baseline
        else "normal_contiguous_eager"
    )
    if args.skip_baseline_bench and args.correctness_iters == 0:
        baseline_execution = "skipped"
    ll_baseline_capacity_tokens = int(sym_buffer.cuda_graph_max_tokens_per_rank)
    if ll_masked_baseline and num_tokens > ll_baseline_capacity_tokens:
        raise ValueError(
            "--baseline-kind ll-masked uses the MegaMoE graph capacity; "
            f"num_tokens={num_tokens} exceeds capacity={ll_baseline_capacity_tokens}"
        )
    if ll_masked_baseline:
        needs_deepep_baseline = (
            args.correctness_iters > 0
            or (not args.skip_bench and not args.skip_baseline_bench)
            or (
                args.cuda_graph
                and not args.cuda_graph_skip_baseline
            )
        )
    else:
        needs_deepep_baseline = (
            args.correctness_iters > 0
            or (not args.skip_bench and not args.skip_baseline_bench)
            or (
                args.cuda_graph
                and not args.cuda_graph_skip_baseline
            )
        )
    ep_buffer = None
    ep_config = None
    if needs_deepep_baseline:
        if ll_masked_baseline:
            ep_buffer = create_deepep_low_latency_buffer(
                group,
                ll_baseline_capacity_tokens,
                hidden,
                num_ranks,
                num_experts,
                num_experts_per_rank,
            )
        else:
            ep_buffer = deep_ep.Buffer(
                group,
                DEEPEP_BUFFER_BYTES,
                0,
                explicitly_destroy=True,
            )
            ep_config = deep_ep.Config(*DEEPEP_CONFIG)

    x_bf16 = (torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * args.input_scale)
    l1_bf16 = (
        torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device="cuda")
        * args.weight_scale
    )
    l2_bf16 = (
        torch.randn((num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda")
        * args.weight_scale
    )
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_weights = torch.softmax(topk_weights.float(), dim=-1)
    apply_route_pattern(
        topk_idx,
        topk_weights,
        pattern=args.route_pattern,
        target_rank=args.route_target_rank,
        num_ranks=num_ranks,
        num_experts_per_rank=num_experts_per_rank,
        num_topk=num_topk,
    )

    l1_fp8, l1_scale = megamoe.cast_grouped_weight_to_fp8_channelwise(l1_bf16)
    l2_fp8, l2_scale = megamoe.cast_grouped_weight_to_fp8_channelwise(l2_bf16)
    baseline_contiguous_weight_layout = None
    masked_l1_weights = None
    masked_l2_weights = None
    if ll_masked_baseline or weight_layout == "ll_pro_masked":
        masked_l1_weights = (megamoe.weight8bit_nt_kpack2_marlin_masked(l1_fp8), l1_scale)
        masked_l2_weights = (megamoe.weight8bit_nt_kpack2_marlin_masked(l2_fp8), l2_scale)
    if ll_masked_baseline:
        baseline_l1_weights = masked_l1_weights
        baseline_l2_weights = masked_l2_weights
    else:
        baseline_l1_packed, baseline_contiguous_weight_layout = pack_deepgemm_contiguous_weight(l1_fp8)
        baseline_l2_packed, l2_contiguous_weight_layout = pack_deepgemm_contiguous_weight(l2_fp8)
        if l2_contiguous_weight_layout != baseline_contiguous_weight_layout:
            raise RuntimeError(
                "DeepGEMM contiguous baseline selected inconsistent L1/L2 layouts: "
                f"{baseline_contiguous_weight_layout} vs {l2_contiguous_weight_layout}"
            )
        baseline_l1_weights = (baseline_l1_packed, l1_scale)
        baseline_l2_weights = (baseline_l2_packed, l2_scale)
    if quant_mode == "int8":
        fused_l1_weights, fused_l2_weights = (
            megamoe.transform_int8_weights_for_mega_moe_normal(
                l1_bf16,
                l2_bf16,
            )
        )
        layout_desc = "YGZP signed-INT8 normal ASM plain pack5"
    elif weight_layout == "normal":
        fused_l1_weights = {
            "normal": (megamoe.flatten_pack5_weight_asm_normal(l1_fp8), l1_scale),
        }
        fused_l2_weights = {
            "normal": (megamoe.flatten_pack5_weight_asm_normal(l2_fp8), l2_scale),
        }
        layout_desc = "single normal ASM plain pack5"
    elif weight_layout == "ll_pro_masked":
        fused_l1_weights = {
            "ll_pro_masked": masked_l1_weights,
        }
        fused_l2_weights = {
            "unified": (megamoe.flatten_pack5_weight(l2_fp8), l2_scale),
        }
        layout_desc = "Pro LL masked-K1 L1 layout plus unified L2 pack5"
    else:
        fused_l1_weights = {
            "unified": (megamoe.flatten_pack5_weight(l1_fp8), l1_scale),
        }
        fused_l2_weights = {
            "unified": (megamoe.flatten_pack5_weight(l2_fp8), l2_scale),
        }
        layout_desc = "single unified transposed pack5"
    print_once(rank, f" > V3 staged layout: {layout_desc}")
    if baseline_contiguous_weight_layout is not None:
        print_once(
            rank,
            f" > DeepGEMM contiguous baseline weight layout: {baseline_contiguous_weight_layout}",
        )
    stats_initial = torch.randint(0, 100, (num_experts_per_rank,), dtype=torch.int32, device="cuda")
    stats_fused = stats_initial.clone()
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    baseline_layout_cache = None
    normal_baseline_predispatch_buffers = {}

    def copy_inputs_to_sym_buffer():
        pre_dispatch = (
            megamoe.mega_moe_pre_dispatch_int8
            if quant_mode == "int8"
            else megamoe.mega_moe_pre_dispatch
        )
        pre_dispatch(
            x_bf16,
            topk_idx,
            topk_weights,
            sym_buffer.x,
            sym_buffer.x_sf,
            sym_buffer.topk_idx,
            sym_buffer.topk_weights,
            num_tokens=num_tokens,
        )

    def run_fused(reset_stats: bool = False):
        if reset_stats:
            stats_fused.copy_(stats_initial)
        copy_inputs_to_sym_buffer()
        fused_api = (
            megamoe.int8_w8a8_mega_moe
            if quant_mode == "int8"
            else megamoe.fp8_w8a8_mega_moe
        )
        fused_api(
            y_fused,
            fused_l1_weights,
            fused_l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats=stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            megamoe_backend=v3_backend,
            capacity_num_tokens=backend_selector_tokens,
        )
        return y_fused, stats_fused

    def get_baseline_layout_cache():
        nonlocal baseline_layout_cache
        if baseline_kind != BASELINE_NORMAL_CONTIGUOUS:
            return None
        if baseline_layout_cache is None:
            layout_result = ep_buffer.get_dispatch_layout(topk_idx, num_experts)
            if len(layout_result) != 5:
                raise RuntimeError(f"unexpected DeepEP layout return arity: {len(layout_result)}")
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, event = layout_result
            if hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
                event.current_stream_wait()
            baseline_layout_cache = (num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)
        return baseline_layout_cache

    def run_selected_baseline(
        x_bf16_arg,
        topk_idx_arg,
        topk_weights_arg,
        expected_tokens_per_rank: int,
        return_stats: bool,
        out: torch.Tensor | None = None,
        use_layout_cache: bool = True,
    ):
        if baseline_kind == BASELINE_LL_MASKED:
            # The DeepEP LL buffer is allocated for the graph maximum, but each
            # baseline graph is captured per runtime bucket.  Keep dispatch
            # capacity aligned with MegaMoE's runtime token bucket instead of
            # the larger allocation cap, otherwise small replay buckets measure
            # extra baseline-only padding work.
            baseline_dispatch_tokens = max(
                int(expected_tokens_per_rank),
                int(x_bf16_arg.size(0)),
            )
            return run_deepep_ll_deepgemm_masked_baseline(
                ep_buffer,
                x_bf16_arg,
                topk_idx_arg,
                topk_weights_arg,
                baseline_l1_weights,
                baseline_l2_weights,
                baseline_dispatch_tokens,
                expected_tokens_per_rank,
                num_ranks,
                num_experts,
                num_experts_per_rank,
                num_topk,
                intermediate_hidden,
                hidden,
                args.activation_clamp,
                bool(args.fast_math),
                out=out,
                return_stats=return_stats,
            )

        rows = int(x_bf16_arg.size(0))
        cache_key = (rows, int(x_bf16_arg.size(1)), int(topk_idx_arg.size(1)))
        cached = normal_baseline_predispatch_buffers.get(cache_key)
        if cached is None:
            cached = (
                torch.empty(
                    (rows, int(x_bf16_arg.size(1))),
                    device=x_bf16_arg.device,
                    dtype=torch.float8_e4m3fn,
                ),
                torch.empty((rows,), device=x_bf16_arg.device, dtype=torch.float32),
                torch.empty((rows, topk_idx_arg.size(1)), device=x_bf16_arg.device, dtype=torch.int64),
                torch.empty((rows, topk_idx_arg.size(1)), device=x_bf16_arg.device, dtype=torch.float32),
            )
            normal_baseline_predispatch_buffers[cache_key] = cached
        baseline_x_fp8, baseline_x_scale, baseline_topk_idx, baseline_topk_weights = cached
        megamoe.mega_moe_pre_dispatch(
            x_bf16_arg,
            topk_idx_arg,
            topk_weights_arg,
            baseline_x_fp8,
            baseline_x_scale,
            baseline_topk_idx,
            baseline_topk_weights,
            num_tokens=rows,
        )
        layout_cache = get_baseline_layout_cache() if use_layout_cache else None
        return run_deepgemm_megamoe_baseline(
            ep_buffer,
            ep_config,
            baseline_x_fp8,
            baseline_x_scale,
            baseline_topk_idx,
            baseline_topk_weights,
            baseline_l1_weights,
            baseline_l2_weights,
            num_experts,
            num_experts_per_rank,
            intermediate_hidden,
            hidden,
            args.activation_clamp,
            DEEPEP_EXPERT_ALIGNMENT,
            args.prepost_backend,
            layout_cache=layout_cache,
            return_stats=return_stats,
        )

    expected_local_counts_cache = None

    def get_expected_local_counts_eager():
        nonlocal expected_local_counts_cache
        if expected_local_counts_cache is not None:
            return expected_local_counts_cache
        gathered_topk = uneven_all_gather_topk_idx(topk_idx, group=group)
        local_begin = rank * num_experts_per_rank
        local_end = local_begin + num_experts_per_rank
        local_mask = (gathered_topk >= local_begin) & (gathered_topk < local_end)
        local_ids = (gathered_topk[local_mask] - local_begin).to(torch.int64)
        counts = torch.bincount(local_ids, minlength=num_experts_per_rank).to(torch.int32)
        expected_local_counts_cache = counts.contiguous()
        return expected_local_counts_cache

    ll_baseline_graph_cache = {}

    def clean_ll_baseline_buffer():
        if ll_masked_baseline and hasattr(ep_buffer, "clean_low_latency_buffer"):
            ep_buffer.clean_low_latency_buffer(
                ll_baseline_capacity_tokens,
                hidden,
                num_experts,
            )

    def get_ll_masked_baseline_graph(
        cache_name: str,
        token_count: int,
        local_token_count: int,
        baseline_x_bf16,
        baseline_topk_idx,
        baseline_topk_weights,
    ):
        key = (cache_name, int(token_count), int(local_token_count))
        cached = ll_baseline_graph_cache.get(key)
        if cached is not None:
            return cached

        baseline_graph_y = torch.empty(
            (local_token_count, hidden),
            device=baseline_x_bf16.device,
            dtype=torch.bfloat16,
        )

        def run_baseline_graph_once():
            run_selected_baseline(
                baseline_x_bf16[:local_token_count],
                baseline_topk_idx[:local_token_count],
                baseline_topk_weights[:local_token_count],
                token_count,
                return_stats=False,
                out=baseline_graph_y,
            )

        clean_ll_baseline_buffer()
        torch.cuda.synchronize()
        for _ in range(args.cuda_graph_warmup):
            run_baseline_graph_once()
        torch.cuda.synchronize()

        clean_ll_baseline_buffer()
        torch.cuda.synchronize()
        baseline_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline_graph):
            run_baseline_graph_once()
        cached = (baseline_graph, baseline_graph_y)
        ll_baseline_graph_cache[key] = cached
        return cached

    def run_baseline(return_stats: bool):
        if ll_masked_baseline:
            baseline_graph, baseline_y = get_ll_masked_baseline_graph(
                "main",
                backend_selector_tokens,
                num_tokens,
                x_bf16,
                topk_idx,
                topk_weights,
            )
            baseline_graph.replay()
            baseline_stats = (
                stats_initial + get_expected_local_counts_eager()
                if return_stats
                else None
            )
            return baseline_y, baseline_stats, {
                "valid_m": int(topk_idx.numel()),
                "recv_rows": int(num_tokens),
                "grouped_rows": int(num_tokens),
                "baseline_graph": True,
            }
        baseline_y, baseline_counts, meta = run_selected_baseline(
            x_bf16,
            topk_idx,
            topk_weights,
            backend_selector_tokens,
            return_stats,
        )
        baseline_stats = stats_initial + baseline_counts if baseline_counts is not None else None
        return baseline_y, baseline_stats, meta

    def run_zero_weight_reuse_check():
        if v3_backend != "normal":
            raise ValueError("--check-zero-weight-reuse requires --megamoe-backend normal")
        if num_tokens <= 0:
            raise ValueError("--check-zero-weight-reuse requires at least one local token")
        saved_topk_weights = topk_weights.clone()
        try:
            nonzero_y, _ = run_fused(reset_stats=True)
            torch.cuda.synchronize()
            if nonfinite_count(nonzero_y):
                raise AssertionError("zero-weight reuse precondition produced nonfinite output")
            if int(torch.count_nonzero(nonzero_y).item()) == 0:
                raise AssertionError(
                    "zero-weight reuse precondition did not populate combine output"
                )

            topk_weights.zero_()
            zero_y, zero_stats = run_fused(reset_stats=True)
            torch.cuda.synchronize()
            zero_nonfinite = nonfinite_count(zero_y)
            zero_max_abs = zero_y.float().abs().max().item() if zero_y.numel() else 0.0
            expected_stats = stats_initial + get_expected_local_counts_eager()
            if zero_nonfinite:
                raise AssertionError(
                    f"zero-weight reuse output has {zero_nonfinite} nonfinite values"
                )
            if zero_max_abs > args.atol:
                raise AssertionError(
                    "zero-weight reuse exposed stale combine data: "
                    f"max_abs={zero_max_abs} exceeds --atol={args.atol}"
                )
            if not torch.equal(zero_stats, expected_stats):
                raise AssertionError(
                    "zero-weight reuse stats mismatch: "
                    f"fused={zero_stats}, expected={expected_stats}"
                )
            print_once(
                rank,
                "Zero-weight same-buffer reuse: "
                f"max_abs={zero_max_abs:.6g}, stats_exact=True",
            )
        finally:
            topk_weights.copy_(saved_topk_weights)

    def run_cuda_graph_bucket_check():
        max_capture_tokens = int(sym_buffer.cuda_graph_max_tokens_per_rank)
        if args.cuda_graph_test_tokens:
            token_list = parse_int_list(args.cuda_graph_test_tokens)
        else:
            token_list = [max_capture_tokens if args.num_tokens == 0 else num_tokens]
        if not token_list:
            raise ValueError("--cuda-graph-test-tokens did not contain any token counts")
        for token in token_list:
            if token <= 0 or token > max_capture_tokens:
                raise ValueError(
                    f"graph test token {token} is outside 1..{max_capture_tokens}"
                )

        graph_x_bf16 = (
            torch.randn((max_capture_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            * args.input_scale
        )
        graph_scores = torch.randn(
            (max_capture_tokens, num_experts),
            dtype=torch.float32,
            device="cuda",
        )
        graph_topk_weights, graph_topk_idx = torch.topk(
            graph_scores, num_topk, dim=-1, largest=True, sorted=False
        )
        graph_topk_weights = torch.softmax(graph_topk_weights.float(), dim=-1)
        apply_route_pattern(
            graph_topk_idx,
            graph_topk_weights,
            pattern=args.route_pattern,
            target_rank=args.route_target_rank,
            num_ranks=num_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_topk=num_topk,
        )
        y_graph = torch.empty(
            (max_capture_tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )

        def local_graph_token_count(token_count: int) -> int:
            if exact_num_tokens_per_rank is not None:
                return min(max(exact_num_tokens_per_rank[rank], 0), token_count)
            if not (args.num_tokens == 0 and args.num_max_removed_tokens > 0):
                return token_count
            span = min(args.num_max_removed_tokens, token_count)
            if span <= 0:
                return token_count
            return max(0, token_count - ((rank * 17 + 7) % (span + 1)))

        def fill_graph_inputs(token_count: int) -> int:
            local_token_count = local_graph_token_count(token_count)
            sym_buffer.cuda_graph_num_tokens.fill_(local_token_count)
            return local_token_count

        def capture_megamoe_graph(graph_capture_tokens: int):
            def run_graph_bucket_once():
                megamoe.mega_moe_pre_dispatch(
                    graph_x_bf16,
                    graph_topk_idx,
                    graph_topk_weights,
                    sym_buffer.x,
                    sym_buffer.x_sf,
                    sym_buffer.topk_idx,
                    sym_buffer.topk_weights,
                    num_tokens=graph_capture_tokens,
                )
                megamoe.fp8_w8a8_mega_moe(
                    y_graph,
                    fused_l1_weights,
                    fused_l2_weights,
                    sym_buffer,
                    cumulative_local_expert_recv_stats=None,
                    activation_clamp=args.activation_clamp,
                    fast_math=bool(args.fast_math),
                    megamoe_backend=v3_backend,
                    graph=True,
                    capacity_num_tokens=graph_capture_tokens,
                )

            fill_graph_inputs(graph_capture_tokens)
            torch.cuda.synchronize()
            graph_warmup_stream = torch.cuda.Stream()
            graph_warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(graph_warmup_stream):
                for _ in range(args.cuda_graph_warmup):
                    run_graph_bucket_once()
            torch.cuda.current_stream().wait_stream(graph_warmup_stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_graph_bucket_once()
            return graph

        graph_timing_rows = []
        single_capture_graph = None
        if args.cuda_graph_single_capture:
            single_capture_graph = capture_megamoe_graph(max_capture_tokens)
        for token in token_list:
            graph_capture_tokens = (
                max_capture_tokens if args.cuda_graph_single_capture else token
            )
            graph = (
                single_capture_graph
                if single_capture_graph is not None
                else capture_megamoe_graph(graph_capture_tokens)
            )
            local_token = fill_graph_inputs(token)
            for _ in range(args.cuda_graph_replays):
                graph.replay()
            torch.cuda.synchronize()
            if args.cuda_graph_skip_baseline:
                print_once(
                    rank,
                    f"CUDA graph bucket token={token}/{graph_capture_tokens}, local={local_token}: "
                    f"replays={args.cuda_graph_replays}, baseline_skipped=True",
                )
                continue
            if baseline_kind == BASELINE_LL_MASKED:
                baseline_capacity_token = (
                    graph_capture_tokens if args.cuda_graph_single_capture else token
                )
                baseline_graph, baseline_y = get_ll_masked_baseline_graph(
                    "graph_bucket",
                    baseline_capacity_token,
                    local_token,
                    graph_x_bf16,
                    graph_topk_idx,
                    graph_topk_weights,
                )
                for _ in range(args.cuda_graph_replays):
                    baseline_graph.replay()
                torch.cuda.synchronize()
            else:
                baseline_y, _, _ = run_selected_baseline(
                    graph_x_bf16[:local_token],
                    graph_topk_idx[:local_token],
                    graph_topk_weights[:local_token],
                    token,
                    return_stats=False,
                    use_layout_cache=False,
                )
            graph_nonfinite = nonfinite_count(y_graph[:local_token])
            baseline_nonfinite = nonfinite_count(baseline_y)
            diff = (y_graph[:local_token].float() - baseline_y.float()).abs()
            diff_nonfinite = nonfinite_count(diff)
            max_abs = diff.max().item() if diff.numel() else 0.0
            mean_abs = diff.mean().item() if diff.numel() else 0.0
            if graph_nonfinite or baseline_nonfinite or diff_nonfinite:
                raise AssertionError(
                    f"cuda graph bucket token={token} local_token={local_token} "
                    f"nonfinite graph={graph_nonfinite} baseline={baseline_nonfinite} "
                    f"diff={diff_nonfinite}"
                )
            if max_abs > args.atol:
                flat_idx = int(diff.argmax().item()) if diff.numel() else 0
                row_idx = flat_idx // hidden if hidden else 0
                col_idx = flat_idx % hidden if hidden else 0
                graph_value = (
                    float(y_graph[:local_token].flatten()[flat_idx].float().item())
                    if diff.numel()
                    else 0.0
                )
                baseline_value = (
                    float(baseline_y.flatten()[flat_idx].float().item())
                    if diff.numel()
                    else 0.0
                )
                graph_runtime_value = (
                    int(sym_buffer.cuda_graph_num_tokens.detach().cpu().item())
                    if hasattr(sym_buffer, "cuda_graph_num_tokens")
                    else -1
                )
                graph_active_tiles_value = (
                    int(sym_buffer.route_scratch.view(torch.int32)[64].detach().cpu().item())
                    if hasattr(sym_buffer, "route_scratch")
                    else -1
                )
                raise AssertionError(
                    f"cuda graph bucket token={token} local_token={local_token} "
                    f"max_abs={max_abs} exceeds --atol={args.atol}; "
                    f"argmax=({row_idx},{col_idx}) "
                    f"graph={graph_value} baseline={baseline_value}; "
                    f"graph_runtime={graph_runtime_value} "
                    f"active_tiles={graph_active_tiles_value}"
                )
            print_once(
                rank,
                f"CUDA graph bucket token={token}/{graph_capture_tokens}, local={local_token}: "
                f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, "
                f"replays={args.cuda_graph_replays}",
            )
            if args.cuda_graph_bench:
                local_token = fill_graph_inputs(token)
                torch.cuda.synchronize()
                graph_timing = bench_tilelang_ms(
                    lambda: graph.replay(),
                    args.warmup,
                    args.repeat,
                )
                rank_graph_metrics = gather_rank_metrics(
                    [
                        graph_timing["median_ms"] / 1e3,
                        graph_timing["min_ms"] / 1e3,
                    ],
                    y_graph.device,
                    group,
                )
                baseline_rank_graph_metrics = None
                baseline_graph_timing = None
                if baseline_kind == BASELINE_LL_MASKED and not args.cuda_graph_skip_baseline:
                    baseline_capacity_token = (
                        graph_capture_tokens if args.cuda_graph_single_capture else token
                    )
                    baseline_graph, _ = get_ll_masked_baseline_graph(
                        "graph_bucket",
                        baseline_capacity_token,
                        local_token,
                        graph_x_bf16,
                        graph_topk_idx,
                        graph_topk_weights,
                    )
                    baseline_graph_timing = bench_tilelang_ms(
                        lambda: baseline_graph.replay(),
                        args.warmup,
                        args.repeat,
                    )
                    baseline_rank_graph_metrics = gather_rank_metrics(
                        [
                            baseline_graph_timing["median_ms"] / 1e3,
                            baseline_graph_timing["min_ms"] / 1e3,
                        ],
                        y_graph.device,
                        group,
                    )
                if rank == 0:
                    graph_avg_s, graph_min_s = rank_graph_metrics.mean(dim=0).cpu().tolist()
                    graph_timing_rows.append(
                        {
                            "tokens": token,
                            "local_tokens_rank0": local_token,
                            "graph_max_tokens_per_rank": max_capture_tokens,
                            "graph_capture_tokens_per_rank": graph_capture_tokens,
                            "cuda_graph_single_capture": bool(args.cuda_graph_single_capture),
                            "graph_replay_median_ms_avg_per_rank": graph_avg_s * 1e3,
                            "graph_replay_min_ms_avg_per_rank": graph_min_s * 1e3,
                            "bench_backend": f"tilelang_{graph_timing['backend']}",
                            "graph_execution": graph_execution,
                            "includes_input_update": False,
                            "includes_host_input_update": False,
                        }
                    )
                    if baseline_rank_graph_metrics is not None:
                        baseline_graph_avg_s, baseline_graph_min_s = (
                            baseline_rank_graph_metrics.mean(dim=0).cpu().tolist()
                        )
                        graph_timing_rows[-1].update(
                            {
                                "baseline_graph_replay_median_ms_avg_per_rank": baseline_graph_avg_s * 1e3,
                                "baseline_graph_replay_min_ms_avg_per_rank": baseline_graph_min_s * 1e3,
                                "baseline_graph_kind": BASELINE_LL_MASKED,
                                "baseline_graph_bench_backend": f"tilelang_{baseline_graph_timing['backend']}",
                            }
                        )
                    print_once(
                        rank,
                        f"CUDA graph bucket bench token={token}/{graph_capture_tokens}: "
                        f"local_rank0={local_token}, "
                        f"median={graph_avg_s * 1e6:.1f} us, min={graph_min_s * 1e6:.1f} us "
                        "(replay only)",
                    )
        return graph_timing_rows

    print_once(rank, "Config:")
    print_once(rank, f" > ranks={num_ranks}, tokens={num_tokens}/{sym_buffer.num_max_tokens_per_rank}")
    print_once(rank, f" > megamoe_backend={v3_backend} (mode={backend_mode}, selector_tokens={backend_selector_tokens})")
    print_once(rank, f" > hidden={hidden}, intermediate={intermediate_hidden}")
    print_once(rank, f" > experts={num_experts}, topk={num_topk}, local_experts={num_experts_per_rank}")
    print_once(rank, f" > scale modes=input/weight/l2_act channelwise fp32")
    print_once(rank, f" > correctness atol={args.atol} ({atol_source})")
    baseline_desc = baseline_kind_description(baseline_kind, args.prepost_backend)
    print_once(rank, f" > baseline={baseline_desc} (kind={baseline_kind}, requested={args.baseline_kind})")
    print_once(
        rank,
        f" > route_pattern={args.route_pattern}, route_target_rank={args.route_target_rank}",
    )
    print_once(rank, " > router weights=CUDA MegaMoE compatible SwiGLU-pre-L2-quant")
    print_once(
        rank,
        f" > sym_buffer={sym_buffer.buffer.nbytes / 2 ** 30:.3f} GiB "
        f"peer_mode={sym_buffer.handle.peer_memory_mode}",
    )
    print_once(rank, f" > route_scratch={sym_buffer.route_scratch.nbytes / 2 ** 30:.3f} GiB")

    if ll_masked_baseline and (args.correctness_iters > 0 or not args.skip_bench):
        get_ll_masked_baseline_graph(
            "main",
            backend_selector_tokens,
            num_tokens,
            x_bf16,
            topk_idx,
            topk_weights,
        )

    if args.check_zero_weight_reuse:
        run_zero_weight_reuse_check()

    correctness_metrics = []
    for i in range(args.correctness_iters):
        fused_y, fused_stats = run_fused(reset_stats=True)
        baseline_y, baseline_stats, _ = run_baseline(return_stats=True)
        # The DeepEP baseline oracle uses internal async streams; isolate it
        # before comparing or starting the next fused iteration.
        torch.cuda.synchronize()
        fused_nonfinite = nonfinite_count(fused_y)
        baseline_nonfinite = nonfinite_count(baseline_y)
        diff = (fused_y.float() - baseline_y.float()).abs()
        diff_nonfinite = nonfinite_count(diff)
        if baseline_stats is None:
            raise AssertionError("baseline stats missing in correctness check")
        stats_ok = torch.equal(fused_stats, baseline_stats)
        correctness_status = torch.tensor(
            [
                int(fused_nonfinite),
                int(baseline_nonfinite),
                int(diff_nonfinite),
                int(not stats_ok),
            ],
            dtype=torch.int64,
            device=diff.device,
        )
        dist.all_reduce(correctness_status, op=dist.ReduceOp.SUM, group=group)
        global_status = [int(value) for value in correctness_status.cpu().tolist()]
        if any(global_status[:3]):
            raise AssertionError(
                "fused/baseline nonfinite rank-counts "
                f"fused={global_status[0]} baseline={global_status[1]} "
                f"diff={global_status[2]}"
            )
        if global_status[3]:
            raise AssertionError(
                f"stats mismatch on {global_status[3]} rank(s): "
                f"local fused={fused_stats}, baseline={baseline_stats}"
            )

        global_max_abs_tensor = (
            diff.max()
            if diff.numel()
            else torch.zeros((), dtype=torch.float32, device=diff.device)
        )
        dist.all_reduce(global_max_abs_tensor, op=dist.ReduceOp.MAX, group=group)
        global_sum_count = torch.tensor(
            [float(diff.sum().item()), float(diff.numel())],
            dtype=torch.float64,
            device=diff.device,
        )
        dist.all_reduce(global_sum_count, op=dist.ReduceOp.SUM, group=group)
        global_max_abs = float(global_max_abs_tensor.item())
        global_abs_sum, global_element_count = global_sum_count.cpu().tolist()
        global_mean_abs = (
            float(global_abs_sum / global_element_count)
            if global_element_count
            else 0.0
        )
        correctness_metrics.append(
            {
                "iteration": i + 1,
                "max_abs": global_max_abs,
                "mean_abs": global_mean_abs,
            }
        )
        if global_max_abs > args.atol:
            raise AssertionError(
                f"fused/baseline global max_abs={global_max_abs} "
                f"exceeds --atol={args.atol}"
            )
        print_once(
            rank,
            f"Correctness {i + 1}/{args.correctness_iters}: "
            f"max_abs={global_max_abs:.6g}, mean_abs={global_mean_abs:.6g} "
            "(all ranks)",
        )

    cuda_graph_timing_rows = (
        run_cuda_graph_bucket_check()
        if args.cuda_graph
        else []
    )

    if args.skip_bench:
        if rank == 0:
            correctness_max_abs = max(
                (row["max_abs"] for row in correctness_metrics),
                default=0.0,
            )
            correctness_mean_abs = (
                sum(row["mean_abs"] for row in correctness_metrics)
                / len(correctness_metrics)
                if correctness_metrics
                else 0.0
            )
            result = {
                "correct": True,
                "bench_skipped": True,
                "reason": "--skip-bench",
                "num_ranks": num_ranks,
                "num_tokens_per_rank": num_tokens,
                "backend_selector_tokens": backend_selector_tokens,
                "megamoe_backend": v3_backend,
                "megamoe_backend_mode": backend_mode,
                "quant_mode": quant_mode,
                "weight_layout": weight_layout,
                "hidden": hidden,
                "intermediate_hidden": intermediate_hidden,
                "num_experts": num_experts,
                "num_topk": num_topk,
                "prepost_backend": args.prepost_backend,
                "baseline_kind": baseline_desc,
                "baseline_kind_resolved": baseline_kind,
                "baseline_kind_requested": args.baseline_kind,
                "baseline_execution": baseline_execution,
                "fused_execution": fused_execution,
                "route_pattern": args.route_pattern,
                "route_target_rank": args.route_target_rank,
                "fused_timing_scope": "eager_main_call",
                "baseline_timing_scope": baseline_execution,
                "cuda_graph_requested": bool(args.cuda_graph),
                "graph_execution": graph_execution,
                "correctness_iters": args.correctness_iters,
                "atol": args.atol,
                "atol_source": atol_source,
                "max_abs": correctness_max_abs,
                "mean_abs": correctness_mean_abs,
                "correctness_metrics": correctness_metrics,
                "cuda_graph_timings": cuda_graph_timing_rows,
            }
            print("Correctness-only:")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dist.barrier(group=group)
        sym_buffer.destroy()
        if ep_buffer is not None:
            dist.barrier(group=group)
        if ep_buffer is not None:
            ep_buffer.destroy()
        dist.barrier(group=group)
        dist.destroy_process_group()
        return

    # Count local received tokens. Keep this structurally aligned with the CUDA
    # MegaMoE test; only the all-gather helper is local because this script does
    # not import the CUDA deep_gemm package.
    gathered_topk_idx = uneven_all_gather_topk_idx(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()
    num_touched_experts = torch.unique(gathered_topk_idx.flatten()).numel() - 1  # Exclude "-1"

    # Benchmark
    fused_timing = bench_tilelang_ms(lambda: run_fused(reset_stats=False), args.warmup, args.repeat)
    baseline_timing = (
        None
        if args.skip_baseline_bench
        else bench_tilelang_ms(lambda: run_baseline(return_stats=False), args.warmup, args.repeat)
    )

    fused_s = fused_timing["median_ms"] / 1e3
    fused_min_s = fused_timing["min_ms"] / 1e3
    baseline_s = float("nan") if baseline_timing is None else baseline_timing["median_ms"] / 1e3
    baseline_min_s = float("nan") if baseline_timing is None else baseline_timing["min_ms"] / 1e3
    matmul_flops = 2 * num_recv_tokens * (hidden * intermediate_hidden * 3)
    num_hbm_bytes = dcu_w8a8_effective_hbm_bytes(
        int(num_recv_tokens),
        int(num_touched_experts),
        hidden,
        intermediate_hidden,
    )
    num_xhcl_bytes = dcu_w8a8_effective_xhcl_bytes(int(num_recv_tokens), hidden)
    tflops = safe_div(matmul_flops / 1e12, fused_s)
    baseline_tflops = safe_div(matmul_flops / 1e12, baseline_s)
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, fused_s)
    xhcl_gbs = safe_div(num_xhcl_bytes / 1e9, fused_s)
    baseline_hbm_gbs = safe_div(num_hbm_bytes / 1e9, baseline_s)
    baseline_xhcl_gbs = safe_div(num_xhcl_bytes / 1e9, baseline_s)
    rank_metrics = gather_rank_metrics(
        [
            float(num_recv_tokens),
            float(num_touched_experts),
            fused_s,
            fused_min_s,
            baseline_s,
            baseline_min_s,
            float(num_hbm_bytes),
            float(num_xhcl_bytes),
            tflops,
            baseline_tflops,
            hbm_gbs,
            xhcl_gbs,
            baseline_hbm_gbs,
            baseline_xhcl_gbs,
        ],
        topk_idx.device,
        group,
    )

    if rank == 0:
        rank_avg = rank_metrics.mean(dim=0).cpu().tolist()
        (
            avg_num_recv_tokens,
            avg_num_touched_experts,
            avg_fused_s,
            avg_fused_min_s,
            avg_baseline_s,
            avg_baseline_min_s,
            avg_num_hbm_bytes,
            avg_num_xhcl_bytes,
            avg_tflops,
            avg_baseline_tflops,
            avg_hbm_gbs,
            avg_xhcl_gbs,
            avg_baseline_hbm_gbs,
            avg_baseline_xhcl_gbs,
        ) = rank_avg
        speedup = safe_div(avg_baseline_s, avg_fused_s)
        result = {
            "correct": True,
            "metric_scope": "per-card rank average",
            "num_ranks": num_ranks,
            "num_tokens_per_rank": num_tokens,
            "backend_selector_tokens": backend_selector_tokens,
            "megamoe_backend": v3_backend,
            "megamoe_backend_mode": backend_mode,
            "quant_mode": quant_mode,
            "weight_layout": weight_layout,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "num_recv_tokens_avg_per_rank": avg_num_recv_tokens,
            "num_touched_experts_avg_per_rank": avg_num_touched_experts,
            "fused_median_ms_avg_per_rank": avg_fused_s * 1e3,
            "fused_min_ms_avg_per_rank": avg_fused_min_s * 1e3,
            "baseline_median_ms_avg_per_rank": json_finite_or_none(avg_baseline_s * 1e3),
            "baseline_min_ms_avg_per_rank": json_finite_or_none(avg_baseline_min_s * 1e3),
            "baseline_kind": baseline_desc,
            "baseline_kind_resolved": baseline_kind,
            "baseline_kind_requested": args.baseline_kind,
            "baseline_execution": baseline_execution,
            "fused_execution": fused_execution,
            "route_pattern": args.route_pattern,
            "route_target_rank": args.route_target_rank,
            "fused_timing_scope": "eager_main_call",
            "baseline_timing_scope": baseline_execution,
            "cuda_graph_requested": bool(args.cuda_graph),
            "graph_execution": graph_execution,
            "bench_backend": f"tilelang_{fused_timing['backend']}",
            "baseline_bench_skipped": bool(args.skip_baseline_bench),
            "router_weight_stage": "swiglu_pre_l2_quant",
            "cuda_graph_timings": cuda_graph_timing_rows,
            "speedup_vs_deepep_deepgemm_baseline": json_finite_or_none(speedup),
            "dcu_reference_fp8_peak_tflops_per_card": DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD,
            "dcu_reference_hbm_gbps_per_card": DCU_REFERENCE_HBM_GBPS_PER_CARD,
            "dcu_reference_xhcl_gbps_per_card": DCU_REFERENCE_XHCL_GBPS_PER_CARD,
            "reference_note": "Raw effective per-card metrics averaged across ranks. HBM uses DCU W8A8 FP8 channelwise bytes: FP8 weights, FP32 input/weight/L2-act scales, FP8 activations, and BF16 output. Baseline bandwidth uses the same logical byte numerator as fused for apples-to-apples throughput comparison; split-kernel temporary traffic is not added.",
            "fused_tflops_median_avg_per_rank": avg_tflops,
            "baseline_tflops_median_avg_per_rank": json_finite_or_none(avg_baseline_tflops),
            "fused_compute_efficiency_pct": pct(avg_tflops, DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD),
            "baseline_compute_efficiency_pct": json_finite_or_none(pct(avg_baseline_tflops, DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD)),
            "dcu_w8a8_hbm_bytes_avg_per_rank": avg_num_hbm_bytes,
            "fused_dcu_w8a8_hbm_effective_gbps": avg_hbm_gbs,
            "fused_dcu_w8a8_hbm_efficiency_pct": pct(avg_hbm_gbs, DCU_REFERENCE_HBM_GBPS_PER_CARD),
            "baseline_dcu_w8a8_hbm_effective_gbps": json_finite_or_none(avg_baseline_hbm_gbs),
            "baseline_dcu_w8a8_hbm_efficiency_pct": json_finite_or_none(pct(avg_baseline_hbm_gbs, DCU_REFERENCE_HBM_GBPS_PER_CARD)),
            "dcu_w8a8_xhcl_bytes_avg_per_rank": avg_num_xhcl_bytes,
            "fused_dcu_w8a8_xhcl_effective_gbps": avg_xhcl_gbs,
            "fused_dcu_w8a8_xhcl_efficiency_pct": pct(avg_xhcl_gbs, DCU_REFERENCE_XHCL_GBPS_PER_CARD),
            "baseline_dcu_w8a8_xhcl_effective_gbps": json_finite_or_none(avg_baseline_xhcl_gbs),
            "baseline_dcu_w8a8_xhcl_efficiency_pct": json_finite_or_none(pct(avg_baseline_xhcl_gbs, DCU_REFERENCE_XHCL_GBPS_PER_CARD)),
        }
        print("Performance:")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.skip_baseline_bench:
            print(
                f" > EP: {rank:2}/{num_ranks} | "
                f"{avg_tflops:4.2f} TFLOPS, "
                f"HBM {avg_hbm_gbs:4.2f} GB/s, "
                f"xHCL {avg_xhcl_gbs:4.2f} GB/s | "
                f"{avg_fused_s * 1e6:4.0f} us | baseline skipped",
                flush=True,
            )
        else:
            print(
                f" > EP: {rank:2}/{num_ranks} | "
                f"{avg_tflops:4.2f} TFLOPS, "
                f"HBM {avg_hbm_gbs:4.2f} GB/s, "
                f"xHCL {avg_xhcl_gbs:4.2f} GB/s | "
                f"{avg_fused_s * 1e6:4.0f} us | "
                f"{result['speedup_vs_deepep_deepgemm_baseline']:.2f}x baseline "
                f"(baseline {avg_baseline_tflops:4.2f} TFLOPS, "
                f"HBM {avg_baseline_hbm_gbs:4.2f} GB/s, "
                f"xHCL {avg_baseline_xhcl_gbs:4.2f} GB/s)",
                flush=True,
            )
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dist.barrier(group=group)
    sym_buffer.destroy()
    if ep_buffer is not None:
        dist.barrier(group=group)
    if ep_buffer is not None:
        ep_buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DCU MegaMoE fused W8A8 channelwise baseline")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=32)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--num-tokens-per-rank-list", type=str, default="",
                        help="comma/space separated exact local token count per rank")
    parser.add_argument("--num-max-removed-tokens", type=int, default=0,
                        help="when --num-tokens=0, each rank uses max_tokens-random(0, this), clamped to at least 0")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate-hidden", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-topk", type=int, default=2)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument(
        "--quant-mode",
        choices=("fp8", "int8"),
        default="fp8",
        help="fused MegaMoE activation/weight quantization; INT8 is YGZP EP8 normal eager only",
    )
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument(
        "--route-pattern",
        choices=(ROUTE_PATTERN_RANDOM, ROUTE_PATTERN_SINGLE_LOCAL_RANK),
        default=ROUTE_PATTERN_RANDOM,
        help="router distribution for correctness/perf probes",
    )
    parser.add_argument(
        "--route-target-rank",
        type=int,
        default=0,
        help="target rank for --route-pattern=single-local-rank",
    )
    parser.add_argument("--prepost-backend", choices=("hip", "triton"), default="hip")
    parser.add_argument(
        "--baseline-kind",
        choices=(BASELINE_AUTO, BASELINE_NORMAL_CONTIGUOUS, BASELINE_LL_MASKED),
        default=BASELINE_AUTO,
        help=(
            "baseline flow: auto selects ll-masked for LL MegaMoE and "
            "normal-contiguous for normal MegaMoE; explicit values keep the "
            "previous comparison modes"
        ),
    )
    parser.add_argument("--correctness-iters", type=int, default=1)
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help=(
            "absolute correctness tolerance; defaults to 0.0035 for FP8 and "
            "0.01 for the INT8-vs-FP8 cross-quantization oracle"
        ),
    )
    parser.add_argument(
        "--check-zero-weight-reuse",
        action="store_true",
        help=(
            "run an eager-normal same-buffer nonzero-to-zero route regression "
            "and require exact cumulative local-expert stats"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument(
        "--skip-baseline-bench",
        action="store_true",
        help="benchmark MegaMoE only; correctness still requires the selected baseline",
    )
    parser.add_argument("--megamoe-backend", choices=("auto", "ll", "normal"),
                        default=os.getenv(BACKEND_ENV, V3_BACKEND_AUTO),
                        help=f"V3 backend selector used by this test; defaults to {BACKEND_ENV}=auto. "
                             f"auto compares the per-run selector token bucket with {NORMAL_LL_TOKEN_THRESHOLD_ENV}.")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="capture the selected V3 staged backend as a CUDA graph")
    parser.add_argument("--cuda-graph-test-tokens", type=str, default="",
                        help="comma/space separated token counts to capture/replay as graph buckets")
    parser.add_argument("--cuda-graph-warmup", type=int, default=1)
    parser.add_argument("--cuda-graph-replays", type=int, default=3)
    parser.add_argument("--cuda-graph-bench", action="store_true",
                        help="benchmark graph.replay() for each --cuda-graph-test-tokens entry")
    parser.add_argument(
        "--cuda-graph-single-capture",
        action="store_true",
        help="reuse one max-capacity MegaMoE graph for every replay bucket",
    )
    parser.add_argument("--cuda-graph-skip-baseline", action="store_true",
                        help="smoke-test graph capture/replay without the DeepEP baseline check")
    parser.add_argument("--out", type=str, default="hygon_tmp/megamoe_dcu_baseline/default_perf.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
