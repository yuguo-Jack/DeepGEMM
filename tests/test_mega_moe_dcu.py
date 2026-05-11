import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import megamoe


ROOT = Path(__file__).resolve().parents[1]
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
from tilelang_ops import swiglu_apply_weight_to_fp8_dcu
from triton_ops import (
    triton_ep_gather_channelwise,
    triton_ep_scatter_channelwise,
)


def print_once(rank: int, msg: str = ""):
    if rank == 0:
        print(msg, flush=True)


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

    l2_a_fp8, l2_a_scale, _ = swiglu_apply_weight_to_fp8_dcu(
        l1_out,
        grouped["route_weights"],
        None,
        num_per_channels=intermediate_hidden,
        use_col_major_scales=False,
        round_scale=False,
        ue8m0_scale=False,
        output_bf16=True,
        clamp_value=activation_clamp,
    )
    if l2_a_scale.dim() == 2:
        l2_a_scale = l2_a_scale[:, 0].contiguous()

    l2_out = torch.empty(
        (grouped["a"][0].shape[0], hidden),
        device=x_fp8.device,
        dtype=torch.bfloat16,
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

    num_tokens = args.num_tokens
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_experts % num_ranks == 0
    assert num_tokens <= num_max_tokens_per_rank
    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0

    print_once(rank, "DCU MegaMoE channelwise W8A8 baseline:")
    print_once(rank, f" > megamoe: {getattr(megamoe, '__file__', None)}")
    print_once(rank, f" > deep_ep: {getattr(deep_ep, '__file__', None)}")
    print_once(rank, f" > deepgemm: {getattr(deepgemm, '__file__', None)}")
    print_once(rank, f" > build config: {megamoe.get_mega_moe_hip_build_config()}")

    sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
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

    x_fp8, x_scale = megamoe.cast_to_fp8_channelwise(x_bf16)
    l1_weights, l2_weights = megamoe.transform_fp8_weights_for_mega_moe(l1_bf16, l2_bf16)
    stats_initial = torch.randint(0, 100, (num_experts_per_rank,), dtype=torch.int32, device="cuda")
    stats_fused = stats_initial.clone()
    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    baseline_layout_cache = None

    def copy_inputs_to_sym_buffer():
        sym_buffer.x[:num_tokens].copy_(x_fp8)
        sym_buffer.x_sf[:num_tokens].copy_(x_scale)
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def run_fused(reset_stats: bool = False):
        if reset_stats:
            stats_fused.copy_(stats_initial)
        copy_inputs_to_sym_buffer()
        megamoe.fp8_w8a8_mega_moe(
            y_fused,
            l1_weights,
            l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats=stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
        )
        return y_fused, stats_fused

    def get_baseline_layout_cache():
        nonlocal baseline_layout_cache
        if baseline_layout_cache is None:
            layout_result = ep_buffer.get_dispatch_layout(topk_idx, num_experts)
            if len(layout_result) != 5:
                raise RuntimeError(f"unexpected DeepEP layout return arity: {len(layout_result)}")
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, event = layout_result
            if hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
                event.current_stream_wait()
            baseline_layout_cache = (num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)
        return baseline_layout_cache

    def run_baseline(return_stats: bool):
        layout_cache = get_baseline_layout_cache()
        baseline_y, baseline_counts, meta = run_deepgemm_megamoe_baseline(
            ep_buffer,
            ep_config,
            x_fp8,
            x_scale,
            topk_idx,
            topk_weights,
            l1_weights,
            l2_weights,
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
        baseline_stats = stats_initial + baseline_counts if baseline_counts is not None else None
        return baseline_y, baseline_stats, meta

    print_once(rank, "Config:")
    print_once(rank, f" > ranks={num_ranks}, tokens={num_tokens}/{sym_buffer.num_max_tokens_per_rank}")
    print_once(rank, f" > hidden={hidden}, intermediate={intermediate_hidden}")
    print_once(rank, f" > experts={num_experts}, topk={num_topk}, local_experts={num_experts_per_rank}")
    print_once(rank, f" > scale modes=input/weight/l2_act channelwise fp32")
    print_once(rank, f" > baseline preprocess={args.prepost_backend} SGLang DeepEP normal ep_scatter/ep_gather style")
    print_once(rank, " > router weights=CUDA MegaMoE compatible SwiGLU-pre-L2-quant")
    print_once(rank, f" > sym_buffer={sym_buffer.buffer.nbytes / 2 ** 30:.3f} GiB")

    for i in range(args.correctness_iters):
        fused_y, fused_stats = run_fused(reset_stats=True)
        baseline_y, baseline_stats, _ = run_baseline(return_stats=True)
        diff = (fused_y.float() - baseline_y.float()).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        stats_ok = torch.equal(fused_stats, baseline_stats)
        if max_abs > args.atol:
            raise AssertionError(f"fused/baseline max_abs={max_abs} exceeds --atol={args.atol}")
        if not stats_ok:
            raise AssertionError(f"stats mismatch: fused={fused_stats}, baseline={baseline_stats}")
        print_once(
            rank,
            f"Correctness {i + 1}/{args.correctness_iters}: "
            f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}",
        )

    if args.skip_bench:
        if rank == 0:
            result = {
                "correct": True,
                "bench_skipped": True,
                "reason": "--skip-bench",
                "num_ranks": num_ranks,
                "num_tokens_per_rank": num_tokens,
                "hidden": hidden,
                "intermediate_hidden": intermediate_hidden,
                "num_experts": num_experts,
                "num_topk": num_topk,
                "prepost_backend": args.prepost_backend,
                "correctness_iters": args.correctness_iters,
                "atol": args.atol,
            }
            print("Correctness-only:")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sym_buffer.destroy()
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
    baseline_timing = bench_tilelang_ms(lambda: run_baseline(return_stats=False), args.warmup, args.repeat)

    fused_s = fused_timing["median_ms"] / 1e3
    fused_min_s = fused_timing["min_ms"] / 1e3
    baseline_s = baseline_timing["median_ms"] / 1e3
    baseline_min_s = baseline_timing["min_ms"] / 1e3
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
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "num_recv_tokens_avg_per_rank": avg_num_recv_tokens,
            "num_touched_experts_avg_per_rank": avg_num_touched_experts,
            "fused_median_ms_avg_per_rank": avg_fused_s * 1e3,
            "fused_min_ms_avg_per_rank": avg_fused_min_s * 1e3,
            "baseline_median_ms_avg_per_rank": avg_baseline_s * 1e3,
            "baseline_min_ms_avg_per_rank": avg_baseline_min_s * 1e3,
            "baseline_kind": f"deepep_dispatch_{args.prepost_backend}_scatter_deepgemm_tilelang_swiglu_{args.prepost_backend}_gather_deepep_combine_channelwise",
            "bench_backend": f"tilelang_{fused_timing['backend']}",
            "router_weight_stage": "swiglu_pre_l2_quant",
            "speedup_vs_deepep_deepgemm_baseline": speedup,
            "dcu_reference_fp8_peak_tflops_per_card": DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD,
            "dcu_reference_hbm_gbps_per_card": DCU_REFERENCE_HBM_GBPS_PER_CARD,
            "dcu_reference_xhcl_gbps_per_card": DCU_REFERENCE_XHCL_GBPS_PER_CARD,
            "reference_note": "Raw effective per-card metrics averaged across ranks. HBM uses DCU W8A8 FP8 channelwise bytes: FP8 weights, FP32 input/weight/L2-act scales, FP8 activations, and BF16 output. Baseline bandwidth uses the same logical byte numerator as fused for apples-to-apples throughput comparison; split-kernel temporary traffic is not added.",
            "fused_tflops_median_avg_per_rank": avg_tflops,
            "baseline_tflops_median_avg_per_rank": avg_baseline_tflops,
            "fused_compute_efficiency_pct": pct(avg_tflops, DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD),
            "baseline_compute_efficiency_pct": pct(avg_baseline_tflops, DCU_REFERENCE_FP8_PEAK_TFLOPS_PER_CARD),
            "dcu_w8a8_hbm_bytes_avg_per_rank": avg_num_hbm_bytes,
            "fused_dcu_w8a8_hbm_effective_gbps": avg_hbm_gbs,
            "fused_dcu_w8a8_hbm_efficiency_pct": pct(avg_hbm_gbs, DCU_REFERENCE_HBM_GBPS_PER_CARD),
            "baseline_dcu_w8a8_hbm_effective_gbps": avg_baseline_hbm_gbs,
            "baseline_dcu_w8a8_hbm_efficiency_pct": pct(avg_baseline_hbm_gbs, DCU_REFERENCE_HBM_GBPS_PER_CARD),
            "dcu_w8a8_xhcl_bytes_avg_per_rank": avg_num_xhcl_bytes,
            "fused_dcu_w8a8_xhcl_effective_gbps": avg_xhcl_gbs,
            "fused_dcu_w8a8_xhcl_efficiency_pct": pct(avg_xhcl_gbs, DCU_REFERENCE_XHCL_GBPS_PER_CARD),
            "baseline_dcu_w8a8_xhcl_effective_gbps": avg_baseline_xhcl_gbs,
            "baseline_dcu_w8a8_xhcl_efficiency_pct": pct(avg_baseline_xhcl_gbs, DCU_REFERENCE_XHCL_GBPS_PER_CARD),
        }
        print("Performance:")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
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

    sym_buffer.destroy()
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
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--intermediate-hidden", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--num-topk", type=int, default=2)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--prepost-backend", choices=("hip", "triton"), default="hip")
    parser.add_argument("--correctness-iters", type=int, default=1)
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--out", type=str, default="hygon_tmp/megamoe_dcu_baseline/default_perf.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.local_rank_idx is not None:
        test(args.local_rank_idx, args.num_processes, args)
    else:
        torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
