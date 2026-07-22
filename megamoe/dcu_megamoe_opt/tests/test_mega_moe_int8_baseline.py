"""YGZP EP8 eager-normal INT8 MegaMoE versus true INT8 baseline.

This executable test is intentionally separate from ``test_mega_moe_dcu.py``.
It supports only the YGZP shape and compares the fused signed-INT8 path with a
DeepEP + DeepGEMM signed-INT8 normal-contiguous baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import megamoe


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third-party"))

import deep_ep  # noqa: E402
import deepgemm  # noqa: E402
from megamoe.dcu_megamoe_opt.K2_fused.k2_fused import (  # noqa: E402
    swiglu_quant_int8_channelwise_out,
)
from megamoe.dcu_megamoe_opt.tests.test_mega_moe_dcu import (  # noqa: E402
    DEEPEP_BUFFER_BYTES,
    DEEPEP_CONFIG,
    DEEPEP_EXPERT_ALIGNMENT,
    ROUTE_PATTERN_RANDOM,
    ROUTE_PATTERN_SINGLE_LOCAL_RANK,
    apply_route_pattern,
    bench_tilelang_ms,
    gather_rank_metrics,
    init_dist,
    nonfinite_count,
    parse_dispatch_result,
    parse_int_list,
    print_once,
)


NUM_RANKS = 8
NUM_EXPERTS = 288
NUM_TOPK = 8
HIDDEN = 4096
INTERMEDIATE_HIDDEN = 2048
NUM_LOCAL_EXPERTS = NUM_EXPERTS // NUM_RANKS
INT8_BASELINE_ATOL = 0.001
INT8_WEIGHT_LAYOUT = "legacy_n16_flat"


def _dispatch_int8(
    ep_buffer,
    ep_config,
    x_int8,
    x_scale,
    topk_idx,
    topk_weights,
    layout_cache,
):
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = layout_cache
    kwargs = dict(
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        config=ep_config,
        async_finish=False,
        expert_alignment=DEEPEP_EXPERT_ALIGNMENT,
    )
    try:
        return ep_buffer.dispatch((x_int8, x_scale.view(-1, 1)), **kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "true INT8 baseline requires DeepEP expert_alignment=256 support"
        ) from exc


def _unpack_dispatched_int8(recv_x):
    if not isinstance(recv_x, tuple) or len(recv_x) != 2:
        raise RuntimeError("true INT8 baseline expects DeepEP to preserve INT8 data and scale")
    recv_x_int8, recv_x_scale = recv_x
    if recv_x_int8.dtype != torch.int8:
        raise ValueError(f"baseline dispatch data must be signed INT8, got {recv_x_int8.dtype}")
    if recv_x_scale.dtype != torch.float32:
        raise ValueError(f"baseline dispatch scale must be FP32, got {recv_x_scale.dtype}")
    if recv_x_scale.device != recv_x_int8.device:
        raise ValueError("baseline dispatch data and scale must be on the same device")
    rows = int(recv_x_int8.size(0))
    if recv_x_scale.dim() == 2 and recv_x_scale.size(1) == 1:
        recv_x_scale = recv_x_scale[:, 0]
    elif recv_x_scale.dim() != 1:
        raise ValueError(
            "baseline dispatch scale must have shape [rows] or [rows, 1], "
            f"got {tuple(recv_x_scale.shape)}"
        )
    if recv_x_scale.numel() != rows:
        raise ValueError(
            f"baseline dispatch has {rows} rows but {recv_x_scale.numel()} scales"
        )
    return recv_x_int8.contiguous(), recv_x_scale.contiguous()


def _counts_and_rows(recv_counts, recv_counts_cuda, device, recv_topk_idx):
    if recv_counts_cuda is not None:
        counts_gpu = recv_counts_cuda.to(
            device=device,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()
    elif torch.is_tensor(recv_counts):
        counts_gpu = recv_counts.to(
            device=device,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()
    else:
        counts_gpu = torch.tensor(recv_counts, dtype=torch.int32, device=device)

    if not torch.is_tensor(recv_counts):
        rows = int(sum(int(value) for value in recv_counts))
    elif recv_counts.device.type == "cpu":
        rows = int(recv_counts.sum().item())
    else:
        # Preserve the eager timed path without a device-to-host count read.
        # Padding rows retain m_indices=-1; the INT8 HIP preprocess sanitizes
        # every empty 256-row tile head before DeepGEMM reads it.
        rows = int(recv_topk_idx.numel())
    return counts_gpu, max(rows, 1)


def _run_int8_baseline(
    ep_buffer,
    ep_config,
    x_bf16,
    topk_idx,
    topk_weights,
    layout_cache,
    l1_weights,
    l2_weights,
    activation_clamp,
    fast_math,
    return_stats,
):
    rows = int(x_bf16.size(0))
    x_int8 = torch.empty((rows, HIDDEN), device=x_bf16.device, dtype=torch.int8)
    x_scale = torch.empty((rows,), device=x_bf16.device, dtype=torch.float32)
    dispatch_topk_idx = torch.empty(
        (rows, NUM_TOPK), device=x_bf16.device, dtype=torch.int64
    )
    dispatch_topk_weights = torch.empty(
        (rows, NUM_TOPK), device=x_bf16.device, dtype=torch.float32
    )
    megamoe.mega_moe_pre_dispatch_int8(
        x_bf16,
        topk_idx,
        topk_weights,
        x_int8,
        x_scale,
        dispatch_topk_idx,
        dispatch_topk_weights,
        num_tokens=rows,
    )

    recv = _dispatch_int8(
        ep_buffer,
        ep_config,
        x_int8,
        x_scale,
        dispatch_topk_idx,
        dispatch_topk_weights,
        layout_cache,
    )
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_counts,
        recv_counts_cuda,
        handle,
    ) = parse_dispatch_result(recv)
    recv_x_int8, recv_x_scale = _unpack_dispatched_int8(recv_x)
    if recv_topk_idx.dim() == 1:
        recv_topk_idx = recv_topk_idx.view(-1, 1)
        recv_topk_weights = recv_topk_weights.view(-1, 1)
    recv_topk_idx = recv_topk_idx.contiguous()
    recv_topk_weights = recv_topk_weights.contiguous()
    counts_gpu, grouped_rows = _counts_and_rows(
        recv_counts,
        recv_counts_cuda,
        recv_x_int8.device,
        recv_topk_idx,
    )
    grouped_x, grouped_x_scale, route_weights, m_indices, output_index = (
        megamoe.deepep_deepgemm_preprocess_channelwise(
            recv_x_int8,
            recv_x_scale,
            recv_topk_idx,
            recv_topk_weights,
            counts_gpu,
            grouped_rows,
        )
    )

    l1_out = torch.empty(
        (grouped_rows, INTERMEDIATE_HIDDEN * 2),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    deepgemm.m_grouped_i8_gemm_nt_contiguous(
        (grouped_x, grouped_x_scale),
        l1_weights,
        l1_out,
        m_indices,
    )
    l2_x_int8 = torch.empty(
        (grouped_rows, INTERMEDIATE_HIDDEN),
        device=x_bf16.device,
        dtype=torch.int8,
    )
    l2_x_scale = torch.empty(
        (grouped_rows,), device=x_bf16.device, dtype=torch.float32
    )
    l2_out = torch.empty(
        (grouped_rows, HIDDEN), device=x_bf16.device, dtype=torch.bfloat16
    )
    swiglu_quant_int8_channelwise_out(
        l1_out,
        route_weights,
        l2_x_int8,
        l2_x_scale,
        l2_out,
        num_per_channels=INTERMEDIATE_HIDDEN,
        output_bf16=False,
        clamp_value=activation_clamp,
        row_combine_ptrs=None,
        fast_math=fast_math,
    )
    deepgemm.m_grouped_i8_gemm_nt_contiguous(
        (l2_x_int8, l2_x_scale),
        l2_weights,
        l2_out,
        m_indices,
    )

    recv_y = torch.empty(
        (recv_x_int8.size(0), HIDDEN),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    megamoe.deepep_deepgemm_postprocess_channelwise(
        recv_y,
        l2_out,
        recv_topk_idx,
        recv_topk_weights,
        output_index,
        apply_topk_weights=False,
    )
    combined = ep_buffer.combine(recv_y, handle=handle, config=ep_config)
    y_baseline = combined[0]
    combine_event = combined[-1]
    if (
        hasattr(combine_event, "current_stream_wait")
        and getattr(combine_event, "event", None) is not None
    ):
        combine_event.current_stream_wait()

    baseline_stats = None
    if return_stats:
        valid_routes = output_index >= 0
        valid_experts = recv_topk_idx[valid_routes]
        baseline_stats = (
            torch.bincount(
                valid_experts.to(torch.int64),
                minlength=NUM_LOCAL_EXPERTS,
            ).to(torch.int32)
            if valid_experts.numel()
            else torch.zeros(
                (NUM_LOCAL_EXPERTS,),
                device=x_bf16.device,
                dtype=torch.int32,
            )
        )
    return y_baseline, baseline_stats


def _global_correctness(fused_y, baseline_y, fused_stats, baseline_stats, group):
    torch.cuda.synchronize()
    diff = (fused_y.float() - baseline_y.float()).abs()
    status = torch.tensor(
        [
            nonfinite_count(fused_y),
            nonfinite_count(baseline_y),
            nonfinite_count(diff),
            int(not torch.equal(fused_stats, baseline_stats)),
        ],
        dtype=torch.int64,
        device=fused_y.device,
    )
    dist.all_reduce(status, op=dist.ReduceOp.SUM, group=group)
    global_status = [int(value) for value in status.cpu().tolist()]
    if any(global_status[:3]):
        raise AssertionError(
            "INT8 fused/baseline nonfinite rank-counts "
            f"fused={global_status[0]} baseline={global_status[1]} "
            f"diff={global_status[2]}"
        )
    if global_status[3]:
        raise AssertionError(
            f"INT8 fused/baseline stats mismatch on {global_status[3]} rank(s)"
        )

    max_abs = (
        diff.max()
        if diff.numel()
        else torch.zeros((), dtype=torch.float32, device=diff.device)
    )
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    sum_count = torch.tensor(
        [float(diff.sum().item()), float(diff.numel())],
        dtype=torch.float64,
        device=diff.device,
    )
    dist.all_reduce(sum_count, op=dist.ReduceOp.SUM, group=group)
    abs_sum, element_count = sum_count.cpu().tolist()
    return float(max_abs.item()), float(abs_sum / element_count) if element_count else 0.0


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    if world_size != NUM_RANKS:
        raise ValueError(f"YGZP INT8 baseline test requires EP8, got world_size={world_size}")
    torch.manual_seed(args.seed + rank)

    exact_tokens = None
    if args.num_tokens_per_rank_list:
        exact_tokens = parse_int_list(args.num_tokens_per_rank_list)
        if len(exact_tokens) != NUM_RANKS:
            raise ValueError(
                f"--num-tokens-per-rank-list needs {NUM_RANKS} entries, "
                f"got {len(exact_tokens)}"
            )
        if max(exact_tokens) > args.num_max_tokens_per_rank:
            raise ValueError(
                "--num-tokens-per-rank-list exceeds "
                f"--num-max-tokens-per-rank={args.num_max_tokens_per_rank}"
            )
        num_tokens = exact_tokens[rank]
        if max(exact_tokens) < 512:
            raise ValueError("eager validation requires a maximum token size of at least 512")
    else:
        num_tokens = args.num_tokens
        if num_tokens < 512:
            raise ValueError("eager validation requires --num-tokens >= 512")
    if args.num_max_tokens_per_rank < 512:
        raise ValueError("--num-max-tokens-per-rank must be at least 512")
    if not 0 <= num_tokens <= args.num_max_tokens_per_rank:
        raise ValueError(
            f"rank {rank} token count {num_tokens} is outside buffer capacity "
            f"{args.num_max_tokens_per_rank}"
        )
    capacity_tokens = max(exact_tokens) if exact_tokens is not None else num_tokens

    sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
        group,
        NUM_EXPERTS,
        args.num_max_tokens_per_rank,
        NUM_TOPK,
        HIDDEN,
        INTERMEDIATE_HIDDEN,
        quant_mode="int8",
    )
    ep_buffer = deep_ep.Buffer(
        group,
        DEEPEP_BUFFER_BYTES,
        0,
        explicitly_destroy=True,
    )
    ep_config = deep_ep.Config(*DEEPEP_CONFIG)

    x_bf16 = (
        torch.randn((num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16)
        * args.input_scale
    )
    l1_bf16 = (
        torch.randn(
            (NUM_LOCAL_EXPERTS, INTERMEDIATE_HIDDEN * 2, HIDDEN),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * args.weight_scale
    )
    l2_bf16 = (
        torch.randn(
            (NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE_HIDDEN),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * args.weight_scale
    )
    scores = torch.randn((num_tokens, NUM_EXPERTS), device="cuda", dtype=torch.float32)
    topk_weights, topk_idx = torch.topk(
        scores, NUM_TOPK, dim=-1, largest=True, sorted=False
    )
    topk_weights = torch.softmax(topk_weights.float(), dim=-1)
    apply_route_pattern(
        topk_idx,
        topk_weights,
        pattern=args.route_pattern,
        target_rank=args.route_target_rank,
        num_ranks=NUM_RANKS,
        num_experts_per_rank=NUM_LOCAL_EXPERTS,
        num_topk=NUM_TOPK,
    )

    l1_int8, l1_scale = megamoe.cast_grouped_weight_to_int8_channelwise(l1_bf16)
    l2_int8, l2_scale = megamoe.cast_grouped_weight_to_int8_channelwise(l2_bf16)
    fused_l1_weights, fused_l2_weights = (
        megamoe.transform_int8_weights_for_mega_moe_normal(
            (l1_int8, l1_scale),
            (l2_int8, l2_scale),
        )
    )
    baseline_l1_weights = (
        megamoe.weight8bit_nt_kpack2_marlin(l1_int8),
        l1_scale,
    )
    baseline_l2_weights = (
        megamoe.weight8bit_nt_kpack2_marlin(l2_int8),
        l2_scale,
    )

    layout_result = ep_buffer.get_dispatch_layout(topk_idx, NUM_EXPERTS)
    if len(layout_result) != 5:
        raise RuntimeError(f"unexpected DeepEP layout return arity: {len(layout_result)}")
    (
        num_tokens_per_rank,
        _,
        num_tokens_per_expert,
        is_token_in_rank,
        layout_event,
    ) = layout_result
    if (
        hasattr(layout_event, "current_stream_wait")
        and getattr(layout_event, "event", None) is not None
    ):
        layout_event.current_stream_wait()
    layout_cache = (num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank)

    stats_fused = torch.zeros(
        (NUM_LOCAL_EXPERTS,), device=x_bf16.device, dtype=torch.int32
    )
    y_fused = torch.empty((num_tokens, HIDDEN), device=x_bf16.device, dtype=torch.bfloat16)

    def run_fused(reset_stats=False):
        if reset_stats:
            stats_fused.zero_()
        megamoe.mega_moe_pre_dispatch_int8(
            x_bf16,
            topk_idx,
            topk_weights,
            sym_buffer.x,
            sym_buffer.x_sf,
            sym_buffer.topk_idx,
            sym_buffer.topk_weights,
            num_tokens=num_tokens,
        )
        megamoe.int8_w8a8_mega_moe(
            y_fused,
            fused_l1_weights,
            fused_l2_weights,
            sym_buffer,
            cumulative_local_expert_recv_stats=stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math),
            megamoe_backend="normal",
            capacity_num_tokens=capacity_tokens,
        )
        return y_fused, stats_fused

    def run_baseline(return_stats=False):
        return _run_int8_baseline(
            ep_buffer,
            ep_config,
            x_bf16,
            topk_idx,
            topk_weights,
            layout_cache,
            baseline_l1_weights,
            baseline_l2_weights,
            args.activation_clamp,
            bool(args.fast_math),
            return_stats,
        )

    print_once(
        rank,
        "YGZP INT8 EP8 eager-normal: fused vs true INT8 normal-contiguous baseline",
    )
    print_once(
        rank,
        f" > tokens={exact_tokens if exact_tokens is not None else num_tokens}, "
        f"capacity={args.num_max_tokens_per_rank}, route={args.route_pattern}",
    )
    print_once(rank, f" > shape={NUM_EXPERTS}/{NUM_TOPK}/{HIDDEN}/{INTERMEDIATE_HIDDEN}")
    print_once(rank, f" > baseline weight layout={INT8_WEIGHT_LAYOUT}, atol={args.atol}")

    correctness_metrics = []
    for iteration in range(args.correctness_iters):
        fused_y, fused_stats = run_fused(reset_stats=True)
        baseline_y, baseline_stats = run_baseline(return_stats=True)
        max_abs, mean_abs = _global_correctness(
            fused_y,
            baseline_y,
            fused_stats,
            baseline_stats,
            group,
        )
        if max_abs > args.atol:
            raise AssertionError(
                f"INT8 fused/baseline max_abs={max_abs} exceeds --atol={args.atol}"
            )
        correctness_metrics.append(
            {"iteration": iteration + 1, "max_abs": max_abs, "mean_abs": mean_abs}
        )
        print_once(
            rank,
            f"Correctness {iteration + 1}/{args.correctness_iters}: "
            f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}, stats_exact=True",
        )

    fused_timing = baseline_timing = None
    if not args.skip_bench:
        fused_timing = bench_tilelang_ms(
            lambda: run_fused(reset_stats=False), args.warmup, args.repeat
        )
        baseline_timing = bench_tilelang_ms(
            lambda: run_baseline(return_stats=False), args.warmup, args.repeat
        )
        rank_timings = gather_rank_metrics(
            [
                fused_timing["median_ms"],
                fused_timing["min_ms"],
                baseline_timing["median_ms"],
                baseline_timing["min_ms"],
            ],
            x_bf16.device,
            group,
        )
        avg_timings = rank_timings.mean(dim=0).cpu().tolist()
    else:
        avg_timings = [None, None, None, None]

    if rank == 0:
        fused_median, fused_min, baseline_median, baseline_min = avg_timings
        result = {
            "correct": True,
            "execution": "ygzp_ep8_normal_int8_eager",
            "baseline_execution": "deepep_deepgemm_int8_normal_contiguous_eager",
            "quant_mode": "int8",
            "num_ranks": NUM_RANKS,
            "num_experts": NUM_EXPERTS,
            "num_topk": NUM_TOPK,
            "hidden": HIDDEN,
            "intermediate_hidden": INTERMEDIATE_HIDDEN,
            "num_max_tokens_per_rank": args.num_max_tokens_per_rank,
            "num_tokens_per_rank": (
                exact_tokens if exact_tokens is not None else [num_tokens] * NUM_RANKS
            ),
            "route_pattern": args.route_pattern,
            "route_target_rank": args.route_target_rank,
            "fused_weight_layout": "normal_plain_pack5",
            "baseline_weight_layout": INT8_WEIGHT_LAYOUT,
            "baseline_expert_alignment": DEEPEP_EXPERT_ALIGNMENT,
            "correctness_iters": args.correctness_iters,
            "atol": args.atol,
            "max_abs": max(
                (metric["max_abs"] for metric in correctness_metrics), default=0.0
            ),
            "mean_abs": (
                sum(metric["mean_abs"] for metric in correctness_metrics)
                / len(correctness_metrics)
                if correctness_metrics
                else 0.0
            ),
            "stats_exact": True,
            "correctness_metrics": correctness_metrics,
            "bench_skipped": bool(args.skip_bench),
            "bench_backend": (
                None if fused_timing is None else f"tilelang_{fused_timing['backend']}"
            ),
            "fused_median_ms_avg_per_rank": fused_median,
            "fused_min_ms_avg_per_rank": fused_min,
            "baseline_median_ms_avg_per_rank": baseline_median,
            "baseline_min_ms_avg_per_rank": baseline_min,
            "speedup_vs_int8_baseline": (
                baseline_median / fused_median
                if fused_median is not None and fused_median > 0.0
                else None
            ),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    dist.barrier(group=group)
    sym_buffer.destroy()
    dist.barrier(group=group)
    ep_buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="YGZP EP8 INT8 MegaMoE fused versus true INT8 baseline"
    )
    parser.add_argument("--num-processes", type=int, default=NUM_RANKS)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-tokens-per-rank-list", type=str, default="")
    parser.add_argument(
        "--route-pattern",
        choices=(ROUTE_PATTERN_RANDOM, ROUTE_PATTERN_SINGLE_LOCAL_RANK),
        default=ROUTE_PATTERN_RANDOM,
    )
    parser.add_argument("--route-target-rank", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1)
    parser.add_argument("--correctness-iters", type=int, default=1)
    parser.add_argument("--atol", type=float, default=INT8_BASELINE_ATOL)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="hygon_tmp/megamoe_int8_baseline/default.json",
    )
    args = parser.parse_args()
    if args.num_processes != NUM_RANKS:
        parser.error(f"YGZP INT8 baseline test requires --num-processes {NUM_RANKS}")
    if args.correctness_iters < 1:
        parser.error("--correctness-iters must be at least 1")
    if args.atol < 0:
        parser.error("--atol must be non-negative")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.local_rank_idx is not None:
        run(cli_args.local_rank_idx, cli_args.num_processes, cli_args)
    else:
        torch.multiprocessing.spawn(
            run,
            args=(cli_args.num_processes, cli_args),
            nprocs=cli_args.num_processes,
        )
