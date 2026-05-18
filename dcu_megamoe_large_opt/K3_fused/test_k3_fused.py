from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third-party"))
sys.path.insert(0, str(ROOT / "tests"))

import deep_ep
import deepgemm
import megamoe
import test_mega_moe_dcu as mega_test
from tilelang_ops import swiglu_apply_weight_to_fp8_dcu

from k3_fused import (
    build_asm_tail_signal_addrs,
    build_row_combine_ptrs,
    k3_l2_fused_asm_to_combine,
    k3_l2_asm_to_combine,
    rank_barrier,
    reduce_local_combine,
    reset_asm_tail_signal_slots,
)


DEEPEP_BUFFER_BYTES = int(2.0e9)
DEEPEP_CONFIG = (24, 8, 256)
DEEPEP_EXPERT_ALIGNMENT = 256


def print_once(rank: int, msg: str = "") -> None:
    if rank == 0:
        print(msg, flush=True)


def init_dist(local_rank: int, num_local_ranks: int):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "8373"))
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


def sync_group(group: dist.ProcessGroup) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=group)


def wall_bench_ms(fn, warmup: int, repeat: int, group: dist.ProcessGroup) -> float:
    for _ in range(warmup):
        fn()
    sync_group(group)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync_group(group)
    return (time.perf_counter() - start) * 1e3 / repeat


def all_gather_equal(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, tensor, group=group)
    return torch.cat(gathered, dim=0)


def _fp8_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def map_recv_rows_to_global_tokens(
    recv_x_fp8: torch.Tensor,
    recv_x_scale: torch.Tensor,
    global_x_bytes: torch.Tensor,
    global_x_scale: torch.Tensor,
) -> list[int]:
    hidden = global_x_bytes.size(1)
    probe = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4095]
    probe_cols = torch.tensor([c for c in probe if c < hidden], device=global_x_bytes.device, dtype=torch.long)
    global_probe = global_x_bytes.index_select(1, probe_cols).cpu()
    recv_bytes = _fp8_bytes(recv_x_fp8)
    recv_probe = recv_bytes.index_select(1, probe_cols).cpu()
    global_x_bytes_cpu = global_x_bytes.cpu()
    recv_bytes_cpu = recv_bytes.cpu()
    global_scale_cpu = global_x_scale.cpu()
    recv_scale_cpu = recv_x_scale.cpu()

    buckets: dict[tuple[int, ...], list[int]] = {}
    for row in range(global_probe.size(0)):
        key = tuple(int(v) for v in global_probe[row].tolist())
        buckets.setdefault(key, []).append(row)

    mapped: list[int] = []
    for row in range(recv_probe.size(0)):
        key = tuple(int(v) for v in recv_probe[row].tolist())
        candidates = buckets.get(key, [])
        match = -1
        for candidate in candidates:
            if torch.equal(global_x_bytes_cpu[candidate], recv_bytes_cpu[row]) and torch.equal(
                global_scale_cpu[candidate], recv_scale_cpu[row]
            ):
                match = candidate
                break
        if match < 0:
            raise AssertionError("failed to map DeepEP recv_x row back to a global source token")
        mapped.append(match)
    return mapped


def sort_rows_by_expert_and_combine_ptr(
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    m_indices: torch.Tensor,
    row_combine_ptrs: torch.Tensor,
    row_valid_mask: torch.Tensor,
):
    total_rows = int(act_fp8.shape[0])
    ptrs = row_combine_ptrs[:total_rows].detach().cpu().tolist()
    valid = row_valid_mask[:total_rows].detach().cpu().tolist()
    order = []
    tile_rows = 256
    for start in range(0, total_rows, tile_rows):
        end = min(start + tile_rows, total_rows)
        order.extend(
            sorted(
                range(start, end),
                key=lambda row: (
                    0 if valid[row] else 1,
                    int(ptrs[row]),
                    row,
                ),
            )
        )
    perm = torch.tensor(order, device=act_fp8.device, dtype=torch.long)
    sorted_row_ptrs = torch.cat(
        [
            row_combine_ptrs[:total_rows].index_select(0, perm),
            row_combine_ptrs[total_rows:],
        ],
        dim=0,
    ).contiguous()
    return {
        "act_fp8": act_fp8.index_select(0, perm).contiguous(),
        "act_scale": act_scale.index_select(0, perm).contiguous(),
        "m_indices": m_indices[:total_rows].index_select(0, perm).contiguous(),
        "row_combine_ptrs": sorted_row_ptrs,
        "row_valid_mask": row_valid_mask[:total_rows].index_select(0, perm).contiguous(),
        "row_permutation": perm,
    }


def run_swiglu_quant(
    l1_out: torch.Tensor,
    route_weights: torch.Tensor,
    intermediate_hidden: int,
    activation_clamp: float,
):
    act_fp8, act_scale, act_bf16 = swiglu_apply_weight_to_fp8_dcu(
        l1_out,
        route_weights,
        None,
        num_per_channels=intermediate_hidden,
        use_col_major_scales=False,
        round_scale=False,
        ue8m0_scale=False,
        output_bf16=True,
        clamp_value=activation_clamp,
    )
    if act_scale.dim() == 2:
        if act_scale.size(0) == 1:
            act_scale = act_scale[0].contiguous()
        elif act_scale.size(1) == 1:
            act_scale = act_scale[:, 0].contiguous()
        else:
            raise AssertionError(f"unexpected SwiGLU scale shape: {tuple(act_scale.shape)}")
    return act_fp8, act_scale, act_bf16


def env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def selected_k3_path(args: argparse.Namespace) -> str:
    if args.k3_mode == "asm-scatter":
        return "asm-scatter+rank-barrier+reduce"
    if env_flag("K3_USE_ASM_TAIL_REDUCE"):
        return "asm-combine+asm-tail-reduce"
    return "asm-combine+rank-barrier+reduce"


def validate_k3_path(args: argparse.Namespace) -> None:
    fused_flags = [
        "K3_USE_ASM_TAIL_REDUCE",
    ]
    enabled = [name for name in fused_flags if env_flag(name)]
    if len(enabled) > 1:
        raise ValueError(f"K3 path flags are mutually exclusive: {enabled}")
    if args.k3_mode == "asm-scatter" and enabled:
        raise ValueError("--k3-mode=asm-scatter cannot be combined with K3 fused path flags")


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    validate_k3_path(args)
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
    assert hidden == 4096 and intermediate_hidden == 2048
    assert num_experts == 256 and num_topk == 6

    print_once(rank, "DCU MegaMoE K3 fused asm+combine test:")
    print_once(rank, f" > ranks={num_ranks}, tokens/rank={num_tokens}")
    print_once(rank, f" > hidden={hidden}, intermediate={intermediate_hidden}")
    print_once(rank, f" > experts={num_experts}, local_experts={num_experts_per_rank}, topk={num_topk}")
    print_once(rank, " > scope=K3 only: L2 FP8 grouped GEMM + combine path")
    print_once(rank, f" > fused_path={selected_k3_path(args)}")

    sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    ep_buffer = deep_ep.Buffer(group, DEEPEP_BUFFER_BYTES, 0, explicitly_destroy=True)
    ep_config = deep_ep.Config(*DEEPEP_CONFIG)
    baseline_layout_cache = None

    try:
        x_bf16 = (
            torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
            * args.input_scale
        )
        l1_bf16 = (
            torch.randn(
                (num_experts_per_rank, intermediate_hidden * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * args.weight_scale
        )
        l2_bf16 = (
            torch.randn(
                (num_experts_per_rank, hidden, intermediate_hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * args.weight_scale
        )
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_weights = torch.softmax(topk_weights.float(), dim=-1)

        x_fp8, x_scale = megamoe.cast_to_fp8_channelwise(x_bf16)
        l1_weights, l2_weights = megamoe.transform_fp8_weights_for_mega_moe(l1_bf16, l2_bf16)
        global_x_bytes = all_gather_equal(_fp8_bytes(x_fp8), group)
        global_x_scale = all_gather_equal(x_scale, group)

        def copy_inputs_to_sym_buffer() -> None:
            with torch.no_grad():
                sym_buffer.x[:num_tokens].copy_(x_fp8)
                sym_buffer.x_sf[:num_tokens].copy_(x_scale)
                sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
                sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)

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

        def build_common_prefix():
            copy_inputs_to_sym_buffer()
            rank_barrier(sym_buffer, rank_idx=rank, num_ranks=num_ranks, verbose_build=bool(args.verbose_build))
            num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = get_baseline_layout_cache()
            recv = mega_test.dispatch_deepep_normal(
                ep_buffer,
                x_fp8,
                x_scale,
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                ep_config,
                DEEPEP_EXPERT_ALIGNMENT,
            )
            recv_x, recv_topk_idx, recv_topk_weights, recv_counts, recv_counts_cuda, handle = (
                mega_test.parse_dispatch_result(recv)
            )
            recv_x_fp8, recv_x_scale = mega_test.unpack_recv_x_fp8_channelwise(recv_x)
            grouped = mega_test.deepep_deepgemm_preprocess_channelwise(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_counts,
                recv_counts_cuda,
                args.prepost_backend,
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
            act_fp8, act_scale, _ = run_swiglu_quant(
                l1_out,
                grouped["route_weights"],
                intermediate_hidden,
                args.activation_clamp,
            )
            recv_to_global_token = map_recv_rows_to_global_tokens(
                recv_x_fp8,
                recv_x_scale,
                global_x_bytes,
                global_x_scale,
            )
            recv_to_global_token_t = torch.tensor(
                recv_to_global_token,
                device=x_fp8.device,
                dtype=torch.int32,
            )
            row_combine_ptrs = build_row_combine_ptrs(
                sym_buffer,
                grouped["output_index"],
                recv_to_global_token_t,
                total_rows=int(act_fp8.shape[0]),
                num_ranks=num_ranks,
                num_experts=num_experts,
                num_tokens=num_tokens,
                num_topk=num_topk,
                hidden=hidden,
                verbose_build=bool(args.verbose_build),
            )
            valid_routes = int((grouped["output_index"] >= 0).sum().item())
            row_valid_mask = torch.zeros((act_fp8.shape[0],), device=x_fp8.device, dtype=torch.bool)
            valid_grouped_rows = grouped["output_index"][grouped["output_index"] >= 0].long()
            if valid_grouped_rows.numel():
                row_valid_mask[valid_grouped_rows] = True
            fused_rows = {
                "act_fp8": act_fp8,
                "act_scale": act_scale,
                "m_indices": grouped["m_indices"],
                "row_combine_ptrs": row_combine_ptrs,
                "row_valid_mask": row_valid_mask,
                "row_permutation": None,
            }
            if args.sort_fused_rows_by_combine_ptr:
                fused_rows = sort_rows_by_expert_and_combine_ptr(
                    act_fp8,
                    act_scale,
                    grouped["m_indices"],
                    row_combine_ptrs,
                    row_valid_mask,
                )
            prefix = {
                "act_fp8": act_fp8,
                "act_scale": act_scale,
                "m_indices": grouped["m_indices"],
                "output_index": grouped["output_index"],
                "recv_topk_idx": grouped["recv_topk_idx"],
                "recv_topk_weights": grouped["recv_topk_weights"],
                "recv_rows": grouped["recv_rows"],
                "handle": handle,
                "row_combine_ptrs": row_combine_ptrs,
                "row_valid_mask": row_valid_mask,
                "valid_routes": valid_routes,
                "grouped_rows": int(act_fp8.shape[0]),
            }
            prefix.update(
                {
                    "fused_act_fp8": fused_rows["act_fp8"],
                    "fused_act_scale": fused_rows["act_scale"],
                    "fused_m_indices": fused_rows["m_indices"],
                    "fused_row_combine_ptrs": fused_rows["row_combine_ptrs"],
                    "fused_row_valid_mask": fused_rows["row_valid_mask"],
                    "fused_row_permutation": fused_rows["row_permutation"],
                }
            )
            if env_flag("K3_USE_ASM_TAIL_REDUCE"):
                reset_asm_tail_signal_slots(
                    sym_buffer,
                    rank_idx=rank,
                    num_ranks=num_ranks,
                    verbose_build=bool(args.verbose_build),
                )
                sync_group(group)
                total_wgs = ((int(fused_rows["act_fp8"].shape[0]) + 255) // 256) * ((hidden + 255) // 256)
                prefix.update(
                    {
                        "asm_tail_done_counter": torch.zeros((1,), device=x_fp8.device, dtype=torch.int32),
                        "asm_tail_signal_addrs": build_asm_tail_signal_addrs(
                            sym_buffer, rank_idx=rank, num_ranks=num_ranks
                        ),
                        "asm_tail_done_target": 0,
                        "asm_tail_signal_generation": 0,
                        "asm_tail_total_wgs": total_wgs,
                    }
                )
            return prefix

        def baseline_k3(prefix: dict):
            l2_out = torch.empty(
                (prefix["act_fp8"].shape[0], hidden),
                device=x_fp8.device,
                dtype=torch.bfloat16,
            )
            deepgemm.m_grouped_fp8_gemm_nt_contiguous(
                (prefix["act_fp8"], prefix["act_scale"]),
                l2_weights,
                l2_out,
                prefix["m_indices"],
            )
            recv_y = torch.empty((prefix["recv_rows"], hidden), device=x_fp8.device, dtype=torch.bfloat16)
            if args.prepost_backend == "hip":
                megamoe.deepep_deepgemm_postprocess_channelwise(
                    recv_y,
                    l2_out,
                    prefix["recv_topk_idx"],
                    prefix["recv_topk_weights"],
                    prefix["output_index"],
                    apply_topk_weights=False,
                )
            else:
                raise RuntimeError("K3 fused test currently supports --prepost-backend=hip only")
            combined = ep_buffer.combine(recv_y, handle=prefix["handle"], config=ep_config)
            y = combined[0]
            if hasattr(combined[-1], "current_stream_wait") and getattr(combined[-1], "event", None) is not None:
                combined[-1].current_stream_wait()
            return y, l2_out

        def fused_k3(prefix: dict):
            y = torch.empty((num_tokens, hidden), device=x_fp8.device, dtype=torch.bfloat16)
            if args.k3_mode == "asm-combine":
                asm_tail_reduce_enabled = env_flag("K3_USE_ASM_TAIL_REDUCE")
                asm_tail_kwargs = {}
                if asm_tail_reduce_enabled:
                    prefix["asm_tail_done_target"] += prefix["asm_tail_total_wgs"]
                    prefix["asm_tail_signal_generation"] += 1
                    asm_tail_kwargs = {
                        "asm_done_counter": prefix["asm_tail_done_counter"],
                        "asm_signal_addrs": prefix["asm_tail_signal_addrs"],
                        "asm_done_target": prefix["asm_tail_done_target"],
                        "asm_signal_num_ranks": num_ranks,
                    }
                    asm_tail_kwargs.update(
                        {
                            "asm_signal_generation": prefix["asm_tail_signal_generation"],
                            "asm_reduce_y": y,
                            "sym_buffer": sym_buffer,
                            "num_ranks": num_ranks,
                            "num_experts": num_experts,
                            "num_tokens": num_tokens,
                            "num_topk": num_topk,
                            "hidden": hidden,
                        }
                    )
                l2_out = k3_l2_fused_asm_to_combine(
                    prefix["fused_act_fp8"],
                    prefix["fused_act_scale"],
                    prefix["fused_m_indices"],
                    l2_weights,
                    prefix["fused_row_combine_ptrs"],
                    **asm_tail_kwargs,
                    verbose_build=bool(args.verbose_build),
                )
            else:
                asm_tail_enabled = False
                l2_out = k3_l2_asm_to_combine(
                    sym_buffer,
                    prefix["fused_act_fp8"],
                    prefix["fused_act_scale"],
                    prefix["fused_m_indices"],
                    l2_weights,
                    prefix["fused_row_combine_ptrs"],
                    verbose_build=bool(args.verbose_build),
                )
            if env_flag("K3_USE_ASM_TAIL_REDUCE") and args.k3_mode == "asm-combine":
                pass
            else:
                rank_barrier(sym_buffer, rank_idx=rank, num_ranks=num_ranks, verbose_build=bool(args.verbose_build))
                reduce_local_combine(
                    y,
                    sym_buffer,
                    num_ranks=num_ranks,
                    num_experts=num_experts,
                    num_tokens=num_tokens,
                    num_topk=num_topk,
                    hidden=hidden,
                    verbose_build=bool(args.verbose_build),
                )
            return y, l2_out

        prefix = build_common_prefix()
        print_once(
            rank,
            f" > grouped_rows={prefix['grouped_rows']}, valid_routes={prefix['valid_routes']}",
        )

        final_max_abs = 0.0
        final_l2_max_abs = 0.0
        for i in range(args.correctness_iters):
            baseline_y, baseline_l2 = baseline_k3(prefix)
            sync_group(group)
            fused_y, fused_l2 = fused_k3(prefix)
            sync_group(group)
            valid_rows = prefix["row_valid_mask"]
            if fused_l2 is not None and bool(valid_rows.any().item()):
                fused_l2_for_compare = fused_l2
                if prefix["fused_row_permutation"] is not None:
                    fused_l2_for_compare = torch.empty_like(fused_l2)
                    fused_l2_for_compare.index_copy_(0, prefix["fused_row_permutation"], fused_l2)
                l2_diff = (fused_l2_for_compare[valid_rows].float() - baseline_l2[valid_rows].float()).abs()
                l2_max_abs = l2_diff.max().item()
            else:
                l2_max_abs = 0.0
            diff = (fused_y.float() - baseline_y.float()).abs()
            max_abs = diff.max().item() if diff.numel() else 0.0
            mean_abs = diff.mean().item() if diff.numel() else 0.0
            final_max_abs = max_abs
            final_l2_max_abs = l2_max_abs
            if l2_max_abs > args.l2_atol:
                raise AssertionError(f"K3 asm L2 max_abs={l2_max_abs} exceeds --l2-atol={args.l2_atol}")
            if max_abs > args.atol:
                raise AssertionError(f"K3 fused/baseline max_abs={max_abs} exceeds --atol={args.atol}")
            print_once(
                rank,
                f"Correctness {i + 1}/{args.correctness_iters}: "
                f"y_max_abs={max_abs:.6g}, y_mean_abs={mean_abs:.6g}, l2_max_abs={l2_max_abs:.6g}",
            )

        result = {
            "correct": True,
            "num_ranks": num_ranks,
            "num_tokens_per_rank": num_tokens,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "grouped_rows": prefix["grouped_rows"],
            "valid_routes": prefix["valid_routes"],
            "y_max_abs": final_max_abs,
            "l2_max_abs": final_l2_max_abs,
            "baseline_kind": "deepep_dispatch_deepgemm_l1_swiglu_deepgemm_l2_deepep_gather_deepep_combine",
            "fused_kind": f"same_prefix_then_{selected_k3_path(args)}",
            "sort_fused_rows_by_combine_ptr": bool(args.sort_fused_rows_by_combine_ptr),
        }

        if not args.skip_bench:
            fused_ms = wall_bench_ms(lambda: fused_k3(prefix), args.warmup, args.repeat, group)
            baseline_ms = wall_bench_ms(lambda: baseline_k3(prefix), args.warmup, args.repeat, group)
            result.update(
                {
                    "fused_k3_wall_ms": fused_ms,
                    "baseline_k3_wall_ms": baseline_ms,
                    "speedup_vs_baseline_k3": baseline_ms / fused_ms if fused_ms else float("nan"),
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                }
            )
            print_once(
                rank,
                f"K3 timing: fused={fused_ms:.3f} ms, baseline={baseline_ms:.3f} ms, "
                f"speedup={result['speedup_vs_baseline_k3']:.3f}x",
            )
            if args.profile_k3_stages:
                profile_y = torch.empty((num_tokens, hidden), device=x_fp8.device, dtype=torch.bfloat16)
                profile_recv_y = torch.empty(
                    (prefix["recv_rows"], hidden), device=x_fp8.device, dtype=torch.bfloat16
                )
                profile_l2_out = torch.empty(
                    (prefix["act_fp8"].shape[0], hidden), device=x_fp8.device, dtype=torch.bfloat16
                )

                def baseline_l2_only():
                    deepgemm.m_grouped_fp8_gemm_nt_contiguous(
                        (prefix["act_fp8"], prefix["act_scale"]),
                        l2_weights,
                        profile_l2_out,
                        prefix["m_indices"],
                    )

                baseline_l2_only()
                sync_group(group)

                def baseline_postprocess_only():
                    megamoe.deepep_deepgemm_postprocess_channelwise(
                        profile_recv_y,
                        profile_l2_out,
                        prefix["recv_topk_idx"],
                        prefix["recv_topk_weights"],
                        prefix["output_index"],
                        apply_topk_weights=False,
                    )

                baseline_postprocess_only()
                sync_group(group)

                def baseline_combine_only():
                    combined = ep_buffer.combine(profile_recv_y, handle=prefix["handle"], config=ep_config)
                    if hasattr(combined[-1], "current_stream_wait") and getattr(combined[-1], "event", None) is not None:
                        combined[-1].current_stream_wait()

                def fused_l2_only():
                    asm_tail_reduce_stage = (
                        env_flag("K3_USE_ASM_TAIL_REDUCE") and args.k3_mode == "asm-combine"
                    )
                    asm_tail_kwargs = {}
                    if asm_tail_reduce_stage:
                        prefix["asm_tail_done_target"] += prefix["asm_tail_total_wgs"]
                        prefix["asm_tail_signal_generation"] += 1
                        asm_tail_kwargs = {
                            "asm_done_counter": prefix["asm_tail_done_counter"],
                            "asm_signal_addrs": prefix["asm_tail_signal_addrs"],
                            "asm_done_target": prefix["asm_tail_done_target"],
                            "asm_signal_num_ranks": num_ranks,
                        }
                        asm_tail_kwargs.update(
                            {
                                "asm_signal_generation": prefix["asm_tail_signal_generation"],
                                "asm_reduce_y": profile_y,
                                "sym_buffer": sym_buffer,
                                "num_ranks": num_ranks,
                                "num_experts": num_experts,
                                "num_tokens": num_tokens,
                                "num_topk": num_topk,
                                "hidden": hidden,
                            }
                        )
                    k3_l2_fused_asm_to_combine(
                        prefix["fused_act_fp8"],
                        prefix["fused_act_scale"],
                        prefix["fused_m_indices"],
                        l2_weights,
                        prefix["fused_row_combine_ptrs"],
                        **asm_tail_kwargs,
                        verbose_build=bool(args.verbose_build),
                    )

                fused_l2_only()
                rank_barrier(sym_buffer, rank_idx=rank, num_ranks=num_ranks, verbose_build=bool(args.verbose_build))
                sync_group(group)

                def fused_barrier_only():
                    rank_barrier(sym_buffer, rank_idx=rank, num_ranks=num_ranks, verbose_build=bool(args.verbose_build))

                def fused_reduce_only():
                    reduce_local_combine(
                        profile_y,
                        sym_buffer,
                        num_ranks=num_ranks,
                        num_experts=num_experts,
                        num_tokens=num_tokens,
                        num_topk=num_topk,
                        hidden=hidden,
                        verbose_build=bool(args.verbose_build),
                    )

                stage_times = {
                    "baseline_l2_deepgemm_ms": wall_bench_ms(baseline_l2_only, args.warmup, args.repeat, group),
                    "baseline_postprocess_ms": wall_bench_ms(
                        baseline_postprocess_only, args.warmup, args.repeat, group
                    ),
                    "baseline_deepep_combine_ms": wall_bench_ms(
                        baseline_combine_only, args.warmup, args.repeat, group
                    ),
                    "fused_l2_asm_to_combine_ms": wall_bench_ms(
                        fused_l2_only, args.warmup, args.repeat, group
                    ),
                    "fused_rank_barrier_ms": wall_bench_ms(fused_barrier_only, args.warmup, args.repeat, group),
                    "fused_reduce_local_combine_ms": wall_bench_ms(
                        fused_reduce_only, args.warmup, args.repeat, group
                    ),
                }
                stage_times["baseline_stage_sum_ms"] = (
                    stage_times["baseline_l2_deepgemm_ms"]
                    + stage_times["baseline_postprocess_ms"]
                    + stage_times["baseline_deepep_combine_ms"]
                )
                stage_times["fused_stage_sum_ms"] = (
                    stage_times["fused_l2_asm_to_combine_ms"]
                    + (
                        0.0
                        if env_flag("K3_USE_ASM_TAIL_REDUCE")
                        else stage_times["fused_rank_barrier_ms"] + stage_times["fused_reduce_local_combine_ms"]
                    )
                )
                result["k3_stage_times_ms"] = stage_times
                print_once(rank, "K3 stage timing:")
                for name, value in stage_times.items():
                    print_once(rank, f" > {name}={value:.3f} ms")
        else:
            result["bench_skipped"] = True

        if rank == 0:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        try:
            sym_buffer.destroy()
        finally:
            ep_buffer.destroy()
            dist.barrier(group=group)
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DCU MegaMoE K3 fused asm + combine test")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--local-rank-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=2048)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--input-scale", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--prepost-backend", choices=("hip",), default="hip")
    parser.add_argument("--atol", type=float, default=0.003)
    parser.add_argument("--l2-atol", type=float, default=0.003)
    parser.add_argument("--k3-mode", choices=("asm-combine", "asm-scatter"), default="asm-combine")
    parser.add_argument("--correctness-iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--profile-k3-stages", action="store_true")
    parser.add_argument("--sort-fused-rows-by-combine-ptr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--out", type=str, default="hygon_tmp/large_opt/K3_fused/k3_fused_result.json")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.local_rank_idx is not None:
        test(parsed.local_rank_idx, parsed.num_processes, parsed)
    else:
        torch.multiprocessing.spawn(test, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
