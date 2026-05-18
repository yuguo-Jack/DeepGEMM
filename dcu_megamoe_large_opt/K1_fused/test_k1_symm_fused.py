from __future__ import annotations

import argparse
import json
import math
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

from k1_fused import k1_symm_fused_l1_asm, load_extension


DEEPEP_BUFFER_BYTES = int(2.0e9)
DEEPEP_CONFIG = (24, 8, 256)
DEEPEP_EXPERT_ALIGNMENT = 256
PREPOST_BACKEND = "hip"
ACTIVATION_CLAMP = 10.0
INPUT_SCALE = 0.05
WEIGHT_SCALE = 0.05
L1_ATOL = 0.003
SWIGLU_ATOL = 0.003
ROUTE_WEIGHT_ATOL = 1e-6
K1_NUM_RANKS = 8
K1_NUM_EXPERTS = 256
K1_NUM_TOPK = 6
K1_HIDDEN = 4096
K1_INTERMEDIATE = 2048
K1_AUTO_COMPACT_MIN_SAVING = 0.10


def k1_mode_name() -> str:
    mode = os.getenv("K1_PREBUILD_MODE", "auto")
    if mode == "compact":
        return "compact-prebuild-asm-pull-l1"
    if mode == "auto":
        return "auto-asm-or-compact-pull-l1"
    return "asm-route-pull-l1"


def k1_capacity_tiles(
    num_ranks: int,
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> int:
    total_tasks = num_ranks * num_tokens * num_topk
    expected_per_expert = (total_tasks + num_experts - 1) // num_experts
    rows_per_expert_target = max(
        DEEPEP_EXPERT_ALIGNMENT, expected_per_expert + 64
    )
    tiles_per_expert = (
        rows_per_expert_target + DEEPEP_EXPERT_ALIGNMENT - 1
    ) // DEEPEP_EXPERT_ALIGNMENT
    return num_experts_per_rank * tiles_per_expert


def k1_prebuild_strategy(
    *,
    num_ranks: int,
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> str:
    mode = os.getenv("K1_PREBUILD_MODE", "auto")
    if mode in ("asm", "asm_route"):
        return "asm"
    if mode == "compact":
        return "compact"
    if mode != "auto":
        return f"invalid:{mode}"
    if k1_auto_uses_compact(
        num_ranks=num_ranks,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_topk=num_topk,
        num_experts_per_rank=num_experts_per_rank,
    ):
        return "compact"
    return "asm"


def k1_auto_uses_compact(
    *,
    num_ranks: int,
    num_tokens: int,
    num_experts: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> bool:
    capacity_tiles = k1_capacity_tiles(
        num_ranks, num_tokens, num_experts, num_topk, num_experts_per_rank
    )
    asm_tiles_per_expert = capacity_tiles // num_experts_per_rank
    if asm_tiles_per_expert <= 1:
        return False
    total_tasks = num_ranks * num_tokens * num_topk
    p = 1.0 / float(num_experts)
    mean = float(total_tasks) * p
    sigma = math.sqrt(max(1.0, mean * (1.0 - p)))
    compact_tiles = 0.0
    for tile in range(asm_tiles_per_expert):
        threshold = float(tile * DEEPEP_EXPERT_ALIGNMENT)
        if tile == 0:
            prob = 1.0 - math.exp(math.log1p(-p) * float(total_tasks))
        else:
            z = (threshold + 0.5 - mean) / sigma
            prob = 0.5 * math.erfc(z / math.sqrt(2.0))
        compact_tiles += min(1.0, max(0.0, prob))
    compact_tiles = min(float(asm_tiles_per_expert), compact_tiles)
    saving = (float(asm_tiles_per_expert) - compact_tiles) / float(
        asm_tiles_per_expert
    )
    return saving >= K1_AUTO_COMPACT_MIN_SAVING


def k1_shape_contract() -> str:
    return (
        "K1_fused currently supports only ranks=8, experts=256, "
        "local_experts=32, topk=6, hidden=4096, intermediate=2048, "
        "DeepEP alignment=256, and 0<num_tokens_per_rank<=num_max_tokens_per_rank"
    )


def print_once(rank: int, msg: str = "") -> None:
    if rank == 0:
        print(msg, flush=True)


def init_dist(local_rank: int, num_local_ranks: int):
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "8372"))
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


_DEVICE_BARRIER_TENSORS: dict[tuple[int, int], torch.Tensor] = {}


def device_rank_barrier(group: dist.ProcessGroup) -> None:
    key = (id(group), torch.cuda.current_device())
    token = _DEVICE_BARRIER_TENSORS.get(key)
    if token is None:
        token = torch.empty((1,), dtype=torch.int32, device="cuda")
        _DEVICE_BARRIER_TENSORS[key] = token
    token.zero_()
    dist.all_reduce(token, op=dist.ReduceOp.SUM, group=group)


def import_ready_barrier(group: dist.ProcessGroup) -> None:
    device_rank_barrier(group)


def wall_bench_ms(fn, warmup: int, repeat: int, group: dist.ProcessGroup) -> float:
    keepalive = []
    for _ in range(warmup):
        keepalive.append(fn())
    torch.cuda.synchronize()
    dist.barrier(group=group)
    keepalive.clear()
    start = time.perf_counter()
    for _ in range(repeat):
        keepalive.append(fn())
    torch.cuda.synchronize()
    dist.barrier(group=group)
    keepalive.clear()
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
    probe_cols = torch.tensor(
        [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4095],
        device=global_x_bytes.device,
        dtype=torch.long,
    )
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


def align_route_rows(
    *,
    baseline: dict,
    fused: dict,
    recv_to_global_token: list[int],
    global_topk_idx: torch.Tensor,
    rank: int,
    num_experts_per_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_output_index = baseline["output_index"].detach().cpu()
    base_topk_idx = baseline["recv_topk_idx"].detach().cpu()
    fused_output_index = fused["output_index"].detach().cpu()
    global_topk_idx_cpu = global_topk_idx.detach().cpu()

    def add_unique(mapping: dict[tuple[int, int], int], key: tuple[int, int], row: int, label: str) -> None:
        if key in mapping:
            raise AssertionError(f"duplicate {label} route key {key}")
        mapping[key] = row

    base_map: dict[tuple[int, int], int] = {}
    for recv_row, slot in (base_output_index >= 0).nonzero(as_tuple=False).tolist():
        token = recv_to_global_token[recv_row]
        expert = int(base_topk_idx[recv_row, slot].item())
        if expert >= num_experts_per_rank:
            expert -= rank * num_experts_per_rank
        add_unique(base_map, (token, expert), int(base_output_index[recv_row, slot].item()), "baseline")

    fused_map: dict[tuple[int, int], int] = {}
    for token, slot in (fused_output_index >= 0).nonzero(as_tuple=False).tolist():
        expert = int(global_topk_idx_cpu[token, slot].item()) - rank * num_experts_per_rank
        add_unique(fused_map, (token, expert), int(fused_output_index[token, slot].item()), "fused")

    if set(base_map) != set(fused_map):
        missing = sorted(set(base_map) - set(fused_map))[:8]
        extra = sorted(set(fused_map) - set(base_map))[:8]
        raise AssertionError(f"route key mismatch before swiglu: missing={missing}, extra={extra}")

    keys = sorted(base_map)
    device = fused["l1_out"].device
    base_rows = torch.tensor([base_map[key] for key in keys], device=device, dtype=torch.long)
    fused_rows = torch.tensor([fused_map[key] for key in keys], device=device, dtype=torch.long)
    return base_rows, fused_rows


def run_swiglu_quant(l1_out, route_weights, intermediate_hidden: int, activation_clamp: float):
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
    return act_fp8, act_scale, act_bf16


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    num_tokens = args.num_tokens
    num_max_tokens_per_rank = args.num_max_tokens_per_rank or num_tokens
    hidden = K1_HIDDEN
    intermediate_hidden = K1_INTERMEDIATE
    num_experts = K1_NUM_EXPERTS
    num_topk = K1_NUM_TOPK
    if (
        num_ranks != K1_NUM_RANKS
        or num_tokens <= 0
        or num_tokens > num_max_tokens_per_rank
    ):
        raise ValueError(k1_shape_contract())
    num_experts_per_rank = num_experts // num_ranks

    print_once(rank, "DCU MegaMoE K1 fused asm + swiglu-quant test:")
    print_once(rank, f" > ranks={num_ranks}, tokens/rank={num_tokens}")
    print_once(rank, f" > hidden={hidden}, intermediate={intermediate_hidden}")
    print_once(rank, f" > experts={num_experts}, local_experts={num_experts_per_rank}, topk={num_topk}")

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
            * INPUT_SCALE
        )
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        topk_weights = torch.softmax(topk_weights.float(), dim=-1)

        x_fp8, x_scale = megamoe.cast_to_fp8_channelwise(x_bf16)
        l1_bf16 = (
            torch.randn(
                (num_experts_per_rank, intermediate_hidden * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * WEIGHT_SCALE
        )
        dummy_l2 = torch.empty(
            (num_experts_per_rank, hidden, intermediate_hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        l1_weights, _ = megamoe.transform_fp8_weights_for_mega_moe(l1_bf16, dummy_l2)
        global_x_bytes = all_gather_equal(_fp8_bytes(x_fp8), group)
        global_x_scale = all_gather_equal(x_scale, group)
        global_topk_idx = all_gather_equal(topk_idx, group)

        def copy_inputs_to_sym_buffer():
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

        def run_baseline_k1():
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
            recv_x, recv_topk_idx, recv_topk_weights, recv_counts, recv_counts_cuda, _ = (
                mega_test.parse_dispatch_result(recv)
            )
            recv_x_fp8, recv_x_scale = mega_test.unpack_recv_x_fp8_channelwise(recv_x)
            grouped = mega_test.deepep_deepgemm_preprocess_channelwise(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_counts,
                recv_counts_cuda,
                PREPOST_BACKEND,
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
            return {
                "l1_out": l1_out,
                "a": grouped["a"],
                "route_weights": grouped["route_weights"],
                "m_indices": grouped["m_indices"],
                "output_index": grouped["output_index"],
                "recv_x_fp8": recv_x_fp8,
                "recv_x_scale": recv_x_scale,
                "recv_topk_idx": grouped["recv_topk_idx"],
                "rows": int(l1_out.shape[0]),
            }

        def add_swiglu_quant(result: dict) -> dict:
            act_fp8, act_scale, act_bf16 = run_swiglu_quant(
                result["l1_out"],
                result["route_weights"],
                intermediate_hidden,
                ACTIVATION_CLAMP,
            )
            result = dict(result)
            result.update(
                {
                    "act_fp8": act_fp8,
                    "act_scale": act_scale,
                    "act_bf16": act_bf16,
                }
            )
            return result

        def run_baseline_k1_swiglu():
            return add_swiglu_quant(run_baseline_k1())

        def run_fused_k1(*, import_inputs: bool = True):
            if import_inputs:
                copy_inputs_to_sym_buffer()
                import_ready_barrier(group)
            k1_result = k1_symm_fused_l1_asm(
                sym_buffer,
                l1_weights,
                rank_idx=rank,
                num_ranks=num_ranks,
                num_experts=num_experts,
                num_tokens=num_tokens,
                num_topk=num_topk,
                hidden=hidden,
                alignment=DEEPEP_EXPERT_ALIGNMENT,
                verbose_build=False,
            )
            (
                l1_out,
                route_weights,
                m_indices,
                output_index,
            ) = k1_result
            return {
                "l1_out": l1_out,
                "route_weights": route_weights,
                "m_indices": m_indices,
                "output_index": output_index,
                "rows": int(l1_out.shape[0]),
            }

        def run_fused_k1_swiglu():
            return add_swiglu_quant(run_fused_k1())

        if args.bench_scope == "k1":
            run_fused_validate = run_fused_k1
            run_fused_bench = run_fused_k1
            run_baseline_bench = run_baseline_k1
            benchmark_scope = "k1_l1_only"
            fused_time_key = "fused_k1_l1_wall_ms"
            baseline_time_key = "deepep_deepgemm_k1_l1_baseline_wall_ms"
            speedup_key = "speedup_vs_deepep_deepgemm_k1_l1"
            fused_kind = "symm_buffer_asm_route_or_compact_metadata_asm_pull_l1"
            baseline_kind = f"deepep_dispatch_{PREPOST_BACKEND}_scatter_deepgemm_l1"
        else:
            run_fused_validate = run_fused_k1_swiglu
            run_fused_bench = run_fused_k1_swiglu
            run_baseline_bench = run_baseline_k1_swiglu
            benchmark_scope = "k1_plus_swiglu_quant_only"
            fused_time_key = "fused_k1_plus_swiglu_quant_wall_ms"
            baseline_time_key = "deepep_deepgemm_k1_plus_swiglu_quant_baseline_wall_ms"
            speedup_key = "speedup_vs_deepep_deepgemm_k1_plus_swiglu_quant"
            fused_kind = "symm_buffer_asm_route_or_compact_metadata_asm_pull_l1_swiglu_quant"
            baseline_kind = f"deepep_dispatch_{PREPOST_BACKEND}_scatter_deepgemm_l1_swiglu_quant"

        load_extension(verbose=False)
        fused = run_fused_validate()
        if args.fused_only:
            flat_output_index = fused["output_index"].reshape(-1)
            fused_rows = flat_output_index[flat_output_index >= 0].long()
            route_weight_values = fused["route_weights"][fused_rows] if fused_rows.numel() else fused["route_weights"][:0]
            m_index_values = fused["m_indices"][fused_rows] if fused_rows.numel() else fused["m_indices"][:0]
            l1_values = fused["l1_out"][fused_rows] if fused_rows.numel() else fused["l1_out"][:0]
            l1_nonfinite = int((~torch.isfinite(l1_values.float())).sum().item()) if l1_values.numel() else 0
            if "act_bf16" in fused:
                swiglu_values = fused["act_bf16"][fused_rows] if fused_rows.numel() else fused["act_bf16"][:0]
                swiglu_nonfinite = (
                    int((~torch.isfinite(swiglu_values.float())).sum().item()) if swiglu_values.numel() else 0
                )
            else:
                swiglu_nonfinite = 0
            route_weight_nonfinite = (
                int((~torch.isfinite(route_weight_values)).sum().item()) if route_weight_values.numel() else 0
            )
            invalid_m_indices = (
                int(((m_index_values < 0) | (m_index_values >= num_experts_per_rank)).sum().item())
                if m_index_values.numel()
                else 0
            )
            if l1_nonfinite or swiglu_nonfinite or route_weight_nonfinite or invalid_m_indices:
                raise AssertionError(
                    "fused-only validation failed: "
                    f"l1_nonfinite={l1_nonfinite}, "
                    f"swiglu_nonfinite={swiglu_nonfinite}, "
                    f"route_weight_nonfinite={route_weight_nonfinite}, "
                    f"invalid_m_indices={invalid_m_indices}"
                )
            print_once(
                rank,
                (
                    "Fused-only validation: "
                    f"rows={fused['rows']}, matched_routes={int(fused_rows.numel())}"
                ),
            )
            result = {
                "correct": True,
                "validation_scope": "fused_only_no_baseline",
                "num_ranks": num_ranks,
                "num_tokens_per_rank": num_tokens,
                "hidden": hidden,
                "intermediate_hidden": intermediate_hidden,
                "num_experts": num_experts,
                "num_topk": num_topk,
                "benchmark_scope": benchmark_scope,
                "fused_rows": fused["rows"],
                "matched_routes": int(fused_rows.numel()),
                "l1_nonfinite": l1_nonfinite,
                "swiglu_bf16_nonfinite": swiglu_nonfinite,
                "route_weight_nonfinite": route_weight_nonfinite,
                "invalid_m_indices": invalid_m_indices,
                "baseline_kind": "skipped",
                "k1_mode": k1_mode_name(),
                "prebuild_strategy": k1_prebuild_strategy(
                    num_ranks=num_ranks,
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    num_topk=num_topk,
                    num_experts_per_rank=num_experts_per_rank,
                ),
                "fused_kind": fused_kind,
            }
            if not args.skip_bench:
                fused_ms = wall_bench_ms(run_fused_bench, args.warmup, args.repeat, group)
                result[fused_time_key] = fused_ms
            if rank == 0:
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                if args.out:
                    out_path = Path(args.out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
            return

        baseline = run_baseline_bench()
        torch.cuda.synchronize()

        recv_to_global_token = map_recv_rows_to_global_tokens(
            baseline["recv_x_fp8"],
            baseline["recv_x_scale"],
            global_x_bytes,
            global_x_scale,
        )
        base_rows, fused_rows = align_route_rows(
            baseline=baseline,
            fused=fused,
            recv_to_global_token=recv_to_global_token,
            global_topk_idx=global_topk_idx,
            rank=rank,
            num_experts_per_rank=num_experts_per_rank,
        )

        l1_diff = (fused["l1_out"][fused_rows].float() - baseline["l1_out"][base_rows].float()).abs()
        if "act_bf16" in fused and "act_bf16" in baseline:
            swiglu_diff = (
                fused["act_bf16"][fused_rows].float() - baseline["act_bf16"][base_rows].float()
            ).abs()
        else:
            swiglu_diff = torch.empty(0, device=fused["l1_out"].device)
        route_weight_diff = (
            fused["route_weights"][fused_rows].float() - baseline["route_weights"][base_rows].float()
        ).abs()
        m_match = torch.equal(fused["m_indices"][fused_rows], baseline["m_indices"][base_rows])

        l1_nonfinite = int((~torch.isfinite(l1_diff)).sum().item()) if l1_diff.numel() else 0
        swiglu_nonfinite = int((~torch.isfinite(swiglu_diff)).sum().item()) if swiglu_diff.numel() else 0
        route_weight_nonfinite = (
            int((~torch.isfinite(route_weight_diff)).sum().item()) if route_weight_diff.numel() else 0
        )
        l1_max_abs = l1_diff.max().item() if l1_diff.numel() else 0.0
        l1_mean_abs = l1_diff.mean().item() if l1_diff.numel() else 0.0
        swiglu_max_abs = swiglu_diff.max().item() if swiglu_diff.numel() else 0.0
        swiglu_mean_abs = swiglu_diff.mean().item() if swiglu_diff.numel() else 0.0
        route_weight_max_abs = route_weight_diff.max().item() if route_weight_diff.numel() else 0.0

        if l1_nonfinite or swiglu_nonfinite or route_weight_nonfinite:
            raise AssertionError(
                "non-finite diff before swiglu: "
                f"l1={l1_nonfinite}, swiglu={swiglu_nonfinite}, "
                f"route_weight={route_weight_nonfinite}"
            )
        if route_weight_max_abs > ROUTE_WEIGHT_ATOL:
            raise AssertionError(
                f"route weight max_abs={route_weight_max_abs} exceeds tolerance {ROUTE_WEIGHT_ATOL}"
            )
        if not m_match:
            raise AssertionError("m_indices mismatch before swiglu")
        if l1_max_abs > L1_ATOL:
            raise AssertionError(f"K1 L1 max_abs={l1_max_abs} exceeds tolerance {L1_ATOL}")
        if swiglu_max_abs > SWIGLU_ATOL:
            raise AssertionError(
                f"swiglu bf16 max_abs={swiglu_max_abs} exceeds tolerance {SWIGLU_ATOL}"
            )

        print_once(
            rank,
            (
                f"Correctness: l1_max_abs={l1_max_abs:.6g}, "
                f"swiglu_max_abs={swiglu_max_abs:.6g}"
            ),
        )

        result = {
            "correct": True,
            "num_ranks": num_ranks,
            "num_tokens_per_rank": num_tokens,
            "hidden": hidden,
            "intermediate_hidden": intermediate_hidden,
            "num_experts": num_experts,
            "num_topk": num_topk,
            "benchmark_scope": benchmark_scope,
            "fused_rows": fused["rows"],
            "baseline_rows": baseline["rows"],
            "matched_routes": int(base_rows.numel()),
            "l1_max_abs": l1_max_abs,
            "l1_mean_abs": l1_mean_abs,
            "l1_nonfinite": l1_nonfinite,
            "swiglu_bf16_max_abs": swiglu_max_abs,
            "swiglu_bf16_mean_abs": swiglu_mean_abs,
            "swiglu_bf16_nonfinite": swiglu_nonfinite,
            "route_weight_max_abs": route_weight_max_abs,
            "route_weight_nonfinite": route_weight_nonfinite,
            "baseline_kind": baseline_kind,
            "k1_mode": k1_mode_name(),
            "prebuild_strategy": k1_prebuild_strategy(
                num_ranks=num_ranks,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_topk=num_topk,
                num_experts_per_rank=num_experts_per_rank,
            ),
            "fused_kind": fused_kind,
        }
        if not args.skip_bench:
            fused_ms = wall_bench_ms(run_fused_bench, args.warmup, args.repeat, group)
            baseline_ms = wall_bench_ms(run_baseline_bench, args.warmup, args.repeat, group)
            result.update(
                {
                    fused_time_key: fused_ms,
                    baseline_time_key: baseline_ms,
                    speedup_key: (
                        baseline_ms / fused_ms if fused_ms else float("nan")
                    ),
                }
            )
        if rank == 0:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
    finally:
        sym_buffer.destroy()
        ep_buffer.destroy()
        dist.barrier(group=group)
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DCU MegaMoE K1 fused asm + swiglu quant test")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--bench-scope", choices=("k1", "k1_swiglu"), default="k1_swiglu")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--fused-only", action="store_true")
    parser.add_argument("--out", type=str, default="hygon_tmp/large_opt/K1_fused/k1_symm_fused_swiglu_result.json")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    torch.multiprocessing.spawn(test, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
