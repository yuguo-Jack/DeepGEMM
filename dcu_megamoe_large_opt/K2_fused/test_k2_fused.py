from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third-party"))

from tilelang_ops import swiglu_apply_weight_to_fp8_dcu

from k2_fused import swiglu_quant_channelwise


def wall_bench_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / repeat


def fp8_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def normalize_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 1:
        return scale.contiguous()
    if scale.dim() == 2 and scale.size(0) == 1:
        return scale[0].contiguous()
    if scale.dim() == 2 and scale.size(1) == 1:
        return scale[:, 0].contiguous()
    raise AssertionError(f"unexpected scale shape: {tuple(scale.shape)}")


def run_tilelang(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    hidden: int,
    output_bf16: bool,
    clamp_value: float | None,
    fast_math: bool,
):
    return swiglu_apply_weight_to_fp8_dcu(
        x,
        topk_weights,
        None,
        num_per_channels=hidden,
        use_col_major_scales=False,
        round_scale=False,
        ue8m0_scale=False,
        output_bf16=output_bf16,
        clamp_value=clamp_value,
        fast_math=fast_math,
    )


def run_fused(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    hidden: int,
    output_bf16: bool,
    clamp_value: float | None,
    verbose_build: bool,
):
    return swiglu_quant_channelwise(
        x,
        topk_weights,
        num_per_channels=hidden,
        use_col_major_scales=False,
        round_scale=False,
        ue8m0_scale=False,
        output_bf16=output_bf16,
        clamp_value=clamp_value,
        verbose_build=verbose_build,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DCU MegaMoE K2 fused SwiGLU channelwise FP8 quant test")
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=0.5)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--no-topk-weights", action="store_true")
    parser.add_argument("--clamp-value", type=float, default=10.0)
    parser.add_argument("--no-clamp", action="store_true")
    parser.add_argument("--output-bf16", action="store_true")
    parser.add_argument("--tilelang-fast-math", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-atol", type=float, default=1.0e-6)
    parser.add_argument("--fp8-atol", type=float, default=32.0)
    parser.add_argument("--dequant-atol", type=float, default=0.08)
    parser.add_argument("--max-byte-mismatch-ratio", type=float, default=1.0e-4)
    parser.add_argument("--bf16-atol", type=float, default=0.003)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--out", type=str, default="hygon_tmp/largesize/K2_fused/k2_fused_result.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = (torch.randn((args.rows, args.hidden * 2), device=device, dtype=torch.float32) * args.input_scale).to(torch.bfloat16)
    topk_weights = None
    if not args.no_topk_weights:
        topk_weights = (torch.rand((args.rows,), device=device, dtype=torch.float32) * args.weight_scale).contiguous()
    clamp_value = None if args.no_clamp else args.clamp_value

    tile = run_tilelang(
        x,
        topk_weights,
        args.hidden,
        args.output_bf16,
        clamp_value,
        args.tilelang_fast_math,
    )
    fused = run_fused(
        x,
        topk_weights,
        args.hidden,
        args.output_bf16,
        clamp_value,
        args.verbose_build,
    )
    torch.cuda.synchronize()

    tile_fp8, tile_scale = tile[:2]
    fused_fp8, fused_scale = fused[:2]
    tile_scale_1d = normalize_scale(tile_scale)
    fused_scale_1d = normalize_scale(fused_scale)
    scale_diff = (fused_scale_1d - tile_scale_1d).abs()
    scale_max_abs = scale_diff.max().item() if scale_diff.numel() else 0.0
    scale_mean_abs = scale_diff.mean().item() if scale_diff.numel() else 0.0

    byte_diff = fp8_bytes(fused_fp8) != fp8_bytes(tile_fp8)
    fp8_mismatch = int(byte_diff.sum().item())
    fp8_mismatch_ratio = fp8_mismatch / max(1, byte_diff.numel())
    fp8_float_diff = (fused_fp8.float() - tile_fp8.float()).abs()
    fp8_nonfinite = int((~torch.isfinite(fp8_float_diff)).sum().item())
    fp8_max_abs = fp8_float_diff.max().item() if fp8_float_diff.numel() else 0.0
    fp8_mean_abs = fp8_float_diff.mean().item() if fp8_float_diff.numel() else 0.0
    dequant_diff = (
        fused_fp8.float() * fused_scale_1d.view(-1, 1) -
        tile_fp8.float() * tile_scale_1d.view(-1, 1)
    ).abs()
    dequant_max_abs = dequant_diff.max().item() if dequant_diff.numel() else 0.0
    dequant_mean_abs = dequant_diff.mean().item() if dequant_diff.numel() else 0.0

    bf16_max_abs = 0.0
    bf16_mean_abs = 0.0
    if args.output_bf16:
        tile_bf16 = tile[2]
        fused_bf16 = fused[2]
        bf16_diff = (fused_bf16.float() - tile_bf16.float()).abs()
        bf16_max_abs = bf16_diff.max().item() if bf16_diff.numel() else 0.0
        bf16_mean_abs = bf16_diff.mean().item() if bf16_diff.numel() else 0.0

    if scale_max_abs > args.scale_atol:
        raise AssertionError(f"scale max_abs={scale_max_abs} exceeds --scale-atol={args.scale_atol}")
    if fp8_nonfinite:
        raise AssertionError(f"fp8 diff has {fp8_nonfinite} non-finite elements")
    if fp8_max_abs > args.fp8_atol:
        raise AssertionError(f"fp8 value max_abs={fp8_max_abs} exceeds --fp8-atol={args.fp8_atol}")
    if fp8_mismatch_ratio > args.max_byte_mismatch_ratio:
        raise AssertionError(
            f"fp8 byte mismatch ratio={fp8_mismatch_ratio} exceeds "
            f"--max-byte-mismatch-ratio={args.max_byte_mismatch_ratio}"
        )
    if dequant_max_abs > args.dequant_atol:
        raise AssertionError(f"dequant max_abs={dequant_max_abs} exceeds --dequant-atol={args.dequant_atol}")
    if args.output_bf16 and bf16_max_abs > args.bf16_atol:
        raise AssertionError(f"bf16 max_abs={bf16_max_abs} exceeds --bf16-atol={args.bf16_atol}")

    result = {
        "correct": True,
        "rows": args.rows,
        "hidden": args.hidden,
        "input_dtype": "bf16",
        "output_dtype": "fp8_e4m3fn",
        "scale_dtype": "fp32",
        "has_topk_weights": topk_weights is not None,
        "clamp_value": clamp_value,
        "output_bf16": bool(args.output_bf16),
        "kernel_policy": "auto",
        "scale_max_abs": scale_max_abs,
        "scale_mean_abs": scale_mean_abs,
        "fp8_max_abs": fp8_max_abs,
        "fp8_mean_abs": fp8_mean_abs,
        "fp8_nonfinite": fp8_nonfinite,
        "fp8_byte_mismatch": fp8_mismatch,
        "fp8_byte_mismatch_ratio": fp8_mismatch_ratio,
        "dequant_max_abs": dequant_max_abs,
        "dequant_mean_abs": dequant_mean_abs,
        "bf16_max_abs": bf16_max_abs,
        "bf16_mean_abs": bf16_mean_abs,
    }

    if not args.skip_bench:
        fused_ms = wall_bench_ms(
            lambda: run_fused(
                x,
                topk_weights,
                args.hidden,
                args.output_bf16,
                clamp_value,
                args.verbose_build,
            ),
            args.warmup,
            args.repeat,
        )
        tile_ms = wall_bench_ms(
            lambda: run_tilelang(
                x,
                topk_weights,
                args.hidden,
                args.output_bf16,
                clamp_value,
                args.tilelang_fast_math,
            ),
            args.warmup,
            args.repeat,
        )
        bytes_read = args.rows * args.hidden * 2 * 2
        bytes_written = args.rows * args.hidden + args.rows * 4
        if args.output_bf16:
            bytes_written += args.rows * args.hidden * 2
        traffic_gb = (bytes_read + bytes_written) / 1.0e9
        result.update(
            {
                "fused_k2_wall_ms": fused_ms,
                "tilelang_k2_wall_ms": tile_ms,
                "speedup_vs_tilelang": tile_ms / fused_ms if fused_ms else float("nan"),
                "estimated_min_traffic_gb": traffic_gb,
                "estimated_fused_bandwidth_gbps": traffic_gb / (fused_ms * 1.0e-3),
                "warmup": args.warmup,
                "repeat": args.repeat,
            }
        )
        print(
            f"K2 timing: fused={fused_ms:.3f} ms, tilelang={tile_ms:.3f} ms, "
            f"speedup={result['speedup_vs_tilelang']:.3f}x, "
            f"est_bw={result['estimated_fused_bandwidth_gbps']:.1f} GB/s",
            flush=True,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
