"""Isolated Python stage helpers for DCU MegaMoE V2 prototypes."""

from __future__ import annotations

from pathlib import Path

import torch

_K2_EXT = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_k2_ext(verbose: bool = False):
    global _K2_EXT
    if _K2_EXT is not None:
        return _K2_EXT
    try:
        from megamoe.dcu_megamoe_large_opt.K2_fused import k2_fused_ext

        _K2_EXT = k2_fused_ext
        return _K2_EXT
    except Exception:
        pass

    from torch.utils.cpp_extension import load

    root = _repo_root()
    source = root / "megamoe" / "dcu_megamoe_large_opt" / "K2_fused" / "k2_fused_ext.cu"
    build_dir = root / "hygon_tmp" / "dcu_megamoe_v2" / "torch_extensions" / "k2_fused_ext"
    build_dir.mkdir(parents=True, exist_ok=True)
    _K2_EXT = load(
        name="dcu_megamoe_v2_k2_fused_ext",
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-DNDEBUG", "-ffast-math"],
        verbose=verbose,
    )
    return _K2_EXT


def swiglu_reference(
    l1_out: torch.Tensor,
    route_weights: torch.Tensor | None = None,
    *,
    num_per_channels: int,
    clamp_value: float | None = 10.0,
) -> torch.Tensor:
    """Torch reference for the current DCU K2 SwiGLU convention."""
    if l1_out.dim() != 2 or l1_out.size(1) != num_per_channels * 2:
        raise ValueError("l1_out must be [rows, 2 * num_per_channels]")
    gate = l1_out[:, :num_per_channels].float()
    up = l1_out[:, num_per_channels:].float()
    if clamp_value is not None:
        up = torch.clamp(up, min=-float(clamp_value), max=float(clamp_value))
        gate = torch.minimum(gate, torch.tensor(float(clamp_value), device=gate.device))
    y = gate * torch.sigmoid(gate) * up
    if route_weights is not None and route_weights.numel() > 0:
        y = y * route_weights.float().view(-1, 1)
    return y


def swiglu_quant_channelwise_out_v2(
    l1_out: torch.Tensor,
    route_weights: torch.Tensor,
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    out_bf16: torch.Tensor,
    *,
    num_per_channels: int,
    output_bf16: bool = False,
    clamp_value: float | None = 10.0,
    row_combine_ptrs: torch.Tensor | None = None,
    verbose_build: bool = False,
) -> None:
    """V2 K2 wrapper around the existing optimized DCU K2 implementation."""
    if l1_out.dim() != 2:
        raise ValueError("l1_out must be [rows, 2 * hidden]")
    hidden = l1_out.shape[1] // 2
    if hidden != num_per_channels or l1_out.shape[1] != hidden * 2:
        raise ValueError("num_per_channels must match l1_out.shape[1] // 2")
    ext = _load_k2_ext(verbose=verbose_build)
    weights = (
        route_weights
        if route_weights is not None
        else torch.empty(0, device=l1_out.device, dtype=torch.float32)
    )
    ext.swiglu_quant_channelwise_out(
        l1_out.contiguous(),
        weights.contiguous(),
        act_fp8,
        act_scale,
        out_bf16,
        bool(output_bf16),
        clamp_value is not None,
        float(clamp_value or 0.0),
        row_combine_ptrs.contiguous() if row_combine_ptrs is not None else None,
    )
