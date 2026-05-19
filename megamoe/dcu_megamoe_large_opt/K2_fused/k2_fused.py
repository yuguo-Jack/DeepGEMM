from __future__ import annotations

import torch

from . import k2_fused_ext as _ext


def load_extension(verbose: bool = False):
    return _ext


def swiglu_quant_channelwise_out(
    x: torch.Tensor,
    topk_weights: torch.Tensor | None,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    out_bf16: torch.Tensor,
    *,
    num_per_channels: int,
    use_col_major_scales: bool = False,
    round_scale: bool = False,
    ue8m0_scale: bool = False,
    output_bf16: bool = False,
    clamp_value: float | None = 10.0,
    row_combine_ptrs: torch.Tensor | None = None,
    verbose_build: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("x must be [rows, 2 * hidden]")
    hidden = x.shape[1] // 2
    if x.shape[1] != hidden * 2:
        raise ValueError("x hidden dimension must be even")
    if num_per_channels != hidden:
        raise ValueError("K2 fused currently implements channelwise one-scale-per-row quant only")
    if use_col_major_scales:
        raise ValueError("K2 fused currently supports use_col_major_scales=False only")
    if round_scale:
        raise ValueError("K2 fused currently supports round_scale=False only")
    if ue8m0_scale:
        raise ValueError("K2 fused currently supports ue8m0_scale=False only")

    ext = load_extension(verbose=verbose_build)
    weights = topk_weights if topk_weights is not None else torch.empty(0, device=x.device, dtype=torch.float32)
    ext.swiglu_quant_channelwise_out(
        x.contiguous(),
        weights.contiguous(),
        out_fp8,
        out_scale,
        out_bf16,
        bool(output_bf16),
        clamp_value is not None,
        float(clamp_value or 0.0),
        row_combine_ptrs.contiguous() if row_combine_ptrs is not None else None,
    )
    out_scale_view = out_scale.view(1, x.shape[0])
    if output_bf16:
        return out_fp8, out_scale_view, out_bf16
    return out_fp8, out_scale_view
