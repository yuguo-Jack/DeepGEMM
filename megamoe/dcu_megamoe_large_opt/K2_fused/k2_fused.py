from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


THIS_DIR = Path(__file__).resolve().parent


def _find_scratch_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "setup.py").exists():
            return path
    return Path.cwd()


SCRATCH_ROOT = _find_scratch_root(THIS_DIR)
SCRATCH_DIR = SCRATCH_ROOT / "hygon_tmp" / "large_opt" / "K2_fused"


def _prepend_env_path(name: str, values: list[str]) -> None:
    old = os.environ.get(name, "")
    parts = [value for value in values if value and Path(value).exists()]
    if old:
        parts.append(old)
    os.environ[name] = ":".join(parts)


def configure_dtk_env() -> None:
    dtk = Path(os.environ.get("DTK_ROOT", os.environ.get("ROCM_HOME", "/opt/dtk")))
    os.environ.setdefault("ROCM_HOME", str(dtk))
    os.environ.setdefault("ROCM_PATH", str(dtk))
    os.environ.setdefault("HIP_PATH", str(dtk / "hip"))
    _prepend_env_path(
        "LD_LIBRARY_PATH",
        [
            "/opt/hyhal/lib",
            str(dtk / ".hyhal" / "rocm_smi" / "lib"),
            str(dtk / "lib"),
            str(dtk / "lib64"),
            str(dtk / "hip" / "lib"),
            str(dtk / "hip" / "lib64"),
        ],
    )
    _prepend_env_path(
        "PATH",
        [
            str(dtk / "bin"),
            str(dtk / "hip" / "bin"),
            str(dtk / "aillvm" / "bin"),
        ],
    )


@lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    configure_dtk_env()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx938"
    os.environ["AMDGPU_TARGETS"] = "gfx938"
    os.environ["TORCH_EXTENSIONS_DIR"] = str(SCRATCH_DIR / "torch_extensions")
    return load(
        name="k2_fused_ext",
        sources=[str(THIS_DIR / "k2_fused_ext.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx938",
            "-DNDEBUG",
            "-ffast-math",
        ],
        with_cuda=True,
        verbose=verbose,
    )


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
