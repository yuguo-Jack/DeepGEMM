from __future__ import annotations

import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SCRATCH_DIR = REPO_ROOT / "hygon_tmp" / "large_opt" / "K1_fused"

ORIGINAL_ASM_NAME = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16"
ORIGINAL_ASM_SRC = THIS_DIR / f"{ORIGINAL_ASM_NAME}.s"
ORIGINAL_ASM_CO = SCRATCH_DIR / f"{ORIGINAL_ASM_NAME}.co"

FUSED_L1_ASM_NAME = (
    "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_"
    "MEGAMOE_DISPATCH_PULL_L1"
)
FUSED_L1_ASM_SRC = THIS_DIR / f"{FUSED_L1_ASM_NAME}.s"
FUSED_L1_ASM_CO = SCRATCH_DIR / f"{FUSED_L1_ASM_NAME}.co"

K1_SUPPORTED_RANKS = 8
K1_SUPPORTED_EXPERTS = 256
K1_SUPPORTED_TOPK = 6
K1_SUPPORTED_HIDDEN = 4096
K1_SUPPORTED_ALIGNMENT = 256
K1_SHAPE_CONTRACT = (
    "K1_fused dispatch-pull L1 asm currently supports only ranks=8, "
    "experts=256, local_experts=32, topk=6, hidden=4096, alignment=256, "
    "and 0<num_tokens_per_rank<=num_max_tokens_per_rank"
)


def _check_fused_l1_shape(
    *,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    alignment: int,
) -> None:
    if (
        int(num_ranks) != K1_SUPPORTED_RANKS
        or int(num_experts) != K1_SUPPORTED_EXPERTS
        or int(num_topk) != K1_SUPPORTED_TOPK
        or int(hidden) != K1_SUPPORTED_HIDDEN
        or int(alignment) != K1_SUPPORTED_ALIGNMENT
        or int(num_tokens) <= 0
        or int(num_tokens) > int(num_max_tokens_per_rank)
    ):
        raise ValueError(K1_SHAPE_CONTRACT)


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


def _compile_asm_code_object(src: Path, co: Path) -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"asm source not found: {src}")
    if co.exists() and co.stat().st_mtime >= src.stat().st_mtime:
        return co

    lock_dir = co.with_name(f"{co.name}.lock")
    lock_acquired = False
    while not lock_acquired:
        try:
            lock_dir.mkdir()
            lock_acquired = True
        except FileExistsError:
            if co.exists() and co.stat().st_mtime >= src.stat().st_mtime:
                return co
            try:
                lock_age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age > 180:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            time.sleep(0.1)

    configure_dtk_env()
    try:
        if co.exists() and co.stat().st_mtime >= src.stat().st_mtime:
            return co
        dtk = Path(os.environ["ROCM_HOME"])
        clang = Path(os.environ.get("K1_CLANG", str(dtk / "aillvm" / "bin" / "clang")))
        if not clang.exists():
            clang = Path("clang")
        obj = SCRATCH_DIR / f"{src.stem}.o"
        subprocess.run(
            [
                str(clang),
                "-x",
                "assembler",
                "-target",
                "amdgcn-amd-amdhsa",
                "-mcode-object-version=4",
                "-mcpu=gfx938",
                "-c",
                "-o",
                str(obj),
                str(src),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(clang),
                "-target",
                "amdgcn-amd-amdhsa",
                str(obj),
                "-o",
                str(co),
            ],
            check=True,
        )
        return co
    finally:
        if lock_acquired:
            try:
                lock_dir.rmdir()
            except OSError:
                pass


def ensure_original_asm_code_object() -> Path:
    return _compile_asm_code_object(ORIGINAL_ASM_SRC, ORIGINAL_ASM_CO)


def ensure_fused_l1_asm_code_object() -> Path:
    return _compile_asm_code_object(FUSED_L1_ASM_SRC, FUSED_L1_ASM_CO)


@lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    configure_dtk_env()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(SCRATCH_DIR / "torch_extensions"))
    return load(
        name="k1_fused_ext",
        sources=[str(THIS_DIR / "k1_fused_ext.cu")],
        extra_include_paths=[str(REPO_ROOT / "deep_gemm" / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx938",
            "-DNDEBUG",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dcu_workspace_offset(num_ranks: int) -> int:
    signal_ptrs_offset = _align(num_ranks * 8, 16)
    return _align(signal_ptrs_offset + num_ranks * 8, 16)


def _symm_x_addr_range(sym_buffer, num_ranks: int, hidden: int) -> tuple[int, int]:
    torch.cuda.synchronize()
    ptrs = (
        sym_buffer.buffer[: int(num_ranks) * 8]
        .view(torch.int64)
        .detach()
        .cpu()
        .tolist()
    )
    ptrs = [int(ptr) for ptr in ptrs if int(ptr) != 0]
    if not ptrs:
        raise RuntimeError("failed to read symm peer pointers")
    x_offset = _dcu_workspace_offset(int(num_ranks))
    base = min(ptrs) + x_offset
    end = max(ptrs) + x_offset + int(sym_buffer.num_max_tokens_per_rank) * int(hidden)
    return base, end - base


def _cached_symm_x_addr_range(sym_buffer, num_ranks: int, hidden: int) -> tuple[int, int]:
    key = (
        int(sym_buffer.buffer.data_ptr()),
        int(num_ranks),
        int(hidden),
        int(sym_buffer.num_max_tokens_per_rank),
    )
    cached = getattr(sym_buffer, "_k1_symm_x_addr_range_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    value = _symm_x_addr_range(sym_buffer, num_ranks, hidden)
    setattr(sym_buffer, "_k1_symm_x_addr_range_cache", (key, value))
    return value


def k1_symm_fused_l1_asm(
    sym_buffer,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    rank_idx: int,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    alignment: int = 256,
    verbose_build: bool = False,
):
    _check_fused_l1_shape(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_tokens=num_tokens,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        alignment=alignment,
    )
    l1_weight, l1_scale = l1_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_fused_l1_asm_code_object()
    symm_base_addr, symm_x_span = _cached_symm_x_addr_range(
        sym_buffer, num_ranks, hidden
    )
    return ext.k1_symm_fused_l1(
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        l1_weight.contiguous(),
        l1_scale.contiguous(),
        int(rank_idx),
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(num_tokens),
        int(num_topk),
        int(hidden),
        int(symm_base_addr),
        int(symm_x_span),
        int(alignment),
        str(code_object),
    )


def k1_l1_original_asm(
    grouped_x: torch.Tensor,
    grouped_x_scale: torch.Tensor,
    m_indices: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    verbose_build: bool = False,
):
    l1_weight, l1_scale = l1_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_original_asm_code_object()
    return ext.k1_l1_original_asm(
        grouped_x.contiguous(),
        grouped_x_scale.contiguous(),
        m_indices.contiguous(),
        l1_weight.contiguous(),
        l1_scale.contiguous(),
        str(code_object),
    )
