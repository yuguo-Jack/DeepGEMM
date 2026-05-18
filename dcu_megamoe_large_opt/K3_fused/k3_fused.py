from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SCRATCH_DIR = REPO_ROOT / "hygon_tmp" / "large_opt" / "K3_fused"

ORIGINAL_ASM_NAME = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16"
ORIGINAL_ASM_SRC = THIS_DIR / f"{ORIGINAL_ASM_NAME}.s"
ORIGINAL_ASM_CO = SCRATCH_DIR / f"{ORIGINAL_ASM_NAME}.co"

K3_COMBINE_ASM_NAME = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE"
K3_COMBINE_ASM_SRC = THIS_DIR / f"{ORIGINAL_ASM_NAME}_K3COMBINE.s"
K3_COMBINE_ASM_CO = SCRATCH_DIR / f"{K3_COMBINE_ASM_NAME}.co"
K3_COMBINE_TAIL_REDUCE_ASM_SRC = THIS_DIR / f"{ORIGINAL_ASM_NAME}_K3COMBINE_TAILREDUCE.s"
K3_COMBINE_TAIL_REDUCE_ASM_CO = SCRATCH_DIR / f"{K3_COMBINE_ASM_NAME}_TAILREDUCE.co"


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

    configure_dtk_env()
    dtk = Path(os.environ["ROCM_HOME"])
    clang = Path(os.environ.get("K3_CLANG", str(dtk / "aillvm" / "bin" / "clang")))
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


def ensure_original_asm_code_object() -> Path:
    return _compile_asm_code_object(ORIGINAL_ASM_SRC, ORIGINAL_ASM_CO)


def ensure_k3_combine_asm_code_object() -> Path:
    return _compile_asm_code_object(K3_COMBINE_ASM_SRC, K3_COMBINE_ASM_CO)


def ensure_k3_combine_tail_reduce_asm_code_object() -> Path:
    return _compile_asm_code_object(K3_COMBINE_TAIL_REDUCE_ASM_SRC, K3_COMBINE_TAIL_REDUCE_ASM_CO)


@lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    configure_dtk_env()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(SCRATCH_DIR / "torch_extensions")
    return load(
        name="k3_fused_ext",
        sources=[str(THIS_DIR / "k3_fused_ext.cu")],
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


def _dcu_signal_ptrs_offset(num_ranks: int) -> int:
    return _align(int(num_ranks) * 8, 16)


def build_asm_tail_signal_addrs(sym_buffer, *, rank_idx: int, num_ranks: int) -> torch.Tensor:
    torch.cuda.synchronize()
    signal_offset = _dcu_signal_ptrs_offset(num_ranks)
    signal_ptrs = (
        sym_buffer.buffer[signal_offset : signal_offset + int(num_ranks) * 8]
        .view(torch.int64)
        .detach()
        .cpu()
        .tolist()
    )
    signal_ptrs = [int(ptr) for ptr in signal_ptrs]
    if len(signal_ptrs) != int(num_ranks) or any(ptr == 0 for ptr in signal_ptrs):
        raise RuntimeError("failed to read DCU symm signal pointers")
    addrs = [0] * 16
    for peer_rank in range(int(num_ranks)):
        # Slots [8, 15] are reserved for K3 ASM tail generation signals.
        # The regular MegaMoE rank barrier uses [0, num_ranks), while the
        # local-block barrier uses 16 and 17.
        addrs[peer_rank] = signal_ptrs[peer_rank] + (8 + int(rank_idx)) * 4
        addrs[8 + peer_rank] = signal_ptrs[int(rank_idx)] + (8 + peer_rank) * 4
    return torch.tensor(addrs, device=sym_buffer.buffer.device, dtype=torch.int64)


def k3_l2_original_asm(
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    m_indices: torch.Tensor,
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    *,
    verbose_build: bool = False,
) -> torch.Tensor:
    l2_weight, l2_scale = l2_weights
    ext = load_extension(verbose=verbose_build)
    code_object = ensure_original_asm_code_object()
    return ext.k3_l2_original_asm(
        act_fp8.contiguous(),
        act_scale.contiguous(),
        m_indices.contiguous(),
        l2_weight.contiguous(),
        l2_scale.contiguous(),
        str(code_object),
    )


def build_row_combine_ptrs(
    sym_buffer,
    output_index: torch.Tensor,
    recv_to_global_token: torch.Tensor,
    *,
    total_rows: int,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    verbose_build: bool = False,
) -> torch.Tensor:
    ext = load_extension(verbose=verbose_build)
    return ext.build_row_combine_ptrs(
        sym_buffer.buffer,
        output_index.contiguous(),
        recv_to_global_token.contiguous(),
        int(total_rows),
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(num_tokens),
        int(num_topk),
        int(hidden),
    )


def rank_barrier(
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    ext.rank_barrier(sym_buffer.buffer, int(rank_idx), int(num_ranks))


def reset_asm_tail_signal_slots(
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    ext.reset_asm_tail_signal_slots(
        sym_buffer.buffer,
        int(rank_idx),
        int(num_ranks),
    )


def scatter_l2_to_combine(
    l2_out: torch.Tensor,
    row_combine_ptrs: torch.Tensor,
    *,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    ext.scatter_l2_to_combine(l2_out.contiguous(), row_combine_ptrs.contiguous())


def reduce_local_combine(
    y: torch.Tensor,
    sym_buffer,
    *,
    num_ranks: int,
    num_experts: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    ext.reduce_local_combine(
        y,
        sym_buffer.buffer,
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(num_tokens),
        int(num_topk),
        int(hidden),
    )


def k3_l2_asm_to_combine(
    sym_buffer,
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    m_indices: torch.Tensor,
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    row_combine_ptrs: torch.Tensor,
    *,
    verbose_build: bool = False,
) -> torch.Tensor:
    l2_out = k3_l2_original_asm(
        act_fp8,
        act_scale,
        m_indices,
        l2_weights,
        verbose_build=verbose_build,
    )
    scatter_l2_to_combine(l2_out, row_combine_ptrs, verbose_build=verbose_build)
    return l2_out


def k3_l2_fused_asm_to_combine(
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    m_indices: torch.Tensor,
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    row_combine_ptrs: torch.Tensor,
    *,
    asm_done_counter: torch.Tensor | None = None,
    asm_signal_addrs: torch.Tensor | None = None,
    asm_done_target: int = 0,
    asm_signal_num_ranks: int = 0,
    asm_signal_generation: int = 0,
    asm_reduce_y: torch.Tensor | None = None,
    sym_buffer=None,
    num_ranks: int = 0,
    num_experts: int = 0,
    num_tokens: int = 0,
    num_topk: int = 0,
    hidden: int = 0,
    verbose_build: bool = False,
) -> torch.Tensor:
    l2_weight, l2_scale = l2_weights
    ext = load_extension(verbose=verbose_build)
    if asm_reduce_y is not None:
        if asm_done_counter is None or asm_signal_addrs is None:
            raise ValueError("asm_done_counter and asm_signal_addrs are required with asm_reduce_y")
        if sym_buffer is None:
            raise ValueError("sym_buffer is required with asm_reduce_y")
        code_object = ensure_k3_combine_tail_reduce_asm_code_object()
        l2_out = ext.k3_l2_combine_asm_tail_reduce(
            act_fp8.contiguous(),
            act_scale.contiguous(),
            m_indices.contiguous(),
            l2_weight.contiguous(),
            l2_scale.contiguous(),
            row_combine_ptrs.contiguous(),
            asm_done_counter.contiguous(),
            asm_signal_addrs.contiguous(),
            asm_reduce_y.contiguous(),
            sym_buffer.buffer,
            int(asm_done_target),
            int(asm_signal_num_ranks),
            int(asm_signal_generation),
            int(num_ranks),
            int(num_experts),
            int(sym_buffer.num_max_tokens_per_rank),
            int(num_tokens),
            int(num_topk),
            int(hidden),
            str(code_object),
        )
    else:
        if asm_done_counter is not None or asm_signal_addrs is not None:
            raise ValueError("standalone asm tail-signal path was removed; pass asm_reduce_y for tail-reduce")
        code_object = ensure_k3_combine_asm_code_object()
        l2_out = ext.k3_l2_combine_asm(
            act_fp8.contiguous(),
            act_scale.contiguous(),
            m_indices.contiguous(),
            l2_weight.contiguous(),
            l2_scale.contiguous(),
            row_combine_ptrs.contiguous(),
            str(code_object),
        )
    return None
