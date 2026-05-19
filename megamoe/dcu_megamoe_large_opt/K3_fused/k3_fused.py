from __future__ import annotations

import os
import subprocess
import time
import hashlib
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


def _find_deep_gemm_include_dir(start: Path) -> Path:
    env_dir = os.getenv("MEGAMOE_DCU_INCLUDE_DIR")
    if env_dir:
        return Path(env_dir)
    candidates = [
        start.parents[2] / "deep_gemm" / "include",
        start.parents[1] / "include",
    ]
    for path in candidates:
        if (path / "deep_gemm").exists():
            return path
    return candidates[0]


SCRATCH_ROOT = _find_scratch_root(THIS_DIR)
DEEP_GEMM_INCLUDE_DIR = _find_deep_gemm_include_dir(THIS_DIR)
SCRATCH_DIR = SCRATCH_ROOT / "hygon_tmp" / "large_opt" / "K3_fused"

K3_COMBINE_ASM_NAME = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE"
K3_COMBINE_ASM_SRC = THIS_DIR / (
    "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.s"
)
K3_COMBINE_ASM_CO = SCRATCH_DIR / f"{K3_COMBINE_ASM_NAME}.co"
K3_COMBINE_TAIL_REDUCE_ASM_SRC = THIS_DIR / (
    "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE.s"
)
K3_COMBINE_TAIL_REDUCE_ASM_CO = SCRATCH_DIR / f"{K3_COMBINE_ASM_NAME}_TAILREDUCE.co"
ASM_CACHE_VERSION = "gfx938-cov4-v1"


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


def _asm_source_signature(src: Path) -> str:
    digest = hashlib.sha256()
    digest.update(ASM_CACHE_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(src.read_bytes())
    return digest.hexdigest()


def _asm_signature_path(co: Path) -> Path:
    return co.with_suffix(f"{co.suffix}.sha256")


def _asm_code_object_current(src: Path, co: Path) -> bool:
    sig_path = _asm_signature_path(co)
    if not co.exists() or not sig_path.exists():
        return False
    try:
        return sig_path.read_text(encoding="ascii").strip() == _asm_source_signature(src)
    except OSError:
        return False


def _write_asm_signature(src: Path, co: Path) -> None:
    _asm_signature_path(co).write_text(_asm_source_signature(src) + "\n", encoding="ascii")


def _compile_asm_code_object(src: Path, co: Path) -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"asm source not found: {src}")
    if _asm_code_object_current(src, co):
        return co

    lock_dir = co.with_name(f"{co.name}.lock")
    lock_acquired = False
    while not lock_acquired:
        try:
            lock_dir.mkdir()
            lock_acquired = True
        except FileExistsError:
            if _asm_code_object_current(src, co):
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
        if _asm_code_object_current(src, co):
            return co
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
        _write_asm_signature(src, co)
        return co
    finally:
        if lock_acquired:
            try:
                lock_dir.rmdir()
            except OSError:
                pass


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
        extra_include_paths=[str(DEEP_GEMM_INCLUDE_DIR)],
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


def build_asm_tail_signal_addrs(
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    out: torch.Tensor | None = None,
    verbose_build: bool = False,
) -> torch.Tensor:
    handle = getattr(sym_buffer, "handle", None)
    signal_ptrs = getattr(handle, "signal_ptrs", None)
    if signal_ptrs is None:
        raise RuntimeError("sym_buffer.handle.signal_ptrs is required for K3 tail-reduce signal setup")
    signal_ptrs = [int(ptr) for ptr in signal_ptrs[: int(num_ranks)]]
    if len(signal_ptrs) != int(num_ranks) or any(ptr == 0 for ptr in signal_ptrs):
        raise RuntimeError("failed to get DCU symm signal pointers")
    addrs = [0] * 16
    for peer_rank in range(int(num_ranks)):
        # Slots [8, 15] are reserved for K3 ASM tail generation signals.
        # The regular MegaMoE rank barrier uses [0, num_ranks), while the
        # local-block barrier uses 16 and 17.
        addrs[peer_rank] = signal_ptrs[peer_rank] + (8 + int(rank_idx)) * 4
        addrs[8 + peer_rank] = signal_ptrs[int(rank_idx)] + (8 + peer_rank) * 4
    if out is None:
        return torch.tensor(addrs, device=sym_buffer.buffer.device, dtype=torch.int64)
    ext = load_extension(verbose=verbose_build)
    ext.fill_i64_tensor_from_host(out.contiguous(), addrs)
    return out


def rank_barrier(
    sym_buffer,
    *,
    rank_idx: int,
    num_ranks: int,
    asm_done_counter: torch.Tensor | None = None,
    reset_tail_signal_slots: bool = False,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    ext.rank_barrier(
        sym_buffer.buffer,
        int(rank_idx),
        int(num_ranks),
        asm_done_counter,
        bool(reset_tail_signal_slots),
    )


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
    output_workspace: torch.Tensor | None = None,
    prob_storage: torch.Tensor | None = None,
    verbose_build: bool = False,
) -> torch.Tensor | None:
    l2_weight, l2_scale = l2_weights
    ext = load_extension(verbose=verbose_build)
    if output_workspace is None or prob_storage is None:
        raise ValueError("integrated K3 path requires output_workspace and prob_storage")
    if asm_reduce_y is not None:
        if asm_done_counter is None or asm_signal_addrs is None:
            raise ValueError("asm_done_counter and asm_signal_addrs are required with asm_reduce_y")
        if sym_buffer is None:
            raise ValueError("sym_buffer is required with asm_reduce_y")
        code_object = ensure_k3_combine_tail_reduce_asm_code_object()
        ext.k3_l2_combine_asm_tail_reduce_out(
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
            output_workspace,
            prob_storage,
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
            raise ValueError("asm tail-signal path requires asm_reduce_y for tail-reduce")
        code_object = ensure_k3_combine_asm_code_object()
        ext.k3_l2_combine_asm_out(
            act_fp8.contiguous(),
            act_scale.contiguous(),
            m_indices.contiguous(),
            l2_weight.contiguous(),
            l2_scale.contiguous(),
            row_combine_ptrs.contiguous(),
            output_workspace,
            prob_storage,
            str(code_object),
        )
    return None
