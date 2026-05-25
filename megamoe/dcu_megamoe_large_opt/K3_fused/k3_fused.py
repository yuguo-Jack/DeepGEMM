from __future__ import annotations

from pathlib import Path

import torch

from . import k3_fused_ext as _ext


THIS_DIR = Path(__file__).resolve().parent

K3_COMBINE_ASM_NAME = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE"
K3_COMBINE_ASM_CO = THIS_DIR / f"{K3_COMBINE_ASM_NAME}.co"
K3_COMBINE_TAIL_REDUCE_ASM_CO = THIS_DIR / f"{K3_COMBINE_ASM_NAME}_TAILREDUCE.co"


def _ensure_prebuilt_code_object(co: Path, label: str) -> Path:
    if not co.exists():
        raise FileNotFoundError(
            f"prebuilt {label} asm code object not found: {co}. "
            "Rebuild and reinstall the megamoe wheel."
        )
    return co


def ensure_k3_combine_asm_code_object() -> Path:
    return _ensure_prebuilt_code_object(K3_COMBINE_ASM_CO, "K3 combine")


def ensure_k3_combine_tail_reduce_asm_code_object() -> Path:
    return _ensure_prebuilt_code_object(K3_COMBINE_TAIL_REDUCE_ASM_CO, "K3 tail-reduce")


def load_extension(verbose: bool = False):
    return _ext


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
    k1_graph_reset_layout: tuple[int, int, int, int] | None = None,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    if k1_graph_reset_layout is None:
        route_scratch = None
        flags_offset = flags_numel = meta_flags_offset = meta_flags_numel = 0
    else:
        route_scratch = sym_buffer.route_scratch
        flags_offset, flags_numel, meta_flags_offset, meta_flags_numel = (
            int(v) for v in k1_graph_reset_layout
        )
    ext.rank_barrier(
        sym_buffer.buffer,
        int(rank_idx),
        int(num_ranks),
        asm_done_counter,
        bool(reset_tail_signal_slots),
        route_scratch,
        flags_offset,
        flags_numel,
        meta_flags_offset,
        meta_flags_numel,
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


def reduce_local_combine_graph(
    y: torch.Tensor,
    sym_buffer,
    *,
    num_ranks: int,
    num_experts: int,
    graph_num_tokens: int,
    runtime_num_tokens: torch.Tensor,
    num_topk: int,
    hidden: int,
    verbose_build: bool = False,
) -> None:
    ext = load_extension(verbose=verbose_build)
    ext.reduce_local_combine_graph(
        y,
        sym_buffer.buffer,
        int(num_ranks),
        int(num_experts),
        int(sym_buffer.num_max_tokens_per_rank),
        int(graph_num_tokens),
        runtime_num_tokens.contiguous(),
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
    active_tiles: torch.Tensor | None = None,
    verbose_build: bool = False,
) -> torch.Tensor | None:
    l2_weight, l2_scale = l2_weights
    ext = load_extension(verbose=verbose_build)
    if output_workspace is None or prob_storage is None:
        raise ValueError("integrated K3 path requires output_workspace and prob_storage")
    if asm_reduce_y is not None:
        if active_tiles is not None:
            raise ValueError("K3 graph active-tile gate is not supported with tail-reduce asm")
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
            active_tiles.contiguous() if active_tiles is not None else None,
        )
    return None
