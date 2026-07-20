from __future__ import annotations

from pathlib import Path

import torch

from ..v3_config import (
    V3_QUANT_FP8,
    V3_QUANT_INT8,
    normalize_v3_backend,
    normalize_v3_quant,
)
from . import k3_fused_ext as _ext


THIS_DIR = Path(__file__).resolve().parent

K3_COMBINE_ASM_NAME = "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE"
K3_COMBINE_PACK5_ASM_CO = THIS_DIR / f"{K3_COMBINE_ASM_NAME}_PACK5.co"
K3_COMBINE_TAIL_REDUCE_PACK5_ASM_CO = THIS_DIR / f"{K3_COMBINE_ASM_NAME}_TAILREDUCE_PACK5.co"
K3_COMBINE_UNIFIED_PACK5_ASM_CO = THIS_DIR / f"{K3_COMBINE_ASM_NAME}_UNIFIED_PACK5.co"
K3_COMBINE_TAIL_REDUCE_UNIFIED_PACK5_ASM_CO = (
    THIS_DIR / f"{K3_COMBINE_ASM_NAME}_TAILREDUCE_UNIFIED_PACK5.co"
)
K3_COMBINE_INT8_PACK5_ASM_CO = (
    THIS_DIR
    / "DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co"
)


def _ensure_prebuilt_code_object(co: Path, label: str) -> Path:
    if not co.exists():
        raise FileNotFoundError(
            f"prebuilt {label} asm code object not found: {co}. "
            "Rebuild and reinstall the megamoe wheel."
        )
    return co


def ensure_k3_combine_pack5_asm_code_object(
    use_unified_weight_layout: bool = False,
    quant_mode: str = V3_QUANT_FP8,
) -> Path:
    quant_mode = normalize_v3_quant(quant_mode)
    if quant_mode == V3_QUANT_INT8:
        if use_unified_weight_layout:
            raise ValueError("INT8 K3 does not support the unified weight layout")
        return _ensure_prebuilt_code_object(
            K3_COMBINE_INT8_PACK5_ASM_CO,
            "K3 YGZP INT8 pack5 combine",
        )
    co = (
        K3_COMBINE_UNIFIED_PACK5_ASM_CO
        if use_unified_weight_layout
        else K3_COMBINE_PACK5_ASM_CO
    )
    return _ensure_prebuilt_code_object(co, "K3 V3 pack5 combine")


def ensure_k3_combine_tail_reduce_pack5_asm_code_object(
    use_unified_weight_layout: bool = False,
) -> Path:
    co = (
        K3_COMBINE_TAIL_REDUCE_UNIFIED_PACK5_ASM_CO
        if use_unified_weight_layout
        else K3_COMBINE_TAIL_REDUCE_PACK5_ASM_CO
    )
    return _ensure_prebuilt_code_object(
        co,
        "K3 V3 pack5 tail-reduce",
    )


def load_extension(verbose: bool = False):
    return _ext


def load_v3_ll_extension(verbose: bool = False):
    try:
        from . import k3_v3_fused_ext as _v3_ext
    except ImportError as exc:
        raise RuntimeError(
            "V3 K3 LL pack5 extension is not built. Rebuild and reinstall the megamoe wheel."
        ) from exc
    return _v3_ext


def _tail_signal_slot_base(num_ranks: int) -> int:
    return 8 if int(num_ranks) <= 8 else int(num_ranks)


def _start_barrier_signal_slot_base(num_ranks: int) -> int:
    if int(num_ranks) <= 8:
        return 18
    return _tail_signal_slot_base(num_ranks) + int(num_ranks)


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
    tail_signal_slot_base = _tail_signal_slot_base(num_ranks)
    addrs = [0] * (2 * int(num_ranks))
    for peer_rank in range(int(num_ranks)):
        # EP8 preserves the historical [8, 15] layout. EP16/EP32 move this
        # window after the rank-barrier slots to avoid signal overlap.
        addrs[peer_rank] = signal_ptrs[peer_rank] + (
            tail_signal_slot_base + int(rank_idx)
        ) * 4
        addrs[int(num_ranks) + peer_rank] = signal_ptrs[int(rank_idx)] + (
            tail_signal_slot_base + peer_rank
        ) * 4
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
    graph_runtime_num_tokens: torch.Tensor | None = None,
    graph_runtime_num_tokens_out: torch.Tensor | None = None,
    graph_tail_signal_generation_out: torch.Tensor | None = None,
    graph_max_tokens: int = -1,
    barrier_signal_slot_base: int | None = None,
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
        graph_runtime_num_tokens.contiguous() if graph_runtime_num_tokens is not None else None,
        graph_runtime_num_tokens_out.contiguous() if graph_runtime_num_tokens_out is not None else None,
        graph_tail_signal_generation_out.contiguous()
        if graph_tail_signal_generation_out is not None
        else None,
        int(graph_max_tokens),
        int(
            _start_barrier_signal_slot_base(num_ranks)
            if barrier_signal_slot_base is None
            else barrier_signal_slot_base
        ),
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


def k3_l2_fused_v3_to_combine(
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
    m_indices: torch.Tensor,
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    row_combine_ptrs: torch.Tensor,
    *,
    backend: str,
    quant_mode: str = V3_QUANT_FP8,
    asm_done_counter: torch.Tensor | None = None,
    asm_signal_addrs: torch.Tensor | None = None,
    asm_done_target: int = 0,
    asm_signal_num_ranks: int = 0,
    asm_signal_generation: int = 0,
    asm_signal_generation_tensor: torch.Tensor | None = None,
    asm_reduce_y: torch.Tensor | None = None,
    sym_buffer=None,
    rank_idx: int = 0,
    num_ranks: int = 0,
    num_experts: int = 0,
    num_tokens: int = 0,
    num_topk: int = 0,
    hidden: int = 0,
    output_workspace: torch.Tensor | None = None,
    prob_storage: torch.Tensor | None = None,
    active_tiles: torch.Tensor | None = None,
    graph_runtime_offset_from_active_tiles: int = 0,
    active_tiles_host_hint: int | None = None,
    graph_runtime_num_tokens: torch.Tensor | None = None,
    ll_split_tail: bool = False,
    ll_block_m: int = 32,
    use_unified_weight_layout: bool = False,
    verbose_build: bool = False,
) -> torch.Tensor | None:
    """V3 K3 dispatch point for staged opt.

    Phase 3 wires staged combine paths. V3 LL pack5 kernels consume the
    per-expert row counts returned by K1 so graph replay can keep a fixed
    captured capacity while doing GEMM work for the runtime-token rows.
    """
    backend = normalize_v3_backend(backend)
    quant_mode = normalize_v3_quant(quant_mode)
    if backend not in ("normal", "ll"):
        raise NotImplementedError("V3 K3 staged path supports normal or LL backend only")
    if quant_mode == V3_QUANT_INT8:
        if backend != "normal":
            raise NotImplementedError("INT8 K3 supports the normal backend only")
        if use_unified_weight_layout:
            raise NotImplementedError("INT8 K3 does not support unified weights")
        if asm_reduce_y is not None:
            raise NotImplementedError(
                "INT8 K3 uses no-tail combine followed by generic reduction"
            )
    if active_tiles_host_hint is not None and quant_mode != V3_QUANT_INT8:
        raise ValueError("active_tiles_host_hint is available for INT8 K3 only")
    if sym_buffer is None and asm_signal_addrs is not None:
        raise ValueError("V3 K3 no-tail path must not receive peer signal tensors")
    if asm_signal_num_ranks or asm_signal_generation:
        if sym_buffer is None:
            raise ValueError("V3 K3 no-tail signal metadata requires sym_buffer")
    if output_workspace is None or prob_storage is None:
        raise ValueError("integrated V3 K3 path requires output_workspace and prob_storage")

    l2_weight, l2_scale = l2_weights
    expected_dtype = torch.int8 if quant_mode == V3_QUANT_INT8 else torch.float8_e4m3fn
    if act_fp8.dtype != expected_dtype or l2_weight.dtype != expected_dtype:
        raise TypeError(
            f"K3 {quant_mode} requires matching activation/weight dtype "
            f"{expected_dtype}; got {act_fp8.dtype} and {l2_weight.dtype}"
        )
    if backend == "normal":
        ext = load_extension(verbose=verbose_build)
        if asm_reduce_y is None:
            code_object = ensure_k3_combine_pack5_asm_code_object(
                use_unified_weight_layout=use_unified_weight_layout,
                quant_mode=quant_mode,
            )
            ext.k3_l2_combine_asm_pack5_out(
                output_workspace,
                act_fp8.contiguous(),
                act_scale.contiguous(),
                m_indices.contiguous(),
                l2_weight.contiguous(),
                l2_scale.contiguous(),
                row_combine_ptrs.contiguous(),
                prob_storage.contiguous(),
                str(code_object),
                active_tiles.contiguous() if active_tiles is not None else None,
                int(graph_runtime_offset_from_active_tiles),
                -1 if active_tiles_host_hint is None else int(active_tiles_host_hint),
            )
            return None
        if sym_buffer is None or asm_done_counter is None or asm_signal_addrs is None:
            raise ValueError("V3 K3 ASM-pack5 tail path requires sym_buffer, done counter, and signal addrs")
        if not num_ranks or not num_experts or num_tokens < 0 or not num_topk or not hidden:
            raise ValueError("V3 K3 ASM-pack5 tail path requires shape metadata")
        code_object = ensure_k3_combine_tail_reduce_pack5_asm_code_object(
            use_unified_weight_layout=use_unified_weight_layout
        )
        ext.k3_l2_combine_asm_tail_reduce_pack5_out(
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
            active_tiles.contiguous() if active_tiles is not None else None,
            int(graph_runtime_offset_from_active_tiles),
            asm_signal_generation_tensor.contiguous()
            if asm_signal_generation_tensor is not None
            else None,
        )
        return None

    ext = load_v3_ll_extension(verbose=verbose_build)
    if backend == "ll":
        if asm_reduce_y is None:
            raise RuntimeError("V3 K3 LL no-tail / tail-reduce-0 path has been retired")
        if sym_buffer is None or asm_done_counter is None or asm_signal_addrs is None:
            raise ValueError("V3 K3 LL tail path requires sym_buffer, done counter, and signal addrs")
        if not num_ranks or not num_experts or num_tokens < 0 or not num_topk or not hidden:
            raise ValueError("V3 K3 LL tail path requires shape metadata")
        signal_generation_arg = (
            asm_signal_generation_tensor.contiguous()
            if asm_signal_generation_tensor is not None
            else None
        )
        runtime_tokens_arg = (
            graph_runtime_num_tokens.contiguous()
            if graph_runtime_num_tokens is not None
            else None
        )
        common_args = (
            output_workspace,
            act_fp8.contiguous(),
            act_scale.contiguous(),
            m_indices.contiguous(),
            l2_weight.contiguous(),
            l2_scale.contiguous(),
            row_combine_ptrs.contiguous(),
            sym_buffer.buffer,
            asm_done_counter.contiguous(),
            asm_signal_addrs.contiguous(),
            asm_reduce_y.contiguous(),
            int(num_ranks),
            int(num_experts),
            int(sym_buffer.num_max_tokens_per_rank),
            int(num_tokens),
            int(num_topk),
            int(ll_block_m),
        )
        if ll_split_tail:
            ext.k3_v3_ll_combine_tail_split(
                output_workspace,
                act_fp8.contiguous(),
                act_scale.contiguous(),
                m_indices.contiguous(),
                l2_weight.contiguous(),
                l2_scale.contiguous(),
                row_combine_ptrs.contiguous(),
                sym_buffer.buffer,
                asm_done_counter.contiguous(),
                asm_signal_addrs.contiguous(),
                asm_reduce_y.contiguous(),
                int(rank_idx),
                int(num_ranks),
                int(num_experts),
                int(sym_buffer.num_max_tokens_per_rank),
                int(num_tokens),
                int(num_topk),
                int(ll_block_m),
                signal_generation_arg,
                runtime_tokens_arg,
            )
        else:
            ext.k3_v3_ll_combine_tail(
                *common_args,
                signal_generation_arg,
                runtime_tokens_arg,
            )
        return None
    raise NotImplementedError(f"unsupported V3 K3 backend: {backend}")
