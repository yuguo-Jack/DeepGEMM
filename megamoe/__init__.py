import os
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist

from . import _C
from .dcu_megamoe_opt.v3_config import (
    V3_BACKEND_NORMAL,
    normalize_v3_backend,
)
from .dcu_megamoe_opt.v3_layout import (
    flatten_pack5_weight,
    flatten_pack5_weight_asm_normal,
)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_hip_backend() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _staged_pack5_shape_supported(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    return (num_ranks, num_experts, num_topk, hidden, intermediate_hidden) == (
        8,
        256,
        6,
        4096,
        2048,
    )


def _check_staged_pack5_shape(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> None:
    if not _staged_pack5_shape_supported(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    ):
        raise ValueError(
            "DCU MegaMoE staged LL/normal path currently supports only "
            "EP8 experts=256 topk=6 hidden=4096 intermediate=2048. "
            "Add a shape-specific staged layout before enabling new model sizes."
        )


def _symm_warmup_alloc_enabled(
    requested_num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
) -> bool:
    if not _is_hip_backend():
        return False
    if requested_num_max_tokens_per_rank < 512:
        return False
    if (num_experts, num_topk, hidden, intermediate_hidden) != (256, 6, 4096, 2048):
        return False
    return True


def _is_current_stream_capturing() -> bool:
    checker = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(checker()) if checker is not None else False


def cast_to_fp8_channelwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("channelwise FP8 cast expects a 2D tensor")
    scale = x.abs().float().amax(dim=1).clamp(min=1.0e-4) / 448.0
    fp8 = (x.float() / scale.view(-1, 1)).to(torch.float8_e4m3fn)
    return fp8.contiguous(), scale.float().contiguous()


def pre_dispatch_fp8_channelwise_out(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    out_fp8: torch.Tensor,
    out_scale: torch.Tensor,
    out_topk_idx: torch.Tensor,
    out_topk_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse LL pre-dispatch input quantization and top-k buffer staging."""
    if x.dim() != 2:
        raise ValueError("pre-dispatch x must be a 2D tensor")
    if topk_idx.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError("pre-dispatch topk tensors must be 2D")
    if topk_idx.shape != topk_weights.shape:
        raise ValueError("pre-dispatch topk_idx/topk_weights shape mismatch")
    rows, hidden = x.shape
    if topk_idx.size(0) != rows:
        raise ValueError("pre-dispatch topk rows must match x rows")
    topk = topk_idx.size(1)
    if out_fp8.dim() != 2 or out_fp8.size(0) < rows or out_fp8.size(1) != hidden:
        raise ValueError("out_fp8 must have shape [>=rows, hidden]")
    if out_scale.numel() < rows:
        raise ValueError("out_scale must have at least one scale per row")
    if (
        out_topk_idx.dim() != 2
        or out_topk_weights.dim() != 2
        or out_topk_idx.size(0) < rows
        or out_topk_weights.size(0) < rows
        or out_topk_idx.size(1) != topk
        or out_topk_weights.size(1) != topk
    ):
        raise ValueError("output topk tensors must have shape [>=rows, topk]")

    out_fp8_view = out_fp8[:rows, :hidden]
    out_scale_view = out_scale.view(-1)[:rows]
    out_topk_idx_view = out_topk_idx[:rows, :topk]
    out_topk_weights_view = out_topk_weights[:rows, :topk]
    if rows == 0:
        return out_fp8_view, out_scale_view, out_topk_idx_view, out_topk_weights_view
    if not (
        out_fp8_view.is_contiguous()
        and out_scale_view.is_contiguous()
        and out_topk_idx_view.is_contiguous()
        and out_topk_weights_view.is_contiguous()
    ):
        raise ValueError("pre-dispatch output slices must be contiguous")

    x_arg = x if x.is_contiguous() else x.contiguous()
    topk_idx_arg = topk_idx if topk_idx.is_contiguous() else topk_idx.contiguous()
    topk_weights_arg = (
        topk_weights if topk_weights.is_contiguous() else topk_weights.contiguous()
    )
    try:
        pre_dispatch = _C.pre_dispatch_fp8_channelwise_out
    except AttributeError:
        raise RuntimeError(
            "DCU W8A8 MegaMoE requires _C.pre_dispatch_fp8_channelwise_out; "
            "rebuild the megamoe HIP extension"
        ) from None

    pre_dispatch(
        x_arg,
        topk_idx_arg,
        topk_weights_arg,
        out_fp8_view,
        out_scale_view,
        out_topk_idx_view,
        out_topk_weights_view,
    )
    return out_fp8_view, out_scale_view, out_topk_idx_view, out_topk_weights_view


def cast_grouped_weight_to_fp8_channelwise(
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weights.dim() != 3:
        raise ValueError("MegaMoE grouped weights must be 3D tensors")
    num_groups, rows, k = weights.shape
    flat_weights = weights.reshape(num_groups * rows, k)
    chunk_rows = int(os.getenv("MEGAMOE_DCU_WEIGHT_CAST_CHUNK_ROWS", "8192"))
    if chunk_rows <= 0 or flat_weights.size(0) <= chunk_rows:
        fp8, scale = cast_to_fp8_channelwise(flat_weights)
    else:
        fp8 = torch.empty(flat_weights.shape, dtype=torch.float8_e4m3fn, device=flat_weights.device)
        scale = torch.empty((flat_weights.size(0),), dtype=torch.float32, device=flat_weights.device)
        for offset in range(0, flat_weights.size(0), chunk_rows):
            end = min(offset + chunk_rows, flat_weights.size(0))
            chunk = flat_weights[offset:end]
            chunk_scale = chunk.abs().float().amax(dim=1).clamp(min=1.0e-4) / 448.0
            fp8[offset:end].copy_((chunk.float() / chunk_scale.view(-1, 1)).to(torch.float8_e4m3fn))
            scale[offset:end].copy_(chunk_scale)
    return fp8.reshape(num_groups, rows, k).contiguous(), scale.reshape(num_groups, rows).contiguous()


def weight8bit_nt_kpack2_marlin(
    weight: torch.Tensor,
    k_tile: int = 16,
    n_tile: int = 16,
) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError("MegaMoE Marlin weight layout expects a 3D grouped weight tensor")
    if k_tile != 16 or n_tile != 16:
        raise ValueError("DCU W8A8 MegaMoE currently supports only 16x16 kpack2 Marlin tiles")
    num_groups, rows, k = weight.shape
    if rows % n_tile != 0 or k % k_tile != 0:
        raise ValueError("DCU W8A8 MegaMoE Marlin weights require rows and K divisible by 16")
    return (
        weight.reshape(num_groups, rows // n_tile, n_tile, k // k_tile, k_tile)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .reshape(num_groups, rows // n_tile, k * n_tile)
    )


def weight8bit_nt_kpack2_marlin_masked(
    weight: torch.Tensor,
    k_tile: int = 16,
    k_tile_group: int = 4,
    n_tile: int = 16,
    n_tile_group: int = 16,
) -> torch.Tensor:
    if weight.dim() not in (2, 3):
        raise ValueError("masked Marlin weight layout expects a 2D or 3D weight tensor")
    if k_tile != 16 or k_tile_group != 4 or n_tile != 16 or n_tile_group != 16:
        raise ValueError("DCU masked DeepGEMM currently supports only 16x4 by 16x16 kpack2 tiles")

    if weight.dim() == 2:
        rows, k = weight.shape
        if rows % (n_tile * n_tile_group) != 0 or k % (k_tile * k_tile_group) != 0:
            raise ValueError("masked Marlin weights require rows divisible by 256 and K divisible by 64")
        return (
            weight.reshape(
                rows // (n_tile * n_tile_group),
                n_tile_group,
                n_tile,
                k // (k_tile * k_tile_group),
                k_tile_group,
                k_tile,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .reshape(rows // k_tile, k * k_tile)
        )

    num_groups, rows, k = weight.shape
    if rows % (n_tile * n_tile_group) != 0 or k % (k_tile * k_tile_group) != 0:
        raise ValueError("masked Marlin weights require rows divisible by 256 and K divisible by 64")
    return (
        weight.reshape(
            num_groups,
            rows // (n_tile * n_tile_group),
            n_tile_group,
            n_tile,
            k // (k_tile * k_tile_group),
            k_tile_group,
            k_tile,
        )
        .permute(0, 1, 4, 2, 5, 3, 6)
        .contiguous()
        .reshape(num_groups, rows // k_tile, k * k_tile)
    )


def get_mega_moe_hip_build_config():
    config = _C.get_mega_moe_hip_build_config()
    config["dcu_megamoe_main_path"] = "v3_staged"
    config["v3_backend_policy"] = (
        "caller supplies megamoe_backend='ll' or 'normal'; "
        "auto selection belongs to the framework/test layer"
    )
    config["supported_staged_shape"] = (
        "EP8 experts=256 topk=6 hidden=4096 intermediate=2048"
    )
    config["cuda_graph_max_tokens_source"] = "requested num_max_tokens_per_rank"
    config["cuda_graph_execution"] = "graph=True captures the selected v3_staged backend"
    return config


class SymmBuffer:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
        cuda_graph_max_tokens_per_rank: Optional[int] = None,
        prepare_opt_3stage: bool = True,
    ):
        if not _is_hip_backend():
            raise RuntimeError("megamoe is the HIP/DCU MegaMoE package")

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.cuda_graph_max_tokens_per_rank = int(
            cuda_graph_max_tokens_per_rank
            if cuda_graph_max_tokens_per_rank is not None
            else num_max_tokens_per_rank
        )
        if (
            self.cuda_graph_max_tokens_per_rank <= 0
            or self.cuda_graph_max_tokens_per_rank > self.num_max_tokens_per_rank
        ):
            raise ValueError(
                "cuda_graph_max_tokens_per_rank must be in "
                f"1..{self.num_max_tokens_per_rank}"
            )
        _check_staged_pack5_shape(
            num_ranks=group.size(),
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
        )

        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(),
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch,
            activation,
        )
        route_scratch_num_bytes = _C.get_mega_moe_route_scratch_size_for_mega_moe(
            group.size(),
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            use_fp8_dispatch,
            activation,
        )

        self.buffer, buffer_ptr, local_handle = _C.allocate_hip_ipc_buffer(num_bytes)
        self.route_scratch = torch.empty(
            (route_scratch_num_bytes,),
            dtype=torch.int8,
            device="cuda",
        )
        ipc_handles = [None] * group.size()
        dist.all_gather_object(ipc_handles, local_handle, group)
        buffer_ptrs = _C.open_hip_ipc_handles(ipc_handles, group.rank())
        buffer_ptrs[group.rank()] = buffer_ptr

        # Slots [0, 17] are used by the legacy rank/local barriers and K3
        # tail-reduce signals. The staged K1/K2/K3 path uses 18..21 for rank
        # barriers.
        signal_num_bytes = _align(max(group.size(), 22) * 4, 128)
        signal_ptr, signal_handle = _C.allocate_hip_ipc_signal_buffer(signal_num_bytes)
        signal_handles = [None] * group.size()
        dist.all_gather_object(signal_handles, signal_handle, group)
        signal_ptrs = _C.open_hip_ipc_handles(signal_handles, group.rank())
        signal_ptrs[group.rank()] = signal_ptr
        self.handle = SimpleNamespace(
            buffer_ptrs=buffer_ptrs,
            buffer_ptr=buffer_ptr,
            signal_ptrs=signal_ptrs,
            signal_buffer_ptr=signal_ptr,
        )

        self.buffer.zero_()
        _C.set_mega_moe_peer_ptrs(self.buffer, buffer_ptrs, signal_ptrs)
        torch.cuda.synchronize()
        self.group.barrier()

        slices = slice_input_buffers(self.buffer)
        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = slices[:8]
        self.cuda_graph_num_tokens = slices[8]
        self.cuda_graph_num_tokens.fill_(self.cuda_graph_max_tokens_per_rank)

        if prepare_opt_3stage:
            from .opt import prepare_opt_3stage

            prepare_opt_3stage(
                self,
                verbose_build=os.getenv("MEGAMOE_DCU_OPT_VERBOSE_BUILD", "0") == "1",
            )

    def destroy(self):
        if self.handle is not None:
            torch.cuda.synchronize()
            remote_ptrs = [
                ptr if rank != self.group.rank() else 0
                for rank, ptr in enumerate(self.handle.buffer_ptrs)
            ]
            remote_signal_ptrs = [
                ptr if rank != self.group.rank() else 0
                for rank, ptr in enumerate(self.handle.signal_ptrs)
            ]
            _C.close_hip_ipc_handles(remote_ptrs)
            _C.close_hip_ipc_handles(remote_signal_ptrs)
            _C.free_hip_ipc_signal_buffer(self.handle.signal_buffer_ptr)
            _C.free_hip_ipc_buffer(self.handle.buffer_ptr)
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.route_scratch = None
        self.cuda_graph_num_tokens = None


def get_symm_buffer_for_mega_moe(
    group: dist.ProcessGroup,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
    cuda_graph_max_tokens_per_rank: Optional[int] = None,
) -> SymmBuffer:
    requested_num_max_tokens_per_rank = int(num_max_tokens_per_rank)
    requested_cuda_graph_max_tokens = (
        int(cuda_graph_max_tokens_per_rank)
        if cuda_graph_max_tokens_per_rank is not None
        else requested_num_max_tokens_per_rank
    )
    num_max_tokens_per_rank = _align(
        num_max_tokens_per_rank,
        _C.get_token_alignment_for_mega_moe(),
    )
    if _symm_warmup_alloc_enabled(
        requested_num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    ):
        dummy_tokens = _align(1, _C.get_token_alignment_for_mega_moe())
        dummy_buffer = None
        try:
            dummy_buffer = SymmBuffer(
                group,
                num_experts,
                dummy_tokens,
                num_topk,
                hidden,
                intermediate_hidden,
                use_fp8_dispatch,
                activation,
                cuda_graph_max_tokens_per_rank=1,
                prepare_opt_3stage=False,
            )
        finally:
            if dummy_buffer is not None:
                dummy_buffer.destroy()
    return SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
        cuda_graph_max_tokens_per_rank=requested_cuda_graph_max_tokens,
    )


def transform_fp8_weights_for_mega_moe(
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_fp8, l1_scale = cast_grouped_weight_to_fp8_channelwise(l1_weights)
    l2_fp8, l2_scale = cast_grouped_weight_to_fp8_channelwise(l2_weights)
    return (
        (weight8bit_nt_kpack2_marlin(l1_fp8), l1_scale),
        (weight8bit_nt_kpack2_marlin(l2_fp8), l2_scale),
    )


def transform_fp8_weights_for_masked_deepgemm(
    l1_weights: torch.Tensor,
    l2_weights: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    l1_fp8, l1_scale = cast_grouped_weight_to_fp8_channelwise(l1_weights)
    l2_fp8, l2_scale = cast_grouped_weight_to_fp8_channelwise(l2_weights)
    return (
        (weight8bit_nt_kpack2_marlin_masked(l1_fp8), l1_scale),
        (weight8bit_nt_kpack2_marlin_masked(l2_fp8), l2_scale),
    )


def _select_v3_weight_layout(weights: Any, backend: str):
    if isinstance(weights, dict):
        key = "unified" if "unified" in weights else backend
        if key not in weights:
            raise KeyError(
                f"missing V3 weight layout {key!r}; provide keys 'll'/'normal' "
                "or a single 'unified' layout"
            )
        return weights[key], key
    return weights, backend


def deepep_deepgemm_preprocess_channelwise(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    all_tokens: int,
):
    return _C.deepep_deepgemm_preprocess_channelwise(
        recv_x,
        recv_x_scale,
        recv_topk_ids,
        recv_topk_weights,
        num_recv_tokens_per_expert,
        all_tokens,
    )


def deepep_deepgemm_postprocess_channelwise(
    recv_y: torch.Tensor,
    l2_out: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    apply_topk_weights: bool = False,
):
    _C.deepep_deepgemm_postprocess_channelwise(
        recv_y,
        l2_out,
        recv_topk_ids,
        recv_topk_weights,
        output_index,
        apply_topk_weights,
    )


def fp8_mega_moe(
    y: torch.Tensor,
    l1_weights: Any,
    l2_weights: Any,
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    megamoe_backend: str = V3_BACKEND_NORMAL,
    graph: bool = False,
    capacity_num_tokens: Optional[int] = None,
):
    call_y = y
    v3_backend = normalize_v3_backend(megamoe_backend)
    l1_weights, l1_layout = _select_v3_weight_layout(l1_weights, v3_backend)
    l2_weights, l2_layout = _select_v3_weight_layout(l2_weights, v3_backend)
    if l1_layout != l2_layout:
        raise ValueError(
            f"V3 L1/L2 weight layouts must match, got {l1_layout!r} and {l2_layout!r}"
        )
    use_unified_weight_layout = l1_layout == "unified"
    if recipe != (1, 1, 32):
        raise ValueError("DCU W8A8 MegaMoE expects recipe=(1, 1, 32)")
    if activation != "swiglu":
        raise ValueError("DCU W8A8 MegaMoE supports swiglu only")

    intermediate_hidden = int(l1_weights[1].size(1) // 2)
    staged_shape_supported = _staged_pack5_shape_supported(
        num_ranks=sym_buffer.group.size(),
        num_experts=sym_buffer.num_experts,
        num_topk=sym_buffer.num_topk,
        hidden=int(y.size(1)),
        intermediate_hidden=intermediate_hidden,
    )
    if not staged_shape_supported:
        _check_staged_pack5_shape(
            num_ranks=sym_buffer.group.size(),
            num_experts=sym_buffer.num_experts,
            num_topk=sym_buffer.num_topk,
            hidden=int(y.size(1)),
            intermediate_hidden=intermediate_hidden,
        )

    if graph:
        graph_max_tokens = int(sym_buffer.cuda_graph_max_tokens_per_rank)
        capture_rows = int(y.size(0))
        if capture_rows < graph_max_tokens:
            raise ValueError(
                "CUDA graph mode requires y to have at least "
                f"{graph_max_tokens} rows; pass a fixed graph-bucket output "
                "buffer and consume only the valid prefix"
            )
        if cumulative_local_expert_recv_stats is not None:
            raise ValueError("CUDA graph mode does not support cumulative_local_expert_recv_stats")
        from .opt import _run_opt_3stage_graph

        _run_opt_3stage_graph(
            y,
            l1_weights,
            l2_weights,
            None,
            sym_buffer,
            rank_idx=sym_buffer.group.rank(),
            num_ranks=len(sym_buffer.handle.buffer_ptrs),
            num_experts=sym_buffer.num_experts,
            graph_max_tokens=graph_max_tokens,
            num_topk=sym_buffer.num_topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            v3_backend=v3_backend,
            use_unified_weight_layout=use_unified_weight_layout,
        )
        return
    else:
        if _is_current_stream_capturing():
            raise RuntimeError(
                "DCU MegaMoE CUDA Graph capture requires an explicit graph mode: "
                "pass graph=True with megamoe_backend='ll' or 'normal' "
                "to capture the V3 staged K1/K2/K3 path."
            )
        from .opt import fp8_mega_moe_opt_3stage

        fp8_mega_moe_opt_3stage(
            call_y,
            l1_weights,
            l2_weights,
            cumulative_local_expert_recv_stats,
            sym_buffer,
            rank_idx=sym_buffer.group.rank(),
            num_ranks=len(sym_buffer.handle.buffer_ptrs),
            num_experts=sym_buffer.num_experts,
            num_topk=sym_buffer.num_topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            v3_backend=v3_backend,
            capacity_num_tokens=capacity_num_tokens,
            use_unified_weight_layout=use_unified_weight_layout,
        )
        return


fp8_w8a8_mega_moe = fp8_mega_moe


try:
    _C.init(os.path.dirname(os.path.abspath(__file__)), os.environ.get("ROCM_HOME", "/opt/dtk"))
except Exception:
    pass


__all__ = [
    "SymmBuffer",
    "cast_to_fp8_channelwise",
    "pre_dispatch_fp8_channelwise_out",
    "cast_grouped_weight_to_fp8_channelwise",
    "weight8bit_nt_kpack2_marlin",
    "weight8bit_nt_kpack2_marlin_masked",
    "get_mega_moe_hip_build_config",
    "get_symm_buffer_for_mega_moe",
    "transform_fp8_weights_for_mega_moe",
    "transform_fp8_weights_for_masked_deepgemm",
    "flatten_pack5_weight",
    "flatten_pack5_weight_asm_normal",
    "deepep_deepgemm_preprocess_channelwise",
    "deepep_deepgemm_postprocess_channelwise",
    "fp8_mega_moe",
    "fp8_w8a8_mega_moe",
]

__version__ = "0.1"
