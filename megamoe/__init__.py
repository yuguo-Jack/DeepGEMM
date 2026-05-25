import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from . import _C


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_hip_backend() -> bool:
    return getattr(torch.version, "hip", None) is not None


_LARGE_OPT_FORCE_VALUES = {"1", "true", "yes", "on", "large_opt", "3stage"}
_LARGE_OPT_AUTO_VALUES = {"auto", "threshold", "adaptive"}
_LARGE_OPT_THRESHOLD_ENV = "MEGAMOE_DCU_LARGE_OPT_3STAGE_TOKEN_THRESHOLD"
_LARGE_OPT_DEFAULT_THRESHOLD = 128
_CUDA_GRAPH_MAX_TOKENS_ENV = "MEGAMOE_DCU_CUDA_GRAPH_MAX_TOKENS_PER_RANK"
_CUDA_GRAPH_DEFAULT_MAX_TOKENS = 256


def _large_opt_3stage_mode() -> str:
    value = os.getenv("MEGAMOE_DCU_USE_LARGE_OPT_3STAGE", "auto").strip().lower()
    if value in _LARGE_OPT_FORCE_VALUES:
        return "force"
    if value in _LARGE_OPT_AUTO_VALUES:
        return "auto"
    return "off"


def _large_opt_3stage_threshold() -> int:
    return int(os.getenv(_LARGE_OPT_THRESHOLD_ENV, str(_LARGE_OPT_DEFAULT_THRESHOLD)))


def _large_opt_3stage_selected(num_tokens: int, mode: str, threshold: int) -> bool:
    if mode == "force":
        return True
    if mode == "auto":
        return num_tokens > threshold
    return False


def _cuda_graph_max_tokens_per_rank() -> int:
    value = int(os.getenv(_CUDA_GRAPH_MAX_TOKENS_ENV, str(_CUDA_GRAPH_DEFAULT_MAX_TOKENS)))
    if value <= 0:
        raise ValueError(f"{_CUDA_GRAPH_MAX_TOKENS_ENV} must be positive")
    return value


def _is_current_stream_capturing() -> bool:
    checker = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(checker()) if checker is not None else False


def cast_to_fp8_channelwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("channelwise FP8 cast expects a 2D tensor")
    scale = x.abs().float().amax(dim=1).clamp(min=1.0e-4) / 448.0
    fp8 = (x.float() / scale.view(-1, 1)).to(torch.float8_e4m3fn)
    return fp8.contiguous(), scale.float().contiguous()


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


def get_mega_moe_hip_build_config():
    config = _C.get_mega_moe_hip_build_config()
    config["large_opt_3stage_env"] = "MEGAMOE_DCU_USE_LARGE_OPT_3STAGE"
    config["large_opt_3stage_auto_values"] = sorted(_LARGE_OPT_AUTO_VALUES)
    config["large_opt_3stage_threshold_env"] = _LARGE_OPT_THRESHOLD_ENV
    config["large_opt_3stage_default_threshold"] = _LARGE_OPT_DEFAULT_THRESHOLD
    config["large_opt_3stage_shape"] = "EP8 experts=256 topk=6 hidden=4096 intermediate=2048"
    config["cuda_graph_max_tokens_env"] = _CUDA_GRAPH_MAX_TOKENS_ENV
    config["cuda_graph_default_max_tokens_per_rank"] = _CUDA_GRAPH_DEFAULT_MAX_TOKENS
    config["cuda_graph_execution"] = "persistent_fused_only"
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
    ):
        if not _is_hip_backend():
            raise RuntimeError("megamoe is the HIP/DCU MegaMoE package")

        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self._large_opt_3stage_mode = _large_opt_3stage_mode()
        self._large_opt_3stage_threshold = _large_opt_3stage_threshold()
        self.cuda_graph_max_tokens_per_rank = _cuda_graph_max_tokens_per_rank()
        self._cuda_graph_max_tokens_per_rank = self.cuda_graph_max_tokens_per_rank

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
        self.cuda_graph_num_tokens = torch.empty((1,), dtype=torch.int32, device="cuda")
        self.cuda_graph_num_tokens.fill_(self.cuda_graph_max_tokens_per_rank)
        ipc_handles = [None] * group.size()
        dist.all_gather_object(ipc_handles, local_handle, group)
        buffer_ptrs = _C.open_hip_ipc_handles(ipc_handles, group.rank())
        buffer_ptrs[group.rank()] = buffer_ptr

        signal_num_bytes = _align(max(group.size(), 18) * 4, 128)
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

        (
            self.x,
            self.x_sf,
            self.topk_idx,
            self.topk_weights,
            self.l1_acts,
            self.l1_acts_sf,
            self.l2_acts,
            self.l2_acts_sf,
        ) = slice_input_buffers(self.buffer)

        if self._large_opt_3stage_mode in {"force", "auto"}:
            from .large_opt import prepare_large_opt_3stage

            prepare_large_opt_3stage(
                self,
                verbose_build=os.getenv("MEGAMOE_DCU_LARGE_OPT_VERBOSE_BUILD", "0") == "1",
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
) -> SymmBuffer:
    num_max_tokens_per_rank = _align(
        num_max_tokens_per_rank,
        _C.get_token_alignment_for_mega_moe(),
    )
    return SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
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
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    cuda_graph: bool = False,
):
    call_y = y
    if cuda_graph:
        graph_max_tokens = getattr(sym_buffer, "_cuda_graph_max_tokens_per_rank", None)
        if graph_max_tokens is None:
            graph_max_tokens = _cuda_graph_max_tokens_per_rank()
        if graph_max_tokens > int(sym_buffer.num_max_tokens_per_rank):
            raise ValueError(
                f"{_CUDA_GRAPH_MAX_TOKENS_ENV}={graph_max_tokens} exceeds "
                f"SymmBuffer capacity {sym_buffer.num_max_tokens_per_rank}"
            )
        if int(y.size(0)) < graph_max_tokens:
            raise ValueError(
                "cuda_graph=True requires y to have at least "
                f"{graph_max_tokens} rows; pass a fixed graph-bucket output "
                "buffer and consume only the valid prefix"
            )
        call_y = y[:graph_max_tokens] if int(y.size(0)) != graph_max_tokens else y
        if cumulative_local_expert_recv_stats is not None:
            raise ValueError("cuda_graph=True does not support cumulative_local_expert_recv_stats")
    else:
        large_opt_mode = getattr(sym_buffer, "_large_opt_3stage_mode", None)
        large_opt_threshold = getattr(sym_buffer, "_large_opt_3stage_threshold", None)
        if large_opt_mode is None or large_opt_threshold is None:
            large_opt_mode = _large_opt_3stage_mode()
            large_opt_threshold = _large_opt_3stage_threshold()
        if _large_opt_3stage_selected(int(y.size(0)), large_opt_mode, large_opt_threshold):
            if _is_current_stream_capturing():
                raise RuntimeError(
                    "DCU MegaMoE staged K1/K2/K3 path does not support CUDA Graph capture; "
                    "pass cuda_graph=True with a fixed graph-bucket output buffer to capture "
                    "the persistent fused path, or run staged path outside graph replay"
                )
            from .large_opt import fp8_mega_moe_large_opt_3stage

            fp8_mega_moe_large_opt_3stage(
                y,
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
            )
            return

    common_args = (
        call_y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
        sym_buffer.route_scratch,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.handle.signal_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.num_experts,
        sym_buffer.num_topk,
        recipe,
        activation,
        activation_clamp,
        fast_math,
    )
    if cuda_graph:
        _C.fp8_mega_moe_with_graph_tokens(*common_args, sym_buffer.cuda_graph_num_tokens)
    else:
        _C.fp8_mega_moe(*common_args)


fp8_w8a8_mega_moe = fp8_mega_moe


try:
    _C.init(os.path.dirname(os.path.abspath(__file__)), os.environ.get("ROCM_HOME", "/opt/dtk"))
except Exception:
    pass


__all__ = [
    "SymmBuffer",
    "cast_to_fp8_channelwise",
    "cast_grouped_weight_to_fp8_channelwise",
    "weight8bit_nt_kpack2_marlin",
    "get_mega_moe_hip_build_config",
    "get_symm_buffer_for_mega_moe",
    "transform_fp8_weights_for_mega_moe",
    "deepep_deepgemm_preprocess_channelwise",
    "deepep_deepgemm_postprocess_channelwise",
    "fp8_mega_moe",
    "fp8_w8a8_mega_moe",
]

__version__ = "0.1"
