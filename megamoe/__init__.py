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
    fp8, scale = cast_to_fp8_channelwise(weights.reshape(num_groups * rows, k))
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
    return _C.get_mega_moe_hip_build_config()


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

        self.buffer, buffer_ptr, local_handle = _C.allocate_hip_ipc_buffer(num_bytes)
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
        self.group.barrier()
        torch.cuda.synchronize()

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
):
    _C.fp8_mega_moe(
        y,
        l1_weights,
        l2_weights,
        cumulative_local_expert_recv_stats,
        sym_buffer.buffer,
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
