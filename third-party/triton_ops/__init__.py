try:
    from .sglang_prepost_dcu import (
        ep_gather_channelwise as triton_ep_gather_channelwise,
        ep_scatter_channelwise as triton_ep_scatter_channelwise,
    )
except Exception:
    triton_ep_gather_channelwise = None
    triton_ep_scatter_channelwise = None

__all__ = [
    "triton_ep_gather_channelwise",
    "triton_ep_scatter_channelwise",
]
