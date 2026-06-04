"""Isolated DCU MegaMoE V2 entry points.

V2 is intentionally opt-in through this package. Importing this module must not
change the existing ``megamoe.fp8_w8a8_mega_moe`` behavior.
"""

from .api import (
    fp8_w8a8_mega_moe_v2,
    get_v2_backend,
    get_v2_stage_plan,
    normalize_v2_backend,
)
from .layout import (
    flatten_pack5_weight,
    pack5_flat_offset,
    pack5_logical_to_physical_ni,
    pack5_physical_to_logical_indices,
    pack5_shape,
    pack5_weight,
    transform_fp8_weights_for_mega_moe_v2_pack5,
    unpack_pack5_weight,
)

__all__ = [
    "flatten_pack5_weight",
    "fp8_w8a8_mega_moe_v2",
    "get_v2_backend",
    "get_v2_stage_plan",
    "normalize_v2_backend",
    "pack5_flat_offset",
    "pack5_logical_to_physical_ni",
    "pack5_physical_to_logical_indices",
    "pack5_shape",
    "pack5_weight",
    "transform_fp8_weights_for_mega_moe_v2_pack5",
    "unpack_pack5_weight",
]
