try:
    from .swiglu_apply_weight_to_fp8 import swiglu_apply_weight_to_fp8
except Exception:
    swiglu_apply_weight_to_fp8 = None
from .swiglu_apply_weight_to_fp8_dcu import swiglu_apply_weight_to_fp8_dcu
