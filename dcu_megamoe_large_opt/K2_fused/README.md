# DCU MegaMoE K2 Fused

This folder isolates the large-shape K2 path:

- input `l1_out`: BF16 `[rows, 2 * intermediate_hidden]`
- optional `topk_weights`: FP32 `[rows]`
- output `act_fp8`: FP8 E4M3 `[rows, intermediate_hidden]`
- output `act_scale`: FP32 channelwise scale, one scale per row

The fused implementation is a hand-written HIP kernel for `SwiGLU + topk weight + channelwise FP8 quant`. It is compared directly with `tilelang_ops.swiglu_apply_weight_to_fp8_dcu`.

Scratch build products are written under `hygon_tmp/largesize/K2_fused`.

Current tuned path:

- `hidden=2048` and `hidden=4096` use register staging for the per-row `y` values.
- Kernel thread policy is selected in C++ from the input shape and `output_bf16`.
- The fastest `hidden=2048, output_bf16=False` policy uses one 64-lane wave per row and a register-only row max reduction.
- `hidden=2048, output_bf16=True` currently selects the 128-thread policy because the extra BF16 stores favor more lanes.
- FP8 output is packed with `v_cvt_pk_fp8_f32` and written as 32-bit stores.
- Other hidden sizes fall back to the generic LDS staging path.

Representative 2048-token large-shape result on the DCU test node:

- `rows=16384, hidden=2048, output_bf16=False`: fused `0.137 ms`, tilelang `0.854 ms`, `6.22x`.
- `rows=16384, hidden=2048, output_bf16=True`: fused `0.170 ms`, tilelang `0.966 ms`, `5.69x`.
