# DCU MegaMoE Large K1_fused

Current scope:

- import per-rank inputs into the MegaMoE symmetric buffer
- build K1 route metadata through either asm-route or compact prebuild
- run fused dispatch pull/stage + L1 FP8 grouped GEMM
- run SwigLU + FP8 quant for the local test surface

L2 grouped GEMM and combine/reduce are not part of this directory's benchmark
surface.

## Shape Contract

```text
EP/ranks                 = 8
experts                  = 256
local_experts/rank        = 32
topk                     = 6
hidden                   = 4096
intermediate             = 2048
L1 output features       = 4096
route row tile M         = 256
expert row alignment     = 256
active tokens/rank       = 1..num_max_tokens_per_rank
```

`num_max_tokens_per_rank` is only the symmetric-buffer capacity. The active
`--num-tokens` value is dynamic as long as it fits that capacity. 2048
tokens/rank is a validation point, not a support limit.

## Active Paths

The fused asm source is:

```text
DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1.s
```

`K1_PREBUILD_MODE` controls route metadata generation:

- `auto` (default): choose `asm` or `compact` by estimated 256-row tile saving.
- `asm`: route build, tail clear, dispatch pull, and L1 all happen in the asm
  kernel.
- `compact`: a small external compact prebuild builds
  `ceil(expert_count / 256)` tile metadata, then the same fused asm consumes it.

There is no external fixed-layout prebuild branch in the active path.

For symmetric-buffer x spans under 4GB, asm-route uses the fast
`symm_base + uint32_offset` source addressing. For larger spans, asm-route
stores `{rank-local x offset, source rank}` and reconstructs per-rank MUBUF
source descriptors in the pull stage. Compact prebuild still uses absolute
pointers for its wide-address fallback.

## Auto Rule

K1_fused pads L1 rows in 256-row tiles. Auto switches to compact only near
fixed-layout tile jumps where estimated overcompute is at least 10%.

Representative token windows:

```text
1..1024       -> asm      (1 tile/expert; compact cannot save a full tile)
1025..1441    -> compact  (asm just jumped to 2 tiles/expert)
1442..2389    -> asm      (compact estimated saving <10%)
2390..2797    -> compact  (asm just jumped to 3 tiles/expert)
2798..3754    -> asm
3755..4136    -> compact  (asm just jumped to 4 tiles/expert)
4137..5120    -> asm
5121..5464    -> compact  (asm just jumped to 5 tiles/expert)
5465..6485    -> asm
6486..6781    -> compact  (asm just jumped to 6 tiles/expert)
6782..7850    -> asm
7851..8085    -> compact  (asm just jumped to 7 tiles/expert)
8086..8192    -> asm
```

`fused_rows` in test output is the host capacity rows that the fused path
allocated for the selected mode.

## Commands

Run from the repository root inside the DCU container.

Normal quick validation:

```bash
bash dcu_megamoe_large_opt/K1_fused/run_k1_quick_validate.sh
```

Correctness only:

```bash
K1_SKIP_BENCH=1 bash dcu_megamoe_large_opt/K1_fused/run_k1_quick_validate.sh
```

Small smoke:

```bash
K1_TOKENS=16 K1_SKIP_BENCH=1 bash dcu_megamoe_large_opt/K1_fused/run_k1_quick_validate.sh
```

Force compact or asm:

```bash
K1_PREBUILD_MODE=compact K1_BENCH_SCOPE=k1 K1_TOKENS=1025 bash dcu_megamoe_large_opt/K1_fused/run_k1_quick_validate.sh
K1_PREBUILD_MODE=asm K1_BENCH_SCOPE=k1 K1_TOKENS=2050 bash dcu_megamoe_large_opt/K1_fused/run_k1_quick_validate.sh
```

Fused-only larger-token timing:

```bash
K1_FUSED_ONLY=1 K1_BENCH_SCOPE=k1 K1_TOKENS=4096 bash dcu_megamoe_large_opt/K1_fused/run_k1_quick_validate.sh
```

Direct test script knobs:

```text
--num-processes                  default 8
--num-tokens                     active tokens per rank, default 512
--num-max-tokens-per-rank         symm-buffer capacity, default = --num-tokens
--bench-scope                    k1 or k1_swiglu
--warmup / --repeat              benchmark loop, defaults 2 / 5
--skip-bench                     correctness only
--fused-only                     skip baseline; validate fused finite outputs only
--seed
--out
```

The baseline is fixed to `DeepEP dispatch + HIP preprocess + DeepGEMM L1`
with the same optional SwigLU/quant scope. The DeepEP+DeepGEMM baseline path is
known to hang for some `>2050` token cases in this environment, so use
`K1_FUSED_ONLY=1` for larger-token fused smoke or fused-only timing.

## Files

- `k1_fused.py`: Python entry points and asm code-object compilation.
- `k1_fused_ext.cu`: PyTorch extension, auto rule, compact prebuild kernels,
  metadata layout, and asm launch.
- `test_k1_symm_fused.py`: correctness and wall-time comparison for K1 or
  K1+SwigLU/quant.
- `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s`: original
  DeepGEMM asm reference.

Generated artifacts belong under:

```text
hygon_tmp/large_opt/K1_fused/
```

`run_with_idle_timeout.py` kills a command after 180 seconds without output by
default. Override with `K1_IDLE_TIMEOUT_SECONDS=...` or pass `--idle-timeout`
directly when calling the helper.
