# DCU MegaMoE V2

DCU MegaMoE V2 is an isolated prototype for the three-stage fused path:

1. K1: dispatch pull plus L1 FP8 groupgemm in one large kernel.
2. K2: SwiGLU plus channelwise quant.
3. K3: L2 FP8 groupgemm plus combine reduce in one large kernel.

The V2 code is intentionally separate from the existing DCU MegaMoE, large-opt
3-stage, and big-fused paths.

## Current Source Layout

- `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`: standalone V2 K1 pure
  groupgemm harness, derived from the current best C FP8 groupgemm skeleton.
- `csrc/kernels/dcu_megamoe_v2/Makefile`: independent hipcc/aicc build entry.
- `csrc/kernels/dcu_megamoe_v2/pack5_layout_check.cpp`: host-side C++
  pack5 offset verifier used by tests.
- `csrc/kernels/dcu_megamoe_v2/layout.py`: V2 pack5 layout helper used by tests.
- `csrc/kernels/dcu_megamoe_v2/stages.py`: isolated Python stage helpers,
  currently containing the V2 K2 wrapper only.
- `scripts/build_dcu_megamoe_v2.sh`: build and quick benchmark wrapper.
- `tests/test_dcu_megamoe_v2.py`: V2 pack5 layout and K2 correctness tests.

## Pack5 Weight Layout

V2 uses pack5:

```text
[expert, k64_outer, n256_outer, n16_outer, k16_segment, n16_physical, k_inner]
```

Inside each N16 group:

```text
logical_ni = (physical_ni & 3) * 4 + (physical_ni >> 2)
```

The transform is setup-only and must not be included in benchmark timing.

Current layout verification:

- L1 fixture: `N=4096, K=4096`.
- L2 fixture: `N=4096, K=2048`.
- Python `pack5_weight` and C++ `pack5_layout_check` agree on selected flat
  offsets for L1, L2, and a small base shape.

```bash
MODE=layout-check bash scripts/build_dcu_megamoe_v2.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py
```

## K3 Layout Status

Accepted V2 K3 must use the same pack5 layout as K1. The previous K3
ASM/kpack2 experiment has been removed from the active source tree because it
violated the V2 requirement to use the C groupgemm implementation and unified
pack5 layout.

The next accepted K3 path is:

1. Adapt the C pack5 groupgemm skeleton for L2 shape `N=4096, K=2048`.
2. Establish pure K3 C pack5 timings for tokens 32, 128, 1024, and 4096.
3. Fuse row-combine/reduce into the C groupgemm epilogue, with no standalone
   combine kernel in the timed path.
4. Validate real communication with 4 ranks or 8 ranks.

## Build

Run inside the DTK container:

```bash
source /opt/dtk/env.sh
cd /workspace/DeepGEMM
bash scripts/build_dcu_megamoe_v2.sh
```

Quick K1 pure groupgemm checks:

```bash
MODE=small TOKENS="32 128" CHECK=1 bash scripts/build_dcu_megamoe_v2.sh
MODE=large TOKENS="1024 4096" CHECK=1 bash scripts/build_dcu_megamoe_v2.sh
```

Quick K1 fused checks:

```bash
MODE=small-symm-stage TOKENS="32 128" CHECK=1 bash scripts/build_dcu_megamoe_v2.sh
MODE=large-symm-stage TOKENS="1024 4096" CHECK=1 bash scripts/build_dcu_megamoe_v2.sh
```

The active K1 fused modes are the staged paths above. Earlier direct pull modes
were kept in findings as rejected experiments and removed from the build script.

## Baseline Snapshot

Initial pure K1 baseline and V2 parity on remote GPU 0:

| path | tokens | best observed ms | correctness |
|---|---:|---:|---|
| small c-ll BM32/CU64 | 32 | 0.299648 | max_abs 0.000488281, value_mismatch 0 |
| small c-ll BM32/CU64 | 128 | 0.307349 | max_abs 0.000488281, value_mismatch 0 |
| large pack5 aicc | 1024 | 0.752475 | max_abs 0, value_mismatch 0 |
| large pack5 aicc | 4096 | 2.26414 | max_abs 0, value_mismatch 0 |
| V2 small c-ll BM32/CU64 | 32 | 0.299546 | max_abs 0.000488281, value_mismatch 0 |
| V2 small c-ll BM32/CU64 | 128 | 0.307141 | max_abs 0.000488281, value_mismatch 0 |
| V2 large pack5 aicc | 1024 | 0.745766 | max_abs 0, value_mismatch 0 |
| V2 large pack5 aicc | 4096 | 2.26284 | max_abs 0, value_mismatch 0 |

## Rejected K1 Pull Prototype

The first V2 K1 fused prototype was `--mode c-ll-pull` / `MODE=small-pull`.
It was removed from active CLI entry points after staged communication became
the accepted V2 direction.

| tokens | pure V2 ms | pull fused ms | degradation | correctness |
|---:|---:|---:|---:|---|
| 32 | 0.299546 | 0.303910 | +1.46% | max_abs 0.000244141, value_mismatch 0 |
| 128 | 0.307141 | 0.314656 | +2.45% | max_abs 0.000488281, value_mismatch 0 |

Rejected experiment: loading the source-row mapping in every K iteration was
correct but slowed 32/128 tokens by about 57-58%. Caching source rows in
registers per tile removed most of that overhead.

Profile note: `hipprof --hip-trace --stats` for 32 tokens showed seven calls
to the single V2 low-latency kernel for two warmup and five repeat launches.
No extra dispatch/prebuild kernel appears in the timed path.

## Rejected K1 Symm-Pull Prototype

`--mode c-ll-symm-pull` / `MODE=small-symm-pull` used the existing DCU MegaMoE
sym-buffer peer pointer header. The standalone harness filled per-rank
`x/x_sf/topk_idx/topk_weights`; the V2 K1 kernel scans peer routes, builds
source rows in route scratch, and pulls token rows through peer sym-buffer
pointers inside the same low-latency groupgemm kernel.

| tokens | pure V2 ms | symm-pull ms | degradation | correctness |
|---:|---:|---:|---:|---|
| 32 | 0.299269 | 0.333588 | +11.47% | max_abs 0.000244141, value_mismatch 0 |
| 128 | 0.307615 | 0.341599 | +11.05% | max_abs 0.000244141, value_mismatch 0 |

Profile note: `hipprof --hip-trace --stats --check 0` for 32 tokens showed
seven calls to `V2_K1_LowLatencyMaskedGroupGemmKernel` for two warmup and five
repeat launches, with no separate dispatch/prebuild kernel in the timed path.

`--symm-devices 2` distributes rank sym buffers over two visible DCUs and makes
rank0 read peer device pointers directly inside the same fused K1 kernel. This
is retained only as rejected real-communication evidence:

| tokens | pure V2 ms | symm-pull 2 DCUs ms | degradation | correctness |
|---:|---:|---:|---:|---|
| 32 | 0.299269 | 0.394100 | +31.70% | max_abs 0.000244141, value_mismatch 0 |
| 128 | 0.307615 | 0.432901 | +40.73% | max_abs 0.000244141, value_mismatch 0 |

The two-DCU path is correct but above the K1 <=20% target. hipprof still shows
only the fused K1 kernel in the timed path, so the next K1 optimization is to
stage and overlap remote pulls inside the same kernel rather than direct remote
A loads in the compute loop.

`--symm-ranks` is the logical rank count used by the MegaMoE symmetric-buffer
layout and route mapping. `--symm-devices` is how many visible DCUs this
standalone harness uses to place those rank buffers. They are equal for the
normal 8-rank / 8-HCU acceptance run, but can intentionally differ for local
simulation such as 8 logical ranks over 4 visible devices.

`--mode c-ll-symm-stage` is the accepted quick K1 real-communication path so
far. It stages peer token rows into local scratch inside the same fused K1
kernel, then groupgemm reads local staged A. The earlier 2-DCU result was only
smoke; the current acceptance run uses 8 ranks on 8 HCUs.

| tokens | pure V2 ms | symm-stage 2 DCUs ms | degradation | correctness |
|---:|---:|---:|---:|---|
| 32 | 0.299269 | 0.349984 | +16.95% | max_abs 0.000244141, value_mismatch 0 |
| 128 | 0.307615 | 0.358944 | +16.69% | max_abs 0.000244141, value_mismatch 0 |

8-rank / 8-HCU acceptance:

```bash
DEVICE=0,1,2,3,4,5,6,7 SYMM_RANKS=8 SYMM_DEVICES=8 \
  MODE=small-symm-stage TOKENS=32 CHECK=1 bash scripts/build_dcu_megamoe_v2.sh
```

| tokens | pure V2 ms | symm-stage 8 ranks ms | degradation | correctness |
|---:|---:|---:|---:|---|
| 32 | 0.299675 | 0.352000 | +17.46% | max_abs 0.000244141, mean_abs 1.32851e-09, value_mismatch 0 |
| 128 | 0.307098 | 0.363023 | +18.21% | max_abs 0.000244141, mean_abs 1.0992e-09, value_mismatch 0 |

hipprof `--hip-trace --stats` for 32 and 128 tokens shows seven HIPOPS calls
to `V2_K1_LowLatencyMaskedGroupGemmKernel` for warmup=2/repeat=5, 100% of
HIPOPS kernel time, with no standalone dispatch/prebuild kernel in the timed
path. Direct `dccobjdump` on the linked executable only reports host ELF; the
save-temps device assembly under `hygon_tmp/dcu_megamoe_v2/save_temps_k1_hipcc`
contains the staged K1 specialization, `v_mmac_f32_16x16x32_fp8_fp8`, and
`s_waitcnt`.

Rejected staging experiment: scalar byte staging was correct but too slow
(`0.517333 ms` for 32 tokens on one device). The retained version uses 16B
vector copies and only stages rows that K1 block tiles actually read.

## K2 Reuse

V2 K2 currently reuses the existing optimized DCU K2 implementation through
`stages.py`. In repo-local environments where `megamoe._C` is not built
inplace, the wrapper JIT-builds only `K2_fused/k2_fused_ext.cu` into
`hygon_tmp/dcu_megamoe_v2/torch_extensions`.

Current remote V2 pytest result after the K3 reset:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py
7 passed in 13.05s
```

K2 metric snippet: rows=32, hidden=128, max_abs 0, mean_abs 0, mismatch 0.

## Rejected K3 ASM/kpack2 Experiment

The previous K3 ASM/kpack2 prototype produced useful failure data, but it is
not an accepted V2 path:

- It used a K3-specific kpack2 layout instead of the unified V2 pack5 layout.
- It used an ASM code object instead of the requested C groupgemm skeleton.
- It only measured large-token cases and did not cover small-token K3.
- It did not satisfy the 4-rank or 8-rank communication acceptance gate.

Those timings remain only as failure history in the planning logs. They must
not be used as the K3 denominator or as evidence that Phase 7 is complete.

## Next Steps

- Add the large-token K1 fused C pack5 path.
- Rebuild K3 on C pack5 for both small tokens 32/128 and large tokens
  1024/4096.
- Connect K1/K2/K3 into an end-to-end stages-fused V2 test.
- Add uneven tokens per rank and graph replay support after eager correctness
  and quick perf converge.
