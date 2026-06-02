# Findings: DCU MegaMoE V2

## Maintenance Convention

- The single maintained location for V2 plan/progress/findings/overview is `.planning/dcu_megamoe_v2/`.
- Do not add new V2 progress/findings files under `docs/`.
- Previous `docs/` progress and findings summaries were consolidated into the active planning logs. Unique overview/build/layout notes live in `overview.md`.

## Requirements
- Build a new isolated DCU MegaMoE V2, not a patch on the current DCU MegaMoE path.
- All runtime stages use the 3-stage fused flow:
  - K1: prebuild / dispatch pull fused with L1 FP8 groupgemm in one large kernel.
  - K2: SwiGLU plus quant, preferably reusing current optimized DCU implementation.
  - K3: L2 FP8 groupgemm fused with combine reduce in one large kernel.
- Source, tests, build scripts, and docs must enter git-tracked workspace outside hygon_tmp.
- Temporary logs, profiles, dumps, and caches belong under hygon_tmp or remote /workspace/DeepGEMM/hygon_tmp.
- Do not break existing dcu_megamoe, large_opt 3-stage, or big-fused paths.
- V2 needs independent files, symbols, tests, and build entry.
- Weight layout must be V2-owned; tests must explicitly transform weights to V2 layout.
- Layout transform is setup work and excluded from benchmark timing.
- Correctness target is max_abs <= 1e-3, and tests should report max_abs, mean_abs, mismatch.
- Quick benchmark tokens:
  - Small K1_LowLatencyMaskedGroupGemmKernel path: 32, 128.
  - Large DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16 plus --c-lowlat-pack 1 path: 1024, 4096.
- Full sweep after convergence: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192.
- K1 fused degradation target versus pure K1 groupgemm: <= 20%.
- K3 fused degradation target versus pure K3 groupgemm: <= 25%.
- Avoid hipMalloc, hipFree, D2H, and unnecessary new kernels in execution path.
- Later: uneven tokens per rank and CUDA graph one-graph multi-size support.

## Known Baseline References From User
- hygon_tmp/K1_groupgemm_fp8/README.md
- hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp
- Small C lowlat baseline:
  - 32 tokens: about 0.299 ms
  - 64 tokens: about 0.294 ms
  - 128 tokens: about 0.308 ms
  - 256 tokens: about 0.323 ms
- Large C pack5 baseline:
  - 512 tokens: about 0.69 ms
  - 1024 tokens: about 0.74 ms
  - 2048 tokens: about 1.36 ms
  - 4096 tokens: about 2.25 ms
  - 8192 tokens: fast band about 4.47 ms
- Recommended pack5 layout:
  - [expert, k64_outer, n256_outer, n16_outer, k16_segment, n16_physical, k_inner]
  - logical_ni = (physical_ni & 3) * 4 + (physical_ni >> 2)

## Research Findings
- hygon_tmp/K1_groupgemm_fp8/README.md states the forward-looking C layout is pack5 for both small `--mode c-ll` and large `--mode c --c-lowlat-pack 1`.
- The pack5 host transform in k1_gemm.cpp is represented by `marlin2_k64_n256_n16_transposed_weight` and wrapped by `make_lowlat_pack5_weight`.
- Small-token recommended C path is `K1_LowLatencyMaskedGroupGemmKernel`, built with hipcc. Tuned presets: tokens 32 -> BM32/CU64, 64 -> BM16/CU64, 128 -> BM32/CU64, 256 -> BM48/CU64.
- Large-token recommended C path is `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<256,256>` with `--c-lowlat-pack 1`, built with aicc.
- K1_groupgemm_fp8 harness defaults to MegaMoE-like random FP8 values with input and weight value scale 0.02 and allowed max_abs 0.001.
- `tests/test_mega_moe_dcu.py` baseline currently dispatches with DeepEP, preprocesses by `megamoe.deepep_deepgemm_preprocess_channelwise`, runs DeepGEMM L1, uses current DCU K2 `swiglu_quant_channelwise_out`, runs DeepGEMM L2, then DeepEP combine.
- Existing big fused DCU path is `csrc/kernels/mega_moe_fused_hip.cu`, exposed through `launch_mega_moe_multirank_persistent_hip_w8a8_channelwise`; V2 should not modify it.
- Existing large-opt 3-stage Python orchestration is `megamoe/large_opt.py` and calls independent K1/K2/K3 extension modules under `megamoe/dcu_megamoe_large_opt`.
- Current setup.py builds the main HIP package `megamoe._C` plus three large-opt extensions only when `IS_HIP_EXTENSION and package_name == 'megamoe'`.
- K2 reusable entry is `megamoe.dcu_megamoe_large_opt.K2_fused.k2_fused.swiglu_quant_channelwise_out`; it launches a channelwise SwiGLU+FP8 quant kernel and accepts optional `row_combine_ptrs` to skip inactive rows.
- Existing K1 large-opt path uses prebuilt ASM and HIP metadata kernels. Metadata fields include `route_weights`, `output_index`, `row_combine_ptrs`, `m_indices`, `staged_x`, `staged_flags`, and compact/asm-route prebuild modes.
- Existing K3 large-opt path uses prebuilt ASM for L2+combine and optional tail-reduce. It requires rows padded to 256 and current shape hidden=4096, intermediate=2048, topk=6.
- For V2 isolation, an independent standalone Makefile/harness can validate pure groupgemm and layout first without modifying setup.py or the existing Python package API.
- CUDA/SM100 MegaMoE uses distinct thread roles in one persistent kernel: dispatch warps, non-epilogue GEMM load/issue warps, and epilogue/combine warps.
- CUDA K1 overlap pattern: dispatch warps count expert routes, publish per-expert/per-rank metadata, pass an NVLink barrier, then pull remote token data into local L1 buffers; GEMM load warps wait on per-L1-block arrival counters before consuming.
- CUDA K3 overlap pattern: L2 epilogue writes BF16 output directly to rank/topk combine buffers using source metadata, then epilogue warps run a staged combine reduction with two load stages and one store stage.
- CUDA scheduler uses per-expert pool block offsets and arrival counters/masks. The DCU V2 analogue should avoid full HBM regrouping when possible and use compact metadata / row pointers only where they do not disrupt the groupgemm pipeline.
- V2 `c-ll-symm-pull` now uses the same sym-buffer peer pointer header as existing DCU MegaMoE. The standalone harness allocates per-rank sym-buffer sections, fills `x/x_sf/topk_idx/topk_weights`, and launches one fused K1 kernel that scans peer routes and pulls token rows through `dcu_peer_sym_buffer_ptrs`.
- V2 `c-ll-symm-pull --symm-devices 2` distributes rank sym buffers over two visible DCUs and makes rank0 read peer device pointers directly inside the fused K1 kernel. This is the first real cross-device communication-fused result in V2.
- Direct peer-memory A loads inside the low-latency compute loop are too exposed: two-DCU degradation is +31.70% at 32 tokens and +40.73% at 128 tokens versus pure local K1. This points to a K1 staging/overlap redesign rather than more scalar metadata tuning.
- V2 `c-ll-symm-stage --symm-devices 2` stages peer token rows into local scratch within the same fused K1 kernel. The vectorized staging version satisfies the K1 <=20% target for quick small-token cases: +16.95% at 32 tokens and +16.69% at 128 tokens.
- V2 `c-ll-symm-stage` now has 8-rank/8-HCU small-token acceptance. With `SYMM_RANKS=8`, `SYMM_DEVICES=8`, and `DEVICE=0,1,2,3,4,5,6,7`, 32 tokens are +17.46% versus pure K1 and 128 tokens are +18.21%, both within the <=20% K1 target and correctness-passing.
- 8-rank K1 hipprof evidence shows the timed path launches only `V2_K1_LowLatencyMaskedGroupGemmKernel`: 7 HIPOPS calls for warmup=2/repeat=5 at both 32 and 128 tokens. Setup `hipMalloc/hipMemcpy/peer access/free` appears only because the full process was profiled.
- `dccobjdump` on the linked hipcc executable reports only host ELF, but `hipcc -save-temps=obj` under `hygon_tmp/dcu_megamoe_v2/save_temps_k1_hipcc` produced device assembly containing the staged K1 specialization plus `v_mmac_f32_16x16x32_fp8_fp8` and `s_waitcnt`.
- Large-token K1 direct remote pull in the C pack5 MT256x256 kernel is correct but far too slow. At 1024 tokens, direct pull was 2.30032 ms (+208.45% versus pure 0.745766 ms); after hoisting remote `x` base pointers out of the K loop it improved to 1.47248 ms but still degraded +97.45%. This path is rejected for acceptance.
- The large-token failure suggests remote A traffic must be staged once per row tile inside a persistent/resident compute kernel, rather than pulled again by every N tile.
- Accepted large-token K1 uses a row-cooperative C pack5 staging design, not ASM. Four N-group blocks for the same row tile cooperatively stage different K partitions of remote A into local scratch, then consume the staged rows through the existing MT256 C pack5 groupgemm pipeline. Per-row `count` and `epoch` flags in the kernel argument `grid_barrier` avoid adding a separate memset/prebuild kernel to the timed loop.
- The accepted K1 large-token 8-rank results are:
  - 1024 tokens: pure C pack5 0.772241 ms, fused row-stage 0.916992 ms, degradation +18.74%, max_abs 0, mean_abs 0, mismatch 0.
  - 4096 tokens: pure C pack5 2.28153 ms, fused row-stage 2.62962 ms, degradation +15.26%, max_abs 0, mean_abs 0, mismatch 0.
- Large-token K1 profile evidence shows the timed warmup/repeat loop is a single fused C pack5 compute kernel: hipprof HIPOPS recorded 7 calls to `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16` for both 1024 and 4096 token runs, with 100% of HIPOPS kernel time in that kernel.
- Important K1 large correctness finding: row-cooperative staging with multiple N groups needs an end-of-N-tile block barrier. Without it, 4096-token correctness failed because loader and compute waves could overlap the next N-group iteration through shared LDS/scale state. The accepted kernel makes loader and invalid waves join the barrier before continuing.
- A 64-block global-stage probe was correct but slower, and an occupancy probe showed the large C stage variants run at one block per CU. This makes a full 128-block grid barrier unsafe for the current design; row-cooperative per-row synchronization is the accepted direction.
- Scalar byte staging was a bad experiment (32-token single-device 0.517333 ms) because it parsed sym-buffer sections per byte. The accepted staging path uses 16B copies and skips rows that no groupgemm block tile will read.
- The current `c-ll-symm-pull` correctness reference relies on deterministic modulo routes so rows can be compared directly. Arbitrary router output will need `output_index`/row metadata based validation and downstream K2/K3 consumption instead of assuming expert-internal row order.
- Unified V2 pack5 is now locked for both L1 `N=4096,K=4096` and L2 `N=4096,K=2048` fixtures. Python `pack5_weight` and the C++ `pack5_layout_check` helper agree on selected flat offsets for L1, L2, and a small base shape.
- Layout transform remains setup-only and is not in the timed groupgemm loops. The C++ helper is only a mapping verifier, not an execution-path kernel.
- The previous V2 K3 ASM/kpack2 prototype is rejected for acceptance. It used a K3-specific kpack2 layout and ASM code objects, while the corrected V2 requirement is C groupgemm plus unified pack5 for both small and large tokens.
- The rejected K3 ASM/kpack2 timing data remains useful only as failure history:
  - 1024 tokens: route_rows=6144, padded rows=8192, temporary pure ASM 0.528159 ms, fused ASM 0.632447 ms, degradation +19.75%, max_abs 0.000244140625, mismatch 0.
  - 4096 tokens: route_rows=24576, padded rows=24576, temporary pure ASM 1.187039 ms, fused ASM 1.391935 ms, degradation +17.26%, max_abs 0.000244140625, mismatch 0.
- These rejected K3 results must not be used as the K3 pure denominator, not be reported as accepted Phase 7 performance, and not be treated as evidence that communication is hidden in the required C pack5 pipeline.
- Valid K3 next step: adapt the C pack5 groupgemm skeleton for L2 shape `N=4096,K=2048`, establish pure C pack5 timings for 32/128/1024/4096, then fuse combine in the C epilogue and validate real 4-rank or 8-rank communication.
- Pure K3 C pack5 is now established for `N=4096,K=2048` using the unified V2 pack5 layout:
  - small 32 tokens: 0.155407 ms, max_abs 0.000244141, value_mismatch 0.
  - small 128 tokens: 0.163861 ms, max_abs 0.000244141, value_mismatch 0.
  - large 1024 tokens: 0.439760 ms, max_abs 0, value_mismatch 0.
  - large 4096 tokens: 1.290590 ms, max_abs 0, value_mismatch 0.
- The original K3 C pack5 VMFault was caused by K1-specific constants in the large C skeleton: `kProblemK=4096` and fixed K-stage ordering were unsafe for the L2 `K=2048` shape. The accepted fix parameterizes `kProblemK` while keeping `N=4096`.
- K3 large C pack5 random correctness required a scheduler barrier after direct-to-LDS global loads for the `K=2048` template. Hygon KB references for gfx938 direct-to-LDS load patterns also show a wait plus scheduler barrier sequence before consuming LDS data. The accepted implementation scopes this to the K3/K=2048 large C path rather than changing all K1 wait behavior.
- K1 fused and K3 fused must be optimized and accepted as separate lines. K1 already has accepted 8-rank fused dispatch-pull/staging evidence; K3 still needs its own fused combine implementation, 4/8-rank communication evidence, and <=25% degradation proof against the K3 pure C pack5 denominator.
- K3 C pack5 identity row-combine pointer output is now correctness-clean for small and large paths. With current same-binary pure denominators, identity rowptr degradation is:
  - 32 tokens: pure 0.155658 ms, rowptr 0.160789 ms, +3.30%.
  - 128 tokens: pure 0.164406 ms, rowptr 0.165125 ms, +0.44%.
  - 1024 tokens: pure 0.440655 ms, rowptr 0.518218 ms, +17.60%.
  - 4096 tokens: pure 1.29907 ms, rowptr 1.37193 ms, +5.61%.
- K3 rowptr identity correctness:
  - 32 tokens: max_abs 0.000244141, mean_abs 6.76449e-10, bit_mismatch 19, value_mismatch 0.
  - 128 tokens: max_abs 0.000244141, mean_abs 6.32578e-10, bit_mismatch 75, value_mismatch 0.
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- K3 dynamic raw-buffer row-resource stores are rejected: they were slower and incorrect at 1024 tokens (`max_abs=0.0157471`, `value_mismatch=767`). Correctness recovered after reverting to flat row-address stores and reducing repeated row pointer loads by prefetching row addresses once per row group.
- Identity rowptr is not communication acceptance. The next accepted K3 checkpoint must use real 4-rank or 8-rank row-combine targets in per-rank sym-buffer combine sections and then add in-kernel combine reduce / tail reduce.
- K3 real 8-rank row-combine targets are correctness-clean for 32/128/1024/4096. Small-token row-combine is within the <=25% target, but large-token row-combine is not.
- K3 small real 8-rank row-combine timings:
  - 32 tokens: pure `0.155658` ms, fused `0.15945` ms, degradation `+2.44%`, `value_mismatch=0`.
  - 128 tokens: pure `0.164406` ms, fused `0.163456` ms, effectively no degradation, `value_mismatch=0`.
- K3 large direct remote row-combine stores are too exposed:
  - 1024 tokens: pure `0.440655` ms, direct combine `0.699353` ms, degradation `+58.71%`.
  - 4096 tokens: pure `1.29907` ms, direct combine `2.14729` ms, degradation `+65.29%`.
- K3 large same-kernel copy-stage improves large row-combine but still misses target:
  - 1024 tokens: pure `0.440655` ms, copy-stage `0.619803` ms, degradation `+40.66%`.
  - 4096 tokens: pure `1.29907` ms, copy-stage `1.86016` ms, degradation `+43.19%`.
  - Correctness passes with `max_abs=0`, `mean_abs=0`, `value_mismatch=0` for both 1024 and 4096.
- K3 copy-worker sweep showed 16 workers is best for 1024 tokens. 1/2/4/8 workers were much slower (`~6.13/3.11/1.61/0.865` ms), so reducing copy workers does not improve overlap.
- K3 copy-worker counts above 16 are rejected: `--k3-copy-workers 64` can launch enough copy-waiter blocks to occupy all CUs before compute blocks run, deadlocking on tile-ready flags. V2 now rejects values outside `[1, 16]`.
- K3 vectorized direct row-address epilogue is rejected. The first attempt failed correctness because non-writer lanes returned before shuffle; after fixing lane participation it passed correctness but slowed 1024 tokens to about `1.26` ms, worse than scalar direct remote stores and copy-stage.
- Current K3 row-combine still stores partial rows. It is not final combine-reduce acceptance until in-kernel tail reduce is added and measured.
- K3 same-kernel local-rank tail-reduce is correctness-clean but not performance-accepted:
  - 1024 tokens: pure `0.440655` ms, tail-reduce `0.773269` ms, degradation `+75.49%`, `max_abs=0`, `value_mismatch=0`.
  - 4096 tokens: pure `1.29907` ms, tail-reduce `2.44608` ms, degradation `+88.30%`, `max_abs=0`, `value_mismatch=0`.
  - The prototype reduces only the local rank combine buffer into `out[token, hidden]`; full all-rank/end-to-end reduce acceptance remains pending.
- K3 copy-worker rows must stay at the front of the grid for the current copy-stage design. Moving copy-worker rows after the compute grid preserved correctness but slowed 1024 tokens to about `0.794` ms, worse than the accepted 16-worker front-scheduled copy-stage.
- K3 same-block self-copy is rejected despite promising raw timing. Using `K3_COPY_WORKERS=0` made 1024 tokens run at about `0.57-0.63` ms but failed correctness with missing/zero combine rows, even after adding a store wait/fence and fixing the copy-loop stride for compute-wave-only participation. Per the correctness rule, this experiment was reverted.
- A repaired K3 same-block self-copy, with loader waves and invalid compute waves also participating in the post-epilogue copy, fixed correctness but still lost on performance: 1024 tokens stabilized at `0.665637` ms, slower than the 16-worker copy-stage. The branch remains rejected and was removed again.
- K3/L2 `--c-tile-n 64` with pack5 is rejected for large tokens. Pure 1024 correctness passed but timing was `0.934237` ms, much slower than the MT256 pure denominator (`0.440655` ms), so reducing row padding did not compensate for the less efficient tile schedule.
- Sorting K3 standard-layout rows by `source_rank, partial_row` is rejected. It preserved correctness and slightly improved 1024 copy-stage timing, but 4096 regressed to `3.50465` ms because it disrupted the large-token row/A access order.
- Replacing the K3 copy-stage compute ready `__threadfence_system()` with device-scope `__threadfence()` is rejected. It initially produced faster timings, but a forced rebuild and repeated 1024-token correctness exposed nondeterministic NaN/mismatch failures. The system fence is required for this publication path.
- Increasing K3 copy workers above 16 is rejected. With the system-fence path restored, 24 workers slowed 4096 tokens to `2.31437` ms and 32 workers slowed it to `3.74851` ms. More copy waiters steal too much GEMM residency.
- Current restored K3 large copy-stage baseline remains correctness-clean but above target:
  - 1024 tokens: pure `0.440655` ms, fused `0.613866` ms, degradation `+39.31%`.
  - 4096 tokens: pure `1.29907` ms, fused `1.7904` ms, degradation `+37.82%`.
- ISA/save-temps evidence for the current K3 copy-stage specialization shows no private scratch and about 213 VGPRs, but every copy-worker block still reserves the GEMM kernel's 64 KiB LDS because it is the same kernel. That explains why extra copy-worker blocks steal full-CU residency even though the copy branch itself does not use LDS.
- K3 small same-kernel local-rank tail-reduce is now correctness-clean and within target:
  - 32 tokens: pure `0.155658` ms, tail-reduce fused `0.164469` ms, degradation `+5.66%`, `max_abs=0.000244141`, `value_mismatch=0`.
  - 128 tokens: pure `0.164406` ms, tail-reduce fused `0.184448` ms, degradation `+12.19%`, `max_abs=0.000488281`, `value_mismatch=0`.
  - hipprof for both 32 and 128 shows 7 calls to the same `V2_K1_LowLatencyMaskedGroupGemmKernel<...,2048,...>` low-latency C pack5 kernel and no standalone reduce kernel in the timed loop.
- The K3 small tail-reduce result is still a local-rank prototype, matching the current large tail-reduce correctness scope. Full all-rank/end-to-end combine-reduce acceptance remains pending.
- K3 large PMC triage confirms two different bottlenecks:
  - Direct remote row-combine has modest extra VMEM instruction count but much higher write/TCP stalls (`TCC_EA_WRREQ_STALL` about 4.50M vs pure 0.51M; `TCP_TCP_TA_DATA_STALL_CYCLES` about 15.07M vs pure 2.99M). This is remote scalar store latency/traffic, not launch count.
  - Copy-stage reduces write stalls (`TCC_EA_WRREQ_STALL` about 0.71M) but raises local reads (`SQ_INSTS_VMEM_RD` about 786k vs pure 463k) and keeps the full 64 KiB LDS occupancy cost for copy-worker blocks.
- K3 large direct 32-bit pair-store epilogue is rejected. It was correct at 1024 (`max_abs=0`) but slowed 1024 to `0.711039` ms and 4096 to `2.38252` ms, worse than both the previous direct row-combine and the copy-stage path. The experiment was reverted.
- K3 large tail-reduce local-copy filtering is correctness-clean but only a modest prototype improvement:
  - 1024 tokens: tail-reduce improves from `0.773269` ms to `0.764443` ms, still `+73.49%` versus pure `0.440655` ms.
  - 4096 tokens: tail-reduce improves from `2.44608` ms to `2.39046` ms, still `+84.01%` versus pure `1.29907` ms.
  - This confirms that copying non-local rows is not the dominant large tail-reduce bottleneck; reduce sweep plus copy-worker/synchronization structure remains the issue.
- K3 large local-rank tail-reduce topk-slot skipping improves the prototype further but remains far above target:
  - 1024 tokens: `0.744538` ms, `+68.96%` versus pure `0.440655` ms.
  - 4096 tokens: `2.34016` ms, `+80.14%` versus pure `1.29907` ms.
  - This skip is valid only for the current local-rank prototype because full all-rank reduce must eventually sum all topk slots after every rank has written its contribution.
- K3 large copy-stage synchronization experiments on 2026-05-29 should not be repeated as-is:
  - Publishing tile-ready flags with `atomicExch` preserved 8-rank correctness but regressed timing (`1024` min `0.622042` ms, `4096` min `1.82321` ms), so atomic publication is rejected for the current copy-stage path.
  - A raw-buffer GLC/cache-bypass flag load/store variant compiled but hung during 8-rank correctness and caused the `megamoe` container to exit; the host later reported `No hycu Driver loaded`. This variant is rejected until a safer standalone synchronization microbench proves the exact flag semantics on this platform.
- K3 large copy-worker row-pointer half-wave broadcast is rejected. It reduced repeated `row_output_ptrs[row]` loads in source, and passed 8-rank correctness at 1024/4096 with `max_abs=0`, but timing regressed to `0.625124` ms at 1024 and `1.81698` ms at 4096. The shuffle/address-broadcast overhead outweighed metadata-load savings.
- K3 large copy-worker row-tile scheduling is rejected. Assigning each worker a subset of row tiles and sweeping hidden tiles in order preserved 8-rank correctness (`max_abs=0` for 1024/4096), but timing regressed to `0.755052` ms at 1024 and `2.02361` ms at 4096. The original linear `tile += worker_count` order better overlaps with tile-ready publication and should remain the baseline for this copy-stage design.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Start with isolated skeleton and pure groupgemm parity before fusing communication | Keeps regressions contained and gives a fair degradation denominator. |
| Keep all profile evidence in planning/docs, with raw artifacts in hygon_tmp | Meets git cleanliness requirement while preserving performance reasoning. |
| Use GPU 0 for initial quick baseline | hy-smi showed all HCUs idle; GPU 0 had 0% VRAM and 0% HCU use. |

## Measured Quick Baselines
| Stage | Tokens | Best observed ms | Correctness |
|-------|--------|------------------|-------------|
| K1 pure small c-ll BM32/CU64 | 32 | 0.299648 | max_abs 0.000488281, value_mismatch 0 |
| K1 pure small c-ll BM32/CU64 | 128 | 0.307349 | max_abs 0.000488281, value_mismatch 0 |
| K1 pure large pack5 aicc | 1024 | 0.752475 | max_abs 0, value_mismatch 0 |
| K1 pure large pack5 aicc | 4096 | 2.26414 | max_abs 0, value_mismatch 0 |
| V2 K1 pure small c-ll BM32/CU64 | 32 | 0.299546 | max_abs 0.000488281, value_mismatch 0 |
| V2 K1 pure small c-ll BM32/CU64 | 128 | 0.307141 | max_abs 0.000488281, value_mismatch 0 |
| V2 K1 pure large pack5 aicc | 1024 | 0.745766 | max_abs 0, value_mismatch 0 |
| V2 K1 pure large pack5 aicc | 4096 | 2.26284 | max_abs 0, value_mismatch 0 |
| V2 K1 pull fused cached rows | 32 | 0.303910 | max_abs 0.000244141, value_mismatch 0, degradation +1.46% |
| V2 K1 pull fused cached rows | 128 | 0.314656 | max_abs 0.000488281, value_mismatch 0, degradation +2.45% |
| V2 K1 symm-pull fused | 32 | 0.333588 | max_abs 0.000244141, value_mismatch 0, degradation +11.47% |
| V2 K1 symm-pull fused | 128 | 0.341599 | max_abs 0.000244141, value_mismatch 0, degradation +11.05% |
| V2 K1 symm-pull fused, 2 DCUs | 32 | 0.394100 | max_abs 0.000244141, value_mismatch 0, degradation +31.70% |
| V2 K1 symm-pull fused, 2 DCUs | 128 | 0.432901 | max_abs 0.000244141, value_mismatch 0, degradation +40.73% |
| V2 K1 symm-stage fused, 2 DCUs | 32 | 0.349984 | max_abs 0.000244141, value_mismatch 0, degradation +16.95% |
| V2 K1 symm-stage fused, 2 DCUs | 128 | 0.358944 | max_abs 0.000244141, value_mismatch 0, degradation +16.69% |

## Failed / Rejected Experiments
| Experiment | Result | Decision |
|------------|--------|----------|
| V2 small `c-ll-pull` loading `pull_src_rows` inside every K iteration | Correctness passed, but 32 tokens slowed to 0.474704 ms (+58.47%) and 128 tokens to 0.482117 ms (+56.97%) | Rejected; cached source rows in registers per tile instead. |
| First V2 `c-ll-symm-pull` row assignment with pure atomic append | Computation was valid but direct K1 row-by-row reference failed because expert-internal rows were permuted | Kept as a finding; current deterministic harness routes use direct row id, while arbitrary route support must carry row metadata into K2/K3. |
| Direct two-DCU peer reads in the compute loop | Correctness passed, but slowdown exceeded K1 target: +31.70% at 32 tokens, +40.73% at 128 tokens | Keep as real-communication baseline; next change should stage/overlap remote pulls within the same kernel. |
| Scalar byte staging for `c-ll-symm-stage` | Correctness passed, but 32-token single-device time was 0.517333 ms | Rejected; vectorized to 16B copies and limited staging to block-tile rows. |
| Large `c-symm-pull` direct remote A loads | Correctness passed, but 1024 tokens still degraded +97.45% after metadata hoisting | Rejected; remote A must be staged once per row tile inside the compute kernel. |
| Large all-block `c-symm-stage` group4 | Correct at 1024, but about 1.085 ms, around +40.5% | Rejected in favor of row-cooperative staging. |
| Large staged group2/group8/global-stage probes | Correct or partly correct but slower than row-cooperative group4; global-stage also constrained by one-block-per-CU occupancy | Rejected for accepted K1 large path. |
| K3 pure via V2 C pack5 harness at `N=4096,K=2048` | VMFault | Rejected as a K3 denominator until the C harness is adapted for the L2 shape/layout. |
| K3 `K=2048` no-K-stage-reorder experiment | Removed the K-stage XOR for `K=2048`, but correctness became much worse with millions of mismatches | Reverted; the sparse random mismatch was not fixed by disabling stage reordering. |
| K3 `K=2048` force-masked-store experiment | Forced masked store even for full row tiles, but correctness still failed and mismatch locations moved | Reverted; store masking was not the root cause. |
| K3 pure via `K3COMBINE` ASM with null row-combine pointers | VMFault | Rejected; V2 wrapper now refuses this path until a dedicated no-combine K3 code object exists. |
| V2 K3 ASM/kpack2 prototype | Correct for the measured large-token smoke path, but used ASM and K3-specific kpack2 layout and omitted small-token plus 4/8-rank acceptance | Rejected; remove from active source/tests/docs and rebuild K3 on C pack5. |

## Profile Notes
- hipprof on `c-ll-pull` tokens=32 with warmup=2/repeat=5 showed 7 HIPOPS calls to the V2 low-latency kernel and no separate dispatch/prebuild kernel in the timed path.
- hipprof on `c-ll-symm-pull` tokens=32 with warmup=2/repeat=5 and `--check 0` showed 7 HIPOPS calls to the V2 low-latency kernel, average 345,462 ns, 100% of kernel time. This confirms no standalone dispatch/prebuild kernel in the timed path.
- hipprof on `c-ll-symm-pull --symm-devices 2` tokens=32 with warmup=2/repeat=5 and `--check 0` showed 7 HIPOPS calls to the V2 low-latency kernel, average 405,234 ns, 100% of kernel time. This isolates the two-DCU slowdown to in-kernel work rather than launch count.
- hipprof on `c-ll-symm-stage --symm-devices 2` tokens=32 with warmup=2/repeat=5 and `--check 0` showed 7 HIPOPS calls to the V2 low-latency kernel, average 359,154 ns, 100% of kernel time.
- Rejected K3 ASM/kpack2 hipprof data: fused 1024 tokens with warmup=2/repeat=5 showed 8 calls to `DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE`, average 585,599 ns. This is retained only as rejected experiment evidence because accepted K3 must be C pack5.
- Rejected K3 ASM/kpack2 pure+fused hipprof data: temporary pure ASM averaged 527,999 ns and `K3COMBINE` averaged 577,379 ns. Do not use this as accepted K3 denominator evidence.
- Direct `dccobjdump` on the linked HIP executable did not expose device ISA; use save-temps or code-object extraction for the next ISA pass.

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Existing git status has unrelated deletion DCU_MEGAMOE_KERNEL_ANALYSIS.md and untracked third-party/composable_kernel/ | Treat as pre-existing user workspace state; do not revert or include unless needed. |

## Cleanup Findings
- Retired ASM/balanced-ASM host plumbing is not needed for accepted V2 because K1/K3 must use the C pack5 implementation.
- Retired `small-pull`, `small-symm-pull`, and `large-symm-pull` modes are kept only as historical measurements in findings; active code now routes K1 fused work through staged variants.
- `--symm-ranks` and `--symm-devices` are intentionally separate:
  - `--symm-ranks` is the logical rank count used by symmetric-buffer layout and route mapping.
  - `--symm-devices` is how many visible DCUs the standalone harness uses to allocate/place those logical rank buffers.
  - They are equal for normal 8-rank/8-HCU acceptance, but can differ for local simulation such as 8 logical ranks over 4 visible devices.

## 2026-06-02 K3 Large Copy-Stage Sync Finding
- K3 large copy-stage tile-ready publication must wait for every compute thread's output stores, not only thread 0's stores. A thread-0-only `s_waitcnt` before `__threadfence_system()` allowed the copy worker to observe the ready flag before other lanes' `out` stores were visible, causing sparse 4096-token mismatches on 8 ranks.
- The corrected all-thread VMEM wait restores 1024/4096 correctness but increases large-token overhead to about +41%; optimization must now work from this correctness-safe baseline.
- After the corrected sync fix, sorted K3 route tasks are a small but stable win for copy-stage large tokens and are correctness-clean on real 8-rank tests. The temporary `K3_SORT_ROWS` script switch was later removed during cleanup; copy-stage now sorts internally by default.
- Direct rowptr 4096 without sorting can still fail correctness sparsely. Direct remote-store experiments should use sorted rows unless the experiment is specifically about proving a new row order.
- `global_store_short ... glc slc` compiles on gfx938 but is rejected for K3 direct rowptr stores: it preserved correctness but slowed 1024 to about 1.50 ms and 4096 to about 5.61 ms.

## 2026-06-02 K3 Large Tail-Reduce Findings
- Same-kernel local-rank tail-reduce now uses the large C pack5 copy-stage kernel, not ASM and not an extra reduce kernel.
- The useful tail-reduce optimizations are:
  - precomputed local topk slot masks, avoiding repeated topk/expert scans in the device reduce loop;
  - a separate final tail output buffer, allowing zero-mask tokens to skip full-row zero stores;
  - an active token list, avoiding the reduce sweep over tokens with no local-rank contribution.
- The active token list is the largest tail-reduce-specific win. It reduced the large fused tail-reduce path to roughly `0.641 ms` at 1024 and `1.891 ms` at 4096 with 8 ranks, both correctness-clean.
- A local tail-row list for copy-stage rows was correctness-clean but only a tiny 4096-token improvement. It was removed during cleanup because the extra kernel arguments and copy branch were not worth the marginal gain.
- Current large tail-reduce degradation remains about `+44%` at 1024 and `+45%` at 4096 versus pure K3 C pack5. The remaining gap is now mostly the copy-stage/synchronization structure and full in-kernel combine-reduce scheduling, not local-rank reduce arithmetic.

## Resources
- .vscode/sftp.json: remote host 10.17.176.13, user hg, remote path /home/hg/yuguo/DeepGEMM.
- Remote execution template: ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa hg@10.17.176.13 "docker exec megamoe bash -lc 'source /opt/dtk/env.sh && cd /workspace/DeepGEMM && <cmd>'"

## Visual/Browser Findings
- None.
