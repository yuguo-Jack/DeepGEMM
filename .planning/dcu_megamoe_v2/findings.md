# Findings: DCU MegaMoE V2

## Maintenance Convention

- The single maintained location for V2 plan/progress/findings/overview is `.planning/dcu_megamoe_v2/`.
- Do not add new V2 progress/findings files under `docs/`.
- Previous `docs/` progress and findings summaries were consolidated into the active planning logs. Unique overview/build/layout notes live in `overview.md`.

## 2026-06-03 K1 Normal Integrated Metadata Optimization Findings
- K1 normal integrated-vs-pure gap was not only caused by communication. After same-size 4-rank cross-rank testing, the useful optimization target was the real-flow in-kernel metadata/prebuild organization before staged GEMM.
- Grouping the normal K1 metadata scan by `source_rank` is accepted:
  - It keeps the pure groupgemm denominator untouched because the pure path does not instantiate `kUseSymmRowStage`.
  - It reduces repeated `get_sections(...)` / runtime token count resolution inside the flat route loop.
  - 4-rank cross-rank normal 4096 K1 improved to `6.418871 ms` with correctness clean.
- Cooperative normal K1 metadata scan across the 4 x-blocks of each expert's first row tile is accepted:
  - It partitions the route scan by `blockIdx.x` and uses an independent metadata count/epoch region in the existing normal grid barrier allocation.
  - It avoids a full-grid barrier and avoids reusing the later staging row count/epoch, which would risk corrupting the groupgemm pipeline synchronization.
  - 4-rank cross-rank normal 4096 K1 improved to `4.923194 ms`, with max_abs `0.000721931` and mismatch `0`.
  - Normal 2048 K1 improved to `2.707197 ms`; normal 1024 K1 is `1.558238 ms`, a slight regression versus metadata-by-rank but acceptable for the high-throughput backend focus on larger sizes.
- Rejected normal K1 metadata/staging experiments:
  - 64B staged-X copy chunks: correctness-clean but worsened normal 4096 K1 to `6.681271 ms`; preserving copy parallelism is more important than reducing source pointer resolution in that form.
  - Lazy `topk_weight` load after expert filtering: correctness-clean but worsened normal 4096 K1 to `6.721431 ms`; the extra output-index initialization and branch shape outweighed saved weight reads.
  - Direct-pull from sym-buffer without staged-X remains rejected from the previous gate because it made normal 4096 K1 about `34.468 ms`; repeated remote/source activation reads are not hidden by GEMM.
- Remaining K1 normal gap after accepted patches:
  - normal K1 4096 pure `2.339350 ms`, integrated `4.923194 ms`, degradation `+110.45%`;
  - normal K1 2048 pure `1.408670 ms`, integrated `2.707197 ms`, degradation `+92.18%`;
  - normal K1 1024 pure `0.854590 ms`, integrated `1.558238 ms`, degradation `+82.34%`.
- Next conclusion: K1 normal is substantially improved but not near the <=20% target. The immediate largest normal large gap is now K3 4096, where integrated K3 remains about `3.41 ms` versus pure `1.342780 ms`.

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

## Real-Flow Integration Decisions
- V2 real-flow integration is opt-in through independent V2 files and tests. Do not add a separate `MEGAMOE_DCU_USE_V2` switch, and do not change the default `megamoe.fp8_w8a8_mega_moe` behavior during the first integration pass.
- The V2 backend selector is `MEGAMOE_DCU_V2_BACKEND=ll|normal`.
  - `ll` means the low-latency backend: small-token prototype kernels are forced for all tested sizes.
  - `normal` means the high-throughput backend: large-token C pack5 prototype kernels are forced for all tested sizes.
  - Automatic small/large threshold selection is deferred until both backends have been run across the full token sweep.
- The V2 test entry remains `tests/test_dcu_megamoe_v2.py`; extend it for real-flow V2 coverage instead of creating a second V2 test file.
- Real-flow V2 production code should be split by responsibility under `megamoe/dcu_megamoe_v2/` with K1/K2/K3/layout/runtime wrappers. The standalone prototype `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`, its Makefile, and `scripts/build_dcu_megamoe_v2.sh` are references and benchmark harnesses, not the long-term production implementation surface.
- V2 real-flow execution should minimize repeated H2D control traffic. Static setup metadata can be prepared outside timed execution, but per-iteration H2D in the runtime path requires an explicit reason.
- Standalone rank barriers should be avoided in the V2 eager path. Communication synchronization should be embedded inside K1/K3 fused kernels where possible; any external barrier must be justified and recorded.
- K3 large being correctness-clean but above the <=25% degradation target is accepted as an integration-stage caveat, not as final performance acceptance.

## Real Test Call Stack Findings
- `tests/test_mega_moe_dcu.py` constructs the production-style test flow in `test(local_rank, num_local_ranks, args)`: initialize distributed state, create `sym_buffer`, allocate optional DeepEP baseline buffer/config, generate BF16 input and weights, create topk routes, cast input to FP8, and transform weights with the current baseline layout transform.
- The current fused path copies `x_fp8`, `x_scale`, `topk_idx`, and `topk_weights` into `sym_buffer`, then calls `megamoe.fp8_w8a8_mega_moe(y, l1_weights, l2_weights, sym_buffer, ...)`. V2 should mirror this high-level call shape through an independent V2 wrapper while using V2 pack5 weights.
- The current baseline path is: DeepEP `get_dispatch_layout`, DeepEP `dispatch`, `megamoe.deepep_deepgemm_preprocess_channelwise`, DeepGEMM L1, optimized DCU K2 `swiglu_quant_channelwise_out`, DeepGEMM L2, `megamoe.deepep_deepgemm_postprocess_channelwise`, and DeepEP `combine`.
- Correctness in the real test compares fused output to baseline output and checks `cumulative_local_expert_recv_stats` against baseline route counts. V2 real-flow tests should keep both checks.
- Baseline weights are currently produced by `megamoe.transform_fp8_weights_for_mega_moe`; V2 must not use that transform for its fused path. The V2 path needs a separate pack5 transform while baseline comparison keeps the existing baseline transform.

## V2 Package API Findings
- The independent V2 package entry is `megamoe.dcu_megamoe_v2`. It exports `fp8_w8a8_mega_moe_v2`, backend parsing helpers, V2 pack5 layout helpers, and `transform_fp8_weights_for_mega_moe_v2_pack5`.
- `fp8_w8a8_mega_moe_v2` intentionally does not fall back to baseline or big-fused code. Until K1/K3 pybind wrappers are connected, it raises `NotImplementedError` through `runtime.run_stages_fused_v2`.
- `MEGAMOE_DCU_V2_BACKEND` validation is strict: only `ll` and `normal` are accepted. The default is `ll` for explicit V2 calls that do not pass a backend.
- V2 K2 is exposed under `megamoe.dcu_megamoe_v2.K2_fused` as a wrapper around the existing optimized DCU K2 extension. Source remains independent; temporary JIT build output stays under `hygon_tmp/dcu_megamoe_v2/torch_extensions`.
- `tests/test_dcu_megamoe_v2.py` is the single V2 test entry. It now includes package contract tests in addition to prototype layout/K2 tests; package import checks skip in environments that lack the built `megamoe._C` parent package extension.
- The real-flow V2 source layout is now split by responsibility under `megamoe/dcu_megamoe_v2/`. K1 and K3 have Python wrapper files plus dedicated `.cu` extension surfaces; the prototype `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` remains a reference harness rather than the production entry point.
- V2 weight transform API is `transform_fp8_weights_for_mega_moe_v2_pack5`. It returns flattened pack5 FP8 weight tensors plus row-wise scales for L1 and L2 and intentionally does not use the baseline `transform_fp8_weights_for_mega_moe`. Its FP8 scale floor matches the current baseline weight-cast convention: `amax(row).clamp(min=1.0e-4) / 448.0`.
- K1/K3 now have V2 extension loader scaffolds that build into `hygon_tmp/dcu_megamoe_v2/torch_extensions`, plus callable C pack5 launch entry points. The remaining gap is runtime workspace/metadata orchestration, not pybind exposure.
- K1/K3 C pack5 launch entry points are now exposed through V2 pybind extension surfaces:
  - K1 ll uses `launch_k1_ll_symm_stage`, mapping to the low-latency C pack5 dispatch-pull/staged L1 kernel.
  - K1 normal uses `launch_k1_normal_symm_stage`, mapping to the normal MT256 C pack5 row-stage dispatch-pull L1 kernel.
  - K3 ll uses `launch_k3_ll_rowptr_tail_reduce`, mapping to low-latency C pack5 L2 rowptr output plus same-kernel tail reduce.
  - K3 normal uses `launch_k3_normal_copy_stage_tail_reduce`, mapping to normal MT256 C pack5 L2 copy-stage plus same-kernel tail reduce.
- The current K1/K3 extension implementation uses a transitional include bridge: `k1_groupgemm_v2.cpp` now has `DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN`, and V2 extension translation units include the prototype kernel definitions while exporting stage-specific pybind launchers. This avoids changing kernel bodies during integration, but it is not the final source organization. After runtime metadata is stable, move accepted C pack5 kernel definitions into stage-owned V2 headers/sources.
- Default torch JIT is rejected for K1/K3 real-flow loading. The extension compile remained inside hipcc for more than 15 minutes even after splitting raw `.cu` from pybind `.cpp` and forcing a single gfx938 arch. V2 K1/K3 loaders now require prebuilt package extension modules, and `setup.py` registers `megamoe.dcu_megamoe_v2.K1_fused.k1_fused_ext` plus `megamoe.dcu_megamoe_v2.K3_fused.k3_fused_ext`.
- The transitional include bridge now defines `DCU_MEGAMOE_V2_KERNEL_ONLY` for K1/K3 extension builds, so the extension path skips the standalone benchmark host harness, CLI parsing, CPU reference, random-data generation, and host layout helper code. This is a compile-scope reduction only; it does not change standalone benchmark behavior.
- Remote `DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` completed successfully after the K1/K3 package-build switch. The full package build, including existing `_C` and large-opt extensions, took about 235 seconds and produced/imported the V2 K1/K3 extension modules. Import smoke confirmed the four expected launcher symbols are present.
- The include bridge is still not the final source organization. It is acceptable only as a short-term way to preserve known-correct kernel bodies while exposing Python call entry points. Before final real-flow implementation is considered clean, accepted C pack5 kernels should be split into stage-owned V2 headers/sources under the organized K1/K3 directories.
- The pybind launcher layer intentionally does not allocate device memory and does not perform H2D copies. It requires the V2 runtime to pass pre-existing staged workspace, route scratch views, grid barriers, row pointers, and metadata tensors.
- Real-flow V2 is not end-to-end connected yet because the current C pack5 K1 launcher does not produce the full Python-visible metadata set needed by K2/K3: route weights, row expert / m_indices, output_index, row_combine_ptrs, local_topk_mask, and tail token list. This metadata must be generated in an isolated V2 path without repeated per-iteration H2D before `runtime.run_stages_fused_v2` can stop raising.
- `V2StagePlan` records the backend-to-kernel mapping so K1/K3 cannot be silently crossed:
  - `ll`: K1 low-latency C pack5 `K=4096`, K3 low-latency C pack5 `K=2048`, K3 tail-reduce enabled, no copy-stage.
  - `normal`: K1 normal C pack5 `K=4096`, K3 normal C pack5 `K=2048`, K3 copy-stage and tail-reduce enabled.
- Remote sync finding: per-file SCP created too many SSH sessions and triggered connection reset/refused on `10.17.176.13`. Use a single archive/scp upload for the next remote validation attempt.

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

## 2026-06-03 V2 Runtime Workspace Findings
- V2 runtime now creates views over `sym_buffer.route_scratch` instead of allocating new execution tensors:
  - staged input FP8 uses the existing DCU route tile `x_fp8` region;
  - L1 BF16 output uses the `act_bf16` region and is aliased as the K3 normal `l2_workspace`;
  - K2 FP8 activation and scale use the `act_fp8` and `act_scale` regions;
  - route weights and staged input scales use the route tile `tile_route_weight` and `tile_x_scale` regions;
  - V2-only metadata (`problem_size`, `route_scratch_i32`, `grid_barrier`, `row_expert`, `row_output_ptrs`, `local_topk_mask`, `tail_tokens`) is packed into the later route scratch queue region.
- The workspace helper performs no `hipMalloc`, `hipFree`, D2H, or H2D by itself. It only reinterprets the preallocated int8 scratch tensor.
- Capacity currently follows the accepted prototype's balanced per-expert contract: `valid_rows_per_expert = ceil(num_max_tokens_per_rank * topk / local_experts)`, padded to 64 rows for `ll` and 256 rows for `normal`. Uneven or overloaded expert routing still needs an overflow/compaction policy before real-flow acceptance.
- Normal backend grid barrier storage is sized for the K3 copy-stage worst case, `16 * ceil(launch_rows / 256) + 2`; ll backend needs only the two-int grid barrier used by low-latency K1/K3.
- Remote validation after the workspace change: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` passed with `17 passed in 13.32s`.

## 2026-06-03 K1 Metadata Writeback Findings
- V2 K1 ll now has optional metadata output pointers in the same fused low-latency C kernel. During the existing route traversal and staging phase it can write:
  - `route_weights[row]` from the source rank's `topk_weights`;
  - `row_expert/m_indices[row]` as the local expert id;
  - `output_index[source_rank, token, topk]` as the grouped row or `-1`;
  - `row_combine_ptrs[row]` as the source rank combine-buffer partial row pointer;
  - `local_topk_mask[token]` as a bit mask for the current local rank;
  - dense `tail_tokens[token] = token`.
- This does not add a standalone metadata kernel and does not introduce H2D/D2H by itself. It extends the existing K1 ll communication/compute kernel signature and pybind wrapper.
- The dense `tail_tokens[:num_tokens]` form is correctness-oriented. It avoids a separate compaction kernel for now; if K3 tail reduce overhead becomes material in the integrated path, replace it with an in-kernel compact active-token list and explicit count.
- K1 normal is not yet real-topk metadata-clean. Its current row-stage still relies on the deterministic helper used by the prototype route pattern. Real-flow normal backend needs the same topk-driven assignment contract as ll before the metadata/prebuild parent item can be marked complete.
- Remote validation after the K1 ll metadata writeback change:
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - `PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` first passed with `17 passed in 14.20s`, then passed with the K1 ll metadata correctness test added: `18 passed in 14.84s`.
  - K1/K3 V2 extension import smoke printed `k1 True True` and `k3 True True`.

## 2026-06-03 K1 normal Metadata Barrier Finding
- K1 normal real-topk metadata cannot use a full-grid in-kernel barrier across the 4 x 32 block grid. The normal C pack5 row-stage kernel can run at one block per CU, and blocks that are already resident may spin while later blocks needed by the barrier never become resident. This produced a pytest timeout in the `normal` metadata test.
- A reduced metadata barrier over only `blockIdx.x == 0` blocks is still unsafe because the scheduler may resident non-metadata `x > 0` blocks for earlier row tiles before all later metadata blocks have launched.
- The accepted correctness-first normal metadata path avoids cross-block metadata barriers entirely. Each `blockIdx.x == 0` row tile owns one local expert, clears only that expert's rows, scans topk routes for that expert, writes row metadata, and then publishes that row tile's row-stage epoch. Other x blocks for the same row tile wait only on that per-row epoch.
- This path stays inside the K1 fused compute kernel and does not add H2D/D2H or a standalone metadata/prebuild kernel.
- The current normal metadata assignment is sequential inside each local-expert owner block. It is accepted for correctness and deadlock safety, but not final for performance. If integrated normal backend overhead is high, optimize this per-expert route scan before measuring final K1 normal degradation.
- K1 V2 package builds do not reliably rebuild `k1_fused_ext.o` when only the included prototype C++ file changes. For remote validation after C kernel-body edits, delete the V2 K1 extension object/so or otherwise force a rebuild before trusting pytest.
- Remote validation after the accepted K1 normal metadata fix:
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - Forced V2 K1 extension rebuild through `setup.py build_ext --inplace` passed.
  - `test_v2_k1_writes_route_metadata_from_sym_buffer[ll-64]` passed.
  - `test_v2_k1_writes_route_metadata_from_sym_buffer[normal-256]` passed.
  - Full `tests/test_dcu_megamoe_v2.py` passed with `19 passed in 14.21s`.

## 2026-06-03 Runtime K1 Connection Finding
- `run_stages_fused_v2` now calls the V2 K1 extension wrapper instead of stopping at the metadata/workspace stub.
- Runtime K1 inputs are all V2-owned scratch views sliced from `sym_buffer.route_scratch`; the call does not allocate new tensors and does not add H2D/D2H.
- The runtime initializes `grid_barrier` once when a new cached V2 state is created. This is setup for the K1 ll grid barrier counter, not a repeated per-iteration H2D control path.
- The runtime maintains a per-state epoch and passes it to K1 normal. Any repeated execution using the same sym buffer must use monotonically increasing epochs; do not hardcode epoch 1 in the real-flow loop.
- `problem_size` is currently passed through as a scratch view but not filled by runtime. K1 ll with symm-stage uses in-kernel route counts rather than `actual_m`, and K1 normal uses topk-driven row metadata. K3 ll may still need an actual problem-size contract later.
- The new runtime test uses a fake K1 launcher to validate parameter wiring without launching a device kernel. GPU metadata correctness is still covered by the dedicated K1 ll/normal metadata tests.
- Remote validation after K1 runtime connection: full `tests/test_dcu_megamoe_v2.py` passed with `19 passed in 14.24s`.

## 2026-06-03 Runtime K2 Connection Finding
- Runtime now calls the V2 K2 wrapper after K1. The wrapper still reuses the existing optimized DCU K2 extension through an isolated V2 boundary.
- K2 consumes dense padded rows: `launch_rows = local_experts * rows_aligned_per_expert`. It relies on `row_combine_ptrs[row] == 0` to skip inactive rows when `output_bf16=False`. This avoids adding a compaction kernel during integration.
- K2 writes `act_fp8` and `act_scale` views from `sym_buffer.route_scratch`; no new activation tensor allocation is added.
- Runtime passes `state.empty_bf16` with `output_bf16=False`, matching the existing large-opt flow and avoiding a BF16 activation output buffer.
- K2 `num_per_channels` must be intermediate hidden (`act_fp8.shape[1]`), not the final hidden size from L2 weights.
- The runtime K2 test is interface-level with a fake launcher. Numerical K2 correctness remains covered by the dedicated GPU reference test.
- Remote validation after K2 runtime connection: full `tests/test_dcu_megamoe_v2.py` passed with `19 passed in 13.25s`.

## 2026-06-03 Runtime K3 Connection Finding
- Runtime now calls K3 after K2, so the eager V2 function has K1/K2/K3 launcher wiring for both forced backends.
- Backend mapping remains explicit:
  - `ll`: K1 low-latency C pack5, K2 optimized DCU wrapper, K3 low-latency C pack5 rowptr tail-reduce.
  - `normal`: K1 normal C pack5 row-stage, K2 optimized DCU wrapper, K3 normal C pack5 copy-stage tail-reduce.
- K3 normal must use a different epoch from K1 normal. K1 normal leaves row-stage counter/epoch data in the shared `grid_barrier` allocation, and K3 copy-stage uses the same allocation for ready flags. Reusing the epoch can let copy workers observe stale ready flags.
- K3 ll `problem_size` is currently `route_scratch_i32[:local_experts]`, populated by K1 ll as in-kernel per-expert route counts. The standalone `problem_size` scratch view remains unused for the connected ll runtime path.
- K3 runtime currently uses dense `tail_tokens[:num_max_tokens_per_rank]`; this is correctness-first and avoids an extra active-token compaction kernel.
- The runtime K3 test is interface-level with fake launchers. It verifies scratch-view and epoch wiring, not numerical end-to-end correctness.
- Remote validation after K3 runtime connection: full `tests/test_dcu_megamoe_v2.py` passed with `19 passed in 13.08s`.

## 2026-06-03 Workspace Capacity And Runtime Smoke Finding
- Forced `MEGAMOE_DCU_V2_BACKEND=normal` can require many more padded rows than the actual route count at small token sizes. For 1 rank, 32 tokens, topk=1, and 32 local experts, normal needs `rows_aligned_per_expert=256` and `launch_rows=8192`.
- The existing route layout's BF16 L1 region is sized as `capacity_rows * intermediate_hidden * sizeof(bf16)`, while V2 L1 output shape is `[rows, 2 * intermediate_hidden]`. Therefore `l1_capacity_rows = capacity_rows / 2`, and V2 must request `capacity_rows >= 2 * launch_rows`.
- V2 `_v2_route_scratch_min_bytes` is now backend-aware. With `backend=None`, it reserves for the maximum of ll and normal so tests/helpers do not accidentally size only for low-latency.
- For normal 32-token one-rank smoke, V2 scratch min bytes is `170877136`.
- A real one-rank runtime smoke with zero weights now completes K1/K2/K3 for both backends:
  - `ll`: finite output, `max=0.0`.
  - `normal`: finite output, `max=0.0`.
- This smoke proves launcher wiring, shape contracts, scratch sizing, and K1/K3 epoch separation at a minimal level. It is not baseline correctness and does not satisfy the 4/8-rank communication acceptance requirement.
- The existing `SymmBuffer.route_scratch` allocation is still produced by the baseline `_C.get_mega_moe_route_scratch_size_for_mega_moe` path. Before real-flow forced-normal small-token tests use a production `SymmBuffer`, confirm whether that allocation is large enough or add minimal V2-specific allocation glue without changing baseline execution behavior.

## 2026-06-03 Real-Flow Correctness Findings
- V2 K1 ll/normal now accepts a uniform runtime `num_tokens` kernel argument. This avoids adding a repeated Python-side H2D write to the sym-buffer runtime token fields for the current uniform-token integration tests. The fallback to `sections.num_tokens[0]` remains for standalone/prototype paths.
- Uneven per-rank token support still needs a separate per-rank token contract. The uniform runtime `num_tokens` argument is correct only for equal token counts across ranks.
- DeepEP baseline dispatch does not support 1-rank validation in this environment. The V2 real-flow test is env-gated and now defaults to 4 ranks; communication/correctness acceptance should use 4 or 8 ranks.
- The real-flow harness disables baseline `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE` inside spawned V2 test workers. Otherwise `SymmBuffer` may try to prepare the existing large-opt 3-stage path and reject non-EP8 shapes such as 4-rank local-expert tests.
- The current real-flow harness uses local-only routing as the first integration target. This validates V2 pack5 layout, real sym-buffer input copy, fused K1/K2/K3 launch wiring, and DeepEP+DeepGEMM output comparison, but it is not full cross-rank combine-reduce acceptance.
- Passing local-only 4-rank correctness against unchanged DeepEP+DeepGEMM baseline:
  - backend `ll`, 32 tokens: `max_abs=0.00049591064453125`, `mean_abs=3.37823512381874e-05`, `mismatch=0`.
  - backend `normal`, 32 tokens: `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`.
  - backend `ll`, 512 tokens: `max_abs=0.000698089599609375`, `mean_abs=2.4401795599260367e-05`, `mismatch=0`.
  - backend `normal`, 512 tokens: `max_abs=0.00075531005859375`, `mean_abs=6.157375173643231e-05`, `mismatch=0`.
  - backend `ll`, 1024 tokens: `max_abs=0.0006885528564453125`, `mean_abs=2.3948752641445026e-05`, `mismatch=0`.
  - backend `ll`, 2050 tokens: `max_abs=0.00069427490234375`, `mean_abs=2.0935844077030197e-05`, `mismatch=0`.
- The earlier normal backend local-only 4-rank 1024-token failure was traced to the K1 normal C pack5 LDS read overlap path, not to K3 normal copy-stage or Python metadata:
  - `normal_no_summary` and `normal_sync_after_v2` probes showed the failure was not explained by the metadata D2H summary path alone.
  - `k1_ll_k3_normal` passed while `k1_normal_k3_ll` failed before the fix, narrowing the root to K1 normal.
  - Failure diagnostics showed correct `src_rank/src_token`, `route_weight`, row pointer, and scale metadata, but sporadic oversized `l1_absmax` on specific K1 expert rows.
  - Serializing normal metadata clearing and using `__threadfence_system()` did not eliminate the failure.
  - Changing the normal C kernel's 16xB128 LDS read helper from `s_waitcnt lgkmcnt(6)` to `s_waitcnt lgkmcnt(0)` eliminated the observed nondeterminism in the tested real-flow path.
- Post-fix K1 normal isolation:
  - `k1_normal_k3_ll`, 4 ranks, 1024 tokens, 5 consecutive runs all passed.
  - Per-rank max_abs stayed around `1.9e-6` to `2.0e-6`, mean_abs below `1e-9`, and mismatch was `0`.
- Post-fix formal real-flow correctness against the unchanged DeepEP+DeepGEMM baseline:
  - backend `normal`, 1024 tokens, 3 consecutive 4-rank runs: `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`.
  - backend `normal`, 2050 tokens: `max_abs=0.000812530517578125`, `mean_abs=6.164611841086298e-05`, `mismatch=0`.
  - full requested 4-rank local-only sweep passed for both `ll` and `normal` at tokens `32,512,1024,2050`.
- First integrated quick performance harness is env-gated with `MEGAMOE_DCU_V2_REAL_FLOW_PERF=1`. It measures only the V2 staged call and the unchanged DeepEP+DeepGEMM baseline call with HIP events; layout transforms, input construction, scratch allocation, and V2 scratch expansion are outside the timed regions.
- First 4-rank local-only integrated quick performance, warmup=3/repeat=5, rank-max median:
  - `ll`, 32 tokens: V2 `0.6668800115585327 ms`, baseline e2e `2.296478033065796 ms`, `max_abs=0.00049591064453125`, `mismatch=0`.
  - `ll`, 128 tokens: V2 `0.6860790252685547 ms`, baseline e2e `2.299837112426758 ms`, `max_abs=0.000751495361328125`, `mismatch=0`.
  - `ll`, 1024 tokens: V2 `2.676637887954712 ms`, baseline e2e `2.8078370094299316 ms`, `max_abs=0.0006618499755859375`, `mismatch=0`.
  - `ll`, 4096 tokens: V2 `11.191666603088379 ms`, baseline e2e `6.714233875274658 ms`, `max_abs=0.0006885528564453125`, `mismatch=0`.
  - `normal`, 32 tokens: V2 `17.107980728149414 ms`, baseline e2e `2.515038013458252 ms`, `max_abs=0.0006389617919921875`, `mismatch=0`.
  - `normal`, 128 tokens: V2 `20.457258224487305 ms`, baseline e2e `2.5651180744171143 ms`, `max_abs=0.00074005126953125`, `mismatch=0`.
  - `normal`, 1024 tokens: V2 `81.61079406738281 ms`, baseline e2e `2.864957094192505 ms`, `max_abs=0.00083160400390625`, `mismatch=0`.
  - `normal`, 4096 tokens: V2 `1138.2191162109375 ms`, baseline e2e `12.349268913269043 ms`, `max_abs=0.000835418701171875`, `mismatch=0`.
- The `normal` backend integrated timing is clearly pathological and must not be used to choose a backend threshold yet. The next evidence needed is stage timing for K1 normal, K2, and K3 normal. The most likely source is the correctness-first K1 normal metadata/row-stage path or another dense padded-row integration overhead, because the standalone prototype normal kernels previously ran in the millisecond range.
- Stage breakdown for the pathological `normal` backend confirmed K1 dominates:
  - 1024 tokens, warmup=1/repeat=3: total V2 `81.6631088256836 ms`; K1 `80.19960021972656 ms`, K2 `0.14032000303268433 ms`, K3 `1.0281590223312378 ms`.
  - 4096 tokens, warmup=1/repeat=3: total V2 `1129.4111328125 ms`; K1 `1127.633056640625 ms`, K2 `0.38784000277519226 ms`, K3 `3.551356077194214 ms`.
- K1 normal performance root cause in the real-flow integration is the correctness-first metadata build inside the normal C pack5 K1 kernel:
  - the first version used `threadIdx.x == 0` to scan all routes for each local expert, which serialized `num_ranks * num_tokens * topk` route checks per expert;
  - for `rows_aligned_per_expert > 256`, multiple row tiles of the same expert repeated the same metadata build, further multiplying the overhead;
  - this is an integration metadata path issue, not a K2 or K3 bottleneck.
- The accepted K1 normal metadata optimization changes K1 normal metadata to:
  - clear rows in parallel within the metadata owner block;
  - scan routes in parallel within the block and use `atomicAdd` to allocate expert rows;
  - build metadata only from the first row tile of each expert;
  - publish the row-stage epoch for all row tiles belonging to that expert.
- The K1 normal metadata optimization is accepted after 4-rank real-flow validation:
  - backend `normal`, 1024 tokens correctness passed with `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`;
  - backend `normal`, requested correctness tokens `32,512,1024,2050` passed with `mismatch=0` and `max_abs <= 0.00083160400390625`;
  - K1 normal stage time dropped from about `80.20 ms` to `1.35 ms` at 1024 tokens and from about `1127.63 ms` to `7.16 ms` at 4096 tokens.
- Updated integrated quick performance after the K1 normal optimization, 4 ranks local-only, warmup=3/repeat=5:
  - `ll`, 32 tokens: V2 `0.6806390285491943 ms`, baseline e2e `2.493597984313965 ms`, `max_abs=0.00049591064453125`, `mismatch=0`.
  - `ll`, 128 tokens: V2 `0.7094389796257019 ms`, baseline e2e `2.504957914352417 ms`, `max_abs=0.000690460205078125`, `mismatch=0`.
  - `ll`, 1024 tokens: V2 `2.6755170822143555 ms`, baseline e2e `2.826237916946411 ms`, `max_abs=0.000804901123046875`, `mismatch=0`.
  - `ll`, 4096 tokens: V2 `11.154061317443848 ms`, baseline e2e `6.653112888336182 ms`, `max_abs=0.000698089599609375`, `mismatch=0`.
  - `normal`, 32 tokens: V2 `1.6804779767990112 ms`, baseline e2e `2.457437038421631 ms`, `max_abs=0.0006389617919921875`, `mismatch=0`.
  - `normal`, 128 tokens: V2 `1.7204780578613281 ms`, baseline e2e `2.566396951675415 ms`, `max_abs=0.00074005126953125`, `mismatch=0`.
  - `normal`, 1024 tokens: V2 `2.4735970497131348 ms`, baseline e2e `2.838076114654541 ms`, `max_abs=0.00083160400390625`, `mismatch=0`.
  - `normal`, 4096 tokens: V2 `10.875988960266113 ms`, baseline e2e `6.751031875610352 ms`, `max_abs=0.000835418701171875`, `mismatch=0`.
- The normal backend is no longer pathological after the owner-block metadata optimization. It is still not ready for automatic threshold selection because:
  - the integrated numbers are compared to baseline e2e, not pure V2 prototype K1/K3 groupgemm timings;
  - 4096 remains slower than baseline e2e, with the post-optimization stage breakdown showing K1 `7.16 ms` and K3 `3.55 ms`;
  - full backend sweep and prototype-kernel comparison are still required before choosing a size threshold.
- Prototype-vs-integrated timing comparison is now collected for forced `ll` and `normal` backends at quick sizes 32/128/1024/4096:
  - K1 prototype fused communication is within the <=20% target for all measured forced-backend quick sizes: `ll` degradation is `+16.06%`, `+16.55%`, `+10.91%`, `+18.37%`; `normal` degradation is `+5.07%`, `+5.55%`, `+7.54%`, `+12.69%`.
  - K3 prototype `ll` fused tail-reduce is within the <=25% target at all quick sizes: `+5.63%`, `+10.05%`, `+18.03%`, `+18.43%`.
  - K3 prototype `normal` copy-stage tail-reduce is still above target: `+30.31%`, `+33.91%`, `+51.24%`, `+56.21%`.
  - Integrated K1 `ll` tracks the standalone fused prototype reasonably: overhead `+15.63%`, `+12.10%`, `+2.21%`, `-2.52%`.
  - Integrated K1 `normal` still has large real-flow overhead at larger sizes, especially 4096: overhead `+31.27%`, `+29.75%`, `+43.16%`, `+166.97%`.
  - Integrated K3 has substantial overhead versus standalone fused prototype for both backends: K3 `ll` overhead is `+52.75%`, `+53.97%`, `+44.34%`, `+67.32%`; K3 `normal` overhead is `+27.84%`, `+20.32%`, `+45.41%`, `+69.50%`.
- The current integration performance conclusion is therefore:
  - K1 prototype fusion itself is not the blocking issue;
  - K3 normal prototype fusion remains above target and needs kernel-level work later;
  - real-flow K3 overhead and normal K1 4096 metadata overhead are the main integration overhead gaps before threshold selection.
- V2 K1 currently rejects `cumulative_local_expert_recv_stats`. The real-flow test compares output only; stats alignment remains a separate integration item.

## 2026-06-03 V2 K1 Cumulative Stats Alignment Finding
- V2 K1 ll/normal now supports `cumulative_local_expert_recv_stats` inside the fused K1 compute kernel. The kernel does not clear stats; it `atomicAdd`s the finalized per-local-expert route counts, matching the existing MegaMoE cumulative semantics.
- The default performance path still passes an empty tensor that maps to a null device pointer, so the stats path is disabled when the caller passes `None`.
- No standalone stats/prebuild kernel, repeated H2D, D2H, or external rank barrier was added. The stats write is piggybacked on K1 metadata/counts already produced inside the fused kernel.
- Fast validation:
  - Single-HCU K1 metadata/stats test passed for both `ll` and `normal`: `2 passed in 13.26s`.
  - Full non-distributed V2 test suite passed: `20 passed, 2 skipped in 13.46s`.
- 4-rank local-only real-flow validation on `HIP_VISIBLE_DEVICES=1,2,3,4` passed for both backends at requested tokens `32,512,1024,2050`; all rows had `stats_ok=True`, `mismatch=0`, and `max_abs <= 0.00083160400390625`.
- Per-token correctness from the stats-enabled real-flow run:
  - `ll`, 32: `max_abs=0.00049591064453125`, `mean_abs=3.37823512381874e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 512: `max_abs=0.0006957054138183594`, `mean_abs=2.5930632546078414e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 1024: `max_abs=0.0007266998291015625`, `mean_abs=2.3697797587374225e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 2050: `max_abs=0.0006694793701171875`, `mean_abs=2.3335776859312318e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 32: `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 512: `max_abs=0.00075531005859375`, `mean_abs=6.150374247226864e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 1024: `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 2050: `max_abs=0.000812530517578125`, `mean_abs=6.164611841086298e-05`, `mismatch=0`, `stats_ok=True`.
- This closes the real-flow stats alignment gap. Stage-level K1/K2/K3 numerical references remain a separate pending Phase 8 item.

## 2026-06-03 V2 Real-Flow Stage Reference Finding
- Real-flow stage metrics are now collected automatically for the 32-token correctness case. Larger requested correctness tokens keep only end-to-end and stats checks to avoid making stage-reference construction dominate validation.
- Reliable stage references in the current harness:
  - K1 metadata/counts/stats are checked through the fused K1 output metadata and cumulative stats.
  - K2 is checked by running the same optimized K2 wrapper in BF16-output mode and comparing against the Torch SwiGLU reference on active rows.
  - K3 is checked by running baseline DeepGEMM L2 on V2 K2 activations in V2 row order, then reducing through V2 `output_index` and comparing to final V2 `y`.
- 4-rank local-only 32-token stage metrics:
  - `ll`: K2 `max_abs=9.5367431640625e-07`, `mismatch=0`; K3 `max_abs=0.000553131103515625`, `mismatch=0`.
  - `normal`: K2 `max_abs=0.0`, `mismatch=0`; K3 `max_abs=0.0006389617919921875`, `mismatch=0`.
- Direct K1 L1 comparison against baseline DeepGEMM in V2 row order is diagnostic-only for now:
  - `ll` diagnostic K1 L1 max_abs was about `0.168`, with many element mismatches.
  - `normal` diagnostic K1 L1 max_abs was about `0.123`, with many element mismatches.
  - Because K2/K3 stage references and end-to-end output still pass, this diagnostic is not accepted as a reliable K1 numerical denominator. The likely issue is a reference construction/layout/row-order mismatch that needs a narrower K1-only investigation before it can be used as an acceptance gate.
- Updated full 4-rank local-only correctness with stats and stage metrics passed for both `ll` and `normal` at tokens `32,512,1024,2050`; 32-token prints include stage metrics, larger tokens keep `stage_metrics={}` and `stage_ok=True`.

## 2026-06-03 V2 Real-Flow Full Sweep Finding
- The Phase 9 full token sweep completed successfully with exit code 0:
  - command log: `hygon_tmp/dcu_megamoe_v2/real_flow_full_sweep_20260603_141843.log`
  - ranks: 4
  - route mode: `local_only`
  - warmup/repeat: 3/5
  - backends: `ll`, `normal`
  - tokens: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
- Correctness passed for every row in the sweep:
  - all `mismatch=0`
  - worst `ll` max_abs: `0.0007801055908203125` at 8192 tokens
  - worst `normal` max_abs: `0.00087738037109375` at 8192 tokens
  - all results are under the `max_abs <= 1e-3` target.
- Integrated V2 staged rank-max median times vs unchanged baseline e2e:
  - `ll`, 32: V2 `0.690559 ms`, baseline `2.498238 ms`, degradation `-72.36%`
  - `ll`, 64: V2 `0.699679 ms`, baseline `2.563997 ms`, degradation `-72.71%`
  - `ll`, 128: V2 `0.716319 ms`, baseline `2.514237 ms`, degradation `-71.51%`
  - `ll`, 256: V2 `0.940319 ms`, baseline `2.638237 ms`, degradation `-64.36%`
  - `ll`, 512: V2 `1.483838 ms`, baseline `2.654877 ms`, degradation `-44.11%`
  - `ll`, 1024: V2 `2.720157 ms`, baseline `2.848477 ms`, degradation `-4.50%`
  - `ll`, 2048: V2 `5.418393 ms`, baseline `4.291355 ms`, degradation `+26.26%`
  - `ll`, 4096: V2 `11.249908 ms`, baseline `6.592633 ms`, degradation `+70.64%`
  - `ll`, 8192: V2 `22.648611 ms`, baseline `12.176468 ms`, degradation `+86.00%`
  - `normal`, 32: V2 `1.690878 ms`, baseline `2.517437 ms`, degradation `-32.83%`
  - `normal`, 64: V2 `1.699518 ms`, baseline `2.546078 ms`, degradation `-33.25%`
  - `normal`, 128: V2 `1.727198 ms`, baseline `2.496638 ms`, degradation `-30.82%`
  - `normal`, 256: V2 `1.773597 ms`, baseline `2.666718 ms`, degradation `-33.49%`
  - `normal`, 512: V2 `2.121918 ms`, baseline `2.631037 ms`, degradation `-19.35%`
  - `normal`, 1024: V2 `2.476957 ms`, baseline `2.819997 ms`, degradation `-12.16%`
  - `normal`, 2048: V2 `4.995835 ms`, baseline `4.257914 ms`, degradation `+17.33%`
  - `normal`, 4096: V2 `10.866069 ms`, baseline `6.748951 ms`, degradation `+61.00%`
  - `normal`, 8192: V2 `21.458380 ms`, baseline `11.821106 ms`, degradation `+81.53%`
- Backend advantage interval from this local-only integrated sweep:
  - `ll` is faster from 32 through 512 tokens.
  - `normal` is faster from 1024 through 8192 tokens.
  - The provisional crossover is therefore between 512 and 1024 tokens.
- Do not hard-code the threshold yet:
  - this sweep is local-only routing and not the final uneven/full cross-rank combine-reduce acceptance;
  - K3 large performance remains above the target in both integrated large-token paths;
  - final threshold selection should be made after cross-rank routing, uneven token support, and the K3 large caveat are revisited.

## 2026-06-03 V2 Uneven Token Local-Only Finding
- V2 eager runtime now supports `dispatch_num_tokens=-1` as a sentinel for K1 to read runtime token counts from each rank's sym-buffer header instead of forcing a uniform token count.
- The implementation does not add a new device tensor or repeated per-iteration H2D array for per-rank counts. It reuses the existing sym-buffer `cuda_graph_num_tokens` scalar on each rank as the local runtime token count.
- K1 pybind validation now accepts `num_tokens=-1`; the C kernels already treat negative `runtime_num_tokens` as "read `sections.num_tokens[0]`".
- K3 still receives the local output token upper bound and currently keeps dense tail-token scanning up to `num_max_tokens_per_rank`. This is correctness-first and may do extra work for uneven ranks; compact active-token tail reduction remains a later optimization/cleanup item.
- Remote validation:
  - prior to running, container process scan found no residual V2 pytest/real_flow processes.
  - unrelated sglang/test_low_latency processes were present, so validation avoided occupied-memory cards and used `HIP_VISIBLE_DEVICES=0,1,2,7`.
  - 4-rank correctness passed with uneven token counts `[32,512,1024,2050]` for both `ll` and `normal`.
  - printed rank-0 metrics:
    - `ll`: local tokens `32`, max_tokens `2050`, max_abs `0.00049591064453125`, mean_abs `1.7092428606702015e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
    - `normal`: local tokens `32`, max_tokens `2050`, max_abs `0.0006389617919921875`, mean_abs `6.178580224514008e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
- This is an uneven-token eager smoke against the real baseline, but still `local_only` routing. It does not close the full cross-rank remote communication/combine acceptance requirement.

## 2026-06-03 Integrated Stage-vs-Prototype Requirement Finding
- The user clarified that after K1/K3 are connected to the real communication-enabled MegaMoE V2 flow, the integrated K1 and K3 stage timings must be compared against the corresponding standalone V2 prototype fused kernel timings.
- If integrated K1 or K3 is much slower than the prototype fused kernel, the gap must be explained and repaired as much as possible before performance acceptance.
- The stage-vs-prototype comparison is now a diagnostic guardrail, not the final hard acceptance rule. A large gap should be treated as theoretically fixable. If a checkpoint still has a large gap, V2 can proceed only if the root cause is documented with profile evidence, the same-size V2 path beats DCU MegaMoE V1, and a concrete follow-up repair item remains open.
- The hard performance gate is V2 versus current DCU MegaMoE V1 at the same rank count, token count, route mode, warmup/repeat policy, and timing harness. Cross-rank correctness alone is not enough, and any V2-vs-V1 regression must be fixed before performance acceptance.
- Likely attribution buckets for this comparison:
  - K1 metadata generation and route scanning;
  - K1/K3 synchronization and in-kernel barriers;
  - remote staging and row pointer setup;
  - dense padded rows or dense tail-token scans;
  - Python/runtime glue and shape/scratch adaptation.
- This requirement is separate from comparing V2 e2e against the unchanged DeepEP+DeepGEMM reference baseline. For final performance acceptance, compare against the current DCU MegaMoE V1 implementation, not only the reference baseline.

## 2026-06-03 V2 Cross-Rank K1 Metadata Visibility Finding
- Cross-rank route validation exposed a real K1 race: K1 could scan peer `topk_idx/topk_weights` from remote sym-buffer sections before every rank had finished publishing local inputs and entered the fused K1 kernel.
- The symptom before the fix was `stats_ok=False` with undercounted K1 metadata in cross-rank route mode, while local-only routing still passed.
- The accepted fix keeps synchronization inside the fused K1 kernels:
  - K1 `ll` runs `mega_moe_rank_barrier` from block 0 and then uses the existing grid barrier before peer route scans.
  - K1 `normal` runs `mega_moe_rank_barrier` from the expert0 metadata owner block and wakes other metadata owner blocks with `grid_barrier[0] = launch_epoch`.
- A counted local barrier across K1 normal metadata owner blocks is rejected. It hung at 32-token cross-rank validation because the assumed metadata block count can exceed the number of metadata owner blocks actually launched for the current shape.
- The epoch-flag approach is correctness-accepted for 4-rank cross-rank route mode:
  - `ll` and `normal` pass tokens 32, 512, 1024, and 2050 with `mismatch=0`, `stats_ok=True`, and max_abs below `1e-3`.
- This is a correctness checkpoint only. Cross-rank performance, same-size V1 comparison, and integrated K1/K3 stage-vs-prototype diagnostic analysis remain required before performance acceptance.

## 2026-06-03 V2 Distributed Test Port Finding
- The real-flow distributed test previously used a random `MASTER_PORT` in a fixed range. A repeated cross-rank run hit `EADDRINUSE` after the `ll` token sweep passed and before the `normal` sweep started.
- The test now reserves an available local TCP port with `socket.bind(("127.0.0.1", 0))` for each spawned distributed run.
- This is a test-harness robustness fix only. It does not affect the MegaMoE V2 execution path or kernel timing.

## 2026-06-03 V2 Cross-Rank Performance Finding
- The first cross-rank integrated performance run is partial because the 8-row run timed out at 1200s after completing all `ll` rows and `normal` 32.
- Collected 4-rank cross-rank stage timings show that correctness-clean communication is not yet performance-accepted:
  - `ll`, 32: K1 `0.473280 ms`, K3 `0.262720 ms`.
  - `ll`, 128: K1 `0.494240 ms`, K3 `0.335199 ms`.
  - `ll`, 1024: K1 `1.708638 ms`, K3 `1.871837 ms`.
  - `ll`, 4096: K1 `6.556473 ms`, K3 `7.856471 ms`.
  - `normal`, 32: K1 `1.050879 ms`, K3 `0.646559 ms`.
- Compared to the recorded standalone V2 prototype fused kernels:
  - K1 `ll` overhead is about `+35.35%`, `+37.41%`, `+21.65%`, and `+11.20%` at 32/128/1024/4096.
  - K3 `ll` overhead is about `+57.95%`, `+83.90%`, `+127.67%`, and `+143.57%` at 32/128/1024/4096.
  - `normal` 32 overhead is about `+39.76%` for K1 and `+26.96%` for K3.
- This confirms the user's requested guardrail is needed: comparing V2 e2e against baseline is not enough. Integrated K3b is materially slower than the prototype fused K3 path under cross-rank routing.
- The next kernel optimization target should be K3b cross-rank combine/tail-reduce:
  - identify whether the overhead comes from remote combine stores, dense tail-token scans, missing overlap with L2 groupgemm, or synchronization placement;
  - then rerun cross-rank performance for 32/128/1024/4096 and complete missing `normal` rows.
- After the timeout, no V2 residual process was found, but unrelated workloads occupied most devices. Continue using pre-run process/card checks and avoid killing unrelated tasks.

## 2026-06-03 V2 K3 Normal Tail Mask Caveat
- K3 `ll` rowptr tail-reduce currently scans all topk slots for each local output token.
- K3 `normal` copy-stage tail-reduce uses the K1-produced `local_topk_mask`, which marks only topk slots whose expert belongs to the current rank.
- That mask is equivalent to the valid-slot mask in local-only routing, but it is not equivalent under cross-rank routing. The final combine for a local token should include remote expert slots after those ranks store into this rank's combine buffer.
- The current cross-rank E2E correctness uses small MegaMoE-like values and `max_abs <= 1e-3`; this may not be strong enough to expose a missing-slot normal K3 reduce bug.
- Do not treat normal K3 cross-rank combine as final semantic acceptance until:
  - K3 normal tail-reduce mask is changed to a valid topk-slot mask for local source tokens, or an equivalent all-slot reduce is used;
  - 4-rank correctness is rerun at 32/512/1024/2050;
  - and stage/prototype performance is remeasured.
- Follow-up implementation direction: change K1-produced `local_topk_mask` from local-expert-only bits to valid global-topk-slot bits (`expert >= 0`, `expert < num_global_experts`, and nonzero route weight). This preserves the K3 normal same-kernel reduce interface while making cross-rank final reduce include remote expert slots stored into this rank's combine buffer.
- The valid-global-topk-slot mask is now correctness-accepted for 4-rank cross-rank real-flow validation on HCU2-HCU5:
  - backend `normal`, tokens `32/512/1024/2050`: max_abs up to `0.0006942749`, mean_abs up to `9.4935e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - backend `ll`, tokens `32/512/1024/2050`: max_abs up to `0.0005726814`, mean_abs up to `6.2686e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
- This closes the specific K3 normal cross-rank tail-mask semantic caveat. It does not close the open performance work: missing `normal` cross-rank perf rows, same-size DCU MegaMoE V1 timing, stage-vs-prototype overhead profiling, and later 8-rank data remain pending.
- Current optimization priority is stage-vs-prototype gap repair, not V1 timing. For the next tuning loop, compare real-flow integrated K1/K3 stage timings against standalone V2 prototype fused K1/K3 kernels first, then optimize the largest gap. Same-size V1 timing remains a later final acceptance gate.

## 2026-06-03 K1 Normal Cooperative Metadata Experiment Rejection
- A cooperative K1 normal metadata scan variant was briefly drafted to let multiple x-blocks share route scanning for each expert.
- The draft used per-row-stage counters/epochs (`metadata_worker_block`, `metadata_reset_epoch`, `metadata_scan_epoch`) and therefore changed synchronization behavior in the C pack5 prototype bridge.
- It was not rebuilt, correctness-tested, or profiled, and it touched `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`, which is now supposed to be prototype/reference material rather than the long-term production implementation surface.
- The experiment was reverted back to the last remote correctness-accepted `metadata_owner_block` implementation.
- Future attempts to reduce K1 normal metadata overhead should be introduced through a stage-owned V2 source path or a narrowly named experimental branch, with a micro gate that checks:
  - 4-rank correctness at 32/512/1024/2050;
  - no hang under `normal` at tokens above 32;
  - integrated K1 stage timing vs standalone K1 fused prototype;
  - no full-grid barrier or counted barrier whose participant count can exceed launched blocks.

## 2026-06-03 K3 Tail Reduce All-Rank Synchronization Finding
- Cross-rank K3 final reduce has a semantic synchronization requirement beyond local grid barriers:
  - K3 expert-owner ranks remote-store partial L2 results into each source rank's combine buffer;
  - the source rank then reads that local combine buffer during tail reduce;
  - therefore tail reduce must not start until peer ranks have completed the relevant remote stores.
- The current K3 fused paths previously had only local synchronization before tail reduce:
  - `ll`: rowptr direct stores followed by a local `v2_device_grid_barrier`;
  - `normal`: copy-stage workers waited for local copy workers but not for peer ranks.
- A candidate fix now adds an in-kernel `mega_moe_rank_barrier` before K3 tail reduce on both paths. This preserves the requirement that communication synchronization stays inside the fused compute kernel and does not add a standalone rank barrier or extra kernel.
- Validation state:
  - source contract pytest and forced remote extension rebuild passed;
  - no 4-rank or 8-rank correctness result has accepted this candidate yet;
  - a temporary 2-rank smoke timed out and is inconclusive, and the remote SSH session became unreachable afterward.
- This candidate must be treated as unaccepted until:
  - residual temporary smoke processes are cleared when the remote is reachable;
  - 4-rank `ll` token 32 cross-rank correctness passes without hang;
  - 4-rank `normal` token 32 cross-rank correctness passes without hang;
  - then 32/512/1024/2050 correctness and 32/128/1024/4096 stage-vs-prototype timing are rerun.
- If 4-rank hangs, do not proceed with performance tuning on this version. Redesign the K3 in-kernel all-rank synchronization or revert the candidate.

## 2026-06-03 K3 Performance Gap Ranking Finding
- Recomputed the recorded 4-rank cross-rank integrated-vs-prototype stage gaps after the user asked why performance-gap investigation was not continuing.
- The largest current gaps are K3, not K1:
  - `ll`, 4096 K3: `+143.60%`.
  - `ll`, 1024 K3: `+127.67%`.
  - `ll`, 128 K3: `+83.90%`.
  - `ll`, 32 K3: `+57.96%`.
  - `normal`, 32 K3: `+26.96%`.
- K1 `ll` is comparatively less problematic at large size:
  - `ll`, 4096 K1 gap is about `+11.21%`.
  - `ll`, 1024 K1 gap is about `+21.64%`.
- Previous local-only 4-rank integrated data already showed K3 overhead before cross-rank route mode:
  - K3 `ll` local-only overhead vs standalone fused prototype: `+52.75%`, `+53.97%`, `+44.34%`, `+67.32%` at 32/128/1024/4096.
  - K3 `normal` local-only overhead vs standalone fused prototype: `+27.84%`, `+20.32%`, `+45.41%`, `+69.50%` at 32/128/1024/4096.
- Conclusion:
  - K3 integrated overhead is not explained solely by cross-rank communication.
  - The likely buckets are dense tail-token scanning, row-output pointer locality/scatter, real-flow row layout versus standalone harness layout, and copy-stage/tail-reduce synchronization.
  - Next remote performance debug should compare standalone K3 fused, integrated local-only K3, and integrated cross-rank K3 under the same visible 4-rank device set before touching K1 again.
- The unaccepted K3 all-rank barrier candidate was reverted from active source after an inconclusive 2-rank timeout, so it does not block this performance-gap investigation. The all-rank synchronization semantic issue remains open for a redesigned candidate after K3 gap attribution is clearer.

## Resources
- .vscode/sftp.json: remote host 10.17.176.13, user hg, remote path /home/hg/yuguo/DeepGEMM.
- Remote execution template: ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa hg@10.17.176.13 "docker exec megamoe bash -lc 'source /opt/dtk/env.sh && cd /workspace/DeepGEMM && <cmd>'"

## Visual/Browser Findings
- None.

## 2026-06-03 Pure GroupGEMM Denominator Correction
- The active performance denominator is the corresponding V2 pure groupgemm path, not the standalone fused prototype path.
- The user clarified the intent: measure how much the real MegaMoE V2 flow degrades relative to pure groupgemm after communication and real metadata are fused into the K1/K3 compute kernels.
- Do not modify the pure groupgemm prototype/harness merely to make an A/B run easier. Treat it as the stable reference denominator.
- A brief `symm_rank_barrier` prototype-harness experiment was started to make a standalone fused K1 denominator run, but that was the wrong comparison target and was reverted from source.
- Next accepted performance reports should state:
  - backend: `ll` or `normal`;
  - token count and rank count;
  - pure K1 or K3 groupgemm time;
  - integrated K1 or K3 fused stage time;
  - fused-vs-pure degradation ratio;
  - correctness max_abs / mean_abs / mismatch;
  - profile evidence or static attribution for any large gap.
- LL tuning focus remains sizes up to/around 1024 tokens; normal tuning focus remains 1024+ tokens.

## 2026-06-03 Same-Size Pure-vs-Integrated Performance Finding
- The first accepted 4-rank performance diagnostic now compares integrated real-flow K1/K3 fused stage timings against same-size V2 pure groupgemm timings. The pure groupgemm prototype/harness remains unmodified and is treated as the stable denominator.
- Worst current same-size degradation is `normal` K1 at 4096 tokens:
  - pure K1 groupgemm `2.339350 ms`;
  - integrated real-flow K1 fused stage `7.527326 ms`;
  - degradation `+221.77%`;
  - correctness remains clean with max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Local-only versus cross-rank isolation shows this `normal` K1 gap is mostly internal:
  - local-only K1 4096 `7.289430 ms`;
  - cross-rank K1 4096 `7.527326 ms`;
  - cross-rank routing adds only about `0.238 ms`.
- hipprof for `normal` 4096 confirms the K1 stage time is dominated by the named V2 K1 fused kernel, not Python glue or an unexpected standalone setup kernel.
- Static root cause:
  - the current real-flow K1 normal fused path generates route metadata in-kernel, stages every selected activation row into `staged_x` HBM, waits on a row-tile barrier, then the groupgemm reads those staged rows;
  - this adds a large memory pass and synchronization that pure groupgemm does not have;
  - therefore the current path is correctness-clean but not yet the intended fine-grained dispatch-pull/GEMM overlap.
- Next optimization direction:
  - short experiment: adjust K1 normal real-flow launch grouping to increase parallelism in row staging and reduce the row-stage bottleneck, without changing the pure groupgemm denominator;
  - deeper fix: remove or shrink the `staged_x` HBM middleman by loading source rows directly from sym-buffer/source pointers in the GEMM load path so dispatch-pull work is hidden inside the compute pipeline.
- K3 remains a separate large gap:
  - `ll` K3 512 and 1024 degrade by `+182.26%` and `+169.79%`;
  - `normal` K3 4096 degrades by `+155.48%`;
  - local-only probes show K3 has both internal overhead and real cross-rank remote-store/tail-reduce cost.

## 2026-06-03 K1 Normal Launch Grouping Rejection
- A K1 normal real-flow experiment changed only the extension launch grouping from `c_stage_n_group=4` to `2`, leaving the pure groupgemm denominator untouched.
- It was correctness-clean but slower:
  - token `4096`, backend `normal`, 4-rank cross-rank;
  - K1 stage worsened from `7.527326 ms` to `9.135353 ms`;
  - pure K1 denominator remained `2.339350 ms`;
  - degradation worsened from `+221.77%` to `+290.50%`;
  - max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Conclusion:
  - increasing launch x-block parallelism is not a viable fix for the current K1 normal overhead. It likely adds block scheduling and row-stage barrier cost while the full HBM staging pass still remains.
  - Do not retry `c_stage_n_group=2` as a performance fix without a new reason or microbench evidence.
  - The next K1 normal fix should reduce the HBM-staged row path itself or make source-row loads happen directly in the GEMM pipeline.

## 2026-06-03 K1 Normal Direct-Pull Build Finding
- A direct-pull K1 normal experiment is staged but not performance-accepted:
  - the large C pack5 `V2_DeepGemm...` template now has a default-off `kUseDirectSymmLoad` parameter;
  - the real-flow K1 normal extension enables it;
  - pure groupgemm default instantiations remain default-off.
- The experiment removes the large FP8 `staged_x` row copy while keeping metadata and `x_scale` staging. Compute waves resolve each grouped row through `symm_src_ranks/symm_src_tokens` and load FP8 packs directly from the source rank's sym-buffer row.
- Build validation passed:
  - V2 K1 extension rebuild;
  - standalone `make -C csrc/kernels/dcu_megamoe_v2 hipcc aicc`.
- Correctness and timing are still pending because all 8 HCUs became occupied by an unrelated `sglang` workload before the run. Do not treat this experiment as accepted until the 4-rank normal 4096 gate passes.
- Expected risk:
  - direct pull removes the staged write/read middleman but may re-read source activation rows for each N-group. If remote HBM reads dominate, it can be slower than staging.
  - If it regresses, reject it and move to a tiled LDS/source-row overlap design or metadata-scan reduction instead.

## 2026-06-03 Same-Size Denominator Row Expansion Finding
- The V2 standalone pure groupgemm harness expands `--tokens` into grouped rows with the same formula as the real-flow runtime:
  - `valid_rows_per_expert = ceil(tokens * topk / local_experts)`;
  - `rows_aligned_per_expert = align(valid_rows_per_expert, row_tile)`;
  - `launch_rows = local_experts * rows_aligned_per_expert`.
- Therefore same-size pure-vs-integrated comparison must keep `tokens`, `topk`, backend row tile, and local expert count aligned.
- For the current real-flow tests:
  - `num_topk=6`;
  - `local_experts=32`;
  - normal 4096 uses `valid_rows_per_expert=768`, `rows_aligned_per_expert=768`, `launch_rows=24576`.
- The prior pure normal 4096 denominator is valid only if the harness run used the same default or explicit `topk=6`. Future reports should print or record `valid_rows_per_expert`, `rows_aligned_per_expert`, and `launch_rows` with the pure denominator to avoid ambiguity.

## 2026-06-03 K3 Tail-Reduce Runtime Token Count Finding
- K3 tail reduce should use the real runtime token count, not the full `num_max_tokens_per_rank` buffer capacity, whenever the two can differ.
- Before the cleanup, K3 `ll` always reduced `num_max_tokens_per_rank` rows, and K3 `normal` pybind passed `tail_tokens.numel()` as the tail count. In fixed perf runs where the sym-buffer max token count equals the requested token count, this does not explain the large 4096-token gap. It can still add unnecessary work for aligned/padded sizes, uneven tokens per rank, and future CUDA graph replay with a larger reserved token capacity.
- The staged cleanup now propagates `runtime_num_tokens` through the K3 wrapper, pybind layer, raw launcher, and kernel launch. K3 `ll` computes the reduce loop count with `v2_effective_num_tokens(...)`; K3 `normal` accepts a `tail_token_count`, with `-1` meaning derive the count from sym-buffer runtime token metadata in-kernel.
- This is a correctness/performance hygiene fix, not yet an accepted optimization result. Remote rebuild and 4-rank correctness/performance validation remain pending because SSH started closing connections before the sync/build step.

## 2026-06-04 K3 Normal Tail-Reduce Micro-Optimization Findings
- Restoring `K3_COPY_WORKERS=16` is required. The interrupted `K3_COPY_WORKERS=32` state violates the extension launcher contract, which enforces copy workers in `[1,16]`; previous `8`-worker testing was correctness-clean but slower.
- hipprof on 4-rank cross-rank `normal` 4096 showed no standalone K3 combine kernel in the timed path. The remaining K3 gap is inside the fused normal groupgemm/copy-stage/tail-reduce kernel.
- A K3 normal topk=6/all-slot tail-reduce fast path is accepted:
  - `num_topk == 6 && slot_mask == 0x3f` is the common real-flow case for current tests;
  - unrolling those six BF16 packed-row loads and using a shared `accumulate_bf16x8_device(...)` helper reduced K3 normal 4096 from about `3.422396 ms` to `3.084637 ms`;
  - the same patch improved normal K3 1024 from `1.003998 ms` to `0.919839 ms` and 2048 from `1.855198 ms` to `1.687358 ms`;
  - correctness stayed clean with max_abs below `1e-3` and mismatch `0`.
- Hoisting K3 normal combine-buffer address arithmetic into vector indexing is accepted:
  - precompute `combine_vecs`, `token_vec_base`, and `slot_stride_vecs`;
  - access `combine_vecs[token_vec_base + topk_slot * slot_stride_vecs]` instead of recomputing byte offsets through `partial_row * kProblemN`;
  - this further improved normal K3 4096 to `3.030876 ms`, with correctness still clean.
- The K3 normal 4096 same-size pure gap is reduced but still large:
  - pure K3 groupgemm `1.342780 ms`;
  - integrated K3 after the two accepted tail-reduce changes `3.030876 ms`;
  - degradation `+125.72%`.
- Conclusion:
  - scalar loop/address-control cleanup helps, but does not close the gap. The remaining K3 normal overhead likely sits in the fused copy-stage/tail-reduce schedule, memory traffic for six combine rows per output token, and synchronization/work partitioning rather than Python/runtime glue.
  - Continue with small measurable K3 changes first, and rerun 1024/2048 after the combine-vector-index patch before declaring the new K3 champion table.
- Dense identity tail-token specialization is accepted for the current fixed-size normal real-flow contract:
  - K1 writes `tail_tokens[token] = token`, and K3 normal passes `tail_token_count == runtime_num_tokens` for fixed-size runs.
  - K3 can therefore avoid reading `tail_tokens[reduce_token_idx]` once per 16B output vector and can use the linear reduce `task` as the output/combine vector base.
  - On `HIP_VISIBLE_DEVICES=2,3,4,5`, this improved K3 normal from `0.904000 -> 0.886879 ms` at 1024 and `1.661758 -> 1.626238 ms` at 2048; 4096 reached `2.969597 ms` with correctness clean.
  - This is not a permanent compact-tail-token contract. If V2 later replaces dense `tail_tokens[:num_tokens]` with compact active-token output, K3 must keep the non-dense path or expose an explicit identity/compact flag.
- Fixed full-topk6 mask and helper/direct reduce are accepted for the current fixed-size normal perf path:
  - the route generator currently gives six valid topk slots per token for these fixed-size correctness/perf runs, so the full-slot path can avoid per-vector mask reloads;
  - using a direct `reduce_full_topk6_bf16x8_device(...)` helper improved the same-device normal K3 champion to `0.881279 ms` at 1024, `1.612638 ms` at 2048, and `2.932156 ms` at 4096 after restore sanity, with correctness clean.
- Pair-vector full-topk6 tail reduce is rejected:
  - reducing two adjacent 16B output vectors per thread worsened K3 to `0.930079 ms`, `1.697118 ms`, and `3.104637 ms` at 1024/2048/4096;
  - do not retry this shape without ISA/resource evidence showing why the extra per-thread work would improve occupancy or memory behavior.
- K3 copy/reduce worker count above 16 is rejected for the current normal copy-stage design:
  - relaxing the runtime/pybind contract to `K3_COPY_WORKERS=32` was correctness-clean but slowed normal 4096 K3 to `4.174075 ms`;
  - the restored 16-worker helper/direct build returned to `2.938077 ms`;
  - more resident spin/wait worker blocks appear to steal scheduling capacity from the compute/copy pipeline, so larger worker counts should not be retried without redesigning the synchronization model.
- Packed BF16 conversion through `__builtin_hcu_cvt_pk_bf16_f32` is accepted for K3 tail-reduce packback:
  - DCU KB and FlashMLA source evidence back the builtin on gfx938;
  - the usable C++ shape is non-const `auto packed = __builtin_hcu_cvt_pk_bf16_f32(...)` followed by local union reinterpret to `uint32_t`;
  - direct `static_cast<uint32_t>` does not compile, and `const auto` makes the union member const-qualified;
  - replacing paired scalar f32-to-bf16 conversions in tail reduce improved normal K3 to `0.879839 ms`, `1.609598 ms`, and `2.931357 ms` at 1024/2048/4096 with correctness clean.
- Current helper/direct plus packed-convert K3 still has a structural gap:
  - hipprof on normal 4096 shows no standalone K3 combine kernel; the K3-like V2 large-C fused kernel remains the timed location of the gap;
  - local-only normal 4096 K3 is `3.053757 ms`, worse than cross-rank `2.93 ms`, because the current `local_only` generator routes all six topk slots to the local rank and therefore increases local combine/tail-reduce pressure;
  - do not use current `local_only` as a direct remote-overhead subtraction unless the route generator is changed to preserve the same per-rank local slot count.
- Same-device pure denominators for the current `HIP_VISIBLE_DEVICES=2,3,4,5` normal gate are now:
  - K3 pure medians: 1024 `0.469150 ms`, 2048 `0.799483 ms`, 4096 `1.332190 ms`;
  - K1 pure medians: 1024 `0.855040 ms`, 2048 `1.411390 ms`, 4096 `2.343970 ms`.
- Current same-device helper/direct gaps remain large:
  - K3 integrated-vs-pure degradation is now `+87.54%` at 1024, `+101.33%` at 2048, and `+120.04%` at 4096 after packed convert;
  - K1 integrated-vs-pure degradation is `+74.51%` at 1024, `+94.20%` at 2048, and `+111.10%` at 4096;
  - the next optimization should be evidence-driven around K3 copy-stage/tail-reduce scheduling and/or K1/K3 source staging, rather than another local scalar cleanup.
- Direct row-pointer K3 epilogue scatter is rejected for the normal large path:
  - The pure-vs-integrated logic comparison showed an attractive hypothesis: use the existing K3 rowptr store path to write GEMM epilogue output directly to combine storage, then have copy workers only wait and tail-reduce, avoiding `l2_workspace -> combine` vector copy.
  - This compiled and was correctness-clean at `normal` 4096 cross-rank, but K3 worsened to `10.009423 ms` versus the accepted copy-stage champion around `2.924638-2.931357 ms`.
  - Same-device K3 degradation versus pure `1.332190 ms` became about `+651.34%`.
  - Conclusion: keeping the K3 GEMM epilogue as contiguous writes is more important than removing the later vector copy. Rowptr/scatter stores from the GEMM epilogue are too expensive in this shape and should not be retried as a simple no-copy replacement.
  - Future K3 work should preserve contiguous GEMM output and attack tail-reduce traffic/scheduling or find a more structured vectorized epilogue-reduce design.
- Direct local-slot K3 tail-reduce through `output_index` is also rejected:
  - This narrower pure-vs-fused shortcut preserved contiguous GEMM output and only tried to skip the combine copy for local topk slots by reading those rows directly from `l2_workspace` during tail reduce.
  - It was correctness-clean at `normal` 4096 cross-rank, but worsened K3 to `3.316477 ms` versus the restored accepted champion `2.930398 ms`.
  - The saved copy work is too small in the current cross-rank route, while added `output_index` loads, branches, and helper complexity hit every tail-reduce vector.
  - Do not reintroduce per-vector `output_index` direct-local reduce unless profiling shows combine-copy bandwidth, not tail-reduce index/control cost, is the dominant bottleneck.
- Device-scope K3 compute-tile publish fence is not accepted:
  - Replacing the copy-stage compute tile's `__threadfence_system()` with `__threadfence()` was correctness-clean but produced `2.940317 ms` then `2.923676 ms` at normal 4096, indistinguishable from the restored accepted `2.930398 ms` within current run noise.
  - Treat fence-scope tweaks as secondary cleanup only; they do not explain the current roughly 2x pure gap.
