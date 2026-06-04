# Progress Log: DCU MegaMoE V2

## 2026-06-04 09:40 +08:00 - Reject/Inconclusive K3 Device-Scope Publish Fence Experiment

- User feedback:
  - current optimization direction may be off; summarize ratios versus the original pure groupgemm denominator before continuing.
- Experiment:
  - Temporarily changed only the normal K3 copy-stage compute-tile publish fence from `__threadfence_system()` to `__threadfence()`.
  - Rationale: pure K3 has no compute-to-copy worker publish path, and copy workers read same-device `l2_workspace`, so a system-scope fence might be a fused-only excess cost.
- Validation:
  - local `git diff --check` and `py_compile` passed;
  - remote residual V2 process scan was empty;
  - all eight HCUs were idle before the run;
  - rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_device_fence_publish_20260604.log`;
  - perf logs: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_device_fence_publish_20260604.log` and `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_device_fence_publish_rerun_20260604.log`.
- Result on `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`:
  - first run: K3 `2.940317 ms`, correctness max_abs `0.000721931`, mismatch `0`;
  - rerun: K3 `2.923676 ms`, correctness max_abs `0.000721931`, mismatch `0`.
- Decision:
  - reject/inconclusive and revert. The result is noise-level around the restored accepted K3 `2.930398 ms`, with no reliable multi-size evidence.
  - Do not continue with fence-scope tweaks as a primary direction before summarizing pure ratios and reprioritizing.

## 2026-06-04 09:28 +08:00 - Reject K3 Direct Local Reduce Via Output Index

- User-directed analysis focus:
  - continue optimizing from the code-logic differences between integrated fused K3 and the corresponding pure K3 groupgemm.
  - After rejecting full rowptr/no-copy scatter, test a narrower shortcut: keep contiguous GEMM output but avoid copying local slots from `l2_workspace` to combine storage by reading local GEMM rows directly during tail reduce.
- Experiment:
  - Temporarily extended `reduce_full_topk6_bf16x8_device(...)` and the K3 copy-stage path to accept `output_index`/workspace context.
  - Copy workers skipped the combine copy for local slots, and tail reduce loaded those local slot rows directly from `l2_workspace` using `output_index`.
  - K3 wrapper/pybind/runtime/tests were temporarily wired to pass `output_index` into normal K3.
- Validation:
  - rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_direct_local_reduce_20260604.log`;
  - perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_direct_local_reduce_20260604.log`.
- Result on `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`:
  - V2 staged `8.414713 ms`, baseline e2e `8.649434 ms`, K1 `4.957116 ms`, K2 `0.254720 ms`, K3 `3.316477 ms`;
  - correctness remained clean: max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`;
  - versus same-device pure K3 4096 `1.332190 ms`, direct-local-reduce K3 degradation was about `+148.95%`.
- Decision:
  - reject and revert the direct-local-reduce experiment.
  - Saving the local-slot combine copy is outweighed by added `output_index` traffic, branches, and helper complexity; in the current cross-rank route only about one sixth of topk slots are local, so the saved copy volume is too small.
- Restore validation:
  - restore rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_restore_after_direct_local_reduce_reject_20260604.log`;
  - restore sanity log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_restore_after_direct_local_reduce_reject_20260604.log`;
  - restored 4096 result: V2 staged `8.051835 ms`, baseline e2e `8.699833 ms`, K1 `4.967517 ms`, K2 `0.251040 ms`, K3 `2.930398 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Next judgment:
  - keep the accepted contiguous-output plus vector-copy K3 champion.
  - Future K3 optimization should be driven by pure-vs-fused logic diffs and profiling, but should not use epilogue scatter or per-vector `output_index` direct-local shortcuts without new resource evidence.

## 2026-06-04 09:11 +08:00 - Reject K3 Direct RowPtr No-Copy Experiment

- User-directed analysis focus:
  - compare integrated fused logic against the corresponding pure groupgemm logic and optimize the extra integrated work.
  - K3 normal pure and integrated share the same MT256 C pack5 GEMM body, but integrated currently adds `kUseK3CopyStage`: GEMM writes contiguous `l2_workspace`, copy workers move BF16 rows through `row_output_ptrs` into combine storage, then copy workers run the dense topk6 tail reduce.
- DCU KB check:
  - `dcu-rag-kb-optimize` for fused GEMM scatter/reduce patterns returned Hygon/Flux guidance that communication/reduce semantics can live in the GEMM epilogue, but must be selected by capability/comm mode.
  - BF16 builtin searches only confirmed scalar BF16<->F32 conversion and the already accepted packed F32->BF16 conversion. No source-backed packed BF16 add path was found, so no invented packed BF16 add builtin was introduced.
- Experiment:
  - Changed `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` so a new `kUseRowPtrs && kUseK3CopyStage` combination would make copy workers wait for compute tile flags and run tail reduce without copying `l2_workspace`.
  - Changed `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_ext.cu` to instantiate normal K3 as `kUseRowPtrs=true, kUseK3CopyStage=true`.
  - Intended effect: test whether directly scattering the K3 GEMM epilogue to combine storage can eliminate the workspace+copy overhead while preserving same-kernel tail reduce.
- Validation:
  - local `git diff --check` and `py_compile` passed;
  - remote residual V2 process scan was empty;
  - remote HCU2-HCU5 were free, while HCU1 still had unrelated VRAM occupancy;
  - rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_direct_rowptr_no_copy_20260604.log`;
  - perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_direct_rowptr_no_copy_20260604.log`.
- Result on `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`:
  - V2 staged `15.638533 ms`, baseline e2e `8.729111 ms`, K1 `5.087675 ms`, K2 `0.252800 ms`, K3 `10.009423 ms`;
  - correctness remained clean: max_abs `0.000844955`, mean_abs `6.53070e-05`, mismatch `0`;
  - versus same-device pure K3 4096 `1.332190 ms`, direct rowptr K3 degradation was about `+651.34%`.
- Decision:
  - reject and revert the direct-rowptr/no-copy normal K3 experiment.
  - The result proves the K3 gap cannot be solved by simply replacing contiguous GEMM output plus vector copy with row-pointer scatter epilogue. Scatter/rowptr stores are much more expensive than the copy they remove in this shape.
- Restore validation:
  - restore rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_restore_after_direct_rowptr_reject_20260604.log`;
  - restore sanity log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_restore_after_direct_rowptr_reject_20260604.log`;
  - restored 4096 result: V2 staged `8.049914 ms`, baseline e2e `8.645754 ms`, K1 `4.945436 ms`, K2 `0.249440 ms`, K3 `2.924638 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
  - Same-device restored K3 degradation versus pure `1.332190 ms` is about `+119.53%`.
- Next judgment:
  - keep the current contiguous-output + vector-copy K3 champion.
  - Further K3 work should reduce tail-reduce memory/compute or copy-worker scheduling overhead without turning the GEMM epilogue into per-row scatter stores.

## 2026-06-04 09:18 +08:00 - K3 Packed BF16 Convert Accepted, Profiling Updated

- Evidence and profiling before the source change:
  - DCU KB lookup for gfx938 packed BF16 arithmetic found `__builtin_hcu_cvt_pk_bf16_f32` in Hygon `gfx938-builtin-reference` / FlashMLA source evidence.
  - Current 16-worker helper/direct hipprof log: `hygon_tmp/dcu_megamoe_v2/hipprof_integrated_normal4096_helper_direct_20260604.log`.
  - hipprof run used `normal` 4096, `cross_rank`, ranks `4`, warmup `1`, repeat `2`, stage breakdown enabled.
  - Event timing in the profiled run: V2 staged `8.089831 ms`, K1 `4.974955 ms`, K2 `0.267280 ms`, K3 `2.949516 ms`, correctness max_abs `0.000721931`, mismatch `0`.
  - HIPOPS showed the timed V2 path still uses the two V2 large-C fused kernels, with 7 calls each per worker process and no standalone K3 combine kernel. The K3-like V2 kernel averaged about `2.89-2.92 ms` on the rank processes; K1 varied by rank and dominated the rankmax in some processes.
- Local-only isolation after the helper/direct line:
  - log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_local_only_helper_direct_20260604.log`;
  - `normal` 4096, ranks `4`, route `local_only`, warmup `3`, repeat `5`;
  - V2 staged `7.942710 ms`, baseline e2e `6.675512 ms`, K1 `4.733915 ms`, K2 `0.251040 ms`, K3 `3.053757 ms`, max_abs `0.000835419`, mean_abs `6.16246e-05`, mismatch `0`.
  - Interpretation: current `local_only` is not a clean lower bound for K3 because all six topk slots are local, whereas `cross_rank` distributes slots across ranks. It increases local combine/tail-reduce pressure and should not be used as a direct remote-overhead subtraction for the current route generator.
- Accepted K3 experiment 6: packed BF16 output conversion in tail reduce.
  - Changed `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`.
  - Added `pack2_bf16_f32_device(float lo, float hi)` and used `__builtin_hcu_cvt_pk_bf16_f32` on `gfx938`.
  - Replaced paired scalar f32-to-bf16 conversions in the K3 full-topk6 helper and generic tail-reduce packback paths.
  - Initial compile attempts failed and were fixed:
    - direct `static_cast<uint32_t>` from the builtin's `__bf16x2` vector result is invalid;
    - `const auto packed` made the union member const-qualified, so the final accepted form uses non-const `auto packed` plus a local union reinterpret to `uint32_t`.
  - Successful rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_tail_pkbf16_nonconst_20260604.log`.
- Performance/correctness logs:
  - `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_tail_pkbf16_20260604.log`;
  - `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_k3_tail_pkbf16_20260604.log`.
- Results on `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`:
  - 1024: V2 staged `2.475517 ms`, baseline e2e `3.388317 ms`, K1 `1.546559 ms`, K2 `0.112799 ms`, K3 `0.879839 ms`, max_abs `0.000665665`, mean_abs `9.48523e-05`, mismatch `0`.
  - 2048: V2 staged `4.419836 ms`, baseline e2e `5.291674 ms`, K1 `2.710876 ms`, K2 `0.156160 ms`, K3 `1.609598 ms`, max_abs `0.000694275`, mean_abs `9.49171e-05`, mismatch `0`.
  - 4096: V2 staged `8.044153 ms`, baseline e2e `8.668790 ms`, K1 `4.931995 ms`, K2 `0.249760 ms`, K3 `2.931357 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Same-device K3 pure gap after packed convert:
  - 1024: pure `0.469150 ms`, integrated `0.879839 ms`, degradation `+87.54%`;
  - 2048: pure `0.799483 ms`, integrated `1.609598 ms`, degradation `+101.33%`;
  - 4096: pure `1.332190 ms`, integrated `2.931357 ms`, degradation `+120.04%`.
- Decision:
  - accept the packed BF16 conversion patch. It is small, source-backed by the DCU KB, correctness-clean, and improves all three normal K3 sizes slightly.
  - The remaining K3 gap is still structural: copy-stage/tail-reduce scheduling and combine-buffer traffic dominate over packback scalar overhead.

## 2026-06-04 08:52 +08:00 - Reject K3 Copy/Reduce Workers 32

- Experiment:
  - temporarily changed `megamoe/dcu_megamoe_v2/runtime.py` to `K3_COPY_WORKERS=32`;
  - temporarily relaxed `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_pybind.cpp` contract from `[1,16]` to `[1,32]`;
  - did not change the pure groupgemm denominator or K3 math.
- Rationale:
  - test whether more copy-stage/tail-reduce worker blocks reduce the remaining normal K3 4096 gap after scalar tail-reduce cleanup.
- Validation setup:
  - local `py_compile` passed;
  - remote process scan found no residual V2 jobs;
  - `hy-smi` showed HCU2-HCU5 free, while HCU0-HCU1 had unrelated `44%` VRAM occupancy;
  - remote sync used a small tar archive under `hygon_tmp/dcu_megamoe_v2/`;
  - rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_copyworkers32_20260604.log`;
  - perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_copyworkers32_20260604.log`.
- Result on `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`:
  - V2 staged `9.257426 ms`, baseline e2e `8.678868 ms`, V2-vs-baseline e2e degradation `+6.66628%`;
  - stage times: K1 `4.991034 ms`, K2 `0.250080 ms`, K3 `4.174075 ms`;
  - correctness remained clean: max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`;
  - versus same-device pure K3 4096 `1.332190 ms`, degradation worsened to `+213.32%`.
- Decision:
  - reject the 32-worker contract and restore `K3_COPY_WORKERS=16` plus the `[1,16]` pybind check.
  - The likely cause is copy-worker spin-block residency/scheduling contention overpowering any tail-reduce parallelism gain.
- Restore validation:
  - restore rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_restore_copyworkers16_20260604.log`;
  - restore sanity log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_restore_copyworkers16_sanity_20260604.log`;
  - restored 4096 result: V2 staged `8.040151 ms`, baseline e2e `8.667509 ms`, K1 `4.941595 ms`, K2 `0.251200 ms`, K3 `2.938077 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.

## 2026-06-04 08:36 +08:00 - K3 Full-TopK6 Helper Direct Accepted, Pair-Vector Rejected

- Remote pre-run and restore state:
  - residual V2 process scan was empty;
  - `hy-smi` after `source /opt/dtk/env.sh` showed all eight HCUs at `VRAM 0%` and `HCU 0%`;
  - all same-device measurements in this checkpoint used `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`;
  - after rejecting the pair-vector experiment, the remote V2 K3 extension was rebuilt back to the helper/direct implementation, then sanity-tested at 4096.
- Pure groupgemm denominators on the same visible device set:
  - `hygon_tmp/dcu_megamoe_v2/pure_groupgemm_k1_k3_normal_2048_2_5_w3r5_20260604.log`;
  - `hygon_tmp/dcu_megamoe_v2/pure_groupgemm_k1_k3_normal_1024_4096_2_5_w3r5_20260604.log`;
  - K1 pure medians: 1024 `0.855040 ms`, 2048 `1.411390 ms`, 4096 `2.343970 ms`;
  - K3 pure medians: 1024 `0.469150 ms`, 2048 `0.799483 ms`, 4096 `1.332190 ms`.
- Command/log hygiene notes:
  - a pure-denominator attempt with nested `TOKENS="1024 2048 4096"` quoting failed as `2048: command not found`;
  - a remote loop attempt also expanded `$t` locally in PowerShell when it was not escaped, so it repeated default token settings instead of the intended sweep;
  - direct execution of `scripts/build_dcu_megamoe_v2.sh` failed due executable bit/permission state; use `bash scripts/build_dcu_megamoe_v2.sh` or single-token commands for these remote runs.
- Accepted K3 experiment 4: fixed full-topk6 mask for the current fixed-size normal perf path.
  - Changed only `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`.
  - For `num_topk == 6`, dense identity tail tokens, and `tail_token_count == runtime_num_tokens`, the current generator produces six valid local-topk slots for each token, so the tail reduce can use `slot_mask=0x3f` without reloading `k3_local_topk_mask` for every 16B vector.
  - Rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_fixed_full_topk6_mask_20260604.log`.
  - Perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_4096_k3_fixed_full_topk6_mask_20260604.log`.
  - Results:
    - 1024: V2 staged `2.454078 ms`, baseline e2e `3.419036 ms`, K1 `1.599838 ms`, K2 `0.104480 ms`, K3 `0.886559 ms`, max_abs `0.000665665`, mismatch `0`.
    - 2048: V2 staged `4.406875 ms`, baseline e2e `5.192314 ms`, K1 `2.735196 ms`, K2 `0.150400 ms`, K3 `1.611838 ms`, max_abs `0.000694275`, mismatch `0`.
    - 4096: V2 staged `8.046549 ms`, baseline e2e `8.721589 ms`, K1 `4.929435 ms`, K2 `0.253279 ms`, K3 `2.949597 ms`, max_abs `0.000721931`, mismatch `0`.
- Accepted K3 experiment 5: helper/direct full-topk6 reduce.
  - Added a small device helper that unrolls all six packed BF16 combine-vector loads and returns the packed reduced `uint4`.
  - The fixed full-topk6 path now stores the helper result directly and continues, avoiding the generic mask branch after the dense identity/full-slot checks.
  - Rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_full_topk6_helper_direct_20260604.log`.
  - Perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_4096_k3_full_topk6_helper_direct_20260604.log`.
  - Results:
    - 1024: V2 staged `2.466078 ms`, baseline e2e `3.376797 ms`, K1 `1.492158 ms`, K2 `0.104799 ms`, K3 `0.881279 ms`, max_abs `0.000665665`, mismatch `0`.
    - 2048: V2 staged `4.411195 ms`, baseline e2e `5.248315 ms`, K1 `2.740957 ms`, K2 `0.153440 ms`, K3 `1.612638 ms`, max_abs `0.000694275`, mismatch `0`.
    - 4096: V2 staged `8.038550 ms`, baseline e2e `8.659029 ms`, K1 `4.954555 ms`, K2 `0.248640 ms`, K3 `2.938557 ms`, max_abs `0.000721931`, mismatch `0`.
- Restore sanity after rejecting the next experiment:
  - restore rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_restore_full_topk6_helper_direct_20260604.log`;
  - sanity log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_restore_helper_direct_sanity_20260604.log`;
  - 4096 sanity result: V2 staged `8.048628 ms`, baseline e2e `8.693108 ms`, K1 `4.948153 ms`, K2 `0.250399 ms`, K3 `2.932156 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
  - This confirms the remote `.so` is back on the helper/direct line rather than the rejected pair-vector line.
- Rejected K3 experiment: pair-vector tail reduce.
  - Rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_full_topk6_pair_vec_20260604.log`.
  - Perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_4096_k3_full_topk6_pair_vec_20260604.log`.
  - Results worsened versus helper/direct: K3 1024 `0.930079 ms`, 2048 `1.697118 ms`, 4096 `3.104637 ms`, correctness still clean.
  - Decision: reject and revert. One thread reducing two adjacent 16B output vectors reduces loop overhead on paper but hurts scheduling/register/memory behavior on this path.
- Same-device integrated-vs-pure status after the accepted helper/direct line:
  - K3 1024: pure `0.469150 ms`, integrated `0.881279 ms`, degradation `+87.85%`;
  - K3 2048: pure `0.799483 ms`, integrated `1.612638 ms`, degradation `+101.71%`;
  - K3 4096: pure `1.332190 ms`, integrated restore sanity `2.932156 ms`, degradation `+120.10%`;
  - K1 1024: pure `0.855040 ms`, integrated `1.492158 ms`, degradation `+74.51%`;
  - K1 2048: pure `1.411390 ms`, integrated `2.740957 ms`, degradation `+94.20%`;
  - K1 4096: pure `2.343970 ms`, integrated restore sanity `4.948153 ms`, degradation `+111.10%`.
- Decision:
  - keep helper/direct as the current K3 normal champion.
  - scalar/control cleanup has reduced K3 4096 from about `3.422396 ms` to `2.932156 ms`, but the same-device pure gap remains about `+120%`; the next useful step needs profiling or a deeper K3 copy-stage/tail-reduce scheduling change, not another tiny scalar refactor.

## 2026-06-04 08:11 +08:00 - K3 Dense Identity Tail-Token Specialization Accepted

- Remote pre-run state before validation:
  - residual V2 process scans found only the scan commands themselves;
  - HCU1 had `45%` VRAM but `0%` HCU use, so all new 4-rank validation used `HIP_VISIBLE_DEVICES=2,3,4,5`.
- Confirmed the second accepted K3 combine-vector-index patch at `normal` 1024/2048 on the new free-card device set before adding another source change:
  - log: `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_k3_combine_vec_index_20260604.log`;
  - 1024: V2 staged `2.506718 ms`, baseline e2e `3.404797 ms`, K1 `1.525279 ms`, K2 `0.100960 ms`, K3 `0.904000 ms`, max_abs `0.000665665`, mismatch `0`;
  - 2048: V2 staged `4.447996 ms`, baseline e2e `5.300154 ms`, K1 `2.712157 ms`, K2 `0.152480 ms`, K3 `1.661758 ms`, max_abs `0.000694275`, mismatch `0`.
- Accepted K3 experiment 3: dense identity `tail_tokens` specialization.
  - Changed `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` only.
  - K3 normal fixed-size real-flow currently uses dense identity `tail_tokens[token] = token`; the tail reduce previously reloaded that token id once per 16B output vector.
  - The new path treats `k3_tail_token_count >= 0 && runtime_num_tokens == k3_tail_token_count` as dense identity, uses `reduce_token_idx` directly, reuses `task`/`token_vec_base` for output indexing, and hoists `combine_vecs` plus `slot_stride_vecs` out of the reduce loop.
  - The non-dense/dynamic `runtime_num_tokens == -1` path still reads `tail_tokens`.
  - KB lookup did not provide a source-backed packed BF16 add builtin, so no invented HCU builtin was introduced.
- Validation:
  - local `git diff --check` passed;
  - local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py tests/test_dcu_megamoe_v2.py` passed;
  - rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_dense_tail_identity_20260604.log`;
  - perf/correctness log: `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_4096_k3_dense_tail_identity_20260604.log`.
- Results on `HIP_VISIBLE_DEVICES=2,3,4,5`, backend `normal`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`:
  - 1024: V2 staged `2.519198 ms`, baseline e2e `3.422716 ms`, K1 `1.519998 ms`, K2 `0.101280 ms`, K3 `0.886879 ms`, max_abs `0.000665665`, mean_abs `9.48523e-05`, mismatch `0`.
  - 2048: V2 staged `4.431196 ms`, baseline e2e `5.250396 ms`, K1 `2.731517 ms`, K2 `0.152160 ms`, K3 `1.626238 ms`, max_abs `0.000694275`, mean_abs `9.49171e-05`, mismatch `0`.
  - 4096: V2 staged `8.099513 ms`, baseline e2e `8.716631 ms`, V2-vs-baseline e2e degradation `-7.07978%`, K1 `4.924635 ms`, K2 `0.248800 ms`, K3 `2.969597 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Same-device internal comparison against the pre-change combine-vector-index run:
  - K3 1024 improved `0.904000 -> 0.886879 ms`;
  - K3 2048 improved `1.661758 -> 1.626238 ms`;
  - 4096 improved versus the prior `0,1,2,3` run `3.030876 -> 2.969597 ms`, but strict pure/degradation rows should be recollected on one consistent visible-device set because HCU1 is currently occupied.
- Decision:
  - keep the dense identity specialization. It is correctness-clean and gives a small but consistent K3 normal improvement without touching the pure groupgemm denominator.
  - next K3 target remains the larger fused copy-stage/tail-reduce schedule gap; current scalar/control cleanup alone is not enough to approach pure K3.

## 2026-06-04 08:01 +08:00 - K3 Normal Tail-Reduce TopK6 Fast Path Accepted

- Resumed from session `019e6ecc-aaed-74e1-aa6e-78b8ee3133f3` and preserved the active Phase 9 pure-vs-integrated performance gate.
- Restored `megamoe/dcu_megamoe_v2/runtime.py` from the invalid interrupted `K3_COPY_WORKERS=32` state back to the accepted launcher limit `K3_COPY_WORKERS=16`.
- Remote pre-run state:
  - SSH/container access was healthy;
  - `hy-smi` showed all 8 HCUs idle;
  - residual V2 process scans found only the scan commands themselves, no old V2 pytest/prototype/build job.
- Validation before K3 source experiments:
  - local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed;
  - local `git diff --check` passed;
  - remote `py_compile` and `git diff --check` passed after syncing.
- Baseline recovery after restoring copy workers:
  - log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_copy_workers16_resume_20260603.log`;
  - backend `normal`, tokens `4096`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`;
  - V2 staged median `8.521912 ms`, baseline e2e `8.765114 ms`, V2-vs-baseline e2e degradation `-2.77466%`;
  - correctness max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`;
  - stages: K1 `4.959357 ms`, K2 `0.250719 ms`, K3 `3.422396 ms`.
- Profile evidence:
  - log: `hygon_tmp/dcu_megamoe_v2/hipprof_integrated_normal4096_resume_20260603.log`;
  - hipprof stage timing K1 `4.961676 ms`, K2 `0.258800 ms`, K3 `3.427117 ms`;
  - HIP trace showed two V2 normal groupgemm kernels, with the K3-like kernel averaging about `3.383985 ms`;
  - no standalone combine kernel appeared in the timed path, so the K3 gap is inside the fused K3 kernel/copy-stage tail reduce.
- Accepted K3 experiment 1: topk=6/all-slot tail-reduce fast path.
  - Changed `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`.
  - Added `accumulate_bf16x8_device(...)` and a `num_topk == 6 && slot_mask == 0x3f` unrolled path in the normal K3 copy-stage tail reduce.
  - Pure groupgemm denominator remains unchanged.
  - Rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_topk6_fastpath_20260603.log`.
  - 4096 log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_topk6_fastpath_20260603.log`.
  - 4096 result: V2 staged `8.219354 ms`, baseline e2e `8.717273 ms`, V2-vs-baseline e2e degradation `-5.71187%`, K1 `4.958876 ms`, K2 `0.249919 ms`, K3 `3.084637 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
  - 1024/2048 log: `hygon_tmp/dcu_megamoe_v2/integrated_normal1024_2048_k3_topk6_fastpath_20260603.log`.
  - 1024 result: V2 staged `2.498398 ms`, baseline e2e `3.351997 ms`, K1 `1.548479 ms`, K2 `0.104000 ms`, K3 `0.919839 ms`, max_abs `0.000665665`, mismatch `0`.
  - 2048 result: V2 staged `4.482235 ms`, baseline e2e `5.221914 ms`, K1 `2.700317 ms`, K2 `0.152000 ms`, K3 `1.687358 ms`, max_abs `0.000694275`, mismatch `0`.
  - Versus the prior accepted K1 checkpoint, K3 improved from `1.003998 -> 0.919839 ms` at 1024, `1.855198 -> 1.687358 ms` at 2048, and about `3.42 -> 3.084637 ms` at 4096.
- Accepted K3 experiment 2: combine-buffer vector index hoist.
  - Changed only the same K3 normal tail-reduce block in `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`.
  - Replaced repeated `local_sections.combine + partial_row * kProblemN` byte-address math with a precomputed `combine_vecs`, `token_vec_base`, and `slot_stride_vecs` vector-index form.
  - Rebuild log: `hygon_tmp/dcu_megamoe_v2/rebuild_k3_combine_vec_index_20260603.log`.
  - 4096 log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_k3_combine_vec_index_20260603.log`.
  - 4096 result: V2 staged `8.146230 ms`, baseline e2e `8.650072 ms`, V2-vs-baseline e2e degradation `-5.82472%`, K1 `4.928155 ms`, K2 `0.248319 ms`, K3 `3.030876 ms`, max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Same-size pure groupgemm denominator comparison after experiment 2:
  - normal K3 4096: pure `1.342780 ms`, integrated `3.030876 ms`, degradation `+125.72%`.
  - normal K1 4096: pure `2.339350 ms`, integrated `4.928155 ms`, degradation `+110.66%`.
  - topk6-fast-path-only normal K3 1024: pure `0.465890 ms`, integrated `0.919839 ms`, degradation `+97.43%`.
  - topk6-fast-path-only normal K3 2048: pure `0.797534 ms`, integrated `1.687358 ms`, degradation `+111.57%`.
- Decision:
  - keep both K3 source changes because they are correctness-clean and materially improve K3 normal 1024/2048/4096 without changing the pure denominator.
  - next immediate validation is to rerun normal 1024/2048 after the combine-vector-index patch, then continue reducing the remaining K3 4096 gap.

## 2026-06-03 21:46 +08:00 - K1 Normal Integrated Metadata Optimization

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing the performance gate.
- Remote pre-run state:
  - container `megamoe` was up;
  - `hy-smi` showed all 8 HCUs at `VRAM 0%` and `HCU 0%`;
  - residual V2 process scan found only the scan command itself, no old V2 pytest/prototype job.
- Synchronized current local V2 sources to remote through `hygon_tmp/dcu_megamoe_v2/local_sync_20260603.tgz`.
- Forced V2 K1/K3 extension rebuild after deleting V2 extension `.so` and K1/K3 object files. Rebuild log:
  - `hygon_tmp/dcu_megamoe_v2/rebuild_integrated_current_20260603.log`
- Current clean gate before new optimization:
  - log: `hygon_tmp/dcu_megamoe_v2/integrated_normal4096_current_gate_20260603.log`;
  - backend `normal`, tokens `4096`, ranks `4`, route `cross_rank`, warmup `3`, repeat `5`;
  - correctness: max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`;
  - stage medians: K1 `7.760150 ms`, K2 `0.251680 ms`, K3 `3.425755 ms`.
- Accepted experiment 1: group K1 normal metadata scan by `source_rank`.
  - Changed only the `kUseSymmRowStage` real-flow metadata path in `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`.
  - `get_sections(...)` and `effective_tokens` are now computed once per source rank per thread instead of inside the flat `route_linear` loop.
  - The pure groupgemm denominator path is unchanged because it does not instantiate `kUseSymmRowStage`.
  - 4-rank cross-rank results:
    - 4096 log `integrated_normal4096_metadata_by_rank_20260603.log`: K1 `6.418871 ms`, K2 `0.249920 ms`, K3 `3.411835 ms`, max_abs `0.000721931`, mismatch `0`.
    - 2048 log `integrated_normal1024_2048_metadata_by_rank_20260603.log`: K1 `2.936796 ms`, K3 `1.848797 ms`, max_abs `0.000694275`, mismatch `0`.
    - 1024 same log: K1 `1.544478 ms`, K3 `0.999198 ms`, max_abs `0.000665665`, mismatch `0`.
  - Compared with the previous same-size integrated table, K1 improved at normal 4096 from `7.527326 ms` to `6.418871 ms` and at 2048 from `3.243505 ms` to `2.936796 ms`.
- Rejected experiment 1: 64B staged-X copy chunks.
  - Hypothesis: copy four contiguous 16B vectors per thread to reduce repeated source pointer resolution.
  - Log: `integrated_normal4096_stage64_metadata_by_rank_20260603.log`.
  - Correctness stayed clean, but K1 worsened to `6.681271 ms`.
  - Decision: reverted. The lost copy parallelism / extra register pressure outweighed fewer metadata lookups.
- Rejected experiment 2: lazy `topk_weight` loads after expert filtering.
  - Hypothesis: read `topk_idx` first and only read `topk_weight` for the current local expert.
  - Log: `integrated_normal4096_metadata_by_rank_lazy_weight_20260603.log`.
  - Correctness stayed clean, but K1 worsened to `6.721431 ms`.
  - Decision: reverted. Extra `output_index` initialization and branch shape hurt more than the saved weight reads.
- Diagnostic local-only probe after metadata-by-rank:
  - log: `integrated_normal4096_local_only_metadata_by_rank_20260603.log`;
  - local-only K1 `6.040792 ms`, cross-rank K1 `6.418871 ms`;
  - cross-rank adds about `0.378 ms`, so the dominant remaining normal K1 gap is still internal metadata/staging/sync, not raw remote communication.
- Accepted experiment 2: cooperative K1 normal metadata scan across the first row-tile's x-blocks.
  - Let the 4 x-blocks for each expert's first row tile partition the route scan.
  - Added an independent per-row metadata count/epoch region under the existing normal `grid_barrier` allocation; it is separate from the later staging row count/epoch to avoid corrupting the groupgemm pipeline synchronization.
  - Correctness smoke:
    - log `correctness_normal32_coop_metadata_20260603.log`;
    - backend `normal`, token `32`, 4-rank cross-rank, max_abs `0.000576496`, mean_abs `9.47803e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
  - Performance:
    - 4096 log `integrated_normal4096_coop_metadata_20260603.log`: K1 `4.923194 ms`, K2 `0.250250 ms`, K3 `3.415516 ms`, total V2 `8.538066 ms`, baseline e2e `8.664148 ms`, max_abs `0.000721931`, mismatch `0`.
    - 2048 log `integrated_normal1024_2048_coop_metadata_20260603.log`: K1 `2.707197 ms`, K3 `1.855198 ms`, max_abs `0.000694275`, mismatch `0`.
    - 1024 same log: K1 `1.558238 ms`, K3 `1.003998 ms`, max_abs `0.000665665`, mismatch `0`.
  - Pure same-size denominator degradation after this accepted patch:
    - normal K1 4096: pure `2.339350 ms`, integrated `4.923194 ms`, degradation `+110.45%`.
    - normal K1 2048: pure `1.408670 ms`, integrated `2.707197 ms`, degradation `+92.18%`.
    - normal K1 1024: pure `0.854590 ms`, integrated `1.558238 ms`, degradation `+82.34%`.
- Current judgment:
  - K1 normal is materially better but still far above the <=20% pure-groupgemm degradation target.
  - The next largest same-size normal gap is K3 4096: integrated K3 remains around `3.41 ms` versus pure `1.342780 ms`.
  - Continue with K3 normal large integrated-vs-pure attribution before returning to deeper K1 staging architecture.

## 2026-06-03 19:46 +08:00 - K3 Runtime Token Count Tail-Reduce Cleanup, Remote Validation Pending

- User emphasized that all current pure-vs-integrated degradation ratios are too high and asked to continue optimization.
- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before changing code.
- Remote status:
  - initial remote container scan found no residual V2 pytest/build/prototype processes;
  - `hy-smi` showed the visible HCUs idle at that moment;
  - the subsequent archive `scp` and direct `ssh` login attempts were repeatedly closed by the remote host (`Connection closed by 10.17.176.13 port 22`), so no new remote build, correctness, or performance run is accepted in this checkpoint.
- Files changed in this checkpoint:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_pybind.cpp`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implementation:
  - added explicit K3 `runtime_num_tokens` propagation through the K3 Python wrapper, pybind layer, raw extension entry, and kernel launch.
  - K3 `ll` tail reduce now derives `total_reduce_vecs` from `v2_effective_num_tokens(...)` instead of always scanning `num_max_tokens_per_rank`.
  - K3 `normal` copy-stage tail reduce now accepts a `tail_token_count`; for dynamic dispatch it can use `-1` and derive the count from the sym-buffer runtime token counter inside the kernel.
  - `runtime.py` now passes the same runtime-token semantics to K3 that K1 uses: fixed token count for eager fixed-size runs and `-1` for dynamic/uneven dispatch.
  - Updated the local runtime contract test expectations so fixed-size K3 receives the fixed token count and dynamic dispatch passes `-1` to K3, matching K1.
- Validation completed:
  - local `python -m py_compile megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py setup.py` passed.
  - local `git diff --check` passed.
  - static C++ signature review confirmed the K3 raw prototype, extension call, pybind wrapper, and Python wrapper pass `runtime_num_tokens` in the same order; normal additionally passes `tail_token_count`.
  - local pytest could not run because the local Windows Python environment still has no `pytest` module.
- Tokens tested: none in this checkpoint because SSH/remote validation is unavailable.
- Pure groupgemm time: not newly measured.
- Fused time: not newly measured.
- Degradation ratio: not newly measured.
- Correctness max_abs / mean_abs / mismatch: pending.
- Profile evidence: not collected.
- Next judgment:
  - when SSH recovers, sync and rebuild before any GPU run;
  - first validate the already-staged K1 normal direct-pull experiment at 4-rank `normal` 4096 against prior K1 `7.527326 ms`;
  - then validate this K3 tail-count cleanup with 4-rank correctness and same-size K3 stage timing, especially non-4096 or uneven cases where aligned `num_max_tokens_per_rank` can exceed the real local token count.

## 2026-06-03 16:39 +08:00 - Priority Shift Back To Prototype Gap Optimization

- User clarified the immediate priority: do not compare against V1 yet.
- Plan update:
  - current optimization loop first compares integrated real-flow K1/K3 stage timings against the corresponding standalone V2 K1 fused / K3 fused prototype kernels;
  - same-size V1 timing remains a later final acceptance gate, but it should not interrupt the current prototype-gap repair work.
- Latest 4-rank cross-rank `normal` integrated quick performance already collected with `warmup=1`, `repeat=2`, stage breakdown:
  - 32 tokens: V2 staged `1.707838` ms, baseline e2e `2.397597` ms, K1 `1.025839` ms, K2 `0.085280` ms, K3 `0.649199` ms, max_abs `0.000576496`, mismatch `0`;
  - 128 tokens: V2 staged `1.769438` ms, baseline e2e `2.516797` ms, K1 `1.053919` ms, K2 `0.086560` ms, K3 `0.670639` ms, max_abs `0.000576496`, mismatch `0`;
  - 1024 tokens: V2 staged `2.685917` ms, baseline e2e `3.254477` ms, K1 `1.581679` ms, K2 `0.120400` ms, K3 `0.996639` ms, max_abs `0.000665665`, mismatch `0`;
  - 4096 tokens: V2 staged `11.308869` ms, baseline e2e `8.570392` ms, K1 `7.663913` ms, K2 `0.249999` ms, K3 `3.420796` ms, max_abs `0.000721931`, mismatch `0`.
- Pure groupgemm time: not measured in this integrated perf run.
- Fused prototype time: pending standalone prototype rerun under the same 4-rank device set.
- Degradation ratio: pending prototype denominator.
- Profile evidence: not collected yet.
- Next judgment:
  - rerun standalone V2 K1/K3 prototype fused kernels for matching tokens/backends on `HIP_VISIBLE_DEVICES=2,3,4,5`;
  - use the biggest integrated-vs-prototype gap to choose the next smallest optimization, likely K1 normal 4096 or K3 normal cross-rank tail/copy-stage.

## 2026-06-03 16:34 +08:00 - 4-Rank Cross-Rank Correctness Revalidated After Tail Mask Fix

- Synced local `dcu_mega_v2` V2 files to the remote workspace by archive/scp overwrite; remote branch state was intentionally ignored.
- Remote pre-run checks:
  - no V2 pytest/build/prototype residual process was found before build/test runs;
  - HCU0/HCU1 still had high VRAM usage, while HCU2-HCU5 were 0% VRAM / 0% HCU and were used for 4-rank validation.
- Build/static checks:
  - remote `git diff --check` passed;
  - remote `python3 -m py_compile tests/test_dcu_megamoe_v2.py megamoe/dcu_megamoe_v2/api.py megamoe/dcu_megamoe_v2/runtime.py` passed;
  - forced remote `DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` rebuilt the V2 K1/K3 extension modules successfully.
- Smoke:
  - `HIP_VISIBLE_DEVICES=2 python3 -m pytest -q tests/test_dcu_megamoe_v2.py -s`
  - result: `20 passed, 2 skipped`.
- 4-rank cross-rank correctness after changing K1-produced `local_topk_mask` to valid global topk slots:
  - devices: `HIP_VISIBLE_DEVICES=2,3,4,5`;
  - route mode: `cross_rank`;
  - ranks: 4;
  - backend `normal` passed tokens `32,512,1024,2050`;
  - backend `ll` passed tokens `32,512,1024,2050`.
- Correctness max_abs / mean_abs / mismatch:
  - `normal`, 32: max_abs `0.0005764961`, mean_abs `9.4780e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `normal`, 512: max_abs `0.0006141663`, mean_abs `9.4935e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `normal`, 1024: max_abs `0.0006656647`, mean_abs `9.4852e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `normal`, 2050: max_abs `0.0006942749`, mean_abs `9.4919e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `ll`, 32: max_abs `0.0004589558`, mean_abs `6.2686e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `ll`, 512: max_abs `0.0003800392`, mean_abs `1.7668e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `ll`, 1024: max_abs `0.0005016327`, mean_abs `3.0853e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`;
  - `ll`, 2050: max_abs `0.0005726814`, mean_abs `3.9676e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
- Pure groupgemm time: not measured in this correctness checkpoint.
- Fused time: not measured in this correctness checkpoint.
- Degradation ratio: not measured in this correctness checkpoint.
- Profile evidence: not collected in this correctness checkpoint.
- Failure note:
  - the first combined `ll,normal` run failed before kernel validation with `TCPStore` `EADDRINUSE` on the selected `MASTER_PORT`;
  - no residual pytest/spawn process was found afterwards, and separate backend reruns passed, so this is recorded as a distributed rendezvous port collision rather than a kernel correctness issue.
- Plan update:
  - marked the K3 normal cross-rank tail-mask semantic repair item complete.
- Next judgment:
  - run 4-rank cross-rank performance for missing `normal` rows with stage breakdown;
  - add/collect same-size DCU MegaMoE V1 timing in the same harness so the hard V2-vs-V1 gate can be judged.

## 2026-06-03 16:08 +08:00 - K3 Normal Tail Mask Semantics Patched

- Re-read the K3 normal tail-mask caveat and patched the smallest source-level issue before rerunning 4-rank validation.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implementation:
  - K1-produced `local_topk_mask` now marks valid global topk slots for local source tokens instead of only slots whose expert belongs to the current rank.
  - This is intended to make K3 normal cross-rank tail reduce include remote expert slots after remote ranks store their partial rows into the local combine buffer.
- Tokens tested: pending remote rebuild/validation.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: pending.
- Profile evidence: not collected.
- Next judgment:
  - Sync local files to remote, force rebuild V2 K1/K3 extensions, rerun 4-rank cross-rank correctness at 32/512/1024/2050, then collect normal cross-rank perf rows.

## 2026-06-03 16:05 +08:00 - Resume With 4-Rank First, 8-Rank Later

- Re-read the active plan/findings/progress and the remote workflow before resuming.
- User clarified the run cadence: tune with 4 ranks first, then produce 8-rank data when 8 cards are available.
- Remote pre-run status:
  - no V2 pytest/build/prototype residual process was found;
  - `hy-smi` showed HCU0/HCU1 with high VRAM, and HCU2-HCU7 at 0% VRAM / 0% HCU;
  - next 4-rank runs should use `HIP_VISIBLE_DEVICES=2,3,4,5`.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.
- Next judgment:
  - Sync local `dcu_mega_v2` files to remote by archive/scp overwrite, regardless of remote branch.
  - Start with 4-rank cross-rank normal perf and K3 normal tail-mask correctness repair before 8-rank data collection.

## 2026-06-03 16:00 +08:00 - Task Plan Completed Marker Style Updated

- Updated `.planning/dcu_megamoe_v2/task_plan.md` completed checklist markers from Markdown `[x]` to visible `✅` at the user's request.
- Pending checklist items remain `[ ]` so unfinished work is still easy to scan.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.

## 2026-06-03 15:50 +08:00 - Prototype Gap Repair Wording Tightened

- Re-read the active plan/finding wording after the user clarified that prototype gaps should be repaired as much as possible, not treated as optional cleanup.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Plan update:
  - Removed wording that implied integrated-vs-prototype gaps can be treated as permanent.
  - Kept V2-vs-V1 same-size speedup as the hard performance gate.
  - Clarified that large prototype gaps remain active optimization issues: if not closed at a checkpoint, they require root-cause evidence and an explicit next repair item.
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.
- Next judgment:
  - When cards are free, measure V2-vs-V1 first for the hard gate, but still profile K3b/K1 prototype gaps and keep repairing them instead of treating them as accepted permanent overhead.

## 2026-06-03 15:49 +08:00 - Performance Gate Recalibrated To V1

- Re-read the active plan and findings after the user clarified that prototype gap should not be the hard gate if it remains open at a checkpoint.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Plan update:
  - Reclassified integrated K1/K3b stage-vs-prototype comparison as a diagnostic guardrail.
  - Added the hard acceptance rule: V2 must beat the current DCU MegaMoE V1 path for the same rank count, token count, route mode, warmup/repeat policy, and timing harness.
  - Added Phase 9 checklist items to measure same-size V1 performance and fix any V2-vs-V1 regression before acceptance.
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.
- Next judgment:
  - When cards are free, collect both V2 cross-rank stage timings and same-size DCU MegaMoE V1 timings.
  - If V2 beats V1 but remains far from prototype, record the current cause with profile evidence and keep a concrete repair item open without making the prototype gap itself the final hard gate.

## 2026-06-03 15:49 +08:00 - Cross-Rank Performance Gate Reaffirmed

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` after the user emphasized that performance must be guaranteed once cards are available.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Plan update:
  - Made the integrated K1/K3b stage-vs-prototype comparison a hard performance gate.
  - Added explicit next checklist items for missing cross-rank `normal` perf rows, K3b overhead profiling, K1 sync/metadata overhead profiling, and targeted fixes before performance acceptance.
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.
- Next judgment:
  - When 4 or 8 clean cards are available, resume with cross-rank performance, not unrelated feature work.
  - Current priority remains explaining and reducing the integrated K3b overhead versus the prototype fused K3 kernel; K1 small-token sync overhead is the secondary profiling target.

## 2026-06-03 09:20 +08:00 - Milestone G Build Path Correction For K1/K3 Extensions

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing.
- Recorded the K1/K3 default torch JIT path as rejected for real-flow loading:
  - after the raw `.cu` / pybind `.cpp` split, K1 JIT still stayed in hipcc for more than 15 minutes under gfx938-only compile;
  - continuing the same JIT path would repeat a known failed experiment.
- Files changed:
  - `setup.py`
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_ext.cu`
  - `tests/test_dcu_megamoe_v2.py`
  - planning files
- Implementation notes:
  - Added `DCU_MEGAMOE_V2_KERNEL_ONLY` around standalone host-only code in the prototype C++ file so extension translation units can include current kernel definitions without compiling CLI parsing, host random generation, CPU reference helpers, or host pack5 helper code.
  - K1/K3 raw extension `.cu` files now define both `DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN` and `DCU_MEGAMOE_V2_KERNEL_ONLY`.
  - K1/K3 Python loaders now require prebuilt extension modules and no longer default to `torch.utils.cpp_extension.load`.
  - `setup.py` now registers V2 K1/K3 pybind extensions and V2 Python subpackages under the existing HIP/megamoe build condition.
  - Added static tests to ensure V2 K1/K3 extensions are registered in setup and K1/K3 Python wrappers do not fall back to default JIT.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py tests/test_dcu_megamoe_v2.py setup.py` passed.
  - Local `git diff --check` passed.
  - Remote archive sync passed after removing root-owned V2 `__pycache__` directories generated by previous container runs and excluding pycache from the archive.
  - Remote `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` passed: `15 passed in 13.05s`.
  - Remote `DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed in about `234.6s`.
  - Remote import smoke confirmed:
    - K1 `launch_k1_ll_symm_stage=True`, `launch_k1_normal_symm_stage=True`
    - K3 `launch_k3_ll_rowptr_tail_reduce=True`, `launch_k3_normal_copy_stage_tail_reduce=True`
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.
- Next judgment:
  - Re-read the next Milestone G item and start the V2 runtime workspace/metadata orchestration.
  - Keep the stage-owned kernel-source split on the plan as cleanup/finalization; the prebuilt package extension path is usable enough to continue runtime integration.

## 2026-06-03 08:21 +08:00 - Milestone G K1/K3 C Pack5 Extension Launchers

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing Milestone G.
- Completed the Milestone G subitem to split callable K1/K3 prototype launch entry points out of the standalone `main` path.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - planning files
- Implementation notes:
  - Added `DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN` around the standalone prototype `main`, allowing V2 extension translation units to include the current C pack5 kernel definitions without pulling in the benchmark executable entry.
  - Replaced K1/K3 placeholder extension functions with C pack5 launchers:
    - K1 ll: `launch_k1_ll_symm_stage`
    - K1 normal: `launch_k1_normal_symm_stage`
    - K3 ll: `launch_k3_ll_rowptr_tail_reduce`
    - K3 normal: `launch_k3_normal_copy_stage_tail_reduce`
  - Python wrappers now call these extension functions when explicit workspace/metadata tensors are provided.
  - The launcher layer does not allocate device memory and does not perform H2D copies; it requires staged workspace, route scratch views, barriers, row pointers, and metadata to be provided by the future V2 runtime orchestration.
  - Runtime still refuses end-to-end execution because workspace and metadata orchestration is not connected yet.
  - This is a transitional include bridge. The plan now tracks replacing it with stage-owned C pack5 kernel headers once the real-flow metadata contract is stable.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Local `rg` confirmed `k1_not_connected` / `k3_not_connected` no longer exist in V2 extension source; remaining occurrences are static test assertions.
  - Local `python -m pytest -q tests/test_dcu_megamoe_v2.py` could not run because local Python still lacks `pytest`.
- Tokens tested: none in this checkpoint.
- Pure groupgemm time: not measured.
- Fused time: not measured.
- Degradation ratio: not measured.
- Correctness max_abs / mean_abs / mismatch: not measured.
- Profile evidence: not collected.
- Next judgment:
  - Attempt remote validation with a single archive-style sync once SSH is available.
  - Then implement V2 runtime workspace views and metadata production so K1/K2/K3 can be connected without falling back to baseline or big-fused paths.

## 2026-06-02 - Milestone G K1/K3 Extension Loader Scaffold

- Re-read the next Milestone G checklist item after completing the V2 pack5 weight transform API.
- Added K1/K3 Python extension loader scaffolds under the independent V2 package.
- Files changed:
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py`
  - planning files
- Implementation notes:
  - K1 and K3 wrappers now have `_load_k1_ext` / `_load_k3_ext` JIT loaders targeting `hygon_tmp/dcu_megamoe_v2/torch_extensions`.
  - The wrappers validate `backend in {'ll', 'normal'}`.
  - The current `.cu` extension functions are still placeholders and intentionally fail if called.
  - Added `V2StagePlan` / `get_v2_stage_plan` to make the ll/normal K1/K3 mapping explicit and testable before launch code is wired.
  - `ll`: K1 low-latency C pack5 with `K=4096`, K3 low-latency C pack5 with `K=2048`, no K3 copy-stage, tail-reduce enabled.
  - `normal`: K1 normal C pack5 with `K=4096`, K3 normal C pack5 with `K=2048`, K3 copy-stage and tail-reduce enabled.
  - The Milestone G pybind item remains open until callable prototype kernel launch entry points are split out of the standalone `main`.
- Verification:
  - Local `python -m py_compile` passed for K1/K3 wrapper files, runtime/API files, and the V2 test file.
  - Local runtime/layout smoke script passed without importing the parent `megamoe` package.
  - `git diff --check` passed.
- Remote validation:
  - Attempted explicit file sync to `hg@10.17.176.13:/home/hg/yuguo/DeepGEMM`.
  - SSH began rejecting connections after repeated per-file upload attempts (`kex_exchange_identification: read: Connection reset`, then `Connection refused`).
  - No remote pytest result is recorded for this checkpoint. Future remote sync should use a single archive/scp operation rather than many SSH/SCP sessions.

## 2026-06-02 - Milestone G Item 6: V2 Pack5 Weight Transform API

- Re-read the next Milestone G checklist item after completing the V2 source layout.
- Completed Milestone G item 6: added a V2-owned pack5 weight transform API and explicit tests.
- Files changed:
  - `megamoe/dcu_megamoe_v2/layout.py`
  - `tests/test_dcu_megamoe_v2.py`
  - planning files
- API added:
  - `transform_fp8_weights_for_mega_moe_v2_pack5(l1_bf16, l2_bf16)`
  - returns `((l1_pack5_flat_fp8, l1_scale), (l2_pack5_flat_fp8, l2_scale))`
  - uses row-wise FP8 scale and V2 pack5 flattening; it does not call baseline `megamoe.transform_fp8_weights_for_mega_moe`.
- Tests added:
  - package layout file mapping matches the existing prototype layout helper;
  - V2 transform returns flattened pack5 FP8 tensors and contiguous scale tensors;
  - repacking after unpacking preserves the pack5 layout.
- Verification:
  - Local `python -m py_compile tests/test_dcu_megamoe_v2.py megamoe/dcu_megamoe_v2/layout.py` passed.
  - `git diff --check` passed.
- Notes:
  - Corrected V2 weight cast scale floor to match the existing baseline convention: `amax(row).clamp(min=1.0e-4) / 448.0`.
  - Actual full-size L1/L2 transform correctness against kernels still needs remote end-to-end validation after K1/K3 pybind entry points are connected.

## 2026-06-02 - Milestone G Items 4-5: Organized V2 Source Layout

- Re-read the next Milestone G checklist item after extending `tests/test_dcu_megamoe_v2.py`.
- Completed Milestone G item 4: created the organized V2 real-flow source layout under `megamoe/dcu_megamoe_v2/`.
- Completed Milestone G item 5: preserved `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`, its Makefile, and `scripts/build_dcu_megamoe_v2.sh` as prototype/reference harness material only.
- Files changed:
  - `megamoe/dcu_megamoe_v2/K1_fused/__init__.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K2_fused/__init__.py`
  - `megamoe/dcu_megamoe_v2/K2_fused/k2_fused.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/__init__.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/__init__.py`
  - `megamoe/dcu_megamoe_v2/api.py`
  - `megamoe/dcu_megamoe_v2/layout.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - planning files
- Implementation notes:
  - K1/K3 Python wrappers currently raise `NotImplementedError` until the prototype kernels are split into callable extension entry points.
  - K1/K3 `.cu` files are explicit V2 extension surfaces and do not contain production kernel bodies yet.
  - The standalone prototype C++/Makefile/script were not modified or repurposed.
- Verification:
  - Local `python -m py_compile` passed for new Python wrapper files.
  - `git diff --check` passed.

## 2026-06-02 - Milestone G Item 3: V2 Test Entry Extended

- Re-read the next Milestone G checklist item after adding the independent V2 package entry.
- Completed Milestone G item 3: kept the V2 test entry as `tests/test_dcu_megamoe_v2.py` and extended it with package-level contract tests instead of creating another V2 test file.
- Files changed:
  - `tests/test_dcu_megamoe_v2.py`
  - planning files
- Tests added:
  - backend contract for `MEGAMOE_DCU_V2_BACKEND=ll|normal`;
  - V2 package pack5 mapping check against the prototype layout helper;
  - runtime guard check that `fp8_w8a8_mega_moe_v2` refuses to run before K1/K3 extension entry points are connected.
- Verification:
  - Local `python -m py_compile tests/test_dcu_megamoe_v2.py ...` passed.
  - Local `python -m pytest -q tests/test_dcu_megamoe_v2.py` could not run because the local Python environment does not have `pytest` installed.
  - `git diff --check` passed.
- Notes:
  - Package import tests skip when `megamoe.dcu_megamoe_v2` cannot be imported because the local parent package lacks the built `megamoe._C` extension. Remote pytest in the DCU container is still required.

## 2026-06-02 - Milestone G Item 2: Independent V2 Package Entry

- Re-read the next Milestone G checklist item after completing the call-stack map.
- Completed Milestone G item 2: added an independent V2 real-flow package/wrapper entry under `megamoe/dcu_megamoe_v2/` without changing the default `megamoe.fp8_w8a8_mega_moe` behavior.
- Files changed:
  - `megamoe/dcu_megamoe_v2/__init__.py`
  - `megamoe/dcu_megamoe_v2/api.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `megamoe/dcu_megamoe_v2/layout.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/__init__.py`
  - `megamoe/dcu_megamoe_v2/K2_fused/__init__.py`
  - `megamoe/dcu_megamoe_v2/K2_fused/k2_fused.py`
  - `megamoe/dcu_megamoe_v2/K3_fused/__init__.py`
  - planning files
- API contract added:
  - `megamoe.dcu_megamoe_v2.fp8_w8a8_mega_moe_v2(...)`
  - `MEGAMOE_DCU_V2_BACKEND=ll|normal` via `get_v2_backend()` / `normalize_v2_backend()`
  - V2 pack5 helpers and `transform_fp8_weights_for_mega_moe_v2_pack5(...)`
  - V2 K2 wrapper around the existing optimized DCU K2 implementation.
- Current runtime state:
  - `fp8_w8a8_mega_moe_v2` dispatches to `runtime.run_stages_fused_v2`.
  - `run_stages_fused_v2` intentionally raises `NotImplementedError` until K1/K3 pybind entry points are split out of the prototype harness.
  - This prevents accidental fallback to baseline or big-fused paths.
- Verification:
  - Local `python -m py_compile` passed for all new V2 package files.
  - `git diff --check` passed.
- Notes:
  - Local direct import is not used for this checkpoint because this workspace lacks the built `megamoe._C` extension required by parent package initialization. Remote import/pytest will be done after extension entry points are added.

## 2026-06-02 - Milestone G Item 1: Real Test Call Stack Mapped

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before starting implementation.
- Completed Milestone G item 1: mapped the real `tests/test_mega_moe_dcu.py` call stack for V2 integration.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Call-stack summary:
  - Test setup initializes distributed state, creates `sym_buffer` through `megamoe.get_symm_buffer_for_mega_moe`, creates optional DeepEP baseline buffers, generates BF16 input/weights, produces FP8 input through `megamoe.cast_to_fp8_channelwise`, and currently transforms weights through baseline `megamoe.transform_fp8_weights_for_mega_moe`.
  - Fused path copies `x_fp8`, `x_scale`, `topk_idx`, and `topk_weights` into `sym_buffer`, then calls `megamoe.fp8_w8a8_mega_moe`.
  - Baseline path gets DeepEP dispatch layout, dispatches FP8 input, runs `megamoe.deepep_deepgemm_preprocess_channelwise`, runs DeepGEMM L1, runs current optimized DCU K2 `swiglu_quant_channelwise_out`, runs DeepGEMM L2, runs `megamoe.deepep_deepgemm_postprocess_channelwise`, then calls DeepEP combine.
  - Correctness compares fused output against baseline output and checks `cumulative_local_expert_recv_stats` against baseline route counts.
- Integration implications:
  - V2 must use a separate pack5 weight transform before its real-flow call; baseline transform remains unchanged for baseline comparison.
  - V2 real-flow wrapper should accept the same core objects as the current fused path: output tensor, V2 L1/L2 pack5 weights, `sym_buffer`, optional stats, activation controls, and backend selector.
  - V2 tests should reuse the existing data-generation and baseline helpers in `tests/test_dcu_megamoe_v2.py` where possible, without changing default `tests/test_mega_moe_dcu.py` behavior.
- No runtime code or benchmark was changed in this checkpoint.

## 2026-06-02 - Plan Update: Real MegaMoE Flow Integration

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md`.
- Updated the active phase from K3 large performance optimization to real MegaMoE flow integration.
- New integration requirements recorded in `task_plan.md`:
  - keep baseline behavior, baseline layout transform, existing DCU MegaMoE, large_opt, and big-fused paths unchanged;
  - use independent V2 files/symbols/build/test/wrapper APIs where possible, with only minimal opt-in glue in existing `megamoe` files if required;
  - route all V2 runtime sizes through staged fused K1/K2/K3, never big fused;
  - keep separate low-latency and high-throughput prototype kernels and initially select them with `MEGAMOE_DCU_V2_BACKEND=ll|normal` instead of automatic thresholding;
  - run integrated correctness against baseline at tokens 32, 512, 1024, and 2050;
  - keep integrated performance testing on the existing quick set: 32/128/1024/4096;
  - compare integrated timings against prototype-kernel timings to isolate Python/glue/call-stack overhead;
  - leave K3 large performance caveat explicit: correctness-clean, but still above the <=25% degradation target;
  - minimize nonessential H2D copies in the execution path;
  - avoid standalone rank barriers unless in-kernel synchronization is proven insufficient and the exception is documented;
  - split real-flow V2 implementation into organized K1/K2/K3/layout/runtime wrapper files, using the existing prototype C++/Makefile/script as references rather than the production implementation surface;
  - keep the V2 test entry as `tests/test_dcu_megamoe_v2.py`;
  - update `findings.md` whenever a reusable decision, interface contract, benchmark conclusion, failure root cause, rejected experiment, or profiling conclusion is established;
  - after each small checklist item is completed, update plan/progress and re-read the next item before continuing.
- No code or benchmark was changed in this checkpoint; this is the plan confirmation for the next implementation phase.

## 2026-06-02 - Consolidate V2 Docs Into Planning Directory

- Consolidated V2 documentation out of `docs/` into `.planning/dcu_megamoe_v2/` so there is one maintained planning location:
  - `docs/DCU_MEGAMOE_V2.md` -> `.planning/dcu_megamoe_v2/overview.md`
- Removed duplicate `docs/dcu_megamoe_v2_progress.md` and `docs/dcu_megamoe_v2_findings.md`; the active `.planning` progress/findings files are now the source of truth.
- Updated `task_plan.md`, `findings.md`, and `overview.md` to state that V2 planning/progress/findings/overview are maintained only under `.planning/dcu_megamoe_v2/`.
- Active maintenance should use:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
  - `.planning/dcu_megamoe_v2/overview.md`
- No benchmark was run; this was documentation consolidation only.

## Session: 2026-05-29

### Milestone E: K3 Pure C Pack5 Baseline
- **Status:** complete.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `scripts/build_dcu_megamoe_v2.sh`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implementation notes:
  - Parameterized the V2 C groupgemm template for `K=2048` while keeping `N=4096` and the unified V2 pack5 layout.
  - Added independent build-script modes `k3-small`, `k3-large`, and `k3` so K3 pure C pack5 timing is separate from K1 timing.
  - Small-token K3 uses the low-latency C path with `K=2048`.
  - Large-token K3 uses the MT256 C pack5 path with `K=2048`.
  - The previous K3 VMFault came from K1 assumptions: the large C kernel had `kProblemK=4096` and the K-stage order could address stages beyond `K=2048`.
  - Random-data K=2048 large pack5 still had sparse correctness failures until a scheduler barrier was added after direct-to-LDS global loads for the `K=2048` large C template. The global wait helper was restored; the extra barrier is scoped to the K3/K=2048 large C path.
- Correctness:
  - 32 tokens, small lowlat K3: max_abs 0.000244141, mean_abs 6.76449e-10, bit_mismatch 19, value_mismatch 0, checked 786,432 values.
  - 128 tokens, small lowlat K3: max_abs 0.000244141, mean_abs 6.32578e-10, bit_mismatch 75, value_mismatch 0, checked 3,145,728 values.
  - 1024 tokens, large C pack5 K3: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0, checked 25,165,824 values.
  - 4096 tokens, large C pack5 K3: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0, checked 100,663,296 values.
- Pure groupgemm timing:
  - 32 tokens: min 0.155407 ms, rounds 0.155407/0.155418/0.155471.
  - 128 tokens: min 0.163861 ms, rounds 0.163861/0.163957/0.164133.
  - 1024 tokens: min 0.439760 ms, rounds 0.442288/0.440341/0.439760.
  - 4096 tokens: min 1.290590 ms, rounds 1.29059/1.29099/1.29129.
- Fused time: N/A for this milestone; this is the pure K3 C pack5 denominator.
- Degradation ratio: N/A until K3 combine is fused.
- Profile evidence:
  - Not collected for pure K3 in this checkpoint; event timing and correctness establish the denominator. K3 fused combine checkpoints will require hipprof evidence with real 4/8-rank communication.
- Rejected/debug attempts:
  - No-K-stage-reorder experiment for `K=2048` made correctness much worse and was reverted.
  - Forcing masked store for `K=2048` large pack5 did not fix correctness and was reverted.
  - CPU scalar calculation of the first failing random mismatch agreed with the non-pack C reference, proving the issue was in pack5 large C load/scheduling rather than the reference denominator.
- Next judgment:
  - Re-read plan and findings, then start Milestone F: K3 C pack5 fused combine in the groupgemm epilogue. K3 fused is optimized separately from K1 fused; do not reuse K1 row-stage communication conclusions as K3 acceptance evidence.

### Milestone D: K1 Large-Token C Pack5 8-Rank Acceptance
- **Status:** complete.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `scripts/build_dcu_megamoe_v2.sh`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Accepted implementation:
  - `MODE=large-symm-stage`
  - `C_ROW_STAGE=1`
  - `C_STAGE_N_GROUP=4`
  - `DEVICE=0,1,2,3,4,5,6,7`
  - `SYMM_RANKS=8`
  - `SYMM_DEVICES=8`
  - The large-token path stays on the C pack5 MT256 groupgemm skeleton and stages remote A rows inside the same compute kernel. It does not use the rejected ASM/kpack2 path.
- Tokens tested: 1024 and 4096.
- Pure groupgemm times from the same acceptance pass:
  - 1024 tokens: min 0.772241 ms, rounds 0.774688/0.774715/0.772241.
  - 4096 tokens: min 2.28153 ms, rounds 2.28209/2.28153/2.28361.
- Fused 8-rank row-cooperative staging times:
  - 1024 tokens: min 0.916992 ms, rounds 0.918752/0.916992/0.917173.
  - 4096 tokens: min 2.62962 ms, rounds 2.63046/2.63143/2.62962.
- Degradation:
  - 1024 tokens: +18.74% versus same-pass pure K1 C pack5.
  - 4096 tokens: +15.26% versus same-pass pure K1 C pack5.
- Correctness:
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0, checked 25,165,824 values.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0, checked 100,663,296 values.
- Profile evidence:
  - 1024-token hipprof run wrote `hygon_tmp/dcu_megamoe_v2/hipprof_large_row_stage1024_8rank_nocheck`; HIPOPS showed 7 calls to `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16`, total 6,533,112 ns, average 933,301 ns, 100% of HIPOPS kernel time.
  - 4096-token hipprof run wrote `hygon_tmp/dcu_megamoe_v2/hipprof_large_row_stage4096_8rank_nocheck`; HIPOPS showed 7 calls to the same V2 large C fused kernel, total 18,531,516 ns, average 2,647,359 ns, 100% of HIPOPS kernel time.
  - Full-process hipprof also records setup `hipMalloc`, peer enable, host/device copies, and `hipFree`; the timed loop itself launches only the fused C pack5 compute kernel.
- Rejected experiments retained as failure history:
  - Direct large `c-symm-pull`: 1024 tokens improved from 2.30032 ms to 1.47248 ms after hoisting metadata, but still degraded +97.45% and was rejected.
  - All-block `large-symm-stage` group4: correct at 1024 but around 1.085 ms, about +40.5%, rejected after row-cooperative staging.
  - `C_STAGE_N_GROUP=2`, `C_STAGE_N_GROUP=8`, and 64-block global staging were correct or partly correct but slower than row-cooperative group4.
  - 4096 staged failures were traced to a missing end-of-N-tile barrier across N-group iterations; adding that barrier and making loader/invalid waves join it restored exact correctness.
- Next judgment: re-read plan and findings before coding. Proceed to Milestone E: K3 pure C pack5 for L2 `N=4096,K=2048`, covering small tokens 32/128 and large tokens 1024/4096 before any fused combine work.

### Plan Correction: Communication Acceptance and K3 Direction
- **Status:** complete
- User clarified that communication-fused acceptance must use 4 ranks or 8 ranks, not just local/2-DCU smoke.
- User clarified that communication must be hidden inside the compute kernel; standalone dispatch/prebuild/combine kernels in the timed path do not satisfy the V2 goal.
- User clarified that K3 must use the C groupgemm implementation and the same V2 pack5 layout, not the existing ASM/kpack2 path.
- Updated `task_plan.md` with:
  - explicit 4/8-rank communication acceptance gate,
  - mandatory plan re-read cadence after each small milestone,
  - corrected K3 C pack5 Phase 7,
  - large milestones A-G with small checklist items.
- Next implementation sequence:
  1. Remove K3 ASM/kpack2 source/tests/docs from active V2.
  2. Lock unified pack5 for L1 and L2.
  3. Re-run K1 small-token communication on 4/8 ranks.
  4. Build K1 large-token C pack5 fused.
  5. Build K3 pure C pack5 and then K3 C fused combine.

### Prompt Recheck: K3 Small Tokens
- **Status:** complete
- User pointed out that K3 fused plan also needs small-token coverage.
- Updated `task_plan.md` so K3 pure C pack5 and K3 C fused combine both require:
  - small quick tokens: 32 and 128,
  - large quick tokens: 1024 and 4096,
  - real 4-rank or 8-rank communication acceptance,
  - pure-vs-fused degradation against C pack5, target <=25%.
- Added an `Original Prompt Recheck Checklist` to `task_plan.md` covering isolation, hygon_tmp usage, unified pack5, no ASM/kpack2 accepted paths, stage/end-to-end correctness, full sweep, uneven tokens, CUDA graph, and round-output requirements.

### Milestone A: Correct The Source Tree
- **Status:** complete
- User asked to start executing the corrected plan and optimize communication aggressively inside the GEMM pipeline.
- Removed active K3 ASM/kpack2 source surface from V2:
  - `csrc/kernels/dcu_megamoe_v2/stages.py` now contains the isolated K2 wrapper only.
  - `csrc/kernels/dcu_megamoe_v2/layout.py` now contains only the unified V2 pack5 helpers.
  - `tests/test_dcu_megamoe_v2.py` now tests pack5 and K2 only; rejected K3 ASM/kpack2 tests are gone.
  - `csrc/kernels/dcu_megamoe_v2/k3_stage_v2_ext.cu` and `scripts/bench_dcu_megamoe_v2_k3.py` were removed from the active V2 tree.
- Updated source docs and planning findings to state that K3 ASM/kpack2 results are rejected failure history, not accepted Phase 7 evidence.
- Tokens tested in this reset: none yet.
- Pure groupgemm time: N/A for this reset.
- Fused time: N/A for this reset.
- Degradation ratio: N/A for this reset.
- Verification:
  - Local `python -m py_compile csrc/kernels/dcu_megamoe_v2/layout.py csrc/kernels/dcu_megamoe_v2/stages.py tests/test_dcu_megamoe_v2.py`: passed.
  - Local `git diff --check`: passed.
  - Remote `HIP_VISIBLE_DEVICES=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py`: 5 passed in 12.88s.
- Correctness after cleanup: pack5 layout tests and K2 BF16 reference test pass; no K3 C pack5 correctness yet.
- Next judgment: re-read the plan before starting Milestone B pack5 L1/L2 fixtures.

### Milestone B: Lock Unified Pack5
- **Status:** complete
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/layout.py`
  - `csrc/kernels/dcu_megamoe_v2/pack5_layout_check.cpp`
  - `csrc/kernels/dcu_megamoe_v2/Makefile`
  - `csrc/kernels/dcu_megamoe_v2/.gitignore`
  - `scripts/build_dcu_megamoe_v2.sh`
  - `tests/test_dcu_megamoe_v2.py`
- Added Python pack5 shape/flat-offset helpers and a small C++ pack5 offset helper.
- Added L1 fixture for `N=4096,K=4096` and L2 fixture for `N=4096,K=2048`; both use the same pack5 mapping.
- Added Python-vs-C pack5 offset checks for L1, L2, and a small base shape.
- Confirmed layout transform remains setup-only. This milestone did not add any layout transform work to timed benchmark loops.
- Tokens tested: none; layout-only milestone.
- Pure groupgemm time: N/A.
- Fused time: N/A.
- Degradation ratio: N/A.
- Correctness:
  - Local `py_compile`: passed.
  - Local `git diff --check`: passed.
  - Remote `MODE=layout-check bash scripts/build_dcu_megamoe_v2.sh`: passed; sample offsets 20119625 and 14811201.
  - Remote `HIP_VISIBLE_DEVICES=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py`: 7 passed in 13.05s.
- Profile evidence: N/A for layout-only validation.
- Next judgment: re-read plan and move to Milestone C, real 4-rank or 8-rank K1 communication acceptance.

### Milestone C: K1 8-Rank Communication Acceptance
- **Status:** complete for small-token K1.
- Files changed:
  - `scripts/build_dcu_megamoe_v2.sh`
- Added `SYMM_RANKS`, `SYMM_DEVICES`, and `RANK_IDX` passthrough to the V2 build script so the standalone K1 harness can run multi-rank communication tests without manual binary invocation.
- Device/rank setup:
  - `DEVICE=0,1,2,3,4,5,6,7`
  - `SYMM_RANKS=8`
  - `SYMM_DEVICES=8`
  - `RANK_IDX=0`
- Tokens tested: 32 and 128.
- Pure groupgemm times:
  - 32 tokens: min 0.299675 ms, rounds 0.299675/0.300187/0.299713.
  - 128 tokens: min 0.307098 ms, rounds 0.307141/0.307098/0.307226.
- Fused 8-rank staged dispatch-pull times:
  - 32 tokens: min 0.352000 ms, rounds 0.352560/0.352000/0.352171.
  - 128 tokens: min 0.363023 ms, rounds 0.363050/0.363407/0.363023.
- Degradation:
  - 32 tokens: +17.46%.
  - 128 tokens: +18.21%.
- Correctness:
  - 32 tokens: max_abs 0.000244141, mean_abs 1.32851e-09, bit_mismatch 34, value_mismatch 0.
  - 128 tokens: max_abs 0.000244141, mean_abs 1.0992e-09, bit_mismatch 128, value_mismatch 0.
- Profile evidence:
  - `hipprof --hip-trace --stats` 32-token run wrote `hygon_tmp/dcu_megamoe_v2/hipprof_symm_stage32_8rank_nocheck`; HIPOPS showed 7 calls to `V2_K1_LowLatencyMaskedGroupGemmKernel`, average 360548 ns, 100% of HIPOPS kernel time.
  - `hipprof --hip-trace --stats` 128-token run wrote `hygon_tmp/dcu_megamoe_v2/hipprof_symm_stage128_8rank_nocheck`; HIPOPS showed 7 calls to `V2_K1_LowLatencyMaskedGroupGemmKernel`, average 372914 ns, 100% of HIPOPS kernel time.
  - HIP API setup includes `hipMalloc`, `hipMemcpy`, peer access setup, and `hipFree` because the whole process was profiled, but the event-timed loop launches only the fused K1 compute kernel.
  - Direct `dccobjdump` on `k1_groupgemm_v2_hipcc` still reports only host ELF, so ISA evidence was collected through `-save-temps=obj` under `hygon_tmp/dcu_megamoe_v2/save_temps_k1_hipcc`.
  - The generated device assembly contains the staged K1 specialization symbol and `v_mmac_f32_16x16x32_fp8_fp8` plus `s_waitcnt` instructions.
- Errors:
  - Repeated the known nested-shell failure once with `TOKENS="32 128"`; reran each token separately and kept that as the stable command form.
- Next judgment: K1 small-token 8-rank acceptance is clean and within target. Proceed to Milestone D: large-token K1 C pack5 dispatch-pull fusion.

### Milestone D: K1 Large-Token C Pack5 Direct Pull Attempt
- **Status:** rejected experiment, Milestone D still in progress.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `scripts/build_dcu_megamoe_v2.sh`
- Added experimental `c-symm-pull` / `MODE=large-symm-pull` path on the C pack5 MT256x256 kernel.
- The path is C pack5, not ASM. It pulls remote `x/x_sf` from the 8-rank sym-buffer inside the groupgemm compute kernel.
- Tokens tested:
  - 1024 only. 4096 was not run because 1024 already missed the performance target.
- Pure groupgemm denominator:
  - V2 K1 large C pack5 1024 tokens: 0.745766 ms from the established V2 baseline.
- Fused direct-pull timings:
  - Initial direct pull: 2.30032 ms, correctness max_abs 0, mean_abs 0, mismatch 0, degradation +208.45%.
  - After hoisting per-row remote `x` base pointers out of the K loop: 1.47248 ms, correctness max_abs 0, mean_abs 0, mismatch 0, degradation +97.45%.
- Profile evidence:
  - Not collected for this rejected path; timing alone is far outside the <=20% target.
- Failure conclusion:
  - Direct remote A loads remain too exposed for the large C path. Even after reducing metadata recomputation, every N tile still pulls remote A, so remote bandwidth is not hidden enough by compute.
- Next judgment:
  - Do not accept `large-symm-pull` as the large-token K1 fused path.
  - Next large-token attempt should use a resident-grid/persistent staged design so remote A is staged once into local scratch inside the same compute kernel before C pack5 tiles consume it. The design must avoid a grid-wide barrier deadlock by launching only resident persistent blocks or by using a bounded producer/consumer schedule.

## Session: 2026-05-28

### Phase 1: Discovery and Baseline Mapping
- **Status:** complete
- **Started:** 2026-05-28 21:37:20 +08:00
- Actions taken:
  - Loaded planning-with-files, remote SSH Docker workflow, Hygon HIP kernel optimizer, and Karpathy guidance.
  - Loaded superpowers skill-selection context from the actual plugin cache after the listed .system path was missing.
  - Confirmed active plan pointer is dcu_megamoe_v2.
  - Created persistent planning files for the V2 task.
  - Inspected .vscode/sftp.json; remote path is /home/hg/yuguo/DeepGEMM and container repo path should be /workspace/DeepGEMM.
  - Captured initial git status: pre-existing deletion of DCU_MEGAMOE_KERNEL_ANALYSIS.md and untracked third-party/composable_kernel/.
  - Verified remote SSH target `hg@10.17.176.13` and host path `/home/hg/yuguo/DeepGEMM`.
  - Verified Docker container `megamoe` is running.
  - Ran quick pure K1 groupgemm baseline on remote GPU 0 for small tokens 32 and 128.
  - Ran quick pure K1 groupgemm baseline on remote GPU 0 for large pack5 tokens 1024 and 4096.
  - Read CUDA/SM100 MegaMoE scheduler and implementation enough to capture dispatch-pull and combine-reduce overlap patterns.
- Files created/modified:
  - .planning/dcu_megamoe_v2/task_plan.md
  - .planning/dcu_megamoe_v2/findings.md
  - .planning/dcu_megamoe_v2/progress.md

## Benchmark / Correctness Results
| Stage | Tokens | Pure groupgemm ms | Fused ms | Degradation | max_abs | mean_abs | mismatch | Notes |
|-------|--------|-------------------|----------|-------------|---------|----------|----------|-------|
| K1 pure small c-ll | 32 | min 0.299648, rounds 0.299718/0.299648/0.299984 | N/A | N/A | 0.000488281 | 2.27487e-09 | bit 36, value 0 | `K1_LowLatencyMaskedGroupGemmKernel`, BM32/CU64, CHECK=1. |
| K1 pure small c-ll | 128 | min 0.307349, rounds 0.307488/0.307349/0.307680 | N/A | N/A | 0.000488281 | 9.94316e-10 | bit 118, value 0 | `K1_LowLatencyMaskedGroupGemmKernel`, BM32/CU64, CHECK=1. |
| K1 pure large pack5 | 1024 | min 0.752475, rounds 0.753745/0.753264/0.752475 | N/A | N/A | 0 | 0 | bit 0, value 0 | aicc `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<256,256,true>`, CHECK=1. |
| K1 pure large pack5 | 4096 | min 2.26414, rounds 2.26469/2.26418/2.26414 | N/A | N/A | 0 | 0 | bit 0, value 0 | aicc pack5, CHECK=1. |
| V2 K1 pure small c-ll | 32 | min 0.299546, rounds 0.299546/0.299962/0.301872 | N/A | N/A | 0.000488281 | 2.27487e-09 | bit 36, value 0 | V2 independent hipcc harness, BM32/CU64, CHECK=1. |
| V2 K1 pure small c-ll | 128 | min 0.307141, rounds 0.307141/0.307472/0.307498 | N/A | N/A | 0.000488281 | 9.94316e-10 | bit 118, value 0 | V2 independent hipcc harness, BM32/CU64, CHECK=1. |
| V2 K1 pure large pack5 | 1024 | min 0.745766, rounds 0.746299/0.745766/0.746085 | N/A | N/A | 0 | 0 | bit 0, value 0 | V2 independent aicc harness, pack5, CHECK=1. |
| V2 K1 pure large pack5 | 4096 | min 2.26284, rounds 2.26409/2.26437/2.26284 | N/A | N/A | 0 | 0 | bit 0, value 0 | V2 independent aicc harness, pack5, CHECK=1. |
| V2 K1 pull fused small, initial | 32 | 0.299546 | min 0.474704 | +58.47% | 0.000244141 | 8.93813e-10 | bit 29, value 0 | Correct but rejected as too slow; repeated `pull_src_rows` load in every K iteration. |
| V2 K1 pull fused small, initial | 128 | 0.307141 | min 0.482117 | +56.97% | 0.000488281 | 1.32013e-09 | bit 138, value 0 | Correct but rejected as too slow. |
| V2 K1 pull fused small, cached src rows | 32 | 0.299546 | min 0.303910 | +1.46% | 0.000244141 | 8.93813e-10 | bit 29, value 0 | Single fused lowlat kernel with source-token row cache. |
| V2 K1 pull fused small, cached src rows | 128 | 0.307141 | min 0.314656 | +2.45% | 0.000488281 | 1.32013e-09 | bit 138, value 0 | Single fused lowlat kernel with source-token row cache. |
| V2 K1 symm-pull small | 32 | min 0.299269, rounds 0.299269/0.299391/0.299391 | min 0.333588, rounds 0.333588/0.333604/0.334319 | +11.47% | 0.000244141 | 1.32851e-09 | bit 34, value 0 | Single fused lowlat kernel scans sym-buffer peer routes, builds source rows, and pulls `x/x_sf` through peer sym-buffer pointers. |
| V2 K1 symm-pull small | 128 | min 0.307615, rounds 0.307647/0.307615/0.307775 | min 0.341599, rounds 0.341946/0.341599/0.341701 | +11.05% | 0.000244141 | 1.0992e-09 | bit 128, value 0 | Same fused sym-buffer path; route scan cost remains within K1 <=20% target. |
| V2 K1 symm-pull 2 DCUs | 32 | min 0.299269, rounds 0.299269/0.299391/0.299391 | min 0.394100, rounds 0.394100/0.394149/0.394527 | +31.70% | 0.000244141 | 1.32851e-09 | bit 34, value 0 | Real peer reads from rank buffers distributed over 2 DCUs; correct but above <=20% target. |
| V2 K1 symm-pull 2 DCUs | 128 | min 0.307615, rounds 0.307647/0.307615/0.307775 | min 0.432901, rounds 0.432901/0.433418/0.433402 | +40.73% | 0.000244141 | 1.0992e-09 | bit 128, value 0 | Direct remote A loads in the compute loop are not hidden. |
| V2 K1 symm-stage 2 DCUs | 32 | min 0.299269, rounds 0.299269/0.299391/0.299391 | min 0.349984, rounds 0.350287/0.349984/0.350394 | +16.95% | 0.000244141 | 1.32851e-09 | bit 34, value 0 | Same fused kernel stages remote rows into local scratch with vectorized 16B copies before groupgemm. |
| V2 K1 symm-stage 2 DCUs | 128 | min 0.307615, rounds 0.307647/0.307615/0.307775 | min 0.358944, rounds 0.359855/0.358944/0.359306 | +16.69% | 0.000244141 | 1.0992e-09 | bit 128, value 0 | Back within K1 <=20% target for quick small-token cases. |
| V2 K2 reused SwiGLU+quant | 32 rows, hidden 128 | N/A | N/A | N/A | 0 | 0 | mismatch 0 | Existing optimized K2 ext via V2 wrapper, output_bf16=True reference check. |

## Profile Evidence
- Event timing from K1_groupgemm_fp8 and V2 harness is recorded above.
- `hipprof --hip-trace --stats` on V2 `c-ll-pull` tokens=32, warmup=2, repeat=5 wrote `hygon_tmp/dcu_megamoe_v2/hipprof_pull32/pull32.db`.
- hipprof HIPOPS stats showed 7 calls to `void V2_K1_LowLatencyMaskedGroupGemmKernel<32, 4096, 4096, ...>` totaling 2,150,886 ns, average 307,269 ns. This matches one fused kernel per launch; no extra dispatch/prebuild kernel appears in the timed repeat path.
- `hipprof --hip-trace --stats` on V2 `c-ll-symm-pull` tokens=32, warmup=2, repeat=5, `--check 0` wrote `hygon_tmp/dcu_megamoe_v2/hipprof_symm_pull32_nocheck/`.
- hipprof HIPOPS stats showed 7 calls to the same `V2_K1_LowLatencyMaskedGroupGemmKernel<32, 4096, 4096, ...>` totaling 2,418,239 ns, average 345,462 ns, 100% of HIPOPS kernel time. This confirms the timed path has no separate dispatch/prebuild kernel.
- `hipprof --hip-trace --stats` on V2 `c-ll-symm-pull --symm-devices 2` tokens=32, warmup=2, repeat=5, `--check 0` wrote `hygon_tmp/dcu_megamoe_v2/hipprof_symm_pull32_2dev_nocheck/`.
- Two-DCU hipprof HIPOPS stats showed 7 calls to the V2 low-latency kernel totaling 2,836,643 ns, average 405,234 ns, 100% of kernel time. No extra launch explains the slowdown; the cost is in-kernel peer route/input reads.
- `hipprof --hip-trace --stats` on V2 `c-ll-symm-stage --symm-devices 2` tokens=32, warmup=2, repeat=5, `--check 0` wrote `hygon_tmp/dcu_megamoe_v2/hipprof_symm_stage32_2dev_nocheck/`.
- Two-DCU staged hipprof HIPOPS stats showed 7 calls to the V2 low-latency kernel totaling 2,514,078 ns, average 359,154 ns, 100% of kernel time.
- `dccobjdump` directly on the linked executable only reported host ELF; ISA evidence still needs a save-temps/code-object path.

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-28 21:36 +08:00 | Get-Content failed for C:/Users/Administrator/.codex/skills/.system/superpowers/using-superpowers/SKILL.md | 1 | Loaded C:/Users/Administrator/.codex/plugins/cache/openai-curated/superpowers/719ed655/skills/using-superpowers/SKILL.md instead. |
| 2026-05-28 21:45 +08:00 | Remote docker exec command had mismatched nested quotes | 1 | Rerun with simpler command string avoiding embedded sed quotes. |
| 2026-05-28 21:48 +08:00 | Remote `TOKENS="32 128"` env var was split by nested shell quoting | 1 | Rerun each token as a separate command with scalar TOKENS value. |
| 2026-05-28 22:05 +08:00 | Local `pytest` command is not on PATH; `python -m pytest` also reports no pytest module | 1 | Run pytest in the remote DCU container after syncing V2 files. |
| 2026-05-28 22:13 +08:00 | Initial V2 large pack5 check failed loading missing DeepGEMM ASM code object | 1 | Patched V2 harness to skip ASM module loading when pack5 C correctness uses the C baseline reference. |
| 2026-05-28 22:09 +08:00 | Initial scp command copied V2 docs/scripts/tests into remote repo root | 1 | Removed only the accidentally copied root-level files and re-copied each file to its intended directory. |
| 2026-05-28 23:10 +08:00 | First `c-ll-symm-pull` correctness failed because atomic route row assignment produced a different expert-internal row order than the grouped reference | 1 | For the standalone deterministic modulo-route harness, assign row id as `route_linear / num_global_experts`; keep generic atomic fallback for non-modulo routes. |
| 2026-05-28 23:19 +08:00 | V2 K2 test initially skipped because importing `megamoe.dcu_megamoe_large_opt.K2_fused.k2_fused` requires repo-local `megamoe._C` | 1 | Added an isolated JIT fallback that compiles only existing `K2_fused/k2_fused_ext.cu` into `hygon_tmp/dcu_megamoe_v2/torch_extensions`. |
| 2026-05-28 23:27 +08:00 | Real two-DCU peer-read K1 exceeded the <=20% target despite passing correctness | 1 | Keep the result as the real-communication baseline; next K1 optimization needs same-kernel staged/overlapped remote pull instead of direct remote A loads inside the compute loop. |
| 2026-05-28 23:45 +08:00 | First `c-ll-symm-stage` staged bytes scalarly and parsed sym-buffer sections per byte, making 32-token single-device time 0.517333 ms | 1 | Replaced scalar byte staging with 16B vector copies and skipped rows beyond the block tiles groupgemm reads. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5: K1 fused dispatch-pull prototype. |
| Where am I going? | Implement correctness-first K1 fused dispatch pull, then profile overlap and continue to K2/K3. |
| What's the goal? | Isolated DCU MegaMoE V2 three-stage fused prototype with correctness and performance evidence. |
| What have I learned? | K1 pack5 C baselines, existing large-opt K1/K2/K3 API boundaries, and CUDA overlap patterns are captured in findings.md. |
| What have I done? | Completed discovery, skeleton, V2 pack5 layout tests, and V2 pure K1 groupgemm validation. |

### Phase 2: Isolated V2 Skeleton
- **Status:** complete
- **Started:** 2026-05-28 21:57:00 +08:00
- Actions taken:
  - Created isolated V2 source directory, K1 groupgemm harness, Makefile, layout helper, build script, tests, and docs.
  - Synced V2 files to remote and cleaned an accidental root-level scp copy.
  - Added V2 directory .gitignore for generated binaries, dumps, and codegen artifacts.
- Files created/modified:
  - csrc/kernels/dcu_megamoe_v2/.gitignore
  - csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp
  - csrc/kernels/dcu_megamoe_v2/Makefile
  - csrc/kernels/dcu_megamoe_v2/layout.py
  - scripts/build_dcu_megamoe_v2.sh
  - tests/test_dcu_megamoe_v2.py
  - docs/DCU_MEGAMOE_V2.md
  - docs/dcu_megamoe_v2_progress.md
  - docs/dcu_megamoe_v2_findings.md

### Phase 3: Pure GroupGEMM Port
- **Status:** complete
- **Started:** 2026-05-28 22:08:00 +08:00
- Actions taken:
  - Built `k1_groupgemm_v2_hipcc` and `k1_groupgemm_v2_aicc` in the remote DTK container.
  - Ran V2 K1 pure groupgemm quick correctness/perf for small tokens 32 and 128.
  - Fixed V2 large pack5 check to avoid unnecessary ASM code-object loading when using C baseline reference.
  - Ran V2 K1 pure groupgemm quick correctness/perf for large tokens 1024 and 4096.
- Files created/modified:
  - csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp
  - csrc/kernels/dcu_megamoe_v2/Makefile
  - scripts/build_dcu_megamoe_v2.sh

### Phase 4: V2 Weight Layout Transform
- **Status:** complete
- **Started:** 2026-05-28 22:04:00 +08:00
- Actions taken:
  - Implemented V2 pack5 layout helper.
  - Added explicit physical-to-logical N16 mapping test and roundtrip test.
  - Verified remotely with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py`: 4 passed in 1.59s.
- Files created/modified:
  - csrc/kernels/dcu_megamoe_v2/layout.py
  - tests/test_dcu_megamoe_v2.py

### Phase 5: K1 Fused Dispatch Pull Prototype
- **Status:** in_progress
- **Started:** 2026-05-28 22:18:00 +08:00
- Actions taken:
  - Added `c-ll-pull` mode that reads A from source token rows through route mapping inside the low-latency groupgemm kernel.
  - Added `MODE=small-pull` to `scripts/build_dcu_megamoe_v2.sh`.
  - Ran initial pull prototype; correctness passed but degradation was about 57-58%, so the change was not accepted as final.
  - Cached source token rows in registers per tile; correctness remained passing and degradation fell to +1.46% at 32 tokens and +2.45% at 128 tokens.
  - Captured hipprof launch-level evidence for tokens=32.
  - Re-ran pure V2 small path after pull-template edits: 32 tokens min 0.299496 with max_abs 0.000488281; 128 tokens min 0.307311 with max_abs 0.000488281.
  - Ran `git diff --check` successfully.
  - Added `c-ll-symm-pull` mode that uses DCU MegaMoE sym-buffer peer pointer layout and builds route source rows inside the same low-latency groupgemm kernel.
  - Verified `c-ll-symm-pull` correctness/perf for 32 and 128 tokens. Degradation is +11.47% and +11.05% versus same-run pure K1.
  - Captured hipprof `--check 0` evidence showing only the V2 fused kernel in the timed warmup/repeat path.
  - Added `--symm-devices` to allocate rank sym buffers over multiple visible DCUs and verified real two-DCU peer-read correctness for 32 and 128 tokens.
  - Measured two-DCU peer-read degradation: +31.70% at 32 tokens and +40.73% at 128 tokens, above target; recorded as the next optimization target.
  - Added `c-ll-symm-stage`, a single-kernel staged remote pull variant.
  - Rejected scalar byte staging as too slow, then vectorized staging to 16B copies and limited staging to rows read by block tiles.
  - Verified two-DCU staged results: +16.95% at 32 tokens and +16.69% at 128 tokens, both correctness-passing and within K1 target.
- Files created/modified:
  - csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp
  - scripts/build_dcu_megamoe_v2.sh

### Phase 6: K2 Reuse Wrapper
- **Status:** complete
- **Started:** 2026-05-28 23:17:00 +08:00
- Actions taken:
  - Added `csrc/kernels/dcu_megamoe_v2/stages.py` with a V2 K2 wrapper around the existing optimized DCU K2 implementation.
  - Added JIT fallback to build only `megamoe/dcu_megamoe_large_opt/K2_fused/k2_fused_ext.cu` into `hygon_tmp/dcu_megamoe_v2/torch_extensions` when the local repo package lacks `megamoe._C`.
  - Added optional GPU K2 correctness test in `tests/test_dcu_megamoe_v2.py`.
  - Verified remote pytest: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` -> 5 passed in 83.42s.
  - Ran a K2 metric snippet: rows=32, hidden=128, max_abs 0, mean_abs 0, mismatch 0, scale range [2.2321428616578487e-07, 1.6352820239262655e-05].
- Files created/modified:
  - csrc/kernels/dcu_megamoe_v2/stages.py
  - tests/test_dcu_megamoe_v2.py

### Rejected Phase 7 Attempt: K3 ASM/kpack2 Prototype
- **Status:** rejected for acceptance
- **Started:** 2026-05-28 23:54:00 +08:00
- Actions taken:
  - Added V2-owned K3 kpack2 Marlin layout transform and CPU mapping/roundtrip tests.
  - Added K3 ASM code-object build helper that writes `.co` outputs under `hygon_tmp/dcu_megamoe_v2/code_objects`.
  - Added V2-local K3 extension source `csrc/kernels/dcu_megamoe_v2/k3_stage_v2_ext.cu` so the combine launch path is no longer only a Python call into the existing large-opt extension.
  - Added K3 wrapper APIs in `stages.py` for L2+row-pointer combine and tail-reduce wiring.
  - Added K3 correctness test against Torch dequant+matmul reference.
  - Added `scripts/bench_dcu_megamoe_v2_k3.py` for K3 fused timing with setup/layout outside the measured event region.
  - Verified remote pytest: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` -> 9 passed in 17.04s.
  - Ran K3 fused quick timing for tokens 1024 and 4096 with `--check-rows 256`.
  - Captured hipprof `--hip-trace --stats` for K3 fused 1024 tokens in `hygon_tmp/dcu_megamoe_v2/hipprof_k3_fused1024`.
  - Compiled a temporary pure K3 ASM code object from `hygon_tmp/K1_groupgemm_fp8` into `hygon_tmp/dcu_megamoe_v2/code_objects`.
  - Added a named-ASM launcher entry in the V2 K3 extension so experiments can load the temporary pure code object without changing the official K3 combine wrapper.
  - Ran K3 pure-vs-fused quick timing for tokens 1024 and 4096; these numbers are now retained only as rejected experiment history because the path used ASM/kpack2, not C pack5.
  - Captured hipprof `--hip-trace --stats` for K3 pure+fused 1024 tokens in `hygon_tmp/dcu_megamoe_v2/hipprof_k3_pure_fused1024`.
- Files created/modified:
  - csrc/kernels/dcu_megamoe_v2/k3_stage_v2_ext.cu
  - csrc/kernels/dcu_megamoe_v2/layout.py
  - csrc/kernels/dcu_megamoe_v2/stages.py
  - scripts/bench_dcu_megamoe_v2_k3.py
  - tests/test_dcu_megamoe_v2.py
  - docs/DCU_MEGAMOE_V2.md
  - docs/dcu_megamoe_v2_progress.md
  - docs/dcu_megamoe_v2_findings.md

## Rejected K3 ASM/kpack2 Benchmark / Correctness Results
| Stage | Tokens | Pure groupgemm ms | Fused ms | Degradation | max_abs | mean_abs | mismatch | Notes |
|-------|--------|-------------------|----------|-------------|---------|----------|----------|-------|
| V2 K3 fused combine smoke | rows 256 | N/A | N/A | N/A | 0.000244140625 | 1.067741e-08 | 0 | Rejected path: Torch dequant+matmul reference with V2 kpack2 Marlin layout. |
| V2 K3 fused combine | 1024 | temp pure min 0.528159, rounds 0.528159/0.559583/0.538848 | min 0.632447, rounds 0.636575/0.643743/0.632447 | +19.75% | 0.000244140625 | 1.10733529e-08 | 0 | Rejected path: ASM/kpack2, identity row-combine pointers; pure `.co` from hygon_tmp. |
| V2 K3 fused combine | 4096 | temp pure min 1.187039, rounds 1.201535/1.187039/1.198239 | min 1.391935, rounds 1.394559/1.391935/1.397535 | +17.26% | 0.000244140625 | 9.89006921e-09 | 0 | Rejected path: ASM/kpack2, identity row-combine pointers; pure `.co` from hygon_tmp. |

These rejected K3 numbers must not be used as accepted K3 C pack5 denominator or fused-overlap evidence.

## Rejected K3 ASM/kpack2 Profile Evidence
- `hipprof --hip-trace --stats` on `scripts/bench_dcu_megamoe_v2_k3.py --tokens 1024 --warmup 2 --repeat 5 --measure-rounds 1 --check-rows 0` wrote `hygon_tmp/dcu_megamoe_v2/hipprof_k3_fused1024`.
- HIPOPS stats showed 8 calls to `DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE`, total 4,684,794 ns, average 585,599 ns. The 8 calls are first build/load launch + 2 warmup + 5 repeat. This is rejected evidence because the path is ASM/kpack2.
- No separate K3 reduce/combine kernel appeared for that rejected launch path. PyTorch random/quant setup kernels were present because the profiler covered the whole Python process; they are outside the event-timed fused loop.
- `hipprof --hip-trace --stats` on K3 pure+fused 1024 tokens wrote `hygon_tmp/dcu_megamoe_v2/hipprof_k3_pure_fused1024`.
- Pure+fused HIPOPS stats showed 8 calls to temporary pure ASM `DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16`, total 4,223,995 ns, average 527,999 ns, and 8 calls to `K3COMBINE`, total 4,619,036 ns, average 577,379 ns. This is retained only as rejected experiment history.

## K3 Errors / Rejected Attempts
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-05-28 16:00 UTC | `deep_gemm` import failed because repo-local `deep_gemm._C` is unavailable | Tried to use pure DeepGEMM Python API as K3 correctness denominator | Switched K3 stage correctness to Torch dequant+matmul reference for the smoke test. |
| 2026-05-28 16:02 UTC | V2 C pack5 harness VMFaulted for `--n 4096 --k 2048` | Tried to reuse K1 C groupgemm harness as K3 pure denominator | Rejected as an invalid denominator until K/layout assumptions are adapted. |
| 2026-05-28 16:08 UTC | `K3COMBINE` ASM VMFaulted with null row-combine pointers | Tried to use the combine code object as a no-combine pure K3 launcher | Protected `stages.k3_l2_asm_out_v2` to raise until a dedicated no-combine K3 code object is integrated. |

### Milestone F: K3 C Pack5 Identity Row-Combine Pointer
- **Status:** identity row-pointer checkpoint complete; real 4/8-rank combine still pending.
- **Started:** 2026-05-29
- Actions taken:
  - Added `--k3-rowptr` and script modes `k3-small-rowptr`, `k3-large-rowptr`, and `k3-rowptr`.
  - Added C pack5 epilogue row pointer output for both the small low-latency K3 path and the large MT256 C pack5 K3 path.
  - Kept K3 weights on the unified V2 pack5 layout; no ASM/kpack2 path was reintroduced.
  - First rowptr implementation used flat pointer stores and was correct but slow for large tokens.
  - Rejected a dynamic raw-buffer row-resource experiment because 1024-token correctness failed (`max_abs=0.0157471`, `value_mismatch=767`).
  - Accepted row-address prefetching in the large C epilogue: row pointers are loaded once per row group and reused across the hidden-lane stores.
- Files modified:
  - csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp
  - scripts/build_dcu_megamoe_v2.sh
- Correctness:
  - K3 rowptr identity 32 tokens: `max_abs=0.000244141`, `mean_abs=6.76449e-10`, `bit_mismatch=19`, `value_mismatch=0`, `checked=786432`.
  - K3 rowptr identity 128 tokens: `max_abs=0.000244141`, `mean_abs=6.32578e-10`, `bit_mismatch=75`, `value_mismatch=0`, `checked=3145728`.
  - K3 rowptr identity 1024 tokens: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
  - K3 rowptr identity 4096 tokens: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=100663296`.
- Stable timing (`CHECK=0`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - Pure K3 C pack5 current denominators:
    - 32 tokens: min `0.155658` ms.
    - 128 tokens: min `0.164406` ms.
    - 1024 tokens: min `0.440655` ms.
    - 4096 tokens: min `1.29907` ms.
  - K3 identity rowptr:
    - 32 tokens: min `0.160789` ms, degradation `+3.30%`.
    - 128 tokens: min `0.165125` ms, degradation `+0.44%`.
    - 1024 tokens: min `0.518218` ms, degradation `+17.60%`.
    - 4096 tokens: min `1.37193` ms, degradation `+5.61%`.
- Profile evidence:
  - Not yet captured for identity rowptr; this checkpoint is a local row-pointer epilogue validation, not real communication acceptance.
- Next step:
  - Build real 8-rank row-combine targets that point K3 epilogue stores into per-rank sym-buffer combine sections, then validate remote combine writes for 32/128/1024/4096 before adding in-kernel tail reduce.

### Milestone F: K3 C Pack5 Real 8-Rank Row-Combine
- **Status:** small-token row-combine within target; large-token row-combine correctness-clean but still above target.
- **Date:** 2026-05-29
- Actions taken:
  - Added real `--k3-combine` row targets into per-rank sym-buffer combine sections for K3 C pack5.
  - Added an experimental `--k3-copy-stage` large-token path: compute blocks write local C output, then same-kernel copy worker blocks vector-copy completed tiles to real 8-rank combine targets.
  - Added `--k3-copy-workers` and swept 1/2/4/8/16 workers; 16 workers is best, so the default is 16.
  - Added script support for `MODE=k3-large-copy-stage` and `K3_COPY_STAGE=1` on `MODE=k3-large-combine` / `MODE=k3-combine`.
  - Tried vectorizing the direct row-address epilogue with wave shuffle plus `uint4` stores; first version failed correctness because non-writer lanes returned before shuffle, the fixed version passed correctness but slowed 1024 tokens to about 1.26 ms, so the experiment was reverted.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - Planning/docs files.
- Correctness:
  - Small 32 real 8-rank row-combine: `max_abs=0.00012207`, `mean_abs=1.21564e-09`, `bit_mismatch=25`, `value_mismatch=0`, `checked=786432`.
  - Small 128 real 8-rank row-combine: `max_abs=0.000244141`, `mean_abs=9.99455e-10`, `bit_mismatch=73`, `value_mismatch=0`, `checked=3145728`.
  - Large 1024 copy-stage real 8-rank row-combine: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
  - Large 4096 copy-stage real 8-rank row-combine: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=100663296`.
- Stable timing (`CHECK=0`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - Pure K3 C pack5 denominators from the current checkpoint remain:
    - 32 tokens: `0.155658` ms.
    - 128 tokens: `0.164406` ms.
    - 1024 tokens: `0.440655` ms.
    - 4096 tokens: `1.29907` ms.
  - Small real 8-rank row-combine:
    - 32 tokens: min `0.15945` ms, degradation `+2.44%`.
    - 128 tokens: min `0.163456` ms, degradation `-0.58%` versus denominator noise.
  - Large direct remote row-combine:
    - 1024 tokens: min `0.699353` ms, degradation `+58.71%`.
    - 4096 tokens: min `2.14729` ms, degradation `+65.29%`.
  - Large same-kernel copy-stage row-combine:
    - 1024 tokens: min `0.619803` ms, degradation `+40.66%`.
    - 4096 tokens: min `1.86016` ms, degradation `+43.19%`.
- Profile evidence:
  - `hipprof --hip-trace --stats` for K3 small 32 wrote `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_combine32`. HIPOPS shows 7 calls to `V2_K1_LowLatencyMaskedGroupGemmKernel<..., 2048, ...>`, average `164022` ns, 100% of HIPOPS kernel time.
  - `hipprof --hip-trace --stats` for K3 small 128 wrote `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_combine128`. HIPOPS shows 7 calls to the same V2 low-latency fused kernel, average `164159` ns, 100% of HIPOPS kernel time.
- Next step:
  - Continue large-token K3 optimization. Current best large path is same-kernel copy-stage, but it still misses the <=25% target.
  - Add in-kernel combine reduce / tail reduce; current row-combine writes partial rows and is not final combine-reduce acceptance.
- Verification:
  - Remote `bash -n scripts/build_dcu_megamoe_v2.sh`: passed.
  - Remote `MODE=k3-large-copy-stage TOKENS=1024 CHECK=0 WARMUP=1 REPEAT=1`: ran successfully and reported `0.623037` ms.
  - A later `--k3-copy-workers 64` sweep hung because 64 copy-waiter blocks can occupy every CU while waiting for compute flags. Killed the stuck `k1_groupgemm_v2_aicc` process and restored the accepted worker limit to 16.
  - After the guard change, remote aicc rebuild passed and `--k3-copy-workers 32` now rejects before launch with an explanatory error.

### Milestone F: K3 Tail-Reduce Prototype And Copy-Stage Rejected Experiments
- **Status:** local-rank tail-reduce correctness prototype exists; performance and full all-rank reduce are still pending.
- **Date:** 2026-05-29
- Actions taken:
  - Added a correctness-first same-kernel K3 tail-reduce prototype behind `--k3-tail-reduce`.
  - Tail-reduce currently requires the standard topk-slot combine layout and positive copy workers; it reduces the local rank combine buffer into `out[token, hidden]` inside the same C pack5 kernel after row-combine copy workers finish.
  - Re-read the task plan before continuing K3 large optimization; K1 remains accepted and untouched.
  - Checked the remote container for stuck work: no stale `k1_groupgemm_v2` benchmark process was present, and all 8 HCUs reported 0% HCU use.
  - Tried moving K3 copy-worker rows behind the compute grid to avoid early CU reservation. The experiment remained correct but regressed 1024-token timing to about `0.794` ms, so it was reverted.
  - Tried a same-block self-copy mode using `K3_COPY_WORKERS=0` so compute blocks copy their own local `d_out` tile to combine targets. It was faster in raw timing (`~0.57-0.63` ms for 1024) but failed correctness with zeros in combine rows, even after adding a store wait/fence and fixing the copy-loop stride. The experiment was reverted.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `scripts/build_dcu_megamoe_v2.sh`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
  - `docs/dcu_megamoe_v2_progress.md`
  - `docs/dcu_megamoe_v2_findings.md`
- Tail-reduce correctness:
  - 1024 tokens: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=4194304`.
  - 4096 tokens: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=16777216`.
- Tail-reduce stable timing (`CHECK=0`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - 1024 tokens: pure `0.440655` ms, tail-reduce `0.773269` ms, degradation `+75.49%`.
  - 4096 tokens: pure `1.29907` ms, tail-reduce `2.44608` ms, degradation `+88.30%`.
- Current restored correctness baseline:
  - 1024-token K3 large copy-stage with 16 workers: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, single-run `c_avg_ms=0.622238`.
- Next step:
  - Profile the restored large-token copy-stage path and attack the remaining overhead without reintroducing incorrect self-copy or over-subscribed copy-worker designs.

### Milestone F: K3 Large Copy-Stage Fence And Worker Experiments
- **Status:** device-fence and >16 worker experiments rejected; current correct large-token baseline remains the 16-worker system-fence copy-stage.
- **Date:** 2026-05-29
- Actions taken:
  - Captured 1024-token hipprof for the restored 16-worker copy-stage path in `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_copy_stage1024`.
  - Verified hipprof HIPOPS shows only 7 calls to `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16` for warmup=2/repeat=5; no standalone combine kernel appears in the timed loop.
  - Rejected a K3/L2 `--c-tile-n 64` pack5 experiment: correctness passed for pure 1024, but pure time was `0.934237` ms, far slower than the MT256 pure denominator and therefore not a valid optimization direction.
  - Rejected standard-layout row sorting by `source_rank, partial_row`: 1024 improved only slightly to `0.614698` ms, while 4096 regressed to `3.50465` ms, so it was reverted.
  - Tried changing the K3 copy-stage compute-tile ready fence from `__threadfence_system()` to `__threadfence()`.
  - Initially saw promising timings, but a forced rebuild and repeated 1024-token correctness exposed nondeterministic NaN/mismatch failures, so the fence change was reverted to `__threadfence_system()`.
  - Temporarily allowed 24/32 copy workers to test whether more copy bandwidth could help 4096 without deadlocking. Both were much slower and were rejected; the guard is back to `[1,16]`.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - Planning/docs files.
- Rejected timing observations:
  - Device-fence variant reached `0.615066` ms at 1024 and `1.79409` ms at 4096, but it is invalid because repeated correctness failed.
  - 24 copy workers: 4096 tokens `2.31437` ms, worse than 16 workers.
  - 32 copy workers: 4096 tokens `3.74851` ms, worse than 16 workers.
- Restored correctness:
  - 1024-token 16-worker system-fence copy-stage after forced rebuild: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
  - 4096-token 16-worker system-fence copy-stage: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=100663296`.
- Current restored stable timing (`CHECK=0`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - 1024 tokens: pure `0.440655` ms, fused copy-stage `0.613866` ms, degradation `+39.31%`.
  - 4096 tokens: pure `1.29907` ms, fused copy-stage `1.7904` ms, degradation `+37.82%`.
- Profile evidence:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_copy_stage4096_device_fence` is retained only as rejected evidence because the device-fence variant later failed correctness.
  - Save-temps under `hygon_tmp/dcu_megamoe_v2/save_temps_k3_aicc` shows the K3 copy-stage specialization uses `amdhsa_next_free_vgpr 213`, `amdhsa_group_segment_fixed_size 65536`, and no private scratch segment.
- Follow-up rejected self-copy repair:
  - Fixed the earlier self-copy correctness issue by making loader waves and invalid compute waves participate in the post-epilogue tile copy.
  - Correctness then passed for 1024 tokens (`max_abs=0`, `value_mismatch=0`), but stable timing was `0.665637` ms, slower than the 16-worker copy-stage (`0.615066` ms).
  - Removed the self-copy branch and restored the accepted `[1,16]` copy-worker guard.
- Next step:
  - Continue reducing the remaining large-token overhead without weakening the system-scope publication needed for correctness. The likely limiter is that same-kernel copy-worker blocks still reserve the GEMM kernel's 64 KiB LDS and occupy full CUs while waiting/copying.

### Milestone F: K3 Small Same-Kernel Tail-Reduce Prototype
- **Status:** small-token local-rank tail-reduce is correctness-clean and within the <=25% target; full all-rank reduce is still pending.
- **Date:** 2026-05-29
- Actions taken:
  - Added `k3_tail_reduce` support to the low-latency K3 C pack5 row-combine kernel path.
  - Reused the existing `K3_TAIL_REDUCE=1` script knob for `MODE=k3-small-combine`; no new environment variable was added.
  - The kernel now writes real 8-rank row-combine targets, publishes with `__threadfence_system()`, performs an in-kernel grid barrier, then reduces the local rank's topk combine rows into `out[token, hidden]` inside the same low-latency compute kernel.
  - Updated the small-token correctness branch to compare tail-reduce output against a pure C pack5 reference summed by `source_rank/source_token/topk_slot`, rather than accidentally falling back to row-combine remap correctness.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `scripts/build_dcu_megamoe_v2.sh`
  - `.planning/dcu_megamoe_v2/task_plan.md`
- Correctness:
  - 32 tokens, 8 ranks / 8 devices: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0.000244141`, `mean_abs=1.86265e-09`, `bit_mismatch=1`, `value_mismatch=0`, `checked=131072`.
  - 128 tokens, 8 ranks / 8 devices: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0.000488281`, `mean_abs=1.97906e-09`, `bit_mismatch=4`, `value_mismatch=0`, `checked=524288`.
- Stable timing (`CHECK=0`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - Pure K3 C pack5 denominators:
    - 32 tokens: `0.155658` ms.
    - 128 tokens: `0.164406` ms.
  - Small same-kernel local-rank tail-reduce:
    - 32 tokens: min `0.164469` ms, degradation `+5.66%`.
    - 128 tokens: min `0.184448` ms, degradation `+12.19%`.
- Profile evidence:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_tail_reduce32`: HIPOPS shows 7 calls to `V2_K1_LowLatencyMaskedGroupGemmKernel<..., 2048, ...>`, average `166308` ns, 100% of HIPOPS kernel time.
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_tail_reduce128`: HIPOPS shows 7 calls to the same fused low-latency kernel, average `185211` ns, 100% of HIPOPS kernel time.
  - Setup `hipMalloc/hipMemcpy/peer access/free` appears because the full process was profiled; it is outside the event-timed fused loop.
- Next step:
  - Return to K3 large-token combine optimization. The immediate target remains reducing the large 1024/4096 overhead without relying on copy-worker blocks that reserve the full GEMM LDS footprint.

### Milestone F: K3 Large PMC Triage And Pair-Store Rejection
- **Status:** profiling clarified the large-token overhead axis; pair-store direct epilogue experiment was reverted.
- **Date:** 2026-05-29
- Actions taken:
  - Rebuilt the aicc binary after the small tail-reduce changes and revalidated large copy-stage correctness.
  - Captured PMC CSVs for pure K3 large 1024, direct remote row-combine 1024, and copy-stage row-combine 1024.
  - Tried a direct row-combine epilogue experiment that packed adjacent hidden columns into 32-bit stores, aiming to reduce scalar 16-bit remote store pressure without reintroducing copy-worker blocks.
  - The pair-store experiment passed correctness at 1024 but slowed both large quick sizes, so it was fully reverted.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` during the experiment; pair-store code was removed again.
  - Planning/docs files.
- Correctness:
  - Pair-store direct 1024 correctness before revert: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
  - Restored copy-stage 1024 correctness after revert: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
- Timing:
  - Pair-store direct 1024: min `0.711039` ms, worse than previous direct remote row-combine `0.699353` ms and copy-stage.
  - Pair-store direct 4096: min `2.38252` ms, worse than previous direct remote row-combine `2.14729` ms and copy-stage.
  - Restored copy-stage 1024 after rebuild: min `0.623141` ms, still above target versus pure `0.440655` ms.
- PMC evidence (`hipprof --pmc --pmc-type 3`, profiler timing itself is not used as benchmark timing):
  - Pure 1024: `SQ_INSTS_VMEM_RD≈462848`, `SQ_INSTS_VMEM_WR≈393216`, `TCC_EA_WRREQ_STALL≈510555`, `TCP_TCP_TA_DATA_STALL_CYCLES≈2990733`, `GRBM_COUNT≈631285`.
  - Direct remote row-combine 1024: `SQ_INSTS_VMEM_RD≈487424`, `SQ_INSTS_VMEM_WR≈393216`, `TCC_EA_WRREQ_STALL≈4497406`, `TCP_TCP_TA_DATA_STALL_CYCLES≈15071619`, `GRBM_COUNT≈959822`.
  - Copy-stage row-combine 1024: `SQ_INSTS_VMEM_RD≈786378`, `SQ_INSTS_VMEM_WR≈442880`, `TCC_EA_WRREQ_STALL≈706997`, `TCP_TCP_TA_DATA_STALL_CYCLES≈5510910`, `GRBM_COUNT≈871635`.
  - Interpretation: direct scalar remote epilogue is dominated by write/TCP stalls; copy-stage reduces remote write stalls through vector copies but pays extra local reads and full-LDS copy-worker occupancy.
- Next step:
  - Avoid more scalar-to-small-vector direct-store variants. The next promising direction is to keep vectorized remote writes but remove or reduce full-LDS copy-worker residency, or restructure final reduce so copy traffic is not paid for non-local rows in prototype-only paths.

### Milestone F: K3 Large Tail-Reduce Local-Copy Filter
- **Status:** correctness-clean prototype-only optimization; still not performance-accepted.
- **Date:** 2026-05-29
- Actions taken:
  - Added a `k3_tail_reduce != 0` copy-stage filter that only copies row-combine outputs whose destination pointer falls inside the local rank combine buffer.
  - This is intentionally scoped to the local-rank tail-reduce prototype and does not affect the accepted row-combine copy-stage path (`K3_TAIL_REDUCE=0`).
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - Planning/docs files.
- Correctness:
  - 1024 tokens: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=4194304`.
  - 4096 tokens: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=16777216`.
- Timing:
  - 1024 tokens: min `0.764443` ms versus previous tail-reduce `0.773269` ms; still `+73.49%` versus pure `0.440655` ms.
  - 4096 tokens: min `2.39046` ms versus previous tail-reduce `2.44608` ms; still `+84.01%` versus pure `1.29907` ms.
- Judgment:
  - Keep the local-copy filter because it is isolated, correct, and modestly improves the prototype.
  - It does not solve the large-token final reduce target. The main remaining cost is the reduce sweep and same-kernel copy-worker/synchronization structure, not just copying non-local rows.

### Milestone F: K3 Large Tail-Reduce TopK-Slot Skip
- **Status:** correctness-clean prototype-only optimization; still not performance-accepted.
- **Date:** 2026-05-29
- Actions taken:
  - Added a `topk_idx` guard inside the large copy-stage local-rank tail-reduce loop so the prototype skips topk slots whose expert is not owned by the local rank.
  - This matches the current local-rank correctness reference, where only rows produced by this rank's local experts are summed. It is not the final all-rank reduce behavior.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - Planning/docs files.
- Correctness:
  - 1024 tokens: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=4194304`.
  - 4096 tokens: `correctness_ref=c_tail_reduce_local_rank`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=16777216`.
- Timing:
  - 1024 tokens: min `0.744538` ms versus the local-copy-filter result `0.764443` ms and original tail-reduce `0.773269` ms; still `+68.96%` versus pure `0.440655` ms.
  - 4096 tokens: min `2.34016` ms versus the local-copy-filter result `2.39046` ms and original tail-reduce `2.44608` ms; still `+80.14%` versus pure `1.29907` ms.
- Judgment:
  - Keep the topk-slot skip because it is isolated to the local-rank prototype and improves both quick large sizes.
  - It is not a final solution; full all-rank reduce will need a different synchronization/data-availability contract because every topk slot should eventually be produced by some rank.

### Milestone F: K3 Large Copy-Stage Sync Experiments And Pointer-Broadcast Rejection
- **Status:** synchronization experiments rejected; copy-worker pointer-load reduction was correctness-clean but slower and reverted.
- **Date:** 2026-05-29
- Actions taken:
  - Re-read the plan/progress/findings before touching K3 large.
  - Re-established the current K3 large copy-stage baseline before experiments:
    - 1024 tokens: correctness-clean, single checked run `0.635038` ms; no-check rounds `0.616522/0.617434/0.617162` ms.
    - 4096 tokens: no-check rounds `1.79062/1.79354/1.81670` ms.
  - Tried replacing the non-tail-reduce tile-ready publication with `atomicExch`.
  - Tried a GLC/cache-bypass flag load/store variant based on Stream-K style synchronization guidance from the DCU knowledge base.
  - Reverted both synchronization experiments.
  - Tried a copy-worker patch that reads `row_output_ptrs[row]` once per 32-vector row and broadcasts it within the half-wave, instead of loading the same row pointer for every 16B vector chunk.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - Planning/docs files.
- Correctness:
  - Atomic flag experiment: 1024 and 4096 8-rank correctness passed with `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - GLC flag experiment: did not complete correctness; run hung and the container exited. Reverted without keeping any code.
  - Row pointer broadcast patch: 1024 and 4096 8-rank correctness passed with `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
- Timing:
  - Atomic flag experiment regressed:
    - 1024 tokens: min `0.622042` ms, worse than current copy-stage baseline around `0.617` ms.
    - 4096 tokens: min `1.82321` ms, worse than current copy-stage baseline around `1.79` ms.
  - GLC flag experiment: no timing accepted because correctness did not complete and it caused the runtime to hang.
  - Row pointer broadcast patch regressed:
    - 1024 tokens: min `0.625124` ms, worse than current baseline around `0.617` ms.
    - 4096 tokens: min `1.81698` ms, worse than current baseline around `1.790` ms.
- Issues:
  - After the GLC flag hang, `docker megamoe` exited and host `hy-smi` reported `No hycu Driver loaded`; the machine reboot restored `hy-smi`, docker, `/dev/kfd`, and 8 idle HCUs.
- Judgment:
  - Do not retry atomic or GLC flag publication as-is.
  - Do not keep the row-pointer broadcast patch; the shuffle overhead outweighed metadata load savings.
  - The source and remote binary were restored to the system-fence copy-stage baseline after the failed row-pointer broadcast experiment.

### Milestone F: K3 Large Copy-Worker Row-Tile Scheduling Rejection
- **Status:** correctness-clean but slower; reverted to the linear tile scheduling baseline.
- **Date:** 2026-05-29
- Actions taken:
  - Re-read the task plan/progress/findings before continuing K3 large optimization.
  - Tried assigning each copy worker a set of row tiles and then sweeping all hidden tiles for that row (`tile_y` outer, `tile_x` inner), instead of the restored linear tile order (`tile += worker_count`).
  - The goal was to improve destination-row locality and reduce row-combine scattering while keeping all communication inside the same C pack5 compute kernel.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` during the experiment; the source was reverted to the linear copy-worker tile order.
- Correctness:
  - 1024 tokens, 8 ranks / 8 devices: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
  - 4096 tokens, 8 ranks / 8 devices: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=100663296`.
- Timing (`CHECK=0`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - 1024 tokens: rounds `0.75598/0.756242/0.755052` ms, min `0.755052` ms, `+71.35%` versus pure `0.440655` ms.
  - 4096 tokens: rounds `2.02709/2.02361/2.0275` ms, min `2.02361` ms, `+55.77%` versus pure `1.29907` ms.
- Judgment:
  - Rejected. The row-tile order delays consumption of ready hidden tiles and is much worse than the restored copy-stage baseline around `0.616` ms / `1.79` ms.
  - Next step should focus on either reducing copy-worker full-LDS residency, reducing the copy traffic that survives into final reduce, or collecting deeper profile evidence for the current baseline before another source change.

### Code Cleanup: Remove Retired K1/ASM Experiment Entrypoints
- **Status:** source cleanup only; no GPU benchmark run because the cards were busy.
- **Date:** 2026-05-29
- Actions taken:
  - Removed retired ASM/balanced-ASM host launch plumbing from the V2 standalone harness.
  - Removed active CLI/build-script entrypoints for `small-pull`, `small-symm-pull`, and `large-symm-pull`.
  - Removed direct-pull template branches from the low-latency and large C kernels, leaving the accepted K1 staged paths plus current K3 rowptr/combine/copy-stage/tail-reduce code.
  - Kept K3 recent experiment knobs: `--k3-rowptr`, `--k3-combine`, `--k3-copy-stage`, `--k3-copy-workers`, and `--k3-tail-reduce`.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `scripts/build_dcu_megamoe_v2.sh`
  - `docs/DCU_MEGAMOE_V2.md`
  - `.planning/dcu_megamoe_v2/progress.md`
- Correctness/performance:
  - Not rerun in this cleanup step.
- Validation:
  - Local `git diff --check` passed.
  - Remote `make -C csrc/kernels/dcu_megamoe_v2 hipcc` passed; only existing `mega_moe_dcu.cuh` missing-return warnings.
  - Remote `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - Remote `bash -n scripts/build_dcu_megamoe_v2.sh` passed.
  - No GPU benchmark run because the cards were busy.
- Next validation:
  - Rerun quick K1 pure/fused and K3 large copy-stage smoke when cards are available.

### Milestone F: K3 Large Copy-Stage Ready Publication Sync Fix
- **Status:** correctness fix kept; performance still above target and needs optimization.
- **Date:** 2026-06-02
- Actions taken:
  - Re-read the task plan, progress, and findings before continuing K3 large.
  - Re-established current pure K3 C pack5 timing with 8-rank acceptance context available:
    - 1024 tokens: rounds `0.444725/0.443114/0.443349` ms, best `0.443114` ms.
    - 4096 tokens: rounds `1.30169/1.30154/1.30203` ms, best `1.30154` ms.
  - Re-ran the large copy-stage fused path with real 8 ranks / 8 devices.
  - Found that the previous compute-block ready publication could publish the tile-ready flag after only thread 0 had waited for its own VMEM stores. That was not a block-wide output visibility guarantee.
  - Moved `wait_vmem_lds_store_device()` before the block barrier so every compute thread waits for its own output stores before thread 0 publishes the ready flag.
  - Added the same all-thread VMEM wait before the copy-worker tail-reduce done counter publication.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
- Correctness before fix:
  - 1024 fused copy-stage: correctness-clean.
  - 4096 fused copy-stage: correctness failed repeatedly. Example: `max_abs=0.0143127`, `mean_abs=1.19198e-07`, `bit_mismatch=4743`, `value_mismatch=3557`.
- Correctness after fix:
  - 1024 fused copy-stage, 8 ranks / 8 devices: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=25165824`.
  - 4096 fused copy-stage, 8 ranks / 8 devices: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, `checked=100663296`.
- Timing after fix (`CHECK=1`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - 1024 fused copy-stage: rounds `0.627325/0.626941/0.627005` ms, best `0.626941` ms, `+41.49%` versus pure `0.443114` ms.
  - 4096 fused copy-stage: rounds `1.83131/1.83909/1.83452` ms, best `1.83131` ms, `+40.70%` versus pure `1.30154` ms.
- Judgment:
  - Keep the sync fix because 4096 correctness is otherwise not reliable.
  - This moves the current baseline farther from the <=25% large-token K3 target, so the next small step is to sweep copy-worker count under the corrected synchronization and then profile the best correct setting.

### Milestone F: K3 Large Copy-Worker Count Sweep After Sync Fix
- **Status:** default 16 workers remains best; lower worker counts are rejected.
- **Date:** 2026-06-02
- Actions taken:
  - Swept `K3_COPY_WORKERS=4/8/12/16` with real 8 ranks / 8 devices.
  - Kept `CHECK=1` for every candidate because this phase is sensitive to synchronization visibility.
- Files modified:
  - Planning/docs only.
- Correctness:
  - All worker-count candidates passed for 1024 and 4096 tokens with `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
- Timing (`CHECK=1`, `WARMUP=5`, `REPEAT=20`, `MEASURE_ROUNDS=2`):
  - 4 workers:
    - 1024 tokens: best `1.43511` ms.
    - 4096 tokens: best `5.41706` ms.
  - 8 workers:
    - 1024 tokens: best `0.779663` ms.
    - 4096 tokens: best `2.85274` ms.
  - 12 workers:
    - 1024 tokens: best `0.629768` ms.
    - 4096 tokens: best `2.04253` ms.
  - 16 workers:
    - 1024 tokens: best `0.626672` ms.
    - 4096 tokens: best `1.83729` ms.
- Judgment:
  - Keep `K3_COPY_WORKERS=16` as the current corrected baseline.
  - Lower worker counts do not free enough compute overlap to compensate for slower tile copy consumption.
  - Next step: profile the corrected 16-worker path and compare it with pure K3 large to decide whether to target wait publication, copy traffic, or copy-worker resource residency.

### Milestone F: K3 Large Corrected Copy-Stage Profile And Pair-Copy Rejection
- **Status:** profile collected; pair-copy experiment rejected and reverted.
- **Date:** 2026-06-02
- Actions taken:
  - Ran HIP trace stats for pure K3 large 1024 and corrected 16-worker copy-stage 1024.
  - Ran PMC type-3 CSV for pure K3 large 1024 and corrected 16-worker copy-stage 1024.
  - Tried changing each copy-worker task from one contiguous `uint4` copy to two contiguous `uint4` copies to reduce row pointer loads and loop overhead.
  - Reverted the pair-copy experiment after correctness passed but timing regressed.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` during the experiment; reverted to the one-`uint4` copy baseline.
  - Planning/docs files.
- Trace evidence:
  - Pure 1024: HIPOPS showed 7 calls to the V2 large C kernel, average `445942` ns, 100% of kernel time.
  - Copy-stage 1024: HIPOPS showed 7 calls to the V2 large C kernel, average `626171` ns, 100% of kernel time.
  - No standalone combine kernel appeared in the traced timed path.
- PMC evidence, copy-stage versus pure at 1024:
  - `GRBM_COUNT`: `887440` vs `629525` (`1.41x`).
  - `SQ_INSTS_VMEM_RD`: `800087` vs `462848` (`1.73x`).
  - `SQ_INSTS_VMEM_WR`: `442880` vs `393216` (`1.13x`).
  - `TCP_TCP_TA_DATA_STALL_CYCLES` sum: `5586487` vs `3048266` (`1.83x`).
  - `TCC_EA_WRREQ_STALL` sum: `690990` vs `405197` (`1.71x`).
  - `SQ_WAIT_INST_LDS`: `1978715` vs `1975549` (`1.00x`), and `SQ_LDS_BANK_CONFLICT=0` for both.
  - PMC resource row: pure `sgpr=32`, copy-stage `sgpr=64`; both have `lds=65536`, `arch_vgpr=216`, `scr=0`.
- Pair-copy correctness:
  - 1024: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - 4096: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
- Pair-copy timing (`CHECK=1`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - 1024: rounds `0.761968/1283.75/0.64113` ms; unstable and slower than corrected baseline `0.626941` ms.
  - 4096: rounds `1.93974/1.94532/1.94341` ms; slower than corrected baseline `1.83131` ms.
- Judgment:
  - Rejected. Pair-copy reduces row pointer load frequency but worsens copy-stage scheduling/parallelism.
  - The main current evidence points to added copy-stage VMEM read traffic and TCP/TCC memory stalls, not LDS wait or bank conflict.

### Milestone F: K3 Large Self-Copy Epilogue Rejection
- **Status:** correctness failed; reverted to copy-worker baseline.
- **Date:** 2026-06-02
- Actions taken:
  - Tried replacing the non-tail-reduce copy-worker path with a self-copy epilogue: compute blocks wait for their own output stores, then copy their own `out` tile to row-combine targets with vector `uint4` stores.
  - Retained the existing copy-worker path for `k3_tail_reduce != 0`.
  - Tried adding device-scope `__threadfence()` before the self-copy, then retested 1024.
  - Reverted the experiment after correctness still failed.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` during the experiment; reverted to copy-worker baseline.
  - Planning/docs files.
- Correctness:
  - Self-copy without the extra device fence failed 1024 and 4096. Example 1024: `max_abs=0.112793`, `mean_abs=0.00718985`, `bit_mismatch=12582912`, `value_mismatch=12020011`.
  - Self-copy with device fence still failed 1024 with the same row-zero pattern.
- Timing:
  - Self-copy produced invalid 1024 timing around `0.573` ms and invalid 4096 timing around `1.897` ms. These are rejected because correctness failed.
- Restore validation:
  - After reverting, remote `aicc` build passed.
  - Restored copy-worker baseline smoke: 1024 real 8-rank / 8-device `CHECK=1`, rounds `0.627935/0.628727` ms, correctness `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
- Judgment:
  - Rejected. The compute block cannot simply read back its own global `out` tile and produce correct combine rows under the current buffer-store/output path.
  - Continue from the corrected copy-worker baseline.

### Milestone F: K3 Large Device-Scope Ready Fence Rejection After All-Thread Wait
- **Status:** correctness-clean in this run, but no stable performance gain; reverted to system-scope fence.
- **Date:** 2026-06-02
- Actions taken:
  - After the all-thread VMEM wait fix, tried changing only the tile-ready publication fence from `__threadfence_system()` to `__threadfence()`.
  - This differs from the earlier rejected device-fence experiment because the earlier path had thread0-only VMEM wait; this run kept the corrected all-thread wait.
  - Reverted after timing did not improve meaningfully.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` during the experiment; reverted to system fence.
  - Planning/docs files.
- Correctness:
  - 1024 real 8-rank / 8-device: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - 4096 real 8-rank / 8-device: `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
- Timing (`CHECK=1`, `WARMUP=10`, `REPEAT=30`, `MEASURE_ROUNDS=3`):
  - 1024: rounds `0.625844/0.625700/0.626228` ms, only about 1 us better than corrected system-fence baseline `0.626941` ms.
  - 4096: rounds `1.83775/1.83435/1.84104` ms, slightly worse than corrected system-fence baseline `1.83131` ms.
- Judgment:
  - Rejected. The tiny 1024 movement is noise-level and 4096 regressed.
  - Given the previous history of device-scope ready fencing being fragile, keep the safer system-scope fence until there is a larger verified win.

### Milestone F: K3 Large Sched-Barrier Ready-Publish Rejection
- **Status:** correctness-clean but no stable performance gain; reverted.
- **Date:** 2026-06-02
- Actions taken:
  - Tried adding `__builtin_amdgcn_sched_barrier(0)` before and after the all-thread VMEM wait in the corrected copy-stage ready-publish path.
  - Reverted after a repeated 4096 check showed no stable improvement.
- Files modified:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` during the experiment; reverted.
  - Planning/docs files.
- Correctness:
  - 1024 and 4096 real 8-rank / 8-device both passed with `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
- Timing:
  - First run:
    - 1024: rounds `0.628704/0.625696/0.625594` ms.
    - 4096: rounds `1.83063/1.83237/1.82992` ms.
  - Repeated 4096 run with 5 measure rounds: `1.83474/1.83572/1.83918/1.83547/1.83324` ms.
- Judgment:
  - Rejected. The initial 4096 best was noise; repeated measurement is not better than the corrected baseline `1.83131` ms.
  - Continue from the corrected system-fence copy-worker baseline or a separate direct remote-store route.

### Milestone F: K3 Large Sort And Direct Store-Policy Experiments
- **Status:** small copy-stage sort improvement kept; direct store-policy experiment rejected.
- **Date:** 2026-06-02
- Actions taken:
  - Re-read the task plan/progress/findings before continuing K3 large.
  - Synced the cleaned V2 source to remote and rebuilt with `aicc`.
  - Added `K3_SORT_ROWS` passthrough to `scripts/build_dcu_megamoe_v2.sh`; default remains `0`.
  - Rechecked direct row-combine sorted/unsorted and corrected copy-stage sorted/unsorted on real 8 ranks / 8 devices.
  - Queried the DCU KB for gfx938 store-policy guidance and created a temporary `global_store_short` probe under `hygon_tmp/dcu_megamoe_v2_probe`.
  - Tried `global_store_short ... glc slc` for direct rowptr BF16 stores, then reverted after a severe performance regression.
- Files modified:
  - `scripts/build_dcu_megamoe_v2.sh`
  - `docs/dcu_megamoe_v2_progress.md`
  - `docs/dcu_megamoe_v2_findings.md`
  - Temporary only: `hygon_tmp/dcu_megamoe_v2_probe/global_store_short_probe.cpp`
- Correctness:
  - Pure K3 1024/4096: max_abs 0.
  - Corrected copy-stage 1024/4096: max_abs 0.
  - Copy-stage + `--k3-sort-rows 1` 1024/4096: max_abs 0.
  - Direct sorted 1024/4096: max_abs 0.
  - Direct unsorted 4096 failed: `max_abs=0.0145264`, `bit_mismatch=191`, `value_mismatch=137`.
  - Direct sorted with `global_store_short ... glc slc` passed correctness but was rejected on timing.
- Timing:
  - Pure K3 large: 1024 best `0.444361` ms; 4096 best `1.30086` ms.
  - Corrected copy-stage: 1024 best `0.627143` ms (`+41.13%`); 4096 best `1.83152` ms (`+40.79%`).
  - Copy-stage + sort, CHECK=0 five-round bests: 1024 `0.624660` ms; 4096 `1.82607` ms.
  - Direct sorted: 1024 best `0.702680` ms; 4096 best `2.21777` ms.
  - Direct sorted with `global_store_short ... glc slc`: 1024 about `1.50064` ms; 4096 about `5.60513` ms.
- Profile evidence:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_copy_stage_sort4096`: `hipprof --hip-trace --stats` shows 7 calls to the same V2 large C fused kernel, average `1,831,358` ns, 100% HIPOPS kernel time; no standalone combine kernel.
- Issues:
  - One remote loop accidentally let PowerShell expand `$mode/$tok`, so it ran default build only; reran with escaped variables.
  - Remote source briefly had trailing NUL bytes after upload; local source had none. Fixed by truncating the remote file before scp and force-rebuilding.
- Judgment:
  - The temporary `K3_SORT_ROWS` script control was useful for this experiment, but was later removed during cleanup; copy-stage sorted routing is now the default active path.
  - Reject direct unsorted 4096 and `global_store_short glc/slc`.
  - Current best large K3 copy-stage+sort is still about `+40.4%` at 4096, so the next optimization needs to reduce copy traffic or copy-worker full-LDS residency rather than store cache policy.
## 2026-06-02 18:30 +08:00 - K3 Large Tail Reduce Optimization

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing Milestone F.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
- Implemented and verified three K3 large same-kernel tail-reduce optimizations:
  - precomputed local topk slot mask on host and passed it into the fused C pack5 kernel;
  - separate `d_k3_tail_out` buffer so zero-mask tokens can skip stores rather than writing a full zero row;
  - precomputed active token list so tail reduce only scans tokens with a local-rank contribution.
- Remote build:
  - Synced local source to `/home/hg/yuguo/DeepGEMM` with remote truncate before scp.
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed in the `megamoe` container.
  - `make -C csrc/kernels/dcu_megamoe_v2 hipcc` also passed; only the existing `mega_moe_dcu.cuh` missing-return warnings were emitted.
  - After cleanup, synced source/script again and re-ran both `aicc` and `hipcc`; both passed with the same existing header warnings.
- Correctness and timing, real 8 ranks / 8 visible devices:
  - Pure K3 C pack5 denominator, 1024 tokens: best `0.444784 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - Pure K3 C pack5 denominator, 4096 tokens: best `1.30267 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - Tail-reduce fused active-token version after cleanup, 1024 tokens: `0.636959/0.637543 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, degradation `+43.21%` versus pure.
  - Tail-reduce fused active-token version after cleanup, 4096 tokens: `1.88046/1.88045 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, degradation `+44.35%` versus pure.
- Profile evidence:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_tail_reduce_active4096`
  - hipprof shows 7 HIPOPS calls to the same `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128...` fused C pack5 kernel, average `1,905,278 ns`, 100% of HIPOPS kernel time.
  - No standalone combine or reduce kernel appears in the timed path.
- Judgment:
  - Tail reduce is now correctness-clean and substantially better than the earlier local-rank prototype (`~2.34-2.45 ms` at 4096 before these optimizations), but large-token K3 fused still misses the <=25% degradation target.
  - Active-token reduction is the meaningful tail-reduce win.
  - The tail-row copy list experiment was correctness-clean but only a marginal 4096-token win, so it was removed before commit to keep the active source simpler.
  - Next optimization should target the copy-stage/synchronization cost or a better in-kernel combine-reduce schedule; local-rank tail reduce alone is no longer the dominant remaining gap.

## 2026-06-03 09:39 +08:00 - Milestone G V2 Route Scratch Workspace Views

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing Milestone G.
- Files changed:
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/findings.md`
  - `.planning/dcu_megamoe_v2/progress.md`
- Implemented:
  - `V2WorkspaceViews` and `V2RuntimeState`.
  - DCU route scratch layout helpers matching `deep_gemm/include/deep_gemm/layout/mega_moe_dcu.cuh`.
  - Backend-specific padded launch rows: 64-row alignment for `ll`, 256-row alignment for `normal`.
  - Route scratch views for staged X, staged X scale, route weights, L1 output, K2 FP8/scale, K1/K3 metadata, grid barriers, row experts, row output pointers, local topk mask, and tail tokens.
  - V2 state cache on `sym_buffer._dcu_megamoe_v2_state`.
- Local validation:
  - `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - `git diff --check` passed.
  - Local Python direct runtime smoke passed; for `normal` with 8 ranks, 256 experts, max tokens 512, topk 6, hidden 512, intermediate 256, it produced `launch_rows=8192`, `valid_rows_per_expert=96`, and `rows_aligned_per_expert=256`.
  - Local `python -m pytest` could not run because Windows Python does not have `pytest`.
- Remote validation:
  - Synced V2 source/test/setup/planning files to `/home/hg/yuguo/DeepGEMM`, excluding `__pycache__` and `*.pyc`.
  - Remote command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py`.
  - Result: `17 passed in 13.32s`.
- Tokens tested:
  - N/A; this was workspace/interface work only.
- Pure groupgemm time:
  - N/A.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - N/A for numerical kernel output; static workspace tests verify dtype, shape, contiguity, cache reuse, and that all V2 execution views are sliced from `route_scratch`.
- Profile evidence:
  - N/A; no kernel timing path changed.
- Judgment:
  - Accepted as the first runtime integration sub-step.
  - Next item is V2 metadata/prebuild generation for `route_weights`, `row_expert/m_indices`, `output_index`, `row_combine_ptrs`, `local_topk_mask`, and `tail_tokens` without repeated per-iteration H2D.

## 2026-06-03 09:55 +08:00 - Milestone G K1 ll Metadata Writeback Plumbing

- Re-read the next Milestone G checklist item before continuing: V2 metadata/prebuild path.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_pybind.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/findings.md`
  - `.planning/dcu_megamoe_v2/progress.md`
- Implemented:
  - Added explicit V2 metadata views for `m_indices`, `row_combine_ptrs`, and `output_index`.
  - Extended the K1 ll fused C kernel with optional metadata outputs for `route_weights`, `row_expert/m_indices`, `output_index`, `row_combine_ptrs`, `local_topk_mask`, and dense `tail_tokens`.
  - Plumbed the new K1 ll metadata tensors through the raw launcher, pybind layer, and Python wrapper.
  - Kept standalone harness compatibility through default kernel parameters.
- Local validation:
  - `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py tests/test_dcu_megamoe_v2.py` passed.
  - `git diff --check` passed.
- Remote validation:
  - Synced V2 source/test/setup/planning files to `/home/hg/yuguo/DeepGEMM`.
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - `PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed.
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` passed with `17 passed in 14.20s`.
  - K1/K3 extension import smoke passed: `k1 True True`, `k3 True True`.
- Tokens tested:
  - N/A; no numerical K1 launch or performance timing was run in this sub-step.
- Pure groupgemm time:
  - N/A.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - N/A for numerical kernel output; this step validated build/import/source contracts only.
- Profile evidence:
  - N/A; no timed kernel path was profiled.
- Judgment:
  - Accepted as partial metadata/prebuild progress for the ll backend.
  - Metadata/prebuild parent item remains open until K1 ll metadata correctness is tested and K1 normal real-topk metadata generation is implemented.

### Follow-up Validation: K1 ll Metadata Correctness

- Added `test_v2_k1_ll_writes_route_metadata_from_sym_buffer`.
- The test constructs a minimal one-rank symmetric buffer layout on GPU, launches the V2 K1 ll fused kernel, and checks:
  - `route_weights[row]` equals source `topk_weights`;
  - `row_expert/m_indices[row]` equals local expert id;
  - `output_index[token, topk]` maps to the grouped row;
  - `row_combine_ptrs[row]` points at the expected combine partial row;
  - `local_topk_mask[token]` and dense `tail_tokens[token]` are populated.
- Remote validation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dcu_megamoe_v2.py` passed with `18 passed in 14.84s`.
- Judgment:
  - K1 ll metadata writeback is correctness-clean for the tested sym_buffer/topk/combine layout.
  - Normal backend metadata is still pending because the current normal row-stage still uses deterministic row-source mapping.

## 2026-06-03 10:24 +08:00 - Milestone G K1 normal TopK Metadata Writeback

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing Milestone G.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_pybind.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - K1 normal C row-stage no longer relies on the deterministic prototype route helper when `route_scratch_i32` is provided.
  - K1 normal now writes real topk-driven `route_weights`, `row_expert/m_indices`, `output_index`, `row_combine_ptrs`, `local_topk_mask`, and dense `tail_tokens` in the fused K1 compute kernel.
  - The normal metadata path avoids standalone metadata/prebuild kernels and avoids H2D/D2H.
  - The first implementation attempted a full-grid metadata barrier and then a 32-block metadata barrier; both are unsafe for this kernel shape because non-metadata compute blocks can occupy residency while waiting.
  - The accepted version makes each row tile / local expert build its own metadata from topk routes and publish its own row-stage epoch, so there is no cross-block metadata barrier.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote sync to `/home/hg/yuguo/DeepGEMM` passed.
  - Remote `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - Remote `PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed; K1 V2 object was force-rebuilt after deleting the stale object/so because the extension build did not track the included prototype C++ dependency.
  - Remote ll metadata test passed: `tests/test_dcu_megamoe_v2.py::test_v2_k1_writes_route_metadata_from_sym_buffer[ll-64]`.
  - Remote normal metadata test passed: `tests/test_dcu_megamoe_v2.py::test_v2_k1_writes_route_metadata_from_sym_buffer[normal-256]`.
  - Remote full V2 pytest passed: `19 passed in 14.21s`.
- Tokens tested:
  - Synthetic one-rank metadata test uses 32 tokens with topk=1 for both `ll` and `normal`.
- Pure groupgemm time:
  - N/A; this was metadata correctness, not a timed benchmark.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - Metadata correctness checks route weights, row expert/m_indices, output_index, row_combine_ptrs, local_topk_mask, and tail_tokens against the GPU sym_buffer layout.
- Profile evidence:
  - N/A; no timed performance profile was collected for this metadata sub-step.
- Judgment:
  - Accepted as the K1 normal real-topk metadata correctness step.
  - Keep dense `tail_tokens[:num_tokens]` for now because it avoids introducing a standalone compaction kernel before end-to-end correctness is connected.
  - Next item: call the K1 ll/normal extension launchers from `runtime.py`, then connect K2 and K3 staged calls.

## 2026-06-03 10:33 +08:00 - Milestone G Runtime K1 Launcher Connection

- Re-read the active Milestone G checklist after completing K1 normal metadata.
- Files changed:
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added a runtime K1 launcher hook that defaults to `megamoe.dcu_megamoe_v2.K1_fused.k1_fused.k1_dispatch_pull_l1_fused_v2`.
  - `run_stages_fused_v2` now creates/reuses V2 scratch views, initializes the cached grid barrier once on state creation, increments a per-state epoch, and calls K1 ll/normal through the V2 extension wrapper.
  - The runtime passes V2-owned scratch views for staged X, staged scales, route weights, row expert/m_indices, output_index, row_combine_ptrs, local_topk_mask, tail_tokens, grid_barrier, and route_scratch_i32.
  - Runtime still raises after K1 because K2/K3 are not connected yet.
  - Added a CPU-safe fake-launcher runtime test that verifies K1 receives the expected V2 scratch views and metadata, without launching a device kernel.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote sync to `/home/hg/yuguo/DeepGEMM` passed.
  - Remote full V2 pytest passed: `19 passed in 14.24s`.
- Tokens tested:
  - Existing K1 metadata tests still cover synthetic 32-token ll/normal cases.
- Pure groupgemm time:
  - N/A.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - Runtime K1 hook correctness is interface-level; numerical K1 metadata correctness remains covered by the ll/normal GPU metadata tests.
- Profile evidence:
  - N/A.
- Judgment:
  - Accepted as the runtime K1 connection sub-step.
  - Next item is to call the existing V2 K2 wrapper from `runtime.py` using K1 `route_weights`, `row_combine_ptrs`, and the padded launch row count.

## 2026-06-03 10:42 +08:00 - Milestone G Runtime K2 Wrapper Connection

- Re-read the active Milestone G checklist before connecting K2.
- Files changed:
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added a runtime K2 launcher hook that defaults to `megamoe.dcu_megamoe_v2.K2_fused.k2_fused.swiglu_quant_channelwise_out_v2`.
  - `run_stages_fused_v2` now calls K2 after K1, using `scratch.l1_out[:launch_rows]`, `route_weights[:launch_rows]`, `act_fp8[:launch_rows]`, `act_scale[:launch_rows]`, and `row_combine_ptrs[:launch_rows]`.
  - K2 uses `output_bf16=False` with `state.empty_bf16`, so no BF16 activation output allocation is added.
  - K2 `num_per_channels` is taken from `scratch.act_fp8.shape[1]` to match intermediate hidden, not L2 hidden.
  - Runtime still raises after K2 because K3 is not connected yet.
  - Extended the CPU-safe runtime test to fake both K1 and K2 and verify the scratch-view contract.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote sync passed.
  - Remote full V2 pytest passed: `19 passed in 13.25s`.
- Tokens tested:
  - Existing GPU metadata tests still cover synthetic 32-token K1 ll/normal.
- Pure groupgemm time:
  - N/A.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - Runtime K2 hook is interface-level; K2 numerical correctness remains covered by `test_v2_k2_swiglu_quant_reuse_matches_bf16_reference`.
- Profile evidence:
  - N/A.
- Judgment:
  - Accepted as the runtime K2 connection sub-step.
  - Next item is to call K3 ll/normal extension launchers from `runtime.py` using K2 act FP8/scale and K1 row-combine metadata.

## 2026-06-03 10:51 +08:00 - Milestone G Runtime K3 Launcher Connection

- Re-read K3 wrapper and pybind contracts before connecting K3.
- Files changed:
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added a runtime K3 launcher hook that defaults to `megamoe.dcu_megamoe_v2.K3_fused.k3_fused.k3_l2_combine_fused_v2`.
  - `run_stages_fused_v2` now calls K3 after K2 for both forced backends:
    - `ll`: K3 low-latency C pack5 rowptr + tail-reduce wrapper.
    - `normal`: K3 normal C pack5 copy-stage + tail-reduce wrapper.
  - K3 consumes K2 `act_fp8/act_scale`, K1 `row_output_ptrs`, `row_expert`, `local_topk_mask`, and dense `tail_tokens`.
  - K3 uses a separate runtime epoch after K1 to avoid stale `grid_barrier` values from the K1 normal row-stage being interpreted as K3 copy-stage ready flags.
  - `problem_size` for K3 ll is passed as `route_scratch_i32[:local_experts]`, i.e. the K1-produced per-expert counts.
  - Extended the CPU-safe runtime test to fake K1/K2/K3 and validate the full staged call contract.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote sync passed.
  - Remote full V2 pytest passed: `19 passed in 13.08s`.
- Tokens tested:
  - Existing GPU metadata tests still cover synthetic 32-token K1 ll/normal.
- Pure groupgemm time:
  - N/A.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - Runtime K3 hook is interface-level; no end-to-end numerical correctness is claimed yet.
- Profile evidence:
  - N/A.
- Judgment:
  - Accepted as the runtime K3 connection sub-step.
  - Next item is a minimal real-kernel runtime smoke for `ll` and `normal` to catch shape, epoch, and scratch contract issues before baseline correctness integration.

## 2026-06-03 11:03 +08:00 - Milestone G Real Runtime Smoke And Workspace Capacity Fix

- Re-read the next Milestone G item after connecting K3.
- Files changed:
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Made V2 route scratch sizing backend-aware.
  - Ensured forced `normal` at 32 tokens has enough padded workspace rows. Because the existing route layout's BF16 L1 region stores `intermediate_hidden` BF16 elements while L1 output is `2 * intermediate_hidden`, the layout capacity must be at least `2 * launch_rows`.
  - Added a normal-small workspace test that verifies `launch_rows=8192`, `capacity_rows >= 2 * launch_rows`, and L1/K2 views cover the padded launch rows.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Local helper check computed normal 32-token V2 scratch size as `170877136` bytes.
  - Remote full V2 pytest passed after the capacity fix: `20 passed in 13.99s`.
  - Remote temporary smoke script in `hygon_tmp/dcu_megamoe_v2/runtime_smoke.py` ran the real K1/K2/K3 runtime for one-rank synthetic 32-token zero-weight input:
    - `ll ok finite=True max=0.0`
    - `normal ok finite=True max=0.0`
- Tokens tested:
  - 32 tokens, 1 rank, topk=1, `ll` and `normal` runtime smoke.
- Pure groupgemm time:
  - N/A.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Correctness:
  - Smoke only checks successful execution and finite zero-weight output; it is not baseline numerical correctness.
- Profile evidence:
  - N/A.
- Judgment:
  - Accepted as the first real K1/K2/K3 runtime smoke.
  - Next item is to add real-flow stage/end-to-end correctness against the unchanged baseline at requested tokens, starting with a small controlled 32-token case.

## 2026-06-03 11:06 +08:00 - Phase 8 Real-Flow Correctness Harness

- Re-read `task_plan.md`, `progress.md`, and `findings.md` before starting Phase 8.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_pybind.cpp`
  - `megamoe/dcu_megamoe_v2/K3_fused/k3_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - K1 V2 ll/normal extension now receives uniform `num_tokens` and K1 fused route scanning uses it instead of requiring Python to write sym-buffer runtime token fields.
  - `run_stages_fused_v2` now defaults `num_tokens` to `y.size(0)` when `dispatch_num_tokens` is not supplied and validates it against `sym_buffer.num_max_tokens_per_rank`.
  - Added an env-gated real-flow pytest harness in `tests/test_dcu_megamoe_v2.py`:
    - `MEGAMOE_DCU_V2_REAL_FLOW=1`
    - `MEGAMOE_DCU_V2_REAL_FLOW_RANKS=4|8`
    - `MEGAMOE_DCU_V2_REAL_FLOW_TOKENS=...`
    - `MEGAMOE_DCU_V2_REAL_FLOW_BACKENDS=ll|normal`
  - The harness uses the same BF16 random input/weight style, FP8 input cast, DeepEP+DeepGEMM baseline helper, and V2-owned pack5 weight transform.
  - The first route mode is local-only per rank. This validates the real V2 call stack and numerical K1/K2/K3 path; it is not full cross-rank combine-reduce acceptance.
  - The harness expands `sym_buffer.route_scratch` for V2 only when the baseline allocation is too small, without changing baseline allocation code.
- Verification:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote sync passed.
  - Remote `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - Remote forced V2 K1/K3 extension rebuild through `PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed.
  - Remote default V2 pytest passed: `20 passed, 1 skipped in 13.29s`.
- Correctness results against unchanged DeepEP+DeepGEMM baseline:
  - 4 ranks, local-only, backend `ll`, 32 tokens: `max_abs=0.00049591064453125`, `mean_abs=3.37823512381874e-05`, `mismatch=0`.
  - 4 ranks, local-only, backend `normal`, 32 tokens: `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`.
  - 4 ranks, local-only, backend `ll`, 512 tokens: `max_abs=0.000698089599609375`, `mean_abs=2.4401795599260367e-05`, `mismatch=0`.
  - 4 ranks, local-only, backend `normal`, 512 tokens: `max_abs=0.00075531005859375`, `mean_abs=6.157375173643231e-05`, `mismatch=0`.
  - 4 ranks, local-only, backend `ll`, 1024 tokens: `max_abs=0.0006885528564453125`, `mean_abs=2.3948752641445026e-05`, `mismatch=0`.
  - 4 ranks, local-only, backend `ll`, 2050 tokens: `max_abs=0.00069427490234375`, `mean_abs=2.0935844077030197e-05`, `mismatch=0`.
  - 4 ranks, local-only, backend `normal`, 1024 tokens: failed, `max_abs=0.0062007904052734375`, `mean_abs=7.264173473231494e-05`, `mismatch=18300`.
- Tokens tested:
  - 32/512/1024/2050 for `ll`; 32/512/1024 for `normal`.
- Pure groupgemm time:
  - N/A; correctness harness only.
- Fused time:
  - N/A.
- Degradation ratio:
  - N/A.
- Profile evidence:
  - N/A; no timed profile was collected in this correctness sub-step.
- Judgment:
  - Accepted the real-flow harness and the passing local-only correctness points.
  - Normal backend 1024 failure must be isolated before claiming normal large-token real-flow correctness.
  - Next step: debug normal 1024 by isolating K1/K2/K3 normal versus ll/stage references.

## 2026-06-03 12:45 +08:00 - Phase 8 K1 Normal Waitcnt Fix And Real-Flow Correctness Completion

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented / accepted:
  - Accepted the K1 normal LDS-read waitcnt ablation: the normal C pack5 16xB128 LDS read helper now waits with `s_waitcnt lgkmcnt(0)` instead of the previous overlapped `lgkmcnt(6)` form.
  - The change is correctness-first and specifically addresses the observed non-deterministic oversized K1 L1 rows in the real-flow normal backend.
  - Earlier experiments that only serialized metadata clearing or switched to system fences did not remove the failure; those are retained as rejected/insufficient isolation steps.
- Remote validation:
  - `k1_normal_k3_ll`, 4 ranks, 1024 tokens, 5 consecutive runs passed.
    - Per-rank max_abs stayed around `1.9e-6` to `2.0e-6`.
    - Mean_abs stayed below `1e-9`.
    - Mismatch was `0`.
  - Formal real-flow backend `normal`, 4 ranks, 1024 tokens, 3 consecutive runs passed:
    - `max_abs=0.00083160400390625`
    - `mean_abs=6.15866738371551e-05`
    - `mismatch=0`
  - Formal real-flow backend `normal`, 4 ranks, 2050 tokens passed:
    - `max_abs=0.000812530517578125`
    - `mean_abs=6.164611841086298e-05`
    - `mismatch=0`
  - Full requested formal real-flow sweep passed with 4 ranks, local-only route, tokens `32,512,1024,2050`, backends `ll,normal`:
    - `ll`, 32: `max_abs=0.0005035400390625`, `mean_abs=3.3859167160699144e-05`, `mismatch=0`
    - `ll`, 512: `max_abs=0.0006866455078125`, `mean_abs=2.571809818618931e-05`, `mismatch=0`
    - `ll`, 1024: `max_abs=0.0007495880126953125`, `mean_abs=2.4467892217217013e-05`, `mismatch=0`
    - `ll`, 2050: `max_abs=0.0007381439208984375`, `mean_abs=2.381613739999011e-05`, `mismatch=0`
    - `normal`, 32: `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`
    - `normal`, 512: `max_abs=0.00075531005859375`, `mean_abs=6.150374247226864e-05`, `mismatch=0`
    - `normal`, 1024: `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`
    - `normal`, 2050: `max_abs=0.000812530517578125`, `mean_abs=6.164611841086298e-05`, `mismatch=0`
- Tokens tested:
  - 1024 for K1 normal isolation, 5 repeated 4-rank runs.
  - 1024 for formal normal, 3 repeated 4-rank runs.
  - 2050 for formal normal, 1 4-rank run.
  - Full formal requested local-only sweep: 32/512/1024/2050 for `ll` and `normal`.
- Pure groupgemm time:
  - N/A; correctness-only real-flow validation.
- Fused time:
  - N/A; correctness-only real-flow validation.
- Degradation ratio:
  - N/A; correctness-only real-flow validation.
- Profile evidence:
  - N/A in this checkpoint. The conclusion is based on deterministic correctness isolation, not timing/profile.
- Judgment:
  - Accepted for Phase 8 requested real-flow end-to-end correctness.
  - The current pass is 4-rank local-only routing; full cross-rank combine-reduce/stat alignment remains a later integration item.
  - Next item: re-read the plan and continue with stage-level correctness references and integrated quick performance at 32/128/1024/4096.

## 2026-06-03 13:02 +08:00 - Phase 8 Integrated Quick Performance Harness And First Results

- Re-read the active Phase 8 items after completing requested end-to-end correctness.
- Files changed:
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added env-gated integrated performance test `test_v2_real_flow_integrated_performance_against_deepep_deepgemm_baseline`.
  - New controls:
    - `MEGAMOE_DCU_V2_REAL_FLOW_PERF=1`
    - `MEGAMOE_DCU_V2_REAL_FLOW_PERF_RANKS`
    - `MEGAMOE_DCU_V2_REAL_FLOW_PERF_TOKENS`
    - `MEGAMOE_DCU_V2_REAL_FLOW_PERF_BACKENDS`
    - `MEGAMOE_DCU_V2_REAL_FLOW_PERF_WARMUP`
    - `MEGAMOE_DCU_V2_REAL_FLOW_PERF_REPEAT`
  - The timing uses HIP events and reports rank-max median/min/max across ranks.
  - V2 timing covers only the staged V2 call. Baseline timing covers `run_deepgemm_megamoe_baseline` with cached DeepEP layout.
  - Layout transforms, random input generation, route_scratch expansion, and weight preparation are outside timing.
- Verification:
  - Local `python -m py_compile tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote perf smoke passed for `ll`, 4 ranks, 32 tokens, warmup=1/repeat=2.
  - Remote integrated quick performance passed for 4 ranks, local-only route, tokens `32,128,1024,4096`, backends `ll,normal`, warmup=3/repeat=5.
- Correctness during performance timing:
  - All measured cases had `mismatch=0`.
  - All measured cases had `max_abs <= 1e-3`.
- Integrated quick performance, rank-max median:
  - `ll`, 32: V2 `0.6668800115585327 ms`, baseline e2e `2.296478033065796 ms`, degradation vs baseline e2e `-70.96%`, `max_abs=0.00049591064453125`.
  - `ll`, 128: V2 `0.6860790252685547 ms`, baseline e2e `2.299837112426758 ms`, degradation vs baseline e2e `-70.17%`, `max_abs=0.000751495361328125`.
  - `ll`, 1024: V2 `2.676637887954712 ms`, baseline e2e `2.8078370094299316 ms`, degradation vs baseline e2e `-4.67%`, `max_abs=0.0006618499755859375`.
  - `ll`, 4096: V2 `11.191666603088379 ms`, baseline e2e `6.714233875274658 ms`, degradation vs baseline e2e `+66.69%`, `max_abs=0.0006885528564453125`.
  - `normal`, 32: V2 `17.107980728149414 ms`, baseline e2e `2.515038013458252 ms`, degradation vs baseline e2e `+580.23%`, `max_abs=0.0006389617919921875`.
  - `normal`, 128: V2 `20.457258224487305 ms`, baseline e2e `2.5651180744171143 ms`, degradation vs baseline e2e `+697.52%`, `max_abs=0.00074005126953125`.
  - `normal`, 1024: V2 `81.61079406738281 ms`, baseline e2e `2.864957094192505 ms`, degradation vs baseline e2e `+2748.59%`, `max_abs=0.00083160400390625`.
  - `normal`, 4096: V2 `1138.2191162109375 ms`, baseline e2e `12.349268913269043 ms`, degradation vs baseline e2e `+9116.89%`, `max_abs=0.000835418701171875`.
- Pure groupgemm time:
  - Not measured in this harness. Prototype-kernel comparison remains pending.
- Fused time:
  - Integrated staged V2 times listed above.
- Profile evidence:
  - Not collected in this checkpoint. This was HIP event timing only.
- Judgment:
  - Perf harness is accepted as an integrated timing tool.
  - The `normal` backend timings are pathological and cannot be used for threshold selection.
  - Next item: add or run stage breakdown for `normal` to isolate whether K1 metadata/row-stage, K2 dense padded rows, or K3 normal copy-stage dominates.

## 2026-06-03 13:28 +08:00 - Phase 8 Normal Backend K1 Stage Breakdown And Metadata Optimization Draft

- Re-read Phase 8 before continuing from the integrated quick performance anomaly.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/api.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added optional `profile_stages` support to `run_stages_fused_v2` and the public V2 API.
  - Added `MEGAMOE_DCU_V2_REAL_FLOW_PERF_STAGE_BREAKDOWN=1` support to the integrated performance test.
  - The stage profiler wraps K1/K2/K3 launchers with HIP events only when explicitly enabled; default V2 runtime path is unchanged.
  - Changed K1 normal real-flow metadata from single-thread route scan to block-parallel route scan with `atomicAdd` row allocation.
  - Restricted K1 normal metadata construction to the first row tile of each expert and made that owner block publish the row-stage epoch for every row tile belonging to that expert.
- Remote evidence before the K1 metadata optimization:
  - `normal`, 1024 tokens, 4 ranks, warmup=1/repeat=3:
    - total V2 rank-max median `81.6631088256836 ms`
    - K1 `80.19960021972656 ms`
    - K2 `0.14032000303268433 ms`
    - K3 `1.0281590223312378 ms`
  - `normal`, 4096 tokens, 4 ranks, warmup=1/repeat=3:
    - total V2 rank-max median `1129.4111328125 ms`
    - K1 `1127.633056640625 ms`
    - K2 `0.38784000277519226 ms`
    - K3 `3.551356077194214 ms`
- Build / validation:
  - Local `python -m py_compile megamoe/dcu_megamoe_v2/runtime.py megamoe/dcu_megamoe_v2/api.py tests/test_dcu_megamoe_v2.py` passed.
  - Local `git diff --check` passed.
  - Remote `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed after the K1 metadata optimization.
  - Remote forced `DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed after deleting V2 K1/K3 build objects and `.so` files.
  - Remote `test_v2_k1_writes_route_metadata_from_sym_buffer` passed for both `ll` and `normal`: `2 passed in 13.74s`.
- Blocker:
  - 4-rank real-flow correctness/performance could not be rerun after the optimization because all 8 DCUs were occupied by an external sglang/model workload.
  - `hy-smi` showed VRAM `97-99%` on cards 0-7.
  - The attempted 4-rank normal 1024 correctness run failed before V2 execution with DeepEP `out of memory` during `deep_ep.Buffer` construction.
- Tokens tested:
  - Stage breakdown before optimization: 1024/4096, backend `normal`, 4 ranks.
  - Metadata unit test after optimization: synthetic 32-token `ll` and `normal`.
- Pure groupgemm time:
  - Not measured.
- Fused time:
  - Pre-optimization stage breakdown listed above.
- Degradation ratio:
  - Not recomputed after optimization because real-flow rerun is blocked by occupied cards.
- Profile evidence:
  - HIP event stage breakdown showed K1 normal dominates the pathological integrated timing.
- Judgment:
  - Stage profiler is accepted.
  - The K1 metadata optimization is built and metadata-unit-clean, but not yet accepted for real-flow correctness/performance.
  - Next action when cards are available: run 4-rank `normal` 1024 correctness, then `normal` 1024/4096 stage breakdown and quick perf.

## 2026-06-03 14:18 +08:00 - Phase 8 K1 Normal Metadata Optimization Accepted

- Re-read the active Phase 8 items after cards became available again.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented / accepted:
  - Accepted the K1 normal metadata owner-block optimization in the real-flow integration.
  - Metadata build is now parallelized inside the first row tile of each local expert and uses `atomicAdd` for row assignment.
  - The owner block publishes the row-stage epoch for all row tiles of that expert, avoiding repeated full route scans on later row tiles.
  - The optimization stays inside the K1 fused compute kernel and does not add a standalone prebuild kernel, repeated H2D control update, or external rank barrier.
- Remote build / validation:
  - `git diff --check` passed.
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - Forced V2 K1/K3 extension rebuild through `PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed after deleting stale V2 build objects and `.so` files.
  - `test_v2_k1_writes_route_metadata_from_sym_buffer` passed for both `ll` and `normal`: `2 passed in 13.74s`.
- Correctness after optimization:
  - 4-rank local-only real-flow, backend `normal`, 1024 tokens passed:
    - `max_abs=0.00083160400390625`
    - `mean_abs=6.15866738371551e-05`
    - `mismatch=0`
  - 4-rank local-only real-flow, backend `normal`, requested tokens `32,512,1024,2050` passed:
    - 32: `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`
    - 512: `max_abs=0.00075531005859375`, `mean_abs=6.150374247226864e-05`, `mismatch=0`
    - 1024: `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`
    - 2050: `max_abs=0.000812530517578125`, `mean_abs=6.164611841086298e-05`, `mismatch=0`
- Stage breakdown after optimization, 4-rank local-only, backend `normal`, warmup=3/repeat=5:
  - 1024 tokens:
    - total V2 `2.4659149646759033 ms`
    - baseline e2e `2.8075170516967773 ms`
    - degradation vs baseline e2e `-12.17%`
    - K1 `1.3508789539337158 ms`
    - K2 `0.10735999792814255 ms`
    - K3 `1.0299190282821655 ms`
    - correctness `max_abs=0.00083160400390625`, `mismatch=0`
  - 4096 tokens:
    - total V2 `10.801589012145996 ms`
    - baseline e2e `6.675033092498779 ms`
    - degradation vs baseline e2e `+61.82%`
    - K1 `7.15823221206665 ms`
    - K2 `0.2542400062084198 ms`
    - K3 `3.5473570823669434 ms`
    - correctness `max_abs=0.000835418701171875`, `mismatch=0`
- Updated integrated quick performance after optimization, 4-rank local-only, warmup=3/repeat=5:
  - `ll`, 32: V2 `0.6806390285491943 ms`, baseline e2e `2.493597984313965 ms`, degradation `-72.70%`, `max_abs=0.00049591064453125`, `mismatch=0`
  - `ll`, 128: V2 `0.7094389796257019 ms`, baseline e2e `2.504957914352417 ms`, degradation `-71.68%`, `max_abs=0.000690460205078125`, `mismatch=0`
  - `ll`, 1024: V2 `2.6755170822143555 ms`, baseline e2e `2.826237916946411 ms`, degradation `-5.33%`, `max_abs=0.000804901123046875`, `mismatch=0`
  - `ll`, 4096: V2 `11.154061317443848 ms`, baseline e2e `6.653112888336182 ms`, degradation `+67.65%`, `max_abs=0.000698089599609375`, `mismatch=0`
  - `normal`, 32: V2 `1.6804779767990112 ms`, baseline e2e `2.457437038421631 ms`, degradation `-31.62%`, `max_abs=0.0006389617919921875`, `mismatch=0`
  - `normal`, 128: V2 `1.7204780578613281 ms`, baseline e2e `2.566396951675415 ms`, degradation `-32.96%`, `max_abs=0.00074005126953125`, `mismatch=0`
  - `normal`, 1024: V2 `2.4735970497131348 ms`, baseline e2e `2.838076114654541 ms`, degradation `-12.84%`, `max_abs=0.00083160400390625`, `mismatch=0`
  - `normal`, 4096: V2 `10.875988960266113 ms`, baseline e2e `6.751031875610352 ms`, degradation `+61.10%`, `max_abs=0.000835418701171875`, `mismatch=0`
- Tokens tested:
  - Correctness: 32/512/1024/2050 for backend `normal`, 4 ranks.
  - Performance: 32/128/1024/4096 for backends `ll` and `normal`, 4 ranks.
- Pure groupgemm time:
  - Not remeasured in this integrated harness. Prototype pure/fused comparison remains a separate pending Phase 8 item.
- Fused time:
  - Integrated staged V2 timings listed above.
- Degradation ratio:
  - Listed above versus unchanged DeepEP+DeepGEMM baseline e2e. Prototype-kernel degradation is still pending.
- Profile evidence:
  - HIP event stage breakdown shows K1 normal dropped from about `80.20 ms` to `1.35 ms` at 1024 and from about `1127.63 ms` to `7.16 ms` at 4096.
- Judgment:
  - Accept the owner-block K1 normal metadata optimization.
  - Normal backend is now usable for integrated timing; 4096 remains slower mainly from K1 normal plus K3 normal copy-stage.
  - Next item: re-read Phase 8 and collect stage-level correctness/prototype-kernel comparison before using these integrated numbers for threshold or final performance claims.

## 2026-06-03 14:55 +08:00 - Phase 8 Prototype Timing Comparison And Stage Breakdown

- Re-read Phase 8 before running prototype comparisons.
- Remote environment:
  - HCU0 was occupied by an unrelated `torchrun bench/test_low_latency.py`; all prototype and integrated follow-up runs used `HIP_VISIBLE_DEVICES=1,2,3,4` for 4-device communication tests or `HIP_VISIBLE_DEVICES=1` for pure single-device prototype tests.
  - An initial prototype batch command had a PowerShell quoting error that expanded `TOKENS=$t` to an empty string. The resulting leftover process was terminated; no measurements from that failed command are used.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Prototype timing method:
  - Standalone V2 harness, warmup=3/repeat=5/measure_rounds=3.
  - Pure tests used one free HCU.
  - Fused communication tests used 4 ranks over 4 visible HCUs.
  - Reported times below use the median of the three printed measure rounds.
- K1 prototype pure vs fused:
  - `ll`, 32: pure `0.301280 ms`, fused `0.349663 ms`, degradation `+16.06%`, correctness PASS, max_abs <= 0.001.
  - `ll`, 128: pure `0.308607 ms`, fused `0.359680 ms`, degradation `+16.55%`, correctness PASS, max_abs <= 0.001.
  - `ll`, 1024 forced path: pure `1.266300 ms`, fused `1.404510 ms`, degradation `+10.91%`, correctness PASS, max_abs <= 0.001.
  - `ll`, 4096 forced path: pure `4.981220 ms`, fused `5.896190 ms`, degradation `+18.37%`, correctness PASS, max_abs <= 0.001.
  - `normal`, 32 forced path: pure `0.715647 ms`, fused `0.751902 ms`, degradation `+5.07%`, correctness PASS, max_abs 0.
  - `normal`, 128 forced path: pure `0.737088 ms`, fused `0.778015 ms`, degradation `+5.55%`, correctness PASS, max_abs 0.
  - `normal`, 1024: pure `0.874271 ms`, fused `0.940160 ms`, degradation `+7.54%`, correctness PASS, max_abs 0.
  - `normal`, 4096: pure `2.358560 ms`, fused `2.657950 ms`, degradation `+12.69%`, correctness PASS, max_abs 0.
- K3 prototype pure vs fused:
  - `ll`, 32: pure `0.157472 ms`, fused tail-reduce `0.166335 ms`, degradation `+5.63%`, correctness PASS, max_abs `0.000488281`.
  - `ll`, 128: pure `0.165632 ms`, fused tail-reduce `0.182272 ms`, degradation `+10.05%`, correctness PASS, max_abs `0.000488281`.
  - `ll`, 1024 forced path: pure `0.696578 ms`, fused tail-reduce `0.822175 ms`, degradation `+18.03%`, correctness PASS, max_abs `0.000488281`.
  - `ll`, 4096 forced path: pure `2.723560 ms`, fused tail-reduce `3.225530 ms`, degradation `+18.43%`, correctness PASS, max_abs `0.000488281`.
  - `normal`, 32 forced path: pure `0.390816 ms`, fused copy-stage tail-reduce `0.509280 ms`, degradation `+30.31%`, correctness PASS, max_abs 0.
  - `normal`, 128 forced path: pure `0.418561 ms`, fused copy-stage tail-reduce `0.560512 ms`, degradation `+33.91%`, correctness PASS, max_abs 0.
  - `normal`, 1024: pure `0.467296 ms`, fused copy-stage tail-reduce `0.706751 ms`, degradation `+51.24%`, correctness PASS, max_abs 0.
  - `normal`, 4096: pure `1.339000 ms`, fused copy-stage tail-reduce `2.091710 ms`, degradation `+56.21%`, correctness PASS, max_abs 0.
- Integrated stage breakdown, 4-rank local-only, warmup=3/repeat=5:
  - `ll`, 32: total `0.679040 ms`, baseline e2e `2.502397 ms`, K1 `0.404320 ms`, K2 `0.073600 ms`, K3 `0.254080 ms`, correctness `max_abs=0.00049591064453125`, `mean_abs=1.7092428606702015e-05`, `mismatch=0`.
  - `ll`, 128: total `0.706559 ms`, baseline e2e `2.560157 ms`, K1 `0.403199 ms`, K2 `0.074880 ms`, K3 `0.280640 ms`, correctness `max_abs=0.0008296966552734375`, `mean_abs=2.617212521727197e-05`, `mismatch=0`.
  - `ll`, 1024: total `2.652637 ms`, baseline e2e `2.854237 ms`, K1 `1.435519 ms`, K2 `0.101599 ms`, K3 `1.186719 ms`, correctness `max_abs=0.0006580352783203125`, `mean_abs=2.4049910280155018e-05`, `mismatch=0`.
  - `ll`, 4096: total `11.190388 ms`, baseline e2e `6.652152 ms`, K1 `5.747352 ms`, K2 `0.264960 ms`, K3 `5.396954 ms`, correctness `max_abs=0.000682830810546875`, `mean_abs=2.214050618931651e-05`, `mismatch=0`.
  - `normal`, 32: total `1.673439 ms`, baseline e2e `2.520156 ms`, K1 `0.987040 ms`, K2 `0.084960 ms`, K3 `0.651039 ms`, correctness `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`.
  - `normal`, 128: total `1.717598 ms`, baseline e2e `2.528797 ms`, K1 `1.009439 ms`, K2 `0.086240 ms`, K3 `0.674399 ms`, correctness `max_abs=0.00074005126953125`, `mean_abs=6.15057215327397e-05`, `mismatch=0`.
  - `normal`, 1024: total `2.471677 ms`, baseline e2e `2.817278 ms`, K1 `1.345918 ms`, K2 `0.104800 ms`, K3 `1.027679 ms`, correctness `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`.
  - `normal`, 4096: total `10.788950 ms`, baseline e2e `6.715353 ms`, K1 `7.095993 ms`, K2 `0.251520 ms`, K3 `3.545436 ms`, correctness `max_abs=0.000835418701171875`, `mean_abs=6.162457430036739e-05`, `mismatch=0`.
- Integrated stage time vs standalone fused prototype:
  - K1 `ll`: overhead is `+15.63%` at 32, `+12.10%` at 128, `+2.21%` at 1024, and `-2.52%` at 4096.
  - K1 `normal`: overhead is `+31.27%` at 32, `+29.75%` at 128, `+43.16%` at 1024, and `+166.97%` at 4096.
  - K3 `ll`: overhead is `+52.75%` at 32, `+53.97%` at 128, `+44.34%` at 1024, and `+67.32%` at 4096.
  - K3 `normal`: overhead is `+27.84%` at 32, `+20.32%` at 128, `+45.41%` at 1024, and `+69.50%` at 4096.
- Judgment:
  - Prototype K1 fused communication is within the <=20% target for all forced-backend quick sizes in this run.
  - Prototype K3 `ll` fused tail-reduce is within the <=25% target for all quick sizes in this run.
  - Prototype K3 `normal` copy-stage tail-reduce is still above target for 32/128/1024/4096.
  - Integrated K1 `normal` at 4096 has large real-flow overhead from topk-driven metadata, despite the owner-block improvement.
  - Integrated K3 has significant overhead versus standalone fused prototype for both backends; this is the next performance-relevant integration gap after stage-level correctness/stat alignment.

## 2026-06-03 15:05 +08:00 - Phase 8 Cumulative Stats Alignment

- Re-read Phase 8 before continuing; active gap was real-flow stats alignment plus stage-level references.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_pybind.cpp`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added optional `cumulative_local_expert_recv_stats` pointer to K1 ll and K1 normal C pack5 fused kernels.
  - K1 ll writes stats after finalized `symm_counts` are clamped.
  - K1 normal writes stats from the per-expert metadata owner block after finalized `symm_counts` are clamped.
  - Python wrapper passes an empty CUDA int32 slice when stats is `None`, so the perf path keeps a null stats pointer.
  - Real-flow correctness now compares V2 cumulative stats against baseline stats, matching `tests/test_mega_moe_dcu.py` semantics.
- Local validation:
  - `python -m py_compile megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - `git diff --check` passed.
- Remote environment:
  - HCU0 was busy; validation used HCU1-4 for 4-rank real-flow and HCU1 for single-device tests.
- Remote build / validation:
  - `git diff --check` passed.
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - Forced V2 extension rebuild through `PYTORCH_ROCM_ARCH=gfx938 MAX_JOBS=2 DG_FORCE_BUILD=1 python3 setup.py build_ext --inplace` passed.
  - Single-device K1 metadata/stats test passed for both backends: `2 passed in 13.26s`.
  - Full non-distributed V2 tests passed: `20 passed, 2 skipped in 13.46s`.
- 4-rank local-only real-flow correctness with stats, `HIP_VISIBLE_DEVICES=1,2,3,4`:
  - `ll`, 32: `max_abs=0.00049591064453125`, `mean_abs=3.37823512381874e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 512: `max_abs=0.0006957054138183594`, `mean_abs=2.5930632546078414e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 1024: `max_abs=0.0007266998291015625`, `mean_abs=2.3697797587374225e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 2050: `max_abs=0.0006694793701171875`, `mean_abs=2.3335776859312318e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 32: `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 512: `max_abs=0.00075531005859375`, `mean_abs=6.150374247226864e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 1024: `max_abs=0.00083160400390625`, `mean_abs=6.15866738371551e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 2050: `max_abs=0.000812530517578125`, `mean_abs=6.164611841086298e-05`, `mismatch=0`, `stats_ok=True`.
- Pure groupgemm time / fused time / degradation:
  - Not remeasured in this stats-alignment checkpoint. Existing Phase 8 prototype-vs-integrated timing remains the current performance reference.
- Profile evidence:
  - Not collected in this checkpoint. The change is disabled on the perf path when stats is `None`; future perf/profile runs should keep stats disabled unless validating cumulative stats overhead.
- Judgment:
  - Accept the K1 cumulative stats alignment.
  - This closes the real-flow output+stats call-stack gap for local-only 4-rank requested tokens.
  - Next item after re-reading Phase 8: add separate stage-level correctness references for K1, K2, and K3 where the reference is reliable, then revisit integrated K3 overhead.

## 2026-06-03 15:35 +08:00 - Phase 8 Available Stage Reference Metrics

- Re-read Phase 8 after closing stats alignment; active gap was stage-reference reporting.
- Files changed:
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implemented:
  - Added 32-token real-flow stage metric collection inside the existing real-flow correctness worker.
  - K2 stage reference compares optimized K2 BF16-output mode with the Torch SwiGLU reference on active V2 rows.
  - K3 stage reference runs baseline DeepGEMM L2 on V2 K2 activations in V2 row order, then reduces through V2 `output_index` and compares to final V2 `y`.
  - K1 direct L1 reference is retained only as a diagnostic metric because the current baseline-DeepGEMM row reference is not reliable.
- Local validation:
  - `python -m py_compile tests/test_dcu_megamoe_v2.py` passed.
  - `git diff --check` passed.
- Remote validation:
  - 4-rank 32-token real-flow stage metrics passed for `ll` and `normal`: `1 passed in 52.15s`.
  - Full requested 4-rank local-only correctness with stats and stage metrics passed for `ll` and `normal` at tokens `32,512,1024,2050`: `1 passed in 171.62s`.
- Stage correctness metrics at 32 tokens:
  - `ll`:
    - end-to-end `max_abs=0.000461578369140625`, `mean_abs=1.68610113178147e-05`, `mismatch=0`, `stats_ok=True`.
    - K2 `max_abs=9.5367431640625e-07`, `mean_abs=3.547029423650594e-12`, `mismatch=0`.
    - K3 `max_abs=0.000553131103515625`, `mean_abs=9.541524923406541e-05`, `mismatch=0`.
    - K1 L1 diagnostic-only: `max_abs=0.16796875`, `mean_abs=0.014402168802917004`, `mismatch=320296`.
  - `normal`:
    - end-to-end `max_abs=0.0006389617919921875`, `mean_abs=6.178580224514008e-05`, `mismatch=0`, `stats_ok=True`.
    - K2 `max_abs=0.0`, `mean_abs=0.0`, `mismatch=0`.
    - K3 `max_abs=0.0006389617919921875`, `mean_abs=6.189954001456499e-05`, `mismatch=0`.
    - K1 L1 diagnostic-only: `max_abs=0.12344837188720703`, `mean_abs=0.020348047837615013`, `mismatch=761953`.
- Correctness tokens tested:
  - `32,512,1024,2050` for both `ll` and `normal`, 4 ranks on `HIP_VISIBLE_DEVICES=1,2,3,4`.
- Pure groupgemm time / fused time / degradation:
  - Not remeasured in this stage-reference checkpoint.
- Profile evidence:
  - Not collected in this checkpoint.
- Judgment:
  - Accept available stage-reference reporting for Phase 8: K1 metadata/stats, K2 numerical, K3 numerical, and end-to-end all pass.
  - Do not use the direct K1 L1 diagnostic mismatch as a failure until the reference construction is narrowed; current evidence points to reference mismatch rather than an execution failure.
  - Next item after re-reading plan: either run a quick integrated perf sanity check after the stats/stage test changes, or proceed to the known K3/normal integration overhead work.

## 2026-06-03 16:25 +08:00 - Phase 9 Real-Flow Full Token Sweep

- Re-read `.planning/dcu_megamoe_v2/task_plan.md`, `progress.md`, and `findings.md` before continuing Phase 9.
- Remote full sweep completed successfully:
  - log: `hygon_tmp/dcu_megamoe_v2/real_flow_full_sweep_20260603_141843.log`
  - exit code: 0
  - pytest result: `1 passed in 368.91s`
  - devices/ranks: `HIP_VISIBLE_DEVICES=0,1,2,3`, 4 ranks
  - route mode: `local_only`
  - warmup/repeat: 3/5
  - backends: `ll`, `normal`
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Tokens tested:
  - 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 for both `ll` and `normal`.
- Correctness:
  - All rows had `mismatch=0`.
  - Worst `ll` max_abs: `0.0007801055908203125` at 8192 tokens.
  - Worst `normal` max_abs: `0.00087738037109375` at 8192 tokens.
  - All results are under `max_abs <= 1e-3`.
- Integrated V2 staged rank-max median vs unchanged baseline e2e:
  - `ll`, 32: V2 `0.690559 ms`, baseline `2.498238 ms`, degradation `-72.36%`, max_abs `0.00049591064453125`, mismatch `0`.
  - `ll`, 64: V2 `0.699679 ms`, baseline `2.563997 ms`, degradation `-72.71%`, max_abs `0.000507354736328125`, mismatch `0`.
  - `ll`, 128: V2 `0.716319 ms`, baseline `2.514237 ms`, degradation `-71.51%`, max_abs `0.00049591064453125`, mismatch `0`.
  - `ll`, 256: V2 `0.940319 ms`, baseline `2.638237 ms`, degradation `-64.36%`, max_abs `0.0007152557373046875`, mismatch `0`.
  - `ll`, 512: V2 `1.483838 ms`, baseline `2.654877 ms`, degradation `-44.11%`, max_abs `0.0007038116455078125`, mismatch `0`.
  - `ll`, 1024: V2 `2.720157 ms`, baseline `2.848477 ms`, degradation `-4.50%`, max_abs `0.0007476806640625`, mismatch `0`.
  - `ll`, 2048: V2 `5.418393 ms`, baseline `4.291355 ms`, degradation `+26.26%`, max_abs `0.0007257461547851562`, mismatch `0`.
  - `ll`, 4096: V2 `11.249908 ms`, baseline `6.592633 ms`, degradation `+70.64%`, max_abs `0.00077056884765625`, mismatch `0`.
  - `ll`, 8192: V2 `22.648611 ms`, baseline `12.176468 ms`, degradation `+86.00%`, max_abs `0.0007801055908203125`, mismatch `0`.
  - `normal`, 32: V2 `1.690878 ms`, baseline `2.517437 ms`, degradation `-32.83%`, max_abs `0.0006389617919921875`, mismatch `0`.
  - `normal`, 64: V2 `1.699518 ms`, baseline `2.546078 ms`, degradation `-33.25%`, max_abs `0.000732421875`, mismatch `0`.
  - `normal`, 128: V2 `1.727198 ms`, baseline `2.496638 ms`, degradation `-30.82%`, max_abs `0.00074005126953125`, mismatch `0`.
  - `normal`, 256: V2 `1.773597 ms`, baseline `2.666718 ms`, degradation `-33.49%`, max_abs `0.0007724761962890625`, mismatch `0`.
  - `normal`, 512: V2 `2.121918 ms`, baseline `2.631037 ms`, degradation `-19.35%`, max_abs `0.00075531005859375`, mismatch `0`.
  - `normal`, 1024: V2 `2.476957 ms`, baseline `2.819997 ms`, degradation `-12.16%`, max_abs `0.00083160400390625`, mismatch `0`.
  - `normal`, 2048: V2 `4.995835 ms`, baseline `4.257914 ms`, degradation `+17.33%`, max_abs `0.000812530517578125`, mismatch `0`.
  - `normal`, 4096: V2 `10.866069 ms`, baseline `6.748951 ms`, degradation `+61.00%`, max_abs `0.000835418701171875`, mismatch `0`.
  - `normal`, 8192: V2 `21.458380 ms`, baseline `11.821106 ms`, degradation `+81.53%`, max_abs `0.00087738037109375`, mismatch `0`.
- Pure groupgemm time:
  - Not remeasured in this integrated full sweep. The current pure/fused prototype references remain the Phase 8 prototype comparison numbers.
- Fused time:
  - Integrated staged V2 times listed above.
- Profile evidence:
  - This checkpoint uses HIP-event rank-max timing from the integrated harness, not hipprof. Existing K1/K3 prototype hipprof evidence remains the communication-fused kernel evidence; full integrated hipprof is still pending for final acceptance.
- Judgment:
  - Full sweep correctness and integrated timing are accepted as Phase 9 evidence.
  - Backend advantage in this local-only integrated sweep is `ll` for 32-512 tokens and `normal` for 1024-8192 tokens.
  - Do not implement the automatic threshold yet. The current evidence is local-only routing and K3 large still has known performance/acceptance caveats.
  - Next item after re-reading the plan: plan or implement uneven tokens per rank, then CUDA graph one-graph multi-size support, while keeping K3 large performance caveat visible.

## 2026-06-03 16:45 +08:00 - Phase 9 Uneven Token Local-Only Correctness

- User clarified before the next run:
  - check and clean residual remote processes before running;
  - after real communication is connected, integrated K1/K3 stage timings must be compared against prototype fused K1/K3 kernel timings, and large unexplained degradation must be debugged/fixed.
- Remote process/card check before running:
  - `hy-smi` showed HCU use 0% on all cards, but memory 86% on HCU3-6.
  - `hy-smi --showpids` / `rocm-smi --showpids` failed with process-directory errors.
  - Container process scan found no residual V2 pytest/real_flow processes.
  - Host process scan showed unrelated sglang/test_low_latency activity; those were not killed.
  - Subsequent validation used `HIP_VISIBLE_DEVICES=0,1,2,7`.
- Files changed:
  - `megamoe/dcu_megamoe_v2/api.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_pybind.cpp`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implementation:
  - Added `dispatch_num_tokens=-1` V2 eager sentinel.
  - Runtime passes `num_tokens=-1` only to K1, causing K1 to read per-rank runtime token counts from each peer sym-buffer section.
  - Runtime still passes the local output token upper bound to K3; K3 dense tail scanning remains correctness-first.
  - K1 pybind now accepts `num_tokens=-1`.
  - V2 real-flow correctness test accepts `MEGAMOE_DCU_V2_REAL_FLOW_UNEVEN_TOKENS`, one token count per rank.
- Local validation:
  - `python -m py_compile megamoe/dcu_megamoe_v2/api.py megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - `git diff --check` passed.
- Remote build / validation:
  - Remote `git diff --check` passed.
  - Remote `python3 -m py_compile megamoe/dcu_megamoe_v2/api.py megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - Forced V2 extension rebuild passed after deleting stale V2 K1/K3 build objects and `.so` files.
  - Remote non-distributed V2 tests passed: `20 passed, 2 skipped in 14.80s`.
  - Remote 4-rank uneven correctness passed:
    - command shape: `HIP_VISIBLE_DEVICES=0,1,2,7`, ranks `4`, backends `ll,normal`, uneven tokens `[32,512,1024,2050]`.
    - pytest result: `1 passed in 53.15s`.
- Correctness:
  - Rank-0 printed `ll`: local tokens `32`, max_tokens `2050`, max_abs `0.00049591064453125`, mean_abs `1.7092428606702015e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
  - Rank-0 printed `normal`: local tokens `32`, max_tokens `2050`, max_abs `0.0006389617919921875`, mean_abs `6.178580224514008e-05`, mismatch `0`, stats_ok `True`, stage_ok `True`.
- Tokens tested:
  - Uneven per-rank local token counts: 32, 512, 1024, 2050.
- Pure groupgemm time:
  - Not measured in this correctness checkpoint.
- Fused time:
  - Not measured in this correctness checkpoint.
- Degradation ratio:
  - Not measured in this correctness checkpoint.
- Profile evidence:
  - Not collected in this checkpoint.
- Judgment:
  - Accept first eager uneven-token local-only correctness.
  - This does not close full remote communication acceptance because routes are still local-only.
  - Next item after re-reading the plan: add real cross-rank route correctness/performance validation, then compare integrated K1/K3 stage times against prototype fused kernel timings.

## 2026-06-03 15:13 +08:00 - Phase 9 Cross-Rank Route Correctness And K1 In-Kernel Sync

- User clarified:
  - scan and clean residual remote processes before running;
  - after real communication is connected, integrated K1/K3b stage timings must be compared against the corresponding prototype fused kernels, and large degradation must be debugged/fixed.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Remote pre-run checks:
  - scanned container and host for V2 pytest/build/prototype residual processes before build and test runs;
  - no V2 residual process was found, so nothing was killed;
  - `hy-smi` showed HCU3-6 with high memory usage, so validation used `HIP_VISIBLE_DEVICES=0,1,2,7`.
- Implementation:
  - Added an in-kernel rank barrier to K1 `ll` before any peer sym-buffer route/topk scan.
  - Added an in-kernel rank barrier to K1 `normal` through the metadata owner block for expert 0, followed by a local `launch_epoch` flag so other metadata owner blocks do not scan peer topk before all ranks have entered K1.
  - This keeps synchronization inside the K1 fused compute kernel and avoids adding a standalone Python-side rank barrier.
  - Replaced random `MASTER_PORT` selection in the real-flow distributed test with a socket-reserved local free port to avoid non-kernel `EADDRINUSE` failures during repeated spawned runs.
- Failed experiment recorded:
  - The first K1 normal sync attempt used `v2_device_grid_barrier(grid_barrier, metadata_sync_blocks)` across metadata owner blocks.
  - `normal` cross-rank 32-token validation hung until the 300s timeout, with an empty log.
  - Root cause: the actual number of launched metadata owner blocks can be smaller than the assumed local expert count for a given problem size, so the counted barrier can wait for blocks that do not exist.
  - Resolution: replaced the counted metadata barrier with an expert0-owned rank barrier plus `launch_epoch` flag.
- Local validation:
  - `python -m py_compile megamoe/dcu_megamoe_v2/api.py megamoe/dcu_megamoe_v2/runtime.py tests/test_dcu_megamoe_v2.py` passed.
  - `git diff --check` passed.
- Remote build / validation:
  - Remote `git diff --check` passed.
  - Remote V2 extension rebuild passed after deleting stale V2 K1/K3 build objects and `.so` files.
  - Remote non-distributed V2 tests passed: `20 passed, 2 skipped in 13.78s`.
- Cross-rank correctness, 4 ranks, route mode `cross_rank`, `HIP_VISIBLE_DEVICES=0,1,2,7`:
  - `ll`, 32: `max_abs=0.00047016143798828125`, `mean_abs=6.436857074731961e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 512: `max_abs=0.0004673302173614502`, `mean_abs=3.7984413211233914e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 1024: `max_abs=0.0005214214324951172`, `mean_abs=3.259707955294289e-05`, `mismatch=0`, `stats_ok=True`.
  - `ll`, 2050: `max_abs=0.0005538463592529297`, `mean_abs=3.972289414377883e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 32: `max_abs=0.0005764961242675781`, `mean_abs=9.478030551690608e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 512: `max_abs=0.000614166259765625`, `mean_abs=9.493521065451205e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 1024: `max_abs=0.0006656646728515625`, `mean_abs=9.485232294537127e-05`, `mismatch=0`, `stats_ok=True`.
  - `normal`, 2050: `max_abs=0.00069427490234375`, `mean_abs=9.491927630733699e-05`, `mismatch=0`, `stats_ok=True`.
- Tokens tested:
  - 32, 512, 1024, 2050 for both `ll` and `normal`.
- Pure groupgemm time:
  - Not measured in this correctness checkpoint.
- Fused time:
  - Not measured in this correctness checkpoint.
- Degradation ratio:
  - Not measured in this correctness checkpoint.
- Profile evidence:
  - Not collected in this correctness checkpoint.
- Logs:
  - `hygon_tmp/dcu_megamoe_v2/cross_rank_ll32_k1_rank_barrier_20260603.log`
  - `hygon_tmp/dcu_megamoe_v2/cross_rank_normal32_k1_epoch_flag_20260603.log`
  - `hygon_tmp/dcu_megamoe_v2/cross_rank_correctness_32_512_1024_2050_k1_epoch_flag_20260603.log`
  - `hygon_tmp/dcu_megamoe_v2/cross_rank_normal_correctness_32_512_1024_2050_20260603.log`
  - `hygon_tmp/dcu_megamoe_v2/non_distributed_after_cross_rank_k1_barrier_20260603.log`
- Judgment:
  - Accept K1 cross-rank metadata visibility fix for correctness.
  - This closes requested 4-rank cross-rank correctness for 32/512/1024/2050 on both forced backends.
  - Cross-rank performance, stage timing, prototype fused comparison, and any K3b-specific performance gap analysis remain open.

## 2026-06-03 15:13 +08:00 - Phase 9 Partial Cross-Rank Performance And Stage-vs-Prototype Signal

- Re-read Phase 9 after cross-rank correctness. Active item was cross-rank performance plus integrated K1/K3 stage-vs-prototype comparison.
- Files changed:
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implementation:
  - Extended the real-flow performance worker to accept `MEGAMOE_DCU_V2_REAL_FLOW_PERF_ROUTE_MODE=local_only|cross_rank`.
  - The perf worker now uses the same real-flow topk route generator as correctness and prints the actual route mode.
  - Kept the default as local-only so existing perf usage is unchanged unless the env explicitly requests cross-rank.
- Remote pre-run checks:
  - scanned for V2 residual pytest/build/prototype processes before the perf run;
  - no V2 residual process was found;
  - validation used `HIP_VISIBLE_DEVICES=0,1,2,7` because HCU3-6 still had high memory use.
- Remote perf run:
  - command shape: 4 ranks, route mode `cross_rank`, backends `ll,normal`, tokens `32,128,1024,4096`, warmup `2`, repeat `3`, stage breakdown enabled.
  - log: `hygon_tmp/dcu_megamoe_v2/cross_rank_perf_stage_32_128_1024_4096_w2r3_20260603.log`
  - The run timed out at 1200s after completing all `ll` rows and `normal` 32. This is a partial performance checkpoint, not final acceptance.
- Cross-rank integrated perf results collected:
  - `ll`, 32: total `0.784639 ms`, baseline e2e `2.546076 ms`, K1 `0.473280 ms`, K2 `0.071040 ms`, K3 `0.262720 ms`, max_abs `0.00046825408935546875`, mismatch `0`.
  - `ll`, 128: total `0.859680 ms`, baseline e2e `2.553597 ms`, K1 `0.494240 ms`, K2 `0.073440 ms`, K3 `0.335199 ms`, max_abs `0.00044536590576171875`, mismatch `0`.
  - `ll`, 1024: total `3.558716 ms`, baseline e2e `3.357596 ms`, K1 `1.708638 ms`, K2 `0.094719 ms`, K3 `1.871837 ms`, max_abs `0.0005578994750976562`, mismatch `0`.
  - `ll`, 4096: total `14.140305 ms`, baseline e2e `8.801269 ms`, K1 `6.556473 ms`, K2 `0.264800 ms`, K3 `7.856471 ms`, max_abs `0.0006422996520996094`, mismatch `0`.
  - `normal`, 32: total `1.715038 ms`, baseline e2e `2.509917 ms`, K1 `1.050879 ms`, K2 `0.084000 ms`, K3 `0.646559 ms`, max_abs `0.0005764961242675781`, mismatch `0`.
- Integrated stage-vs-prototype comparison from collected rows:
  - K1 `ll`: cross-rank integrated overhead vs prototype fused is about `+35.35%` at 32, `+37.41%` at 128, `+21.65%` at 1024, and `+11.20%` at 4096.
  - K3 `ll`: cross-rank integrated overhead vs prototype fused is about `+57.95%` at 32, `+83.90%` at 128, `+127.67%` at 1024, and `+143.57%` at 4096.
  - K1 `normal`, 32: cross-rank integrated overhead vs prototype fused is about `+39.76%`.
  - K3 `normal`, 32: cross-rank integrated overhead vs prototype fused is about `+26.96%`.
- Post-timeout process/card check:
  - no V2 pytest/build/prototype process remained;
  - `hy-smi` then showed HCU0/1 plus HCU3-6 with high memory use from other visible workloads, leaving only HCU2 and HCU7 clearly free.
  - Remaining 4-rank perf rows were not run to avoid interfering with unrelated users.
- Pure groupgemm time:
  - Not remeasured in this integrated perf checkpoint. Prototype fused comparison uses the current recorded standalone V2 prototype timings.
- Fused time:
  - Integrated staged cross-rank fused times listed above.
- Profile evidence:
  - HIP-event stage timing only; no hipprof/PMC collected in this checkpoint.
- Judgment:
  - Cross-rank correctness is accepted, but cross-rank performance is not accepted.
  - The major new performance gap is K3 cross-rank integrated overhead, especially `ll` large tokens.
  - K1 small-token overhead from the added in-kernel rank barrier/metadata sync is also material and should be profiled, but K3 is the larger gap.
  - Next implementation focus should be K3b cross-rank combine/tail-reduce performance and then completing the missing `normal` 128/1024/4096 perf rows when four clean devices are available.

## 2026-06-03 17:22 +08:00 - K1 Normal Metadata Experiment Quarantined

- User asked why `k1_groupgemm_v2.cpp` was edited during the current optimization loop.
- Clarification:
  - the edit was an unvalidated attempt to make K1 normal metadata scan cooperative across x-blocks;
  - it touched the standalone prototype bridge file, while the long-term plan is to move production logic into stage-owned V2 files;
  - it had not been rebuilt or correctness/performance tested, so it must not be mixed into the accepted baseline.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Action:
  - downloaded the remote source version that was already used for successful 4-rank correctness/build validation;
  - surgically reverted the local K1 normal metadata block from the experimental `metadata_worker_block` implementation back to the accepted `metadata_owner_block` implementation;
  - verified the local `k1_groupgemm_v2.cpp` SHA256 now matches the remote accepted copy exactly;
  - verified `metadata_worker_block`, `metadata_reset_epoch`, and `metadata_scan_epoch` no longer exist in the accepted source.
- Validation:
  - `git diff --check` passed locally.
- Tokens tested:
  - none in this checkpoint.
- Pure groupgemm time:
  - not measured.
- Fused time:
  - not measured.
- Degradation ratio:
  - not measured.
- Correctness:
  - no new correctness run; this checkpoint restores the last correctness-accepted K1 source.
- Profile evidence:
  - not collected.
- Judgment:
  - reject/quarantine the cooperative K1 normal metadata experiment until it can be reintroduced in stage-owned code with a micro correctness/performance gate.
  - Continue from the plan item: compare integrated K1/K3 stage timings against standalone prototype fused timings first, then optimize the largest gap.

## 2026-06-03 17:48 +08:00 - K3 Tail Reduce All-Rank Barrier Candidate

- Active plan item:
  - real-flow K3 must compare against the standalone K3 fused prototype, but first the K3 combine/reduce semantics must be honest for cross-rank remote stores.
- Finding while reading K3 code:
  - K3 `ll` rowptr direct-store path and K3 `normal` copy-stage path both performed only local/intra-kernel-block synchronization before tail reduce.
  - For cross-rank routing, expert-owner ranks remote-store partial L2 rows into the source rank's combine buffer; the source rank's tail reduce then reads that local combine buffer.
  - Without an all-rank synchronization inside K3, tail reduce can theoretically read before peer remote stores have arrived.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Implementation:
  - Added an in-kernel `mega_moe_rank_barrier` before K3 tail reduce for the `ll` rowptr path, after the local grid barrier that confirms all local rowptr stores are issued.
  - Added an in-kernel `mega_moe_rank_barrier` before K3 tail reduce for the `normal` copy-stage path, after copy workers finish remote combine stores and before they start reading local combine rows.
  - Added a source contract assertion so future refactors do not silently remove the K3 in-kernel all-rank synchronization hook.
  - No standalone Python-side barrier or extra combine kernel was introduced.
- Local validation:
  - `python -m py_compile tests/test_dcu_megamoe_v2.py megamoe/dcu_megamoe_v2/api.py megamoe/dcu_megamoe_v2/runtime.py megamoe/dcu_megamoe_v2/K1_fused/k1_fused.py megamoe/dcu_megamoe_v2/K3_fused/k3_fused.py` passed.
  - Local pytest could not run because the local Windows Python does not have `pytest` installed.
  - `git diff --check` passed.
- Remote validation:
  - Synced source to remote using a tar archive under `hygon_tmp`.
  - First tar extraction reported directory chmod/utime warnings on existing directories; rerun with `--no-overwrite-dir --touch --no-same-owner --no-same-permissions` succeeded.
  - Remote `py_compile` and `git diff --check` passed.
  - Remote source/runtime contract pytest passed: `3 passed`.
  - Forced V2 K1/K3 extension rebuild by deleting stale V2 extension objects and `.so` files; rebuild completed and recompiled both V2 K1 and V2 K3 extension objects.
- 2-rank smoke:
  - Attempted a temporary `hygon_tmp/dcu_megamoe_v2_2rank_smoke.py` using HCU2/HCU7 for `ll` and `normal` token 32 cross-rank.
  - First attempt failed before kernels because the temporary script lacked the multiprocessing `if __name__ == "__main__"` guard.
  - Second attempt timed out at 300s.
  - This is not an acceptance result because the official requirement is 4-rank or 8-rank communication; it is also not enough to conclude the K3 barrier candidate is correct.
  - After the timeout, SSH started returning `Connection closed by 10.17.176.13 port 22`, so residual process cleanup and diagnosis are deferred until the remote host is reachable again.
- Tokens tested:
  - no accepted token result in this checkpoint.
- Pure groupgemm time:
  - not measured.
- Fused time:
  - not measured.
- Degradation ratio:
  - not measured.
- Correctness:
  - not accepted for this K3 rank-barrier candidate yet.
- Profile evidence:
  - not collected.
- Next judgment:
  - When remote SSH is reachable, first scan and kill only the temporary 2-rank smoke processes if they remain.
  - Then run a 4-rank correctness check at token 32 for `ll` first, then `normal`, before doing performance.
  - If the 4-rank run hangs, revert or redesign the in-kernel K3 rank synchronization instead of continuing performance work on a questionable semantic path.

## 2026-06-03 18:07 +08:00 - Continue Performance Gap Analysis And Revert Blocking K3 Barrier Candidate

- User asked why the performance-gap investigation did not continue.
- Correction:
  - The remote SSH outage prevents new 4-rank/8-rank timing, but it should not stop local gap analysis.
  - The unaccepted K3 all-rank barrier candidate also should not remain in the active path while it may hang and block performance debugging.
- Files changed:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `tests/test_dcu_megamoe_v2.py`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Action:
  - Re-read plan/progress/findings before making the next decision.
  - Recomputed the recorded integrated-vs-prototype gaps from existing 4-rank data.
  - Reverted the unaccepted K3 all-rank barrier candidate from active source and removed its source-contract assertion.
  - Kept the all-rank K3 synchronization issue recorded as a future design requirement, but not as active code.
- Existing 4-rank cross-rank gap ranking from recorded data:
  - `ll`, 4096 K3: prototype `3.225200 ms`, integrated `7.856471 ms`, gap `+143.60%`.
  - `ll`, 1024 K3: prototype `0.822160 ms`, integrated `1.871837 ms`, gap `+127.67%`.
  - `ll`, 128 K3: prototype `0.182272 ms`, integrated `0.335199 ms`, gap `+83.90%`.
  - `ll`, 32 K3: prototype `0.166320 ms`, integrated `0.262720 ms`, gap `+57.96%`.
  - `normal`, 32 K1: prototype `0.751902 ms`, integrated `1.050879 ms`, gap `+39.76%`.
  - `normal`, 32 K3: prototype `0.509280 ms`, integrated `0.646559 ms`, gap `+26.96%`.
- Interpretation:
  - The biggest current gap is K3 `ll`, especially large-token forced-backend sizes.
  - K1 `ll` is not the blocker at large size; K1 4096 gap is only about `+11.21%`.
  - The older 4-rank local-only integrated run already showed K3 overhead even without cross-rank route mode:
    - K3 `ll` integrated-vs-prototype overhead `+52.75%`, `+53.97%`, `+44.34%`, `+67.32%` at 32/128/1024/4096.
    - K3 `normal` integrated-vs-prototype overhead `+27.84%`, `+20.32%`, `+45.41%`, `+69.50%` at 32/128/1024/4096.
  - Therefore the gap is not only remote communication; it also includes real-flow K3 row layout / dense tail reduce / row pointer setup / workspace path differences.
- Tokens tested:
  - no new accepted run in this checkpoint.
- Pure groupgemm time:
  - not newly measured.
- Fused time:
  - not newly measured.
- Degradation ratio:
  - recalculated from existing recorded data above.
- Correctness:
  - no new correctness run.
- Profile evidence:
  - no new profile because remote SSH is still unavailable.
- Next performance-debug plan when remote returns:
  - clean residual smoke processes first;
  - rerun accepted active source, not the reverted K3 barrier candidate;
  - collect K3-only A/B timing in this order:
    1. standalone prototype K3 fused denominator with the same visible 4-rank device set;
    2. integrated real-flow local-only K3 stage timing;
    3. integrated real-flow cross-rank K3 stage timing;
    4. if possible, K3 `normal` missing 128/1024/4096 rows;
  - then profile the largest K3 row with hipprof to separate remote store/copy-stage, dense tail reduce, and synchronization overhead.

## 2026-06-03 17:49 +08:00 - Correct Performance Denominator Back To Pure GroupGEMM

- User clarified that the comparison target is the pure groupgemm prototype/harness, not a standalone fused prototype kernel.
- Correction:
  - Stop modifying the prototype kernel/harness for fused-denominator convenience.
  - Compare the real MegaMoE V2 integrated K1/K3 fused stages against the corresponding pure groupgemm timings.
  - Optimize the real-flow fused stage degradation relative to pure groupgemm.
- Reverted this checkpoint's mistaken prototype-harness edits:
  - removed the temporary `symm_rank_barrier` / `--symm-rank-barrier` idea from `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`;
  - removed the matching explicit launcher argument from `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`;
  - restored the prior standalone launcher shape instead of trying to make a standalone fused K1 denominator work.
- Files changed for the correction:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/findings.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
- Validation:
  - Local `rg` confirms no `symm_rank_barrier`, `symm-rank-barrier`, or `symm_rank_barrier_enabled` remains.
  - Local `git diff --check` passed.
  - No remote/GPU runs were started after the user reported someone else is using the cards.
- Data handling:
  - The prototype fused timings collected just before this correction are not accepted as the denominator for the next performance decision.
  - Use the existing pure groupgemm rows from the V2 harness where available, and recollect missing pure denominators only when cards are free.
- Next when cards are available:
  - Run 4-rank integrated stage breakdown for `ll` and `normal`.
  - Run or reuse pure K1/K3 groupgemm timings with the same backend family, token count, visible device set, warmup/repeat policy, and layout.
  - Compute fused-vs-pure degradation for K1/K3 `ll` at 32/128/512/1024, with emphasis below/around 1024.
  - Compute fused-vs-pure degradation for K1/K3 `normal` at 1024/2048/4096 and later 8192.
  - Profile the largest fused-vs-pure gap before making another source change.

## 2026-06-03 18:19 +08:00 - Same-Size Pure-vs-Integrated 4-Rank Performance Gate

- User reminder:
  - keep same-size comparisons;
  - do not modify the pure groupgemm prototype/harness;
  - optimize the real MegaMoE V2 integrated K1/K3 fused stage relative to the pure groupgemm denominator.
- Files changed:
  - `.planning/dcu_megamoe_v2/task_plan.md`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Remote setup:
  - synced current local V2 sources to remote through a tar archive under `hygon_tmp`;
  - remote `py_compile` passed for V2 Python files;
  - remote `git diff --check` passed;
  - process scan found no residual V2 pytest/build/prototype processes;
  - `hy-smi` showed all 8 HCUs at `VRAM 0%` and `HCU 0%` before the runs;
  - forced V2 K1/K3 extension rebuild completed;
  - standalone V2 `make hipcc aicc` completed.
- Pure groupgemm denominator log:
  - `hygon_tmp/dcu_megamoe_v2/pure_groupgemm_k1_k3_ll_normal_same_size_w3r5_20260603.log`
  - warmup `3`, repeat `5`, measure rounds `3`, `HIP_VISIBLE_DEVICES=0,1,2,3`, correctness `--check 1`.
- Integrated 4-rank real-flow logs:
  - `hygon_tmp/dcu_megamoe_v2/integrated_cross_rank_ll_32_128_512_1024_w3r5_20260603.log`
  - `hygon_tmp/dcu_megamoe_v2/integrated_cross_rank_normal_1024_2048_4096_w3r5_20260603.log`
  - route mode `cross_rank`, ranks `4`, warmup `3`, repeat `5`, stage breakdown enabled.
- Same-size degradation table, integrated fused stage versus pure groupgemm:
  - `normal` K1 4096: pure `2.339350 ms`, fused `7.527326 ms`, degradation `+221.77%`, correctness max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
  - `ll` K3 512: pure `0.358080 ms`, fused `1.010718 ms`, degradation `+182.26%`, correctness max_abs `0.000443459`, mean_abs `2.40812e-05`, mismatch `0`.
  - `ll` K3 1024: pure `0.687774 ms`, fused `1.855514 ms`, degradation `+169.79%`, correctness max_abs `0.000473738`, mean_abs `3.14684e-05`, mismatch `0`.
  - `normal` K3 4096: pure `1.342780 ms`, fused `3.430545 ms`, degradation `+155.48%`, correctness max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
  - `normal` K3 2048: pure `0.797534 ms`, fused `1.859831 ms`, degradation `+133.20%`, correctness max_abs `0.000694275`, mean_abs `9.49171e-05`, mismatch `0`.
  - `normal` K1 2048: pure `1.408670 ms`, fused `3.243505 ms`, degradation `+130.25%`, correctness max_abs `0.000694275`, mean_abs `9.49171e-05`, mismatch `0`.
  - `normal` K3 1024: pure `0.465890 ms`, fused `1.001436 ms`, degradation `+114.95%`, correctness max_abs `0.000665665`, mean_abs `9.48523e-05`, mismatch `0`.
  - `ll` K3 128: pure `0.164448 ms`, fused `0.334240 ms`, degradation `+103.25%`, correctness max_abs `0.000470161`, mean_abs `6.10542e-05`, mismatch `0`.
  - `normal` K1 1024: pure `0.854590 ms`, fused `1.608312 ms`, degradation `+88.20%`, correctness max_abs `0.000665665`, mean_abs `9.48523e-05`, mismatch `0`.
  - `ll` K3 32: pure `0.158624 ms`, fused `0.267200 ms`, degradation `+68.45%`, correctness max_abs `0.000287414`, mean_abs `2.59288e-05`, mismatch `0`.
  - `ll` K1 32: pure `0.300320 ms`, fused `0.491999 ms`, degradation `+63.82%`, correctness max_abs `0.000287414`, mean_abs `2.59288e-05`, mismatch `0`.
  - `ll` K1 128: pure `0.308895 ms`, fused `0.493280 ms`, degradation `+59.69%`, correctness max_abs `0.000470161`, mean_abs `6.10542e-05`, mismatch `0`.
  - `ll` K1 512: pure `0.649183 ms`, fused `0.936159 ms`, degradation `+44.21%`, correctness max_abs `0.000443459`, mean_abs `2.40812e-05`, mismatch `0`.
  - `ll` K1 1024: pure `1.258660 ms`, fused `1.713755 ms`, degradation `+36.16%`, correctness max_abs `0.000473738`, mean_abs `3.14684e-05`, mismatch `0`.
- Local-only isolation probe:
  - log `hygon_tmp/dcu_megamoe_v2/integrated_local_only_gap_probe_normal4096_ll512_1024_w3r5_20260603.log`.
  - `normal` 4096 local-only K1 `7.289430 ms` versus cross-rank K1 `7.527326 ms`; cross-rank adds only about `0.238 ms`, so the dominant K1 normal gap is internal staging/metadata/sync, not remote communication.
  - `ll` 512 local-only K3 `0.664799 ms` versus cross-rank K3 `1.010718 ms`; cross-rank adds about `0.346 ms`.
  - `ll` 1024 local-only K3 `1.185118 ms` versus cross-rank K3 `1.855514 ms`; cross-rank adds about `0.670 ms`.
- Profile evidence:
  - hipprof log `hygon_tmp/dcu_megamoe_v2/hipprof_normal4096/normal4096_profile_20260603.log`.
  - Profiled `normal` 4096 cross-rank with warmup `1`, repeat `2`.
  - Stage timing in the profiled run: K1 `7.668 ms`, K3 `3.446 ms`, total `11.131 ms`, mismatch `0`.
  - HIP trace stats show the V2 normal K1 fused kernel average aligns with the K1 stage timing, so the worst K1 gap is inside the fused kernel rather than Python glue.
- Static attribution:
  - K1 normal real-flow currently scans route metadata in metadata-owner blocks, fills row maps, stages full activation rows into `staged_x` HBM, waits on a per-row-tile row-stage barrier, and then GEMM reads `staged_x`.
  - For 4096 tokens this introduces roughly a full extra staged activation pass plus synchronization before the groupgemm work; pure groupgemm has none of that.
  - The next source experiment should target the real-flow K1 normal fused path, not the prototype denominator.
- Next step:
  - first small experiment: change only the real-flow K1 normal extension launch grouping to test whether increasing x-block parallelism for row staging reduces the `normal` K1 4096 gap without touching pure groupgemm.
  - If this only gives a small improvement, the next deeper fix is to remove the HBM `staged_x` middleman and pull rows directly from sym-buffer source pointers inside the GEMM load path.

## 2026-06-03 18:31 +08:00 - Reject K1 Normal `c_stage_n_group=2` Launch Experiment

- Experiment:
  - changed only `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`;
  - `dcu_megamoe_v2_launch_k1_normal_symm_stage_raw` used `c_stage_n_group=2` instead of the accepted `4`;
  - pure groupgemm prototype/harness was not modified.
- Rationale:
  - the hypothesis was that more x-blocks per row tile might parallelize K1 normal row staging and reduce the worst `normal` 4096 K1 gap.
- Validation:
  - local `git diff --check` passed before sync;
  - remote pre-run process scan found no residual V2 processes;
  - remote `hy-smi` showed all HCUs free before the build/test;
  - forced V2 K1/K3 extension rebuild completed;
  - 4-rank cross-rank integrated performance ran for backend `normal`, token `4096`, warmup `3`, repeat `5`, stage breakdown enabled.
- Log:
  - `hygon_tmp/dcu_megamoe_v2/integrated_cross_rank_normal4096_stage_ngroup2_w3r5_20260603.log`
- Result:
  - pure K1 denominator stayed `2.339350 ms`;
  - integrated K1 fused stage worsened from prior `7.527326 ms` to `9.135353 ms`;
  - K1 degradation worsened from `+221.77%` to `+290.50%`;
  - total V2 staged time `12.762070 ms`;
  - K3 stage stayed roughly unchanged at `3.431038 ms`;
  - correctness remained clean: max_abs `0.000721931`, mean_abs `9.49635e-05`, mismatch `0`.
- Decision:
  - reject and revert the experiment.
  - The larger block count increases scheduling/synchronization overhead more than it helps row staging.
  - Next K1 normal optimization should target the staging architecture itself: reduce the full `staged_x` HBM middleman or replace it with direct sym-buffer row loads inside the GEMM path.

## 2026-06-03 18:43 +08:00 - K1 Normal Direct-Pull Build-Only Experiment, GPU Validation Pending

- Experiment:
  - added a default-off `kUseDirectSymmLoad` template parameter to the large C pack5 `V2_DeepGemm...` kernel body;
  - enabled it only from `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu` for the real-flow K1 normal launcher;
  - pure groupgemm default instantiations and standalone pure timing modes remain default-off for this path.
- Intended effect:
  - keep in-kernel metadata writeback and the lightweight `x_scale` staging;
  - skip the large FP8 `staged_x` row copy;
  - have compute waves load each grouped row's FP8 activation directly from the source rank's sym-buffer row using `symm_src_ranks/symm_src_tokens`;
  - invalid/padded rows return zero packs.
- Files changed in this checkpoint:
  - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
  - `megamoe/dcu_megamoe_v2/K1_fused/k1_fused_ext.cu`
  - `.planning/dcu_megamoe_v2/progress.md`
  - `.planning/dcu_megamoe_v2/findings.md`
- Validation completed:
  - local `git diff --check` passed;
  - remote `py_compile` and `git diff --check` passed after sync;
  - forced remote V2 K1 extension rebuild passed;
  - standalone V2 `make -C csrc/kernels/dcu_megamoe_v2 hipcc aicc` passed, so the prototype/pure harness still compiles.
- GPU validation status:
  - not run yet.
  - Before the performance run, `hy-smi` showed all eight HCUs occupied by an unrelated `sglang` workload:
    - scheduler PIDs `714829` through `714836`;
    - parent workload PID `706017`;
    - VRAM about `86%` on HCU0-HCU3 and `91%` on HCU4-HCU7.
  - V2 process scan was empty, so these processes were not killed.
- Tokens tested:
  - none for the direct-pull build-only checkpoint.
- Pure groupgemm time:
  - unchanged from prior same-size denominator; no new timing run.
- Fused time:
  - not measured for direct-pull because cards are occupied.
- Degradation ratio:
  - pending.
- Correctness:
  - pending.
- Profile evidence:
  - pending.
- Next when cards free:
  - run backend `normal`, token `4096`, 4-rank cross-rank, warmup `3`, repeat `5`, stage breakdown;
  - if correctness passes and K1 improves versus prior `7.527326 ms`, also run `1024` and `2048`;
  - if correctness fails or K1 slows, revert `kUseDirectSymmLoad` activation and record the rejection.
- Denominator sanity:
  - standalone `--tokens` expands rows using `tokens * topk / local_experts`, matching the real-flow grouped-row shape.
  - For `normal` 4096 with `num_topk=6`, both the pure harness and integrated path should use `valid_rows_per_expert=768`, `rows_aligned_per_expert=768`, and `launch_rows=24576` for 32 local experts.
