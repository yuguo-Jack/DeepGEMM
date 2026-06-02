# DCU MegaMoE V2 Progress

The persistent working log lives under `.planning/dcu_megamoe_v2/`.

## 2026-05-29

- User corrected the K3 direction: V2 K3 must use the C groupgemm
  implementation and the unified pack5 layout, including small-token K3
  coverage. The previous ASM/kpack2 prototype is not accepted.
- User also clarified that communication-fused acceptance requires a real
  4-rank or 8-rank run. The existing 2-DCU K1 staged result is useful smoke
  data but is not final acceptance.
- Reset active K3 work toward C pack5:
  - Removed active K3 ASM helper APIs from `stages.py`.
  - Removed K3 kpack2 layout helpers from `layout.py`.
  - Removed K3 ASM/kpack2 tests from `tests/test_dcu_megamoe_v2.py`.
  - Deleted the V2 K3 ASM extension source and K3 benchmark script from the
    active V2 tree.
- No K3 tokens were benchmarked in this reset yet. Pure/fused K3 C pack5
  timings and correctness remain pending.
- Verification after cleanup:
  - Local `py_compile`: passed.
  - Local `git diff --check`: passed.
  - Remote `tests/test_dcu_megamoe_v2.py`: 5 passed in 12.88s.
- Locked unified pack5 layout for L1 and L2:
  - Added `pack5_layout_check.cpp`, a small host-side C++ offset verifier.
  - Added L1 fixture `N=4096,K=4096` and L2 fixture `N=4096,K=2048` to the
    V2 pytest.
  - Added Python-vs-C pack5 offset checks.
  - Remote `MODE=layout-check bash scripts/build_dcu_megamoe_v2.sh`: passed
    with sample offsets 20119625 and 14811201.
  - Remote `tests/test_dcu_megamoe_v2.py`: 7 passed in 13.05s.
  - No tokens were benchmarked; this was layout-only validation.
- Completed K1 small-token 8-rank communication acceptance:
  - Added `SYMM_RANKS`, `SYMM_DEVICES`, and `RANK_IDX` passthrough to
    `scripts/build_dcu_megamoe_v2.sh`.
  - 8-rank setup: `DEVICE=0,1,2,3,4,5,6,7`, `SYMM_RANKS=8`,
    `SYMM_DEVICES=8`, `RANK_IDX=0`.
  - 32 tokens: pure 0.299675 ms, fused 0.352000 ms, degradation +17.46%,
    max_abs 0.000244141, mean_abs 1.32851e-09, value_mismatch 0.
  - 128 tokens: pure 0.307098 ms, fused 0.363023 ms, degradation +18.21%,
    max_abs 0.000244141, mean_abs 1.0992e-09, value_mismatch 0.
  - hipprof for both 32 and 128 shows 7 HIPOPS calls to
    `V2_K1_LowLatencyMaskedGroupGemmKernel`, 100% of HIPOPS kernel time, with
    no standalone dispatch/prebuild kernel in the timed loop.
  - ISA evidence was collected with `hipcc -save-temps=obj` under
    `hygon_tmp/dcu_megamoe_v2/save_temps_k1_hipcc`; direct dccobjdump on the
    linked executable still only reports host ELF.

## 2026-05-28

- Created file-based plan for the V2 effort.
- Mapped K1_groupgemm_fp8 pure groupgemm baselines and pack5 layout.
- Mapped existing DCU large-opt K1/K2/K3 boundaries.
- Mapped CUDA MegaMoE dispatch-pull and combine-reduce overlap patterns.
- Measured quick pure K1 groupgemm baselines on remote GPU 0.
- Started isolated V2 skeleton under `csrc/kernels/dcu_megamoe_v2`.
- Verified V2 layout tests remotely: 4 passed.
- Built V2 independent K1 harness with hipcc and aicc.
- Verified V2 K1 pure quick tokens:
  - 32: 0.299546 ms, max_abs 0.000488281, value_mismatch 0.
  - 128: 0.307141 ms, max_abs 0.000488281, value_mismatch 0.
  - 1024: 0.745766 ms, max_abs 0, value_mismatch 0.
  - 4096: 2.26284 ms, max_abs 0, value_mismatch 0.
- Added K1 `c-ll-pull` fused prototype.
  - Initial mapping-load variant was correct but too slow: +58.47% at 32 tokens and +56.97% at 128 tokens.
  - Cached source rows in registers per tile: 32 tokens 0.303910 ms (+1.46%), 128 tokens 0.314656 ms (+2.45%).
  - hipprof confirmed one V2 low-latency kernel per measured launch and no extra dispatch/prebuild kernel in the repeat loop.
- Added K1 `c-ll-symm-pull` prototype that uses the existing DCU MegaMoE
  symmetric-buffer layout and peer pointer header.
  - The kernel scans peer-rank `topk_idx/topk_weights`, builds local expert
    source rows in route scratch, then pulls `x/x_sf` from peer sym buffers
    inside the same low-latency L1 groupgemm kernel.
  - 32 tokens: pure 0.299269 ms, symm-pull 0.333588 ms, degradation +11.47%,
    max_abs 0.000244141, value_mismatch 0.
  - 128 tokens: pure 0.307615 ms, symm-pull 0.341599 ms, degradation +11.05%,
    max_abs 0.000244141, value_mismatch 0.
  - hipprof `--check 0` confirmed only 7 calls to
    `V2_K1_LowLatencyMaskedGroupGemmKernel` for warmup=2/repeat=5; no separate
    dispatch/prebuild kernel appears in the timed path.
  - Two-DCU peer-read mode (`--symm-devices 2`) is correct but not yet fast
    enough: 32 tokens 0.394100 ms (+31.70%), 128 tokens 0.432901 ms
    (+40.73%), both max_abs 0.000244141 and value_mismatch 0.
  - Two-DCU hipprof `--check 0` showed only 7 fused K1 kernel calls, average
    405234 ns. This confirms the slowdown is inside the fused kernel's remote
    route/input reads, not from extra launch overhead.
  - Added `c-ll-symm-stage`, still a single fused K1 kernel, that stages remote
    peer token rows into local scratch before the low-latency groupgemm consumes
    local staged A.
  - After vectorizing staging to 16B copies and only staging rows read by the
    K1 block tiles, two-DCU results are back within target:
    - 32 tokens: pure 0.299269 ms, staged 0.349984 ms, degradation +16.95%,
      max_abs 0.000244141, value_mismatch 0.
    - 128 tokens: pure 0.307615 ms, staged 0.358944 ms, degradation +16.69%,
      max_abs 0.000244141, value_mismatch 0.
  - Two-DCU staged hipprof `--check 0` showed only 7 fused K1 kernel calls,
    average 359154 ns.
- Added V2 K2 Python stage wrapper in `csrc/kernels/dcu_megamoe_v2/stages.py`.
  - It reuses the existing optimized DCU K2 source and can JIT-build the K2
    extension into `hygon_tmp/dcu_megamoe_v2/torch_extensions` when the main
    `megamoe` package is not built inplace.
  - Remote pytest: `5 passed in 83.42s`.
  - K2 BF16 reference check: rows=32, hidden=128, max_abs 0, mean_abs 0,
    mismatch 0.
- Rejected K3 ASM/kpack2 experiment:
  - A V2-local K3 ASM/kpack2 prototype was built and measured, but it violates
    the V2 requirement to use C groupgemm and unified pack5.
  - Large-token measurements from that experiment are retained only as failure
    history. They are not accepted K3 results and must not be used as Phase 7
    evidence.
  - The attempt did reveal two useful failures for the C rebuild:
    the unadapted V2 C pack5 harness VMFaulted at `N=4096,K=2048`, and using
    the existing `K3COMBINE` ASM code object with null row-combine pointers also
    VMFaulted.
  - Next valid K3 work is to adapt the C pack5 groupgemm skeleton for L2
    shape, then fuse combine in the C epilogue and validate 32/128/1024/4096
    with 4-rank or 8-rank communication.

## 2026-05-29

- Added K3 C pack5 identity row-combine pointer output in the V2 harness.
  - Files changed:
    - `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`
    - `scripts/build_dcu_megamoe_v2.sh`
  - New script modes:
    - `MODE=k3-small-rowptr`
    - `MODE=k3-large-rowptr`
    - `MODE=k3-rowptr`
- Correctness:
  - 32 tokens: max_abs 0.000244141, mean_abs 6.76449e-10, bit_mismatch 19, value_mismatch 0.
  - 128 tokens: max_abs 0.000244141, mean_abs 6.32578e-10, bit_mismatch 75, value_mismatch 0.
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- Current same-binary pure K3 C pack5 denominators:
  - 32 tokens: 0.155658 ms.
  - 128 tokens: 0.164406 ms.
  - 1024 tokens: 0.440655 ms.
  - 4096 tokens: 1.29907 ms.
- K3 identity rowptr timing:
  - 32 tokens: 0.160789 ms, degradation +3.30%.
  - 128 tokens: 0.165125 ms, degradation +0.44%.
  - 1024 tokens: 0.518218 ms, degradation +17.60%.
  - 4096 tokens: 1.37193 ms, degradation +5.61%.
- Rejected experiment:
  - Dynamic raw-buffer row-resource stores failed correctness at 1024 tokens (`max_abs=0.0157471`, `value_mismatch=767`), so the change was reverted.
- Next:
  - Point K3 row-combine pointers into real 8-rank sym-buffer combine sections and validate remote writes for 32/128/1024/4096.
  - Then add in-kernel combine reduce / tail reduce; identity rowptr is not communication acceptance.

## 2026-05-29 K3 Real 8-Rank Row-Combine Checkpoint

- Added real 8-rank K3 row-combine targets under the V2 C pack5 path.
- K3 small real 8-rank row-combine is within target:
  - 32 tokens: pure 0.155658 ms, fused 0.15945 ms, degradation +2.44%, max_abs 0.00012207, value_mismatch 0.
  - 128 tokens: pure 0.164406 ms, fused 0.163456 ms, effectively no degradation, max_abs 0.000244141, value_mismatch 0.
- K3 large correctness is clean but performance is not yet accepted:
  - Direct remote row-combine: 1024 tokens 0.699353 ms (+58.71%), 4096 tokens 2.14729 ms (+65.29%).
  - Same-kernel copy-stage: 1024 tokens 0.619803 ms (+40.66%), 4096 tokens 1.86016 ms (+43.19%).
  - Copy-stage correctness: max_abs 0, mean_abs 0, value_mismatch 0 for 1024 and 4096.
- Rejected experiments:
  - Reducing copy workers below 16 worsened performance.
  - Vectorized direct row-address epilogue passed after a shuffle participation fix, but slowed 1024 tokens to about 1.26 ms and was reverted.
- hipprof evidence for K3 small:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_combine32`
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_combine128`
  - Both show 7 calls to the V2 low-latency fused kernel and no standalone combine kernel in the timed path.
- Next:
  - Continue K3 large optimization and add in-kernel combine reduce / tail reduce. Current K3 row-combine still writes partial rows and is not final combine-reduce acceptance.

## 2026-05-29 K3 Tail-Reduce And Copy-Stage Experiments

- Added a same-kernel local-rank tail-reduce prototype on top of the K3 large C pack5 copy-stage path.
  - 1024 tokens correctness: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0, checked 4194304.
  - 4096 tokens correctness: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0, checked 16777216.
  - Timing is not accepted yet: 1024 tokens 0.773269 ms (+75.49% vs pure 0.440655 ms), 4096 tokens 2.44608 ms (+88.30% vs pure 1.29907 ms).
- Confirmed the remote run was not stuck before continuing: no stale V2 benchmark process was present and all 8 HCUs were at 0% HCU use.
- Rejected copy-worker-after-compute scheduling:
  - Correct but slow: 1024 tokens regressed to about 0.794 ms.
  - Reverted because front-scheduled copy workers are faster for the current overlap model.
- Rejected same-block self-copy:
  - Raw timing looked close to target at 1024 tokens, but correctness failed with zero/missing combine rows.
  - Store wait/fence and compute-wave stride fixes did not make it correct, so the experiment was reverted.
- Restored current correct K3 large copy-stage baseline:
  - 1024 tokens single correctness run: 0.622238 ms, max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.

## 2026-05-29 K3 Large Fence And Worker Experiments

- Captured hipprof for current K3 large copy-stage:
  - 1024 artifact: `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_copy_stage1024`.
  - 4096 artifact: `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_copy_stage4096_device_fence`.
  - HIPOPS shows seven calls to the single V2 large C fused kernel and no standalone combine kernel in the timed loop.
- Rejected fence optimization:
  - Changing the compute-tile ready fence from `__threadfence_system()` to `__threadfence()` looked faster, but a forced rebuild and repeated correctness failed with NaN/mismatch.
  - Restored `__threadfence_system()` and revalidated 1024 correctness: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- Rejected copy-worker counts above 16:
  - 24 workers: 4096 tokens 2.31437 ms.
  - 32 workers: 4096 tokens 3.74851 ms.
  - Guard restored to `[1,16]`.
- Current restored system-fence copy-stage baseline:
  - 1024 tokens: pure 0.440655 ms, fused 0.613866 ms, degradation +39.31%, max_abs 0, value_mismatch 0.
  - 4096 tokens: pure 1.29907 ms, fused 1.7904 ms, degradation +37.82%, max_abs 0, value_mismatch 0.
- Rejected experiments:
  - K3/L2 `--c-tile-n 64` pack5 pure path was correct but too slow: 1024 pure 0.934237 ms.
  - Sorting standard combine rows by source rank/partial row slightly helped 1024 but regressed 4096 to 3.50465 ms.
- ISA note:
  - Save-temps are under `hygon_tmp/dcu_megamoe_v2/save_temps_k3_aicc`.
  - The current K3 copy-stage specialization has no private scratch, around 213 VGPRs, and 64 KiB group segment. Copy-worker blocks therefore still reserve full GEMM LDS, which is likely the next occupancy/overlap limiter.
- Repaired self-copy follow-up:
  - Making loader waves and invalid compute waves participate fixed the previous self-copy correctness failure.
  - It was still slower than the 16-worker copy-stage: 1024 tokens stabilized at 0.665637 ms.
  - The self-copy branch was removed again.

## 2026-05-29 K3 Small Tail-Reduce Checkpoint

- Added same-kernel local-rank tail-reduce to the K3 small low-latency C pack5 path.
- The path still writes real 8-rank combine rows first, then reduces the local rank combine buffer into `out[token, hidden]` inside the same kernel.
- Correctness:
  - 32 tokens: max_abs 0.000244141, mean_abs 1.86265e-09, bit_mismatch 1, value_mismatch 0.
  - 128 tokens: max_abs 0.000488281, mean_abs 1.97906e-09, bit_mismatch 4, value_mismatch 0.
- Timing:
  - 32 tokens: pure 0.155658 ms, fused tail-reduce 0.164469 ms, degradation +5.66%.
  - 128 tokens: pure 0.164406 ms, fused tail-reduce 0.184448 ms, degradation +12.19%.
- hipprof evidence:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_tail_reduce32`: 7 calls to `V2_K1_LowLatencyMaskedGroupGemmKernel<...,2048,...>`, average 166308 ns.
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_small_tail_reduce128`: 7 calls to the same fused kernel, average 185211 ns.
  - No standalone reduce kernel appears in HIPOPS.
- Scope:
  - This is still local-rank tail-reduce correctness, not full all-rank end-to-end combine-reduce acceptance.

## 2026-05-29 K3 Large PMC And Pair-Store Rejection

- Captured PMC for pure 1024, direct remote row-combine 1024, and copy-stage 1024.
- Key PMC evidence:
  - Pure: VMEM_RD about 462848, VMEM_WR about 393216, TCC_EA_WRREQ_STALL about 510555, TCP data stall about 2990733.
  - Direct row-combine: VMEM_RD about 487424, VMEM_WR about 393216, TCC_EA_WRREQ_STALL about 4497406, TCP data stall about 15071619.
  - Copy-stage: VMEM_RD about 786378, VMEM_WR about 442880, TCC_EA_WRREQ_STALL about 706997, TCP data stall about 5510910.
- Interpretation:
  - Direct remote epilogue is mainly hurt by remote scalar store/write stalls.
  - Copy-stage hides/reduces those write stalls with vector copies, but adds local reads and still occupies full GEMM LDS for copy-worker blocks.
- Rejected experiment:
  - A 32-bit adjacent-hidden pair-store direct epilogue passed 1024 correctness (`max_abs=0`) but was slower.
  - 1024 tokens: 0.711039 ms, worse than prior direct 0.699353 ms and copy-stage.
  - 4096 tokens: 2.38252 ms, worse than prior direct 2.14729 ms and copy-stage.
  - The pair-store code was reverted.
- Restored check:
  - Current copy-stage 1024 after rebuild remains correctness-clean (`max_abs=0`, value_mismatch 0) and measured 0.623141 ms in the latest stable run.

## 2026-05-29 K3 Large Tail-Reduce Local-Copy Filter

- Added a prototype-only filter for `K3_TAIL_REDUCE=1`: copy-stage now copies only rows whose destination is inside the local rank combine buffer.
- This does not affect normal K3 large row-combine with `K3_TAIL_REDUCE=0`.
- Correctness:
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- Timing:
  - 1024 tokens: 0.764443 ms, improved from 0.773269 ms but still +73.49% versus pure 0.440655 ms.
  - 4096 tokens: 2.39046 ms, improved from 2.44608 ms but still +84.01% versus pure 1.29907 ms.
- Judgment:
  - Kept as an isolated prototype improvement.
  - It is not a final large-token combine-reduce solution.

## 2026-05-29 K3 Large Tail-Reduce TopK-Slot Skip

- Added a `topk_idx` guard inside the large local-rank tail-reduce prototype so it skips topk slots whose expert is not owned by the local rank.
- Correctness:
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- Timing:
  - 1024 tokens: 0.744538 ms, improved from 0.764443 ms but still +68.96% versus pure 0.440655 ms.
  - 4096 tokens: 2.34016 ms, improved from 2.39046 ms but still +80.14% versus pure 1.29907 ms.
- Judgment:
  - Kept as an isolated local-rank prototype optimization.
  - Not valid as the final all-rank reduce behavior because final reduce must sum every topk slot after all ranks have written their contributions.

## 2026-05-29 K3 Large Copy-Stage Sync And Pointer-Broadcast Rejections

- Rechecked current K3 large copy-stage before new experiments:
  - 1024 tokens: correctness-clean; no-check rounds 0.616522/0.617434/0.617162 ms.
  - 4096 tokens: no-check rounds 1.79062/1.79354/1.81670 ms.
- Rejected sync experiments:
  - `atomicExch` tile-ready publication passed 8-rank correctness but regressed timing: 1024 min 0.622042 ms, 4096 min 1.82321 ms.
  - Raw-buffer GLC/cache-bypass flag publication/wait compiled but hung during correctness and destabilized the runtime; the experiment was reverted.
- Rejected row-pointer broadcast:
  - K3 large copy workers read `row_output_ptrs[row]` once per 32-vector row and broadcast the address across the half-wave.
  - Correctness passed for 1024/4096 with `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - Timing regressed: 1024 min 0.625124 ms and 4096 min 1.81698 ms, both worse than the restored copy-stage baseline.
  - The source and remote binary were reverted to the system-fence copy-stage baseline.

## 2026-05-29 K3 Large Copy-Worker Row-Tile Scheduling Rejection

- Tried a copy-worker scheduling variant where each worker owns row tiles and sweeps all hidden tiles for that row.
- Correctness passed on real 8-rank / 8-device runs:
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- Timing regressed:
  - 1024 tokens: rounds 0.75598/0.756242/0.755052 ms, min 0.755052 ms, +71.35% versus pure 0.440655 ms.
  - 4096 tokens: rounds 2.02709/2.02361/2.0275 ms, min 2.02361 ms, +55.77% versus pure 1.29907 ms.
- Judgment:
  - Rejected and reverted to the linear `tile += worker_count` copy-worker order.
  - The linear order consumes ready tiles earlier and remains the best current copy-stage baseline.

## 2026-05-29 Code Cleanup: Retired K1/ASM Entrypoints

- Removed retired ASM/balanced-ASM host launch plumbing from the V2 standalone harness.
- Removed active build-script entrypoints for `small-pull`, `small-symm-pull`, and `large-symm-pull`.
- Removed direct-pull template branches from the low-latency and large C kernels.
- Kept current K3 rowptr/combine/copy-stage/tail-reduce knobs for ongoing K3 large optimization.
- Validation:
  - Local `git diff --check` passed.
  - Remote V2 `hipcc` and `aicc` builds passed; `hipcc` only reported existing `mega_moe_dcu.cuh` missing-return warnings.
  - Remote `bash -n scripts/build_dcu_megamoe_v2.sh` passed.
- No GPU benchmark was run in this cleanup step because the cards were busy; next validation is quick K1/K3 smoke when cards are available.

## 2026-06-02 K3 Large Copy-Stage Sync Fix

- Re-established pure K3 C pack5 timing:
  - 1024 tokens: best 0.443114 ms.
  - 4096 tokens: best 1.30154 ms.
- Reproduced a 4096-token correctness failure in the 8-rank copy-stage path before the fix:
  - Example failure: max_abs 0.0143127, mean_abs 1.19198e-07, bit_mismatch 4743, value_mismatch 3557.
- Fixed ready publication by making every compute thread wait for its own VMEM stores before the block publishes its tile-ready flag. Also added the same wait before tail-reduce copy workers publish their done counter.
- Correctness after the fix, real 8 ranks / 8 devices:
  - 1024 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
  - 4096 tokens: max_abs 0, mean_abs 0, bit_mismatch 0, value_mismatch 0.
- Timing after the fix:
  - 1024 tokens: fused 0.626941 ms, +41.49% versus pure 0.443114 ms.
  - 4096 tokens: fused 1.83131 ms, +40.70% versus pure 1.30154 ms.
- Judgment:
  - The all-thread wait is required for reliable correctness; the old faster path published readiness too early.
  - Continue by sweeping copy-worker count and profiling the best corrected setting.

## 2026-06-02 K3 Large Copy-Worker Count Sweep

- Swept `K3_COPY_WORKERS=4/8/12/16` with real 8 ranks / 8 devices and `CHECK=1`.
- All candidates passed 1024/4096 correctness with max_abs 0, mean_abs 0, bit_mismatch 0, and value_mismatch 0.
- Timing:
  - 4 workers: 1024 best 1.43511 ms, 4096 best 5.41706 ms.
  - 8 workers: 1024 best 0.779663 ms, 4096 best 2.85274 ms.
  - 12 workers: 1024 best 0.629768 ms, 4096 best 2.04253 ms.
  - 16 workers: 1024 best 0.626672 ms, 4096 best 1.83729 ms.
- Judgment:
  - Keep 16 copy workers. Lower counts leave copy consumption too far behind the compute pipeline.
  - Profile the corrected 16-worker path before the next source change.

## 2026-06-02 K3 Large Corrected Copy-Stage Profile And Pair-Copy Rejection

- HIP trace stats for 1024 tokens:
  - Pure K3 large: 7 calls to the V2 large C kernel, average 445942 ns, 100% of kernel time.
  - Copy-stage: 7 calls to the V2 large C kernel, average 626171 ns, 100% of kernel time.
  - No standalone combine kernel appeared in the timed path.
- PMC evidence, copy-stage versus pure:
  - GRBM_COUNT: 887440 vs 629525, about 1.41x.
  - SQ_INSTS_VMEM_RD: 800087 vs 462848, about 1.73x.
  - SQ_INSTS_VMEM_WR: 442880 vs 393216, about 1.13x.
  - TCP data stall sum: 5586487 vs 3048266, about 1.83x.
  - TCC write-request stall sum: 690990 vs 405197, about 1.71x.
  - SQ_WAIT_INST_LDS is basically unchanged and LDS bank conflict remains 0.
  - Resource row: pure sgpr 32, copy-stage sgpr 64; both use 64 KiB LDS and no scratch.
- Pair-copy experiment:
  - Changed each copy-worker task to copy two contiguous `uint4` chunks.
  - Correctness passed for 1024 and 4096 with max_abs 0.
  - Timing regressed: 1024 best valid round 0.64113 ms with one huge outlier, 4096 best 1.93974 ms.
  - Reverted to the one-`uint4` copy baseline.
- Judgment:
  - The bottleneck is still memory traffic/stall from the copy-stage path, not LDS wait.
  - Simple loop coarsening is rejected; the next source change should target copy traffic or producer/consumer overlap rather than rowptr-load frequency alone.

## 2026-06-02 K3 Large Self-Copy Epilogue Rejection

- Tried a self-copy epilogue for non-tail-reduce K3 large row-combine:
  - Copy-worker blocks returned immediately.
  - Each compute block waited for its output stores and then vector-copied its own `out` tile into row-combine targets.
  - Tail-reduce kept the old copy-worker path.
- Correctness failed:
  - 1024 self-copy showed many zero rows in combine output: max_abs 0.112793, mean_abs 0.00718985, bit_mismatch 12582912, value_mismatch 12020011.
  - 4096 also failed with many zero rows.
  - Adding device-scope `__threadfence()` before self-copy did not fix 1024.
- Timing was rejected because correctness failed:
  - Invalid 1024 timing was around 0.573 ms.
  - Invalid 4096 timing was around 1.897 ms.
- Reverted and restored the copy-worker baseline:
  - Remote `aicc` rebuild passed.
  - 1024 real 8-rank / 8-device smoke passed with max_abs 0 and rounds 0.627935/0.628727 ms.
- Judgment:
  - Self-copy is not valid under the current buffer-store/output path.
  - Continue from the corrected copy-worker baseline.

## 2026-06-02 K3 Large Device-Scope Ready Fence Rejection

- Retried device-scope ready fencing only after the all-thread VMEM wait fix.
- Correctness passed for real 8-rank / 8-device 1024 and 4096 runs with max_abs 0.
- Timing did not justify the change:
  - 1024: rounds 0.625844/0.625700/0.626228 ms, only about 1 us faster than the system-fence baseline.
  - 4096: rounds 1.83775/1.83435/1.84104 ms, slightly slower than the system-fence baseline.
- Reverted to `__threadfence_system()`.
- Judgment:
  - Not a material win and historically more fragile, so keep the safer system-fence publication.

## 2026-06-02 K3 Large Sched-Barrier Ready-Publish Rejection

- Tried adding scheduler barriers around the corrected all-thread VMEM wait before tile-ready publication.
- Correctness passed for 1024 and 4096 real 8-rank / 8-device runs with max_abs 0.
- First timing looked slightly better at 4096, but repeat measurement did not hold:
  - First 4096 run: best 1.82992 ms.
  - Repeat 4096 run: best 1.83324 ms.
- Reverted.
- Judgment:
  - No stable improvement over the corrected baseline, so do not keep it.

## 2026-06-02 K3 Large Direct Remote-Store Route Recheck

- Rechecked the direct row-combine path after the corrected copy-stage baseline, using real 8 ranks / 8 devices.
- Correct direct rowptr baseline:
  - 1024 tokens: rounds 0.701824 / 0.701989 / 0.701903 ms.
  - 4096 tokens: rounds 2.26419 / 2.28274 / 2.27131 ms.
- `--k3-sort-rows 1` keeps the normal combine row mapping but sorts route tasks by combine row before launch:
  - Correctness passed for 1024 and 4096 with max_abs 0.
  - 1024 tokens: rounds 0.701574 / 0.702811 / 0.701925 ms, effectively unchanged.
  - 4096 tokens: rounds 2.21643 / 2.22685 / 2.20800 ms, about 2.5% better than unsorted direct rowptr.
  - Kept as an explicit experimental switch because it is correctness-clean and useful for direct-store studies.
- Rejected direct-store variants:
  - `--k3-combine-linear 1` was faster but failed 4096 correctness, so it remains invalid.
  - Per-scalar buffer-store from row address passed correctness but slowed to about 0.940747 ms at 1024 and 2.52924 ms at 4096.
  - Precomputing row buffer resources passed correctness but still slowed to about 0.867 ms at 1024 and 2.500 ms at 4096.
- Source cleanup:
  - Removed the rejected row-resource precompute/store experiment from `k1_groupgemm_v2.cpp`.
  - Current source keeps the corrected all-thread VMEM wait and system-scope ready publication.
  - The temporary sort switch was later cleaned up; copy-stage sorted routing is now the default active path.
- Judgment:
  - The direct remote-store route is still correctness-viable but not performance-viable yet.
  - Next direct-route work should target store issue/coalescing or a vectorized safe remote-store pattern, not per-row resource construction.

## 2026-06-02 K3 Large Sort And Store-Policy Experiments

- Synced the cleaned V2 source to remote, rebuilt with `aicc`, and re-established current 8-rank denominators:
  - Pure K3 large 1024: best `0.444361` ms, correctness max_abs 0.
  - Pure K3 large 4096: best `1.30086` ms, correctness max_abs 0.
  - Corrected copy-stage 1024: best `0.627143` ms, `+41.13%`, correctness max_abs 0.
  - Corrected copy-stage 4096: best `1.83152` ms, `+40.79%`, correctness max_abs 0.
- Direct row-combine recheck:
  - Direct unsorted 1024: best `0.701552` ms, correctness max_abs 0.
  - Direct unsorted 4096: best `2.23999` ms, but correctness failed with `max_abs=0.0145264`, `bit_mismatch=191`, `value_mismatch=137`; rejected.
  - Direct sorted 1024 (`--k3-sort-rows 1`): best `0.702680` ms, correctness max_abs 0.
  - Direct sorted 4096: best `2.21777` ms, correctness max_abs 0.
- Corrected copy-stage plus sorted rows:
  - CHECK=1 quick run: 1024 best `0.625799` ms, 4096 best `1.82681` ms, both max_abs 0.
  - CHECK=0 five-round timing:
    - 1024 unsorted best `0.626037` ms; sorted best `0.624660` ms.
    - 4096 unsorted best `1.83129` ms; sorted best `1.82607` ms.
  - Added a temporary `K3_SORT_ROWS` script switch while evaluating sort behavior. This switch was later removed during cleanup.
- Profile evidence for the current best 4096 copy-stage+sort path:
  - Artifact directory: `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_copy_stage_sort4096`.
  - `hipprof --hip-trace --stats` with warmup 2 / repeat 5 / CHECK=0 shows 7 calls to `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128...`, average `1,831,358` ns, 100% of HIPOPS kernel time.
  - No standalone combine kernel appears in the timed loop.
- Store-policy probe:
  - Added temporary probe `hygon_tmp/dcu_megamoe_v2_probe/global_store_short_probe.cpp`.
  - Verified `global_store_short %addr, %value, off` and `global_store_short %addr, %value, off glc slc` compile on gfx938 with `aicc`.
  - Tried applying `global_store_short ... glc slc` to direct rowptr BF16 stores.
  - Correctness passed for direct sorted 1024/4096, but timing regressed badly: 1024 about `1.50064` ms and 4096 about `5.60513` ms.
  - Reverted the main source to the normal pointer store and force-rebuilt the remote binary.
- Hygiene:
  - The first remote loop accidentally let PowerShell expand `$mode/$tok`, so it ran default build only; reran with escaped variables.
  - A remote source copy temporarily contained trailing NUL bytes, detected by `aicc` warnings. Local source had no NULs; the remote file was truncated and recopied before rebuilding.
- Judgment:
  - Keep sorted routing for copy-stage as a small correctness-clean optimization, but do not keep the temporary script environment switch.
  - Reject direct unsorted 4096 and `global_store_short glc/slc`.
  - Current best K3 large copy-stage+sort remains about `+40.4%` at 4096, still above the <=25% target.
## 2026-06-02 - K3 Large Tail Reduce Optimization

- Implemented K3 large same-kernel local-rank tail-reduce optimizations in `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`:
  - host-precomputed local topk slot mask;
  - separate final `d_k3_tail_out` buffer so zero-mask tokens skip writes;
  - active token list so reduce only scans tokens with local-rank contribution.
- Remote validation:
  - `make -C csrc/kernels/dcu_megamoe_v2 aicc` passed.
  - After cleanup, both `aicc` and `hipcc` builds passed; only existing `mega_moe_dcu.cuh` missing-return warnings appeared under hipcc.
  - Real 8-rank / 8-device K3 large copy-stage tail-reduce passed correctness for 1024 and 4096 tokens.
- Timing and correctness:
  - Pure K3 C pack5 1024: best `0.444784 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - Fused tail-reduce 1024 after cleanup: `0.636959/0.637543 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, degradation `+43.21%`.
  - Pure K3 C pack5 4096: best `1.30267 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`.
  - Fused tail-reduce 4096 after cleanup: `1.88046/1.88045 ms`, `max_abs=0`, `mean_abs=0`, `bit_mismatch=0`, `value_mismatch=0`, degradation `+44.35%`.
- Profile evidence:
  - `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_tail_reduce_active4096`
  - hipprof reports 7 HIPOPS calls to the same V2 large C pack5 fused kernel, average `1,905,278 ns`, 100% of HIPOPS kernel time, with no standalone combine/reduce kernel in the timed path.
- Judgment:
  - The tail-reduce path is correctness-clean and meaningfully faster than the earlier prototype, but still misses the <=25% K3 large target.
  - The local tail-row list copy experiment was removed before commit because it added complexity for only a marginal 4096-token win.
  - Next work should focus on reducing copy-stage/synchronization overhead or redesigning the in-kernel combine-reduce schedule.
