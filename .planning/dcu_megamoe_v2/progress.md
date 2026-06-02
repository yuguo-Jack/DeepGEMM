# Progress Log: DCU MegaMoE V2

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
