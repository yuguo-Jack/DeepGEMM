# Task Plan: DCU MegaMoE V2

## Goal
Build an isolated DCU MegaMoE V2 prototype in git-tracked source that uses a three-stage fused flow: K1 dispatch-pull plus L1 FP8 groupgemm, K2 SwiGLU plus quant, and K3 L2 FP8 groupgemm plus combine reduce.

## Current Phase
Milestone G/Phase 9 performance gate: the real MegaMoE V2 staged flow is connected and correctness-clean for the requested 4-rank cases. The active focus is now same-size integrated K1/K3 stage timing versus the corresponding V2 pure groupgemm denominator, then repairing the largest degradation inside the real-flow fused kernels. Latest accepted K1 normal work reduced 4-rank cross-rank normal 4096 K1 from the earlier `7.527326 ms` class to about `4.93-4.95 ms` using source-rank grouped metadata plus first-row-tile x-block cooperative metadata scanning. Latest accepted K3 normal work restored `K3_COPY_WORKERS=16`, added topk=6/all-slot tail-reduce fast paths, hoisted combine-buffer vector indexing, specialized dense identity tail tokens, accepted the helper/direct full-topk6 reduce path, and uses gfx938 packed BF16 conversion for tail-reduce packback; on `HIP_VISIBLE_DEVICES=2,3,4,5`, normal 4096 K3 is now sanity-restored at `2.930398 ms` versus same-device pure K3 `1.332190 ms` (`+119.96%`). Direct K3 shortcuts have both been rejected: rowptr/no-copy epilogue scatter worsened K3 to `10.009423 ms`, and direct-local-reduce via `output_index` worsened K3 to `3.316477 ms`. The next step is to compare fused and pure code paths for extra integrated work, while preserving contiguous GEMM writes and avoiding extra per-vector `output_index` loads/branches.

## Success Criteria
- V2 source, tests, build scripts, and docs are isolated from existing DCU MegaMoE, large_opt, and big-fused paths.
- V2 owns its weight layout transform; tests explicitly transform into pack5-style V2 layout before timing.
- Correctness target: max_abs <= 1e-3, with max_abs, mean_abs, and mismatch reported.
- Benchmark loop reports tokens tested, pure groupgemm time, fused time, and degradation ratio.
- Quick benchmark coverage must include small tokens 32/128 and large tokens 1024/4096 for any stage that has both small and large execution paths.
- K1 fused target: <= 20% slower than pure K1 groupgemm after dispatch-pull fusion.
- K3 fused target: <= 25% slower than pure K3 groupgemm after combine-reduce fusion.
- Communication-fused acceptance requires a real 4-rank or 8-rank run. A 1-rank local pull or 2-DCU smoke is useful for debugging but is not an acceptance result.
- Current tuning cadence: use 4-rank runs first for faster iteration and bottleneck repair; once 8 clean cards are available, produce matching 8-rank correctness/performance data before final handoff.
- Communication work must be hidden inside the compute kernel. Timed K1/K3 fused paths must not add standalone dispatch/prebuild/combine kernels.
- hipprof evidence for communication-fused checkpoints must show the expected compute/fused kernel path and explicitly call out any setup kernels outside the timed loop.
- No hipMalloc, hipFree, D2H, or unnecessary new kernels in execution path.
- Temporary logs, dumps, profiles, and experiment caches stay under hygon_tmp or remote /workspace/DeepGEMM/hygon_tmp.
- V2 plan/progress/findings/overview are maintained only under `.planning/dcu_megamoe_v2/`; do not create parallel V2 progress/findings files under `docs/`.
- Baseline behavior, baseline layout transform, and existing DCU MegaMoE / large_opt / big-fused call paths must not change.
- V2 real-flow integration uses independent V2 files, symbols, build entries, tests, and wrapper APIs as much as possible. Existing `megamoe` files may only receive minimal opt-in glue when required to expose the V2 path.
- Do not keep growing the standalone prototype `k1_groupgemm_v2.cpp` as the production V2 implementation. Use it and its Makefile/script as implementation and build references, then split real-flow V2 code into organized K1/K2/K3/layout/runtime wrapper files similar to the existing DCU MegaMoE structure.
- All V2 runtime sizes must execute staged fused K1/K2/K3. Small and large token paths may use different prototype kernels, but neither path may fall back to big fused.
- Do not implement automatic small/large threshold selection in the first real-flow integration. Add `MEGAMOE_DCU_V2_BACKEND=ll|normal`, where `ll` forces the low-latency backend and `normal` forces the high-throughput C pack5 backend. Measure both backends across the full token sweep later before choosing the threshold.
- Integrated real-flow correctness must pass against the existing baseline at tokens 32, 512, 1024, and 2050 with max_abs <= 1e-3.
- Integrated performance tests keep the existing quick performance size set: small 32/128 and large 1024/4096.
- Integrated K1/K3 fused stage performance must be compared against the corresponding V2 pure groupgemm timing with the same backend family, layout, rank count, visible device set, warmup/repeat policy, and token count. Report fused time, pure groupgemm time, and degradation ratio.
- Do not modify the pure groupgemm prototype/harness just to make an integration comparison easier. Treat it as the stable denominator; optimize the real MegaMoE V2 flow around it.
- Hard performance acceptance rule: for each accepted same-size run, V2 must be faster than the current DCU MegaMoE V1 path under the same rank count, token count, route mode, warmup/repeat policy, and timing harness. Correctness-only cross-rank results are not enough for handoff.
- Minimize nonessential H2D copies in the V2 path. Static/precomputed host metadata is acceptable during setup, but repeated per-iteration H2D control traffic in the execution path must be avoided or explicitly justified.
- Avoid standalone rank barriers in the V2 eager execution path. Communication synchronization should be embedded into the communication/compute fused kernels where possible; use an external barrier only after documenting why in-kernel synchronization is insufficient.
- Before any remote build, test, benchmark, or profile run, scan the container and host for residual V2 pytest/build/prototype processes. Kill only processes that clearly belong to the current V2 work; leave unrelated user workloads untouched and choose free devices explicitly.

## Execution Loop
- Before starting each milestone, read this task plan plus progress.md/findings.md.
- Each milestone must be broken into small checklist items. After completing one item, update its status, append a progress checkpoint, then re-read the next checklist item before continuing.
- Update `findings.md` whenever a reusable technical decision, interface contract, benchmark conclusion, correctness failure root cause, rejected experiment, or profiling conclusion is established. Do not wait until the end of a milestone to record conclusions that will affect later implementation.
- Do not batch many unchecked items and mark them complete only at the end.
- Do not proceed from a failed experiment by widening scope. Record the failure, revert or quarantine it, and pick the next smallest corrective step.
- Every accepted communication benchmark must state rank count, token count, pure time, fused time, degradation, correctness, and profile evidence.
- After real communication is connected, use integrated K1/K3b fused-stage-vs-pure-groupgemm comparison as the first diagnostic: record the degradation ratio, profile the cause, and keep trying to repair large gaps. If a checkpoint still carries a large gap while V2 beats V1, document the current bottleneck and next repair attempt instead of treating the pure-groupgemm gap itself as a final acceptance blocker.
- When cards become available again, resume from the performance gate first: complete missing cross-rank `normal` perf rows, collect same-size pure groupgemm denominators, profile the K3b and K1 overhead buckets, apply the smallest kernel/runtime fix needed to reduce degradation, and rerun the same 4-rank or 8-rank comparison before moving to unrelated features.
- Current K3 performance gate: after cards are free, clear only residual V2 processes, then measure pure K3 groupgemm, integrated local-only K3, and integrated cross-rank K3 on the same visible 4-rank device set. Repair the largest fused-vs-pure degradation before revisiting the K3 all-rank tail-reduce synchronization design.

## Phases

### Phase 1: Discovery and Baseline Mapping
- ✅ Read K1_groupgemm_fp8 README and best C groupgemm skeletons.
- ✅ Map existing DCU MegaMoE K1/K2/K3 implementations and tests.
- ✅ Map CUDA MegaMoE overlap strategy for K1 dispatch pull and K3 combine.
- ✅ Record layout, APIs, build commands, and reusable pieces in findings.md.
- **Status:** complete

### Phase 2: Isolated V2 Skeleton
- ✅ Create csrc/kernels/dcu_megamoe_v2 or equivalent isolated source directory.
- ✅ Create independent build entry under scripts.
- ✅ Create independent Python test entry under tests.
- ✅ Create and consolidate V2 overview/progress/findings under `.planning/dcu_megamoe_v2/`.
- **Status:** complete

### Phase 3: Pure GroupGEMM Port
- ✅ Port/extract the best C FP8 groupgemm skeletons into the V2 build path.
- ✅ Validate independent compile path on remote DCU container.
- ✅ Establish pure groupgemm timing for quick tokens: small 32/128, large 1024/4096.
- **Status:** complete

### Phase 4: V2 Weight Layout Transform
- ✅ Implement pack5 layout transform helper owned by V2.
- ✅ Add tests that verify logical to physical n mapping.
- ✅ Ensure layout transform is excluded from benchmark timing.
- **Status:** complete

### Phase 5: K1 Fused Dispatch Pull Prototype
- ✅ Implement initial small-token local-pull fused kernel mode and launcher independently.
- ✅ Start with correctness-first dispatch-pull plus L1 groupgemm.
- ✅ Benchmark quick small tokens and compare against pure K1 groupgemm.
- ✅ Capture launch-level profile evidence.
- ✅ Extend K1 fused metadata from local deterministic pull map to sym-buffer prebuild/dispatch-pull contract.
- ✅ Upgrade small-token K1 communication validation from 2-DCU smoke to 4-rank or 8-rank acceptance.
- ✅ Add large-token K1 fused path.
- ✅ Profile real 4/8-rank overlap with hipprof/PMC and ISA evidence.
- **Status:** complete

### Phase 6: K2 Integration
- ✅ Reuse existing optimized DCU SwiGLU plus quant implementation where possible.
- ✅ Keep V2 API/build/test entry independent.
- ✅ Validate stage-level correctness.
- [ ] Add K2 timing rows once K1/K2/K3 staged harness is connected.
- **Status:** complete for correctness, timing deferred to staged harness

### Phase 7: K3 C Pack5 Fused Combine Prototype
- ✅ Remove rejected K3 ASM wrapper/code-object path and kpack2 layout from V2 source/tests/docs.
- ✅ Restore K3 to the same V2 pack5 layout contract used by K1.
- ✅ Add pack5 tests that cover both L1 and L2 logical shapes.
- ✅ Adapt the C groupgemm harness for K3 L2 shape: N=4096, K=2048, pack5.
- ✅ Establish pure K3 C pack5 timing for small quick tokens 32 and 128.
- ✅ Establish pure K3 C pack5 timing for large quick tokens 1024 and 4096.
- ✅ Implement K3 C row-combine in the groupgemm epilogue / same-kernel copy-stage with no extra combine kernel.
- ✅ Validate K3 C row-combine correctness for small tokens 32/128 on real 8-rank communication targets.
- ✅ Validate K3 C row-combine correctness for large tokens 1024/4096 on real 8-rank communication targets.
- [ ] Compare K3 C fused against pure K3 C pack5 for small and large tokens; target <=25% degradation.
- [ ] Profile real 4/8-rank overlap and log hipprof/PMC/ISA evidence.
- [ ] Add in-kernel combine reduce / tail reduce; current row-combine stores partial rows and is not final combine-reduce acceptance.
- **Status:** in_progress

### Rejected / Quarantined K3 Work
- ✅ K3 ASM/kpack2 prototype was built and measured, but it violates the V2 requirement to use C groupgemm and the unified pack5 layout.
- ✅ Treat all K3 ASM/kpack2 timings as invalid for acceptance. Keep them only as failure history in findings/progress.
- ✅ Remove the corresponding source/test/doc artifacts from active V2.

## Big Plan With Small Plans

### Milestone A: Correct The Source Tree
- ✅ Delete K3 ASM V2 extension source.
- ✅ Strip K3 ASM helpers from stages.py.
- ✅ Remove K3 kpack2 layout helpers/tests.
- ✅ Update docs/progress/findings to mark K3 ASM as rejected.
- ✅ Verify local py_compile, git diff --check, and remote pytest.

### Milestone B: Lock Unified Pack5
- ✅ Add L1 pack5 test fixtures.
- ✅ Add L2 pack5 test fixtures for N=4096, K=2048 shape reduced where needed.
- ✅ Add Python-vs-C pack5 byte mapping check.
- ✅ Confirm layout transform is outside benchmark timing.

### Milestone C: K1 Communication Acceptance
- ✅ Re-run small-token K1 fused communication with 4 ranks or 8 ranks.
- ✅ Require the communication pull/staging to happen inside the K1 compute kernel.
- ✅ Record 32/128 pure vs fused timing, correctness, and hipprof evidence.
- ✅ Continue K1 large-token C pack5 dispatch-pull fusion only after the small-token 4/8-rank acceptance is clean.

### Milestone D: K1 Large-Token C Pack5 Fused
- ✅ Add large-token dispatch-pull/staging path on the C pack5 groupgemm kernel.
- ✅ Validate tokens 1024/4096 with 8 ranks.
- ✅ Target <=20% degradation against pure large-token K1 C pack5.

### Milestone E: K3 Pure C Pack5
- ✅ Make pure K3 C pack5 run for N=4096, K=2048 without VMFault.
- ✅ Validate small tokens 32/128 correctness max_abs <= 1e-3.
- ✅ Validate large tokens 1024/4096 correctness max_abs <= 1e-3.
- ✅ Record stable pure K3 C timings for 32/128/1024/4096.
- ✅ Decide and record small-token K3 C tiling separately from large-token K3 C tiling if the lowlat and MT256 paths diverge.

### Milestone F: K3 C Pack5 Fused Combine
- ✅ Add row-combine pointer output in C groupgemm epilogue.
- ✅ First validate identity row pointers for small tokens 32/128.
- ✅ First validate identity row pointers for large tokens 1024/4096.
- ✅ Then validate real 8-rank combine targets for 32/128/1024/4096 correctness.
- ✅ Hit <=25% degradation for small-token K3 row-combine 32/128.
- ✅ Add correctness-first same-kernel local-rank tail-reduce prototype.
- ✅ Add small-token same-kernel local-rank tail-reduce prototype for 32/128.
- [ ] Hit <=25% degradation for large-token K3 row-combine 1024/4096.
- [ ] Optimize same-kernel tail reduce and extend it from local-rank prototypes to final all-rank combine-reduce acceptance.

### Milestone G: End-To-End V2
- ✅ Map the real `tests/test_mega_moe_dcu.py` call stack: sym buffer setup, FP8 input cast, baseline layout transform, dispatch metadata, K1, K2, K3/combine, stats, and output comparison.
- ✅ Add an independent V2 real-flow package/wrapper entry, preferably under `megamoe/dcu_megamoe_v2/`, without replacing existing baseline or large_opt code.
- ✅ Keep V2 tests in the independent `tests/test_dcu_megamoe_v2.py` entry; extend that file for real-flow V2 coverage instead of creating a second V2 test file.
- ✅ Create the organized V2 source layout before moving production logic out of the prototype file:
  - `megamoe/dcu_megamoe_v2/__init__.py`
  - `megamoe/dcu_megamoe_v2/api.py`
  - `megamoe/dcu_megamoe_v2/layout.py`
  - `megamoe/dcu_megamoe_v2/runtime.py`
  - `megamoe/dcu_megamoe_v2/K1_fused/`
  - `megamoe/dcu_megamoe_v2/K2_fused/`
  - `megamoe/dcu_megamoe_v2/K3_fused/`
  - matching V2 C++/HIP extension files under those K-stage directories.
- ✅ Keep `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`, `Makefile`, and `scripts/build_dcu_megamoe_v2.sh` as prototype/reference harness material; do not rely on them as the long-term real-flow implementation surface.
- ✅ Add a V2 weight transform API for pack5 L1/L2. It must be separate from baseline `transform_fp8_weights_for_mega_moe` and explicit in tests.
- [ ] Expose V2 K1/K3 kernels through a V2 pybind/extension path usable from Python, rather than relying only on the standalone benchmark `main`.
  - ✅ Add K1/K3 Python extension loader scaffolds and dedicated V2 `.cu` extension surfaces.
  - ✅ Split callable K1/K3 prototype kernel launch entry points out of the standalone `main` path.
  - ✅ Add a `DCU_MEGAMOE_V2_KERNEL_ONLY` include mode so the extension path skips standalone host harness code.
  - ✅ Register V2 K1/K3 pybind extensions and V2 Python subpackages in `setup.py` so real-flow execution uses prebuilt modules instead of default torch JIT.
  - ✅ Validate remote `build_ext --inplace` for the V2 K1/K3 extension modules.
  - [ ] Replace the transitional include bridge with stage-owned C pack5 kernel headers/sources before finalizing real-flow implementation.
- [ ] Connect K1 C fused, existing optimized K2 wrapper, and K3 C fused into one eager staged V2 function.
  - ✅ Add V2 runtime workspace views over `sym_buffer.route_scratch` for L1 output, K2 act FP8/scale, staged X, route scratch i32, grid barriers, row_expert, row_output_ptrs, local_topk_mask, and tail_tokens.
  - [ ] Add a V2 metadata/prebuild path that produces route_weights, row_expert/m_indices, output_index, row_combine_ptrs, and local_topk_mask without repeated per-iteration H2D.
    - ✅ Add explicit V2 metadata views for `m_indices`, `row_combine_ptrs`, and `output_index` as route_scratch slices.
    - ✅ Add K1 ll in-kernel metadata writeback for `route_weights`, `row_expert/m_indices`, `output_index`, `row_combine_ptrs`, `local_topk_mask`, and dense `tail_tokens`.
    - ✅ Add K1 ll metadata correctness test against sym_buffer topk/topk_weights/combine layout.
    - ✅ Extend K1 normal C row-stage metadata from deterministic route helper to real topk-driven row assignment.
    - [ ] Decide whether dense `tail_tokens[:num_tokens]` is acceptable or replace with compact active-token output after correctness is connected.
  - ✅ Call K1 ll/normal extension launchers from `runtime.py`.
  - ✅ Call existing V2 K2 wrapper with K1 route weights and active row count.
  - ✅ Call K3 ll/normal extension launchers from `runtime.py`.
  - ✅ Run a minimal real-kernel K1/K2/K3 runtime smoke for `ll` and `normal`.
- ✅ Route all sizes through staged fused K1/K2/K3; use `MEGAMOE_DCU_V2_BACKEND=ll|normal` to force either the low-latency backend or the high-throughput C pack5 backend for the first integration.
- [ ] Add stage-level correctness checks for K1, K2, and K3 against the existing baseline/reference path.
- ✅ Add real-flow end-to-end correctness checks against baseline at tokens 32, 512, 1024, and 2050.
- ✅ Benchmark integrated quick performance at tokens 32/128/1024/4096 with stage breakdown. The active comparison target is now the corresponding pure groupgemm timing, not standalone fused prototype timing.
- ✅ Defer automatic threshold selection until both V2 backends have been run across the full token sweep; record the advantage interval for each backend before choosing the threshold.
- [ ] Keep K3 large performance caveat recorded: current fused large path is correctness-clean but still above the <=25% target versus pure K3 C pack5.
- [ ] Add uneven tokens per rank and CUDA graph replay only after eager real-flow correctness is stable.
  - ✅ Add first eager uneven-token local-only correctness validation with `dispatch_num_tokens=-1` reading per-rank sym-buffer runtime token counts.
  - [ ] Extend uneven-token validation to real cross-rank routing/remote communication.

## Original Prompt Recheck Checklist
- ✅ V2 source remains isolated under V2-specific source/test/script/doc paths, not under hygon_tmp.
- ✅ hygon_tmp contains only remote logs, profiles, temporary code objects, dumps, and experiment caches.
- ✅ Existing dcu_megamoe, large_opt 3-stage, and big-fused paths are not modified.
- ✅ V2 symbols, tests, build entry, and docs remain independent.
- ✅ K1 small-token path uses fused dispatch/prebuild pull plus L1 FP8 groupgemm in one compute kernel.
- ✅ K1 large-token path uses the C pack5 groupgemm skeleton with dispatch/prebuild pull fused in one compute kernel.
- ✅ K2 reuses the optimized DCU SwiGLU+quant implementation through an isolated V2 boundary.
- [ ] K3 small-token path uses C pack5 L2 groupgemm plus combine fused in one compute kernel.
- [ ] K3 large-token path uses C pack5 L2 groupgemm plus combine fused in one compute kernel.
- [ ] K3 cross-rank tail reduce has accepted in-kernel all-rank synchronization before final local combine read. Candidate barrier is compiled but not correctness-accepted yet.
- ✅ L1 and L2 weights use the unified V2 pack5 layout; no baseline layout and no K3 kpack2/ASM layout in accepted paths.
- ✅ Tests explicitly perform V2 layout transform, and layout transform is excluded from benchmark timing.
- [ ] Stage-level correctness is collected for K1, K2, and K3 before end-to-end correctness. Current real-flow references cover K1 metadata/stats plus K2/K3 numerical stage checks at 32 tokens; K1 direct L1 numerical reference remains diagnostic only.
- ✅ End-to-end correctness compares against baseline with max_abs <= 1e-3.
- ✅ Quick optimization loop covers K1 and pure K3 small 32/128 and large 1024/4096; K3 fused combine quick loop remains in progress.
- ✅ Full sweep later covers 32/64/128/256/512/1024/2048/4096/8192.
- ✅ Communication acceptance uses 4 ranks or 8 ranks and confirms communication is hidden inside compute kernels for accepted K1 checkpoints.
- [ ] No execution path introduces hipMalloc, hipFree, D2H, or unnecessary new kernels.
- [ ] No unnecessary new environment variables are added.
- [ ] Uneven tokens per rank support is added or explicitly planned after eager path.
- [ ] CUDA graph one-graph multi-size replay support is added or explicitly planned after eager path.
- [ ] Every accepted checkpoint records files changed, tokens, pure time, fused time, degradation, correctness, profile evidence, and next-step judgment.

### Phase 8: Real MegaMoE Flow Integration
- ✅ Add a V2 real-flow test that mirrors `tests/test_mega_moe_dcu.py` data generation, baseline call, output comparison, and cumulative expert stats comparison.
- ✅ Run correctness tokens 32, 512, 1024, and 2050 against the unchanged baseline.
  - ✅ 4-rank local-only route, backend `ll`: 32/512/1024 pass.
  - ✅ 4-rank local-only route, backend `normal`: 32/512 pass.
  - ✅ 4-rank local-only route, backend `ll`: 2050 pass.
  - ✅ 4-rank local-only route, backend `normal`: 1024 pass after the K1 normal LDS read waitcnt fix.
  - ✅ 4-rank local-only route, backend `normal`: 2050 pass.
- ✅ Report correctness metrics for K1, K2, K3, and end-to-end where stage references are available. End-to-end metrics and cumulative expert stats alignment are complete for the requested local-only real-flow tokens; K2/K3 numerical stage references pass at 32 tokens, and K1 metadata/stats pass with direct L1 numerical comparison kept diagnostic-only.
- ✅ Run integrated quick performance tokens 32/128/1024/4096 with stage breakdown. Recompute degradation against pure groupgemm before accepting performance.
- ✅ Record forced-backend results for both `ll` and `normal` V2 backends; defer automatic threshold selection to the later full-sweep phase.
- **Status:** in_progress, requested real-flow end-to-end correctness, cumulative stats alignment, and available stage-reference reporting complete; integrated quick performance, stage breakdown, and prototype timing comparison collected; K1 direct L1 diagnostic mismatch and K3/normal performance remain open follow-ups

### Phase 9: Full Sweep and Feature Follow-up
- ✅ Run full tokens 32/64/128/256/512/1024/2048/4096/8192.
- [ ] Add or plan uneven tokens per rank support.
- [ ] Keep remote pre-run cleanup in the loop: scan host/container for residual V2 processes before each build/test/profile, and record any cleanup in progress.md.
- ✅ Add real cross-rank route correctness validation for requested tokens 32, 512, 1024, and 2050 on 4 ranks.
- ✅ Strengthen K3 normal cross-rank combine semantics: replace local-expert-only tail mask with valid topk-slot/all-slot reduce, then rerun 4-rank correctness.
- [ ] Add real cross-rank performance validation, then compare integrated K1/K3 stage timings against corresponding pure groupgemm kernels.
  - [ ] Use 4-rank first for debug/tuning on currently free cards.
  - [ ] Complete missing cross-rank `normal` 128/1024/4096 performance rows.
  - ✅ First compare integrated K1/K3 stage timings against the corresponding pure groupgemm timings on 4 ranks for `ll` 32/128/512/1024 and `normal` 1024/2048/4096.
  - [ ] Profile K3b cross-rank overhead versus pure K3 groupgemm and attribute it to remote store, dense tail reduce, synchronization, row metadata, or runtime staging.
  - ✅ Profile K1 cross-rank overhead versus pure K1 groupgemm for the worst current row (`normal`, 4096); additional LL-specific profiling remains a follow-up if LL becomes the blocker.
  - [ ] Validate the build-only K1 normal direct-pull experiment on 4 ranks at `normal` 4096. Accept only if correctness passes and K1 stage improves versus prior `7.527326 ms`; otherwise revert the direct-pull activation.
  - [ ] Validate the K3 runtime-token tail-reduce cleanup after rebuild. It is a hygiene/perf fix for aligned, uneven, and future graph-capacity cases; accept only after 4-rank correctness remains clean and K3 stage timing does not regress.
  - [ ] Measure current DCU MegaMoE V1 same-size cross-rank performance later, after the pure groupgemm degradation is narrowed, using the same rank count, route mode, tokens, warmup/repeat, and timing harness.
  - [ ] Verify V2 is faster than DCU MegaMoE V1 at accepted same-size points before final performance acceptance.
  - [ ] Implement and validate targeted fixes for any V2-vs-V1 regression. Large fused-vs-pure groupgemm degradations should be repaired as much as possible; if a gap remains at a checkpoint while V2 still beats V1, explain the current bottleneck with evidence and keep a follow-up repair item open.
  - [ ] When 8 ranks are available, rerun accepted correctness/performance sizes on 8 ranks and record matching data.
- [ ] Add or plan CUDA graph one-graph multi-size support.
- **Status:** in_progress; 4-rank local-only full sweep passed for `ll` and `normal`, first 4-rank uneven-token local-only correctness passed, and 4-rank cross-rank correctness passed for both backends at 32/512/1024/2050. Current integrated advantage interval is `ll` for 32-512 tokens and `normal` for 1024-8192 tokens, with the crossover between 512 and 1024. Automatic threshold selection is not implemented yet because K3 large performance, cross-rank performance, same-size V1 performance comparison, integrated fused-vs-pure groupgemm degradation analysis, and CUDA graph support remain open.

### Phase 10: Cleanup, Commit, and Handoff
- [ ] Ensure temporary outputs are not in git.
- [ ] Update docs with final measurements and caveats.
- [ ] Commit coherent milestones locally when useful; do not push.
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use .planning/dcu_megamoe_v2 for persistent planning state | User explicitly requested Planning with Files and this avoids mixing with source docs. |
| Keep V2 under independent source/test/build/doc entries | Prevents regressions in existing DCU MegaMoE, large_opt 3-stage, and big-fused paths. |
| Use pack5 as the V2 recommended weight layout | Matches the current K1_groupgemm_fp8 baseline recommendation and avoids reusing baseline layout accidentally. |
| Treat layout transform as preparation, not timed benchmark work | Matches correctness and benchmark requirements. |
| Maintain V2 planning/docs only under `.planning/dcu_megamoe_v2/` | Avoids duplicated progress/findings under `docs/`; overview content was moved into `.planning`, and progress/findings are kept in the active `.planning` logs. |
| Integrate V2 through an opt-in independent real-flow wrapper first | Preserves the existing baseline and large_opt paths while allowing `tests/test_mega_moe_dcu.py`-style correctness/performance validation. |
| Pause K3 large performance tuning while integrating | Current K3 large fused path is correctness-clean but above target; integration should expose real-flow overhead before further kernel tuning. |
| Use `MEGAMOE_DCU_V2_BACKEND=ll|normal` before automatic thresholding | Lets low-latency and high-throughput V2 backends be validated independently across all sizes before choosing the real dispatch threshold. |
| Minimize H2D and standalone barriers in V2 execution | Keeps communication and synchronization work hidden inside fused kernels and avoids overhead beyond the pure groupgemm denominator. |
| Split real-flow V2 implementation by stage and responsibility | Prevents the standalone prototype C++ file from becoming the production implementation and keeps V2 maintainable like existing DCU MegaMoE stage directories. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Missing superpowers skill at listed .system path | 1 | Loaded actual installed skill path from plugin cache and treated it as skill-selection context. |
| K3 large raw-buffer GLC flag synchronization hung during 8-rank correctness and the container exited | 1 | Reverted the experiment, restored the system-fence copy-stage path, and recorded that this sync variant must not be retried without a standalone microbench. |
| K3 large copy-worker row-pointer half-wave broadcast regressed timing | 1 | Reverted the experiment after 1024/4096 8-rank correctness passed but timing worsened. |
| K3 large copy-worker row-tile scheduling regressed timing | 1 | Reverted to the linear tile scheduling baseline after 1024/4096 8-rank correctness passed but timing worsened. |
| K1/K3 extension default torch JIT compile stayed in hipcc for more than 15 minutes even after splitting raw `.cu` from pybind `.cpp` | 2 | Rejected default JIT as the real-flow build path; changed K1/K3 loaders to require prebuilt setup.py extensions and added a kernel-only include mode to reduce standalone host harness compile work. |
| Local Windows Python has no pytest module | 1 | Used local `py_compile` plus a direct runtime workspace smoke, then ran authoritative pytest in the remote container. |
| K1 normal cross-rank metadata counted barrier hung at 32 tokens | 1 | Rejected counted metadata-owner barrier; replaced it with expert0 in-kernel rank barrier plus launch-epoch flag for other metadata owner blocks. |
| Real-flow distributed test hit `MASTER_PORT` EADDRINUSE during repeated spawned runs | 1 | Replaced random fixed-range port selection with socket-reserved local free ports in the V2 test harness. |

## Round Output Checklist
Each progress checkpoint should report:
- Files changed.
- Tokens tested.
- Pure groupgemm time.
- Fused time.
- Degradation ratio.
- Correctness max_abs, mean_abs, mismatch.
- Profile evidence and next-step judgment.
