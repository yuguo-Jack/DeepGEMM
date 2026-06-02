# Task Plan: DCU MegaMoE V2

## Goal
Build an isolated DCU MegaMoE V2 prototype in git-tracked source that uses a three-stage fused flow: K1 dispatch-pull plus L1 FP8 groupgemm, K2 SwiGLU plus quant, and K3 L2 FP8 groupgemm plus combine reduce.

## Current Phase
Milestone F: K3 C pack5 real 8-rank row-combine is correctness-clean. Small-token row-combine and local-rank same-kernel tail-reduce are within target, while large-token combine still needs optimization and full in-kernel combine reduce / tail reduce.

## Success Criteria
- V2 source, tests, build scripts, and docs are isolated from existing DCU MegaMoE, large_opt, and big-fused paths.
- V2 owns its weight layout transform; tests explicitly transform into pack5-style V2 layout before timing.
- Correctness target: max_abs <= 1e-3, with max_abs, mean_abs, and mismatch reported.
- Benchmark loop reports tokens tested, pure groupgemm time, fused time, and degradation ratio.
- Quick benchmark coverage must include small tokens 32/128 and large tokens 1024/4096 for any stage that has both small and large execution paths.
- K1 fused target: <= 20% slower than pure K1 groupgemm after dispatch-pull fusion.
- K3 fused target: <= 25% slower than pure K3 groupgemm after combine-reduce fusion.
- Communication-fused acceptance requires a real 4-rank or 8-rank run. A 1-rank local pull or 2-DCU smoke is useful for debugging but is not an acceptance result.
- Communication work must be hidden inside the compute kernel. Timed K1/K3 fused paths must not add standalone dispatch/prebuild/combine kernels.
- hipprof evidence for communication-fused checkpoints must show the expected compute/fused kernel path and explicitly call out any setup kernels outside the timed loop.
- No hipMalloc, hipFree, D2H, or unnecessary new kernels in execution path.
- Temporary logs, dumps, profiles, and experiment caches stay under hygon_tmp or remote /workspace/DeepGEMM/hygon_tmp.
- V2 plan/progress/findings/overview are maintained only under `.planning/dcu_megamoe_v2/`; do not create parallel V2 progress/findings files under `docs/`.

## Execution Loop
- Before starting each milestone, read this task plan plus progress.md/findings.md.
- After each small plan is completed, update progress.md and re-read this task plan before selecting the next change.
- Do not proceed from a failed experiment by widening scope. Record the failure, revert or quarantine it, and pick the next smallest corrective step.
- Every accepted communication benchmark must state rank count, token count, pure time, fused time, degradation, correctness, and profile evidence.

## Phases

### Phase 1: Discovery and Baseline Mapping
- [x] Read K1_groupgemm_fp8 README and best C groupgemm skeletons.
- [x] Map existing DCU MegaMoE K1/K2/K3 implementations and tests.
- [x] Map CUDA MegaMoE overlap strategy for K1 dispatch pull and K3 combine.
- [x] Record layout, APIs, build commands, and reusable pieces in findings.md.
- **Status:** complete

### Phase 2: Isolated V2 Skeleton
- [x] Create csrc/kernels/dcu_megamoe_v2 or equivalent isolated source directory.
- [x] Create independent build entry under scripts.
- [x] Create independent Python test entry under tests.
- [x] Create and consolidate V2 overview/progress/findings under `.planning/dcu_megamoe_v2/`.
- **Status:** complete

### Phase 3: Pure GroupGEMM Port
- [x] Port/extract the best C FP8 groupgemm skeletons into the V2 build path.
- [x] Validate independent compile path on remote DCU container.
- [x] Establish pure groupgemm timing for quick tokens: small 32/128, large 1024/4096.
- **Status:** complete

### Phase 4: V2 Weight Layout Transform
- [x] Implement pack5 layout transform helper owned by V2.
- [x] Add tests that verify logical to physical n mapping.
- [x] Ensure layout transform is excluded from benchmark timing.
- **Status:** complete

### Phase 5: K1 Fused Dispatch Pull Prototype
- [x] Implement initial small-token local-pull fused kernel mode and launcher independently.
- [x] Start with correctness-first dispatch-pull plus L1 groupgemm.
- [x] Benchmark quick small tokens and compare against pure K1 groupgemm.
- [x] Capture launch-level profile evidence.
- [x] Extend K1 fused metadata from local deterministic pull map to sym-buffer prebuild/dispatch-pull contract.
- [x] Upgrade small-token K1 communication validation from 2-DCU smoke to 4-rank or 8-rank acceptance.
- [x] Add large-token K1 fused path.
- [x] Profile real 4/8-rank overlap with hipprof/PMC and ISA evidence.
- **Status:** complete

### Phase 6: K2 Integration
- [x] Reuse existing optimized DCU SwiGLU plus quant implementation where possible.
- [x] Keep V2 API/build/test entry independent.
- [x] Validate stage-level correctness.
- [ ] Add K2 timing rows once K1/K2/K3 staged harness is connected.
- **Status:** complete for correctness, timing deferred to staged harness

### Phase 7: K3 C Pack5 Fused Combine Prototype
- [x] Remove rejected K3 ASM wrapper/code-object path and kpack2 layout from V2 source/tests/docs.
- [x] Restore K3 to the same V2 pack5 layout contract used by K1.
- [x] Add pack5 tests that cover both L1 and L2 logical shapes.
- [x] Adapt the C groupgemm harness for K3 L2 shape: N=4096, K=2048, pack5.
- [x] Establish pure K3 C pack5 timing for small quick tokens 32 and 128.
- [x] Establish pure K3 C pack5 timing for large quick tokens 1024 and 4096.
- [x] Implement K3 C row-combine in the groupgemm epilogue / same-kernel copy-stage with no extra combine kernel.
- [x] Validate K3 C row-combine correctness for small tokens 32/128 on real 8-rank communication targets.
- [x] Validate K3 C row-combine correctness for large tokens 1024/4096 on real 8-rank communication targets.
- [ ] Compare K3 C fused against pure K3 C pack5 for small and large tokens; target <=25% degradation.
- [ ] Profile real 4/8-rank overlap and log hipprof/PMC/ISA evidence.
- [ ] Add in-kernel combine reduce / tail reduce; current row-combine stores partial rows and is not final combine-reduce acceptance.
- **Status:** in_progress

### Rejected / Quarantined K3 Work
- [x] K3 ASM/kpack2 prototype was built and measured, but it violates the V2 requirement to use C groupgemm and the unified pack5 layout.
- [x] Treat all K3 ASM/kpack2 timings as invalid for acceptance. Keep them only as failure history in findings/progress.
- [x] Remove the corresponding source/test/doc artifacts from active V2.

## Big Plan With Small Plans

### Milestone A: Correct The Source Tree
- [x] Delete K3 ASM V2 extension source.
- [x] Strip K3 ASM helpers from stages.py.
- [x] Remove K3 kpack2 layout helpers/tests.
- [x] Update docs/progress/findings to mark K3 ASM as rejected.
- [x] Verify local py_compile, git diff --check, and remote pytest.

### Milestone B: Lock Unified Pack5
- [x] Add L1 pack5 test fixtures.
- [x] Add L2 pack5 test fixtures for N=4096, K=2048 shape reduced where needed.
- [x] Add Python-vs-C pack5 byte mapping check.
- [x] Confirm layout transform is outside benchmark timing.

### Milestone C: K1 Communication Acceptance
- [x] Re-run small-token K1 fused communication with 4 ranks or 8 ranks.
- [x] Require the communication pull/staging to happen inside the K1 compute kernel.
- [x] Record 32/128 pure vs fused timing, correctness, and hipprof evidence.
- [x] Continue K1 large-token C pack5 dispatch-pull fusion only after the small-token 4/8-rank acceptance is clean.

### Milestone D: K1 Large-Token C Pack5 Fused
- [x] Add large-token dispatch-pull/staging path on the C pack5 groupgemm kernel.
- [x] Validate tokens 1024/4096 with 8 ranks.
- [x] Target <=20% degradation against pure large-token K1 C pack5.

### Milestone E: K3 Pure C Pack5
- [x] Make pure K3 C pack5 run for N=4096, K=2048 without VMFault.
- [x] Validate small tokens 32/128 correctness max_abs <= 1e-3.
- [x] Validate large tokens 1024/4096 correctness max_abs <= 1e-3.
- [x] Record stable pure K3 C timings for 32/128/1024/4096.
- [x] Decide and record small-token K3 C tiling separately from large-token K3 C tiling if the lowlat and MT256 paths diverge.

### Milestone F: K3 C Pack5 Fused Combine
- [x] Add row-combine pointer output in C groupgemm epilogue.
- [x] First validate identity row pointers for small tokens 32/128.
- [x] First validate identity row pointers for large tokens 1024/4096.
- [x] Then validate real 8-rank combine targets for 32/128/1024/4096 correctness.
- [x] Hit <=25% degradation for small-token K3 row-combine 32/128.
- [x] Add correctness-first same-kernel local-rank tail-reduce prototype.
- [x] Add small-token same-kernel local-rank tail-reduce prototype for 32/128.
- [ ] Hit <=25% degradation for large-token K3 row-combine 1024/4096.
- [ ] Optimize same-kernel tail reduce and extend it from local-rank prototypes to final all-rank combine-reduce acceptance.

### Milestone G: End-To-End V2
- [ ] Connect K1 C fused, K2 optimized wrapper, and K3 C fused.
- [ ] Add stage-level and end-to-end correctness tests.
- [ ] Run quick tokens first, then full sweep.
- [ ] Add uneven tokens per rank and CUDA graph replay after eager path is correct.

## Original Prompt Recheck Checklist
- [x] V2 source remains isolated under V2-specific source/test/script/doc paths, not under hygon_tmp.
- [x] hygon_tmp contains only remote logs, profiles, temporary code objects, dumps, and experiment caches.
- [x] Existing dcu_megamoe, large_opt 3-stage, and big-fused paths are not modified.
- [x] V2 symbols, tests, build entry, and docs remain independent.
- [x] K1 small-token path uses fused dispatch/prebuild pull plus L1 FP8 groupgemm in one compute kernel.
- [x] K1 large-token path uses the C pack5 groupgemm skeleton with dispatch/prebuild pull fused in one compute kernel.
- [x] K2 reuses the optimized DCU SwiGLU+quant implementation through an isolated V2 boundary.
- [ ] K3 small-token path uses C pack5 L2 groupgemm plus combine fused in one compute kernel.
- [ ] K3 large-token path uses C pack5 L2 groupgemm plus combine fused in one compute kernel.
- [x] L1 and L2 weights use the unified V2 pack5 layout; no baseline layout and no K3 kpack2/ASM layout in accepted paths.
- [x] Tests explicitly perform V2 layout transform, and layout transform is excluded from benchmark timing.
- [ ] Stage-level correctness is collected for K1, K2, and K3 before end-to-end correctness.
- [ ] End-to-end correctness compares against baseline with max_abs <= 1e-3.
- [x] Quick optimization loop covers K1 and pure K3 small 32/128 and large 1024/4096; K3 fused combine quick loop remains in progress.
- [ ] Full sweep later covers 32/64/128/256/512/1024/2048/4096/8192.
- [x] Communication acceptance uses 4 ranks or 8 ranks and confirms communication is hidden inside compute kernels for accepted K1 checkpoints.
- [ ] No execution path introduces hipMalloc, hipFree, D2H, or unnecessary new kernels.
- [ ] No unnecessary new environment variables are added.
- [ ] Uneven tokens per rank support is added or explicitly planned after eager path.
- [ ] CUDA graph one-graph multi-size replay support is added or explicitly planned after eager path.
- [ ] Every accepted checkpoint records files changed, tokens, pure time, fused time, degradation, correctness, profile evidence, and next-step judgment.

### Phase 8: End-to-End Stages Fused Test
- [ ] Add end-to-end V2 test against baseline.
- [ ] Report correctness metrics for K1, K2, K3, and end-to-end.
- [ ] Run quick benchmark set.
- **Status:** pending

### Phase 9: Full Sweep and Feature Follow-up
- [ ] Run full tokens 32/64/128/256/512/1024/2048/4096/8192.
- [ ] Add or plan uneven tokens per rank support.
- [ ] Add or plan CUDA graph one-graph multi-size support.
- **Status:** pending

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

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Missing superpowers skill at listed .system path | 1 | Loaded actual installed skill path from plugin cache and treated it as skill-selection context. |
| K3 large raw-buffer GLC flag synchronization hung during 8-rank correctness and the container exited | 1 | Reverted the experiment, restored the system-fence copy-stage path, and recorded that this sync variant must not be retried without a standalone microbench. |
| K3 large copy-worker row-pointer half-wave broadcast regressed timing | 1 | Reverted the experiment after 1024/4096 8-rank correctness passed but timing worsened. |
| K3 large copy-worker row-tile scheduling regressed timing | 1 | Reverted to the linear tile scheduling baseline after 1024/4096 8-rank correctness passed but timing worsened. |

## Round Output Checklist
Each progress checkpoint should report:
- Files changed.
- Tokens tested.
- Pure groupgemm time.
- Fused time.
- Degradation ratio.
- Correctness max_abs, mean_abs, mismatch.
- Profile evidence and next-step judgment.
