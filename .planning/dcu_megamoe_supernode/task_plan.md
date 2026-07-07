# DCU MegaMoE Supernode Plan

## Status Convention

- ✅ completed.
- [ ] required but not complete.
- 🚫 abandoned or explicitly not pursued.
- 🧭 optional backlog / observe until new evidence.
- Do not use `[x]`; checked items are normalized to ✅.

## Goal

Support the DCU MegaMoE V3 staged path on TX32 supernode EP sizes while keeping this branch focused only on supernode enablement.

Target shapes:
- DSV4 Flash model shape: experts=256, topk=6, hidden=4096, intermediate=2048.
- DeepSeek-V4-Pro model shape: experts=384, topk=6, hidden=7168, intermediate=3072.
- EP8 remains supported. Peer-memory mode is independent from EP size: default is HIP IPC, and Fabric/RPC is enabled only with `MEGAMOE_DCU_PEER_MEMORY=rpc`.
- EP16 and EP32 must be accepted for supernode runs, with local experts 16 and 8 respectively.
- Pro EP8/EP16/EP32 must be accepted by source and dispatch gates, with local experts 48, 24, and 12 respectively. Initial runtime validation is Pro EP16 only.

Scope boundaries:
- This branch only implements supernode support. Do not re-open LL_V2 or unrelated kernel experiments.
- TX32 runtime validation is active; keep node-local and cross-node results separated because the two nodes do not share storage.
- Preserve performance-first contracts: no extra hot-path kernels and no runtime weight transforms. EP16/EP32 shape support must not implicitly switch peer-memory mode; `MEGAMOE_DCU_PEER_MEMORY=rpc` is the only Fabric/RPC selector.

## Phase 1: Read Examples And Current Assumptions
Status: ✅ completed

- ✅ Read `hygon_tmp/supernode_code` C++ examples and Galaxy supernode interface PDF.
- ✅ Identify current EP8/256 hard gates in Python/C++ source.
- ✅ Identify rank-count assumptions inside K1/K3 wrappers and source guards.
- ✅ Decide minimal code changes for EP16/EP32 without perturbing legacy non-supernode EP8 behavior.

## Phase 2: Shape And Scratch Generalization
Status: ✅ completed

- ✅ Replace EP8-only validators with DSV4 Flash EP set `{8,16,32}`.
- ✅ Update human-readable build config and error messages.
- ✅ Ensure route scratch and signal-address scratch scale with `num_ranks`.
- ✅ Update source guards to cover EP16/EP32.
- ✅ Static-check all remaining EP8-only constants and direct slot references.

## Phase 3: Supernode Peer Memory Handles
Status: ✅ completed

- ✅ Add a supernode-capable peer mapping path based on the example RPC/Fabric handle flow.
- ✅ Keep legacy non-supernode EP8/local IPC behavior available.
- ✅ Add explicit peer-memory selection: default `ipc`; `MEGAMOE_DCU_PEER_MEMORY=rpc` enables DeepEP-style Fabric/RPC handles.
- 🚫 Abandoned the earlier hybrid same-host-IPC / cross-host-RPC design after re-reading DeepEP supernode normal path. DeepEP normal/MNNVL uses one Fabric-backed shared-memory mode instead of per-peer IPC/RPC mixing.
- ✅ Ensure destroy/cleanup closes the matching handle type.
- ✅ Build-review HSA include/link compatibility against the example code and link `hsa-runtime64`.

## Phase 4: Static Review
Status: ✅ completed

- ✅ Run local Python compile checks and source-guard string checks that do not require DCU hardware.
- ✅ Review hot-path kernels for `kMaxSignalRanks`, 16-slot scratch, or EP8-only assumptions.
- ✅ Update findings/progress with remaining runtime validation checklist.

## Runtime Validation Checklist
Status: [ ] active on TX32

- ✅ Build the HIP wheel on the target DTK image for EP16 node22.
- ✅ EP16 process-group smoke test: buffer allocation, pointer exchange, `set_mega_moe_peer_ptrs`, pre-dispatch, K1-only, K3 no-reduce, and default normal fused smoke.
- ✅ EP16 correctness bring-up fixes: single-node IPC peer mode and K1 compact prebuild default for `num_ranks > 8`.
- 🚫 EP16/EP32 normal K3 ASM tail-reduce1 is no longer the default path. The signal-layout fix remains available for explicit ablation with `K3_USE_ASM_TAIL_REDUCE=1`, but EP16/EP32 normal defaults to tail-reduce0 / external local reduce.
- 🚫 EP16/EP32 normal eager active-tile patch was tried and reverted. It passed correctness but did not improve large-token performance enough to justify extra eager plumbing.
- [ ] Re-run IPC default and `MEGAMOE_DCU_PEER_MEMORY=rpc` Fabric/RPC smoke once hardware is available; EP size should not affect peer-memory selection.
- ✅ 151.1 EP8 8-card RPC smoke on devices `0..7`: `LL graph`, `LL eager`, `normal eager`, `normal graph`, `LL graph uneven`, and `normal eager uneven` pass with `normal-contiguous` baseline after Fabric buffer export size was aligned to 2 MiB.
- ✅ 151.1 true EP16 RPC validation later unblocked; EP16 LL graph, normal eager, and normal graph runs passed. The earlier device15 allocation issue was not reproducible.
- ✅ 151.1 EP8 RPC `normal eager 4096` passed after the node became idle: correct, MegaMoE `5.7636 ms`, baseline `10.0042 ms`.
- ✅ 151.1 EP16 RPC `LL graph` capture512 passed for replay `8,32,64,128,256,512`: correct, graph replay medians `0.3749/0.4069/0.4769/0.5977/0.9869/1.9228 ms`.
- ✅ 151.1 EP16 RPC `normal eager` matrix is complete through `5120`: `512/1024/1025/2048/2050/4096/4097/5120` are correct and faster than baseline.
- ✅ 151.1 EP16 RPC `normal eager 8192` MegaMoE-only timing is collected (`12.6043 ms`); full baseline correctness passes, but baseline benchmark times out after 420s, so stable EP16 baseline timing is not available for this bucket.
- ✅ Baseline comparison on 151.1 shows EP16 baseline is also mildly slower than EP8 at large tokens (`+3.4%` at 4096, `+6.9%` at 5120), suggesting the EP16 large-token gap is partly shared by the baseline communication path.
- 🚫 Do not keep chasing the yuguo old-DeepGEMM baseline as the primary validator in the current TX32 torch runtime. It is ABI/storage-sensitive; use the current environment DeepGEMM with contiguous baseline weight layout for dcu_mega_v3/supernode A/B checks.
- [ ] Build the HIP wheel on node69 once it is free, because TX32 nodes do not share compiled artifacts.
- [ ] EP32 process-group smoke test: cross-node Fabric/RPC peer memory, buffer allocation, pointer exchange, `set_mega_moe_peer_ptrs`, and destroy cleanup.
- ✅ EP16 LL graph capture512 uniform correctness/performance on node22 for replay `8,32,64,128,256,512`, faster than `ll-masked` baseline in the collected run.
- ✅ EP16 LL graph uneven correctness/performance on node22, faster than `ll-masked` baseline in the collected run.
- ✅ EP16 normal eager correctness on node22 for checked boundary buckets through 8192 after the K1 compact-capacity fix.
- ✅ EP16 normal eager performance smoke on node22 for 512 and 2048, faster than normal-contiguous baseline.
- ✅ Finish EP16 normal eager performance matrix for `1024,1025,2050,4096,4097,5120`; all checked buckets are faster than the normal-contiguous baseline.
- ✅ Sample EP16 normal eager `8192` MegaMoE-only performance with baseline timing explicitly skipped because the DeepEP normal baseline does not return cleanly at this bucket.
- [ ] Investigate DeepEP normal baseline `8192` EP16 rank hang only if a full baseline comparison at this bucket becomes required.
- ✅ Root-cause EP16 normal medium/large-token regression versus EP8 references (`1024+`, especially `4096/8192`): stage timing shows the pure K3 GEMM path is healthy, while K3 peer combine scatter/write dominates the extra EP16 cost.
- 🚫 Optimize normal K3 peer combine writes for EP16/EP32 as a supernode-specific required task. Superseded by the 151.1 EP16 RPC data: EP8 `4096` recovered to `5.7636 ms`, EP16 normal eager and normal graph are correct and faster than baseline, so no extra hot-path K3 peer-combine rewrite is currently justified.
- ✅ 151.1 EP16 RPC normal graph validation passed for uniform cap512/cap4096 and uneven cap512/cap1024/cap2048/cap4096. Keep the exact replay numbers in `progress.md` and `findings.md`.
- [ ] EP32 LL eager/graph correctness and performance.
- [ ] EP32 normal eager/graph correctness and performance, especially cross-node Fabric/RPC and signal handling.

## DeepSeek-V4-Pro Shape Support Plan
Status: [] Pro shape support is implemented; Pro EP16/EP8 normal eager validation is collected through 8192, Pro LL split source cleanup is complete locally, and remaining work is final 151.1 rebuild/runtime sanity plus large normal-graph cap8192 instability.

Target:
- Shape: experts=384, topk=6, hidden=7168, intermediate=3072.
- Supported EP sizes: EP8/EP16/EP32.
- Runtime priority: Pro EP16 correctness first was completed; Pro EP8 now has normal eager plus LL cap512 coverage; Pro EP32 remains source-supported and runtime-deferred unless explicitly requested.
- Flash guardrail: existing DeepSeek-V4-Flash correctness and performance must not materially regress. If shared dynamic normal ASM hurts Flash, split Pro-only K1/K3 normal ASM code objects and keep Flash on the original constant-address path.

Current snapshot (2026-07-03):
- ✅ Source and dispatch gates cover Pro EP8/EP16/EP32; runtime validation now covers Pro EP16 plus the requested Pro EP8 normal/LL slices.
- ✅ Pro EP16 normal correctness/performance passed through 8192 against `normal-contiguous`; Pro EP16 LL also passed the same baseline oracle.
- ✅ Flash EP8 normal 4096 guardrail remains in the same performance band after the shared K1 ASM fix, so no Pro-only kernel split is currently justified.
- ✅ Temporary Python debug hooks, backend rerun flags, Pro-only weight-layout override, and extra eager assertion formatting have been removed from the test harness.
- 🚫 K1/K3 publish fences, store-side `glc`, K2/reduce `glc/slc`, K3 sleep/sync/tail-reduce/store probes, and the Pro-only layout override were diagnostic paths with negative or no production value.
- ✅ README documents both Flash and Pro staged shapes and includes a Pro EP16 validation example.
- ✅ Pro EP16 normal eager 8192 is stable for fused correctness/execution. Pro EP16 graph cap8192 correctness is fixed, but replay8192 graph bench/cleanup remains unstable.
- ✅ Pro EP8 512 normal eager is fixed by forcing compact prebuild for `local_experts > 32`; clean IPC sanity and RPC baseline-bench sanity passed.
- ✅ Fix the remote packaging/install flow so source-tree imports do not require manually copying fresh built `.so` files.
- ✅ Pro EP8 normal eager RPC runtime validation is complete for `512,1024,1025,2048,2050,4096,4097,5120,8192`; all buckets passed correctness against `normal-contiguous` with baseline timing. Pro EP8 graph/LL/uneven data remains separate if requested. Pro EP32 runtime validation remains deferred.
- [ ] 2026-07-02 requested 151.1 validation batch is partially complete: Flash EP16 LL graph fair retest passed, Pro EP8 normal eager buckets `512,1024,1025,2048,2050,4096,4097,5120,8192` passed, and Pro LL cap512 checks were collected on the later split path. Remaining: final current-source rebuild/sanity, Flash EP16 normal graph `4096/5120/8192` retest, large-cap graph instability retest, and any still-requested full Pro EP8/EP16 graph/uneven matrix.
- ✅ Rechecked Flash EP16 LL graph with a fair cap512 `--baseline-kind ll-masked` command after 151.1 recovery. It matched historical cap512 within about `-1%..+1%`; no code regression was reproduced.
- 🚫 Current-vs-previous-commit A/B and LL-risk bisect are not pursued now because the fair Flash EP16 LL graph retest passed. Keep this only as a contingency if a future fair cap512 run regresses.
- ✅ Source-side unified LL skew guard is added and smoke-validated on 151.1: Flash LL and Pro `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` LL reserve up to 256 rows/expert, bounded by `num_ranks * num_max_tokens`, while Pro default split masked-K1 keeps its separate 128-row EP8 launch-shape guard. Runtime data is in `hygon_tmp/supernode_debug/151_1_rpc/ll_graph_flash_guard_20260707_103302`.

Completed:
- ✅ Re-read PR #316 shape requirement and existing V3/supernode planning memory before implementation.
- ✅ Added shared Python/C++ V3 staged shape registry and gates for Flash plus Pro.
- ✅ Generalized Pro route scratch sizing, signal tail counters, and shape checks.
- ✅ Added Pro local experts 48/24/12 support in LL dispatch and raised per-expert counter capacity.
- ✅ Generalized K1 normal compact header offsets and forced compact capacity for `num_ranks > 8`.
- ✅ Reserved separate K3 BF16 output workspace for Pro when `hidden > 2 * intermediate`.
- ✅ Fixed normal ASM Flash-only `x_sf`, K3 scatter-stride, and `GLOBAL_OFFSET_A` address constants.
- ✅ Remote source pytest and dynoffsetA4 rebuild passed on 151.1.
- ✅ Pro EP16 LL correctness passed after the workspace fix.
- ✅ Fixed K3 normal staged writeback loop to use the fixed 4096-row half-tile bound instead of hidden size; this removed a Pro-only out-of-tile combine-row walk but did not fully fix correctness.
- ✅ Added targeted combine-slot and route-metadata diagnostics. Latest evidence rules out local reduce and shows row metadata exists for a previously missing-slot token.
- ✅ Pro EP16 LL also passes against the `normal-contiguous` baseline, proving the baseline oracle is usable for the Pro shape.
- ✅ Same-input normal-vs-LL diagnostics isolated the first confirmed bad boundary to normal K3 ASM: the normal combine buffer does not match a Python single-column reconstruction from its own visible `act_fp8`, `act_scale`, and L2 packed weight inputs.
- ✅ Corrected the previous `GLOBAL_OFFSET_A` hypothesis: pack5 `offset0I` is tile-local `ni16`, so its stride is fixed `4 * 256 = 1024`; only `ko` and row/tile base addressing scale with hidden.
- ✅ Rebuilt and retested Pro EP16 normal after restoring fixed pack5 `ni16` stride; correctness still failed, proving the correction is source-correct but not sufficient.
- ✅ Plain-normal fixed-ni16 diagnostics showed K3 can match a Python reconstruction from its visible inputs for the selected bad point, reclassifying the current first bad boundary to K1 normal output production.
- ✅ K1 debug showed valid Pro EP16 normal route rows can have `l1_out_absmax=0.0` before K2, while other slots for the same token are nonzero.
- ✅ Added Python K1 reconstruction from the normal path's own staged input, `l1_weight`, and `l1_scale`. On route `(source_rank=5, token=189, col=3478)`, normal K1 ASM produced identical row samples for two different local experts while Python K1 references were distinct, proving the first bad boundary is inside normal K1 ASM production rather than baseline, compact tile metadata, K2, K3, or reduce.
- ✅ Rejected the K1 metadata cache-coherency hypothesis. The `buffer_wbinvl1/glc` experiment still matched local expert0 until the freshly built K1 runtime extension `.so` was copied into the source-tree import path.
- ✅ Root-caused the expert0 K1 discriminator to stale runtime build artifacts: tests were importing the old source-tree `k1_fused_ext*.so`, so the new `reserved_c4` compact metadata fields were never passed to ASM.
- ✅ After copying fresh built `.so` artifacts from `build/lib.../megamoe/` into the runtime source tree, Pro EP16 normal default layout passed the targeted 512-token correctness run against `normal-contiguous`, and K1 all-expert discriminator matched routed experts instead of local expert0.
- ✅ Removed the temporary K1 metadata `buffer_wbinvl1/glc` experiment, rebuilt, copied fresh `.co` and `.so` runtime artifacts, and confirmed Pro EP16 normal still passes. The correctness fix does not require the experimental metadata-load path.
- ✅ Pro EP16 no-debug 512-token smoke passed for both `normal` and `ll` backends against `normal-contiguous`.
- ✅ Flash EP8 normal 4096 RPC correctness/performance guardrail passed after the Pro changes and no-experiment rebuild: MegaMoE `5.784 ms` versus previous same-node reference `5.764 ms`, within noise.
- ✅ Collected initial Pro EP16 normal 512-token performance: MegaMoE `3.206 ms`, baseline `4.258 ms`, speedup `1.33x`, correctness still clean.
- ✅ Collected larger Pro EP16 normal 1024-token performance: MegaMoE `5.068 ms`, baseline `7.144 ms`, speedup `1.41x`, correctness still clean.
- ✅ Isolated the 151.1 Pro EP16 normal graph cap5376/replay512 failure to K3 normal ASM tail-reduce graph runtime, not eager/K1/K2/plain-K3 graph. Tail-reduce0 passes the minimized replay512 check.
- ✅ Rebuilt on 151.1, copied fresh `build/lib.linux-x86_64-cpython-310` MegaMoE artifacts back into the source-tree import path, and confirmed fresh K3 tail-reduce `.co` plus `k3_fused_ext` runtime artifacts.
- ✅ Verified the K3 tail-reduce graph runtime fix on 151.1: minimized Pro EP16 graph replay512 passed, cap5120 full buckets passed, and cap8192 correctness passed. cap8192 replay8192 graph bench/cleanup remains a separate stability issue.

Resolved Diagnostic History:
- ✅ Pro EP16 LL 2048 passed against the same `normal-contiguous` baseline, so the 2048 oracle is credible and the remaining failure is normal-only.
- 🚫 Tried a bounded K3 direct-combine publish patch (`s_waitcnt vmcnt(0)` plus `buffer_wbinvl1_vol` before `s_endpgm`); Pro EP16 normal 2048 still failed, so the patch was removed.
- ✅ Targeted route diagnostics and explicit sync probes show plain synchronization is insufficient, while route-specific CPU readback can make that route correct and move the max-diff elsewhere.
- 🚫 Built, ISA-verified, and tested the bounded K1 main-output publish patch (`s_waitcnt vmcnt(0)` plus `buffer_wbinvl1_vol` before K1 main `Kernel End`); Pro EP16 normal 2048 still failed, so the patch was removed.
- ✅ Selective route-stage diagnostics for `(source_rank=7, token=1030, col=6286)` show `combine`-only readback keeps slot0 wrong-positive, while `k1,combine` readback makes the original route match the baseline within BF16 rounding.
- 🚫 Built and ISA-verified a K1 BF16 output store-side `glc` experiment (`buffer_store_short ... glc` for all K1 `store D` sites); Pro EP16 normal 2048 still failed.
- 🚫 Tested K1 store-side `glc` plus K2 `glc/slc` input loads together; Pro EP16 normal 2048 still failed.
- ✅ Removed the unproven K3/K1 publish patches, K1 store-side `glc` guard, and the env-gated K2 `glc/slc` input-load diagnostic from local source before the next diagnostic loop, because none provided correctness value and they affect shared Flash/normal hot-path risk.
- ✅ Clean hot-path rebuild still fails Pro EP16 normal 2048; route `(10,559,3146)` showed combine readback can materially change the dominant bad slot, so the problem is not limited to K1 output consumption.
- 🚫 Env-gated `reduce_local_combine` `global_load_dwordx4 ... glc slc` was ISA-verified and tested negative; final reduce read-side coherency alone is not sufficient.
- ✅ K2/K3 visible-input and stage-matrix diagnostics for route `(5,610,1383)` show K3 math/addressing is correct under visible inputs; `k3,combine` or `k2,combine` readback before/around the post-K3 barrier makes the dominant slot correct, while `combine` alone only partially helps.
- 🚫 K3 debug side effect is not pure delay/timing: 10ms sleep after K3, sleep+sync, tail-reduce control, and K3 direct-combine store-side `glc` all still failed Pro EP16 normal 2048.
- ✅ Restored clean direct K3 ASM after the negative store-side `glc` probe, rebuilt/copied fresh `.co`/`.so`, and continued diagnostics from a clean hot path.
- ✅ Added and ran same-input normal-vs-LL/baseline stage comparison for Pro EP16 2048. LL matches `normal-contiguous`; normal diverges before K2/K3 because K1 staged FP8 source-x bytes can differ from the source rank input while x_scale is correct.
- ✅ Fix normal K1 input staging / remote source-x read path for Pro hidden=7168, focusing on hidden-stride/chunk loop bounds and stale low-column chunks.
  - ✅ Local root-cause patch prepared for the rank-local K1 staging path: a wave can straddle two Pro rows because 7168 bytes is 224 x 32-byte chunks, so the source rank must stay lane-specific instead of using one `v_readfirstlane(v254)` rank for the whole wave.
  - ✅ Remote rebuilt/copied fresh K1 artifacts and reran the same-input source-x diagnostic; targeted route source-x mismatches dropped to zero for the sampled K1 rows.
  - ✅ The remaining K1 output mismatch is isolated to K1 N tiles that participate in dispatch-pull staging (`wg0 < 8`, columns `< 2048`). Columns in later tiles match the Python K1 reference/LL within BF16 tolerance.
  - ✅ Prepared a local K1 ASM fix that restores `s62` to the staged_x plane offset `0` before rebuilding B addresses. The rank-local staging loop reused `s62` for source-rank pointer slots, so participating low-column CTAs could read staged_x from a shifted B base.
  - ✅ Rebuilt/copied fresh K1 `.co`/`.so` artifacts on 151.1 and reran Pro EP16 2048 after the `s62` restore. Full correctness passed with `max_abs=0.000976562`, so the K1 low-tile fault is fixed for the active bucket.
- ✅ Enforce ordered same-input stage comparison before further kernel changes.
  - ✅ Added debug-only K1 snapshot capture and made failure backend-compare default to K1-only. The compare now stops after K1, captures full K1 rows for normal and LL, and prints per-slot `MEGAMOE_DCU_DEBUG_K1_COMPARE` with physical-row equality, logical expert metadata, max_abs, argmax column, and top delta columns.
  - ✅ Ran the remote Pro EP16 2048 K1-only same-input compare and confirmed K1 itself is the first failing boundary.
  - ✅ Compared normal K1 output against LL K1 output on the same generated input and logical route. Physical rows differ, so the comparison aligned by route slot; route weights matched.
  - ✅ K1 debug compared source-x bytes, x scale, routed expert metadata, L1 packed weight offsets/scales, and Python full-row refs. Source-x is correct; low N tiles are wrong in normal K1 output.
  - ✅ After the `s62` restoration rebuild, reran the active Pro EP16 2048 bucket. The run no longer failed, so no K2/K3 divergence remains for this bucket.
  - [ ] If a later Pro bucket fails, resume the ordered compare at K2 output (`act_fp8`, `act_scale`, and dequantized values) on the same route.
  - [ ] If K2 passes for a later failure, compare K3/combine output on the same route, then final reduce if needed.
- ✅ After the 2048 correctness fix, reran Pro EP16 normal 2048 no-debug correctness/performance. Result: correct with `max_abs=0.000976562`; MegaMoE `8.032 ms`, normal-contiguous baseline `13.204 ms`, speedup `1.64x`.
- ✅ Flash EP8 4096 guardrail after the shared K1 ASM `s62` fix passed: correct with `max_abs=0.000671387`; MegaMoE `5.803 ms` versus the prior `5.784 ms` reference, a noise-level delta. No Pro-only kernel split is currently justified.
- ✅ Broader Pro EP16 4096 no-debug correctness/performance guardrail passed: `max_abs=0.000991821`; MegaMoE `15.791 ms`, normal-contiguous baseline `25.327 ms`, speedup `1.60x`.
- ✅ Cleaned up temporary Python debug instrumentation after the K1 fix: removed `MEGAMOE_DCU_DEBUG_*` route/stage hooks, K1 snapshot capture, backend failure rerun CLI flags, and combine-slot failure dumps. Added source-test assertions so these debug hooks do not re-enter production/test harness sources.
- ✅ Removed the test-only `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT` override and restored the harness to the general weight-layout policy: explicit unified env uses unified; otherwise normal backend uses normal layout and LL uses unified.
- ✅ Removed the extra eager correctness failure argmax/value formatting and restored the original one-line `max_abs` assertion.
- ✅ Inspected the runtime K1 normal `.co` ISA side-effect files and confirmed the current dynamic compact metadata and routed expert stride code are present.
- ✅ Ran the K1 all-local-expert Python discriminator for the same Pro EP16 route; observed K1 rows match local expert0, not the routed expert.
- ✅ Continued validation after the compact-prebuilt ASM patch that reads routed expert directly from `route_scratch_i32[tile_experts_offset + compact_tile]`. The apparent failure was caused by stale runtime `.so` launch code, not the patched `.co`.
- ✅ Ran the bounded K1 ASM discriminator that forces `sgprScaleFlag=1`; K1 rows switched to local expert1, proving the A-weight/scale expert stride/SRD path can apply a nonzero expert id.
- ✅ Added Python-visible route-scratch header debug for the same forced run; `compact_tile_expert_header` matched each routed tile id/m-index, proving the HIP-built side-channel metadata is correct in memory after K1.
- ✅ Root-caused why the normal non-forced K1 ASM metadata path behaved like expert0 despite correct side-channel data and a working expert-stride path: the runtime launcher was stale and left the new packed metadata argument unset.

Pending / Deferred:
- 🚫 Keep the fresh `.so` copy step in every remote build/test loop. This is no longer needed after `build_dcu_megamoe.sh` started deleting stale runtime artifacts, building inplace, verifying fresh `.so/.co` files, and checking imports resolve to the source tree.
- 🚫 Flash guardrails collected so far do not show material performance regression, so Pro-only normal ASM code objects are not pursued now. Reopen only if a future fair Flash run regresses.
- [ ] Keep Pro EP32 runtime validation deferred until requested; Pro EP8 now has normal eager 512..8192 plus LL cap512 coverage, but not a full graph/uneven matrix.

## Pod2 40-Card Environment Bring-Up
Status: [ ] blocked on compute-node interactive auth or key setup.

Target:
- Jump host: `simsadmin@10.2.208.215:51730`.
- Worker nodes: `c0..c9` at `172.20.2.131..172.20.2.140`, user `p_user`, expected 4 DCUs per node for 40 cards total.
- Docker: `lj_sgl_0512`.
- Storage model: shared storage across Pod2 compute nodes.
- Important mounts: `/data_add/lizhg/lj`, `/data_add/DeepSeek-V4-Pro-FP8-Channel -> /module`, `/public/lijing`.

Checklist:
- ✅ Record the Pod2 profile and command templates without persisting plaintext passwords.
- ✅ Verify jump-host terminal access: `ssh -F NUL -p 51730 simsadmin@10.2.208.215` works with key auth.
- ✅ Established compute-node key auth for terminal automation via ProxyJump to `p_user@c0..c9`.
- ✅ Started existing `lj_sgl_0512` containers on `c0..c9` and verified worker login, `docker exec`, and shared mounts.
- ✅ Checked DCU visibility across `c0..c9` inside `lj_sgl_0512`; current result is no `/dev/kfd` or `/dev/dri/renderD*`, `hy-smi` reports no `hycu` driver / `WAIT GLOBAL TOPO`, and `c2` reports `LINK FAILED`.
- [ ] Resolve Pod2 DCU driver/device exposure before any 40-card GPU run.

## Risks

- Risk: Supernode fabric memory APIs may differ across DTK versions; code should prefer current HIP wrappers when available or use HSA extensions from the examples.
- Risk: EP16/EP32 reduce local expert count and can change K1/K3 grid shape. Validators must not be the only change if kernels assume `local_experts == 32`.
- Risk: Normal K3 ASM tail-reduce signal layout was originally EP8-only. The current fix expands signal/wait slots to 32 ranks for explicit ablation, but EP16/EP32 normal defaults to tail-reduce0 and EP32 tail-reduce1 still requires true 32-card validation if re-enabled.
- Risk: The requested yuguo old DeepGEMM baseline needs an ABI shim in the active torch runtime and has layout/padding sensitivity; keep that baseline isolated from MegaMoE correctness diagnosis.
- Risk: The Pro normal ASM fixes make formerly Flash-constant addressing dynamic. This should preserve Flash behavior, but Flash latency must be checked; if it regresses materially, split Pro kernels instead of sharing the dynamic hot path.

## 2026-07-03 Pro EP8 LL 512 Update

Status: ✅ completed/superseded by Pro LL split finalization after fused-C A/B showed split masked-K1 is faster.

- ✅ Pro EP8 LL 512 fused-only isolation passed on 151.1 with `MEGAMOE_DCU_PEER_MEMORY=rpc`: fused eager median `16.7910 ms`, baseline timing skipped.
- ✅ Pro EP8 LL 512 correctness against `ll-masked` passed after increasing DeepEP LL heap to `ROCSHMEM_HEAP_SIZE=12884901888` and `DUSHMEM_HEAP_SIZE=12884901888`: `max_abs=0.000976562`, `mean_abs=4.86086e-05`.
- ✅ Pro EP8 LL 512 eager timing passed: fused `16.6829 ms`, `ll-masked` baseline `5.6826 ms`, speedup `0.3406x`.
- ✅ Pro EP8 LL graph cap512 replay512 passed: fused graph `16.6293 ms`, `ll-masked` baseline graph `5.7480 ms`.
- ✅ Pro EP16 LL graph cap512 direct comparison is now collected. Replay `8/32/128/256/512` fused graph medians are `3.2558/3.3209/5.1238/8.6320/15.2628 ms`; `ll-masked` baseline graph medians are `2.4204/2.4802/2.6445/3.3827/4.7778 ms`.
- ✅ Pro LL performance triage: split-tail off is worse, `ll_block_m=48/64` only partially improves, and `hipprof` shows K1 LL dominates the fused time.
- ✅ Pro LL action resolved: keep Pro LL on the split stage-only K1 + masked ASM path; do not route small-token auto to normal or continue the fused-C K1 effort in this branch.
  - ✅ Current investigation branch: Flash/Pro EP16 fused-only hipprof shows Pro K1 is about `10.39x` slower while theoretical K1 work is only `2.625x`. Mixed-shape ablations further isolate the main trigger to K1 `K=7168`: `experts=384,hidden=4096,intermediate=2048` stays Flash-like at `1.877 ms`, `hidden=7168,intermediate=2048` jumps to `10.655 ms`, and `hidden=4096,intermediate=3072` is only `2.540 ms`.
  - ✅ Hybrid normal-ASM-K1-as-LL-K1 shortcut was rejected by source contract: LL K2/K3 require per-expert contiguous rows plus `actual_m` counts, while normal ASM K1 emits compact/nondeterministic row order and row-wise metadata.
  - ✅ Resource metadata check does not show catastrophic register spilling for Pro block32 K1 (`vgpr=132`, `sgpr=106`, no VGPR spill) versus Flash block32 (`vgpr=124`, `sgpr=100`), so the next ablation targets tile granularity rather than spill-only tuning.
  - ✅ Temporary ablation result: Pro-only K1 LL `blockN=128` plus Pro `ll_block_m=48` is correct and materially faster on 151.1. Pro EP16 LL graph replay512 improved from `15.2628 ms` to `5.0180 ms`, close to `ll-masked` baseline `4.7777 ms`.
  - ✅ Guardrails for the Pro LL K1 tile change passed: Flash EP16 LL graph cap512 remains in its historical band (`1.9169 ms` replay512), and Pro EP8 LL 512 improves from the prior `16.6293 ms` graph replay512 to `5.3764 ms`, now faster than `ll-masked` baseline `5.7417 ms`.
  - ✅ Production decision: keep the Pro-specific LL K1 tile selector for `K=7168` (`blockN=128`) and Pro `ll_block_m=48` as the current fix. This is narrower than a new kernel split and leaves Flash `K=4096` on the previous blockM32/blockN256 path.
  - ✅ Extracted and measured Pro K1-only/pure groupgemm paths against DeepGEMM masked over multiple buckets; results informed the later split-path decision.
    - ✅ Define the K1-only contract: per-local-expert contiguous rows, `actual_m` counts, unified pack5 weights/scales, staged input/scales, and no K2/K3/combine work.
    - ✅ Build a minimal benchmark path that can time K1 alone without DeepEP baseline noise, using the same generated route distribution as fused LL.
      - ✅ Added `--k1-only-bench` and `--k1-only-ll-block-m` to the DCU MegaMoE test harness; remote source pytest passed on 151.1.
      - ✅ Added a K1-only benchmark epoch guard (`torch.cuda.synchronize()` plus host rank barrier after each K1 call) to avoid unsafe back-to-back K1 start-barrier reuse during warmup/repeat loops.
      - ✅ Added `no-start-barrier` and `pure-gemm` ablation modes. Full K1-only with the in-kernel start barrier still closes SSH before writing JSON, so it remains an unsafe diagnostic mode.
      - ✅ Collected Pro EP16 K1-only ablation data for replay/tokens `8,32,128,256,512`: `no-start-barrier` medians `1.4832/1.6491/1.7492/2.9272/4.5657 ms`; `pure-gemm` medians `1.0704/1.1419/1.1690/2.1947/3.4260 ms`.
      - 🚫 Reject `ll_block_m=64` for Pro LL K1: pure-gemm `256` was about `10.13 ms` and pure-gemm `512` interrupted SSH before result JSON.
    - ✅ Tested Pro EP16 K1-only tokens `8,32,128,256,512`, Pro EP8 LL cap512, and Flash EP16 LL graph cap512 guardrail.
    - ✅ Ran and cleaned the candidate knob/diagnostic surface: `blockN`, `blockM`, barrier/stage-copy ablations, pure-gemm comparisons, and rejected CU/8-wave attempts.
      - ✅ First-order ablation removed K1 internals: `no-start-barrier` removes the kernel start rank barrier, and `pure-gemm` skips dispatch/route scan/stage copy after one initialization K1.
      - ✅ Prepared a pure-gemm-only Pro `blockN` diagnostic switch (`--k1-only-ll-block-n {0,64,128,256}`) so recovery runs can test K1 core tile width without changing the production fused default.
      - ✅ Direct same-input Pro EP16 pure K1 vs DeepGEMM masked comparison collected for tokens `8,32,128,256,512`: pure/DeepGEMM ratios are `1.279/1.298/1.269/1.632/1.679`, so the pure groupgemm core is not yet performance-acceptable.
      - ✅ Stable bm32/bm48 plus blockN `64/128/256` comparisons against same-input DeepGEMM masked were collected for Pro EP16 `256/512`. Best points remain blockN128: token256 bm32/bn128 ratio `1.458`; token512 bm48/bn128 ratio `1.668`. blockN64 is worse, blockN256 is catastrophic, and bm64 remains rejected.
      - 🚫 Reject the MT256 diagnostic branch for Pro K1 pure optimization. The relevant DeepGEMM masked reference is the small-token `MT256x64x128 WGM8 GROUPGEMM_MASKED` path, while `MT256x256x128` is the normal/contiguous large-tile reference; the MT256 diagnostic path was removed after it proved unstable and directionally mismatched.
      - ✅ Cloned DeepGEMM `develop`, checked out package commit `6a53e9c`, and confirmed Pro K1 masked baseline dispatch uses `mode=1002` / `MT256x64x128 WGM8 GROUPGEMM_MASKED` rather than the contiguous `MT256x256x128` path.
      - ✅ Cleaned the local diagnostic control surface: removed `--k1-only-ll-cus`, the unvalidated 128-CU/8-wave pure K1 instantiation, and the associated static assertions. Kept only the stable pure-gemm blockN harness needed for Pro K1 core optimization.
      - ✅ Compared the old scratch `C fp8 groupgemm` LL kernel against the copied masked ASM in the same harness. Flash `E32,N4096,K4096` stays within noise through tokens 256, but Pro `E24,N6144,K7168` reproduces a large C-LL gap: old `blockN=256` is about `4.6x-9.6x` slower than masked ASM, while `blockN=128` improves but is still `1.3x-2.3x` slower.
      - 🚫 Reject simple Pro direct-load launch widening: scratch `CU128`, `8 waves`, and `BM64/BN256` variants are slower than the current `4w/CU64/BM48/BN128` skeleton at tokens `128/256/512`.
      - ✅ Profiled/ISA-inspected the Pro direct C-LL core against masked ASM. The direct core is a fully unrolled global-load + `ds_bpermute` path; masked ASM is an LDS-backed `256x64x128` WGM8 structure. A no-sched-barrier variant was faster but failed correctness, so it is rejected.
      - ✅ Scratch-proved a Pro LDS-backed pack5 C skeleton against the normal/MT256 reference: after fixing Pro `N/K`, packed expert/K strides, and the Flash-only `stage_iter ^ 16` ordering assumption, `row256` improves tokens512 from direct `2.48 ms` to about `1.49 ms`.
      - 🚫 Reject the formal `row256` LDS port for MegaMoE LL: after rebuild on 151.1 it mismatched both DeepGEMM `ll-masked` and the same-input direct128 LL path (`max_abs` about `0.066-0.074`). It is self-consistent with the normal/MT256 orientation, not the LL masked/direct orientation.
      - ✅ New user-approved direction: Pro LL K1 may use an independent weight layout and reorder function. Do not require layout unification with Flash or K3; instead shape-gate Pro-size layout packing and kernel dispatch so Flash/existing LL flows remain unchanged.
      - ✅ Preserve the existing Flash-friendly LL fused kernel that also supports Pro through `blockN=128` and `blockM=48`. Keep it as the Pro LL unified-layout compatibility/fallback path; the new Pro masked-friendly path must be additive and shape/layout-gated.
      - ✅ Implemented the additive Pro-only masked-friendly K1 path for EP16: `ll_pro_masked` L1 layout uses K1 LL stage-only dispatch/packing, DeepGEMM masked K1 groupgemm, and existing unified L2/K2/K3. EP16 512 eager and cap512 graph replay buckets passed correctness and beat the `ll-masked` baseline.
      - ✅ Pro EP8 LL masked-K1 cap512 validation is complete for replay `8,32,128,256,512`; eager 512 and graph buckets passed correctness and beat `ll-masked`.
      - ✅ Cleaned obsolete K1-only/blockN512 layout diagnostic control (`ll_asm_compatible_layout`) after the formal LDS path was rejected.
      - ✅ Flash EP16 LL graph cap512 guardrail passed after the cleanup/rebuild; replay512 remains in the historical fair-run band.
      - ✅ Promoted the `stage-only K1 + bundled DeepGEMM masked ASM K1` path from interim fallback to the final Pro LL production path for this branch.
      - ✅ Stopped comparing against the old Pro unified fused K1 for optimization; keep unified only as the compatibility fallback/Flash guardrail surface.
      - ✅ Built and evaluated Pro-size C pure K1 groupgemm variants against same-input DeepGEMM masked; retained evidence but rejected them for production because they missed the performance target.
        - ✅ Fixed the direct masked-layout prototype's major correctness bug: DeepGEMM masked layout still needs the pack5-style physical N16 mapping inside each N16 group. After adding that mapping, same-input K1 diff drops to BF16-level (`max_abs=0.00390625`, mean about `4e-09`).
        - 🚫 Reject direct-load masked-layout prototypes as the final pure K1 path. Both tested forms are far from the 5% target at Pro EP16 token128: `BM64/BN256/8wave` was about `6.68 ms` vs DeepGEMM `0.91 ms`, and `BM48/BN128/4wave` was about `5.32 ms` vs DeepGEMM `0.91 ms`.
        - ✅ Connected the current C groupgemm as an optional MegaMoE e2e backend for the Pro masked-K1 path (`MEGAMOE_DCU_PRO_LL_MASKED_K1_C_GROUPGEMM=1`) so it can replace the DeepGEMM masked K1 call during optimization without changing the default ASM fallback.
        - ✅ Ported the first masked-orientation LDS row64/N256 C candidate into the formal optional backend. Pro EP16 token128 correctness passes, and it improves the formal C path from `5.86 ms` fused to `2.29 ms`, but it is still slower than the default DeepGEMM masked split path (`1.38 ms`) and therefore remains an optimization scaffold.
        - 🚫 Stop optimizing the formal C backend in this branch; `xlds_wgm8` became bit-exact but stayed about `1.12x-1.24x` slower than DeepGEMM masked, and fused-C e2e stayed slower than split.
          - ✅ Generated save-temps/device `.s` and compared the formal/tune C kernels against the DeepGEMM masked WGM8 ASM before ending the C effort.
          - ✅ Generated current formal M32 LDS C backend device `.s` with `--save-temps=obj` on 151.1 and compared it against the DeepGEMM masked `256x64x128 WGM8` reference. Current C has the same `128` `v_mmac` count but much heavier static control/epilogue shape: `64 KiB` LDS, `211` VGPR, `s_barrier=61`, `s_waitcnt=379`, and `buffer_store=256`; DeepGEMM masked has `32 KiB` LDS, `191` VGPR, `s_barrier=12-16`, `s_waitcnt=14-21`, and `buffer_store=64` in object/source counts.
          - 🚫 Rejected the naive 32KiB single-LDS-stage C port. It matched ASM LDS footprint, but K1-only correctness failed badly (`max_abs~0.99`) because loader waves overwrite the same LDS buffer for the next K stage while compute waves are still consuming the current stage.
          - 🚫 No further C candidate is planned in this branch after the final split decision; keep the double-buffer/ownership notes as evidence for future work only.
          - ✅ Used these comparison targets during save-temps/ISA analysis: launch shape, LDS footprint, VGPR pressure, `v_mmac` grouping/order, B-side LDS offsets, waits/barriers, and scale/store epilogue.
          - 🚫 Do not continue this C-kernel rewrite in this branch; future work should start from the saved evidence if needed.
          - ✅ Extract the Pro masked LDS pure K1 kernel into a tune-only extension so `--save-temps` and same-input K1-vs-DeepGEMM masked loops no longer require rebuilding the whole fused K1 extension. Use this shorter loop to optimize C against the DeepGEMM masked `.s` until performance is ASM-level and numeric diff is minimized.
          - ✅ Matched the masked ASM scale-order in the C store epilogue; Pro EP16 token128 same-input diff improved from BF16-level to bit-exact (`max_abs=0`, `mean_abs=0`).
          - 🚫 Tested and rejected the 512-thread WGM8 weight-LDS candidate: correct but slower (`2.3890 ms` vs DeepGEMM `0.9759 ms`) because N-split compute waves duplicate global `x` loads.
          - ✅ Implemented the input-LDS/direct-masked-weight `xlds_wgm8` candidate. After masked-only epilogue and interleaved N16 ownership, Pro EP16 token128 is bit-exact and improves C K1 to `1.2900 ms` vs DeepGEMM `0.9800 ms` (`1.316x`).
          - ✅ Added an ASM-guided full-tile x-LDS prefetch path and removed redundant loader bounds. Current best Pro EP16 same-input K1-only ratios for tokens `8/32/128/256/512` are `1.243/1.176/1.162/1.120/1.170`, all bit-exact (`max_abs=0`, `mean_abs=0`).
          - 🚫 Rejected directly copying the ASM row-swizzle into the current x-LDS layout; it failed correctness (`max_abs=1.67578125`) and was reverted.
          - ✅ Added inline-asm masked-weight loads and split next-stage prefetch for the current `xlds_wgm8` tune kernel. Best checked token `128/256/512` medians are `1.1491/1.5896/2.4878 ms`, all bit-exact, but still above the `<=1.05x` target.
          - 🚫 Rejected the more aggressive `vmcnt(4)` phase4 prefetch schedule: generated `.s` looked closer to hand ASM, but token128 same-input correctness failed with NaN / `max_abs=0.3076171875`.
          - 🚫 Rejected `#pragma unroll 2` for the x-LDS compute loop: correct and more ASM-like statically, but token512 regressed to `2.6211 ms`.
          - 🚫 Rejected the full-tile/partial-tile store-branch split: static control count improved, but token `128/256/512` runtime was neutral-to-worse (`1.1574/1.6076/2.4895 ms`).
          - 🚫 Rejected moving `s_setprio` outside the K-stage loop: bit-exact but token `128/256/512` regressed to `1.1651/1.6854/2.5505 ms`.
          - ✅ Filled current split-prefetch small-bucket data: token `8/32` medians are `1.1040/1.1511 ms` vs DeepGEMM `0.8599/0.9678 ms`, bit-exact.
          - 🚫 Rejected the reference-style packed epilogue (`v_pk_mul_f32` scale/output pairs): static `.s` improved, but Pro EP16 token `128/256/512` measured `1.1573/1.6495/2.5306 ms`, worse than the retained split-prefetch baseline at the important 256/512 buckets.
          - 🚫 Rejected the LDS-read overlap `lgkmcnt(2)` split: generated `.s` matched the intended partial-wait schedule and stayed bit-exact, but token `128/256/512` measured `1.1465/1.6482/2.5152 ms`; only 128 improved slightly while 256/512 regressed.
          - 🚫 Rejected the four-load inline asm group for masked weights. The first version VMFaulted due to missing early-clobber output constraints; the corrected `=&v` version was bit-exact but token `128/256/512` measured `1.1402/1.6133/2.7117 ms`, with large token512 regression.
          - 🚫 Rejected removing the phase0/phase4 accumulator `s_nop` dependency block: bit-exact, but token256 regressed to `1.6651 ms`; token512's small gain is not enough to justify a bucket-specific path.
          - 🚫 Rejected phase-level priority retiming: bit-exact but token `128/256/512` did not improve (`1.1655/1.5922/2.4961 ms`), so keep the retained split-prefetch priority placement.
          - 🚫 Rejected the `vmcnt(8)` double-buffered next-weight prefetch attempt. It made `.s` look closer to the masked ASM but raised VGPR pressure and VMFaulted in the tune kernel; reverted to the retained split-prefetch baseline and restored token128 bit-exact smoke.
          - 🚫 Rejected unconditional padding-row stores for `xlds_wgm8`. Static `.s` became much closer to the reference store/control shape (`buffer_store_short 128->64`, `s_cbranch 131->19`, `v_cmp 101->5`) and stayed bit-exact, but token `128/256/512` measured `1.1507/1.6186/2.5129 ms`, neutral-to-worse versus the retained split-prefetch baseline.
          - 🚫 Rejected offset-increment weight-address state in `xlds_wgm8`. It reduced a little address arithmetic (`v_lshl 70->65`, `s_mul_i32 10->8`) but raised VGPRs (`123->127`) and regressed token `128/256/512` to `1.1564/1.6807/2.5308 ms`; reverted to direct `K1_XLDS_LOAD_W_AT`.
          - ✅ Restored the retained split-prefetch baseline after the offset-increment rejection. Token128 smoke is bit-exact (`max_abs=0`) with C `1.1435 ms` vs DeepGEMM masked `0.9687 ms`, and generated `.s` for `<24>` is back to `64 v_mmac`, `22 buffer_load_dwordx4`, `8 ds_read_b128`, `128 buffer_store_short`, `217 s_waitcnt`, `123 VGPR`-class behavior.
          - 🚫 Rejected the C-only `xlds_wgm8_n64` structural candidate. It moved N tile from 256 to 64 to resemble `MT256x64x128`, and static counts dropped (`16 v_mmac`, `10 buffer_load_dwordx4`, `32 stores`, `62 waitcnt` per CTA), but token128 slowed to `1.8438 ms` vs DeepGEMM `0.9524 ms`; the extra CTA count/launch scheduling outweighs lower per-CTA state.
          - 🚫 Rejected the next4 early-clobber relaxed-wait schedule. Generated `.s` issued next-stage loads before `vmcnt(4)`, but compiler/control flow still inserted a `vmcnt(0)` and token128 failed correctness with NaN (`max_abs=0.4048`), so C-level pending-VMEM retiming is unsafe without a larger hand-asm block.
          - 🚫 Do not keep blocking fusion on the old `<=1.05x` pure-C target. The retained `xlds_wgm8` pure kernel is bit-exact but still about `1.12x-1.24x` slower than DeepGEMM masked; user direction is now to fuse this current C backbone first and judge by e2e K1/MegaMoE impact.
          - ✅ Fuse the retained Pro `xlds_wgm8` C groupgemm into the Pro LL K1 fused path using the Flash-style structure: route/stage/start-barrier and L1 GEMM in one shape-gated K1 kernel, with the current split `stage-only + DeepGEMM masked K1` path kept as fallback/oracle.
            - ✅ Implemented behind `MEGAMOE_DCU_PRO_LL_MASKED_K1_FUSED_C=1` as an opt-in Pro-only path: `V3_K1_ProMaskedXLdsWgm8FusedKernel` uses the retained x-LDS/WGM8 compute tile after the LL route/stage builder, and the split DeepGEMM masked path remains the default fallback/oracle.
            - ✅ Local Python compile/diff checks passed and the 151.1 full MegaMoE rebuild completed with fresh K1 artifacts.
          - ✅ Ran narrow fused-context A/B/tuning, including barrier retiming and readlane actual-m handling; fused-C stayed slower than split and was rejected.
        - ✅ Use the DeepGEMM masked `256x64x128 WGM8 GROUPGEMM_MASKED` structure as the reference: row tile 64, N tile 256, K tile 128, 512-thread/WGM8 style scheduling, and LDS-backed B-side pipeline.
        - ✅ Preserved LL/masked output orientation and actual-m contract in the split and C/tune experiments.
        - 🚫 Pure-kernel acceptance target was not met and is no longer pursued in this branch; split masked ASM is the final performance path.
      - ✅ Current active implementation step: validate the fused Pro `xlds_wgm8` C K1 e2e path behind a Pro-only env gate, then compare against the split DeepGEMM masked-K1 fallback.
        - ✅ The fused Pro C K1 removes the current split path's extra DeepGEMM launch and uses the route/stage/actual_m metadata produced inside the same K1 kernel.
        - ✅ Keep the current `ll_pro_masked` split path as fallback until the fused C K1 is both correct and faster e2e.
        - ✅ Run Pro EP16 LL eager 128/256/512 A/B: default split DeepGEMM masked K1 versus `MEGAMOE_DCU_PRO_LL_MASKED_K1_FUSED_C=1`. Readlane fused-C is correct but slower than split by about `16.7%/11.2%/7.8%`.
      - 🚫 Extra fused-C validation is not pursued because same-window Pro EP16 LL eager A/B showed fused-C slower than split; split path remains default.
    - ✅ Acceptance for the next optimization phase: fused Pro C K1 should show an e2e benefit over the current split masked-K1 path on at least one important bucket without correctness loss; if not, keep it as an opt-in experiment and retain split DeepGEMM masked as default. Current result chooses the fallback/default split path.

## 2026-07-03 Active Continuation
Status: [] active; queue is partially complete and remaining graph/broad validation waits for 151.1 availability.

- ✅ Flash EP8 normal graph cap8192 replay4096 passed before the new patch (`5607.7 us` median).
- ✅ Verify the K3 tail-reduce hidden-vec external-arg patch with Flash EP8 normal graph cap8192 replay8192 first; this specifically guards against the recent Pro graph hidden fix regressing Flash. Remote build/pytest passed on 151.1, and Flash EP8 normal graph cap8192 now passes replay4096/replay8192 with medians `5636.4/11217.7 us`.
- [ ] Continue the requested queue after replay8192 correctness: Flash LL checks and Pro EP8 normal eager buckets are complete; remaining items are Flash EP16 normal graph retest, prior 8192 graph-bench/cleanup retest, and any still-requested Pro graph/uneven matrix. Treat README HCA/topo settings as reference only: use node-actual settings on 151.1.
  - ✅ Flash EP8 LL graph cap512 replay512 with `ll-masked` baseline passed: fused graph median `1.8144 ms`, baseline graph median `2.1164 ms`, correctness `max_abs=0.000549316`.
  - ✅ Flash EP8 LL graph small-cap direct retest passed: cap8/replay8 fused `0.5446 ms` vs `ll-masked` `0.5452 ms`; cap32/replay32 fused `0.6402 ms` vs `ll-masked` `0.6545 ms`.
  - ✅ Flash EP16 LL graph cap512 with `ll-masked` baseline passed using node-actual 151.1 env, not forced HCA/topo variables.
  - ✅ Pro EP8 512 normal eager passed after the compact-prebuild fix.
  - ✅ Completed Pro EP8 broad normal eager buckets from `1024` through `8192` once container/node state recovered.
  - [] Retest the prior 8192 graph-bench/cleanup instability after the Pro EP8 queue or when specifically prioritized.
- ✅ Flash EP16 LL graph cap512 with `ll-masked` baseline passed after 151.1 recovered. Current replay medians `8/32/128/256/512 -> 0.3763/0.4113/0.5980/0.9915/1.9043 ms`, matching historical cap512 within about `-1%..+1%`; the earlier slow readout is not reproduced under a fair cap512 command.

### 2026-07-03 Revised Priority Queue

- ✅ Recover/confirm 151.1 execution state first: SSH stable, no residual test PIDs, no KFD PIDs, and all 16 HCUs visible before any EP16 run.
- ✅ Flash EP16 LL graph performance regression is the first EP16 priority after recovery. Fair cap512 retest with `--baseline-kind ll-masked` matched historical `ep16_ll_graph_512_rpc_20260701_120729`.
- 🚫 If Flash EP16 LL graph remains slow, run current-vs-previous-commit A/B under the same cap512 command before changing code. Not pursued because the fair retest did not reproduce the slowdown.
- [] Flash EP16 normal graph is the second EP16 priority: retest `4096`, `5120`, and `8192`, with emphasis on the previous `8192` graph-bench instability.
- ✅ Pro EP16 normal graph hidden-vec fix runtime verification: minimized tail-reduce graph replay512 passed, cap5120 full graph buckets passed, and cap8192 graph correctness passed. cap8192 graph bench remains unstable/VMFault during replay benchmarking, matching the broader large-cap graph-bench stability issue rather than the original hidden-vector correctness bug.
- 🚫 Large-cap normal tail-reduce graph post-K3 barrier hypothesis failed and was removed. Flash EP16 normal graph cap8192 still VMFaulted in `rank_barrier_kernel` slot32 after eager correctness passed; adding another barrier kernel is not a valid fix.
- [] Large-cap normal tail-reduce graph instability remains open. Next direction: isolate whether graph replay is corrupting/resetting tail-reduce state before K3 completes, or whether K1 compact-route graph state is being rebuilt with stale graph-runtime/tile metadata. Do not add new hot-path kernels without a narrower proof.
- ✅ Pro EP8 normal eager data collection and 512 isolation are complete; remaining Pro EP8 graph/uneven coverage is separate from this lower-priority isolation item.
  - ✅ Pro EP8 512 fused-only isolation shows the hang is inside MegaMoE normal fused execution, not baseline/correctness comparison.
  - ✅ Pro EP8 512 stage-stop isolation shows `start_barrier` returns and `k1` times out, so the first non-returning stage is normal K1 with `local_experts=48`.
  - ✅ Pro EP8 K1 root cause found in the old normal K1 in-ASM route builder: it publishes/reset-counts only 32 local-expert entries (`rank * 32`, loop bound 32), so Pro EP8 `local_experts=48` can wait forever on experts 32..47.
  - ✅ Narrow fix: force compact prebuild for `local_experts > 32`, matching the already-safe EP16/EP32 path style without changing Flash EP8/EP16/EP32 defaults.
  - ✅ Pro EP8 512 normal eager now returns for K1 stage-stop, full fused-only, and full correctness against `normal-contiguous` (`max_abs=0.000976562`).
  - ✅ Removed the temporary `MEGAMOE_DCU_STAGE_STOP` instrumentation, synced the clean Python source to 151.1, reran remote source pytest (`11 passed`), and reran Pro EP8 512 clean sanity (`max_abs=0.000976562`).
  - ✅ Pro EP8 512 RPC normal eager with baseline bench passed: fused `4.7955 ms`, normal-contiguous baseline `4.7731 ms`, speedup `0.995x`.
  - ✅ No-copy rebuild validation passed on 151.1: build script verified fresh source-tree `.so/.co` artifacts, import paths resolve to `/root/yuguo/DeepGEMM`, source pytest passed (`11 passed`), and Pro EP8 512 RPC normal eager smoke passed (`max_abs=0.000976562`, fused `4.8080 ms`, baseline bench skipped).
  - ✅ Completed broad Pro EP8 RPC normal eager data collection for `512,1024,1025,2048,2050,4096,4097,5120,8192`. Fused medians are `4.7537/5.2085/5.2166/8.0569/8.0917/15.4011/15.3239/17.2820/27.6906 ms`; normal-contiguous baseline medians are `4.8026/7.5860/7.6248/13.1070/13.1054/24.6854/24.7344/30.5266/48.5922 ms`.

## 2026-07-06 Pro LL Split Finalization
Status: ✅ source cleanup, final 151.1 rebuild, and Pro EP16 LL split runtime sanity are complete.

- ✅ Remove Pro LL fused-C and C/tune optimization code paths now that same-window A/B showed they are slower than split DeepGEMM masked-K1.
- ✅ Integrate the DeepGEMM masked FP8 group GEMM ASM launch into MegaMoE's K1 extension so the Pro LL split production path does not import or call the external `deepgemm` package.
- ✅ Make Pro-size LL backend default to the high-performance `ll_pro_masked` L1 layout; only `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` should force the unified-layout LL compatibility path for Pro.
- ✅ Clean static tests and runtime test harness knobs that only supported the rejected fused-C/tune experiments.
- ✅ Remove the remaining pure-K1 groupgemm launcher, `ll_pure_*` pybind/Python arguments, and `--k1-only-*` harness options so the branch exposes only the Pro LL split production path.
- ✅ Local static verification passed after cleanup: Python `py_compile`, `git diff --check`, and source scans for retired Pro LL env knobs / pure-K1 symbols.
- ✅ Update README with the final Pro LL split behavior and the unified-layout compatibility override.
- ✅ Clean the public FP8 weight-transform surface: remove the legacy `transform_fp8_weights_for_mega_moe`, rename the Pro helper to `transform_fp8_weights_for_mega_moe_pro_ll_masked_k1`, and remove unused `v3_layout.py` transform/unpack helpers while keeping the pack5 layout contract helpers.
- ✅ Rebuild on 151.1 and run Pro EP16 LL split sanity/performance after SGLang/prefix-cache released the cards. Default Pro LL now uses packaged develop masked `.co` hash `73184662ec644cf9f4e9cfacec720a15428e84c5f84ad06e6e9e57bfa06543b4`; eager tokens `8/32/64/128/256/512` and graph cap512 replay `8/32/64/128/256/512` all passed against `ll-masked`.
- [ ] Finish Pro EP8 LL split correctness retest on clean cards. Current token256 failing examples show final local reduce matches the visible combine slots, so the next isolation is split-tail on/off under a clean node and then route-slot diagnostics for the first bad `(token, slot, col)`.
- [ ] Remove temporary Pro EP8 LL diagnostics after root cause is fixed: combine-slot assertion detail and route-slot print hooks used for the split-tail/per-route isolation.

## 2026-07-06 Pro LL Correctness And Regression Queue
Status: [] active; ordered by current user priority.

- ✅ Fix Pro EP8 LL uniform correctness first. Start with token256, because it is the known failing bucket and the tightest EP8 LL capacity case.
  - ✅ Reproduce token256 on clean 151.1 cards against `ll-masked`.
  - ✅ Run the same token256 case with split-tail disabled to separate K3 split-tail copy/publish from upstream K1/K2/K3 route values.
  - ✅ Add narrow route-slot diagnostics for the first bad `(token, slot, col)` only. The failing slots had valid rows and pointers but zero K1/K2/K3 values.
  - ✅ Apply the minimal source fix after the failing boundary is proven. Pro EP8 LL masked K1 now uses at least `128` rows/expert so it avoids the failing `size_m=64` masked-ASM launch shape.
- ✅ After token256 passes, validate Pro EP8 LL uniform token512.
- ✅ If token256/512 evidence suggests bucket sensitivity, also validate Pro EP8 LL uniform tokens `8/32/64/128` to rule out a single-bucket coincidence.
- ✅ Brush Pro LL uneven after uniform is clean: EP16 LL uneven first, then EP8 LL uneven.
- [] Reconcile Pro LL split versus `ll-masked` baseline numbers under the runtime-token-aligned baseline policy. The earlier Pro EP8 LL graph baseline numbers were inflated by using graph allocation cap as DeepEP LL dispatch cap; corrected Pro EP8 LL graph cap512 data is now collected, while any final eager/uneven table should cite only runtime-token-aligned runs.
- ✅ Run necessary Flash guardrails after the Pro LL fix to confirm Flash behavior/performance is unaffected.
- ✅ Clean all temporary diagnostics and update README only with stable final behavior. No new README text is needed for the internal Pro EP8 masked-K1 launch-shape guard.
