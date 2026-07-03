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
Status: [] Pro shape support is implemented; EP16 normal eager is validated through 8192, Pro EP8 512 is fixed, no-copy build packaging is fixed, and remaining work is broad Pro EP8 plus large graph instability.

Target:
- Shape: experts=384, topk=6, hidden=7168, intermediate=3072.
- Supported EP sizes: EP8/EP16/EP32.
- Runtime priority: Pro EP16 correctness first; EP8/EP32 remain source-supported and runtime-deferred unless explicitly requested.
- Flash guardrail: existing DeepSeek-V4-Flash correctness and performance must not materially regress. If shared dynamic normal ASM hurts Flash, split Pro-only K1/K3 normal ASM code objects and keep Flash on the original constant-address path.

Current snapshot (2026-07-03):
- ✅ Source and dispatch gates cover Pro EP8/EP16/EP32; active runtime validation focused on EP16 as requested.
- ✅ Pro EP16 normal correctness/performance passed at 512, 1024, 2048, and 4096 tokens against `normal-contiguous`; Pro EP16 LL also passed the same baseline oracle.
- ✅ Flash EP8 normal 4096 guardrail remains in the same performance band after the shared K1 ASM fix, so no Pro-only kernel split is currently justified.
- ✅ Temporary Python debug hooks, backend rerun flags, Pro-only weight-layout override, and extra eager assertion formatting have been removed from the test harness.
- 🚫 K1/K3 publish fences, store-side `glc`, K2/reduce `glc/slc`, K3 sleep/sync/tail-reduce/store probes, and the Pro-only layout override were diagnostic paths with negative or no production value.
- ✅ README documents both Flash and Pro staged shapes and includes a Pro EP16 validation example.
- ✅ Pro EP16 normal eager 8192 is stable for fused correctness/execution. Pro EP16 graph cap8192 correctness is fixed, but replay8192 graph bench/cleanup remains unstable.
- ✅ Pro EP8 512 normal eager is fixed by forcing compact prebuild for `local_experts > 32`; clean IPC sanity and RPC baseline-bench sanity passed.
- ✅ Fix the remote packaging/install flow so source-tree imports do not require manually copying fresh built `.so` files.
- [ ] Pro EP8 broad runtime validation remains incomplete after 512 because the first 1024 bucket attempt was interrupted by 151.1 host/SSH instability. Pro EP32 runtime validation remains deferred.
- [ ] 2026-07-02 requested 151.1 validation batch: resync current code, rebuild, rerun Flash EP8/EP16 normal eager+graph buckets `512,1024,1025,2048,2050,4096,4097,5120,8192`, Flash LL graph buckets `8,32,128,256,512`, sample Flash uneven eager/graph, compare against V3/supernode historical Flash data, then run the same bucket/rule set for Pro EP8/EP16 and record the Pro data.
- ✅ Rechecked Flash EP16 LL graph with a fair cap512 `--baseline-kind ll-masked` command after 151.1 recovery. It matched historical cap512 within about `-1%..+1%`; no code regression was reproduced.
- 🚫 Current-vs-previous-commit A/B and LL-risk bisect are not pursued now because the fair Flash EP16 LL graph retest passed. Keep this only as a contingency if a future fair cap512 run regresses.

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
- [ ] If a future Flash performance run regresses materially, introduce Pro-only normal ASM code objects and route by shape.
- [ ] Keep EP8/EP32 Pro runtime validation deferred until requested or until EP16 is clean.

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

## 2026-07-03 Active Continuation
Status: [] active on 151.1; host/cards are visible and idle, and `sglang_megamoe` has been restarted for the current validation loop.

- ✅ Flash EP8 normal graph cap8192 replay4096 passed before the new patch (`5607.7 us` median).
- ✅ Verify the K3 tail-reduce hidden-vec external-arg patch with Flash EP8 normal graph cap8192 replay8192 first; this specifically guards against the recent Pro graph hidden fix regressing Flash. Remote build/pytest passed on 151.1, and Flash EP8 normal graph cap8192 now passes replay4096/replay8192 with medians `5636.4/11217.7 us`.
- [ ] Continue the requested queue after replay8192 correctness: Flash LL checks are complete; Pro EP8 broad buckets and prior 8192 graph instability retest remain. Treat README HCA/topo settings as reference only: use node-actual settings on 151.1, and do not force stale `mlx5_*` topology when `/sys/class/infiniband` is empty.
  - ✅ Flash EP8 LL graph cap512 replay512 with `ll-masked` baseline passed: fused graph median `1.8144 ms`, baseline graph median `2.1164 ms`, correctness `max_abs=0.000549316`.
  - ✅ Flash EP16 LL graph cap512 with `ll-masked` baseline passed using node-actual 151.1 env, not forced HCA/topo variables.
  - ✅ Pro EP8 512 normal eager passed after the compact-prebuild fix.
  - [] Continue Pro EP8 broad normal eager buckets from `1024` once container/node state is healthy.
  - [] Retest the prior 8192 graph-bench/cleanup instability after the Pro EP8 queue or when specifically prioritized.
- ✅ Flash EP16 LL graph cap512 with `ll-masked` baseline passed after 151.1 recovered. Current replay medians `8/32/128/256/512 -> 0.3763/0.4113/0.5980/0.9915/1.9043 ms`, matching historical cap512 within about `-1%..+1%`; the earlier slow readout is not reproduced under a fair cap512 command.

### 2026-07-03 Revised Priority Queue

- ✅ Recover/confirm 151.1 execution state first: SSH stable, no residual test PIDs, no KFD PIDs, and all 16 HCUs visible before any EP16 run.
- ✅ Flash EP16 LL graph performance regression is the first EP16 priority after recovery. Fair cap512 retest with `--baseline-kind ll-masked` matched historical `ep16_ll_graph_512_rpc_20260701_120729`.
- 🚫 If Flash EP16 LL graph remains slow, run current-vs-previous-commit A/B under the same cap512 command before changing code. Not pursued because the fair retest did not reproduce the slowdown.
- [] Flash EP16 normal graph is the second EP16 priority: retest `4096`, `5120`, and `8192`, with emphasis on the previous `8192` graph-bench instability.
- ✅ Pro EP16 normal graph hidden-vec fix runtime verification: minimized tail-reduce graph replay512 passed, cap5120 full graph buckets passed, and cap8192 graph correctness passed. cap8192 graph bench remains unstable/VMFault during replay benchmarking, matching the broader large-cap graph-bench stability issue rather than the original hidden-vector correctness bug.
- [] Pro EP8 data collection is lower priority than the Flash EP16 graph regression. If 16-card recovery remains blocked, use available 0..7 cards only to isolate the Pro EP8 512 hang: compare IPC vs RPC/fabric and K3 tail-reduce0 vs tail-reduce1 before any broad Pro EP8 batch.
  - ✅ Pro EP8 512 fused-only isolation shows the hang is inside MegaMoE normal fused execution, not baseline/correctness comparison.
  - ✅ Pro EP8 512 stage-stop isolation shows `start_barrier` returns and `k1` times out, so the first non-returning stage is normal K1 with `local_experts=48`.
  - ✅ Pro EP8 K1 root cause found in the old normal K1 in-ASM route builder: it publishes/reset-counts only 32 local-expert entries (`rank * 32`, loop bound 32), so Pro EP8 `local_experts=48` can wait forever on experts 32..47.
  - ✅ Narrow fix: force compact prebuild for `local_experts > 32`, matching the already-safe EP16/EP32 path style without changing Flash EP8/EP16/EP32 defaults.
  - ✅ Pro EP8 512 normal eager now returns for K1 stage-stop, full fused-only, and full correctness against `normal-contiguous` (`max_abs=0.000976562`).
  - ✅ Removed the temporary `MEGAMOE_DCU_STAGE_STOP` instrumentation, synced the clean Python source to 151.1, reran remote source pytest (`11 passed`), and reran Pro EP8 512 clean sanity (`max_abs=0.000976562`).
  - ✅ Pro EP8 512 RPC normal eager with baseline bench passed: fused `4.7955 ms`, normal-contiguous baseline `4.7731 ms`, speedup `0.995x`.
  - ✅ No-copy rebuild validation passed on 151.1: build script verified fresh source-tree `.so/.co` artifacts, import paths resolve to `/root/yuguo/DeepGEMM`, source pytest passed (`11 passed`), and Pro EP8 512 RPC normal eager smoke passed (`max_abs=0.000976562`, fused `4.8080 ms`, baseline bench skipped).
  - [] Continue broad Pro EP8 RPC normal eager data collection for `1024,1025,2048,2050,4096,4097,5120,8192` after 151.1 SSH/node state recovers. The first 1024 batch attempt was interrupted by host-side SSH closure before any result JSON.
