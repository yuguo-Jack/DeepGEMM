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
- EP8 remains supported. Peer-memory mode is independent from EP size: default is HIP IPC, and Fabric/RPC is enabled only with `MEGAMOE_DCU_PEER_MEMORY=rpc`.
- EP16 and EP32 must be accepted for supernode runs, with local experts 16 and 8 respectively.

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
- ✅ EP16 correctness bring-up fixes: single-node IPC peer mode, K1 compact prebuild default for `num_ranks > 8`, and normal K3 ASM tail-reduce disabled for `num_ranks > 8`.
- [ ] Re-run IPC default and `MEGAMOE_DCU_PEER_MEMORY=rpc` Fabric/RPC smoke once hardware is available; EP size should not affect peer-memory selection.
- ✅ 151.1 EP8 8-card RPC smoke on devices `0..7`: `LL graph`, `LL eager`, `normal eager`, `normal graph`, `LL graph uneven`, and `normal eager uneven` pass with `normal-contiguous` baseline after Fabric buffer export size was aligned to 2 MiB.
- [ ] 151.1 true EP16 RPC validation is currently blocked by active `chl_sgl0512` card occupancy, not by a reproducible device15 failure. On 2026-07-01, device15 small torch allocation succeeds in both `chl_sgl0512` and restarted `sglang_megamoe`.
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
- [ ] Investigate DeepEP normal baseline `8192` EP16 rank hang if a full baseline comparison at this bucket is still required.
- ✅ Root-cause EP16 normal medium/large-token regression versus EP8 references (`1024+`, especially `4096/8192`): stage timing shows the pure K3 GEMM path is healthy, while K3 peer combine scatter/write dominates the extra EP16 cost.
- [ ] Optimize normal K3 peer combine writes for EP16/EP32 without adding extra hot-path kernels; start from row pointer locality, destination-rank batching that preserves ASM-friendly row order, or a split local-output plus combine strategy only if profiling proves it wins.
- [ ] EP32 LL eager/graph correctness and performance.
- [ ] EP32 normal eager/graph correctness and performance, especially cross-node Fabric/RPC and signal handling.

## Risks

- Risk: Supernode fabric memory APIs may differ across DTK versions; code should prefer current HIP wrappers when available or use HSA extensions from the examples.
- Risk: EP16/EP32 reduce local expert count and can change K1/K3 grid shape. Validators must not be the only change if kernels assume `local_experts == 32`.
- Risk: Signal scratch slots currently include 16-entry assumptions in tail-reduce setup; EP32 may need larger signal-address storage.
- Risk: The requested yuguo old DeepGEMM baseline needs an ABI shim in the active torch runtime and has layout/padding sensitivity; keep that baseline isolated from MegaMoE correctness diagnosis.
