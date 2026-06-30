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
- EP8 remains supported. On TX32/supernode nodes it must exercise the new supernode-aware hybrid symm buffer selection; legacy 8-DCU standalone environments keep the old IPC-only path.
- EP16 and EP32 must be accepted for supernode runs, with local experts 16 and 8 respectively.

Scope boundaries:
- This branch only implements supernode support. Do not re-open LL_V2 or unrelated kernel experiments.
- TX32 runtime validation is active; keep node-local and cross-node results separated because the two nodes do not share storage.
- Preserve performance-first contracts: no extra hot-path kernels and no runtime weight transforms. In hybrid symm buffer mode, same-host peers use HIP IPC and only cross-host peers use Fabric/RPC.

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
- ✅ Use the hybrid path when the runtime environment indicates a TX32/supernode node, including EP8 sanity runs on 16-DCU nodes.
- ✅ In hybrid mode, use HIP IPC for same-host peers and Fabric/RPC only for cross-host peers.
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
- ✅ EP8 TX32 sanity uses `peer_mode=hybrid`: same-host peer handles are opened with HIP IPC, while the mode selection still validates the supernode-aware buffer path.
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
