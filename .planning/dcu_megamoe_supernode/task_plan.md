# DCU MegaMoE Supernode Plan

## Status Convention

- ✅ completed.
- [ ] required but not complete.
- 🚫 abandoned or explicitly not pursued.
- 🧭 optional backlog / observe until new evidence.
- Do not use `[x]`; checked items are normalized to ✅.

## Goal
Support the DCU MegaMoE V3 staged path on supernode EP sizes while keeping this branch focused only on supernode enablement.

Target shapes:
- DSV4 Flash model shape: experts=256, topk=6, hidden=4096, intermediate=2048.
- EP8 remains supported and must keep the existing behavior.
- EP16 and EP32 must be accepted for supernode runs, with local experts 16 and 8 respectively.

Scope boundaries:
- This branch only implements supernode support. Do not re-open LL_V2 or unrelated kernel experiments.
- Supernode runtime environment is not ready, so final verification is local/source review only in this turn.
- Preserve performance-first contracts: no extra hot-path kernels, no runtime weight transforms, and keep 8-card IPC path unchanged.

## Phase 1: Read Examples And Current Assumptions
Status: ✅ completed

- ✅ Read `hygon_tmp/supernode_code` C++ examples and Galaxy supernode interface PDF.
- ✅ Identify current EP8/256 hard gates in Python/C++ source.
- ✅ Identify rank-count assumptions inside K1/K3 wrappers and source guards.
- ✅ Decide minimal code changes for EP16/EP32 without perturbing EP8 behavior.

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
- ✅ Keep default EP8/local IPC path unchanged.
- ✅ Use the new path only when requested or when rank count exceeds the local IPC comfort zone.
- ✅ Ensure destroy/cleanup closes the matching handle type.
- ✅ Build-review HSA include/link compatibility against the example code and link `hsa-runtime64`.

## Phase 4: Static Review
Status: ✅ completed

- ✅ Run local Python compile checks and source-guard string checks that do not require DCU hardware.
- ✅ Review hot-path kernels for `kMaxSignalRanks`, 16-slot scratch, or EP8-only assumptions.
- ✅ Update findings/progress with remaining runtime validation checklist.

## Runtime Validation Checklist
Status: [ ] deferred until supernode environment is ready

- [ ] Build the HIP wheel on the target DTK image and confirm `hsa_ext_rpc_memory_*` symbols/headers match.
- [ ] EP16/EP32 process-group smoke test: buffer allocation, pointer exchange, `set_mega_moe_peer_ptrs`, and destroy cleanup.
- [ ] EP16/EP32 LL eager/graph correctness and performance.
- [ ] EP16/EP32 normal eager/graph correctness and performance, especially normal ASM tail-reduce signal handling.

## Risks
- Risk: Supernode fabric memory APIs may differ across DTK versions; code should prefer current HIP wrappers when available or use HSA extensions from the examples.
- Risk: EP16/EP32 reduce local expert count and can change K1/K3 grid shape. Validators must not be the only change if kernels assume `local_experts == 32`.
- Risk: Signal scratch slots currently include 16-entry assumptions in tail-reduce setup; EP32 may need larger signal-address storage.
