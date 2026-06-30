# Findings

## Current Status

- ✅ EP8/EP16/EP32 shape gates, scratch sizing, signal-slot helpers, and peer-memory path are implemented on the supernode branch.
- ✅ EP16 single-node runtime bring-up now builds and passes staged smoke on node22.
- [ ] EP16 benchmark matrix and EP32 multi-node validation are still pending.
- 🧭 Future tuning should start from profiler data on real supernode hardware, not from EP8 assumptions.

## Supernode Examples
- `hygon_tmp/supernode_code/p2p_internode.cpp` uses HSA RPC memory APIs:
  - `hsa_ext_rpc_memory_create(ptr, size, handle)` to export a local fine-grained/uncached GPU allocation;
  - `hsa_ext_rpc_memory_attach(handle, 1, &agent, &mapped_ptr)` to import a remote GPU allocation to the current GPU agent;
  - `hsa_ext_rpc_memory_detach(mapped_ptr)` to release imports.
- The example maps HIP device id to HSA agent by comparing HIP PCI bus id with HSA `HSA_AMD_AGENT_INFO_BDFID` and `HSA_AMD_AGENT_INFO_DOMAIN`.
- `p2p_test.cpp` keeps the local rank pointer as the original allocation, imports every peer handle, copies the pointer table to device, and writes remote memory from a GPU kernel.
- `test_shmem_all.cpp` demonstrates an all-rank handle exchange with MPI, attaching all peer buffers through RPC memory, then writing a ring remote pointer.
- Galaxy PDF documents new HIP-level wrappers equivalent to HSA RPC memory:
  - `hipFabricHandle_t` wraps `hsa_ext_rpc_memory_t`;
  - `hipDeviceAgent` wraps `hsa_agent_t`;
  - `hipHsaExtRpcMemoryCreate`, `hipHsaExtRpcMemoryAttach`, `hipHsaExtRpcMemoryAttachDevice`, `hipHsaExtRpcMemoryDetach`.

## Current MegaMoE Assumptions
- `megamoe/__init__.py` previously only accepted `(num_ranks, num_experts, topk, hidden, intermediate) == (8, 256, 6, 4096, 2048)` for the staged path. This is now generalized to DSV4 Flash EP8/EP16/EP32.
- `megamoe/opt.py::_check_shape` repeated the same EP8-only gate. This is now generalized to `{8,16,32}`.
- `megamoe/dcu_megamoe_opt/csrc/apis/mega_dcu.hpp::get_mega_moe_route_scratch_size_for_mega_moe` had an EP8-only staged route scratch gate. This is now generalized and reserves `2 * num_ranks` signal address entries.
- `SymmBuffer` previously allocated and opened peer buffers with HIP IPC only. It now keeps HIP IPC for EP8 and uses HSA RPC/Fabric handles automatically when `num_ranks > 8`.
- `layout.cuh` already computes header offsets from `num_ranks`; the pointer header itself is not fixed to 8.

## Implementation Notes
- Signal slots now preserve the EP8 layout exactly. For EP16/EP32, tail/copy done slots start at `num_ranks`, start-barrier slots follow that window, post-K3 barrier follows start-barrier, and split-tail chunk-ready slots follow post-K3.
- K1 and K3 LL kernels now instantiate `kExperts` for local experts `{8,16,32}`. The readlane broadcast remains bounded by 32 lanes.
- K3 split-tail publish aggregation now keeps its shared-memory fast path up to 32 ranks instead of falling back to per-row publishing above 8 ranks.
- Normal ASM tail-reduce host validation now allows `asm_signal_num_ranks <= 32`; real supernode hardware is still needed to confirm the code object follows `asm_signal_num_ranks` rather than assuming the old 8-rank second-half offset.

## 2026-06-29 EP16 Bring-Up Findings

- Node22 (`10.17.162.22`) is currently the usable 16-card test node. Node69 (`10.17.160.69`) showed persistent 94% VRAM on all cards with no readable KFD PID list, so EP32 is blocked until that node is free.
- The old DeepGEMM whl requested for isolation exists on both nodes and was extracted, but it is ABI-incompatible with the active torch in `yiqa_deepep`; the missing symbol differs by `int` vs `unsigned int` in `c10_hip_check_implementation`.
- EP16 on a single node must use HIP IPC peer memory. The supernode Fabric/RPC path should only be selected when the process-group rank count exceeds the local visible DCU count.
- K1 normal EP16 uncovered an old ASM self-route assumption: the self-route path can VMFault with 16 local experts, while compact prebuild produces valid route metadata. The production default now switches to compact prebuild for `num_ranks > 8`.
- Normal K3 ASM tail-reduce is not safe for EP16 yet. K1/K2/K3 main compute plus external post-K3 reduce passes, while ASM tail-reduce hangs. Normal backend now disables ASM tail-reduce for `num_ranks > 8`; LL backend is unchanged.

## 2026-06-29 Hybrid Symm Buffer Finding

- TX32 EP8/EP16 sanity should select the supernode-aware `peer_mode=hybrid` path even when all ranks are on one 16-DCU node. This validates the new symm-buffer handle exchange and keeps behavior aligned with future EP32 runs.
- Hybrid does not mean every peer is opened through Fabric/RPC. Same-host peers must use HIP IPC; forcing local peers through `hsa_ext_rpc_memory_attach` failed with HSA status `4104` and is abandoned.
- Cross-host peers remain the only intended Fabric/RPC users. This keeps EP8/EP16 single-node latency close to IPC while making EP32 pointer tables compatible with the two-node supernode layout.
- The yuguo old DeepGEMM baseline is still a separate compatibility axis: MegaMoE hybrid EP8 passed against the built-in DeepGEMM baseline, while the old whl path needs ABI/layout/storage workarounds before it can be used as the final comparison baseline.

## 2026-06-30 EP16 Normal Capacity Finding

- EP16 normal 2048 was not a K3 reduce error. The top mismatches showed one missing top-k combine slot while the other slots were valid. K1 route probes showed the corresponding route lacked an `output_index`, so K2/K3 never had a row to compute/write.
- The immediate cause was tight compact route capacity in K1. `k1_build_compact_tiles_kernel` truncates later local experts when the compact active-tile estimate is too small; the observed failures clustered on expert 63, the last local expert for rank 3.
- The low-risk fix is to keep compact prebuild for EP16/EP32, but use fixed capacity when compact is forced for `num_ranks > 8`. EP8 auto compact remains on the old estimated-capacity path.
- Python scratch views must use the same K1 capacity headroom formula as the C++ extension and C API. A mismatch left `l1_out_workspace` too small at 8192 tokens even though the route scratch allocation itself was large enough.
- LL graph tests with the `ll-masked` baseline require DeepEP LL ROCSHMEM sizing. On node22, `ROCSHMEM_HEAP_SIZE=4737418240`, `ROCSHMEM_IPC_MNVL=1`, and `ROCSHMEM_GDR_DISABLE_XDP=1` avoided the DeepEP buffer init failure.
- Large normal 8192 correctness produced valid output but stalled during test finalization/cleanup. Treat this as a cleanup/teardown issue to investigate separately from numeric correctness.

## 2026-06-30 EP16 Normal 8192 Baseline Finding

- EP16 normal 8192 MegaMoE output is numerically correct against the normal-contiguous baseline on rank0 (`max_abs=0.000488281`), but the distributed test does not finish because at least one rank remains inside the correctness path before the cleanup barrier.
- The stall is not caused by `sym_buffer.destroy()` or MegaMoE kernel teardown in the fused-only path. When DeepEP baseline timing is skipped, MegaMoE normal 8192 benchmark exits cleanly.
- `--prepost-backend triton` is not a workaround for the 8192 baseline issue in the current environment; Triton AMD compilation failed with missing temporary `.amdgcn` output.
- Keep 8192 normal data separated in reports:
  - buckets through 5120 have full MegaMoE-vs-baseline comparisons;
  - 8192 currently has MegaMoE-only timing plus correctness evidence, but no safe DeepEP normal baseline timing.

## 2026-06-30 EP16 Versus EP8 Performance Readout

- EP16 LL graph improves over the latest EP8 graph data for small and mid buckets:
  - 8: EP16 `0.3725 ms` vs EP8 `0.5473 ms`;
  - 32: EP16 `0.4033 ms` vs EP8 `0.6089 ms`;
  - 64: EP16 `0.4773 ms` vs EP8 `0.6600 ms`;
  - 128: EP16 `0.6053 ms` vs EP8 `0.7483 ms`;
  - 256: EP16 `1.0378 ms` vs EP8 `~1.0832 ms`.
- EP16 LL 512 does not improve in the current single-node run: EP16 `1.9711 ms` vs EP8 `~1.7739 ms`. This may be a real rank-count/local-expert tradeoff, not necessarily a correctness issue.
- EP16 normal 512 improves over EP8 normal 512: EP16 `1.3090 ms` vs EP8 `1.7629 ms`.
- EP16 normal 1024/1025 regress versus the latest EP8 reference:
  - 1024: EP16 `2.7494 ms` vs EP8 `2.0397 ms`;
  - 1025: EP16 `2.7335 ms` vs EP8 `2.1879 ms`.
- Historical EP8 normal large-token references are also faster than the current EP16 single-node numbers (`4096 ~5.8 ms`, `8192 ~10.9 ms` vs EP16 `4096 8.1634 ms`, `8192 14.9958 ms`). This needs profiler confirmation before treating EP16 as a performance win for normal medium/large buckets.

## 2026-06-30 dcu_mega_v3 EP8 A/B Finding

- `dcu_mega_v3` EP8 reference runs do not need Fabric/RPC buffer changes. They should keep the original single-node HIP IPC peer path. The supernode branch EP8 uses the new hybrid selector only to validate the wrapper; same-host peers still open through HIP IPC.
- The yuguo old DeepGEMM whl is not a reliable primary reference in the current `yiqa_deepep` torch runtime. It can be imported with an ABI shim, but its op is sensitive to current-torch tensor views/storage and old Marlin padding semantics.
- Current environment DeepGEMM requires contiguous normal baseline weight layout (`[E,K/64,N/16,4,16,16]`). The `dcu_mega_v3` compare test needed a temporary baseline-layout helper to run against that package; this does not change MegaMoE fused kernels or fused weight layout.
- `dcu_mega_v3` EP8/4096 normal eager on node22 with current DeepGEMM:
  - correctness `max_abs=0.000671387`;
  - MegaMoE `7.1299 ms`;
  - baseline `9.8677 ms`;
  - speedup `1.384x`.
- Current supernode branch EP8/4096 is in the same band (`~7.2 ms` formal run, stage breakdown also close), so there is no evidence that the supernode wrapper regressed EP8 4096. The EP16 normal large-token slowdown remains attributable to K3 peer combine scatter/write fanout rather than to pure GEMM, K1 capacity, or EP8 hybrid peer-memory selection.

## 2026-06-30 - EP8 Baseline Mismatch Root Cause Narrowing

- The EP8 `4096` normal eager gap is not caused by the supernode hybrid buffer/handle code path. Evidence: normal-node dirty worktree source, rebuilt on TX32 node22 outside the supernode branch source tree, still measures `~7.1-7.3 ms`.
- The historical `~5.8 ms` result reproduces on the normal 8-card node with its own `dcu_mega_v3` worktree, so the historical record itself is not stale.
- TX32 and normal-node environments differ materially: hardware SKU (`BW1301_LC` vs `BW1101`), torch version (`2.10.0` vs `2.9.0`), DeepGEMM package ABI/layout, DTK install path/package, and 16-card vs 8-card local visibility.
- Local topology reports do not explain the gap: both systems report all-local HSW links and 1-hop peer access.
- Until a TX32 runtime/clock/package normalization experiment proves otherwise, use TX32 EP8 `~7.1 ms` as the relevant same-machine baseline for EP16/EP32 supernode work, and keep the normal-node `~5.8 ms` as a cross-machine historical reference only.
