# Findings

## Current Status

- ✅ EP8/EP16/EP32 shape gates, scratch sizing, signal-slot helpers, and peer-memory path are implemented on the supernode branch.
- [ ] Runtime build, correctness, and performance validation are deferred until a 16/32-card supernode environment is ready.
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
- `SymmBuffer` previously allocated and opened peer buffers with HIP IPC only. It now keeps HIP IPC for EP8 and uses HSA RPC/Fabric handles automatically for ranks > 8 or when `MEGAMOE_DCU_SUPERNODE=1`.
- `layout.cuh` already computes header offsets from `num_ranks`; the pointer header itself is not fixed to 8.

## Implementation Notes
- Signal slots now preserve the EP8 layout exactly. For EP16/EP32, tail/copy done slots start at `num_ranks`, start-barrier slots follow that window, post-K3 barrier follows start-barrier, and split-tail chunk-ready slots follow post-K3.
- K1 and K3 LL kernels now instantiate `kExperts` for local experts `{8,16,32}`. The readlane broadcast remains bounded by 32 lanes.
- K3 split-tail publish aggregation now keeps its shared-memory fast path up to 32 ranks instead of falling back to per-row publishing above 8 ranks.
- Normal ASM tail-reduce host validation now allows `asm_signal_num_ranks <= 32`; real supernode hardware is still needed to confirm the code object follows `asm_signal_num_ranks` rather than assuming the old 8-rank second-half offset.
