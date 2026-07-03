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
- Superseded on 2026-07-01: the normal K3 ASM tail-reduce hang was traced to the old EP8-only signal layout. The current fix expands signal/wait handling to EP16/EP32 for explicit ablation, but EP16/EP32 normal defaults back to tail-reduce0 / external local reduce; EP32 tail-reduce1 still needs runtime validation if re-enabled.

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
- Current supernode branch EP8/4096 is in the same band (`~7.2 ms` formal run, stage breakdown also close), so there is no evidence that the supernode wrapper regressed EP8 4096. Historical readout at that time attributed the EP16 normal large-token slowdown to K3 peer combine scatter/write fanout rather than to pure GEMM, K1 capacity, or EP8 hybrid peer-memory selection.
- Superseded on 2026-07-01 by 151.1 EP8/EP16 RPC validation: EP8 `4096` recovered to `5.7636 ms`, and EP16 normal eager/graph are correct and faster than baseline. Treat K3 peer-combine rewrite as not currently required unless a future profiler run produces new evidence.

## 2026-06-30 - EP8 Baseline Mismatch Root Cause Narrowing

- The EP8 `4096` normal eager gap is not caused by the supernode hybrid buffer/handle code path. Evidence: normal-node dirty worktree source, rebuilt on TX32 node22 outside the supernode branch source tree, still measures `~7.1-7.3 ms`.
- The historical `~5.8 ms` result reproduces on the normal 8-card node with its own `dcu_mega_v3` worktree, so the historical record itself is not stale.
- TX32 and normal-node environments differ materially: hardware SKU (`BW1301_LC` vs `BW1101`), torch version (`2.10.0` vs `2.9.0`), DeepGEMM package ABI/layout, DTK install path/package, and 16-card vs 8-card local visibility.
- Local topology reports do not explain the gap: both systems report all-local HSW links and 1-hop peer access.
- Until a TX32 runtime/clock/package normalization experiment proves otherwise, use TX32 EP8 `~7.1 ms` as the relevant same-machine baseline for EP16/EP32 supernode work, and keep the normal-node `~5.8 ms` as a cross-machine historical reference only.

## 2026-07-01 - 151.1 Environment Usage Finding

- Node `10.17.151.1` basic access and mount state:
  - SSH key login as `root` works when the node is healthy.
  - Docker container `sglang_megamoe` is the intended MegaMoE test container.
  - Host `/root/yuguo` is mounted into the container as `/root/yuguo`; repo path is `/root/yuguo/DeepGEMM`.
  - Container OS is Ubuntu 22.04.5; Python is 3.10.12; torch is 2.10.0 with HIP 6.3.26113.
- Before every test on 151.1, check card occupancy first. A running SGLang service can occupy all 16 cards (`--tp-size 16 --ep-size 16 --dp-size 16`, port `10015`), with scheduler processes and a benchmark client. Do not run MegaMoE tests while those jobs own the cards.
- `hy-smi --showpids` can fail with `Unable to open process directory` on this node. When that happens, use host-side `ps` to map the actual SGLang or benchmark processes before deciding whether the node is free.
- Torch import can fail if `/opt/hyhal/lib/libamd_smi.so` wins library resolution and lacks `amdsmi_init`. Keep Python's amdsmi package library first:
  - `export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:${LD_LIBRARY_PATH:-}`
- DTK path must be selected deliberately per run and not mixed accidentally. For the current 151.1 MegaMoE validation flow, use the user-specified `/root/yuguo/dtk-26.04.1/env.sh` path when running from `/root/yuguo/DeepGEMM`; do not append `/opt/dtk` fallback libraries into the same test environment unless the user explicitly switches the run to `/opt/dtk`.
- If SSH starts closing immediately, stop launching new tests and recheck node health later; do not assume the previous MegaMoE run is still valid or complete.

## 2026-07-01 - Normal K3 Tail-Reduce1 EP16/EP32 Finding

- The EP16/EP32 tail-reduce1 bug is in the K3 ASM tail-reduce signal protocol, not in the pure groupgemm path or the external local reduce path.
- The original K3 tail-reduce source published completion for only 8 ranks and waited on fixed offsets that assume `num_ranks == 8`.
- Correct signal addressing must scale with `asm_signal_num_ranks`:
  - EP8 wait base: `8 * sizeof(int64_t) = 64`.
  - EP16 wait base: `16 * sizeof(int64_t) = 128`.
  - EP32 wait base: `32 * sizeof(int64_t) = 256`.
- The code now expands signal and wait macros through rank 31. EP16 smoke confirms correctness for `512` and `4096`; EP32 validation remains pending because no healthy 32-card environment is currently available.

## 2026-07-01 - EP16/EP32 Eager Normal Active-Tile Attempt

- 🚫 Abandoned and reverted. The attempt passed active-tile metadata from eager normal K1 into K2/K3/tail-reduce to avoid capacity-row work after EP16/EP32 compact prebuild.
- Validation on 151.1 showed only marginal recovery at 4096 and no improvement at larger buckets compared with the previous no-tail / external-reduce path:
  - EP16 normal eager tail-reduce1 active-tile `4096`: correct, MegaMoE `6.6835 ms`, baseline `10.4333 ms`.
  - EP16 normal eager tail-reduce1 active-tile `5120`: correct, MegaMoE `8.3056 ms`, baseline `12.8190 ms`.
  - EP16 normal eager tail-reduce1 active-tile `8192`: MegaMoE-only `12.8188 ms`.
- Conclusion: capacity-row inflation is not the dominant remaining EP16 big-token regression in this branch. Do not keep extra eager active-tile plumbing unless a later profiler shows a clearer K2/K3 padding bottleneck.

## 2026-07-01 - EP16 Normal Graph Readout

- EP16 RPC normal graph is functional on single-node 16-card `151.1` with `MEGAMOE_DCU_PEER_MEMORY=rpc` and `K3_USE_ASM_TAIL_REDUCE=1`.
- Correctness passed for uniform cap512, uneven cap512, uniform cap4096, uneven cap4096, plus uneven cap1024 and cap2048.
- Graph replay latency scales smoothly across tested buckets:
  - uniform cap512: `128/256/512 -> 0.9027/1.0053/1.2542 ms`;
  - uneven cap512: `128/256/512 -> 0.8411/0.9076/1.1298 ms`;
  - uneven cap1024: `512/1024 -> 1.1037/1.7890 ms`;
  - uneven cap2048: `1024/2048 -> 1.8058/3.2687 ms`;
  - uniform cap4096: `1024/2048/4096 -> 2.0467/3.3681/6.4547 ms`;
  - uneven cap4096: `1024/2048/4096 -> 1.8613/3.3350/6.4743 ms`.
- Uneven graph cases are not showing a correctness-specific or signal-specific failure. The remaining large-token EP16 concern is still performance scaling versus EP8, not graph correctness.
- The `cached_notify_combine` launch-bound warnings are emitted by the normal-contiguous DeepEP baseline path; they did not block MegaMoE normal graph replay validation.

## 2026-07-01 - DeepSeek-V4-Pro Shape Read-In

- New target shape from DeepGEMM PR #316: DeepSeek-V4-Pro uses experts=384, topk=6, hidden=7168, intermediate=3072. Supported EP sizes should remain EP8/EP16/EP32, which implies local experts 48/24/12.
- The current staged shape contract is still Flash-only in `megamoe/__init__.py`, `megamoe/opt.py`, `K1_fused/k1_fused.py`, and `csrc/apis/mega_dcu.hpp`.
- K1/K3 LL launch dispatch currently instantiates only local experts 8/16/32 and hard-codes Flash dimensions. Pro EP16/EP32 fit the old <=32 local-expert limit, but Pro EP8 needs local_experts=48.
- K1/K3 LL groupgemm headers use `static_assert(kExperts <= 32)` and per-expert done-counter counts of 32. Pro EP8 requires increasing these per-expert counters to 64 and validating readlane assumptions.
- K1 normal compact prebuild has a fixed compact header assumption (`65 + capacity_tiles`) built around 32 local experts. Pro EP8 requires dynamic compact header offsets based on `2 * local_experts + 1`.
- Normal ASM wrappers gate on Flash dims, but the host `GpuProb` already carries dynamic `hidden`/`intermediate` values and the ASM sources contain references to 7168/3072 paths. Treat these as source gates to lift carefully, then validate by build/runtime.

## 2026-07-01 - DeepSeek-V4-Pro EP16 Workspace Finding

- Pro changes an old Flash-only coincidence: Flash has `hidden == 2 * intermediate == 4096`, while Pro has K1 L1 output columns `2 * 3072 = 6144` and K3 output hidden `7168`.
- Therefore K3 cannot reuse the K1 `l1_out` BF16 workspace for Pro. The route scratch layout must reserve a separate `k3_out` BF16 workspace when `hidden > 2 * intermediate`, and Python plus C++ route-scratch sizing must mirror this exactly.
- The first Pro EP16 normal 512 runtime reached K3 and failed its host shape/workspace check after remote source pytest and build had passed. This is a shape-layout issue, not yet a kernel correctness/performance result.
- K1 graph reset metadata must use the same fixed compact capacity as the EP16/EP32 K1 launch path. Otherwise graph reset offsets can be computed from compact capacity while the actual K1 graph launch uses fixed capacity.

## 2026-07-01 - DeepSeek-V4-Pro Normal ASM Stride Finding

- After the separate `k3_out` workspace fix, Pro EP16 LL correctness passed, while Pro EP16 normal still failed numerically with valid route stats and unified weights. This separated the remaining issue from route metadata, scratch sizing, and Python weight-layout selection.
- K1 normal ASM still had a Flash-only symmetric-buffer offset for `x_sf`: `x + num_max_tokens * 4096`. Pro hidden is `7168`, so this pointed K1 at the wrong FP8 scale rows. The fix is to multiply by dynamic `hidden` (`sgprSizeL`) instead of literal `0x1000`.
- K3 normal ASM still had a Flash-only output scatter row stride: `row * (4096 * sizeof(bfloat16))`. Pro output rows are `7168` BF16 values, so K3 scatter wrote with the wrong row stride. The fix is to multiply by dynamic `hidden` (`sgprSizeI`) and then by `2`.
- These two ASM fixes are source-level complete and build successfully on 151.1. Runtime Pro EP16 normal validation is pending card availability because an active SGLang DeepSeek-V4-Pro server currently holds about `90-91%` VRAM on all 16 HCUs.

## 2026-07-01 - DeepSeek-V4-Pro Normal A-Weight Address Finding

- The `dynstride3` run proved the x-scale and K3 scatter fixes were not sufficient: Pro EP16 normal still mismatched while route stats remained correct.
- A remaining Flash-only constant was found in the shared normal ASM `GLOBAL_OFFSET_A` macro: `v_lshlrev ... 10` with comment `no256 * (4096 / 4)`.
- For pack5 A weights the stride must scale with `SizeI`, not with Flash's fixed 4096. For Pro this changes:
  - K1 from Flash L1 rows `4096 / 4 = 1024` to Pro L1 rows `6144 / 4 = 1536`.
  - K3 from Flash hidden `4096 / 4 = 1024` to Pro hidden `7168 / 4 = 1792`.
- The source fix uses `v_mul_lo_u32 tmp, s[sgprSizeI], offset0I` followed by `v_lshrrev_b32 tmp, 2, tmp` in all six normal ASM sources. This keeps Flash behavior identical while allowing Pro dimensions.
- Need remote rebuild and EP16 normal retest once the SGLang workload releases the cards.
- Flash performance guardrail: the shared dynamic ASM change is acceptable only if Flash normal performance remains within noise of the previous Flash path. If Flash shows a material regression after runtime A/B, split Pro into separate K1/K3 normal ASM code objects and route Pro shapes to those kernels, leaving Flash on the original constant-address fast path.
- Superseded on 2026-07-02 by the pack5 tile-index re-read below: the `offset0I * (SizeI / 4)` interpretation was wrong. The dynamic parts that should remain are `ko * hidden * 64`, global-read increments of `hidden * 128`, and row/tile base strides; the tile-local `offset0I` term is fixed `ni16 * (4 * 256)`.

## 2026-07-01 - DeepSeek-V4-Pro Normal dynoffsetA4 Retest Finding

- The dynoffsetA4 rebuild did not fix Pro EP16 normal correctness. Latest retest failed with `max_abs=0.1669921875`, `argmax=(181,6248)`, `fused=0.0`, `baseline=0.1669921875`, and `stats_ok=True`.
- The failed value is an exact zero on the fused side while the baseline row is nonzero. This shifts the next investigation away from pure FP8 numeric drift and toward a missing row/slot: K1 route metadata, `output_index`, `row_combine_ptrs`, or K3 normal combine/writeback for the failing token row.
- Because Pro EP16 LL already passes and the normal route stats are valid, the next evidence should isolate K1-normal metadata from K3-normal writeback before any further ASM edits. Prefer targeted diagnostics for the argmax token row over another broad source patch.

## 2026-07-01 - DeepSeek-V4-Pro Normal Combine-Slot Finding

- A targeted Pro EP16 normal diagnostic with `--debug-combine-on-fail` shows `fused == sum(combine_slots)` at the failing output element.
- Diagnostic point: rank `3`, row `225`, col `5845`, fused `-0.1259765625`, baseline `0.05615234375`.
- Local combine slots were `[0.0, 0.0, -0.1259765625, 0.0, 0.0, 0.0]` for top-k experts `[16,80,96,201,302,378]` owned by ranks `[0,3,4,8,12,15]`.
- This rules out `reduce_local_combine` as the primary remaining bug for Pro EP16 normal. It reads the combine buffer consistently.
- The remaining root-cause search should focus on K3 normal producing/writing only a subset of expected top-k slots, or on the row pointer / `output_index` metadata that K3 consumes for those slots.

## 2026-07-01 - DeepSeek-V4-Pro Normal Staged-Loop And Route-Metadata Finding

- K3 normal staged combine writeback had a hidden-size dependent loop bound in `K3_STORE_STAGED_HALF`. Flash hidden `4096` made this accidentally equal to the tile-local half bound, while Pro hidden `7168` made the loop walk beyond the 128 routed rows in the half tile.
- The staged-loop fix changes that bound back to fixed `4096`, matching the tile-row iteration used by the no-tail and tail-reduce staged stores. This is correctness-relevant and should not materially affect Flash because Flash already used the same effective value.
- After rebuilding and copying the generated `.co` files into the runtime source tree, Pro EP16 normal still failed. The remaining failure had nonzero combine slots whose sum matched the fused value, so local reduce remains ruled out.
- Route-metadata debug for source rank `11`, token `213` showed all six top-k routes had nonzero `row_combine_ptr` entries and valid `output_index`/`m_indices`/`route_weight` values before K3. That evidence makes missing K1 route metadata unlikely for at least the previously missing-slot case.
- Next evidence should compare the actual K1 normal activation row (`act_fp8`/`act_scale`) and K3 normal per-slot result against a reference value, because the remaining bug is now more likely numeric production/addressing than combine reduction.

## 2026-07-01 - DeepSeek-V4-Pro Normal K3 Value Debug Finding

- Env-gated Pro EP16 diagnostics now print route metadata, a Python single-column K3 reference, and actual local combine slots before `reduce_local_combine` for a selected `(source_rank, token, col)`.
- `k3_out` is not a reliable observation point for the normal K3 combine ASM path: debug runs showed `k3_out` can remain zero while the local combine buffer has nonzero slots. The normal ASM path should be diagnosed through the combine buffer or through explicit K1/K3 inputs, not by reading `k3_out`.
- For one selected failing-style point (`source_rank=11`, `token=211`, `col=1270`), route metadata existed for all six slots and the local combine buffer before reduce held `[-0.0220947265625, 0.0, 0.003753662109375, 0.0, 0.0, 0.0]`. The Python single-column reconstruction from `act_fp8`, `act_scale`, `l2_weight`, and `l2_scale` predicted about `[-0.00419, 0, 0.00632, 0, 0, 0]`.
- Pro EP16 normal also fails with plain normal ASM pack5 layout (`MEGAMOE_DCU_PRO_WEIGHT_LAYOUT=normal`) in the same combine-before-reduce shape as unified. This rules out a unified-only pack5 layout bug.
- Pro EP16 LL passes against the `normal-contiguous` baseline, not only against the `ll-masked` baseline: `max_abs=0.000488281`, `mean_abs=1.21163e-05`. This makes the normal baseline oracle credible for Pro and shifts the split to normal K1/K2/K3 internals.
- Same-input normal-versus-LL debug for `(source_rank=5, token=184, col=4778)` shows the LL K3 C++ path is internally consistent: slot3 on owner rank10 had Python single-column ref `0.0318703` and `k3_out=0.0319824`, and the full LL output passed the normal-contiguous baseline.
- The same normal route has valid metadata, but slot3 on rank10 had Python single-column ref `0.0051409` while the actual local combine slot for source rank5/token184/col4778 was `-0.1318359`. Therefore normal K3 ASM is definitely producing or writing a value inconsistent with its own visible `act_fp8`, `act_scale`, and L2 packed weight inputs. K1/K2 may still differ from LL, but K3 normal is the first confirmed faulty boundary.
- Next fix target: normal K3 ASM Pro addressing/writeback, especially the pack5 A-weight/scale address path and staged combine store path for hidden `7168`.

## 2026-07-02 - DeepSeek-V4-Pro Normal Pack5 `GLOBAL_OFFSET_A` Correction

- Re-reading the pack5 flat layout changed the root-cause hypothesis for the shared normal ASM address macro.
- Pack5 A is indexed as `((((ko * (N / 256) + no) * 16 + ni16) * 4 + ks) * 256 + ni * 16 + ki)`.
- In `GLOBAL_OFFSET_A`, `offset0I` is the tile-local `ni16` component, not the hidden-tile `no` component. Its element stride is therefore fixed at `4 * 256 = 1024` for both Flash and Pro.
- The hidden-dependent terms are already handled elsewhere: SRD/base offset covers `no * 256 * 64`, the pack5 `ko` base uses `ko * hidden * 64`, and K-loop increments use `hidden * 128`.
- Current source fix restores `GLOBAL_OFFSET_A` in the two K1 normal ASM sources and four K3 normal ASM sources to `v_lshlrev ... 10`, with the corrected comment `pack5 ni16 * (4 * 256)`.
- Remote Pro EP16 normal retest after rebuilding and copying fresh `.co` files still fails with the same combine-buffer boundary. Therefore the fixed `GLOBAL_OFFSET_A` correction is source-correct but not sufficient; the remaining fault is elsewhere in normal K3 ASM address/scale/store behavior.

## 2026-07-02 - DeepSeek-V4-Pro Normal Boundary Reclassification

- Plain normal layout also fails after the fixed-ni16 rebuild, so the remaining issue is not default unified pack5 layout alone.
- A targeted plain-normal debug point (`source_rank=1`, `token=384`, `col=2265`) showed K3 normal combine matches a Python single-column reconstruction from the normal path's own `act_fp8`, `act_scale`, and L2 packed weight: Python ref `-0.0926068`, combine slot `-0.0927734`.
- The same debug point showed suspicious K2 outputs for other valid slots: several routed rows had identical near-zero `act_scale=2.2321428616578487e-07`. That shifts the current first-suspect boundary back from K3-only to K1/K2 output production or K2 input rows for Pro normal.
- Next diagnostic should compare normal K1/K2 row values against a Python reconstruction for the selected `(source_rank, token, topk_slot)` before another ASM patch.
- Follow-up K1 debug on `source_rank=6`, `token=81` confirmed K1 output rows are missing before K2 for valid routes. For the same token, valid slots 0/2/4 had `l1_out_absmax=0.0`, while slots 1/3/5 had normal nonzero L1 output. The near-zero K2 scales are therefore a symptom of zero K1 rows, not a K2 quantization or K3 combine bug.
- Current target moves to K1 normal compact route/active-tile compute coverage for Pro EP16.

## 2026-07-02 - DeepSeek-V4-Pro K1 Compact Active-Tile Offset Finding

- K1 normal EP16/EP32 is forced through HIP compact prebuild (`force_compact_prebuild || num_ranks > 8`), where the route scratch header is dynamic: `counts[local_experts]`, `tile_bases[local_experts]`, `active_tiles`, then `tile_experts`.
- The HIP compact builder writes `active_tiles` at `route_scratch_i32[2 * local_experts]`. For Pro EP16 this is offset `48`; for Pro EP32 it is `24`; for Pro EP8 it would be `96`.
- The K1 normal ASM prebuilt path still read `route_scratch_i32[64]`, which only matches Flash/EP8 local_experts=32. For Pro EP16 it reads inside `tile_experts` rather than the active tile count, so valid compact row tiles can be skipped before K1 GEMM.
- The same prebuilt ASM path also clamped loaded `m_indices` with a hard-coded local expert limit of `32`, which would break Pro EP8 local_experts=48 even though EP16 is below that limit.
- Current source fix packs `active_tiles_offset` and `local_experts` into `GpuProb.reserved_c4`; the two K1 normal ASM sources read the dynamic offset and clamp against dynamic `local_experts`.

## 2026-07-02 - DeepSeek-V4-Pro K1 Staged-X Row Stride Finding

- After the compact active-tile offset patch, Pro EP16 normal still failed. Targeted K1 debug showed the active-offset issue was not the only problem: for a selected source token only the row tile 0 route had normal K1 output, while valid higher row tiles still had `l1_out_absmax=0.0`.
- Re-reading the K1 dispatch-pull staging block found another Flash-only constant: staged source rows were stored with `global_staged_row << 12`, i.e. `row * 4096` bytes.
- The same Pro path already loops over `256 * hidden` bytes and the downstream GEMM reads `staged_x` with dynamic `strideB1=hidden`; therefore Pro writes and reads different row strides unless the store uses `row * hidden`.
- Current source fix keeps the Flash `hidden==4096` shift path and adds a Pro/dynamic branch that computes `global_staged_row * hidden` before adding the vector byte offset.

## 2026-07-02 - DeepSeek-V4-Pro K1/K2 Content Mismatch Finding

- After fixing both the compact active-tile offset and the staged-x row-store stride, targeted normal debug showed all six selected routes can have nonzero K1 rows and internally consistent K3 combine values.
- Same-input LL debug with the normal-contiguous baseline still passes, but its per-slot K3 references for the same `(source_rank, token, col)` differ sharply from the normal path. For the latest selected point, normal K3 matches its own visible K2 inputs, while LL matches the baseline-side expected result.
- This rules out the baseline oracle, local reduce, K3 combine/writeback, missing compact route tiles, and missing staged rows for the latest selected point.
- The next root-cause search should compare normal K1/K2 produced rows against LL or a Python K1/K2 reconstruction for the same routed rows, then patch the first K1/K2 address/scale/layout mismatch found.
- Follow-up K1/K2 samples on route `(source_rank=5, token=189, col=3478)` show normal slot1 and slot2 on rank5 have different route rows and different local experts (`m_index=5` at row1280, `m_index=13` at row3329), but identical K1 samples and identical K2 raw FP8 samples. Only K2 scale changes with route weight.
- Additional tile-header debug printed `tile_base` and `tile_m_index`; both slots matched their routed row experts (`tile_m_index=5` for row1280 and `tile_m_index=13` for row3329). This rules out the earlier hypothesis that compact tile first-row metadata is selecting the wrong expert.
- The remaining likely root cause is K1 normal ASM expert weight addressing or a clobbered `sgprScaleFlag` / `sgprStrideAK` after the prebuilt route branch. The next source inspection should focus on writes to scalar registers `s96`/`sgprScaleFlag` and `s37`/`sgprStrideAK` before A SRD setup.

## 2026-07-02 - DeepSeek-V4-Pro K1 Python Reconstruction Finding

- Added a normal-only K1 debug reconstruction from the exact staged input view, row scale, `l1_weight`, and `l1_scale` tensors used by the normal path.
- Latest Pro EP16 default-normal diagnostic log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1ref1/normal_ep16.log`.
- On rank5 route `(source_rank=5, token=189, col=3478)`, slot1 used row `1281`, local expert `5`, and slot2 used row `3331`, local expert `13`. Both had valid compact tile metadata.
- Normal K1 ASM printed identical first four `l1_out` samples for slot1 and slot2: `[-0.0756836, -0.0527344, -0.171875, 0.300781]`.
- Python K1 references from the normal path's own visible inputs were distinct:
  - slot1: `[-0.134254, 0.194541, -0.0019329, -0.0420067]`;
  - slot2: `[-0.196096, 0.0715203, 0.0490447, 0.397356]`.
- This proves the current first bad boundary is normal K1 ASM production itself. It rules out the normal-contiguous baseline, compact active-tile count, compact tile expert header, staged-x source tensor construction, K2 quantization, K3 combine/writeback, and local reduce for this route.
- Next source target: K1 normal ASM A-weight and scale addressing after the prebuilt route branch, especially whether the routed expert id is applied before SRDs/global read addresses are finalized.

## 2026-07-02 - DeepSeek-V4-Pro K1 Runtime Code-Object Finding

- `dccobjdump --show-sass` on the standalone K1 `.co` files initially looked empty because DTK writes ISA side-effect files in the current directory; the generated remote ISA files are `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_{PACK5,UNIFIED_PACK5}.co*.ISA`.
- The runtime K1 normal code objects are not stale: ISA contains the dynamic `reserved_c4` load at `GpuProb+0xc4`, low16 active-tile offset, high16 local-experts clamp, `v_readfirstlane_b32 s96` from the tile expert table, and A SRD expert stride math `s[sgprStrideAK] * s96`.
- This shifts the next discriminator away from rebuild freshness and toward whether K1 ASM is reading the wrong expert weight/scale despite the visible route chain, or whether the diagnostic samples match a different expert than the routed one.

## 2026-07-02 - DeepSeek-V4-Pro K1 Expert0 Discriminator Finding

- Added a debug-only all-local-expert K1 Python discriminator. For each K1 sample column it computes Python K1 references for every local expert from the same staged input row, then reports the closest experts to the observed ASM `l1_out`.
- Pro EP16 default-normal log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1match2/normal_ep16.log`.
- Result: every valid printed route on ranks 0/2/5/6/9 matched local expert `0` for almost all or all sampled columns, even when routed `m_index` was 5, 8, 11, 13, 14, or 15. Examples:
  - rank5 slot1 row1281 routed expert5: best expert0 for 27/27 samples;
  - rank5 slot2 row3330 routed expert13: best expert0 for 27/27 samples;
  - rank2 slot0 row2052 routed expert8: best expert0 for 27/27 samples.
- The remaining K1 normal correctness bug is therefore not numerical drift: prebuilt compact K1 ASM is effectively using expert0 A weights/scales.
- Patch direction: in the compact-prebuilt branch, load `sgprScaleFlag` directly from the HIP-built `route_scratch_i32[tile_experts_offset + compact_tile]` side-channel instead of indirectly reading `m_indices[compact_tile * 256]`.

## 2026-07-02 - DeepSeek-V4-Pro Post Tile-Expert Patch Finding

- The compact-prebuilt K1 ASM patch was rebuilt and verified in runtime `.co`, then tested on Pro EP16 normal default layout.
- Pro EP16 normal still fails after the patch: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1tileexpert1/normal_ep16_after_tileexpert.log`.
- The new failure point differs from the earlier missing-row/expert0 evidence: rank `12`, row `497`, col `1907` has all six combine slots present and nonzero, and `fused` equals the slot sum within BF16 rounding.
- This keeps the first visible boundary before or inside K1/K2 numeric production, but the current failure is no longer explained solely by missing compact tiles, local reduce, K3 combine scatter, or a single zero slot.
- Next evidence should use `MEGAMOE_DCU_DEBUG_ROUTE=12,497,1907` to compare routed K1 ASM samples, all-expert Python K1 discriminator output, K2 FP8/scale samples, and K3 single-column reconstruction against the normal-contiguous baseline route.

## 2026-07-02 - DeepSeek-V4-Pro K1 Expert-Stride Fork Finding

- Post-patch targeted debug still shows K1 ASM output matching local expert0 for routed experts across different ranks and local expert ids.
- Runtime K1 `.co` ISA confirms the compact-prebuilt branch now executes the intended side-channel read:
  `active_tiles_offset + 1 + compact_tile -> v_readfirstlane_b32 s96`.
- The compact metadata itself is credible because debug `tile_m_index` values match routed `m_index`, and K3 combines correctly from whatever K2 input it receives.
- Plain pack5 and unified pack5 both fail after the tile-expert patch, so this is not isolated to the unified transposed layout.
- The next fork is precise: if a forced `sgprScaleFlag=1` ASM experiment changes K1 discriminator matches from expert0 to expert1, the metadata load value is the fault; if it still matches expert0, the K1 A-weight/scale expert stride or SRD setup is ignoring `sgprScaleFlag`.

## 2026-07-02 - DeepSeek-V4-Pro K1 Metadata-Load Finding

- The forced `sgprScaleFlag=1` discriminator resolved the fork: K1 rows changed to match local expert1 for nearly every sampled column on all valid routes. Therefore the A-weight/scale expert stride and SRD math can use a nonzero expert id.
- A follow-up forced run printed Python-visible compact route headers. For the selected Pro EP16 route, every valid K1 row had `compact_tile_expert_header == tile_id == m_index`, and `compact_active_tiles=24`.
- This rules out the HIP compact builder producing bad `tile_experts` data and rules out expert-stride math being permanently pinned to expert0.
- Remaining root-cause candidate: the normal non-forced K1 ASM metadata load from `route_scratch` behaves like it reads stale or zero data at use time, despite the correct header being visible later from Python. The next bounded patch should target metadata load coherency/order (`buffer_wbinvl1` before reading compact metadata and `glc` on the `buffer_load_dword` route-scratch metadata loads), then verify with ISA and the K1 all-expert discriminator before broad correctness.

## 2026-07-02 - DeepSeek-V4-Pro Stale Runtime Extension Finding

- The metadata-load coherency hypothesis was rejected: adding `buffer_wbinvl1` and `glc` to the K1 compact metadata MUBUF loads did not stop the K1 discriminator from matching local expert0.
- The actual fault was build-artifact freshness. `setup.py bdist_wheel` produced a new K1 host extension under `build/lib...`, but the tests were run with `PYTHONPATH=.`, so Python imported the older source-tree `megamoe/dcu_megamoe_opt/K1_fused/k1_fused_ext*.so`.
- The old launcher did not populate the new `GpuProb.reserved_c4` compact metadata field. The patched `.co` therefore read zero for the active-tile offset/local-expert metadata pack, which made the compact metadata path fall back to expert0 behavior even though the `.co` and Python-visible route scratch data were correct.
- After copying fresh built `.so` files into the source runtime tree, the same Pro EP16 normal targeted run passed against the `normal-contiguous` baseline and the K1 discriminator matched the routed local experts.
- Follow-up action: remove the temporary K1 `buffer_wbinvl1/glc` experiment and retest Pro EP16 normal with fresh `.co` plus fresh `.so`. Keeping that experiment would be unnecessary risk for the shared Flash normal path unless the no-experiment retest proves it is needed.

## 2026-07-02 - DeepSeek-V4-Pro No-Experiment Correctness Finding

- The follow-up no-experiment retest passed. Removing the temporary K1 compact metadata `buffer_wbinvl1/glc` changes did not break Pro EP16 normal correctness once the fresh runtime `.so` was in place.
- Current correctness-critical pieces are the Pro shape/scratch gates, K3 separate output workspace, dynamic K1 compact metadata offsets in `reserved_c4`, Pro staged-x dynamic row stride, and fresh runtime extension propagation.
- The current evidence does not justify splitting a Pro-only K1 normal kernel for the metadata-load path. Flash still needs an explicit correctness/performance guardrail because earlier Pro work made other shared normal ASM address paths dynamic.

## 2026-07-02 - Flash Guardrail Finding After Pro Changes

- Flash EP8 normal eager 4096 with RPC peer memory still passes correctness after the Pro changes and the removal of the metadata `glc/wbinvl1` experiment.
- Current MegaMoE timing is `5.784 ms`, compared with the earlier 151.1 reference `5.764 ms`; the delta is small enough to treat as noise for this smoke-level guardrail.
- Therefore there is no current Flash-performance reason to split a Pro-only K1/K3 normal ASM code object. Keep this as a future contingency only if broader Flash buckets show a material regression.

## 2026-07-02 - DeepSeek-V4-Pro Initial Performance Finding

- Pro EP16 normal 512 is correct and faster than the normal-contiguous baseline in the current 151.1 run: MegaMoE `3.206 ms`, baseline `4.258 ms`, speedup `1.33x`.
- Pro EP16 normal 1024 is also correct and faster than baseline: MegaMoE `5.068 ms`, baseline `7.144 ms`, speedup `1.41x`.
- The 2048 medium-token guardrail is not yet clean: it fails with a nonzero fused/baseline mismatch at rank `1`, row `1672`, col `5051` while route stats are OK.
- Targeted 2048 debug shows local reduce is not the issue, and the debugged route is internally consistent through K1, K2, K3, and combine. Disabling K2 active-tile skip does not fix 2048.
- This means the Pro EP16 512/1024 fix is real but not sufficient for medium tokens. Continue systematic debug by comparing LL 2048 against the same normal-contiguous baseline before treating Pro EP16 normal as fully correct.

## 2026-07-02 - DeepSeek-V4-Pro 2048 Normal Publish Hypothesis

- Pro EP16 LL 2048 passes against the same `normal-contiguous` baseline used by normal, so the 2048 baseline/test construction is usable and the remaining failure is isolated to the normal backend.
- Targeted normal 2048 debug can make the inspected route locally correct while the max-diff moves to another token/column. The debug helper copies source-rank tensors to CPU, which likely adds a synchronization/visibility side effect.
- The current working hypothesis is that K3 normal direct-combine stores need an explicit publish point before the post-K3 rank barrier and local reduce consume the combine buffer.
- A bounded K3 ASM patch now adds `s_waitcnt vmcnt(0)` and `buffer_wbinvl1_vol` just before the final `s_endpgm` in both direct-combine code objects. Remote build and `dccobjdump` ISA checks confirmed both runtime K3 code objects end with `s_waitcnt vmcnt(0)`, `buffer_wbinvl1_vol`, `s_endpgm`.
- This patch touches shared Flash normal K3 ASM, so if it fixes Pro 2048 correctness the Flash EP8 4096 guardrail must be rerun before accepting the change.
- Result update: Pro EP16 normal 2048 still fails after the K3 publish patch (`process/rank=3`, row `1054`, col `958`, `max_abs=0.142578125`, fused `-0.1240234375`, baseline `0.0185546875`). Therefore K3 final store publish alone is not the root cause. Keep the patch under scrutiny because it changes shared Flash K3 ASM without yet providing correctness value.
- Stage-sync update: `HIP_LAUNCH_BLOCKING=1`, explicit `torch.cuda.synchronize()` after K1, after K2, after K3, and after all three stages all still fail Pro EP16 normal 2048. Plain synchronization is not sufficient.
- Targeted route `6,142,3248` is more diagnostic: no-debug combine slot0 was `+0.036376953125`, but with route debug the same slot's K3 Python ref/combine value becomes about `-0.08984`, matching the expected output direction. K1 for that route matches routed experts under debug.
- K1 main ASM l1_out store publication was tested next because route debug reads `l1_out` after K1 and can fix the selected route, while plain synchronization does not. The bounded K1 patch added `s_waitcnt vmcnt(0)` plus `buffer_wbinvl1_vol` before the two main K1 `s_endpgm` exits in both PACK5 and UNIFIED code objects.
- Result update: Pro EP16 normal 2048 still fails after the K1 publish patch (`process/rank=7`, row `1030`, col `6286`, `max_abs=0.1193695068359375`, fused `0.1162109375`, baseline `-0.0031585693359375`). Therefore K1 final-store publish is not sufficient either.
- The current working set of ruled-out sole causes is: baseline/test construction, local reduce, K2 active-tile skip, K3 final publish, K1 final publish, and plain stream/device synchronization. Next evidence should come from targeted route `(7,1030,6286)` with more selective intermediate readback controls rather than adding more global fences.

## 2026-07-02 - DeepSeek-V4-Pro Selective K1 Readback Finding

- Added `MEGAMOE_DCU_DEBUG_ROUTE_STAGE` so targeted route debug can read only selected stages instead of always materializing K1/K2/K3/meta/combine.
- For original failing route `(source_rank=7, token=1030, col=6286)`, `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=combine` still left the route wrong: combine slot sum `0.04688525199890137`, with slot0 `0.037353515625` while the earlier no-debug baseline value for the route was `-0.0031585693359375`. The run's max diff moved elsewhere, but the original route was not fixed by combine readback alone.
- For the same route, `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k1,combine` made the original route correct within BF16 rounding: combine slot sum `-0.003143310546875` versus earlier baseline `-0.0031585693359375`. Slot0 changed from wrong-positive to correct-negative (`-0.021240234375`) while the other slots mostly stayed stable.
- A K2-side `MEGAMOE_DCU_K2_GLC_SLC_LOAD=1` experiment rebuilt K2 with `global_load_ushort ... glc slc` for BF16 `l1_out` input reads, and ISA confirmed the load form, but Pro EP16 normal 2048 still failed. Therefore making K2 input loads cache-bypassing/fresh is not sufficient.
- Current interpretation: K1 CPU readback has a route-local side effect that fixes the downstream K2/K3/combine result for that route. The next minimal experiment should target K1 output store visibility or store-side cache behavior, not K3 combine, local reduce, or the baseline oracle.
- Follow-up K1 store-side experiment changed all K1 BF16 `l1_out` `buffer_store_short ... // store D` sites in the PACK5 and UNIFIED K1 code objects to `buffer_store_short ... glc`. Remote source pytest passed and `dccobjdump` confirmed 640 `buffer_store_short ... glc` instructions in each runtime K1 `.co`.
- Result update: Pro EP16 normal 2048 still fails with K1 store-side `glc` alone (`process/rank=14`, row `1043`, col `5051`, `max_abs=0.10791015625`, fused `0.0888671875`, baseline `-0.01904296875`). Therefore K1 BF16 store `glc` alone is not the missing correctness condition.
- Result update: Pro EP16 normal 2048 also fails with K1 store-side `glc` plus env-gated K2 `global_load_ushort ... glc slc` input loads (`process/rank=4`, row `1637`, col `3261`, `max_abs=0.115234375`, fused `-0.0927734375`, baseline `0.0224609375`). Therefore the simple two-sided coherent store/load hypothesis is not sufficient.
- Cleanup direction: remove the unproven K1/K3 tail publish and K1 store-side `glc` experiments before the next correctness loop. They touch shared Flash normal ASM and currently provide only negative evidence.

## 2026-07-02 - DeepSeek-V4-Pro Clean Hot-Path Cleanup Finding

- The negative K1 final publish, K3 direct-combine publish, K1 store-side `glc`, and K2 env-gated `glc/slc` input-load experiments have been removed from local source before continuing diagnosis.
- Local static checks now show no remaining experiment markers for those paths, and Python compile plus whitespace checks pass.
- The next correctness evidence must come from a rebuilt clean remote runtime, because previous remote code objects and source-tree `.so` files still reflect the last experiment until refreshed.

## 2026-07-02 - DeepSeek-V4-Pro Clean 2048 Failure Finding

- After syncing, rebuilding, copying fresh `.co`/`.so`, and verifying the negative experiment artifacts are absent, Pro EP16 normal 2048 still fails.
- The current clean first failure is `(source_rank=10, token=559, col=3146)`, with fused `0.078125` versus baseline `-0.04736328125`.
- Slot3 is the large wrong contribution (`0.0771484375`) for expert `91` owned by rank `3`. This is the next targeted route for selective readback and same-input intermediate comparison.

## 2026-07-02 - DeepSeek-V4-Pro Combine Read-Side Finding

- For clean route `(10,559,3146)`, `combine`-only CPU readback changes the observed combine slot3 from the no-debug wrong `+0.0771484375` to `-0.0517578125`, and the route sum moves from `+0.078125` to about `-0.0353`.
- `k1,combine` moves the same route sum to about `-0.0430`, close to the no-debug baseline `-0.04736`, while the run still fails at a different route.
- This differs from the earlier route `(7,1030,6286)`, where `combine` alone was insufficient and `k1,combine` was decisive. The clean evidence suggests there may be more than one stale-consumer boundary: reduce consuming combine, and in some cases K2 consuming K1.
- Next highest-value probe is read-side coherence in `reduce_local_combine`, because it directly consumes the combine buffer after the rank barrier and before the final `y`.

## 2026-07-02 - DeepSeek-V4-Pro Reduce Read-Side Negative Finding

- The env-gated `reduce_local_combine` probe using `global_load_dwordx4 ... glc slc` for combine reads does not fix Pro EP16 normal 2048.
- The new failure `(source_rank=5, token=610, col=1383)` has combine slot values that already sum to the wrong fused value, so final reduce's ordinary vector load is not the sole cause.
- Next comparison should move one stage earlier: inspect whether K3 matches a Python reconstruction from visible K2 outputs, and whether those K2 outputs differ from the LL/baseline path for the same route.

## 2026-07-02 - DeepSeek-V4-Pro K2/K3 Visible-Input Finding

- For route `(5,610,1383)`, `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k2,k3,combine` shows K3 Python reconstruction from visible K2 `act_fp8/act_scale` and L2 packed weights matches the combine slots.
- The dominant slot4 ref is `0.11482040584087372`, and the combine slot is `0.11474609375`; the earlier no-debug/reduce-glc failure had slot4 `-0.007293701171875`.
- This rules out a pure K3 formula/addressing bug for that route under visible inputs. The remaining question is which readback boundary changes the state: K2 output readback, K3 debug readback, or combine readback.

## 2026-07-02 - DeepSeek-V4-Pro Stage Matrix Timing Finding

- Stage matrix for route `(5,610,1383)` separates the readback side effects:
  - `combine` readback alone partially improves slot4 from the no-debug/reduce-probe value `-0.007293701171875` to `0.0211181640625`, but the route sum is still far from the baseline.
  - `k3,combine` readback makes slot4 match the Python K3 reconstruction (`0.11482040584087372` ref, `0.11474609375` combine slot) and moves the route sum near the baseline.
  - `k2,combine` readback also makes slot4 correct (`0.11474609375`).
- Since `k3,combine` fixes the dominant slot while the K3 debug work is a readback/reconstruction after the K3 launch and before the normal post-K3 rank barrier in `opt.py`, the next hypothesis is a timing or memory-visibility boundary around K3 completion and cross-rank combine-buffer consumption, not a K3 arithmetic or L2 weight-address bug.
- A plain `torch.cuda.synchronize()` after K3 was already negative, so the next minimal probe should distinguish "extra host delay before the post-K3 barrier" from "missing system-scope publish/fence for K3 combine stores".

## 2026-07-02 - DeepSeek-V4-Pro K3 Timing/Store Negative Finding

- A controlled host delay after K3 and before the post-K3 rank barrier does not fix Pro EP16 normal 2048. Both `MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3_MS=10` and the same delay with an explicit `torch.cuda.synchronize()` still fail.
- Forcing the existing K3 tail-reduce path with `K3_USE_ASM_TAIL_REDUCE=1` also fails, so the issue is not solved by switching to the tail-reduce protocol.
- A direct-K3 ASM probe that adds `glc` to remote combine `global_store_short` and `global_store_dwordx4` sites was ISA-verified but still fails Pro EP16 normal 2048.
- Current ruled-out sole causes now include: baseline construction, LL oracle, local reduce ordinary loads, pure K3 arithmetic/addressing under visible inputs, plain synchronize, host delay, tail-reduce selection, K1/K3 final publish, K1 store-side `glc`, K2 input-load `glc/slc`, reduce read-side `glc/slc`, and K3 store-side `glc`.
- The next diagnostic should follow the user's same-input suggestion: compare normal and LL/baseline stage values for the same generated tensors and routed slots, so the next patch targets the first actually divergent producer/consumer boundary instead of adding global cache modifiers.

## 2026-07-02 - DeepSeek-V4-Pro Same-Input K1 Source-X Finding

- Added a same-input failure diagnostic that gathers the failing route across ranks, then reruns normal and LL on the same generated tensors with route-stage debug enabled. LL matches the `normal-contiguous` baseline exactly on the selected routes, while normal remains wrong.
- The decisive boundary is before K2/K3: normal K3 combine matches its own visible K2 output and L2 weights, but normal K2 output differs from LL for the dominant slots.
- A debug-only source-x broadcast comparison shows normal K1 staged input rows can have wrong FP8 `x` bytes while the copied `x_scale` is correct. Example route `(source_rank=1, token=1819, col=6306)`:
  - normal final value `-0.01019287109375`, baseline/LL `-0.076171875`;
  - slot1 and slot3 on owner rank7 had `x_raw_sample_mismatch_count=13` and `x_dequant_sample_max_abs_delta=0.201171875`;
  - slot0 on owner rank4 had `x_raw_sample_mismatch_count=12`;
  - unaffected slots showed `x_raw_sample_mismatch_count=0`;
  - normal K1 ASM still matched a Python K1 reconstruction from its own staged input, proving the wrong value enters through normal K1 source-x staging, not K1 math, K2, K3, combine, or final reduce.
- Next source target: normal K1 input staging/remote source-x read path, especially Pro hidden-stride/chunk loop bounds and any Flash-only 4096-byte/4096-column assumptions that can leave the low column chunks stale while later sampled chunks match.

## 2026-07-02 - DeepSeek-V4-Pro K1 Rank-Local Source-Rank Finding

- The Pro K1 staging path uses rank-local row pointers when the symmetric x address span exceeds 32 bits. `row_x_ptrs` packs the source rank in the high dword and the rank-local x offset in the low dword.
- The current ASM rank-local source load scalarizes that high dword with `v_readfirstlane_b32 s63, v254`, then loads one peer-base SRD for the whole wave.
- That is safe only when all active lanes in the wave belong to rows from the same source rank. Flash hidden=4096 has 128 x 32-byte chunks per row, which aligns with 64-lane wave boundaries. Pro hidden=7168 has 224 x 32-byte chunks per row, so every other row starts halfway through a wave; the first 32 chunks of that row can share a wave with the previous row.
- If the previous row came from a different source rank, those low-column chunks use the wrong peer base while still using the current row's rank-local offset. This exactly matches the source-x diagnostic: low-column FP8 bytes can be wrong, `x_scale` remains correct, and unaffected slots have zero raw-byte mismatches.
- Local patch direction: in both normal K1 ASM code objects, make the rank-local source-load branch iterate over the active lanes by `v254` source rank, loading each peer-base SRD under a matching exec mask, then restore the original valid-lane mask before the existing zero-fill/store path.

## 2026-07-02 - DeepSeek-V4-Pro K1 Source-X Patch Result

- After the rank-local source-rank grouping patch and a fresh remote K1 rebuild, the targeted source-x diagnostic no longer reports raw FP8 input mismatches for route `(source_rank=1, token=1819, col=6306)`.
- The remaining Pro EP16 normal 2048 failure is therefore no longer the original source-x staging bug. The next observed boundary is K1 output production: K1 rows can have correct staged input but still diverge from Python K1 references, especially in low L1 output columns.
- The new rank-local loop's use of `s80:s81` is probably not the cause by itself: the surrounding K1 prologue already uses `s80:s81` repeatedly for `saveexec` before those SGPRs are initialized later as `sgprLocalWriteAddrA_ori` and `sgprLoopforPf`.
- Next root-cause candidates are rank-local loop mask/value preservation, K1 packed A-weight addressing, and K1 scale addressing for Pro's `hidden=7168`, `l1_cols=6144` shape.

## 2026-07-02 - DeepSeek-V4-Pro Ordered Same-Input Stage Compare Rule

- Future correctness diagnosis must be gated in order: K1 first, then K2, then K3/combine, then final reduce. Do not spend time on K2/K3 if the same-input K1 output differs.
- The expected invariant is logical equality, not necessarily identical physical row placement. Normal and LL may deterministically arrange scratch rows differently, so compare physical row ids first and only rely on direct row equality if verified.
- If physical rows differ, align by logical route key `(source_rank, token, topk slot, routed expert)` and compare the corresponding K1 output rows.
- K1 pass criteria: normal and LL K1 outputs for each selected routed slot agree within the accepted BF16 tolerance on the full L1 row, with source-x bytes, x scale, routed expert, L1 weight, and L1 scale available for the first failing column if they do not.

## 2026-07-02 - K1-Only Compare Instrumentation Finding

- The previous failure compare mixed K1/K2/K3/combine debug in one rerun, which made it too easy to reason past the first failing producer.
- The diagnostic path now has a K1-only mode: on failure, normal and LL rerun on the same input, capture full K1 rows for the selected route, stop before K2, and print a per-slot normal-vs-LL K1 row comparison.
- This is only instrumentation. It does not prove or fix correctness until the remote Pro EP16 2048 run produces `MEGAMOE_DCU_DEBUG_K1_COMPARE` evidence.

## 2026-07-02 - DeepSeek-V4-Pro K1 Low-Tile Staging Finding

- K1-only same-input comparison proves normal K1 output diverges from LL/Python before K2. This follows the requested stage order and keeps K2/K3 out of the current fix loop.
- Full-row K1 refs show source-x is now correct (`x_raw_full_mismatch_count=0`) while normal K1 still differs from Python/LL in N tiles `0..7`, i.e. columns `< 2048`.
- Those are exactly the K1 CTAs that participate in dispatch-pull staging (`wg0 < s58`, with `s58=8` for the Pro run). CTAs for later N tiles skip staging and match the Python reference.
- Targeted ASM comparison found a staging-register lifetime bug: `s62` is initialized as the staged_x plane offset `0`, but the rank-local staging loop later reuses it for source-rank pointer slots. The B-address restore path then adds stale `s62` to `sgprAddressB`, shifting staged_x reads only for staging-participant CTAs.
- Local fix: restore `s62` to `0` before rebuilding B global offsets in both K1 `PACK5` and `UNIFIED_PACK5` ASM sources. This is a narrow K1 fix and should not require touching K2/K3.
- Remote verification on 151.1 confirms the `s62` restore fixes the active Pro EP16 2048 correctness failure: the same bucket passes end-to-end with `max_abs=0.000976562`. Because the end-to-end bucket is now correct, there is no remaining K2/K3 boundary to split for this failure; ordered K2/K3 comparison remains the rule for any later failing bucket.
- No-debug Pro EP16 2048 performance after the fix is healthy: MegaMoE `8.0324 ms` versus normal-contiguous baseline `13.2038 ms`, speedup `1.64x`. The next risk check is Flash EP8 4096 because the K1 ASM source is shared; expected impact is small because the fix adds one scalar move on the existing staging address-restore path.
- Flash EP8 4096 guardrail after the `s62` restore passed with MegaMoE `5.8028 ms`, compared with the prior same-node reference `5.784 ms`. This is a noise-level delta, so there is no current evidence that the shared K1 fix materially hurts Flash performance. Keep the Pro-only kernel split only as a contingency for future broader Flash regressions.
- A broader Pro EP16 4096 guardrail also passed after the same fix: MegaMoE `15.7914 ms` versus normal-contiguous baseline `25.3273 ms`, speedup `1.60x`. This confirms the fix is not just a 2048-only artifact and keeps the root cause classified as K1 dispatch-pull staging register lifetime, not K2/K3 math or final reduce.

## 2026-07-02 - Temporary Debug Cleanup Finding

- The Python debug scaffolding used for ordered K1/K2/K3 isolation is no longer needed after the K1 `s62` fix and guardrail runs.
- Cleanup removed env-gated `MEGAMOE_DCU_DEBUG_*` route/stage hooks, same-input backend reruns, K1 snapshot caches, source-x/K2/K3/combine dumps, stop-after-K1 control flow, and debug-only CLI flags from the normal harness path.
- The retained failure report is the production-useful minimum: argmax row/col, fused value, baseline value, and stats state. Source-level tests now assert the retired debug hooks and CLI flags stay absent.
- This cleanup should not change fused kernel behavior: the functional K1 ASM fix, Pro shape support, scratch sizing, and correctness/performance paths remain in place.

## 2026-07-02 - Test-Only Pro Layout Override Cleanup Finding

- `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT` was a temporary test harness override for the earlier plain-vs-unified Pro diagnosis. It was not read by production `fp8_mega_moe`.
- With the K1 `s62` correctness fix verified, the Pro-specific override is no longer needed and would confuse future users into thinking there is a production Pro layout switch.
- The harness now follows the same generic rule for all shapes: explicit unified env selects unified, otherwise normal backend uses normal layout and LL uses unified layout.

## 2026-07-02 - DeepSeek-V4-Pro Normal Graph Tail-Reduce Finding

- Pro EP16 normal graph cap5376/replay512 fails only when forcing `K3_USE_ASM_TAIL_REDUCE=1`; the same minimized run passes with tail-reduce0. This isolates the current graph correctness issue to K3 normal ASM tail-reduce graph/reduce, not K1 route building, K2, plain K3 graph, or the `normal-contiguous` baseline.
- The tail-reduce graph runtime macro used `s_lshl_b32 s76, s76, 9`, which hard-coded Flash's BF16 vector count per token (`4096 / 8 = 512`). For Pro hidden=7168 the count must be `7168 / 8 = 896`.
- This explains failures such as row 330/col 2234: with the Flash vector limit, only the first `runtime_tokens * 512` vectors are reduced, so later linear positions in the 512-token Pro output can contain stale/partial values even if the column is below 4096.
- Patch direction: compute graph runtime reduce vectors dynamically as `runtime_tokens * (sgprSizeI >> 3)` in both K3 tail-reduce ASM variants. Flash remains equivalent because `4096 >> 3 = 512`.

## 2026-07-02 - Pod2 40-Card Environment Finding

- Pod2 is a new 40-card shared-storage supernode profile.
- Jump host:
  - `simsadmin@10.2.208.215`, SSH port `51730`.
  - Terminal key auth from this Windows workspace works with:
    `ssh -F NUL -p 51730 simsadmin@10.2.208.215 "hostname && whoami && date"`.
  - Verified hostname/user on 2026-07-02: `sims_508_tiaobj.localdomain`, `simsadmin`.
- Worker nodes:
  - `c0 172.20.2.131`
  - `c1 172.20.2.132`
  - `c2 172.20.2.133`
  - `c3 172.20.2.134`
  - `c4 172.20.2.135`
  - `c5 172.20.2.136`
  - `c6 172.20.2.137`
  - `c7 172.20.2.138`
  - `c8 172.20.2.139`
  - `c9 172.20.2.140`
- Worker user is `p_user`. The worker password was provided out-of-band in the chat, but must not be written into planning files, shell commands, command logs, or repository files.
- One-shot terminal equivalent of the MobaXterm-style jump path:
  `ssh -F NUL -J simsadmin@10.2.208.215:51730 p_user@172.20.2.131`.
- Current non-interactive result after key installation: ProxyJump BatchMode login works for `p_user@c0..c9`. An earlier `c8` failure was observed before the key path was fully verified; later `c8` checks returned `p2c8` / `p_user` and container status successfully.
- Docker/container:
  - Container name: `lj_sgl_0512`.
  - Enter container after worker login: `docker exec -it lj_sgl_0512 /bin/bash`.
  - Use non-interactive checks with: `docker exec lj_sgl_0512 bash -lc '<cmd>'`.
- Mounts and shared storage:
  - `/data_add/lizhg/lj` is mapped to `/data_add/lizhg/lj`.
  - `/data_add/DeepSeek-V4-Pro-FP8-Channel` is mapped to `/module`.
  - `/public/lijing` is mapped to `/public/lijing`.
  - User confirmed Pod2 compute nodes use shared storage. After login, verify this with the same path checksum or `df -h`/`mount` on two workers before relying on one-node build artifacts.
- First checks to run after compute auth is available:
  - Login sanity: `hostname && whoami && date`.
  - Docker sanity: `docker ps --filter name=lj_sgl_0512 --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'`.
  - Mount sanity: `docker inspect lj_sgl_0512 --format '{{json .Mounts}}'`.
  - Container toolchain: `docker exec lj_sgl_0512 bash -lc 'python3 -V; which hipcc || true; hy-smi || true'`.
  - 40-card state on all workers: `docker exec lj_sgl_0512 bash -lc 'hy-smi; hy-smi --showpids || true'`.
- 2026-07-02 startup/state check:
  - `docker start lj_sgl_0512` succeeded on `c0..c9`; every container reported `Up`.
  - Shared mounts were visible in the running container on all workers: `/module`, `/public/lijing`, and `/data_add/lizhg/lj`.
  - DCU devices are not currently exposed in the container: all workers report no `/dev/kfd` and no `/dev/dri/renderD*`.
  - `hy-smi` inside the container does not show usable HCU status. Most nodes print `Hfm Link Status: WAIT GLOBAL TOPO` plus `No hycu Driver loaded`; `c2` prints `Hfm Link Status: LINK FAILED`.
  - Treat Pod2 GPU execution as blocked on driver/device exposure or platform topology state, not on SSH/container login.

## 2026-07-03 - K3 Tail-Reduce Graph Hidden Regression Finding

- Flash EP8 normal graph cap8192 on 151.1 passed replay4096 (`5607.7 us`) but failed replay8192 correctness (`max_abs=5.5331878662109375`, graph row 7716 col 2644).
- The failure is consistent with the recent Pro EP16 normal graph fix. That fix changed the graph runtime reduce limit from Flash's `runtime_tokens << 9` to `runtime_tokens * (sgprSizeI >> 3)`.
- In the K3 tail-reduce extra-reducer path, `sgprSizeI` aliases `s20`, and the grouped-GEMM workgroup mapping reuses `s20` before `K3_TAIL_APPLY_GRAPH_RUNTIME_STATE` runs. The dynamic hidden calculation can therefore read a clobbered value in the late reducer path.
- Local fix direction: keep the Pro dynamic hidden behavior, but pass `hidden / 8` through the existing 256-byte K3 `GpuProb` storage at offset `0xd8` and have both tail-reduce ASM variants read that stable external-arg value. This avoids reintroducing the Flash-only constant and avoids changing SGPR allocation.
- Remote verification confirmed the fix: after rebuild/copying fresh artifacts on 151.1, Flash EP8 normal graph cap8192 passes replay4096/replay8192 with `max_abs=0.000488281` and medians `5636.4/11217.7 us`.

## 2026-07-03 - 151.1 LL Masked Baseline Environment Finding

- The README HCA/topology instructions are a reference for nodes that expose InfiniBand HCAs. They should not be applied blindly on 151.1.
- 151.1 currently exposes no entries under `/sys/class/infiniband` on the host or inside `sglang_megamoe`, so forcing old `mlx5_*` topology data would add an artificial failure mode.
- For 151.1, first try `ll-masked` with the actual node environment plus the generic DeepEP LL sizing/contexts. If DeepEP LL init fails, the next fallback should use the previously validated MNNVL compatibility variables (`ROCSHMEM_IPC_MNVL=1`, `ROCSHMEM_GDR_DISABLE_XDP=1`, appropriately large heap) rather than stale HCA names.

## 2026-07-03 - Flash EP16 Graph Regression Boundary

- Flash EP16 LL graph slowdown remains unresolved and must be treated as a first-priority performance regression candidate once 16 HCUs are healthy.
- The previous slow comparison was not fully fair because cap512 historical data was compared with a later cap768-style retry in at least one run. However, old LL replay should be weakly sensitive to capture capacity, so cap mismatch alone is not enough to dismiss the regression.
- Fair retest requirement: same current worktree command as history with cap512, replay `8,32,128,256,512`, `--baseline-kind ll-masked`, and actual 151.1 environment. If still slow, do current-vs-previous-commit A/B before patching.
- Pro EP16 graph hidden-vec fix is not yet Pro-runtime verified. It solved the Flash EP8 normal graph replay8192 regression after moving `hidden/8` into stable `GpuProb` storage, but a healthy 16-card Pro EP16 graph run is still required.
- Pro EP8 broad benchmarking should not proceed until its 512 normal path is isolated. The first valid question is whether the Pro EP8 512 hang is caused by RPC/fabric, IPC, K3 tail-reduce1, or a Pro EP8 local_experts=48 path.

Update after 151.1 recovery:
- The fair Flash EP16 LL graph cap512 retest with `ll-masked` baseline does not reproduce the slowdown. Current-vs-history deltas for common replay buckets are within about `-1%..+1%`.
- Therefore the current evidence does not support a code regression in the LL graph path. The earlier slow run should be treated as an unfair cap/baseline/env comparison or node-state artifact unless a future fair cap512 run regresses again.
- Skip current-vs-previous commit A/B for this issue for now; keep the A/B method only as the contingency if the fair command regresses later.

## 2026-07-03 - Pro EP8 Normal K1 Boundary Finding

- Pro EP8 normal 512 hangs before any result JSON even with `correctness_iters=0` and baseline benchmarking skipped, so the baseline and final correctness comparison are not the blocker.
- IPC mode with `K3_USE_ASM_TAIL_REDUCE=0` also hangs, so the current failure is not specific to RPC/Fabric peer memory or K3 tail-reduce1.
- Temporary stage-stop instrumentation narrows the first non-returning stage: the function returns after `start_barrier`, but does not return after normal K1. This means K2, K3, combine, and final reduce are not yet in scope for this failure.
- The active hypothesis is a K1 normal ASM or launcher assumption that still depends on Flash-style `local_experts<=32` / old route-scratch compact metadata layout. The next comparisons should target K1 `PACK5` and `UNIFIED_PACK5` ASM diffs around `local_experts`, `active_tiles`, `tile_experts`, and any hard-coded `32`/`64` values before adding new experiments.

Resolution update:
- The hypothesis was confirmed in the old K1 normal in-ASM route builder. It still assumes 32 local experts in several places: `rank * 32`, reset/publish loops bounded by 32, and route meta flag coverage for 32 entries.
- Pro EP8 has `local_experts=48`, so local experts 32..47 can wait on route metadata that the old builder never publishes. This explains the K1 timeout and why the same shape is not blocked by baseline or K3 tail-reduce settings.
- The minimal fix is launcher-side: force compact prebuild for `local_experts > 32`. This does not change Flash EP8 (`local_experts=32`), Flash EP16 (`16`), or Flash EP32 (`8`) default routing.
- Verification on 151.1 after rebuild: Pro EP8 512 K1 stage-stop returns; full fused-only returns; correctness against `normal-contiguous` passes with `max_abs=0.000976562`.
- The temporary `MEGAMOE_DCU_STAGE_STOP` instrumentation is now only cleanup debt and should be removed before broad timing data is collected.

Clean-source update:
- `MEGAMOE_DCU_STAGE_STOP` has been removed from production source and should not be reintroduced. It was only useful for locating the K1 boundary.
- Clean Pro EP8 512 IPC sanity still passes after removal, and Pro EP8 512 RPC with baseline bench also passes. RPC 512 timing is fused `4.7955 ms` versus normal-contiguous baseline `4.7731 ms`.
- The next Pro EP8 question is no longer "does 512 hang"; it is broad-bucket behavior starting from 1024. The first 1024 batch attempt was interrupted by 151.1 host-side SSH closure before result JSON, so no correctness/performance conclusion should be drawn from that interrupted run.

## 2026-07-03 - Current Status Consolidation Finding

- Pro EP16 8192 should be described precisely: normal eager fused correctness/execution is stable; graph cap8192 replay8192 correctness can pass, but graph benchmark/cleanup has shown VMFault/barrier-timeout behavior.
- Flash EP16 LL graph with `ll-masked` baseline is usable on 151.1. The successful fair retest used cap512 and node-actual environment. Do not require README HCA/topology variables on 151.1 while `/sys/class/infiniband` is empty.
- README has enough Pro documentation for now: it lists the Pro shape (`experts=384`, `hidden=7168`, `intermediate=3072`) and local experts for EP8/EP16/EP32, plus one Pro EP16 validation command. A full Pro EP8/EP32 matrix is not required unless requested.
- Current operational blocker is not card occupancy: host-side `hy-smi` shows all 16 HCUs normal/idle and no KFD PIDs. The blocker is that `sglang_megamoe` exited with code 255 and must be restarted before container-side testing resumes.

## 2026-07-03 - No-Copy Build Flow Finding

- The previous stale-extension failure mode was caused by source-tree imports resolving old `.so` files even when the latest build output existed under `build/lib...`.
- The build script is now the source of truth for remote MegaMoE runtime artifacts: it deletes stale source-tree `.so/.co` files, runs `build_ext --inplace`, verifies fresh source-tree artifacts, and imports `megamoe._C` plus K1/K2/K3 extension modules from `/root/yuguo/DeepGEMM`.
- This removes the manual `build/lib... -> source tree` copy step from the normal remote validation loop. If a future run imports from `build/`, site-packages, or another repo path, the build script should fail before GPU testing starts.
- Verification on 151.1 after this change: rebuild passed, source pytest passed (`11 passed`), and Pro EP8 512 normal eager RPC smoke passed with `max_abs=0.000976562`.
