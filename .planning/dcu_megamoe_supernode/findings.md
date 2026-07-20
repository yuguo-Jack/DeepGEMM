# Findings

## Current Status

- ✅ EP8/EP16/EP32 shape gates, scratch sizing, signal-slot helpers, and peer-memory path are implemented on the supernode branch.
- ✅ EP16 single-node runtime bring-up now builds and passes staged smoke on node22.
- [ ] EP32 multi-node validation is still pending; EP16 has broad normal/LL coverage, with large normal graph-bench instability tracked separately in `task_plan.md`.
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

## 2026-07-03 - Pro EP8 Normal Eager Finding

- Pro EP8 normal eager is no longer blocked at K1 after forcing compact prebuild for `local_experts > 32`.
- The full RPC normal eager matrix `512,1024,1025,2048,2050,4096,4097,5120,8192` passed correctness against `normal-contiguous`; every bucket produced baseline timing and cleared KFD PIDs after exit.
- Performance scales from near-baseline at 512 (`4.7537 ms` fused vs `4.8026 ms` baseline) to about `1.75x` faster at 8192 (`27.6906 ms` fused vs `48.5922 ms` baseline).

## 2026-07-03 - Pro EP8 LL 512 Finding

- Pro EP8 LL 512 needs a larger DeepEP LL heap than the README Flash-sized example. DeepEP reported `num_rdma_bytes(11450456192)`, so the successful 151.1 `ll-masked` runs used `ROCSHMEM_HEAP_SIZE=12884901888` and `DUSHMEM_HEAP_SIZE=12884901888`.
- Do not force stale HCA/topology variables on 151.1 while `/sys/class/infiniband` is empty. The successful Pro EP8 LL baseline runs used node-actual settings plus the generic DeepEP LL context variables.
- Correctness is not the blocker: Pro EP8 LL 512 passed against `ll-masked` with `max_abs=0.000976562`, and graph replay512 also passed.
- Performance is the blocker: Pro EP8 LL fused is about `16.63-16.68 ms` for replay/eager 512, while the `ll-masked` baseline is about `5.68-5.75 ms`. The same timing in graph mode shows the gap is inside the fused LL path, not just launch overhead.

## 2026-07-03 - Pro LL Graph Performance Finding

- Pro EP16 LL graph cap512 has now been compared directly against `ll-masked`; it is also abnormal. Fused graph replay medians for `8/32/128/256/512` are `3.2558/3.3209/5.1238/8.6320/15.2628 ms`, while `ll-masked` baseline graph medians are `2.4204/2.4802/2.6445/3.3827/4.7778 ms`.
- This separates Pro behavior from Flash EP16: Flash EP16 LL graph is faster than `ll-masked`, but Pro EP16/EP8 LL graph is slower.
- Split-tail is not the root cause: disabling `MEGAMOE_DCU_LL_K3_SPLIT_TAIL` made Pro EP8 LL graph512 slightly slower.
- `ll_block_m` is only a secondary tuning parameter. For Pro EP16 replay512, block64 improved `15.2628 -> 13.7521 ms`, and block48 improved to `12.9954 ms`, but both remain far behind the `4.78 ms` baseline.
- Kernel profiling isolates the dominant cost to K1 LL. In a Pro EP16 block48 fused-only run, `V3_K1_LowLatencyMaskedGroupGemmKernel<24, 6144, 7168, ...>` took about `11.5-11.8 ms` per call; K2 and K3 were much smaller.
- Root-cause direction: the current LL K1 kernel is a C++ low-latency grouped GEMM path tuned enough for Flash but not competitive for the Pro K1 shape (`N=6144`, `K=7168`). A simple size-support patch is insufficient for Pro LL performance.
- Practical path: keep Pro small-token auto selection on the normal backend unless/until a Pro-optimized LL K1 path is implemented. A proper Pro LL fix likely needs a dedicated/DeepGEMM-derived K1 LL kernel rather than another launch-policy tweak.
- This validates the Pro EP8 local-experts=48 route for eager normal. It does not yet validate Pro EP8 graph, LL, or uneven buckets.

Update after Flash-vs-Pro K1 profile:
- The abnormal Pro LL cost is not explained by the Pro K1 arithmetic size alone. Under the same EP16, token512, block32, fused-only profile, Flash K1 median is `1.3383 ms` and Pro K1 median is `13.9043 ms`.
- The theoretical K1 work ratio at the same token/topk is `(6144 * 7168) / (4096 * 4096) = 2.625x`, while measured K1 ratio is about `10.39x`; per unit work Pro is roughly `3.96x` slower.
- The Pro support diff for LL K1 did not add a Pro-specific optimized kernel. It generalized the old Flash template launch from `<4096,4096>` to `<6144,7168>` and enabled local experts `24/48`.
- Source inspection shows LL K1 does not compute empty capacity rows in the GEMM loop; it uses `cur_tokens` to derive `m_tiles`. Stage-copy traffic scales mainly with hidden (`7168/4096 = 1.75x`), also too small to explain the observed K1 ratio.
- Current working hypothesis: the Pro `<6144,7168>` instantiation exposes a template/codegen problem in the HIP C++ LL K1, especially the fully unrolled `kKIterations = 112` outer loop. Next ablation should test K-loop unroll behavior before any production fix.

Update after K/N dimension ablation:
- The fully unrolled K-loop hypothesis was tested and rejected: `#pragma unroll 1` made Pro EP16 LL fused-only worse (`17.3007 ms`), and disabling block32 parallel stage-copy was also slightly worse (`15.8442 ms`).
- Expert-count/local-expert effects are ruled out for the main gap. A temporary `experts=384, hidden=4096, intermediate=2048` EP16 LL run stayed Flash-like at `1.8772 ms`.
- The dominant trigger is `K=hidden=7168`. With `experts=384`, `hidden=7168, intermediate=2048` already costs `10.6547 ms`, and hipprof shows K1 `<24,4096,7168>` averaging about `9.1-9.7 ms`. In contrast, `hidden=4096, intermediate=3072` costs only `2.5402 ms`, with K1 `<24,6144,4096>` about `1.56-2.09 ms`.
- Therefore the current HIP C++ LL K1 path has a K-dimension/codegen or memory-pipeline scaling failure around `kKIterations=112`, not a route-scratch/local-expert issue and not primarily the N/output width. Another small launch-policy tweak is unlikely to close the `ll-masked` baseline gap.
- All temporary mixed-shape gates were removed after measurement, and the remote default source was rebuilt with source pytest `11 passed`.
- A partial-unroll ablation also failed to improve Pro K1: changing the main K loop to `#pragma unroll 8` produced fused `15.7133 ms`, equivalent to default `15.7026 ms`; `unroll 1` was worse at `17.3007 ms`. This narrows the remaining path away from simple unroll-factor tuning and toward a different K1 implementation/tile pipeline for `K=7168`.

Update after layout/resource check:
- Do not splice normal ASM K1 directly into the LL path as a shortcut. The LL path relies on per-expert contiguous rows and `m_indices` as per-expert `actual_m`; normal ASM K1 produces compact/nondeterministic row order and row-wise metadata. A correct hybrid would need a bridge that rebuilds per-expert row layout and counts, which is no longer a narrow K1 performance ablation.
- Current gfx938 resource metadata for EP16 block32 does not show catastrophic resource growth: Flash `<24,4096,4096>` is `vgpr=124, sgpr=100, private=0`, while Pro `<24,6144,7168>` is `vgpr=132, sgpr=106, private=208`, with no VGPR spill. This makes a pure occupancy/register-spill explanation weak.
- The next useful local ablation is Pro-only K1 tile granularity: try `blockN=128` for the Pro K1 LL template to see whether more N tiles with fewer accumulators improves the `K=7168` path. Keep it temporary unless correctness and timing justify production work.

Update after Pro K1 tile-granularity ablation:
- Pro-only K1 LL `blockN=128` plus Pro `ll_block_m=48` is a strong positive ablation. Pro EP16 LL graph cap512 replay512 improved from `15.2628 ms` to `5.0180 ms`, while `ll-masked` baseline stayed `4.7777 ms`.
- Small replay buckets also improved enough to beat the masked baseline: replay8/32/128/256 fused medians were `1.3350/1.4698/1.6748/3.0813 ms` versus baseline `2.4166/2.4787/2.6485/3.3804 ms`.
- Current root-cause statement: the old Pro `K=7168,N=6144,blockN=256` instantiation carried too much per-tile accumulator/codegen pressure for the LL K1 kernel. Splitting N to 128 removes that collapse without changing Flash's `K=4096` path.
- Keep the finding conditional until guardrails pass: Flash EP16 LL graph must remain in its historical band, and Pro EP8 LL 512 must show the same improvement.

Guardrail update:
- Flash EP16 LL graph cap512 remains effectively unchanged with the candidate: replay512 `1.9169 ms` versus recent fair history `1.9043 ms`, with the same blockM32 path.
- Pro EP8 LL 512 confirms the fix generalizes across EP sizes: graph replay512 improved from `16.6293 ms` to `5.3764 ms`, and is now faster than `ll-masked` baseline `5.7417 ms`.
- This is enough evidence to keep the narrow Pro LL K1 tile selector instead of immediately creating a separate Pro-only kernel family. The change is still shape-gated by `K=7168`, so Flash `K=4096` stays on the previous blockN256 instantiation.

Next optimization direction:
- The current Pro LL K1 fix is a successful tile granularity fix, not proof that K1 is near optimal. The right next step is to extract or emulate a K1-only groupgemm benchmark with the same LL layout contract, tune the GEMM core in isolation, then backport only the winning structure to the fused K1 path.
- The benchmark must keep the LL output contract intact: per-local-expert contiguous output rows and `actual_m` counts. A normal-ASM K1 shortcut remains invalid unless it rebuilds that layout.
- K1-only benchmark harness is now available in `test_mega_moe_dcu.py` through `--k1-only-bench`; blockM can be ablated with `--k1-only-ll-block-m`.
- The first 151.1 Pro EP16 blockM ablation attempt was interrupted by host SSH closure before any result JSON, so it provides no performance conclusion. Resume by checking node health, then rerun a short single blockM smoke before the full matrix.
- After adding a host epoch guard, full K1-only still reproduced the SSH-closure failure before writing JSON. This points away from the Python warmup/repeat loop alone and makes component ablation necessary.
- The next useful split is:
  - `no-start-barrier` to test whether the K1 kernel-side rank barrier is the trigger;
  - `pure-gemm` to test the groupgemm core after dispatch/route scan/stage copy has been initialized once.
- A local pure-gemm path has been prepared, but it requires remote rebuild before it can produce data.

Update after pure-gemm/no-start ablation:
- The K1-only diagnostic split is now producing useful data. Full mode with the in-kernel start barrier remains unsafe and can close SSH before a result JSON, so it should not be used for performance attribution.
- `no-start-barrier` versus `pure-gemm` isolates non-GEMM K1 overhead. For Pro EP16 token buckets `8/32/128/256/512`, no-start medians are `1.4832/1.6491/1.7492/2.9272/4.5657 ms`, while pure-gemm medians are `1.0704/1.1419/1.1690/2.1947/3.4260 ms`.
- Dispatch/route scan/stage copy plus barrier setup therefore costs roughly `0.41/0.51/0.58/0.73/1.14 ms`. This is measurable, but the 512-token GEMM core itself still accounts for about `3.43 ms`, so core GEMM tuning remains the main optimization target.
- `ll_block_m=64` is a negative result for Pro LL K1. It measured about `10.13 ms` at token256 and then interrupted SSH at token512. Do not spend more time on bm64 unless a later code change materially changes the kernel structure.
- bm32 and bm48 have a real token-size tradeoff: bm32 was faster in the noisy token256 sample, while bm48 stayed better at token512. This needs a stable rerun before changing the production selector.
- The next useful core-GEMM axis is blockN, not another bm64 retry. A pure-gemm-only `blockN` diagnostic is prepared so Pro K1 can compare 64/128/256 on the same staged inputs while keeping production fused behavior unchanged until measurements justify a change.

## 2026-07-03 - Normal Tail-Reduce Graph Replay Finding

- Flash EP16 normal graph cap8192 with explicit tail-reduce1 reproduced the large graph instability after eager correctness passed. The failure happens inside graph replay, before the first bucket result is printed.
- The strongest signal is the rank barrier timeout at EP16 start-barrier slot `32` with `generation=2` and `release=1`. That indicates some ranks entered the next replay's start barrier while others did not finish the previous replay epoch.
- Earlier hypothesis: normal tail-reduce graph may need a post-K3 rank barrier so back-to-back graph replay cannot reuse/reset tail-reduce counters before all ranks finish K3.
- This hypothesis was later tested and rejected on 151.1; see the validation update below. It is retained here only as diagnostic history.

Update after validation:
- The post-K3 rank-barrier hypothesis is rejected. Flash EP16 normal graph cap8192 still VMFaulted in `rank_barrier_kernel` slot32 after eager correctness passed, and the extra barrier was removed.
- Do not treat "add another kernel after K3" as the fix path. The next investigation should isolate the graph replay state transition itself: which captured operation first observes stale/reset tail-reduce or compact-route state, and whether tail-reduce graph should be disabled for large EP16/Pro caps until a no-extra-kernel fix exists.

## 2026-07-05 - Pro K1 Pure GroupGEMM Baseline Finding

- The Pro EP16 pure K1 groupgemm core is still slower than same-shape DeepGEMM masked grouped GEMM on the same staged inputs and `actual_m`.
- Matrix result under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_pure_vs_deepgemm_matrix_20260705_085436`: tokens `8/32/128/256/512`, pure K1 medians `1.1403/1.2496/1.2285/2.2936/3.6320 ms`, DeepGEMM masked medians `0.8917/0.9627/0.9679/1.4053/2.1625 ms`, ratios `1.279/1.298/1.269/1.632/1.679`.
- Numerical diff between pure K1 and DeepGEMM masked is small but not bit-identical: max_abs is `0.001953` for 8/32 and `0.003906` for 128/256/512, while mean_abs stays around `4e-09`. Treat this as BF16-level accumulation/order difference unless a later correctness check ties it to an end-to-end failure.
- Current acceptance gate is stricter than "fused is close": the pure Pro K1 groupgemm itself should not be this far behind DeepGEMM masked. Optimize the pure skeleton first, then reconnect dispatch/stage/copy/fused path only after the core is close.
- BlockM/blockN diagnostic under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_pure_bm_bn_deepgemm_20260705_090146` shows parameter tuning alone is insufficient:
  - token256 best: bm32/bn128, pure K1 `2.0946 ms`, DeepGEMM `1.4369 ms`, ratio `1.458`.
  - token512 best: bm48/bn128, pure K1 `3.6236 ms`, DeepGEMM `2.1727 ms`, ratio `1.668`.
  - bn64 is slower than bn128 at both token sizes; bn256 is unusable for Pro (`4.83x-6.40x` slower than DeepGEMM).
- Current optimization axis: inspect/profile the pure groupgemm skeleton against DeepGEMM masked. If the Flash-tuned LL K1 structure cannot match DeepGEMM for `K=7168,N=6144`, create a Pro-only pure K1 kernel family rather than compromising the Flash path.

Correction after re-checking the local DeepGEMM references:
- The DeepGEMM masked baseline should not be described as the `MT256x256x128` path. The matching local reference is `hygon_tmp/K1_groupgemm_fp8/deepgemm_groupgemm_masked_fp8_marlin_balanced_256x64x128_TN_BF16_WGM8.s`, symbol `DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT256x64x128_WGM8_GROUPGEMM_MASKED`.
- `MT256x256x128` is the normal/contiguous large-tile reference in the same scratch workspace, not the masked low-latency reference used for LL baseline comparison.
- The attempted Pro MT256 pure diagnostic branch is therefore not the right primary optimization direction for `ll-masked`; it was also unstable on 151.1 and has been removed from the runnable code path.
- Continue Pro K1 pure optimization from the stable low-latency pure-gemm skeleton and compare/borrow only from the masked `256x64x128` reference unless new package-symbol evidence proves a different masked dispatch.

DeepGEMM package-commit reference check:
- Local reference repo was cloned from `http://42.228.13.241:10068/dcutoolkit/deeplearing/deepgemm.git` on `develop`, then checked out to the environment package commit `6a53e9c45c7d6b46395c3a85231d5f2322a36a2a`.
- The current environment package reports `deepgemm` version `2.1.0+das.opt1.dtk2604.torch2100.2606152155.g6a53e9`, so this checkout matches the active baseline rather than newer develop HEAD.
- For `m_grouped_fp8_gemm_nt_masked`, `deepgemm_masked_config_i8fp8()` maps `EXPECTED_M16/32/64` to `mode=1002`, and values beyond configured keys also fall back to `1002`. Python config names this rows `block_m=64`, cols `block_n=256`.
- In C++ dispatch, `mode=1002` launches `_m_grouped_marlin_fp8_gemm_nt_masked_asm_impl<256, 64, 128, scalar_t, 512>`, which maps to code object `deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co`.
- Offline `llvm-readelf`/`llvm-objdump` on the installed gfx938 `.co` confirmed one kernel symbol, `DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT256x64x128_WGM8_GROUPGEMM_MASKED`, with a 512-thread/WGM8 style kernel. The disassembly has 128 `v_mmac_f32_16x16x32_fp8_fp8` instructions, B-side LDS reads (`ds_read_b128`), direct global `buffer_load_dwordx4`, and scalar `s_waitcnt`/`s_barrier` scheduling.
- Optimization implication: do not change LL weight layout to chase this baseline. The actionable structure to borrow is the Pro masked GEMM pipeline shape: 8 waves / 512 threads, 128 persistent workgroups, row tile 64, column tile 256, and an LDS-backed B-side pipeline. The current MegaMoE pure LL K1 skeleton is direct-load/shuffle and 4-wave on the production path, so the remaining gap is likely the GEMM core pipeline, not dispatch or layout.

Local diagnostic cleanup finding:
- Keep `--k1-only-ll-block-n` because it is still useful for the stable pure-gemm core comparison, but do not keep `--k1-only-ll-cus` or the 128-CU/8-wave instantiation as a casual diagnostic knob.
- Reason: DeepGEMM masked's 512-thread/WGM8 structure is a kernel-design reference, not proof that the existing direct-load LL K1 template can be safely widened by only changing `CUS/WARPS`. A future 8-wave path should be introduced as an intentional Pro-only kernel family with correctness and performance data, not hidden behind the current harness.

Old C LL vs masked ASM shape finding:
- The user's suspicion is valid: the old `C fp8 groupgemm` LL path was designed and measured against Flash `E32,N4096,K4096`, and in that shape it still tracks the copied masked ASM closely on 151.1: C-LL is slightly faster through tokens 256 and only about `1.15x` slower at token512.
- The same old C-LL skeleton does not generalize cleanly to Pro `E24,N6144,K7168`. In the scratch harness, Pro `blockN=256` is `4.6x-9.6x` slower than the masked ASM, while Pro `blockN=128` is much better but still `1.3x-2.3x` slower.
- This means the current Pro pure-K1 gap is not primarily caused by MegaMoE dispatch, stage copy, K2/K3, or graph mechanics. The old direct-load/shuffle C-LL GEMM core has a shape cliff at Pro K/N. Any next optimization should first explain/profile this standalone gap against the masked `256x64x128` ASM structure.
- Scratch candidate sweep shows launch widening is not enough: `CU128`, `8 waves`, and `BM64/BN256` are all slower than the current `4w/CU64/BM48/BN128` direct-load skeleton for Pro E24. This strengthens the hypothesis that the gap is the memory/compute pipeline structure, not merely persistent block count or wave count.

Pro LDS-backed pack5 skeleton finding:
- Direct Pro C-LL profiling and ISA inspection show a clean core gap: tokens512 direct C-LL averages about `2.48 ms`, while masked ASM averages about `1.18 ms`. The direct slice is dominated by fully unrolled global loads plus `ds_bpermute`; the masked reference uses a compact LDS-backed WGM8 `256x64x128` structure.
- Removing the direct kernel's scheduler barrier is not valid even though it is faster. The no-sched variant improves tokens512 to about `2.30 ms` but fails correctness with `max_abs=0.0610352`, so that dependency must be respected for the direct-load/shuffle path.
- A scratch-only Pro parameterization of the existing LDS-backed pack5 C skeleton is correct after three Pro-specific fixes:
  - template `kProblemN/K` must be `6144/7168`, not fixed `4096/4096`;
  - lowlat-pack expert stride must be `N*K`, and K-outer stride must be `N*64`;
  - Flash's `stage_iter ^ 16` ordering is invalid for Pro because K=7168 has 56 K-stages; Pro uses linear K-stage order in the scratch candidate.
- The corrected LDS row256 candidate is exact against the ASM reference at tokens512 (`max_abs=0`, no bit mismatches), and improves large-bucket Pro pure K1 materially:
  - tokens128: masked `0.6142 ms`, direct C-LL `0.7964 ms`, LDS row256 `1.2018 ms`;
  - tokens256: masked `0.6432 ms`, direct C-LL `1.5931 ms`, LDS row256 `1.3151 ms`;
  - tokens512: masked `1.1948 ms`, direct C-LL `2.4797 ms`, LDS row256 `1.4948 ms`.
- The row256 path should not replace the direct path globally: it over-computes small buckets and loses at tokens128. Treat it as a Pro large-bucket kernel candidate, while preserving Flash and Pro small-bucket behavior.

Formal row256 port recheck:
- The formal MegaMoE K1-only check showed the row256 LDS candidate is not correct for the current LL contract. On the same staged input/layout, LDS row256 differed from both direct128 LL and DeepGEMM `ll-masked` by the same amount (`max_abs` about `0.066-0.074`).
- The scratch pass was against the normal/MT256x256 reference path, while MegaMoE LL compares against the masked WGM8/direct orientation. This is a reference-orientation mismatch, not a stale-build problem.
- Do not connect the current row256 LDS skeleton to fused LL. Any future Pro K1 pure kernel must preserve the existing LL weight layout and the LL masked/direct output orientation, or prove a full rewrite against `ll-masked` before entering production code.

Pro masked-K1 fused path finding:
- The fastest currently verified Pro LL K1 route is an independent-layout path, not the old unified-layout C-LL pure kernel. It uses MegaMoE LL K1 only for deterministic route/stage packing, then calls the same DeepGEMM masked grouped GEMM wrapper as the `ll-masked` baseline for L1.
- This is intentionally additive: the old Flash-friendly Pro unified-layout LL path with `blockN=128` and `blockM=48` remains present as a compatibility/fallback path, and Flash does not select `ll_pro_masked`.
- On Pro EP16 512, the new fused path is faster than both the old unified-layout fused path and the end-to-end baseline because K1 uses DeepGEMM masked while K2/K3 still use MegaMoE fused execution: `3.877 ms` vs old unified `5.233 ms` vs baseline `4.877 ms`.
- Graph cap512 replay `8/32/128/256/512` passed correctness and measured `1.014/1.126/1.370/2.239/3.872 ms`, all faster than the `ll-masked` baseline `2.420/2.483/2.646/3.378/4.780 ms`.
- Pro EP8 also validates the same shape-gated path with `local_experts=48`: eager 512 `3.917 ms` vs baseline `5.678 ms`, and graph cap512 replay `8/32/128/256/512` `1.435/1.902/2.100/2.424/3.936 ms` vs baseline `3.325/3.744/3.913/4.147/5.729 ms`. Correctness max_abs stays at `0.000488281` or `0.000976562`. The same-code unified-layout fallback graph replay512 is `5.314 ms`, so the new path is about `1.35x` faster than unified for Pro EP8 512.
- A fresh Pro EP16 unified-layout graph A/B on the same current code gives replay512 `5.185 ms`, which is slower than both the new independent-layout path and the baseline. This keeps the unified path useful as a compatibility/fallback route, but it should not be the preferred Pro LL performance path if the remaining guardrails pass.
- User-adjusted next direction: do not keep comparing against old Pro unified fused K1 during optimization. The meaningful ladder is DeepGEMM masked ASM -> Pro C pure K1 -> Pro Flash-style fused K1.
- The current `ll_pro_masked` split path is therefore best viewed as an interim high-performance fallback and oracle. It proves the independent layout is viable and gives an e2e target, but it does not answer whether K1 can be fused back into MegaMoE without losing the masked-ASM-level GEMM core.
- Next kernel finding to prove: a Pro C pure K1 kernel can match the `256x64x128 WGM8 GROUPGEMM_MASKED` reference while preserving LL/masked orientation and `actual_m`. Only after that proof should route/stage/start-barrier fusion be attempted.

Direct masked-layout prototype finding:
- The masked DeepGEMM L1 layout is not just a linear `[N16,K16]` order. The Pro C prototype must apply the same physical N16 mapping used by the pack5/marlin family inside each 16-wide N group. Missing this mapping caused the large same-input K1 diff; adding it reduced the difference to one BF16 quantum (`max_abs=0.00390625`, mean about `4e-09`) in the Pro EP16 token128 same-input check.
- Correctness alone is not enough: direct masked-layout loads are much slower than the DeepGEMM masked ASM at Pro size. The measured token128 ratios were `7.37x` for the widened `BM64/BN256/8wave` prototype and `5.83x` for the current direct `BM48/BN128/4wave` skeleton.
- Treat the direct masked-layout C path as a diagnostic/orientation oracle only. The next viable pure K1 candidate needs an LDS-backed masked-orientation pipeline that borrows the `256x64x128 WGM8 GROUPGEMM_MASKED` structure without returning to the rejected normal/MT256 row256 contract.

Formal C backend hook finding:
- The Pro masked-K1 path now has a formal MegaMoE C groupgemm backend gate. With `MEGAMOE_DCU_PRO_LL_MASKED_K1_C_GROUPGEMM=1`, `ll_pro_masked` still uses K1 stage-only route/stage packing, but L1 GEMM is produced by the MegaMoE C pure groupgemm extension instead of the DeepGEMM masked wrapper.
- This validates integration shape and correctness in the real e2e path: Pro EP16 token128 eager passed against `ll-masked` with `max_abs=0.000488281`.
- It also quantifies the current gap in the real path: the C backend smoke measured about `5.86 ms`, far slower than the masked baseline timing in that short run. Therefore the C backend is now a formal optimization target/scaffold, not a performance solution.

Formal LDS C backend finding:
- The first formal masked-orientation LDS C candidate is correct but still not fast enough. On Pro EP16 LL token128, the default split path with DeepGEMM masked K1 measures `1.383 ms` fused, while the C LDS backend measures `2.290 ms` fused on the same command. This proves the remaining gap is in the C K1 core, not in the surrounding Pro `ll_pro_masked` e2e flow.
- Resource metadata explains part of the gap: `V3_K1_ProMaskedLdsGroupGemmKernel<24>` uses `64 KiB` LDS and `211` VGPR/thread with no spill. The kernel maps one wave over a full N256 slice, so each compute wave carries 32 `float32x4` accumulators. That is structurally different from the masked `256x64x128` WGM8 reference, whose useful clue is 512 threads / 8 waves with an LDS-backed pipeline and much more compact static MMAC structure.
- The current 256-thread LDS candidate is useful as a correctness-bearing formal scaffold and is much faster than the direct C scaffold, but it should not be treated as the final pure K1 design. Next optimization should intentionally build a WGM8-like Pro C K1 family that splits the N256 tile across more compute waves to reduce accumulator pressure, rather than simply changing `blockN` or widening the old direct-load template.
- 151.1 became unreachable during a K1-only same-input compare of this candidate. Until the node recovers, do not infer a kernel correctness failure from that run; first verify host/container/HCU state, clear any stale distributed processes, then rerun a short K1-only smoke.

ASM-guided current C finding:
- Current C save-temps and object-level extraction show the first formal Pro masked LDS backend is still structurally far from DeepGEMM masked ASM even though the math core has the same `128` `v_mmac` static count.
- The C `<24>` kernel uses `64 KiB` LDS and keeps two compute waves over M halves; DeepGEMM masked uses a 512-thread WGM8 layout with compute waves `0..3` and loader waves `4..7`.
- The ASM line `s_mul_i32 s[sgprTemp0], s[sgprWaveiD], NperWAVE` with `NperWAVE=16` is the key mapping clue: wave id shifts N ownership, while row ownership is derived from lane group plus store row offsets. The earlier 512-thread/M16 C experiment was wrong because it split physical rows by wave id instead of splitting N.
- A naive single-LDS-stage C attempt is invalid. It reduces LDS metadata from `64 KiB` to `32 KiB`, but it lets loader waves overwrite the same LDS buffer for the next K stage while compute waves still read the current stage. Correctness failed with `max_abs~0.99`.
- Optimization implication: keep the current DeepGEMM masked split path as the fallback/oracle. The next real C optimization must rewrite wave ownership and scheduling closer to WGM8 while preserving the LL masked layout/output contract; do not spend more time on single-buffer or blind blockN sweeps.

Tune-loop extraction finding:
- Full `k1_fused_ext.cu + k1_v3_fused_ext.cu` rebuilds are too slow for ASM-guided iteration. The practical loop is now a tune-only extension that compiles only the Pro masked LDS pure K1 launch and the current C kernel header.
- This keeps the same-input contract intact: staged input/scales and masked L1 weights come from the real K1 stage-only path, while timing/diff compares directly with DeepGEMM masked on those tensors.
- Optimization target remains strict: generated C `.s` should converge toward the installed DeepGEMM masked `256x64x128 WGM8 GROUPGEMM_MASKED` structure, and the C pure K1 should approach ASM performance while reducing the residual BF16-level diff where possible.
- First tune-only `.s` comparison confirms the current gap. Per instantiated C kernel (`local_experts=12/24/48`), static counts are `128 v_mmac`, `32 ds_read_b128`, `24 ds_read_b32`, `36 buffer_load_dwordx4`, `256 buffer_store`, `61 s_barrier`, `379 s_waitcnt`, `2 s_setprio`. The DeepGEMM masked source reference has the same `128 v_mmac` but far fewer scheduling/epilogue instructions in the source count: `26 buffer_load_dwordx4`, `29 buffer_store`, `16 s_barrier`, `21 s_waitcnt`, `8 s_setprio`.
- Immediate optimization target is therefore not arithmetic coverage but C schedule/ownership: reduce store fanout, reduce wait/barrier count, and move toward the ASM WGM8 wave ownership where wave id splits N (`NperWAVE=16`) instead of the current two-compute-wave row split. Precision work should follow the same alignment: match ASM accumulation/store/scale order to see whether the current one-BF16-quantum max diff can be reduced.
- Matching the ASM scale order is useful for precision: computing `weight_scale * x_scale` first and then multiplying by the accumulator made the Pro EP16 token128 tune-ext K1 output bit-exact versus DeepGEMM masked (`max_abs=0`, `mean_abs=0`) without materially changing timing or generated instruction structure. Keep this precision alignment unless a later WGM8 rewrite naturally supersedes it.
- The first 512-thread WGM8 rewrite proved that N-split alone is not sufficient. A weight-LDS WGM8 variant lowered VGPR/static store count but slowed token128 to `2.3890 ms` because each N-split compute wave still reloaded the same `x` rows from global memory.
- The useful WGM8 dataflow is input-LDS plus direct masked-weight loads, matching the DeepGEMM masked asm direction. The current best tune-only C candidate stages `x` in LDS, direct-loads masked weights, uses masked-only stores for small partial tiles, and maps wave id to interleaved N16 groups (`n16 = R * 4 + wave_id`). On Pro EP16 token128 it is bit-exact versus DeepGEMM masked and improves C K1 to `1.2900 ms` versus DeepGEMM `0.9800 ms` (`1.316x`).
- The asm row-based LDS swizzle cannot be copied mechanically into the current x-LDS layout. The direct `k_vec ^ (row & 7)` write/read attempt failed correctness with `max_abs=1.67578125`; any future swizzle work must re-derive the exact layout rather than reusing that patch.

Full-tile x-LDS loader finding:
- Removing redundant loader bounds and adding a branchless full-tile x prefetch path is a valid C approximation of the ASM's branch-light prefetch style. It keeps partial tiles guarded and preserves bit-exact output versus DeepGEMM masked.
- The current best Pro EP16 pure C K1 matrix is now tokens `8/32/128/256/512`: C `1.0785/1.1309/1.1506/1.6135/2.5278 ms`, DeepGEMM masked `0.8675/0.9616/0.9898/1.4401/2.1596 ms`, ratios `1.243/1.176/1.162/1.120/1.170`, with `max_abs=0` and `mean_abs=0` for every bucket.
- The remaining gap is no longer a gross layout or precision problem. The likely gap is compiler scheduling around direct masked-weight loads and store/control lowering: generated C still cannot reproduce the ASM's coarse `s_waitcnt vmcnt(8)` grouping, and compiler-emitted waits/control remain heavier than the hand-written `.s`.
- Next high-value options should therefore be source-backed and `.s`-checked: either use a bounded buffer-resource probe to replace partial-tile row branches safely, or introduce a very small inline-asm load/MMAC scheduling block for the weight side so C can hold more outstanding weight loads like the ASM. Do not spend more time on blind blockN/blockM sweeps.

ASM-guided scheduling rejection finding:
- Relaxing the phase4 wait by issuing the next-stage first four weight loads before `s_waitcnt vmcnt(4)` is unsafe in the generated C kernel. Even though the local `.s` window looked closer to hand ASM, token128 produced NaNs and `max_abs=0.3076171875`, which means the compiler/hardware queue ordering does not give the same guarantee as the hand-written ASM schedule.
- Two source-level simplifications were also measured and rejected. `#pragma unroll 2` made static shape more ASM-like but slowed token512; splitting full-tile and partial-tile store epilogues reduced static control but did not improve runtime. The retained kernel is therefore the previous split-prefetch x-LDS baseline.
- Moving `s_setprio` outside the K-stage loop is also a negative result. Keeping high MMAC priority across the entire loop worsened all checked buckets, so the per-stage high/normal priority toggles should remain in the current C schedule.
- Precision is currently not the limiting factor for the checked pure-K1 buckets: all retained and most rejected correct candidates are bit-exact versus DeepGEMM masked (`max_abs=0`, `mean_abs=0`). The remaining work is pipeline/scheduling latency, not numerical accuracy.
- The reference C/ASM-style packed epilogue (`v_pk_mul_f32`) is not a win in this generated C x-LDS kernel. It reduced static wait count and VGPRs, but Pro EP16 token `128/256/512` C times were `1.1573/1.6495/2.5306 ms`, worse than the retained split-prefetch baseline at the important 256/512 buckets. Keep the scalar epilogue unless a larger rewrite changes register scheduling.
- Splitting LDS consumption with `lgkmcnt(2)` is correct but not beneficial in the current phase shape. The `.s` formed the intended partial-wait schedule and stayed bit-exact, but token `128/256/512` C times were `1.1465/1.6482/2.5152 ms`; only 128 marginally improved while 256/512 regressed. Do not retry this exact 4-read/2+2 split without changing the surrounding MMAC/load grouping.
- Grouping four masked-weight `buffer_load_dwordx4` instructions into one inline asm block is also not a win. The first form was invalid because outputs used `=v` instead of early-clobber and clobbered offset VGPRs inside the asm block, causing a token256 VMFault. The corrected `=&v` form stayed bit-exact but measured token `128/256/512` C times `1.1402/1.6133/2.7117 ms`; token512 regressed badly, so the retained per-load helper remains better.
- Removing the phase0/phase4 accumulator `s_nop` dependency block is not a clean win. It remains bit-exact and slightly improves token512 absolute C time in one run (`2.4719 ms`), but token256 regresses (`1.6651 ms`), so keep the dependency block in the retained baseline.
- A phase-priority retiming variant is also rejected. It stayed bit-exact but did not improve the important buckets (`128/256/512` measured `1.1655/1.5922/2.4961 ms`), so the retained per-stage priority placement remains the better baseline.
- A `vmcnt(8)` double-buffered next-weight prefetch variant is unsafe in generated C. It increased VGPR pressure substantially and VMFaulted in `V3_K1_ProMaskedXLdsWgm8GroupGemmKernel<24>` during token128/tune testing. After reverting it, token128 smoke returned to `1.1661 ms` vs DeepGEMM `0.9573 ms`, ratio `1.218x`, with `max_abs=0`. Do not retry loop-carried pending VMEM registers without a tighter inline-asm/control proof.
- The actual DeepGEMM masked code object, not just the macro source `.s`, has `128 v_mmac`, `26 buffer_load_dwordx4`, `16 ds_read_b128`, `64 buffer_store_short`, `12 s_barrier`, `15 s_waitcnt`, and `8 s_setprio`. This confirms the reference really is slimmer than the retained C x-LDS kernel in wait/control and uses 64 stores, not an artifact of macro-source counting.
- Unconditional padding-row stores are rejected despite making the C `.s` statically closer to the reference. The variant reduced `<24>` static control to `64 buffer_store_short`, `77 s_waitcnt`, `19 s_cbranch`, and `5 v_cmp`, and stayed bit-exact, but token `128/256/512` measured `1.1507/1.6186/2.5129 ms`. Extra padding writes outweigh the branch reduction on the checked buckets, so keep the masked partial-tile store path.
- Offset-increment weight addressing is also rejected. It looked like a reasonable C simplification for `K1_XLDS_LOAD_W_AT` by carrying `w_off*` across K stages, but the generated kernel raised VGPR pressure (`123 -> 127`) and regressed token `128/256/512` to `1.1564/1.6807/2.5308 ms` while remaining bit-exact. The retained direct address expression is better because the compiler can schedule/recompute it without carrying eight extra loop-state VGPRs.
- After reverting offset-increment, the restored generated C `.s` for `<24>` matches the retained split-prefetch static shape: `64 v_mmac`, `22 buffer_load_dwordx4`, `8 ds_read_b128`, `128 buffer_store_short`, `6 s_barrier`, `217 s_waitcnt`, `2 s_setprio`, `131 s_cbranch`, `101 v_cmp`, and no `vmcnt(8)`. Token128 smoke is again bit-exact versus DeepGEMM masked (`1.1435 ms` vs `0.9687 ms`, ratio `1.180x`).
- The `MT256x64x128` reference shape should not be copied as a plain C N64 retile while keeping M64. A tune-only `xlds_wgm8_n64` candidate stayed bit-exact and had much lighter per-CTA static counts (`16 v_mmac`, `10 buffer_load_dwordx4`, `32 buffer_store_short`, `62 s_waitcnt`), but token128 slowed to `1.8438 ms` vs DeepGEMM `0.9524 ms`. The reason is likely that N64 increases CTA count by 4x without reproducing the reference's M256 row ownership and hand-scheduled main loop.
- The next4 relaxed-wait idea is rejected even with early-clobber single-load outputs. The generated `.s` did contain `buffer_load_dwordx4` for the next-stage phase0 weights followed by `s_waitcnt vmcnt(4)`, but the compiler/control-flow lowering still emitted a later `s_waitcnt vmcnt(0)` before phase4 MMAC, and runtime produced NaNs (`max_abs=0.40478515625`). This confirms that pending-VMEM retiming cannot be safely expressed with the current C variables and small inline helpers; it needs a larger hand-controlled block if pursued.
- The current evidence says precision is already at the practical target for the checked pure K1 buckets (`max_abs=0`). Further "precision improvement" should come naturally from preserving the ASM scale/order and avoiding unsafe wait changes; the main remaining gap is load/wait/MMAC scheduling. Small C edits that only improve static counts have repeatedly failed, so the next candidate should be a larger ASM-guided C/inline-asm block or a re-derived LDS layout that changes the actual main-loop schedule.

Pro fused-C K1 implementation finding:
- The first fused-C Pro LL K1 implementation deliberately copies the successful Flash-style structure rather than inventing a new driver flow: route/stage/start-rank-barrier is produced inside the K1 kernel, then the retained `xlds_wgm8` C groupgemm consumes the staged tensors and `actual_m` metadata in a persistent tile loop.
- The path is opt-in and shape-gated by `MEGAMOE_DCU_PRO_LL_MASKED_K1_FUSED_C=1`, `hidden=7168`, `l1_rows=6144`, `ll_block_m=48`, and `ll_cus=64`; Flash and the default Pro split DeepGEMM masked fallback should not select it.
- The fused helper needs an extra block-level barrier after the tile store path because the old pure groupgemm kernel returned after one tile, while the fused kernel keeps the same CTA alive for the next persistent tile. Without that barrier, loader waves for the next tile can overwrite x-LDS while compute waves from the previous tile are still finishing scale reads/stores.
- Runtime proof now shows the fused-C path is correct but not yet faster than the default split DeepGEMM masked-K1 path. On Pro EP16 LL eager, split measures `1.3687/2.2304/3.8939 ms` at tokens `128/256/512`; fused-C measures `1.6186/2.5056/4.2445 ms` before barrier retiming, and `1.6108/4.2252 ms` for checked `128/512` after moving the persistent-tile barrier before stores.
- The readlane broadcast tweak for `actual_m` is now measured. It improves fused-C slightly but does not change the decision: readlane fused-C measures `1.6004/2.4841/4.2079 ms` at tokens `128/256/512`, while the same-window default split path measures `1.3710/2.2329/3.9052 ms`.
- Interpretation: the fusion mechanics are viable and correct, but simply embedding the current C x-LDS backbone does not beat the DeepGEMM masked ASM split call. Keep fused-C as an opt-in experiment and keep split DeepGEMM masked-K1 as the Pro LL default. Further work should only proceed if the C backbone/main-loop schedule changes materially; do not change the LL weight layout or default fallback to chase this result.

Pro LL split finalization finding:
- Final user decision is to stop pursuing Pro LL K1 fusion for this branch and keep the split path as the production route.
- Production Pro LL split should mean: K1 stage-only route/stage packing inside MegaMoE, followed by a MegaMoE-packaged launch of the masked FP8 group GEMM ASM code object. It must not import/call the external Python `deepgemm` package in the MegaMoE execution path.
- The external `deepgemm` package remains acceptable for test baselines such as `--baseline-kind ll-masked`, because those are oracle/comparison paths rather than MegaMoE fused execution.
- Pro LL tests should default to `ll_pro_masked` for the Pro shape. `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` is the only Pro LL test-harness override that should force the old unified-layout compatibility kernel.
- The remaining pure-K1/C-fusion scaffolding is not part of the final branch contract. The source should not expose `ll_pure_*`, `--k1-only-*`, Pro LL fused-C env knobs, or tune-only extension paths; future K1 experiments should be reintroduced on a separate branch if needed.
- The first bundled masked K1 integration attempt used an untracked scratch `.s`, not the active DeepGEMM package's masked kernel. The active environment is DeepGEMM commit `6a53e9c45c7d6b46395c3a85231d5f2322a36a2a`, and its installed/develop masked Pro LL kernel is the prebuilt code object `deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co` with hash `73184662ec644cf9f4e9cfacec720a15428e84c5f84ad06e6e9e57bfa06543b4`.
- Correct integration should copy that `.co` as a prebuilt package artifact and never recompile the unrelated scratch `.s`. After switching to the develop `.co`, the MegaMoE default Pro LL split path passed Pro EP16 LL eager and graph cap512 against `ll-masked`, with graph replay512 `3.9464 ms` versus baseline `5.1602 ms`.

Pro EP8 LL split correctness isolation finding:
- Pro EP16 LL split is validated, but Pro EP8 LL token256 needs a clean-card rerun. The first failures have valid route stats and final fused output equal to the sum of visible combine slots for the failing column, which points away from the final local reduce and toward per-route slot production or the split-tail slot copy.
- Because the latest no-split-tail ablation overlapped with a resident SGLang service and hit allocation pressure, it is not usable evidence. The decisive next test is the same input on clean cards with split-tail enabled versus disabled.
- If split-tail-off passes, inspect `k3_v3_ll_combine_tail_split` slot copy/signaling. If it fails the same way, inspect the route owner's staged row through K1 masked GEMM, K2 activation, and K3 `output_workspace` before combine.
- The current capacity-overflow hypothesis is testable and must not be assumed as fact. Pro EP8 LL token256 is the most sensitive bucket because expected rows per local expert are `256 * 6 / 48 = 32`, aligned capacity is `64`, and the current LL headroom threshold does not add slack below expected `48`. If a failing run shows every `actual_m <= 64` and valid `row_combine_ptrs`, the capacity hypothesis should be dropped in favor of K3 split-tail copy/publish or per-route K1/K2/K3 value tracing.
- Clean-card token256 evidence now rules out split-tail as the first boundary: the default and `MEGAMOE_DCU_LL_K3_SPLIT_TAIL=0` runs both fail, and the fused output matches the visible combine-slot sum. Route-slot diagnostics for the default failing point show valid `row_combine_ptrs`, `row_in_expert < actual_m`, and nonzero staged scales, but some route rows have zero L1 output, default/min activation scale, and zero K3 output.
- Current-source token512 passes with the same Pro EP8 LL split path. The 256/512 contrast aligns with the masked K1 launch size: token256 uses `rows_per_expert=64` (`num_MBlocks=1`), while token512 uses `rows_per_expert=128` (`num_MBlocks=2`). The current minimal fix is to force only Pro EP8 LL masked K1 to at least `128` rows/expert across K1 launch and route-scratch sizing; this is a launch-shape guard, not a weight-layout change.
- The Pro EP8 LL min-rows fix is validated. Uniform tokens `8/32/64/128/256/512`, uneven EP8, and uneven EP16 all pass against `ll-masked`. Performance after the fix is acceptable for the split path: Pro EP8 eager is faster than `ll-masked` from token32 upward and graph is faster on every checked replay bucket. Flash EP8 LL graph cap512 still matches the prior `~1.814 ms` replay512 guardrail, so the Pro-shape guard did not produce a Flash LL regression.

LL baseline fairness finding:
- `ll-masked` baseline graph and eager paths both use CUDA graph replay internally, but they were not necessarily doing the same amount of work. The old test harness passed the graph allocation capacity (`sym_buffer.cuda_graph_max_tokens_per_rank`, e.g. Pro EP8 cap512 allocating/printing `512/768`) into DeepEP `low_latency_dispatch()` as `num_max_tokens_per_rank`.
- MegaMoE graph already uses the runtime token bucket for its active work. For a fair `ll-masked` baseline, the DeepEP LL dispatch cap must also be aligned to the current bucket (`expected_tokens_per_rank`, bounded by the actual local input rows), while the larger graph allocation cap remains only a buffer allocation property.
- The corrected Pro EP8 LL graph cap512 baseline medians are `1.4417/1.9496/2.1295/2.4835/3.1822/5.7530 ms` for replay tokens `8/32/64/128/256/512`, versus MegaMoE `1.4314/1.9004/1.9800/2.1053/2.4702/3.9245 ms`. This supersedes the earlier unfair small-bucket graph baseline values around `3.33-4.16 ms`.
- Do not cite old Pro EP8 graph `ll-masked` baseline data collected before this harness fix as fair performance comparison data. Re-run eager/graph/uneven summaries with the runtime-token-aligned harness if they are needed for final reporting.

Unified LL capacity finding:
- The unified LL K1/K2/K3 path is already closer to a masked design than a pure fixed-M design: K1 computes per-expert tile counts from `actual_m`, K2 scans or reads the max actual M, and K3 reads `actual_m` before iterating rows. Therefore a moderate `m_per_expert` guard can reduce skew overflow risk without forcing every later stage to process all padded rows.
- The remaining non-free costs of a larger guard are scratch size, K1 row initialization/staging metadata, and tail-tile/padding stores. This is materially cheaper than making the Pro split DeepGEMM masked ASM always use a larger `size_m`, so the 256-row skew guard is limited to Flash LL and Pro unified-layout LL (`ll_stage_only == false`).
- Pro default LL split remains governed by the separate Pro EP8 128-row masked-K1 launch-shape guard. Raising that path to 256 by default would change the DeepGEMM masked ASM `size_m`/`num_MBlocks` and is not currently justified by measured correctness data.
- 151.1 runtime smoke supports this split: Pro default LL split EP8/EP16 graph remains correct and performant, Pro `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` graph smoke is correct, and Flash EP8/EP16 LL graph guardrails remain in the historical band. The unified 256-row guard did not produce an observed Flash precision or performance regression in the checked cap512 graph buckets.
- Direct Flash EP8 LL small-cap retest also supports the same conclusion: cap8/replay8 measured `0.5446 ms` vs `ll-masked` `0.5452 ms`, and cap32/replay32 measured `0.6402 ms` vs `ll-masked` `0.6545 ms`, both with graph correctness passing.

Skew-safe capacity design finding:
- DeepEP normal's `num_worst_tokens = num_tokens * num_ranks` test pattern is the right correctness reference: capacity is provisioned for worst-case receive count, while actual returned rows remain the real dispatch prefix.
- DCU MegaMoE normal can follow a similar design with smaller performance impact than LL because compact prebuild already emits `tile_experts` and `active_tiles`. If the tile pool is sized for local-rank worst-case route rows (`num_ranks * capacity_tokens * topk`), one expert can occupy most or all compact tiles without clipping, while ASM CTAs beyond `active_tiles` early-exit.
- DCU MegaMoE LL currently uses fixed per-expert `m_per_expert`, so true no-drop correctness requires `m_per_expert >= num_ranks * capacity_tokens`. To keep performance acceptable, all later stages must continue using `actual_m` / max active rows and avoid full-capacity init/copy/reduce where possible.
- Pro split `ll_pro_masked` is the most performance-sensitive path because increasing `m_per_expert` can change DeepGEMM masked ASM `size_m`/`num_MBlocks`. If broad worst-case capacity regresses this path, the durable solution should be compact stage-only metadata for masked K1 rather than overflow fallback.
- 151.1 has a current runtime library pitfall unrelated to MegaMoE code: default Python import of `torch` fails through `/opt/hyhal/lib/libamd_smi.so`. A run-local LD_LIBRARY_PATH filter plus `/root/yuguo/dtk-26.04.1/.hyhal/rocm_smi/lib` restores `torch` import and should be used in validation commands until the container environment is cleaned.
- Implemented design adjustment: normal now defaults to compact prebuild and sizes the compact tile pool with a local-rank worst-case/top-k-unique bound; LL sizes per-expert rows to `num_ranks * capacity_tokens`. K2/K3 normal receive active-tile metadata so enlarged correctness capacity does not force full invalid-row activation/GEMM work.
- Memory-control finding: a single route_scratch buffer previously represented both normal and LL layout capacity. Full LL worst-case rows for a large normal bucket would over-allocate scratch. The fix records `ll_scratch_capacity_tokens_per_rank` on `SymmBuffer`: default covers the auto-LL threshold (512 unless env changes it), explicit larger graph caps can opt in, and forced LL beyond the cap raises a Python configuration error instead of relying on overflow fallback.
- 151.1 validation proved the legal extreme top-k route pattern now passes against both baseline families. Flash EP8 normal, tokens=128, all ranks routing top-k into rank0 local experts, passed against `normal-contiguous` with `max_abs=6.10352e-05`. Flash EP8 LL with the same pattern passed against `ll-masked` with `max_abs=0.000244141`.
- The first LL retry failure was not a MegaMoE correctness failure: DeepEP `ll-masked` baseline initialization required `ROCSHMEM_HEAP_SIZE`/`DUSHMEM_HEAP_SIZE` larger than the reported `num_rdma_bytes(1090523264)`. The historical 151.1 LL env (`ROCSHMEM_HEAP_SIZE=4737418240`, `DUSHMEM_HEAP_SIZE=4737418240`, MNNVL/LL context variables, `MEGAMOE_DCU_PEER_MEMORY=rpc`) resolved it.
- Active-only follow-up finding: the main remaining LL performance risk after worst-capacity sizing was not K2/K3 GEMM math itself; it was extra capacity-proportional bookkeeping. K1 still cleared per-row metadata/scales across `kExperts * m_per_expert`, and K3 split-tail only used `max(actual_m)` for the tiny `<=16` copy-block case. The low-intrusion fix is to keep worst-capacity buffers but initialize/copy only actual rows plus active tile padding, leaving a future compact-LL rewrite only if Pro LL performance still regresses.
- Flash EP8 LL cap512 triage after the first active-only patch showed correctness was clean but performance was not acceptable: graph replay `8/32/64/128/256/512` moved to `1.0037/1.1033/1.1262/1.1996/1.4770/2.2573 ms`, a material regression from the earlier `0.5483/0.6529/0.6983/0.7588/1.0272/1.8229 ms` guardrail. The root cause is not just work inside K3; the host still launched split-tail copy CTAs for the full worst `rows_per_expert` capacity, so random buckets paid for many empty CTAs before early-return.
- The next low-intrusion K3 fix is a bounded CTA-pool loop: host launch uses a historical-sized copy pool (`256` rows/expert for Flash, `128` for Pro) plus reduce blocks, while the kernel reads `max(actual_m)` and grid-stride loops over every active copy block. Random distributions avoid full worst-capacity empty launches; extreme skew remains correct because the pool loops until all active blocks finish and the existing done-counter still targets `active_copy_blocks`.
- The K3 CTA-pool fix is validated on 151.1. Flash EP8 LL graph cap512 after rebuild measured `0.5583/0.6684/0.7029/0.7792/1.0462/1.8411 ms` for replay `8/32/64/128/256/512`, essentially back to the historical post-guard band, and the adversarial skew LL route pattern still passed. This resolves the Flash full-capacity launch regression without giving up exact worst-capacity correctness.
- Pro default split remains the unresolved performance problem. The exact-capacity path sets Pro EP8 cap512 `rows_per_expert` to `4096`, so the bundled masked K1 ASM sees `size_m=4096` and `num_MBlocks=64`. The original fast Pro split path used small logical M values such as `128`, which kept masked K1 work proportional to active rows.
- The masked K1 ASM wrapper cannot be fixed by a PyTorch non-contiguous view. The argument block has strides for `input_scale`, but not for the FP8 `ptr_B` input or BF16 `ptr_C` output. DeepGEMM's reference wrapper also uses `input.size(1)` as both `size_m` and the physical expert stride. Therefore a small logical M with a separate large physical expert stride is not expressible without changing the code object ABI.
- Pro `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` remains a correctness/compatibility path, not the performance replacement. After exact capacity and K3 pool it is correct but slower than the default split path on Pro EP8 cap512 (`2.3846/3.0920/3.1654/3.2647/3.4464/6.0613 ms`), so switching Pro default to unified would hide the K1 stride problem rather than solve it.
- The next real fix is compact active-K1 for Pro LL split. The design should keep the route builder's worst-capacity storage so legal extreme skew has space, then compact the active rows that feed masked K1/K2/K3 into a smaller active layout. This is distinct from an overflow fallback: no valid rows should be dropped, and the ordinary random case should recover the historical masked K1 `size_m` while the skew case remains exact even if it needs more active rows.
- The first compact-head implementation is correctness-preserving but not performance-acceptable. It keeps the worst-capacity fixed layout for downstream K2/K3, compacts only the masked-K1 head rows, copies compact head output back, and runs an offset masked-K1 tail for rows beyond the head. Pro EP8 LL cap512 remained correct, but graph replay moved to `2.2771/2.7466/2.8233/2.9797/3.2819/4.7923 ms`, slower than the exact split run (`2.1586/2.6249/2.6943/2.8541/3.1735/4.6519 ms`) and far from the historical pre-exact target (`1.4357/1.9045/1.9807/2.1098/2.4462/3.9351 ms`).
- Therefore compact-head should stay behind `MEGAMOE_DCU_PRO_LL_COMPACT_HEAD=1` and remain an ablation tool. The next Pro fix must remove the copy-back and mostly-empty tail-launch costs, or change the masked K1/K2/K3 contract to consume a true compact layout. Simply splitting head/tail while preserving the fixed worst-capacity downstream layout is not enough.
- Pro LL graph capture policy matters for performance fairness. With exact capacity, a single cap512 graph for all replay buckets makes small buckets pay max-capacity work in MegaMoE. Capturing MegaMoE per bucket while keeping baseline dispatch cap aligned to runtime tokens restored Pro EP8 graph medians to `1.4381/1.9296/2.0488/2.2953/2.7740/4.6565 ms`, faster than the aligned `ll-masked` baseline on every checked bucket.
- Normal exact capacity is not performance-safe yet. Flash EP8 normal eager 4096 regressed from the historical same-node `5.7636 ms` band to `17.2707 ms` after exact compact capacity; a K1-only active-launch patch reduced it only to `16.1966 ms`. The regression is therefore not mainly K1 launch waste.
- The next Normal bottleneck hypothesis is K3 host launch/done-target capacity scaling. Current `k3_fused_ext.cu` uses `total_rows` for `prob.n`, `wg_n`, and global work items, and `opt.py` computes `asm_done_target` from capacity `rows`. Because K3 tail-reduce waits for `asm_done_target`, eager active launch must adjust both launch rows and done target together, while graph capture needs a capture-safe approach later.

Normal exact-capacity performance finding:
- The Normal exact compact-capacity correctness fix is now performance-acceptable for eager Flash EP8 on 151.1. The final 4096 recheck measured `5.9164 ms` fused versus `9.7064 ms` baseline, about `+2.7%` from the historical same-node `5.7636 ms` MegaMoE reference and much better than the initial exact-capacity regression (`17.2707 ms`).
- The root cause of the large Normal regression was capacity-proportional host work after enlarging the compact tile pool, not extra useful GEMM math. K1 active launch alone barely moved the needle, but K3 active launch plus done-target correction cut 4096 to `6.6010 ms`, and K2 active CTA pooling brought it to the final `~5.9 ms` band.
- The accepted Normal design remains baseline-like: workspace capacity is worst-case enough for a legal expert-skew distribution, while active route metadata drives K1/K2/K3 work in the ordinary random case. No overflow fallback is part of the correctness contract.
- The rejected threshold experiment is a useful caution. Avoiding the active-tile host readback for small capacity buckets looked attractive, but it reintroduced too much empty capacity work (`512 -> 3.327 ms`, `1024 -> 5.086 ms`) and destabilized the matrix run. The current unconditional eager active-readback is the retained low-risk path.
- Current limitation: K1/K3 active host readback is intentionally disabled for stream capture and graph runtime offset cases, so this specifically resolves Normal eager. If Normal graph later needs exact compact capacity plus large caps, the safer next design is a capture-compatible fixed CTA pool or runtime active-tile consumer, not a host readback inside graph capture.

Pro LL compact-active / skew finding:
- The eager-only compact-active path is the right performance direction for Pro LL random buckets, but it is not yet the final correctness story. It improved Pro EP8 LL eager token256/token512 from exact-worst `2.7449/4.6322 ms` to `2.6022/4.0952 ms`, while preserving random correctness against `ll-masked`.
- Graph capture remains on per-bucket exact worst-capacity because compact-active currently reads `actual_m` on the host. This is deliberate and keeps graph replay in the already-recovered band; a future graph compact design needs a device-side active-row consumer or per-bucket compact capture, not a host read inside capture.
- Pro adversarial LL skew now exposes a common LL correctness bug independent of compact-active. The same `single-local-rank` failure reproduces with compact-active disabled, with split-tail disabled, with tail-reduce disabled, and on Pro unified LL. Therefore the next fix should trace K1 stage/K2/K3 production for the first bad route, not patch the compact-active performance path blindly.
- The current most useful diagnostic boundary is before the final reduce: tail-reduce-off still fails and the error scale is `~0.10`, far above BF16 tolerance. The next evidence to collect is whether the bad route's staged row, masked/unified K1 output, K2 activation, or K3 combine slot first diverges from the `ll-masked` oracle.
- Scratch layout changes can mask the issue for specific row counts, so pass/fail at one HOT_M value is not enough. A valid fix must pass the adversarial Pro EP8 route sweep with compact-active on and off, plus the existing Flash LL skew guardrail.
- Pro masked-K1 launch mapping must be treated as code-object-specific. The default packaged `.co` maps expert/N from persistent `workgroup_x`; a naive 2D launch with x=`num_NBlocks` only covers expert0 and recreates the dropped-slot symptom. The scratch balanced `.co` uses an active tile scheduler but expects a different weight/storage layout, so it is not a drop-in production fix.
- Latest trace after restoring the default `.co` shows all six adversarial slots are produced for rank0 token44, but the final result is still off by `max_abs ~= 0.098`. The active bug is now a Pro split numeric/layout mismatch versus `ll-masked`, not a simple overflow or missing-combine-slot fallback problem.

Normal graph exact-capacity finding:
- Normal graph is correctness-safe after the exact compact-capacity fix. The 151.1 Flash EP8 graph check at token4096 passed with graph bucket `max_abs=0.000488281` and `mean_abs=9.37182e-06` against `normal-contiguous`.
- Normal graph performance is not yet acceptable relative to the recovered eager path. Graph replay-only median measured `6.9654 ms`, while the current same-shape Normal eager guardrail is `5.9164 ms`, about `+17.7%` slower.
- This is an implementation-side work inflation, not a precision/capacity failure. Eager uses a host readback of `active_tiles` to shrink K1/K3/K2 launches after metadata build; graph capture cannot do that D2H/synchronize and instead keeps fixed worst-capacity launch dimensions with device-side early exits.
- The next graph fix should therefore decouple graph launch dimensions from worst-capacity rows/tiles using a capture-compatible fixed CTA pool or device-side active-tile consumer. Adding a D2H `active_tiles_host` path to graph would break capture semantics and is not the right direction.

Pro LL skew K1-stage visibility finding:
- The remaining Pro EP8 LL adversarial skew bug is not explained by overflow, final local reduce, split-tail copy, or masked-K1 launch mapping alone. With compact-active disabled and tail/split-tail ablations, failing route slots can still have valid row metadata but zero/default staged data before K1 GEMM.
- The most precise failing signature is: active route row, valid `row_combine_ptr`, correct partial token/slot, but `staged_x` is zero and `staged_x_scale` is the padding default. Downstream L1/K3/combine then stays zero. This places the first bad boundary in the K1 route/stage builder's metadata handoff.
- System-scope count publication improves but does not fully solve the issue, so per-row metadata is now the likely remaining visibility gap. The next correctness patch should make `symm_src_x_ptrs[row]` a release/acquire handoff between route assignment and stage-copy consumers.
- Performance risk to watch: acquiring the source pointer inside the current vector-granular stage-copy loop may add overhead. If correctness passes but random Pro LL performance regresses, optimize the acquire placement or use a cheaper source-backed cache-invalidation/load pattern rather than reverting to capacity-proportional full initialization.

Pro LL skew root-cause finding:
- The actual retained root cause is stride/padding, not source-pointer visibility. Exact LL can choose a per-expert stride that is not a multiple of the Pro stage block M. For Pro EP8 skew, `m_per_expert=512` and `kBlockM=48`, so the old stage copy rounded active rows to `528` and wrote padding past the expert stride.
- Because the physical layout is `[expert, m_per_expert]`, writing row_in `512..527` for expert N aliases row_in `0..15` for expert N+1. This explains why low row_in routes in slots 1..5 were zero while higher row_in routes such as token28 row28 remained correct.
- The correct low-intrusion fix is to clamp stage-copy padding to the physical expert stride: `stage_rows = min(m_per_expert, round_up(expert_count, kBlockM))`. This preserves worst-capacity correctness and active-only behavior without overflow fallback, extra kernels, or performance-heavy synchronization.
- Diagnostic synchronization changes are not required for this bug and should not stay in the final performance path unless future evidence demands them.

LL final performance finding:
- The retained stage-clamp fix has negligible measured performance impact. Pro random eager stays in the same compact-active band as before the root-cause cleanup, Pro graph cap512 stays in the per-bucket recovered band, and Flash graph cap512 remains at or better than the previous guardrail numbers.
- The final LL fix therefore satisfies the intended design: worst-case capacity is real and exact, active/random work remains proportional to actual rows, and the correctness fix does not add hot-path kernels or capacity-proportional initialization.

Temporary cleanup finding:
- The remaining temporary diagnostics after the final LL fix were in two buckets: an explicit failure-detail dump in `test_mega_moe_dcu.py`, and the rejected compact-head Pro LL ablation. Both are now removed from runnable code.
- Compact-head should not remain as an env-gated production ablation in this branch. It was correctness-preserving but slower because it kept fixed worst-capacity downstream layout and paid extra copy-back / offset-GEMM costs. The stable Pro LL path is compact-active for eager random buckets and exact per-bucket capacity for graph.
- Compact-active does not need the old head/tail count split. After cleanup it uses one compact count vector (`pro_compact_m`), reducing scratch bookkeeping and removing misleading names left over from the ablation.
- The precision/correctness risk addressed by this work is specifically the legal extreme-routing overflow/clipping/stride class. Normal no longer clips compact tiles under local-rank skew; LL no longer depends on mean/slack rows as a correctness guarantee; Pro LL stage-copy padding is clamped to physical expert stride. The retained regression evidence is Pro EP8 LL adversarial token128 passing after cleanup plus the earlier Flash/Pro skew matrix.
- A separate 151.1 environment caveat remains: Python `torch` import can pick `/opt/hyhal/lib/libamd_smi.so` and fail on missing `amdsmi_init`. Validation commands should prepend `/usr/local/lib/python3.10/dist-packages/amdsmi` to `LD_LIBRARY_PATH` until the container base environment is cleaned.

EP16 final-retake finding:
- The final retained LL fix also passes Pro EP16. The legal extreme route pattern `single-local-rank` passed for token64 and token128 against `ll-masked`, both with `max_abs=0.000488281`, and graph cap512 passed every checked replay bucket `8/32/64/128/256/512`.
- Compact-active is a performance/work-shaping path, not an overflow fallback. Correctness comes from exact worst-capacity rows plus the Pro stage-copy stride clamp; compact-active keeps eager random Pro LL from paying full worst-capacity work. Graph currently stays on exact per-bucket capacity because this compact-active path needs host-visible active counts.
- The compact-active env was removed after default-on validation. There is no retained Pro LL compact-active user switch; eager Pro LL uses the compact-active path directly, while the rejected compact-head experiment env and source-level debug trace/failure envs remain absent from runnable sources.
- A true single cap512 graph replaying smaller runtime tokens is still correctness-safe, but it is not performance-equivalent to the pre-exact-capacity graph path for small/medium buckets. Pro EP16 single-capture replay measured `1.8298/1.9214/2.0112/2.1938/3.1099/4.8275 ms` for `8/32/64/128/256/512`; the small buckets are dominated by fixed capture-capacity work. Matching the old single-graph small-bucket performance would require graph-compatible active-work/compact consumption rather than another host-side active-count path.
- Caveat: the first single-capture comparison used a baseline graph path still aligned to per-bucket dispatch capacity. After correcting the harness to pass cap512 to the `ll-masked` baseline under `--cuda-graph-single-capture`, the attempted EP16 rerun hung hard enough that 151.1 stopped responding to SSH and ping. Therefore there is no fair baseline-cap512 number yet; this mode needs recovery-first triage before more testing.
- API documentation note: `capacity_num_tokens` now has two explicit meanings. In eager mode it is the current request capacity bound, usually the EP-group maximum local token count. In graph mode it is the capture capacity for the selected LL or normal graph, while replay-time valid tokens remain in `sym_buffer.cuda_graph_num_tokens`.

Graph performance recovery finding:
- The post-fix graph/eager gaps are reproduced on current rebuilt 151.1 artifacts: Flash EP8 Normal token4096 graph replay is `6.9821 ms` versus eager `6.0118 ms`; Pro EP8 LL cap512 graph replay512 is `4.6485 ms` versus eager `4.1925 ms`.
- Disabling Normal ASM tail-reduce is not the recovery path: token4096 graph replay moved to `7.0714 ms`, slower than the default `6.9821 ms`.
- Disabling LL split-tail is not the recovery path: Pro EP8 LL graph replay512 moved to `5.2302 ms`, slower than the default `4.6485 ms`.
- For Pro LL graph, the obvious masked-K1 shortcut is unsafe. The exact graph path uses physical `size_m=4096` for Pro EP8 cap512, so the bundled masked K1 ASM sees `num_MBlocks=64`; eager compact-active uses compact `size_m=128` and `num_MBlocks=2`. However, setting graph `num_MBlocks` from `expected_m_per_group` or the compact active bound would silently skip legal skew rows above that bound. This would reintroduce the exact precision hazard the worst-capacity fix removed.
- A safe Pro LL graph optimization must make the M-block scheduler dynamic from device-side `actual_m/max_count`, or move the whole LL graph path to a true compact active layout consumed by masked K1, K2, and K3. A host D2H readback or a fixed expected-M clamp is not acceptable for graph correctness.
- For Normal graph, the remaining safe direction is still a capture-compatible active-work consumer or CTA pool inside K1/K3, not D2H `active_tiles_host`. Current ASM gates inactive row tiles but still launches the captured capacity grid, so the empty-CTA cost remains on graph replay.
- `hipprof --stats` confirms the source-level diagnosis. In the Normal graph token4096 profile, compact route init/count/build/emit kernels are small, while K1 normal ASM and K3 normal ASM dominate the MegaMoE graph work. In the Pro EP8 LL graph token512 profile, Pro masked K1 is the main exact-stride-specific target; K1 stage-only, K3 split/combine, K3 LL GEMM, and K2 are already active-row driven or not unique to graph's exact-stride overhead.
- Therefore the next performance branch should not start with Python/test harness changes. It should either modify the Normal/K3 ASM work scheduling so a fixed captured CTA pool consumes `active_tiles`, or modify the Pro masked K1 code-object/ABI so physical expert stride and scheduled M-block count are decoupled by a device-side max-count. If that is too invasive, leave graph as correctness-safe and keep production on eager for Normal and per-bucket graph for LL.
- The previously missing fair baseline-cap512 single-capture data is now collected for Pro EP16 LL. With both MegaMoE and `ll-masked` baseline captured at cap512, replay medians for `8/32/64/128/256/512` were MegaMoE `1.8216/1.9240/2.0016/2.2017/3.1074/4.8633 ms` versus baseline `2.5676/2.6352/2.6780/2.8035/3.5800/5.1097 ms`. This confirms the old per-bucket baseline numbers understated the fixed-capacity cost for small buckets; under a fair cap512 baseline, MegaMoE remains faster for every checked bucket.

Graph optimization correction finding:
- The packaged Pro masked K1 `.co` is not the simple fixed-M scheduler implied by the first graph-gap hypothesis. Current disassembly of hash `73184662ec644cf9f4e9cfacec720a15428e84c5f84ad06e6e9e57bfa06543b4` shows a `masked_m` load and `m_block * 64 >= masked_m[expert]` early exit inside the persistent task loop.
- A historical scratch `deepgemm_groupgemm_masked_fp8_marlin_balanced_256x64x128_TN_BF16_WGM8.s` exists on 151.1, but it is not a drop-in source replacement for the packaged `.co`: it was previously tested with a different weight/storage layout. Do not wire it into setup or replace the prebuilt production object.
- The Pro LL graph/eager delta should now be framed as a stride/layout problem rather than missing actual-M M-block scheduling. Current ABI exposes `size_m` as the physical per-expert stride for input/output tensors; even if compute skips inactive M-blocks, a compact graph path still needs either a layout-compatible ABI extension or downstream K2/K3 compact-row consumption.
- A masked-K1-only stride microbench weakens the idea that physical `size_m` alone is the dominant remaining Pro LL graph cost. With fixed active rows, `size_m=4096` was only about `0.6%` slower than `size_m=128` at `E=48`, and about `1.7%` slower at `E=12`. The next graph investigation should therefore profile the whole captured Pro LL exact path and compare stage-only/K2/K3 behavior before touching the packaged `.co`.
- Expected-M clamping remains rejected. It may reduce scheduled M blocks for random graph buckets, but it can skip legal skew rows above the clamp and reintroduce the precision hazard fixed by worst-capacity rows.

Pro LL graph K2 finding:
- The largest Pro LL graph-only regression after the exact-capacity fix was K2, not masked K1. Pro `hidden=3072` uses the generic K2 kernel, and that launch path ignored `max_row_blocks` by launching `dim3(rows)`. With exact graph capacity, this meant `196608` CTAs for EP8 cap512 instead of an active-row pool.
- The retained K2 fix makes the generic kernel match the 2048/4096 register kernels: launch a fixed CTA pool and grid-stride across `effective_rows`. It is correctness-safe for skew because the loop covers all logical active rows even when active rows exceed the CTA pool.
- This is not an overflow fallback and does not change capacity sizing. It only removes empty graph CTA work for Pro hidden=3072. The post-fix EP8 graph token512 replay improved from `4.6485 ms` to `3.9561 ms`, and the K2 kernel profile dropped from `~0.797 ms/call` to `~0.099 ms/call`.
- EP16 benefits from the same hidden=3072 generic-K2 fix. With IPC peer mode on 151.1, all EP16 graph buckets passed against `ll-masked`, and MegaMoE stayed faster than baseline on every checked bucket: `1.103/1.216/1.301/1.495/2.406/4.178 ms` versus `1.147/1.278/1.405/1.709/2.840/5.118 ms`.

Pro LL graph K2 CTA-pool finding:
- The main recoverable Pro LL graph regression was K2 generic launch waste, not masked-K1 M-block scheduling. Pro hidden `3072` uses the generic `swiglu_quant_channelwise_kernel`, and that branch ignored the precomputed `launch_blocks` by launching one CTA per physical row.
- The fix is correctness-safe because it does not clamp rows or change capacity. It launches a fixed CTA pool and grid-strides through `effective_rows`, which is derived from device-side `actual_m/max_m`; skew rows beyond the first pool are still processed by later loop iterations.
- Measured impact is large and favorable. Pro EP8 LL graph replay512 improved from `4.6485 ms` to `3.9561 ms`, and EP16 single cap512 small buckets improved from roughly `1.82/1.92/2.00/2.20/3.11/4.86 ms` to `1.13/1.22/1.31/1.50/2.41/4.14 ms` for `8/32/64/128/256/512`.
- Remaining Pro LL graph hotspots after this fix are K3 combine/reduce, K3 LL GEMM, masked K1, and K1 stage-only. K2 is no longer the dominant graph/eager gap.

Normal graph post-K2-pool finding:
- Flash EP8 Normal token4096 does not materially benefit from the K2 generic CTA-pool fix. The post-fix same-run numbers were graph replay `6.9424 ms` versus eager `5.9765 ms`, compared with the earlier `6.9821 ms` versus `6.0118 ms`.
- This matches the source expectation: Flash Normal uses K2 `hidden=2048`, which already had the register-kernel CTA-pool path before the generic K2 fix. The remaining Normal graph gap is still K1/K3 normal ASM capacity-grid early-exit overhead, not K2 launch waste.
- Pro Normal may still be the normal-backend case that benefits from K2 generic pooling because it uses `intermediate=3072`, but the attempted Pro EP8 Normal graph512 probe could not complete on 151.1 due to HIP OOM during baseline weight packing and persistent `86%` VRAM reporting with no visible KFD PIDs. Do not infer a Pro Normal graph result from that failed run.

## 2026-07-14 - Flash LL Optimization Restart Constraints

- The user reports that framework-integrated Flash decode currently shows no gain, so end-to-end LL latency—not isolated GEMM throughput—is the optimization objective.
- Flash EP8/EP16 with `ll-masked` is the approved fast validation surface. Primary decode buckets are `8,32,64,128,256,512`; per-bucket graph replay is the first timing surface because it reflects the existing LL graph design without single-capacity padding bias.
- Current source already contains the prior active-work recoveries: LL K1 active-only initialization, K3 split-tail active copy/CTA pooling, and K2 fixed CTA pooling. Repeating those ideas without new profile evidence would not be a new optimization.
- The next candidate must be chosen from fresh current-artifact profiling. Likely attribution buckets are K1 stage/groupgemm, K3 local GEMM, K3 split combine/reduce, rank/barrier/signal protocol, and launch overhead; this is a hypothesis list, not a conclusion.
- Correctness remains non-negotiable: no expected-M clamp, no graph host readback, no capacity crop, and no optimization that only passes random routes while failing `single-local-rank` skew.
- Existing uncommitted `task_plan.md` changes only repair mojibake/checkmark characters. Preserve them separately from new kernel work.

### Initial DCU RAG retrieval

- `dcu-rag-kb` resolved its Hygon sources under `D:\git\geak_dcu\mcp_tools\rag-mcp\knowledge-base\amd-knowledge-base\layer-6-extended\hygon-extend` and returned source-backed gfx938/DeepEP references.
- The first optimize query did not return a direct MegaMoE K1/K3 recipe. Its useful Hygon-specific guidance is narrower: keep hot shapes specialized, treat combine as an independently tunable stage, use fixed low-latency layouts/double buffering, overlap at chunk granularity with explicit readiness signals, and compile quantization into the transfer/stage path when message movement dominates.
- The retrieved LDS microbenchmark evidence explicitly pairs `ds_read_*` consumers with `s_waitcnt lgkmcnt(0)`. Any waitcnt removal/reordering in K1/K3 therefore requires generated-ISA and correctness proof; it is not a safe source-only cleanup.
- The KB also points to gfx938 native FP8 `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts`, but the current production kernels already use `v_mmac`/pack5 paths. This is evidence for ISA verification, not justification to rewrite the GEMM core before profiling.
- Relevant original sources to inspect next are the Hygon DeepEP low-latency report plus `DeepEP-main/csrc/kernels/internode_ll.cu`, the gfx938 builtin reference/source sheet, and the LDS microbenchmark sources. These are supporting patterns; current MegaMoE source and 151.1 profile remain authoritative.

### Optimizer method/metrics constraints

- The optimizer catalog confirms that the current Flash LL GEMM path already satisfies the high-priority MMAC/FP8 direction; a rewrite to another builtin is not justified unless `SQ_INSTS_MMOP` or final ISA unexpectedly shows the production kernel is not using `v_mmac`.
- For the expected small/irregular decode workload, the first evidence-driven method candidates are `compute.launch_config_wave64`, `compute.register_pressure_control`, `latency.waitcnt_pipeline`, `latency.reduce_barrier`, `latency.persistent_scheduler`, and possibly `memory.vectorized_global_access`/`memory.epilogue_fusion`. None is selected until current profile/resource evidence triggers it.
- Required evidence mapping is now explicit: low waves/CU activity for launch/persistent changes; high VGPR/SGPR or low occupancy for register control; clustered waits/SQTT stalls for waitcnt movement; barrier count/stalls for sync reduction; packed load/store ISA plus request counters for vectorization; fewer launches/global round trips for fusion.
- Final proof for low-level changes must use `dccobjdump`: `v_mmac_*`, vector global/buffer operations, `ds_read*`, `s_waitcnt vmcnt/lgkmcnt`, `s_barrier`, and resource usage. Missing or empty dumps are inconclusive rather than proof of failure.

### DeepEP original-source comparison

- Deep retrieval reached the original Hygon DeepEP `internode_ll.cu`, not only the KB summary. Its strongest relevant pattern is chunk-granular combine readiness: `combine_sbo` maps `packed_recv_count` into `block_m` chunks, waits on a per-expert/per-chunk `comp_signal`, sends ready token chunks, and keeps clean/finish counters explicit.
- DeepEP also uses shape/quant compile-time specialization, fixed low-latency layouts, double buffers, and explicit send/receive phases. These are architectural patterns rather than direct MegaMoE patches.
- The source contains real system/workgroup-scope atomic and readiness protocols plus block-wide barriers. It reinforces that signal/barrier removal is high risk and must be justified by current MegaMoE profile plus scope analysis.
- Current MegaMoE K3 already has split-tail chunk signals and active copy blocks, so the novel question is not "add chunking" but whether its current chunk size, worker assignment, signal publication/wait, or phase coupling leaves measurable decode bubbles. Fresh K3 profile/ISA must answer that before changes.
- Framework-side delayed receive hooks could matter to the user's reported no-gain decode result, but this round remains kernel-first as requested. If isolated MegaMoE beats `ll-masked` while framework decode still does not, overlap/control-plane integration becomes a separate follow-up rather than evidence that the kernel branch failed.

### 151.1 pre-baseline verification

- SHA256 matches between local workspace and `/root/yuguo/DeepGEMM` for `opt.py`, K1/K3 LL implementation headers, K2 extension source, and the integrated test harness. The remote mapped source is therefore current for the hot paths under study.
- Existing source-tree extensions were built on 2026-07-08 18:12-18:13, after the retained K2 pool change. `hipprof` and `dccobjdump` resolve from `/root/yuguo/dtk-26.04.1/bin`.
- Before the baseline, all 16 HCUs reported `VRAM=0%`, `HCU=0%`, Normal mode, and `hy-smi --showpids` reported no KFD PIDs. The previous persistent-86%-VRAM incident is not present now.

### Flash LL fast-validation contract

- The fair fast matrix is Flash `hidden=4096`, `intermediate=2048`, `experts=256`, `topk=6`, explicit `--megamoe-backend ll --baseline-kind ll-masked --seed 1234`, and default per-bucket graph capture for tokens `8,32,64,128,256,512`.
- `--skip-bench` only removes the top-level eager benchmark. Graph-bucket correctness and graph replay timing still execute, so each JSON contains a scope-aligned MegaMoE-graph versus `ll-masked`-graph pair. `--cuda-graph-skip-baseline` is profile-only and must not be used for speedup claims.
- Branch selection starts with EP8 (`HIP_VISIBLE_DEVICES=0..7`) using graph warmup 1/replays 3 and test warmup 3/repeat 10. Only promising branches move to EP16 (`0..15`); the champion is rerun with 20 replays in at least two clean trials.
- 151.1's framework-aligned Flash LL transport contract is RPC: `MEGAMOE_DCU_PEER_MEMORY=rpc` plus the retained ROCSHMEM/DUSHMEM low-latency environment. IPC and RPC results must never be mixed in one A/B comparison.
- Every candidate keeps random correctness in the paired graph run and adds EP8 `single-local-rank` token128. A champion also runs EP16 skew. Any route/signal/barrier change adds uneven-rank coverage. Historical Flash skew errors were about `0.000244141`, so merely fitting the broad `atol=0.0035` is not sufficient evidence.
- Historical guardrails are EP8 fused `0.4989/0.6451/0.6853/0.7688/1.0380/1.8168 ms` and EP16 fused `0.3732/0.4078/0.4771/0.5932/0.9963/1.8817 ms` for `8/32/64/128/256/512`. These are only drift checks; optimization claims must use same-run paired JSON because old runs used different capacity/capture policies.
- The README phrase describing one captured graph is stale for its shown command: without `--cuda-graph-single-capture`, the current harness captures one graph per token bucket. The current round intentionally keeps per-bucket capture.

### K3 batch3 adjacent retake

- The isolated 3-load/1-wait reducer branch is repeatably positive, but the >=2% evidence is bucket-specific. The first 20-replay comparison gave `2.42%/2.02%/2.08%` at tokens `64/128/256`; the immediately adjacent original-to-candidate retake gave `2.46%/1.71%/1.27%`. Token64 is the only bucket currently above the 2% gate in both trials; no tested bucket regressed.
- Do not claim a broad 2% win yet. Run a focused higher-replay original/candidate A/B for tokens `64/128/256`; retain only if the branch remains precision-clean and its middle-bucket benefit survives the stricter noise check.
- The 100-replay retake confirms the precise interpretation: batch3 is a repeatable token64 optimization (`+2.43%`) and a smaller positive token128/token256 optimization (`+1.79%/+1.02%`), not a broad >=2% middle-bucket win. Because every tested bucket remains non-regressing and the compiled resource tuple is unchanged, advance it to skew and EP16 guardrails while keeping the claim bucket-specific.
- EP16 invalidates the current batch3 implementation despite its EP8 gains: the adjacent original binary passes token512, while batch3 VM-faults there. Source review points to a concrete inline-assembly hazard to verify: the multi-instruction block uses ordinary `=v` outputs, so the compiler may overlap an output with a later address input even though the first load writes that output before the later address is consumed. The outputs likely need early-clobber constraints (`=&v`). Treat this as a hypothesis until ISA/compile evidence and a fixed retest confirm it.
- The project itself confirms the correct constraint convention: K1/K3 multi-load helpers already mark their vector results `=&v`. The batch3 helper now follows that convention. Validate the fix first on EP16 token512; only then repeat performance, skew, and resource/ISA checks because early-clobber can change allocation pressure.
- The fixed EP16 token512 pass strongly supports the register-overlap diagnosis, but does not yet prove the optimization remains worthwhile. Early-clobber constrains allocation and may consume more live VGPRs; all fixed-binary EP8/EP16 performance and resource measurements must replace the pre-fix data.
- Fixed batch3 retains useful performance: EP8 remains centered on token64 (`+2.61%`, token128 `+1.99%`), while EP16 shows a broader middle-bucket effect at token32/64/128 (`+2.14%/+3.98%/+3.60%`). The candidate is now correctness-safe on the previously failing EP16/512 bucket, but an adjacent higher-replay EP16 original/candidate run is still required before retention.
- The high-replay EP16 retake narrows the defensible claim: token64 is repeatably improved (`+3.70%`), while token32/token128 are smaller positives (`+1.25%/+1.02%`) and the earlier >2% readings were not stable. Together with the EP8 high-replay `+2.43%` token64 result, the branch clears the noise gate specifically for token64 on both EP sizes and is non-regressing outside a sub-noise EP8 token512 reading.
- Final compiler assembly validates both intended scheduling and the safety fix: each topk6 half issues three vector loads before one VMEM wait, and the destination/address VGPR ranges are disjoint. This is stronger evidence than the degraded `dccobjdump` rendering and closes the ISA-sequence acceptance criterion.
- High-sample profiling closes the method attribution: batching three reducer loads before one wait lowers the instrumented combine/reduce kernel average by `8.01%` at EP8 token64, while the adjacent K3 local GEMM changes by under 1%. End-to-end gains are smaller because combine/reduce is only one stage of the LL graph.
- Against the requested same-run `ll-masked` baseline, the retained fixed binary wins all six full-matrix buckets. EP8 gains for tokens `8/32/64/128/256/512` are `17.56%/12.34%/14.49%/16.15%/9.28%/12.87%`; EP16 gains are `19.10%/16.24%/12.45%/11.93%/13.49%/4.66%`. These quantify the full MegaMoE LL path versus masked LL; incremental attribution versus the saved original K3 remains the narrower token64 result.

### Fresh Flash EP8 LL baseline (2026-07-14)

- A clean 151.1 EP8 run using the frozen RPC/per-bucket contract passed the eager correctness check and all six graph-bucket checks. Random-route max error was `0.000488281`; graph-bucket max error was `0.000244141` through token128 and `0.000549316` at token256/512.
- Same-run graph medians for MegaMoE LL were `0.5050/0.6549/0.6946/0.7757/1.0664/1.8218 ms` for tokens `8/32/64/128/256/512`.
- The paired `ll-masked` graph medians were `0.6081/0.7382/0.7908/0.9108/1.1587/2.1204 ms`. Baseline/LL speedups are therefore about `1.204x/1.127x/1.139x/1.174x/1.087x/1.164x`; LL wins every isolated bucket, but token256 has the smallest margin.
- The result is inside the prior Flash EP8 stable band, so the current artifact is suitable for branch comparison. It also separates two questions: isolated MegaMoE LL still has a real paired advantage, while the user's missing framework decode gain likely includes integration/overlap/control-plane effects outside this kernel-only timing scope.
- Run directory: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_flash_ll_opt/ep8_baseline_20260714_114627`.

### Framework decode data supplied by the user

- Framework TPOT uses input lengths 4096 and 16384, output length 1024, and max concurrency `1,2,4,8,16,32,64`. For input 4096, DeepEP versus MegaMoe TPOT is `20.16/20.66`, `20.85/22.54`, `22.49/21.85`, `23.99/23.75`, `24.81/24.39`, `29.52/26.39`, and `28.95/30.35 ms`, corresponding to reported changes `-2.48%/-8.11%/+2.85%/+1.00%/+1.69%/+10.60%/-4.84%`.
- For input 16384, DeepEP versus MegaMoe TPOT is `21.25/20.04`, `20.87/21.84`, `22.36/22.10`, `22.19/23.55`, `26.31/23.29`, `25.46/26.50`, and `28.72/29.01 ms`, corresponding to `+5.69%/-4.65%/+1.16%/-6.13%/+11.48%/-4.08%/-1.01%`. The input-51200 rows are present but not yet populated.
- The sign changes across adjacent concurrency points are much larger and less monotonic than the isolated EP8 per-bucket LL wins. This does not prove a specific integration bug, but it means final success cannot be inferred from one isolated token bucket: scheduler batching, graph selection/reuse, communication overlap, or measurement variance must be separated after the kernel branch is chosen.
- The framework table makes low-concurrency/tiny-token latency especially relevant: concurrency 1/2 often regresses even though isolated token8/32 LL currently wins. The fresh token32 profile and the proposed partial-M-repeat K1 branch are therefore aligned with the observed weak framework region.

### Initial Flash EP8 token32 profile

- The graph-only token32 `hipprof --stats --hip-trace --follow-fork --devices 0` collection completed and wrote its database before the wrapper failed. The failure was only the final remote file-list command receiving a trailing CR (`sort\r`); the workload, result JSON, and profiler database completed successfully.
- Coarse profiler statistics are heavily instrumented and include capture/setup, so their absolute kernel times are not benchmark times. Within the MegaMoE kernels visible on device 0, K1 masked group GEMM and K3 combine/reduce dominate; K3 local GEMM and K2 are small. This supports inspecting K1 tiny-tail work and K3 synchronization separately rather than changing K2 again.
- Run directory: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_flash_ll_opt/ep8_profile_t32_20260714_114841`.

### Flash EP8 token512 profile comparison and first branch

- The matching token512 graph-only profile completed cleanly at `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_flash_ll_opt/ep8_profile_t512_20260714_115152`.
- Under the same heavy instrumentation, K1 masked group GEMM was about `4.44 ms/call` at token32 and `4.31 ms/call` at token512; K3 combine/reduce was about `3.25` and `3.17 ms/call`. These values are effectively fixed/noisy profiler costs, not replay latency. K3 local GEMM changed much more clearly from about `0.171` to `0.445 ms/call`, while K2 stayed tiny (`0.032` to `0.037 ms/call`).
- Source audit found that the Flash LL K1 BM32 tiny specialization masks invalid BF16 stores but still generates the second 16-row A-load/shuffle/scale/MMAC repeat. Historical K1-only BM16 evidence improved token32 K1 while K3 BM16 caused the end-to-end regression. The safest new branch is therefore K1-only: keep BM32 scheduling and K3 untouched, but compile out the empty second M16 repeat for the existing tiny specialization.
- The first branch must be compile-time/specialization-limited, not a per-lane divergent shortcut. Evidence requirements are same-run EP8 paired graph timing, random and skew correctness, final ISA showing the second MMAC/load group removed, and no VGPR/SGPR/private-segment regression. A dynamic `remaining_m <= 16` extension is deferred until this narrow ablation wins.
- Lower-priority K1 candidates are row-major 4096-byte staging (only if ISA shows redundant pointer VMEM), disabling an apparently unconsumed LL `row_expert_out` store as a separate A/B, and token-major topk6 route emit only if route-phase profiling justifies it.

### K1 tiny-repeat precision gate

- Exact source inspection adds a critical precondition to the proposed K1 branch. The host selects `kMaskTinyStore` from `valid_rows_per_expert <= 16`, but the kernel schedules per-expert tiles from device-side `cur_tokens = min(gemm_m[expert], m_per_expert)`. The current specialization only masks stores whose row is beyond `cur_tokens`; it does not assert that every expert has at most 16 rows.
- Therefore compiling out `mr=1` is safe only if `valid_rows_per_expert` is proven to be a hard per-expert upper bound. If it is an expected/average row count, random hot experts and `single-local-rank` routes can exceed 16 and the branch would silently omit legal rows. No implementation is allowed until the caller semantics are traced.
- The K3 audit provides a precision-preserving fallback: in split combine/reduce, issue topk6 `global_load_dwordx4` operations in pairs followed by one `s_waitcnt vmcnt(0)`, then retain the exact slot0..slot5 FP32 accumulation order. It changes only the VMEM wait window, leaves signals/barriers/fences unchanged, and has an objective ISA acceptance test.
- Caller tracing proves `valid_rows_per_expert` is `ll_expected_rows_per_expert`, while physical capacity is separately expanded to `ll_worst_rows_per_expert = route_capacity_tokens_per_rank * num_ranks`. It is explicitly an expected count with random-routing headroom, so compile-time removal of `mr=1` is rejected as skew-unsafe.
- A safe K1 ablation remains possible: for the existing tiny specialization only, compute the number of required 16-row repeats for the current expert/tile from device-side `cur_tokens`, and guard A loads, shuffles, MMACs, and scale loads with a block-uniform `mr < active_m_repeats`. Complete skew tiles retain every repeat; only the final partial tile skips empty repeats, and the existing per-row store mask remains for the partial 16-row repeat.

### Rejected K1 device-count partial-repeat branch

- The device-count-driven branch compiled and passed all random correctness checks, but catastrophically regressed exactly the three buckets using the tiny specialization: token `8/32/64` moved from `0.5050/0.6549/0.6946 ms` to `2.0103/2.3879/2.8151 ms`.
- Token128, which does not use this specialization, stayed noise-level (`0.7757` baseline versus `0.7793 ms` branch), and paired `ll-masked` graph timings stayed stable. This isolates the regression to the new inner-pipeline control flow rather than node load or baseline drift.
- The branch is rejected despite correct output. Repeated runtime guards inside unrolled A-load/shuffle/MMAC loops evidently destroy the intended pipeline/code generation; no EP16 or skew run is justified. Local source is restored immediately and the remote extension will be rebuilt before the next branch.
- This result also closes the broader dynamic-repeat direction for the current kernel structure. A future K1 partial-repeat attempt would need a separately compiled/static body selected outside the hot K loop while retaining a skew-safe full-repeat path, not inner-loop runtime `continue` checks.

### K3 paired-load branch design

- Source inspection confirms split reduce currently calls `global_load_uint4_device` once per topk slot, and that helper places `s_waitcnt vmcnt(0)` immediately after every `global_load_dwordx4`. With topk6 this creates six serial load/wait windows before six ordered FP32 accumulations.
- The safest implementation is not a no-wait helper whose result could be consumed early by compiler scheduling. Instead, add a pair helper containing two `global_load_dwordx4` instructions and the single `s_waitcnt vmcnt(0)` in one volatile inline-assembly block; expose both register results only after the wait completes.
- The topk6 path will process fixed pairs `(0,1)`, `(2,3)`, `(4,5)` and accumulate the first result then the second, preserving the exact slot0..slot5 FP32 addition order. Other topk values retain the existing loop/helper. No copy phase, readiness signal, barrier, store wait, or fence changes in this A/B.

### K3 paired-load first EP8 result

- The candidate passed eager and every graph-bucket precision check with exactly the same reported errors as baseline: graph max error `0.000244141` through token128 and `0.000549316` at token256/512.
- Candidate medians for `8/32/64/128/256/512` were `0.5028/0.6496/0.6822/0.7650/1.0505/1.8385 ms`, versus the original fresh baseline `0.5050/0.6549/0.6946/0.7757/1.0664/1.8218 ms`.
- This is about `0.4%/0.8%/1.8%/1.4%/1.5%` faster through token256 but about `0.9%` slower at token512. Token512 minimum (`1.8080 ms`) is effectively equal to baseline (`1.8091 ms`), suggesting median jitter rather than a clear large-token regression.
- The signal is plausible but below the optimizer's 2% noise gate. Preserve both old and candidate `.so` files and run alternating 20-replay A/B without rebuild between trials before deciding. No EP16 claim is allowed from this three-replay screen.

### K3 paired-load alternating 20-replay A/B

- The restored baseline `.so` and candidate `.so` were swapped into the same source-tree runtime path without rebuild, then run back-to-back with identical environment, seed, per-bucket capture, three graph warmups, and 20 replays. The candidate binary was restored afterward.
- Baseline medians were `0.50420/0.65496/0.69450/0.77852/1.06396/1.82360 ms`; candidate medians were `0.50320/0.64916/0.68220/0.76574/1.04872/1.82302 ms` for tokens `8/32/64/128/256/512`.
- Candidate improvements are `0.20%/0.89%/1.77%/1.64%/1.43%/0.03%`. The direction and magnitude reproduce the three-replay screen, so batching two loads is a real middle-bucket improvement with no observed regression, but it remains below the 2% branch-selection threshold.
- Before rejecting the wait-window family, inspect candidate ISA/resource usage and test a separate batch3 branch if register pressure is still safe. Batch3 must preserve pairwise slot order (`0,1,2`, then `3,4,5`) and remain a single-variable replacement, not be combined with copy changes.

### K3 batch3 first 20-replay result

- Batch3 compiled with the same Flash combine/reduce metadata as batch2: 39 VGPR, 100 SGPR, 4 pre-existing SGPR spills, zero VGPR spills, wave64, no private segment, and a 44060-byte function. It did not cross a resource threshold.
- Against the saved original-binary 20-replay baseline, batch3 medians are `0.50380/0.64554/0.67772/0.76280/1.04180/1.82324 ms` versus `0.50420/0.65496/0.69450/0.77852/1.06396/1.82360 ms` for tokens `8/32/64/128/256/512`.
- Improvements are about `0.08%/1.44%/2.42%/2.02%/2.08%/0.02%`. This is the first branch to cross the 2% gate in three contiguous middle decode buckets while remaining flat at the endpoints. Every graph-bucket precision check passed with baseline-level errors.
- Batch3 is a provisional champion, not yet retained. Run an adjacent candidate→baseline→candidate sequence, then skew correctness, EP16, profile, and final generated-code verification before selection.

### 2026-07-14 Pro K3 validation preflight

- The first Pro EP8 launch intentionally reused the Flash validation heap and failed before any fused or baseline kernel ran: DeepEP requested `num_rdma_bytes(11450456192)`, while `ROCSHMEM_HEAP_SIZE`/`DUSHMEM_HEAP_SIZE` were only `4737418240`.
- This exactly matches the recorded 2026-07-03 Pro EP8 requirement. The established successful Pro setting is `12884901888` bytes (12 GiB) for both heaps; use it for all current Pro `ll-masked` runs.
- The failed initializer left no KFD PIDs and did not change the retained runtime SHA. It is an environment-capacity preflight failure, not a correctness or performance observation.
- With the established 12 GiB heaps, the retained fixed K3 Pro EP8 six-bucket matrix passed. Fused graph medians for tokens `8/32/64/128/256/512` are `1.43582/1.90560/1.97538/2.12872/2.45262/3.95552 ms`; same-run `ll-masked` medians are `1.44408/2.08698/2.17298/2.54806/3.19218/5.74220 ms`.
- Pro EP8 fused gains are `0.57%/8.69%/9.09%/16.46%/23.17%/31.11%`. All per-bucket graph comparisons passed with max absolute error from `0.000488281` to `0.000976562`; runtime SHA stayed fixed and no KFD PIDs remained.
- The matching retained-K3 Pro EP16 matrix also passed. Fused medians are `1.09655/1.20865/1.28934/1.47813/2.39638/4.14787 ms`; same-run `ll-masked` medians are `1.14419/1.28061/1.40935/1.71316/2.84173/5.10637 ms`.
- Pro EP16 fused gains are `4.16%/5.62%/8.52%/13.72%/15.67%/18.77%` for tokens `8/32/64/128/256/512`. All graph checks passed with max absolute error `<=0.000976562`, the runtime SHA remained fixed, and all 16 cards were clean afterward.
- Pro EP8 `single-local-rank` token128 passed both eager and graph precision against `ll-masked`; both max absolute errors were `0.000488281`. Run status was 0.
- A later host-level card audit found a separate DeepSeek-V4-Pro SGLang service in container `wanghl_dev1`, parent PID `3325901`, using all 16 cards at roughly `87%-93%` VRAM. Epoch/timestamp comparison proves it started at `2026-07-14 14:25:52 +08:00`, about one hour after the retained Pro EP8/EP16 matrices and EP8 skew results were written at `13:25-13:29`; those results are therefore not load-contaminated.
- Container-local `hy-smi --showpids` cannot resolve process directories owned by the other container and reports a generic error; host-level `hy-smi` is authoritative while that service is active. Do not run EP16 skew or binary A/B until host-level cards are clean, and do not terminate the external service without explicit authority.
- Both attribution artifacts remain intact: saved original K3 SHA256 `41abd49f6f96f430894c0e8aeaeba2c623932d3bbb2fe314574d8976e332676c`; retained early-clobber K3 SHA256 `e32d9b94784d584ad7fdd0aaf699938742d4affa827134b4598a2feb3c53047c`. The source-tree runtime still matches the retained artifact.

### 2026-07-20 generic correctness and teardown conclusions

- Compact-route zero experts must explicitly emit cleared row metadata/output state on buffer reuse; otherwise a previous nonzero route can leave a stale combine row. Commit `3499767` locks this behavior with reuse tests.
- The initial fixed-expert quotient is dead in both normal K1 PACK5 layouts because all produced scalar values are overwritten before use. Commit `3c41e8d` removes it; the repeat-confirmed benefit is `4.241%` at token512 and noise-scale elsewhere.
- Importer mappings must close/detach on every rank before any owner frees its local exported allocation. The repository protocol is therefore `remote close/detach -> group barrier -> local export free -> group barrier`, implemented for MegaMoE `SymmBuffer` by `e12ba9f`.
- `e12ba9f` does not cover external DeepEP. The remaining cap512 exit warning correlates with an extra full Fabric dummy-buffer export/attach/detach/free lifecycle absent at cap128; this is a hypothesis and separate hardening debt, not a proven root cause.

### K3 artifact-attribution correction in progress

- `dccobjdump --show-sass` successfully produced full baseline/candidate gfx938 ISA files after extracting device ELFs and passing an output directory rather than a filename. The device ELFs and SASS hashes differ, but the inspected `k3_v3_fused_ext.so` diff currently shows changed branch offsets without changed `global_load_dwordx4`/`s_waitcnt` sequences.
- This raises a concrete attribution concern: the shared header rebuilds both `k3_fused_ext.so` and `k3_v3_fused_ext.so`, and the split reducer may live in the former even though the staged launch wrapper lives in the latter. If so, swapping only the V3 `.so` did not switch the modified kernel and the apparent 1-2% A/B difference is just run-order noise.
- Do not credit batch2 or start batch3 until the Python/binding symbol path is traced and the actual owning `.so` is saved/swapped. The correct binary must show the intended load-load-wait sequence in SASS.
- Python/binding tracing rules out the wrong-extension hypothesis: LL assigns `ext = load_v3_ll_extension()`, calls `ext.k3_v3_ll_combine_tail_split`, and `k3_v3_fused_ext.cu` directly launches `V3_K3_LowLatencyCombineReduceKernel`, which calls the edited split reducer. The saved/swapped V3 `.so` is the correct owner.
- The remaining question is generated-code equivalence: either the compiler did not materialize the intended branch as expected, or the first diff filters missed the changed body. A complete ISA diff and exact sequence extraction are required before interpreting the A/B.
- The persisted `--show-sass` text dumps have exactly `387868` lines and `24783890` bytes and expose only six changed branch immediates. Initially this looked like a no-op, but ELF symbol sizes prove that conclusion was too strong: Flash combine/reduce grows from `43224` to `44060` bytes (`+836`, exactly 209 4-byte instructions), and the Pro specialization grows from `44248` to `45112` bytes (`+864`, 216 instructions).
- The branch-immediate deltas match those inserted instruction counts, so the candidate body did materialize. The DTK `--show-sass` text output is incomplete/misleading for this internal branch body and cannot be used alone to compare the sequence. Use ELF metadata and a symbol-bounded disassembler/function extraction instead; the batch2 runtime A/B remains potentially valid pending that proof.
- ELF metadata is unchanged for the Flash combine/reduce specialization: baseline and batch2 both use wave64, `VGPR=39`, `SGPR=100`, `SGPR spills=4`, `VGPR spills=0`, and `private_segment_fixed_size=0`. Batch2 therefore did not cross a reported register/spill threshold.
- `dccobjdump --function`, `--recompile`, and `--disassembleAll` still expose only the kernel entry/control stub rather than the inserted internal branch body in this DTK build. Final wait-sequence proof must come from compiler save-temps assembly or another symbol-bounded AMDGPU disassembler; the empty/incomplete output is recorded as degraded tooling, not proof of absence.
