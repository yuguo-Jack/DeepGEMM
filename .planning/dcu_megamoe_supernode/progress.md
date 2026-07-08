# Progress

## 2026-06-27
- ✅ Created the supernode-specific planning workspace under `.planning/dcu_megamoe_supernode`.
- ✅ Read existing V3 planning style and avoided modifying old mojibake files.
- ✅ Read `hygon_tmp/supernode_code` examples and extracted the Galaxy PDF interface summary.
- ✅ Confirmed initial hard gates to remove/generalize: staged shape validators in Python and C++ plus IPC-only peer mapping in `SymmBuffer`.
- ✅ Generalized staged shape gates from EP8-only to DSV4 Flash EP8/EP16/EP32 in Python and C++ route scratch sizing.
- ✅ Added dynamic signal-slot helpers: EP8 keeps historical slots; EP16/EP32 move tail/copy, start barrier, post-K3 barrier, and chunk-ready signals into non-overlapping ranges.
- ✅ Updated K1/K3 LL launchers to instantiate local expert templates for 8, 16, and 32 local experts instead of always using 32.
- ✅ Added a supernode peer-memory path using HSA RPC/Fabric memory handles while preserving the existing HIP IPC path for EP8.
- ✅ Ran local static checks: `python -m py_compile ...`, `git diff --check`, and a source-guard string script.
- [ ] Full pytest remains deferred because the local Python environment has `No module named pytest`.
- [ ] Deferred true build/runtime validation to the future supernode environment.

## 2026-06-29 - Planning Format Cleanup
- ✅ Normalized supernode planning files to the shared status convention: ✅ / [ ] / 🚫 / 🧭.
- ✅ Removed UTF-8 BOM from supernode planning files and kept LF line endings.
- [ ] Runtime validation remains deferred until the supernode environment is available.

## 2026-06-29 - TX32 Supernode Workflow Skill
- ✅ Created project skill `.codex/skills/tx32-supernode-workflow` with TX32 two-node operation rules, root-login safety guardrails, non-shared-storage sync discipline, and multi-node 32-card launch notes.
- ✅ Removed the separate verification script and folded verification/sync/compile/debug/torchrun/hipprof procedures directly into `SKILL.md`, aligned with `remote-ssh-docker-workflow`.
- ✅ Ran `quick_validate.py`; skill structure passed.
- ✅ Verified both nodes accept key-based root SSH, both run `yiqa_deepep`, and both containers bind-mount `/home/yiqa` to `/home/yiqa`.
- ✅ Created `/home/yiqa/DeepGEMM` on both nodes and synced the current local workspace contents to both nodes for initial remote execution setup.
- ✅ Queried `dcu-rag-kb` for Hygon `hipprof` multi-node/session guidance and added a TX32 multi-node hipprof section.
## 2026-06-29 - Sync Workflow Correction
- ? Updated `tx32-supernode-workflow` sync guidance to match `remote-ssh-docker-workflow`: local remains the main editing workspace, remote nodes are execution workspaces, and sync is bidirectional.
- ? Removed the recommendation to prefer `git ls-files -co --exclude-standard` as the general sync policy; remote nodes do not need to rely on Git state.
- ? Documented full-workspace setup sync, targeted file/directory upload to both non-shared-storage nodes, and remote artifact sync-back under `hygon_tmp/`.

## 2026-06-29 - TX32 Launch Model Clarification
- Updated `tx32-supernode-workflow` to distinguish local-spawn tests from standard one-process-per-device training.
- SGLang / DeepEP / MegaMoE tests should use outer `torchrun --nproc-per-node=1` and pass `--num-processes=16` to spawn local workers inside each node.
- Megatron or generic DDP training scripts may use `torchrun --nproc-per-node=16` when torchrun itself owns one worker per local DCU.
- Added an explicit warning not to combine `--nproc-per-node=16` with scripts that also spawn 16 local workers.

## 2026-06-29 - Profiling Guidance Cleanup
- Removed the default multi-node `hipprof` recipe from `tx32-supernode-workflow`.
- Profiling commands should follow the active framework path; SGLang/MegaMoE framework runs normally prefer the existing torchprof-based workflow.

## 2026-06-29 - TX32 Environment Checks Borrowed From Remote Workflow
- Added `rocminfo` / `rocninfo` device-enumeration checks to `tx32-supernode-workflow` for ISA, CU, wavefront, and runtime device visibility validation.
- Kept `hy-smi` / `rocm-smi` as the regular utilization and memory-status check.
- Added `which hipcc`, Python version, pip version, and optional package inventory probes inside `yiqa_deepep`.
- Added `/dev/kfd` and `/dev/dri/renderD*` visibility checks for container device-mount diagnosis.

## 2026-06-29 - TX32 Skill Sync And Temp Directory Cleanup
- Standardized the remote temporary directory in `tx32-supernode-workflow` to `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/`.
- Updated archive sync, remote log paths, and artifact copy-back examples to use `hygon_tmp/supernode_debug/`.
- Removed the empty local `scripts/` directory from the skill; the skill now only carries `SKILL.md` and `agents/openai.yaml`.

## 2026-06-29 - Read-Only Remote Example Validation
- Verified SSH login examples on both TX32 nodes without touching running jobs.
- Verified Docker/mount example on node22; node69 currently reports `Cannot connect to the Docker daemon`, with `docker` service failed and `containerd` active, so no Docker restart/start was attempted.
- Verified node22 container repo/toolchain probe: `/home/yiqa/DeepGEMM`, `/opt/dtk/bin/hipcc`, Python 3.10.12, and pip 26.0.1.
- Verified node22 `hy-smi` card-status command: 16 HCUs visible and idle at the time of check.
- Found and fixed PowerShell/Bash quoting issues in the `rocminfo` and `pip3 list | grep` examples; the corrected examples now use bash-safe `'\''...'\''` grep patterns.
- Verified the corrected node22 `rocminfo` device enumeration and package-inventory commands.

## 2026-06-29 - TX32 Sync Guidance Wording
- Clarified that archive/tar sync is optional bulk sync for many changed files, not the default workflow.
- Normal iterative work should prefer VS Code SFTP multi-node upload or explicit single-file uploads to both nodes.

## 2026-06-29 - TX32 Sync Guidance Simplification
- Removed the optional tar/archive bulk-sync command block from `tx32-supernode-workflow`.
- Kept the core rule: local workspace, node0 `/home/yiqa/DeepGEMM`, and node1 `/home/yiqa/DeepGEMM` must stay synchronized for source/config/test files needed by a run.
- Kept the temp-file rule: local and remote temporary/debug artifacts go under `hygon_tmp/supernode_debug/`.
- Replaced the repeated `mkdir -p /home/yiqa/DeepGEMM` setup snippet with a verify-only `test -d` check; creation is only for first-time setup when the user asks for setup/sync.
- Clarified that remote temporary/debug artifacts do not all need to sync back to local; prefer keeping corresponding run directories organized on both remote nodes, and pull selected artifacts only when local analysis needs them.

## 2026-06-29 - TX32 Build Guidance Correction
- Updated `tx32-supernode-workflow` build guidance to run builds on both nodes.
- Documented that TX32 has no shared storage, so compiled artifacts produced on node0 are not visible on node1.
- Kept single-node build only as a same-node diagnostic path.

## 2026-06-29 - TX32 Compile/Test/Debug Template Expansion
- Expanded `Compile, Test, And Debug Templates` beyond compile/build examples.
- Added source-level pytest/contract-test examples for both nodes.
- Added debug-script examples that write node-specific logs under `hygon_tmp/supernode_debug/`.
- Kept GPU/multi-node execution under the separate 32-card torchrun section.

## 2026-06-29 - Supernode Runtime Selection Cleanup
- Removed the `MEGAMOE_DCU_SUPERNODE` runtime override from the supernode branch.
- Supernode peer-memory selection now depends on environment-derived device/rank topology: TX32 EP8/EP16 select hybrid mode with same-host HIP IPC peers, while EP32 can use Fabric/RPC for cross-host peers.
- Updated source guards and planning notes to match the automatic selection policy.

## 2026-06-29 - DeepEP TX32 Unit Test
- ✅ Followed Feishu doc `Qk13...` section `3.1 单测参考` for DeepEP cross-node unit tests.
- ✅ Checked both TX32 nodes before testing: node0 `10.17.160.69` and node1 `10.17.162.22` showed 16 idle HCUs each and no KFD PIDs.
- ✅ Ran the documented cross-node high-throughput unit command on `/home/yiqa/DeepEP-super_node/tests/test_intranode.py`; this script defaults to `--num-processes 4` per node, so the documented run is 8 ranks total. Both nodes completed with `rc=0`, and log summaries show all correctness cases `passed`.
- ✅ Ran cross-node low-latency unit test with `ROCSHMEM_HEAP_SIZE=4737418240`, `ROCSHMEM_IPC_MNVL=1`, and `ROCSHMEM_GDR_DISABLE_XDP=1` on `/home/yiqa/DeepEP-super_node/tests/test_low_latency.py`; this script defaults to `--num-processes 16` per node, so the run is 32 ranks total. Both nodes completed with `rc=0`.
- 🚫 Tried a 32-rank high-throughput variant by adding `--num-processes 16` to `test_intranode.py`; DeepEP failed inside `intranode_dispatch` with `num_nvl_bytes` assertion, so this is not a valid documented single-test configuration without further DeepEP buffer sizing changes.
- ✅ Confirmed both nodes returned to no KFD PIDs after testing.
- Logs:
  - Remote documented high-throughput: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/deepep_unit_20260629_175810/`
  - Remote low-latency: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/deepep_unit_ll_20260629_180031/`
  - Remote attempted 32-rank high-throughput: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/deepep_unit_intra32_20260629_180513/`
  - Local copied summaries: `hygon_tmp/supernode_debug/deepep_unit_logs_20260629_1802/`

## 2026-06-29 - DeepEP `normal_32.sh` Recheck
- ✅ Added the preferred TX32 DTK path `/home/yiqa/dtk-26.04.1/env.sh` to `.codex/skills/tx32-supernode-workflow/SKILL.md`.
- ✅ Checked both nodes before running `/home/yiqa/DeepEP-super_node/tests/normal_32.sh`: DTK env path exists on both nodes and no KFD PIDs were active.
- ✅ Ran `normal_32.sh` as provided on node0 and node1; both logs show no `FAILED`, `Traceback`, `Assertion`, or `RuntimeError` patterns and both reached `DeepEP test finished`.
- ✅ Confirmed both nodes returned to no KFD PIDs after the run.
- ⚠️ Important: `normal_32.sh` does not pass `--num-processes 16`; `test_intranode.py` defaults to `--num-processes 4`, and the logs confirm `ProcessGroupNCCL initialization options: size: 8`. Therefore this script validates the documented 8-rank normal path, not the forced 32-rank high-throughput variant.
- Logs:
  - Remote wrapper logs: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/deepep_normal32_script_20260629_182719/`
  - Local copied logs: `hygon_tmp/supernode_debug/deepep_normal32_script_20260629_182719/`

## 2026-06-29 - DeepEP High-Throughput 32-Rank Test Code Check
- ✅ Confirmed both TX32 nodes already have `/home/yiqa/DeepEP-super_node/tests/test_intranode.py` updated to restrict the high-throughput tuning loop to `for i in (24,):`.
- ✅ Confirmed both TX32 nodes set `test_intranode.py --num-processes` default to `16`, so `normal_32.sh` can launch the 32-rank path without adding the flag explicitly.
- ✅ Verified node0 and node1 `test_intranode.py` SHA256 match: `00B256DD3B62A5F386B825D4D29BA7EB04BAC3147C30F607229A120C969E4021`.
- [ ] Runtime validation intentionally deferred because the cards are currently being used by others.

## 2026-06-29 - MegaMoE EP16 Bring-Up On Node22
- ✅ Checked both TX32 nodes before MegaMoE testing: node22 `10.17.162.22` was idle; node69 `10.17.160.69` still showed 94% VRAM on all 16 DCUs and `hy-smi --showpids` could not open the process directory, so EP16 validation used node22 only.
- ✅ Extracted the requested old DeepGEMM whl under `/home/yiqa/yuguo/deepgemm_old_01290c7` on both nodes for isolation from the container package.
- 🚫 The old DeepGEMM whl cannot currently be imported in the `yiqa_deepep` torch runtime: it requires the older `c10::hip::c10_hip_check_implementation(..., int, bool)` ABI, while the current torch exports the newer `(..., unsigned int, bool)` ABI. MegaMoE runtime tests therefore used the container's built-in DeepGEMM until a matching torch/deepgemm pair is provided.
- ✅ Fixed EP16 single-node peer-memory selection: when `num_ranks <= torch.cuda.device_count()`, MegaMoE now keeps the HIP IPC path; Fabric/RPC peer memory is reserved for true multi-node rank counts.
- ✅ Isolated K1 EP16 normal route fault: old default ASM self-route path VMFaulted in `MEGAMOE_DISPATCH_PULL_L1`; `K1_PREBUILD_MODE=compact` passed with valid `row_combine_ptrs`, `m_indices`, and `output_index`.
- ✅ Made K1 normal default to compact prebuild for `num_ranks > 8`, preserving the old EP8 auto/ASM behavior and keeping an explicit `K1_PREBUILD_MODE=asm/asm_route` ablation escape hatch.
- ✅ Built MegaMoE on node22 with `/home/yiqa/dtk-26.04.1/env.sh`; wheel produced at `/home/yiqa/DeepGEMM/build/whl/megamoe-0.1-cp310-cp310-linux_x86_64.whl`.
- ✅ Remote static contract test passed on node22: `10 passed in 3.73s`.
- ✅ EP16 stage smoke results on node22:
  - `buffer`: passed, `world_size=16`, `peer_mode=ipc`, `device_count=16`.
  - `pre_dispatch`: passed.
  - `k1_only`: passed after compact prebuild default.
  - `k3_no_reduce`: passed, proving K1/K2/K3 main path is functional.
  - `fused` with `K3_USE_ASM_TAIL_REDUCE=0`: passed.
- ✅ Fixed EP16 normal fused default hang by disabling normal ASM tail-reduce when `num_ranks > 8`; LL backend still keeps its own fused/split tail behavior. Default EP16 `fused` now reaches `stage=fused_ok`.
- [ ] EP16 benchmark matrix is next: LL graph capture512 replay `8,32,64,128,256,512`, and normal eager `512,1024,1025,2048,2050,4096,4097,5120,8192`.
- [ ] EP32 validation remains blocked while node69 cards are occupied; if node69 stays occupied, proceed with EP16-only data as requested.

## 2026-06-29 - MegaMoE EP16 Matrix Bring-Up Debug
- ✅ EP16 LL graph capture512 is now correct on node22 for uniform `8,32,64,128,256,512` and uneven replay buckets; MegaMoE graph replay is faster than the `ll-masked` baseline in the collected runs.
- [ ] EP16 normal eager correctness is still blocked: route stats match the normal-contiguous baseline, but a small set of output tokens are all-zero in MegaMoE while baseline rows are nonzero.
- [ ] Next debug step: inspect the K3 combine slots for the top differing tokens to determine whether the loss happens before combine writeback or during post-K3 local reduce.
- ✅ Rechecked the requested `/home/yiqa/yuguo` DeepGEMM whl. Direct import fails against the active torch 2.10.0/DTK 26.04.1 runtime because the old package expects `c10_hip_check_implementation(..., int, bool)` while torch exports `(..., unsigned int, bool)`.
- ✅ Added a local ABI shim under `/home/yiqa/yuguo/deepgemm_old_01290c7/libc10_hip_abi_shim.so` on node22 and verified the old DeepGEMM package imports with `LD_PRELOAD` plus `PYTHONPATH=/home/yiqa/yuguo/deepgemm_old_01290c7`.
- [ ] Before continuing EP16, run EP8 sanity with the old DeepGEMM baseline to prove the original 8-card path is healthy.

## 2026-06-29 - MegaMoE Supernode Symm Buffer Correction
- ✅ User clarified that EP8 sanity on TX32 supernode must also exercise the new supernode-aware symmetric buffer path, not silently fall back to the original EP8 IPC-only path.
- ✅ Verified the requested `/home/yiqa/yuguo` DeepGEMM whl imports only with the local ABI shim because the active torch exports the newer `c10_hip_check_implementation(..., unsigned int, bool)` symbol.
- ✅ Confirmed yuguo DeepGEMM baseline must use the old Marlin weight layout (`weight8bit_nt_kpack2_marlin`); the newer contiguous layout can produce invalid values with the old whl at large M.
- ✅ Reproduced the old whl's padding sensitivity: `m_indices=-1` can VMFault, so old-baseline padding rows must be clamped to a harmless valid expert because `output_index` filters them later.
- ✅ EP8 normal correctness passed with the original IPC path and yuguo old DeepGEMM baseline after old-layout + old-baseline owning-copy workaround: `max_abs=0.000488281`.
- ✅ EP8 normal correctness also passed with the built-in DeepGEMM baseline, proving the MegaMoE compute path itself is healthy before switching the symm-buffer selection.
- 🚫 Tried forcing every peer handle to Fabric/RPC on single-node EP8; local peer attach failed in `hsa_ext_rpc_memory_attach` with HSA status `4104`, even with all 16 local DCUs visible. This all-Fabric design is abandoned.

## 2026-06-29 - EP8 Uses Supernode-Aware Hybrid Symm Buffer
- ✅ Implemented the hybrid symm-buffer selection in `megamoe/__init__.py`: TX32/supernode environments are detected from physical `/dev/dri/renderD*` count, visible device count, or ranks exceeding local devices.
- ✅ EP8 sanity on a 16-DCU TX32 node now selects `peer_mode=hybrid` instead of the legacy IPC-only path; same-host peer handles still use HIP IPC internally, while cross-host peers are reserved for Fabric/RPC.
- ✅ Added hybrid C++ APIs in `python_api_hip.cpp` to allocate both IPC and Fabric handles, open each peer according to the same-host mask, and close IPC/Fabric handles with the matching API.
- ✅ Rebuilt on node22 and validated EP8 normal correctness with the built-in DeepGEMM baseline under `peer_mode=hybrid`.
- [ ] The requested yuguo old-DeepGEMM baseline still needs follow-up under hybrid mode: MegaMoE compute passed with the built-in baseline, but the old baseline path currently trips old-package tensor/storage compatibility in the full distributed test.
- ✅ Follow-up EP8 confirmation on node22: `test_mega_moe_dcu.py --num-processes 8 --megamoe-backend normal --baseline-kind normal-contiguous --skip-bench` passed with `peer_mode=hybrid`, `correct=true`, `max_abs=0.000488281`, `mean_abs=9.32389e-06`.
- ✅ Node22 card state was clean before and after the run; no KFD PIDs remained after completion.
- Log: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep8_hybrid_confirm_20260630_003544/node22.log`.

## 2026-06-30 - EP16 Normal K1 Compact Capacity Fix
- ✅ Reproduced the EP16 normal eager 2048 failure as a K1 compact route-capacity issue rather than a K3 reduce issue. Debug combine-slot probes showed missing top-k partials with `output_index=-1`, concentrated on the tail local expert of a rank.
- ✅ Minimal ablation confirmed the hypothesis: enlarging K1 route capacity to max tokens made 2048 pass with `max_abs=0.000488281`.
- ✅ Implemented the lower-risk fix in `K1_fused/k1_fused_ext.cu`: when EP16/EP32 are forced onto compact prebuild, use fixed compact capacity instead of the tighter estimated compact capacity, preserving EP8 auto-compact behavior.
- ✅ Fixed the Python scratch layout in `megamoe/opt.py` so `_v3_staged_capacity_rows()` uses the same K1 headroom formula as C++ (`max(slack, ceil(expected/divisor))`). This fixed 8192 `l1_out_workspace` under-allocation.
- ✅ Rebuilt MegaMoE on node22 with `/home/yiqa/dtk-26.04.1/env.sh`.
- ✅ EP16 normal eager correctness now passes on node22 for the checked boundary set:
  - 2048: `max_abs=0.000488281`, log `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep16_normal_2048_fixedcompact_20260630_064143/node22.log`.
  - 2050: `max_abs=0.000488281`.
  - 4097: `max_abs=0.000488281`.
  - 5120: `max_abs=0.000488281`.
  - 8192: `max_abs=0.000488281`; the run produced correct output but needed manual cleanup after finalization stalled.
- ✅ EP16 LL graph capture512/replay uniform matrix passes and is faster than `ll-masked` baseline:
  - 8: MegaMoE 0.3725 ms vs baseline 0.9211 ms.
  - 32: MegaMoE 0.4033 ms vs baseline 0.9340 ms.
  - 64: MegaMoE 0.4773 ms vs baseline 0.9598 ms.
  - 128: MegaMoE 0.6053 ms vs baseline 1.0186 ms.
  - 256: MegaMoE 1.0378 ms vs baseline 1.3803 ms.
  - 512: MegaMoE 1.9711 ms vs baseline 2.0477 ms.
  - Log `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep16_ll_graph_matrix_rocmem_20260630_065750/node22.log`.
- ✅ EP16 LL graph uneven capture512 passes: local rank0 replay 505 tokens, MegaMoE 1.6261 ms vs baseline 1.8078 ms, log `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep16_ll_graph_uneven_20260630_065907/node22.log`.
- ✅ EP16 normal eager performance smoke:
  - 512: MegaMoE 1.3090 ms vs baseline 2.1917 ms.
  - 2048: MegaMoE 4.6880 ms vs baseline 6.1976 ms.
  - Log `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep16_normal_perf_smoke_20260630_070019/`.
- ✅ EP16 normal eager additional performance matrix on node22:
  - 1024: MegaMoE 2.7494 ms vs baseline 3.8750 ms.
  - 1025: MegaMoE 2.7335 ms vs baseline 3.6834 ms.
  - 2050: MegaMoE 5.0300 ms vs baseline 6.3913 ms.
  - 4096: MegaMoE 8.1634 ms vs baseline 11.0204 ms.
  - 4097: MegaMoE 8.2270 ms vs baseline 11.1997 ms.
  - 5120: MegaMoE 10.0424 ms vs baseline 13.3566 ms.
  - Log `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep16_normal_perf_matrix_part_20260630_070330/`.
- [ ] Continue normal eager performance matrix for `8192` after confirming the large-token finalization stall is not hiding a cleanup bug.
- [ ] Node69 remains unavailable for EP32 validation until its persistent VRAM occupancy clears.

## 2026-06-30 - EP16 Normal 8192 Baseline Isolation
- ✅ Confirmed EP16 normal 8192 MegaMoE correctness against the normal-contiguous baseline on rank0 output: `max_abs=0.000488281`.
- ✅ Isolated the apparent 8192 stall: after correctness prints, at least one nonzero rank does not reach the cleanup barrier. The issue is tied to the DeepEP normal baseline path at this bucket, not to MegaMoE K1/K3 numeric output.
- 🚫 Tried `--prepost-backend triton` as a baseline workaround; it failed during Triton AMD compilation with missing `/tmp/*.amdgcn`, so it is not a valid EP16 8192 workaround.
- ✅ Added `--skip-baseline-bench` to `test_mega_moe_dcu.py` so MegaMoE-only timing can be collected when correctness was already established but the selected baseline cannot be timed safely. The JSON marks `baseline_execution=skipped`, `baseline_bench_skipped=true`, and baseline metric fields are `null`.
- ✅ EP16 normal 8192 MegaMoE-only benchmark completed on node22:
  - MegaMoE 14.9958 ms, 164.97 TFLOPS/card effective, HBM 80.63 GB/s, xHCL 40.29 GB/s.
  - Log `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/megamoe_ep16_normal_perf_8192_fusedonly_json_20260630_075516/node22.log`.
- [ ] Full 8192 normal baseline comparison remains pending on a DeepEP baseline fix or a different safe baseline implementation.

## 2026-06-30 - dcu_mega_v3 EP8 Reference Recheck On TX32
- ✅ Answered the EP8 buffer question: the `dcu_mega_v3` branch should use its original single-node HIP IPC path for EP8 reference runs. It does not need Fabric/RPC buffer changes; the supernode branch separately validates the hybrid wrapper while still using HIP IPC for same-host peers.
- ✅ Ran `dcu_mega_v3` branch code in an isolated compare repo on node22 using its own `test_mega_moe_dcu.py` flow and current environment DeepGEMM.
- 🚫 Tried the requested `/home/yiqa/yuguo` old DeepGEMM package first, but it is not a good primary validator in this torch runtime: it needs an ABI shim, uses old `deepgemm` package naming, and its C++ op rejects some current-torch tensor views with `Tensor doesn't have storage`.
- ✅ Patched only the temporary compare repo so the current environment DeepGEMM baseline uses the required contiguous weight layout (`[E,K/64,N/16,4,16,16]`). MegaMoE fused weights and kernels were not changed for this reference run.
- ✅ `dcu_mega_v3` EP8 normal eager 4096 result on node22:
  - correctness: `max_abs=0.000671387`, `mean_abs=9.45305e-06`;
  - MegaMoE: `7.1299 ms` average per rank;
  - normal-contiguous baseline: `9.8677 ms`;
  - speedup: `1.384x`.
  - Remote log/result: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/dcu_v3_ep8_4096_own_currentdg_20260630_100210/`.
- ✅ This matches the current supernode branch EP8/4096 formal result within noise (`~7.21 ms`), so the supernode branch has not shown an EP8 4096 regression. The remaining normal large-token issue is EP16-specific and points back to K3 peer combine scatter/write fanout.

## 2026-06-30 - EP8 Normal Eager Gap Investigation: normal node vs TX32 node22

- User asked to first explain why EP8 `4096` normal eager on TX32/supernode does not match the historical normal-node `~5.8 ms` result before continuing EP16/EP32 performance conclusions.
- Normal node (`hg@10.17.176.11`, docker `sglang_megamoe`, `/workspace/DeepGEMM`) direct run on branch/worktree `dcu_mega_v3`, git HEAD `522e556`, command `--num-processes 8 --num-max-tokens-per-rank 4096 --num-tokens 4096 --megamoe-backend normal --baseline-kind normal-contiguous --warmup 5 --repeat 20`:
  - Correct, `tokens=4096/4224`, `route_scratch=0.470 GiB`.
  - Fused median/min: `5.9189 / 5.8310 ms`; baseline median/min: `9.5958 / 9.4802 ms`; speedup `1.621x`.
  - This matches historical `~5.8 ms`.
- TX32 node22 (`root@10.17.162.22`, docker `yiqa_deepep`) with the normal node actual dirty worktree source copied over, rebuilt in an isolated temp repo, and only baseline layout patched for the current DeepGEMM package:
  - Devices `0..7`: fused `~7.3057 ms`.
  - Devices `8..15`: fused `~7.0569 ms`; repeat=30 with clock watcher: fused `~7.1607 ms`.
  - Correctness remained good (`max_abs=0.000671387`).
- This rules out the supernode branch hybrid buffer as the EP8 slowdown root cause: the same normal-node source rebuilt on TX32 still lands in the `~7.1 ms` band.
- Environment differences observed:
  - Normal node: `torch 2.9.0`, HIP `6.3.26113`, DTK `/opt/dtk`, 8 visible `BW1101` devices, max clock `1300 MHz`, DeepGEMM `op.so` hash `0e8e...`, size `23.9 MB`.
  - TX32 node22: `torch 2.10.0`, HIP `6.3.26113`, DTK `/home/yiqa/dtk-26.04.1`, 16 visible `BW1301_LC` devices, max clock `1350 MHz`, DeepGEMM `op.so` hash `38eb...`, size `654 MB`.
  - Moving the normal-node DeepGEMM package to TX32 failed with a torch/HIP ABI undefined symbol, so that package cannot be used as a drop-in TX32 baseline.
- Topology check: both normal node 8-card and TX32 node22 16-card report HSW links, 1-hop between all local HCUs, weight 0; 8-card subsets on TX32 are not crossing a visibly worse topology.
- Current conclusion: historical normal-node EP8 `5.8 ms` is a valid normal-node reference, but not a same-hardware reference for TX32. The TX32 single-node EP8 baseline for this container/runtime is currently `~7.1 ms`, and EP16/EP32 performance should be compared against this TX32 EP8 local baseline unless the TX32 runtime/clock/driver difference is separately resolved.

## 2026-06-30 - rocBLAS Large GEMM A/B: Normal Node vs TX32

- Ran DTK `rocblas-bench` through the dynamic loader because the installed benchmark ELF lacks execute permission in both DTK trees.
- Card status before the run:
  - Normal node had no active HCU use; `hy-smi --showpids` failed to enumerate processes while VRAM remained high from prior context state.
  - TX32 node22 was clean with no KFD PIDs and 0% VRAM before the run.
- Single-card BF16 square GEMM, `gemm_ex`, device0 on normal node and device8 on TX32:
  - 4096^3: normal `468.8 us / 293.2 TFLOPS`, TX32 `487.5 us / 281.9 TFLOPS`.
  - 8192^3: normal `4504 us / 244.1 TFLOPS`, TX32 `4413 us / 249.2 TFLOPS`.
  - 16384^3: normal `50918 us / 172.8 TFLOPS`, TX32 `51195.2 us / 171.8 TFLOPS`.
- MegaMoE-like BF16 rectangular GEMM:
  - M=4096,N=2048,K=4096: normal `235.75 us / 291.5 TFLOPS`, TX32 `243.25 us / 282.5 TFLOPS`.
  - M=4096,N=4096,K=2048: normal `255.95 us / 268.5 TFLOPS`, TX32 `260.95 us / 263.3 TFLOPS`.
- Logs:
  - Normal node: `/workspace/DeepGEMM/hygon_tmp/debug/rocblas_bench_20260630/`.
  - TX32 node22: `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/rocblas_bench_20260630/`.
- Current readout: raw rocBLAS BF16 GEMM is close between machines, and TX32 is not meaningfully slower on large square GEMM. The MegaMoE EP8 gap is therefore unlikely to be pure matrix-core throughput; it more likely comes from MegaMoE multi-kernel/peer-write/launch/clock behavior or the torch/runtime stack around those kernels.

## 2026-06-30 - DeepEP-Style Fabric Peer Memory Cleanup
- ✅ Re-read DeepEP-super_node normal/MNNVL memory flow: normal supernode shared memory uses one Fabric/RPC-backed mode rather than a per-peer same-host IPC / cross-host RPC hybrid table.
- ✅ Updated MegaMoE supernode `SymmBuffer` selection to use `peer_mode=fabric` whenever `_use_supernode_peer_memory()` is true; legacy non-supernode EP8 still uses the original HIP IPC path.
- ✅ Removed the Python hybrid peer mask flow from `megamoe/__init__.py`; kernels still receive the same flat peer pointer table and do not branch on peer type.
- ✅ Removed unused hybrid C++ binding APIs from `python_api_hip.cpp` and kept only IPC and Fabric allocation/open/close APIs.
- ✅ Changed Fabric signal buffer allocation to `hipDeviceMallocFinegrained`, matching the DeepEP-style exported shared-buffer approach and avoiding the earlier all-RPC signal attach risk seen with uncached signal memory.
- [ ] Runtime validation is pending because TX32 nodes are no longer reachable / have been taken down.

## 2026-06-30 - Explicit Peer-Memory Mode
- ✅ Changed MegaMoE peer-memory selection to match the requested DeepEP-style opt-in behavior: default `ipc`, and `MEGAMOE_DCU_PEER_MEMORY=rpc` enables Fabric/RPC handles.
- ✅ Decoupled EP16/EP32 shape support from IPC/RPC selection. Rank count, local expert count, scratch sizing, and signal slot layout are still shape concerns; peer memory is now controlled only by the environment variable.
- [ ] Runtime validation is pending because TX32 nodes are unavailable.

## 2026-06-30 - EP16/EP32 Normal Path Guardrails
- ✅ Re-checked the staged normal path: `num_ranks > 8` forces K1 compact prebuild from `opt.py`, so EP16/EP32 do not use the in-ASM route builder.
- ✅ Tightened the low-level K1 extension so `K1_PREBUILD_MODE=asm/asm_route` is only honored for `num_ranks <= 8`; direct EP16/EP32 low-level calls now stay on compact prebuild too.
- ✅ Re-checked K3 normal eager/graph: `_tail_reduce_enabled_for_backend()` returns `False` for non-LL `num_ranks > 8`, so EP16/EP32 normal use K3 no-tail ASM plus the separate local combine reduce kernel.
- ✅ Re-checked normal graph support: `graph=True, megamoe_backend="normal"` enters `_run_opt_3stage_graph`; K1 graph uses compact prebuild with `runtime_num_tokens`, K3 graph uses no-tail ASM with `active_tiles`, then `reduce_local_combine_graph`.
- [ ] Runtime validation remains pending because TX32 nodes are unavailable.

## 2026-06-30 - DeepEP LL Baseline MNNVL Selection Cleanup
- ✅ Replaced the outdated DeepEP LL baseline `allow_mnnvl = num_ranks > torch.cuda.device_count()` heuristic with the same explicit peer-memory policy as MegaMoE: default IPC, and `MEGAMOE_DCU_PEER_MEMORY=rpc/fabric/mnvl/1/true` enables Fabric/MNNVL-style handles.
- ✅ Re-scanned the supernode deltas and found no remaining hostname/device-count based peer-memory selection. Rank-count checks that remain are shape/path guardrails only, such as forcing K1 compact prebuild for EP16/EP32.
- ✅ Verified local syntax and whitespace checks: `python -m py_compile megamoe/__init__.py megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` and `git diff --check`.
- [ ] Runtime validation remains pending because TX32 nodes are unavailable.

## 2026-06-30 - 151.1 Single-Node 16-Card RPC Validation Bring-Up
- [ ] Use node `10.17.151.1`, docker `sglang_megamoe`, repo `/root/yuguo/DeepGEMM`, and DTK `/root/yuguo/dtk-26.04.1/env.sh` for EP16 RPC validation.
- ✅ Confirmed repo sync state: branch `supernode`, commit `f8f892c`.
- ✅ Confirmed container mount `/root/yuguo -> /root/yuguo`, Ubuntu 22.04.5, DTK hipcc available.
- ✅ Current card state changed from occupied to idle before testing; `hy-smi --showpids` still reports a tool-side process-directory error, so host `ps` is also used for occupancy checks.
- ✅ Build MegaMoE inside the container with `MEGAMOE_DCU_PEER_MEMORY=rpc`.
- [ ] Validate EP16 RPC `LL graph` and `normal eager` first.
- [ ] If those pass, expand to `LL eager`, `normal graph`, and uneven coverage.
- ✅ Built MegaMoE successfully on 151.1 with `/root/yuguo/dtk-26.04.1/env.sh`; wheel output under `/root/yuguo/DeepGEMM/build/whl/`.
- [ ] EP16 RPC validation is currently blocked by device15/container runtime state: `LL graph` crashes before MoE kernels in `_C.allocate_hip_fabric_buffer`, and the reduced single-card smoke `torch.empty(..., device="cuda")` also segfaults with `HIP_VISIBLE_DEVICES=15`. Device14 Fabric alloc succeeds, so this is not yet a MegaMoE kernel failure.
- ✅ Fixed the first RPC attach issue by matching DeepEP shared-memory allocation behavior: Fabric/RPC exported buffer sizes are now aligned to 2 MiB before `hsa_ext_rpc_memory_create`.
- [ ] EP8 RPC 8-card smoke with `ll-masked` baseline now reaches DeepEP baseline init but fails inside DeepEP low-latency MNNVL/RDMA setup (`ROCSHMEM_HEAP_SIZE`/`deep_ep.cu:376 invalid argument`). Continue MegaMoE RPC validation with `normal-contiguous` baseline to isolate MegaMoE from DeepEP baseline environment setup.
- ✅ EP8 RPC 8-card `LL graph` smoke passed with `normal-contiguous` baseline on devices `0..7`: correct, graph replay token 128/cap512 median `0.7549 ms`, eager main-call median `0.7369 ms`, peer mode `fabric`.
- ✅ EP8 RPC 8-card `normal eager` smoke passed with `normal-contiguous` baseline on devices `0..7`: correct, token 512 median `1.8096 ms`, peer mode `fabric`.
- ✅ Permission to clean stale SGLang/container state was granted; stale SGLang was killed and docker `sglang_megamoe` was restarted.
- [ ] True EP16 validation is still blocked after cleanup: host shows no KFD PIDs and device14 torch allocation succeeds, but physical device15 still segfaults on ordinary `torch.empty(..., device="cuda")` and dmesg reports `libgalaxyhip.so.5` segfaults. This needs host-level device/driver reset or a healthy 16-card node before EP16 can be judged.
- ✅ EP8 RPC 8-card `LL eager` smoke passed on devices `0..7`: correct, token 128/cap512 median `0.7376 ms`, baseline `1.4487 ms`, peer mode `fabric`.
- ✅ EP8 RPC 8-card `normal graph` smoke passed on devices `0..7`: correct, token 512/cap512 graph replay median `1.6795 ms`, eager main-call median `1.9976 ms`, peer mode `fabric`.
- ✅ EP8 RPC 8-card `LL graph uneven` smoke passed on devices `0..7` for local tokens `512,257,128,64,32,7,0,0`; graph replay medians for runtime `7/32/128/512` were `0.4939/0.6324/0.7277/0.9089 ms`.
- ✅ EP8 RPC 8-card `normal eager uneven` smoke passed on devices `0..7` for local tokens `512,257,128,64,32,7,0,0`: correct, median `1.6304 ms`, baseline `1.7379 ms`.

## 2026-07-01 - 151.1 Device15 Recheck While `chl_sgl0512` Is Running
- ✅ Entered `chl_sgl0512` first because it was actively using the cards. It uses the same image as `sglang_megamoe`, sources `/opt/dtk/env.sh`, sees torch `2.10.0`, and `HIP_VISIBLE_DEVICES=15` small CUDA allocation plus synchronize succeeds.
- ✅ Restarted existing `sglang_megamoe` container and reran the same device15 checks with `/root/yuguo/dtk-26.04.1/env.sh`; both inline torch allocation and the previous `torch_alloc_smoke.py` now succeed on device15.
- ✅ Updated conclusion: the earlier device15 `libgalaxyhip.so.5` segfault was a transient container/runtime state, not a reproducible MegaMoE code failure and not a fixed physical-card failure.
- [ ] `normal eager 4096 EP8` on 151.1 is currently not run because `chl_sgl0512` is actively occupying all 16 cards with large VRAM allocations. Wait for the node to become free or get explicit permission from the owner before killing/interrupting that workload.

## 2026-07-01 - 151.1 RPC EP8/EP16 Performance Pass
- ✅ Rechecked card state before the run: all 16 HCUs were idle with only 2 MiB VRAM used and no KFD PIDs.
- ✅ EP8 RPC `normal eager 4096` on devices `0..7` passed: correct, route scratch `0.470 GiB`, MegaMoE median/min `5.7636/5.7153 ms`, normal-contiguous baseline median/min `10.0042/9.8233 ms`, speedup `1.736x`.
- ✅ EP16 RPC `LL graph` capture512 passed on devices `0..15`: correct, graph replay medians for runtime `8/32/64/128/256/512` were `0.3749/0.4069/0.4769/0.5977/0.9869/1.9228 ms`.
- ✅ EP16 RPC `normal eager` partial matrix completed before the node SSH became unresponsive:
  - `512`: correct, MegaMoE `1.2940 ms`, baseline `2.1316 ms`, speedup `1.647x`.
  - `1024`: correct, MegaMoE `2.0580 ms`, baseline `3.5208 ms`, speedup `1.711x`.
  - `2050`: correct, MegaMoE `3.5886 ms`, baseline `5.7776 ms`, speedup `1.610x`.
  - `4096`: correct, MegaMoE `6.5743 ms`, baseline `10.3451 ms`, speedup `1.574x`.
- [ ] Need to collect the remote JSON/logs after 151.1 SSH recovers. The loop had progressed through at least `1025/2048` and started `4097`, but the terminal output was truncated and then the SSH connection closed during the `4097` run.
- [ ] Need to resume remaining EP16 normal eager buckets after recovery: `4097,5120,8192` if they did not complete; collect exact `1025/2048` values from JSON.
- ✅ 151.1 recovered and result JSONs were collected. Exact EP16 RPC `normal eager` values:
  - `1025`: correct, MegaMoE `2.0536 ms`, baseline `3.4505 ms`, speedup `1.680x`.
  - `2048`: correct, MegaMoE `3.6066 ms`, baseline `5.7382 ms`, speedup `1.591x`.
  - `4097`: correct, MegaMoE `6.5976 ms`, baseline `10.3945 ms`, speedup `1.575x`.
  - `5120`: correct, MegaMoE `8.1738 ms`, baseline `12.6119 ms`, speedup `1.543x`.
  - `8192`: MegaMoE-only timing with `correctness-iters=0` passed, MegaMoE `12.6043 ms`; full-baseline rerun passed correctness (`max_abs=0.000488281`, `mean_abs=5.56393e-06`) but timed out in benchmark after 420s, so no stable EP16 baseline timing is recorded for this bucket.
- ✅ EP8 RPC normal eager comparison points on the same 151.1 node:
  - `4096`: MegaMoE `5.7636 ms`, baseline `10.0042 ms`.
  - `5120`: MegaMoE `7.3197 ms`, baseline `11.7945 ms`.
  - `8192`: MegaMoE `10.7700 ms`, baseline `19.0455 ms`.
- ✅ Baseline observation: EP16 normal-contiguous baseline is slightly slower than EP8 at comparable big-token buckets on this node (`4096`: `10.3451` vs `10.0042 ms`, `+3.4%`; `5120`: `12.6119` vs `11.7945 ms`, `+6.9%`). This supports that part of the EP16 large-token slowdown is also present in the baseline/communication path, not only MegaMoE-specific. EP16 `8192` baseline timing remains unusable because the benchmark phase times out.
- ✅ EP16 LL graph cap512 replay small-token effective bandwidth estimate from the recorded JSON: replay8 `0.3749 ms` -> estimated HBM `1077.6 GB/s`, xHCL `1.57 GB/s`; replay32 `0.4069 ms` -> estimated HBM `998.6 GB/s`, xHCL `5.80 GB/s`, assuming all 16 local experts are touched. The JSON stores replay latency, not per-bucket bandwidth fields, so these are reconstructed estimates.
- ✅ Follow-up conclusion after 151.1 EP8/EP16 RPC runs: the earlier "optimize normal K3 peer combine writes" item is no longer a required supernode task. EP8 `4096` on the same 151.1 environment recovered to `5.7636 ms`, EP16 normal eager/graph are correct and faster than baseline, and no extra K3 peer-combine rewrite is justified without new profiler evidence.

## 2026-07-01 - EP16/EP32 Normal Tail-Reduce1 Signal Layout Fix
- ✅ Root-caused the EP16/EP32 normal tail-reduce1 bug to the K3 ASM tail-reduce signal protocol: the old source only signaled ranks `0..7` and waited on fixed second-half offsets `64..120`, which is only correct for EP8.
- ✅ Updated both K3 normal tail-reduce ASM sources to signal ranks up to 31 and to choose wait-base offsets from `asm_signal_num_ranks`: EP8 uses base `64`, EP16 uses base `128`, and EP32 uses base `256`.
- ✅ Removed the Python normal-backend `num_ranks > 8` tail-reduce disable gate. K1 EP16/EP32 still uses compact prebuild; the restored behavior is only for K3 normal ASM tail-reduce.
- ✅ Rebuilt MegaMoE on 151.1 before the node became unreachable; assembler accepted the expanded macros and the wheel was produced successfully.
- ✅ EP16 normal tail-reduce1 smoke on 151.1 passed for `512` and `4096`:
  - `512`: correct, MegaMoE `1.2804 ms`, baseline `2.1008 ms`.
  - `4096`: correct, MegaMoE `6.8729 ms`, baseline `10.2744 ms`.
- [ ] Full EP16 normal tail-reduce1 matrix (`512,1024,1025,2048,2050,4096,4097,5120,8192`) was started but 151.1 became unreachable. Resume only after confirming SSH/card state and using a single DTK environment, not mixed `/root/yuguo/dtk` plus `/opt/dtk` libraries.
- [ ] EP32 tail-reduce1 runtime validation still needs a healthy 32-card environment.
- ✅ Superseded policy update: EP16/EP32 normal now defaults back to tail-reduce0 / external local reduce. Tail-reduce1 code remains available only when explicitly forcing `K3_USE_ASM_TAIL_REDUCE=1`; LL behavior is unchanged.

## 2026-07-01 - EP16/EP32 Eager Normal Active-Tile Patch
- ✅ Found that normal eager K2/K3 did not pass `active_tiles`, unlike normal graph. With EP16/EP32 fixed compact capacity this can inflate K2/K3/tail-reduce work to capacity rows even when correctness is fine.
- ✅ Patched the existing eager start `rank_barrier` path to write `graph_runtime_num_tokens` scratch from `graph_max_tokens` when no graph runtime tensor is provided.
- ✅ Patched eager normal K2 to pass `active_tiles` and `active_tile_m=256`.
- ✅ Patched eager normal K3 no-tail to pass `active_tiles`, and tail-reduce1 to pass `active_tiles`, the runtime-token offset, and the generation tensor so the ASM active-tile gate and reducer work sizing can match graph behavior.
- ✅ Local checks passed: `python -m py_compile megamoe/opt.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` and `git diff --check`.
- [ ] Remote build and EP16 normal eager retest are blocked because `10.17.151.1` currently times out at ping/SSH banner exchange.

## 2026-07-01 - Active-Tile Attempt Superseded
- 🚫 The EP16/EP32 eager normal active-tile patch was tested and reverted.
- Validation on 151.1 passed correctness but did not show enough performance value:
  - `4096`: correct, MegaMoE `6.6835 ms`, baseline `10.4333 ms`.
  - `5120`: correct, MegaMoE `8.3056 ms`, baseline `12.8190 ms`.
  - `8192`: MegaMoE-only `12.8188 ms`.
- The previous "active-tile patch" note above should be treated as historical investigation, not an active work item.

## 2026-07-01 - 151.1 EP16 RPC Normal Graph Validation
- ✅ Rechecked card state before testing on `10.17.151.1`: all 16 HCUs were idle and returned to 0% VRAM after the run.
- ✅ Built and tested from `/root/yuguo/DeepGEMM` in docker `sglang_megamoe` with `/root/yuguo/dtk-26.04.1/env.sh`, `MEGAMOE_DCU_PEER_MEMORY=rpc`, `K3_USE_ASM_TAIL_REDUCE=1`, and normal graph replay enabled.
- ✅ EP16 RPC `normal graph` uniform cap512 passed:
  - eager main-call: MegaMoE `1.3027 ms`, baseline `1.7411 ms`.
  - graph replay medians for runtime `128/256/512`: `0.9027/1.0053/1.2542 ms`.
- ✅ EP16 RPC `normal graph` uneven cap512 passed:
  - actual generated tokens `287`, rank0 replay locals `121/249/505`.
  - eager main-call: MegaMoE `1.2027 ms`, baseline `1.5385 ms`.
  - graph replay medians for runtime `128/256/512`: `0.8411/0.9076/1.1298 ms`.
- ✅ EP16 RPC `normal graph` uniform cap4096 passed:
  - eager main-call: MegaMoE `6.9082 ms`, baseline `9.9314 ms`.
  - graph replay medians for runtime `1024/2048/4096`: `2.0467/3.3681/6.4547 ms`.
- ✅ EP16 RPC `normal graph` uneven cap4096 passed:
  - actual generated tokens `2291`, rank0 replay locals `1017/2041/4089`.
  - eager main-call: MegaMoE `6.2936 ms`, baseline `7.9413 ms`.
  - graph replay medians for runtime `1024/2048/4096`: `1.8613/3.3350/6.4743 ms`.
- ✅ Additional uneven middle buckets passed:
  - cap1024, actual tokens `573`, rank0 locals `505/1017`, graph replay medians `512/1024 -> 1.1037/1.7890 ms`, eager MegaMoE `1.8496 ms`, baseline `2.5275 ms`.
  - cap2048, actual tokens `1146`, rank0 locals `1017/2041`, graph replay medians `1024/2048 -> 1.8058/3.2687 ms`, eager MegaMoE `3.3288 ms`, baseline `4.3669 ms`.
- Note: DeepEP normal-contiguous baseline still emits `cached_notify_combine` launch-bound warnings. They do not block correctness or MegaMoE graph replay timing, but should not be mistaken for MegaMoE kernel warnings.
- Logs:
  - Primary run: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_ep16_normal_graph_20260701_155607/`.
  - Extra uneven run: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_ep16_normal_graph_extra_20260701_161143/`.

## 2026-07-01 - DeepSeek-V4-Pro Shape Support Start
- Re-read `.planning/dcu_megamoe_supernode/{task_plan,findings,progress}.md` and the V3 staged-path source before implementation, per user request.
- Confirmed the Pro target shape: experts=384, topk=6, hidden=7168, intermediate=3072, with EP8/EP16/EP32 local experts 48/24/12.
- Added an initial source-level test in `megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` that expects a shared V3 staged shape registry to cover both Flash and Pro shapes. This was added before the user's "read source before modifying" reminder; no further implementation edits were made until the source/planning re-read above.
- Source read-in found remaining Flash-only gates in Python validators, C++ route scratch sizing, K1 Python/C++ wrappers, K1/K3 LL launch dispatch, K1/K3 per-expert done counters, and K1 normal compact header offsets.
- Local TDD red-run attempt: `python -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py::test_v3_model_shape_registry_covers_flash_and_pro_ep_sizes` failed because this Windows Python does not have `pytest` installed. Defer pytest execution to the remote/container environment and keep local `py_compile`/source checks for quick feedback.
- Implemented the first local Pro-shape pass: shared Python/C++ staged shape contracts, Python entry shape checks, route scratch tail-counter expansion to 112 ints, K1 compact header dynamic local-expert offsets, K1/K3 LL Pro template dispatch, and K1/K3 per-expert done-counter limits expanded to 64.
- Local checks passed: `python -m py_compile megamoe/__init__.py megamoe/opt.py megamoe/dcu_megamoe_opt/v3_config.py megamoe/dcu_megamoe_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`, `git diff --check`, and manual source-level calls for the shape registry, route_scratch mirror, and supernode source support tests. Full local pytest is still unavailable due missing `pytest`.

## 2026-07-01 - DeepSeek-V4-Pro EP16 First Remote Pass
- Re-read the supernode/V3 planning files after context recovery and rechecked the changed source before continuing.
- Remote 151.1 card checks before build/test showed all 16 HCUs idle and no KFD PIDs. Remote source pytest passed before the latest workspace fix: `11 passed in 7.17s`.
- Remote build passed before the latest workspace fix and produced `/root/yuguo/DeepGEMM/build/whl/megamoe-0.1-cp310-cp310-linux_x86_64.whl`.
- First Pro EP16 normal 512 run reached the staged path but failed at K3 shape/workspace validation. Config confirmed the intended Pro shape and local experts: `ranks=16`, `local_experts=24`, `hidden=7168`, `intermediate=3072`.
- Root cause: Flash reused `l1_out` as K3 output workspace because Flash has `hidden == 2 * intermediate`; Pro has `l1_out` columns 6144 but K3 output hidden 7168. Local fix now reserves `k3_out` separately when needed, passes it to all K3 calls, and mirrors the extra BF16 reservation in the C++ route scratch size API.
- Additional local consistency fix: `k1_graph_flag_reset_layout()` now uses fixed compact capacity for `num_ranks > 8`, matching the EP16/EP32 K1 graph launch path.
- Local post-fix checks passed: `python -m py_compile megamoe/opt.py megamoe/dcu_megamoe_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_opt/v3_config.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`, `git diff --check`, and manual source-guard calls with a minimal local pytest import stub.
- Next: sync the latest Python/C++/test changes to 151.1, rerun source pytest, rebuild, recheck cards, and rerun Pro EP16 normal 512.

## 2026-07-01 - DeepSeek-V4-Pro EP16 Normal ASM Stride Fix
- Remote source pytest on 151.1 passed after the Pro shape changes: `11 passed in 7.34s`.
- Pro EP16 LL correctness-only passed after slicing the separate `k3_out` workspace to runtime rows: `max_abs=0.000488281`, `mean_abs=1.21163e-05`, backend `v3_ll_eager`, `weight_layout=unified`.
- Pro EP16 normal 512 still failed after the first shape/scratch fix, even with unified weights: route stats were OK, but output mismatches stayed around `max_abs~0.18`, pointing to the normal ASM data path rather than routing or weight layout.
- Re-read the K1/K3 normal ASM and fixed two remaining Flash-only stride assumptions:
  - K1 symmetric-buffer `x_sf` offset used `num_max_tokens * 4096`; it is now `num_max_tokens * hidden` through `sgprSizeL`.
  - K3 combine scatter used BF16 row stride `4096 * 2`; it is now `hidden * sizeof(bfloat16)` through `sgprSizeI`.
- Local checks after the ASM stride fix passed: `python -m py_compile ...`, `git diff --check`, and source guards confirm the old literal strides only remain in negative test assertions.
- Remote rebuild `hygon_tmp/supernode_debug/pro_ep16_20260701_dynstride3` completed successfully and produced `/root/yuguo/DeepGEMM/build/whl/megamoe-0.1-cp310-cp310-linux_x86_64.whl`.
- Runtime validation is waiting for card availability: `hy-smi --showmemuse` shows all 16 HCUs at about `90-91%` VRAM, and host `pgrep` shows an active `sglang.launch_server` for `/data_add/lijing/DeepSeek-V4-Pro-FP8-Channel` with `tp/ep/dp=16`. Do not run MegaMoE tests until that workload releases the cards.

## 2026-07-01 - DeepSeek-V4-Pro Normal GLOBAL_OFFSET_A Dynamic Fix
- After the `dynstride3` rebuild, Pro EP16 normal 512 still failed numerically: `max_abs=0.194091796875`, argmax `(188,5888)`, fused `-0.1318359375`, baseline `0.062255859375`, `stats_ok=True`.
- Re-read the normal eager path and K2/K3 wrappers. K2 is unlikely to be the remaining root cause because LL Pro EP16 already passes and uses the same K2 kernel; Pro's K2 "hidden" is the intermediate dimension `3072`, which is within the existing generic path.
- Static ASM review found a third Flash-only normal ASM address assumption in `GLOBAL_OFFSET_A`: `offset0I * (4096 / 4)`. This affects K1 and K3 pack5 A-weight addressing. It is now dynamic as `offset0I * (SizeI / 4)` in:
  - K1 PACK5 and UNIFIED_PACK5 normal ASM.
  - K3 PACK5, UNIFIED_PACK5, TAILREDUCE_PACK5, and TAILREDUCE_UNIFIED_PACK5 normal ASM.
- Local checks passed after this patch: `python -m py_compile megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`, `git diff --check`, and `rg` confirms the old `4096 / 4` / `v_lshlrev ... 10` forms no longer exist in ASM.
- Runtime validation is blocked again because `10.17.151.1` is currently occupied by an SGLang DeepSeek-V4-Pro server: all 16 HCUs show about `79-82%` VRAM, and host `pgrep` lists `python -m sglang.launch_server ... --model-path /data_add/lijing/DeepSeek-V4-Pro-FP8-Channel`.

## 2026-07-01 - DeepSeek-V4-Pro dynoffsetA4 Build Complete, Runtime Waiting
- Re-read `.planning/dcu_megamoe_supernode` and `.planning/dcu_megamoe_v3`, then re-read the relevant Pro source paths before continuing: `v3_config.py`, `layout.cuh`, `opt.py`, K1/K3 Python wrappers, K1/K3 host extensions, Pro source guards, and the normal ASM address macros.
- Local pre-remote checks passed again: `python -m py_compile megamoe/__init__.py megamoe/opt.py megamoe/dcu_megamoe_opt/v3_config.py megamoe/dcu_megamoe_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_opt/K3_fused/k3_fused.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` and `git diff --check`.
- Remote rebuild after the `GLOBAL_OFFSET_A` dynamic fix completed successfully. The build log was archived at `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_dynoffsetA4/build.log`, `build.status=0`, and the wheel exists at `/root/yuguo/DeepGEMM/build/whl/megamoe-0.1-cp310-cp310-linux_x86_64.whl`.
- Card check before correctness testing still shows the node is occupied: all 16 HCUs have about `79-82%` VRAM in use, and host `pgrep` shows active SGLang DeepSeek-V4-Pro processes including `python -m sglang.launch_server ... --model-path /data_add/lijing/DeepSeek-V4-Pro-FP8-Channel ... --port 15000` plus scheduler processes. No Pro EP16 normal correctness test was launched while the cards are occupied.
- Rechecked card state three times after build (`20:51`, `20:53`, `20:55`, and `20:57` CST in the remote log window). The same SGLang launch server PID `413458` remained active and HCU VRAM stayed at about `79-82%`. Treat runtime validation as waiting on external card availability, not as a build or source failure.
- User guardrail added: after cards are free, also run Flash normal performance smoke/A-B. If the shared dynamic normal ASM causes a material Flash latency regression, create Pro-only normal ASM kernel/code-object variants and route Pro to them while keeping Flash on the original constant-address kernel path.

## 2026-07-01 - DeepSeek-V4-Pro dynoffsetA4 EP16 Normal Retest
- Rechecked `10.17.151.1` before the retest at about `22:49 CST`: all 16 HCUs showed 0% use and 0% memory, and `hy-smi --showpids` reported no KFD PIDs. Host-side `pgrep` still showed stale `sglang::scheduler` names, but no active `sglang.launch_server` and no KFD/VRAM ownership.
- Ran Pro EP16 normal correctness-only from `/root/yuguo/DeepGEMM` in docker `sglang_megamoe` with `/root/yuguo/dtk-26.04.1/env.sh`, `--num-processes 16`, `--num-max-tokens-per-rank 768`, `--num-tokens 512`, `--hidden 7168`, `--intermediate-hidden 3072`, `--num-experts 384`, `--num-topk 6`, `--megamoe-backend normal`, `--baseline-kind normal-contiguous`, `--skip-bench`, and `--correctness-iters 1`.
- Result failed: status `1`, `max_abs=0.1669921875`, `argmax=(181,6248)`, `fused=0.0`, `baseline=0.1669921875`, `stats_ok=True`. Log path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_dynoffsetA4/normal_ep16.log`.
- Interpretation for next work: the exact fused zero suggests a missing K3 combine/write or skipped row for the failing token/lane, not just numerical error. Next step is targeted diagnostics to locate whether K1 normal produced bad/absent row metadata for token 181 or K3 normal skipped/wrote the wrong hidden lane.

## 2026-07-01 - DeepSeek-V4-Pro EP16 Combine-Slot Diagnostic
- Added a test-only `--debug-combine-on-fail` path that reads the local symmetric-buffer combine slots at the max-diff element and prints the six top-k slot values, top-k experts, owner ranks, and router weights. Local `python -m py_compile megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` passed before syncing.
- Synced only the Python test harness to `10.17.151.1`; no kernel rebuild was needed for this diagnostic.
- Rechecked card state at about `23:00 CST`: all 16 HCUs showed 0% use and 0% memory, and `hy-smi --showpids` reported no KFD PIDs.
- Pro EP16 normal diagnostic run failed as expected. New failure point: rank `3`, `max_abs=0.18212890625`, `argmax=(225,5845)`, fused `-0.1259765625`, baseline `0.05615234375`, `stats_ok=True`.
- Combine diagnostic payload: `combine_slots=[0.0, 0.0, -0.1259765625, 0.0, 0.0, 0.0]`, `combine_slot_sum=-0.1259765625`, top-k experts `[16,80,96,201,302,378]`, owner ranks `[0,3,4,8,12,15]`.
- Interpretation: local reduce is faithful because `fused == combine_slot_sum`. The remaining Pro normal correctness bug is before reduce, in K3 normal combine production/writeback or the row metadata consumed by K3; reduce should not be the next target.
- Log path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_combine_debug1/normal_ep16.log`.

## 2026-07-01 - DeepSeek-V4-Pro EP16 Staged-Loop Fix And Route Debug
- Re-read the K3 normal no-tail and tail-reduce ASM staged-store macro after the combine-slot diagnostic ruled out local reduce.
- Found a Pro-only staged-store bug: `K3_STORE_STAGED_HALF` used `sgprSizeI` as the staged-loop row bound. Flash hidden `4096` made the loop count accidentally correct, but Pro hidden `7168` over-walked each half-tile.
- Patched all four K3 normal combine ASM sources (`PACK5`, `UNIFIED_PACK5`, `TAILREDUCE_PACK5`, `TAILREDUCE_UNIFIED_PACK5`) to keep the staged half-tile bound fixed at `4096`.
- Local checks passed after the patch: `python -m py_compile megamoe/opt.py megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` and `git diff --check`.
- Rechecked `10.17.151.1` before testing: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Remote rebuild completed successfully under `/root/yuguo/DeepGEMM` with `/root/yuguo/dtk-26.04.1/env.sh`; build log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_stagedloop1/build.log`, `build.status=0`.
- Important runtime detail: `bdist_wheel` generated fresh K3 `.co` files under `build/lib...`; these were copied into `/root/yuguo/DeepGEMM/megamoe/dcu_megamoe_opt/K3_fused/` before retesting.
- Pro EP16 normal still failed after the staged-loop fix. Failure point: rank `11`, row `213`, col `5384`, fused `0.083984375`, baseline `-0.10009765625`, `stats_ok=True`; combine slots summed to the fused value.
- Added an env-gated route metadata diagnostic in `megamoe/opt.py` (`MEGAMOE_DCU_DEBUG_ROUTE=source_rank,token`) and synced it to 151.1.
- Route debug for source rank `11`, token `213` showed all six expected top-k routes had valid output rows and nonzero `row_combine_ptr` values before K3, including the previously zero-looking slots. This makes missing K1 route metadata unlikely for that case.
- Next step: isolate whether the remaining wrong combine values are already present in K1 normal activation output/scale or are introduced by K3 normal L2 weight/scale addressing and writeback.

## 2026-07-01 - DeepSeek-V4-Pro EP16 K3/Combine Value Debug
- Added env-gated `MEGAMOE_DCU_DEBUG_ROUTE=source_rank,token,col` diagnostics in `megamoe/opt.py` to print per-slot route rows, Python single-column K3 reconstruction, and actual local combine slots before the local reduce.
- Synced the debug-only Python changes to `10.17.151.1`; local `python -m py_compile megamoe/opt.py` and `git diff --check` passed.
- Rechecked cards before every diagnostic run; all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- K3 value debug showed `k3_out` is not a reliable observation point for normal combine ASM because the combine buffer can be nonzero while `k3_out` reads as zero.
- Combine-before-reduce debug for `source_rank=11`, `token=211`, `col=1270` printed local slots `[-0.0220947265625, 0.0, 0.003753662109375, 0.0, 0.0, 0.0]`, while the Python single-column reconstruction from `act_fp8`, `act_scale`, `l2_weight`, and `l2_scale` predicted roughly `[-0.00419, 0, 0.00632, 0, 0, 0]`.
- Added a test-only `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT=normal|unified` override to avoid forcing Pro normal diagnostics to unified pack5. Local `python -m py_compile` and `git diff --check` passed; the test file was synced to 151.1.
- Rechecked 151.1 before testing: all 16 HCUs showed 0% VRAM/HCU and no KFD PIDs.
- Pro EP16 normal with plain normal ASM pack5 (`MEGAMOE_DCU_PRO_WEIGHT_LAYOUT=normal`) still failed: rank `5`, row `184`, col `4778`, fused `-0.1552734375`, baseline `0.0272216796875`, combine slots `[0,0,0,-0.1552734375,0,0]`. This rules out a unified-only pack5 layout bug.
- User suggested checking LL against `normal-contiguous` baseline before deeper intermediate comparison. Ran Pro EP16 LL correctness-only with `--baseline-kind normal-contiguous`; it passed with `max_abs=0.000488281`, `mean_abs=1.21163e-05`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_ll_normalbaseline1/ll_ep16.log`.
- Patched the debug-only K3 value printer so LL `m_indices` is interpreted as per-expert actual counts instead of per-row local experts; local `python -m py_compile megamoe/opt.py` and `git diff --check` passed, then `opt.py` was synced to 151.1.
- Same-input debug runs used seed default `1234` and `MEGAMOE_DCU_DEBUG_ROUTE=5,184,4778`:
  - Normal log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_sameinput_normal_route1/normal_ep16.log`.
  - LL log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260701_sameinput_ll_route2/ll_ep16.log`.
- LL slot3 on rank10: `act_scale=0.0001689324`, Python K3 ref `0.0318703`, `k3_out=0.0319824`, and full output passed normal-contiguous baseline.
- Normal slot3 on rank10: `act_scale=0.0003850642`, Python K3 ref `0.0051409`, but source-rank5 local combine slot at col4778 was `-0.1318359`. The normal run also failed at row184/col3241 with combine slot3 `0.1064453` versus baseline `-0.1015625`.
- Current confirmed boundary: normal K3 ASM is wrong for Pro because its actual combine write does not match the Python reconstruction from the normal path's own visible K2 output and L2 packed weight. Next fix target is K3 normal ASM Pro addressing/writeback, not baseline or local reduce.

## 2026-07-02 00:00:17 +08:00 - Pro EP16 K3 Normal ASM Source Read
- Re-read active .planning/dcu_megamoe_supernode files before the next patch, per the recurring planning workflow.
- Current evidence still supports the user's suggested baseline split: Pro EP16 LL passes against the normal-contiguous baseline, while Pro EP16 normal fails with local combine slots summing exactly to the fused wrong value.
- Source read focus: K3 normal ASM COMPUTE_ADDRESS_SCALE, GLOBAL_OFFSET_A, pack5 A base calculation, row-combine staged store, and the LL C++ pack5 reference load/store math.
- Intermediate conclusion: normal ASM plain and unified pack5 both fail, so the next patch should target shared normal K3 address/scale/store logic rather than the unified-layout ni permutation alone.

## 2026-07-02 00:18:00 +08:00 - Pro EP16 Pack5 `GLOBAL_OFFSET_A` Correction
- Re-read `.planning/dcu_megamoe_supernode/{task_plan,findings,progress}.md` after context recovery and updated the Pro support plan before running more remote tests.
- Corrected the previous dynamic `GLOBAL_OFFSET_A = offset0I * (SizeI / 4)` hypothesis. In the normal ASM macro, `offset0I` is pack5 tile-local `ni16`; its stride is fixed `4 * 256 = 1024`, while `ko` and K-loop increments remain hidden-dependent.
- Local source now restores fixed `v_lshlrev_b32 ... 10` in all two K1 normal ASM sources and four K3 normal ASM sources, with the corrected comment `pack5 ni16 * (4 * 256)`.
- Updated `test_dcu_megamoe_v3.py` source guards so they reject the bad `SizeI / 4` interpretation and assert the corrected fixed `ni16` stride count.
- Next validation step: local static checks, then 151.1 card check, targeted sync, remote source tests/build, copy generated `.co` files into the runtime source tree, and rerun Pro EP16 normal correctness.

## 2026-07-02 00:25:00 +08:00 - Local Static Check Before Remote Retest
- Local `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py megamoe\dcu_megamoe_opt\tests\test_mega_moe_dcu.py` passed.
- Local `git diff --check` passed after fixing a pre-existing progress-note trailing space.
- Source grep confirmed no remaining `SizeI / 4`, `no256 * (4096 / 4)`, or `v_mul_lo_u32 v[\vgprTmp+0], s[sgprSizeI]` in K1/K3 ASM source directories.
- Positive grep confirmed the corrected fixed pack5 `ni16 * (4 * 256)` offset appears in 2 K1 ASM sources and 4 K3 ASM sources, with matching source-guard assertions in `test_dcu_megamoe_v3.py`.

## 2026-07-02 00:35:00 +08:00 - Remote Retest After `GLOBAL_OFFSET_A` Correction
- Rechecked `10.17.151.1` before GPU work: all 16 HCUs showed 0% HCU/VRAM and `hy-smi --showpids` reported no KFD PIDs.
- Synced 6 normal ASM sources plus `test_dcu_megamoe_v3.py` to `/root/yuguo/DeepGEMM`.
- Remote source checks passed in docker `sglang_megamoe`: `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.15s`.
- Remote `python3 setup.py bdist_wheel` completed successfully. The fresh build/lib `.co` files were newer than the runtime source `.co` files, so the 2 K1 and 4 K3 pack5 `.co` files were copied back into `megamoe/dcu_megamoe_opt/{K1_fused,K3_fused}/` before testing.
- Pro EP16 normal correctness still failed after the `GLOBAL_OFFSET_A` correction. Failure point: rank `13`, row `394`, col `4869`, `max_abs=0.19384765625`, fused `0.1552734375`, baseline `-0.03857421875`, `stats_ok=True`.
- `--debug-combine-on-fail` showed the local combine buffer is still the first bad visible boundary: slots `[0.0, 0.0, 0.1552734375, 0.0, 0.0, 0.0]`, slot sum equals fused, owner ranks `[2,7,9,11,14,5]` for top-k experts `[60,190,217,282,343,138]`.
- Interpretation: restoring fixed `GLOBAL_OFFSET_A` was source-correct but not sufficient. Continue diagnosis from normal K3 ASM scale/address/store behavior for the bad combine slot; baseline and local reduce remain ruled out.

## 2026-07-02 00:45:00 +08:00 - K3 ASM Source Re-read After Failed Retest
- Re-read K3 normal ASM scale addressing (`COMPUTE_ADDRESS_SCALE`), pack5 A/B global offsets, combine scatter/store macros, and the C++ `GpuProb` launch argument filling.
- The remaining failure can still be explained by either shared K3 ASM behavior or the default unified transposed pack5 path. Earlier plain-normal evidence was collected before the corrected fixed `GLOBAL_OFFSET_A` retest, so it is no longer a clean discriminator.
- Next diagnostic: rerun Pro EP16 normal with `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT=normal` after the fixed-ni16 `.co` rebuild. If plain passes, focus on unified pack5 layout/addressing; if plain still fails, keep tracing shared K3 ASM activation/scale/store behavior.

## 2026-07-02 00:55:00 +08:00 - Plain Normal Layout Retest
- Rechecked `10.17.151.1` immediately before the run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Pro EP16 normal with `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT=normal` still failed after the fixed-ni16 rebuild. Failure point: rank `1`, row `384`, col `2265`, `max_abs=0.19873046875`, fused `-0.07666015625`, baseline `0.1220703125`, `stats_ok=True`.
- Combine slots again prove the failure is before local reduce: `[0.0, -0.07666015625, 0.0, 0.0, 0.0, 0.0]`, owner ranks `[3,5,7,13,14,1]` for top-k experts `[85,121,186,317,339,29]`.
- Conclusion: the remaining bug is shared K3 normal ASM behavior, not the default unified transposed pack5 layout alone. Next run should enable `MEGAMOE_DCU_DEBUG_ROUTE=1,384,2265` on the plain layout to compare Python single-column reconstruction against the actual combine slot.

## 2026-07-02 01:05:00 +08:00 - Plain Normal K3 Debug Reclassifies Boundary
- Ran plain normal with `MEGAMOE_DCU_DEBUG_ROUTE=1,384,2265` after another clean card check.
- For the selected row384/col2265 slot1, K3 normal matched the debug Python single-column reconstruction from normal-path inputs: `python_ref=-0.0926068`, local combine slot `-0.0927734`.
- That means the selected row384 wrong output is not caused by K3 reading or writing a different value from its own visible inputs. The mismatch versus the normal-contiguous baseline is now more likely earlier, in K1/K2 output production or in the rows/scales fed into K3.
- Suspicious evidence: slots 0/2/3/4/5 for the same token had valid routes but identical near-zero `act_scale=2.2321428616578487e-07`, while slot1 had a normal-scale `act_scale=0.000479672`.
- Need next diagnostic to print a Python K1+SwiGLU+K2 reference for selected routed rows and compare it against `act_fp8/act_scale` before K3.

## 2026-07-02 01:18:00 +08:00 - Plain Normal K1 Debug Finds Missing Rows
- Added an env-gated K1 debug point in `megamoe/opt.py` that prints `l1_out_absmax`, gate/up absmax, route weight, output row, `m_indices`, and `row_combine_ptr` after K1 and before K2 for `MEGAMOE_DCU_DEBUG_ROUTE=source_rank,token,col`.
- Synced the debug-only Python change to `10.17.151.1`; remote `python3 -m py_compile megamoe/opt.py` passed before running.
- Rechecked cards before the run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Plain Pro EP16 normal debug with `MEGAMOE_DCU_DEBUG_ROUTE=6,81,3794` showed all six routes had valid metadata, but slots 0/2/4 had `l1_out_absmax=0.0` before K2 while slots 1/3/5 had normal nonzero K1 rows.
- The selected K3 values matched Python reconstruction from the visible K2 inputs for the nonzero slots, and the zero K1 rows explain the near-zero K2 activation scales.
- Current root-cause target is K1 normal compact route/active-tile compute coverage for Pro EP16, not baseline, local reduce, or K3 combine for the selected evidence.

## 2026-07-02 01:30:00 +08:00 - K1 Compact Active-Tile Offset Patch
- Re-read K1 normal host compact kernels, launch argument packing, and the active prebuilt branch in both K1 normal ASM sources.
- Found a Flash/EP8-only constant in the compact-prebuilt branch: ASM read `route_scratch_i32[64]` for the active compact tile count, but the HIP compact builder writes it at `2 * local_experts`.
- For Pro EP16 `local_experts=24`, the correct active-tile offset is `48`; reading offset `64` lands inside `tile_experts`, so some valid compact row tiles can be skipped before K1 produces `l1_out`.
- Patched `GpuProb.reserved_c4` to pack low16=`active_tiles_offset` and high16=`local_experts`; patched both K1 normal ASM sources to read that dynamic offset and clamp loaded `m_indices` against dynamic local experts instead of `32`.
- Added static source assertions in `test_dcu_megamoe_v3.py` so the hard-coded active-tile `64` and local-expert `32` checks cannot silently return.

## 2026-07-02 01:42:00 +08:00 - K1 Staged-X Row Stride Patch
- Remote source pytest and rebuild after the active-tile patch succeeded: `test_dcu_megamoe_v3.py` reported `11 passed in 7.47s`, and fresh K1 `.co` files were copied into the runtime source tree.
- Card check before testing showed all 16 HCUs idle and `hy-smi --showpids` reported no KFD PIDs.
- Pro EP16 normal default layout still failed after the active-tile patch: rank `15`, row `188`, col `5424`, fused `0.1083984375`, baseline `-0.115234375`, combine slot1 only.
- Targeted debug for `MEGAMOE_DCU_DEBUG_ROUTE=15,188,5424` showed slot1 row55 had nonzero K1 output, but valid slots on higher row tiles still had `l1_out_absmax=0.0`.
- Re-read K1 dispatch-pull staging and found the producer still stored `staged_x` rows with `row << 12` (`row * 4096`) while Pro consumers read with dynamic hidden stride `7168`.
- Patched both K1 normal ASM sources to keep the Flash `hidden==4096` shift path and use `row * hidden` for Pro/dynamic staged_x row stores. Static tests now assert this dynamic-store branch is present.

## 2026-07-02 01:58:00 +08:00 - Same-Input LL Comparison After K1 Coverage Fixes
- Remote source pytest and rebuild after the staged-x row-stride patch succeeded; the fresh K1 `.co` files were copied into the runtime source tree before retesting.
- Card checks before the normal and LL diagnostic runs showed all 16 HCUs idle and no KFD PIDs.
- Pro EP16 normal still failed, but the targeted route debug for `MEGAMOE_DCU_DEBUG_ROUTE=5,189,3478` showed all six top-k slots now have nonzero K1 rows and K3 combine values match a Python reconstruction from visible K2/K3 inputs.
- Same-input Pro EP16 LL with `--baseline-kind normal-contiguous` passed correctness for the same selected token/column, while LL slot values differed strongly from normal before combine.
- Current conclusion: baseline, local reduce, K3 combine/writeback, compact active-tile coverage, and staged-x row-store coverage are ruled out for the latest selected point. The remaining first-suspect boundary is normal K1/K2 numeric production versus LL/baseline for Pro EP16.

## 2026-07-02 00:49:20 +08:00 - Resume K1/K2 Correctness Diagnosis
- Re-read the active `.planning/dcu_megamoe_supernode` task plan, progress, and findings before continuing, per the file-based planning workflow.
- Current working hypothesis remains evidence-based, not a new fix: the next boundary to prove is whether normal K1 rows, K2 quantized rows/scales, or their weight addressing diverge first from the passing LL/normal-contiguous baseline path.
- Next planned diagnostic is env-gated and should not affect Flash or normal runtime by default: print bounded K1 row samples and K2 `act_fp8`/`act_scale` samples for the same `(source_rank, token, col)` route in normal and LL/default/plain runs.
- Added the local debug-only `MEGAMOE_DCU_DEBUG_K2` path plus bounded K1 row samples in `megamoe/opt.py`. Local `python -m py_compile megamoe\opt.py` and `git diff --check` passed.
- Synced `megamoe/opt.py` and the updated planning files to `10.17.151.1`. Remote `python3 -m py_compile megamoe/opt.py` passed, and remote source pytest passed with `11 passed in 7.30s` after using the DTK LD path.
- Card checks before the normal/LL diagnostic runs showed all 16 HCUs idle and no KFD PIDs.
- Same route `MEGAMOE_DCU_DEBUG_ROUTE=5,189,3478` now shows a first concrete K1 split: normal slot1 and slot2 on rank5 have different local experts (`m_index=5` and `m_index=13`) but identical K1 samples and identical K2 raw FP8 samples; only K2 scale differs with the route weight. LL on the same route has distinct K1/K2 samples for the two slots and passes the normal-contiguous baseline.
- Current root-cause fork: either the compact tile's first-row `m_indices[compact_tile * 256]` differs from the actual route row and makes the ASM load the wrong expert, or `sgprScaleFlag` is correct but the normal ASM A-weight expert stride/addressing is not using it correctly for Pro.

## 2026-07-02 01:07:00 +08:00 - Tile Header Fork Resolved
- Re-read active planning files and ran the planning session catch-up script before continuing after context compaction.
- Updated the K1/K2 debug record with the latest `tile_base` / `tile_m_index` evidence from `MEGAMOE_DCU_DEBUG_ROUTE=5,189,3478`.
- Normal rank5 slot1: row `1280`, `m_index=5`, `tile_base=1280`, `tile_m_index=5`; slot2: row `3329`, `m_index=13`, `tile_base=3328`, `tile_m_index=13`.
- Despite the correct tile headers, slot1 and slot2 still print identical K1 sample values and identical K2 raw FP8 samples. This resolves the fork toward K1 normal ASM expert stride/addressing or scalar-register clobbering after route metadata load, rather than compact tile metadata selection.
- Next action: inspect both K1 normal ASM sources for raw writes to `s96`/`sgprScaleFlag` and `s37`/`sgprStrideAK`, then choose the smallest source patch or discriminator run.

## 2026-07-02 01:24:00 +08:00 - K1 Python Reconstruction Confirms ASM Boundary
- Re-read `.planning/dcu_megamoe_supernode` and restored the post-compaction task state before continuing.
- Recorded the latest Pro EP16 default-normal K1 reference diagnostic from `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1ref1/normal_ep16.log`.
- For rank5 route `(source_rank=5, token=189, col=3478)`, slot1 local expert `5` and slot2 local expert `13` have valid row/tile metadata but identical normal K1 ASM samples.
- The Python K1 reconstruction from the normal path's own `staged_x`, row scale, `l1_weight`, and `l1_scale` gives different values for the two slots and differs from the ASM output.
- Updated conclusion: the first bad visible boundary is inside normal K1 ASM production. The next source read focuses on whether routed expert id and scale SRDs are applied after the compact-prebuilt branch, before global A/scale reads are consumed.

## 2026-07-02 01:29:00 +08:00 - Continue K1 Correctness Root Cause
- Re-read the active planning files after the user's reminder to keep `.planning` current.
- Current next check is non-GPU: inspect the runtime K1 normal `.co` objects with `dccobjdump` to ensure they contain the current compact-prebuilt expert routing and not stale Flash-era code.
- If the code objects are current, add a bounded discriminator that compares the observed K1 ASM row samples against Python references for all local experts for the same staged input row; this should identify whether the ASM is fixed to expert0, tile index, source rank, or another wrong expert id before the next patch.

## 2026-07-02 01:36:00 +08:00 - K1 Expert Discriminator And Patch
- First `.co` inspection attempt failed because a CRLF/pipe quoting path made bash see an incomplete loop. Retried with a UTF-8 base64 script; `dccobjdump` wrote ISA side-effect files into the remote repo root.
- Runtime K1 `.co` ISA contains the current dynamic compact metadata path, so the issue is not stale code objects.
- Added a debug-only all-expert K1 Python discriminator in `megamoe/opt.py`; local `py_compile` and `git diff --check` passed, and remote `python3 -m py_compile megamoe/opt.py` passed.
- Checked `10.17.151.1` before the GPU run: all 16 HCUs were 0% HCU/VRAM and `hy-smi --showpids` reported no KFD PIDs.
- The first diagnostic run failed before import because `PYTHONPATH=.` was missing; reran with `PYTHONPATH=.`.
- Pro EP16 default-normal discriminator log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1match2/normal_ep16.log` shows valid routed K1 rows consistently match local expert0 Python references, not their routed experts.
- Patched both K1 normal ASM sources so the compact-prebuilt branch reads `sgprScaleFlag` from `route_scratch_i32[tile_experts_offset + compact_tile]` instead of `m_indices[compact_tile * 256]`. Added source-guard assertions for the new side-channel read.

## 2026-07-02 01:47:00 +08:00 - Post Tile-Expert Patch Correctness Retest
- Re-read active planning files and confirmed the current Pro support plan before testing.
- Rechecked `10.17.151.1` before GPU work: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal default layout with the fresh K1 `.co` files containing the tile-expert side-channel patch.
- Result: correctness still failed. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1tileexpert1/normal_ep16_after_tileexpert.log`.
- Failure point: rank `12`, row `497`, col `1907`, `max_abs=0.228515625`, fused `-0.0849609375`, baseline `0.1435546875`, `stats_ok=True`.
- Combine debug now shows all six top-k slots are present and nonzero: `[-0.00265503, -0.00282288, -0.00161743, -0.0717773, -0.00558472, -0.000396729]`, owner ranks `[2,4,13,13,14,15]`, experts `[67,117,313,323,342,373]`.
- Interpretation: the tile-expert patch was necessary for the expert0 discriminator, but it is not sufficient for full Pro EP16 normal correctness. The new failure shape looks like wrong K1/K2 numeric content for routed experts rather than missing K1 rows or local reduce loss. Next targeted debug route is `MEGAMOE_DCU_DEBUG_ROUTE=12,497,1907`.

## 2026-07-02 01:56:00 +08:00 - Post Tile-Expert K1 Debug And Plain-Pack5 Check
- Ran targeted debug with `MEGAMOE_DCU_DEBUG_ROUTE=12,497,1907` after another clean card check. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_aftertile_debug1/normal_ep16_route_12_497_1907.log`.
- Parsed JSON debug lines from that log. Valid routed rows on ranks `2/4/13/14/15` still matched local expert0 for 26 or 27 of 27 sampled K1 columns, while the routed expert almost never won the all-expert discriminator.
- K2 rows inherit the same wrong K1 content: slots on the same rank can have identical raw FP8 samples with only route-weight-dependent scales.
- K3 remains internally consistent from visible K2 inputs: Python single-column refs match the combine slots for the selected column, and the combine slot sum equals fused output.
- Re-dumped runtime K1 `.co` with `dccobjdump --show-sass --inputs`; the side-channel patch is definitely present in both PACK5 and UNIFIED_PACK5 code objects. This rules out stale `.co` as the explanation.
- Ran `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT=normal` plain-pack5 correctness after a clean card check. It also failed, with the same rank0/row111/col4250 failure as the subsequent default-layout run. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_plain_aftertile1/plain_normal_ep16_after_tileexpert.log`.
- Current fork: either the value loaded into `sgprScaleFlag` is still effectively zero in prebuilt CTAs, or `sgprScaleFlag` is correct but the K1 A-weight/scale expert stride/SRD path ignores it. Next minimal discriminator is an ASM experiment forcing `sgprScaleFlag=1`.

## 2026-07-02 02:05:00 +08:00 - Force-Expert1 And Header Discriminator
- Ran a bounded temporary ASM experiment that forces `sgprScaleFlag=1` after the compact-prebuilt metadata path in both K1 normal ASM sources, rebuilt on `10.17.151.1`, and verified the runtime `.co` files contain `s_mov_b32 s96, 1`.
- With `MEGAMOE_DCU_DEBUG_ROUTE=12,497,1907`, the K1 all-local-expert discriminator switched from expert0 to expert1 for the valid routed rows: ranks `2/4/13/14/15` reported best expert1 for 26 or 27 of 27 sampled columns.
- Added Python-visible route-scratch header debug in `megamoe/opt.py` and reran the forced experiment. The parsed log showed `compact_tile_expert_header == tile_id == m_index` for all six selected valid K1 rows, with `compact_active_tiles=24`.
- Conclusion: the K1 A-weight/scale expert stride path can apply a nonzero expert id, and the HIP-built `tile_experts` side-channel is correct in Python-visible memory. The remaining fault is the normal non-forced ASM metadata read behaving like zero at load/use time.
- Next action before any real correctness retest: remove the temporary `DEBUG_FORCE_EXPERT1` line, then test a single metadata-read coherency hypothesis (`buffer_wbinvl1` and/or `glc` on the prebuilt `route_scratch` MUBUF metadata loads).

## 2026-07-02 02:12:00 +08:00 - K1 Metadata Coherency Patch Prepared
- Removed the temporary `DEBUG_FORCE_EXPERT1` line from both K1 normal ASM sources.
- Added one `buffer_wbinvl1` before the compact-prebuilt route-scratch metadata reads and added `glc` to both prebuilt `buffer_load_dword` metadata loads (`active_tiles` and `tile_experts`) in PACK5 and UNIFIED_PACK5.
- Added source guards in `test_dcu_megamoe_v3.py` to reject leftover `DEBUG_FORCE_EXPERT1` and to require the new metadata visibility marker.
- Local checks passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`, `git diff --check`, and grep verification for the K1 ASM metadata-load changes.
- Next step: check `10.17.151.1` card state, sync the touched files, run remote source tests/build, copy fresh K1 `.co` files into the runtime source tree, then verify ISA before GPU correctness.

## 2026-07-02 02:18:00 +08:00 - Remote Build And ISA Verification For Metadata Patch
- Checked `10.17.151.1` before remote work: all 16 HCUs showed 0% HCU/VRAM and `hy-smi --showpids` reported no KFD PIDs.
- Synced both K1 normal ASM sources, `megamoe/opt.py`, and `test_dcu_megamoe_v3.py` to `/root/yuguo/DeepGEMM`.
- Remote checks passed in docker `sglang_megamoe`: `PYTHONPATH=. python3 -m py_compile ...` and `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.52s`.
- Remote `python3 setup.py bdist_wheel` completed successfully. Fresh K1 PACK5 and UNIFIED_PACK5 `.co` files from `build/lib.../K1_fused/` were copied back into the runtime source tree.
- ISA verification with `dccobjdump --show-sass --inputs` under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1metaglc_build/isa` confirmed no force-expert marker and confirmed `buffer_wbinvl1` plus `buffer_load_dword ... glc` in both K1 code objects.
- Next step: recheck cards, rerun targeted Pro EP16 normal debug for `MEGAMOE_DCU_DEBUG_ROUTE=12,497,1907`, and inspect whether K1 matches routed experts instead of expert0.

## 2026-07-02 02:24:00 +08:00 - Runtime K1 Extension Staleness Found
- Rechecked cards before the targeted debug run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Targeted Pro EP16 normal debug after the metadata `glc/wbinvl1` patch still failed. The final argmax moved back to `rank=5,row=189,col=3478`, and route `12,497,1907` still showed all valid K1 rows matching local expert0: best0 was 26 or 27 of 27 samples; routed expert was 0 of 27.
- The cache/coherency hypothesis is therefore rejected.
- New build-artifact finding: the built K1 extension under `build/lib.../K1_fused/k1_fused_ext...so` contains the new `reserved_c4` guard string, but the runtime source-tree extension `megamoe/dcu_megamoe_opt/K1_fused/k1_fused_ext...so` is old (`2026-07-01 20:39:53`) and lacks that string.
- This explains why ASM read `GpuProb+0xc4` as effectively zero even though source and `.co` looked correct: the test imports from `PYTHONPATH=.` and was using the old source-tree K1 host launcher, not the freshly built extension.
- Next action: copy fresh built `.so` artifacts from `build/lib.../megamoe/` back into the source runtime tree, then rerun the same targeted K1 discriminator before changing ASM again.

## 2026-07-02 02:32:00 +08:00 - Fresh Runtime Extension Fixes K1 Metadata Path
- Copied fresh built extension artifacts from `build/lib.../megamoe/` into the source runtime tree on `10.17.151.1`, including `_C`, K1, K2, K3, and K3 V3 `.so` files.
- Verified the runtime source-tree K1 extension now contains the `K1 compact metadata exceeds packed reserved_c4 range` guard string and that `PYTHONPATH=.` imports the fresh source-tree extension.
- Rechecked cards before the GPU run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Targeted Pro EP16 normal default-layout correctness for route `MEGAMOE_DCU_DEBUG_ROUTE=12,497,1907` passed with `max_abs=0.000488281` and `mean_abs=1.21734e-05` against `normal-contiguous`.
- The K1 all-local-expert discriminator now matches routed experts instead of local expert0 for the selected route. Example: rank4 slot1 row5493 local expert21 now reports best expert21, and rank13 slots for local experts1/11 also match their routed local experts.
- Conclusion: the main correctness blocker was stale runtime extension launch code, not the normal baseline or K1 metadata cache coherency. Next step is to remove the temporary `buffer_wbinvl1/glc` experiment and retest so Flash performance does not inherit unnecessary metadata-load cost.

## 2026-07-02 02:46:00 +08:00 - Pro EP16 Passes Without Metadata `glc/wbinvl1`
- Removed the temporary K1 compact metadata `buffer_wbinvl1` plus `glc` MUBUF-load experiment from both PACK5 and UNIFIED_PACK5 K1 ASM sources.
- Local checks passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`, `git diff --check`, and grep confirmed only the source guard still mentions the removed marker.
- Rechecked `10.17.151.1` before build/test work: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Remote checks passed after syncing: source pytest `11 passed in 7.55s`, `python3 setup.py bdist_wheel` succeeded, fresh K1 `.co` and all built extension `.so` files were copied into the source runtime tree, and the runtime K1 `.so` contains the `reserved_c4` guard string.
- Rechecked cards before GPU correctness: all 16 HCUs idle, no KFD PIDs.
- Targeted Pro EP16 normal default-layout run without the metadata experiment passed for route `12,497,1907`: `max_abs=0.000488281`, `mean_abs=1.21734e-05`; K1 all-expert discriminator matched routed experts.
- No-debug Pro EP16 512-token smoke passed for both backends against `normal-contiguous`:
  - normal: `max_abs=0.000488281`, `mean_abs=1.21734e-05`;
  - ll: `max_abs=0.000488281`, `mean_abs=1.21163e-05`.
- Conclusion: Pro EP16 correctness at the current tested bucket no longer depends on the experimental metadata-load changes, so shared Flash K1 does not inherit that extra cache/load cost.

## 2026-07-02 02:56:00 +08:00 - Flash EP8 Guardrail After Pro Fix
- Rechecked `10.17.151.1` before the Flash run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Flash shape EP8 normal eager 4096 with `MEGAMOE_DCU_PEER_MEMORY=rpc`, `experts=256`, `topk=6`, `hidden=4096`, `intermediate=2048`, and `baseline-kind normal-contiguous`.
- Correctness passed: `max_abs=0.000671387`, `mean_abs=9.45305e-06`.
- Performance remained in the previous same-node band: MegaMoE median `5.783997 ms`, baseline median `9.882434 ms`, speedup `1.7086x`.
- Comparison to the prior planning reference (`5.7636 ms` MegaMoE, `10.0042 ms` baseline) is about `+0.35%` on MegaMoE, which is within run noise. No evidence currently supports splitting Pro-only normal ASM for Flash performance protection.

## 2026-07-02 03:06:00 +08:00 - Pro EP16 Normal 512 Performance
- Rechecked `10.17.151.1` before the Pro performance run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal eager 512 with default single-node IPC peer mode and `normal-contiguous` baseline.
- Correctness passed: `max_abs=0.000488281`, `mean_abs=1.21734e-05`.
- Performance: MegaMoE median `3.205919 ms`, baseline median `4.257838 ms`, speedup `1.3281x`.
- The TCPStore heartbeat warning appeared after the JSON result during distributed shutdown, but the command exit code was `0` and the correctness/performance JSON was complete.
- Next: run at least one larger Pro EP16 normal bucket to catch capacity/stride regressions beyond the already-fixed 512-token case.

## 2026-07-02 03:14:00 +08:00 - Pro EP16 Normal 1024 Performance
- Rechecked `10.17.151.1` before the 1024 run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal eager 1024 with `num_max_tokens_per_rank=1536`, default single-node IPC peer mode, and `normal-contiguous` baseline.
- Correctness passed: `max_abs=0.000976562`, `mean_abs=1.23586e-05`.
- Performance: MegaMoE median `5.068058 ms`, baseline median `7.143626 ms`, speedup `1.4095x`.
- This larger bucket exercises the Pro dynamic scratch/capacity/stride path beyond the initial 512 failure case and remains correct and faster than baseline.

## 2026-07-02 03:23:00 +08:00 - Pro EP16 Normal 2048 Failure
- Rechecked `10.17.151.1` before the 2048 run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal eager 2048 with `num_max_tokens_per_rank=3072`, default single-node IPC peer mode, and `normal-contiguous` baseline.
- Correctness failed before performance JSON. Failure from process/rank `1`: `max_abs=0.125`, argmax row `1672`, col `5051`, fused `0.0693359375`, baseline `-0.0556640625`, `stats_ok=True`.
- Post-failure card check showed all 16 HCUs back to idle and no KFD PIDs.
- Next debug route is `MEGAMOE_DCU_DEBUG_ROUTE=1,1672,5051` with `--skip-bench --correctness-iters 1 --debug-combine-on-fail`, to determine whether this medium-token issue is missing route/slot coverage, K1/K2 content, K3 combine, or local reduce.

## 2026-07-02 03:35:00 +08:00 - Pro EP16 2048 Debug Narrows Away From Reduce
- Targeted debug on route `1,1672,5051` and then `1,1672,4072` showed the debugged token's `combine_slot_sum` matches the fused value, so `reduce_local_combine` is not the failing layer.
- For debugged routes, K1 all-expert discriminator matched routed experts, K2 route metadata was valid, and K3 Python single-column reconstruction matched the combine slot values. The selected debug route can become locally correct while the run's max-diff moves to another token/column.
- This behavior suggests a broader medium-token normal-path issue rather than a single bad expert address. The first K2 active-tile-skip hypothesis was tested by setting `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=999999`, but 2048 still failed (`max_abs=0.14892578125`, argmax row `25`, col `1808`).
- Next discriminator: run Pro EP16 LL 2048 against the same `normal-contiguous` baseline. If LL passes, baseline is still credible and the bug remains normal-specific; if LL fails, inspect the baseline/test construction at 2048 before more normal ASM edits.

## 2026-07-02 03:42:00 +08:00 - K3 Publish Patch Prepared And Recorded
- Re-read active `.planning` files after the user's reminder to keep planning current.
- The previously pending LL discriminator is now complete: Pro EP16 LL 2048 passes against `normal-contiguous`, so the 2048 baseline is credible and the failure remains normal-only.
- Added and built a bounded K3 direct-combine publish patch in both K3 normal ASM direct-combine sources: `s_waitcnt vmcnt(0)` plus `buffer_wbinvl1_vol` immediately before final `s_endpgm`.
- Local checks already passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py` and `git diff --check`.
- Remote build on `10.17.151.1` docker `sglang_megamoe` already passed source pytest (`11 passed`) and `python3 setup.py bdist_wheel`; fresh K3 `.co` files and all built `.so` files were copied into the source runtime tree.
- Remote ISA verification confirmed both direct K3 runtime code objects end with `s_waitcnt vmcnt(0)`, `buffer_wbinvl1_vol`, `s_endpgm`.
- Next step: recheck card state, then run Pro EP16 normal 2048 correctness with the fresh K3 publish code objects.

## 2026-07-02 03:48:00 +08:00 - K3 Publish Patch Does Not Fix Pro 2048
- Checked `10.17.151.1` before the run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal 2048 correctness-only after the K3 publish patch. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k3publish_2048/normal_ep16_2048_correctness.log`.
- Result: still failed before performance. Process/rank `3` reported `max_abs=0.142578125`, argmax row `1054`, col `958`, fused `-0.1240234375`, baseline `0.0185546875`, `stats_ok=True`.
- Post-failure card check showed all 16 HCUs returned idle and no KFD PIDs.
- Conclusion: K3 final-store publish alone is not sufficient. Continue normal-only route diagnosis from `(source_rank=3, token=1054, col=958)` and keep the K3 publish patch as unproven until Flash guardrail and/or a positive Pro result justify it.

## 2026-07-02 04:10:00 +08:00 - Sync Diagnostics Point To K1 Output Publication
- Ran targeted debug for route `(3,1054,958)`. The debugged route became locally correct: K1 matched routed experts, K3 Python refs matched combine slots, and combine sum was `0.0132141`, close to the earlier baseline value `0.0185547`; the run's max diff moved to another route.
- Queried the DCU KB for Hygon/gfx936 cache/coherency guidance. The useful guidance is that `glc/slc` are real memory-path decisions and Hygon all-reduce references use release/acquire-style synchronization for visibility-sensitive barriers.
- `HIP_LAUNCH_BLOCKING=1` plus `CUDA_LAUNCH_BLOCKING=1` did not fix Pro EP16 normal 2048.
- Added a temporary env-gated Python diagnostic sync hook `MEGAMOE_DCU_DEBUG_SYNC_STAGE`. Tests with `k1`, `k2`, `k3`, and `all` each still failed, so plain stream/device synchronization is not sufficient.
- Targeted route `(6,142,3248)` compared the no-debug failure slot with debug refs: no-debug slot0 was `+0.0363769`, while targeted debug made the same K3 slot/ref about `-0.08984`, matching the baseline direction. The K1 debug for the same route matched routed experts.
- Source inspection found K1 main GEMM epilogues store `l1_out` via `buffer_store_short` and then reach `s_endpgm` without a final publish fence. Added a bounded K1 ASM experiment: `s_waitcnt vmcnt(0)` plus `buffer_wbinvl1_vol` before the main `Kernel End` in both PACK5 and UNIFIED K1 ASM sources.
- Next step: rebuild K1 code objects, copy fresh `.co` and `.so` artifacts, verify ISA contains the K1 publish marker, then rerun Pro EP16 normal 2048 correctness.

## 2026-07-02 04:18:00 +08:00 - K1 Publish Build And ISA Verified
- Synced the K1/K3 ASM sources, `megamoe/opt.py`, and `test_dcu_megamoe_v3.py` to `10.17.151.1:/root/yuguo/DeepGEMM`.
- Checked `10.17.151.1` before remote work: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Remote source checks passed in docker `sglang_megamoe`: `PYTHONPATH=. python3 -m py_compile ...` and `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.47s`.
- Remote `python3 setup.py bdist_wheel` completed; fresh K1 `.co` files and all built `.so` runtime extensions were copied from `build/lib.../megamoe/` into the source tree.
- Re-dumped K1 runtime code objects after clearing stale `dccobjdump` side-effect ISA files. Both PACK5 and UNIFIED K1 code objects contain the intended publish sequence before the two main kernel exits: `s_waitcnt vmcnt(0)`, `buffer_wbinvl1_vol`, `s_endpgm`.
- Next step: recheck card state, then run Pro EP16 normal 2048 correctness with the K1 publish patch and no diagnostic sync env.

## 2026-07-02 04:25:00 +08:00 - K1 Publish Patch Does Not Fix Pro 2048
- Rechecked `10.17.151.1` before the GPU run: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal 2048 correctness-only with fresh K1 publish code objects and no `MEGAMOE_DCU_DEBUG_SYNC_STAGE` or `MEGAMOE_DCU_DEBUG_ROUTE`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1publish_2048/normal_ep16_2048_correctness.log`.
- Result: still failed before performance. Process/rank `7` reported `max_abs=0.1193695068359375`, argmax row `1030`, col `6286`, fused `0.1162109375`, baseline `-0.0031585693359375`, `stats_ok=True`.
- Failure combine debug: slot sum `0.1162261962890625`; slots `[0.09765625, 0.0047607421875, 0.0030670166015625, 0.00201416015625, 0.0032958984375, 0.00543212890625]`; owners `[4,8,9,9,11,3]`; experts `[112,192,223,225,270,77]`.
- Post-failure card check showed all 16 HCUs returned idle and no KFD PIDs.
- Conclusion: K1 final-store publish alone is not sufficient. Next targeted debug route is `(source_rank=7, token=1030, col=6286)`, ideally with more selective intermediate readback controls to isolate whether K1 readback or K2/act readback is changing the route.

## 2026-07-02 04:40:00 +08:00 - Selective Route Readback And K2 GLC Probe
- Added a temporary diagnostic selector `MEGAMOE_DCU_DEBUG_ROUTE_STAGE` in `megamoe/opt.py` so route debug can read only `k1`, `k2`, `k3`, `meta`, or `combine`; default route-debug behavior remains all-stage when the selector is unset.
- Ran targeted route `(7,1030,6286)` with `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k1`. K1 rows for routed owner ranks matched the routed local experts, and the run's max diff moved to a different route `(rank=5,row=1993,col=4755)`. This shows K1-stage readback alone can perturb/fix the selected route, not only all-stage route debug.
- Queried the DCU KB for Hygon/gfx938 cache/coherency load guidance. Retrieved evidence says `glc/slc` are real memory-path controls, and Hygon/CK examples use `global_load_* ... glc slc` / `buffer_load_* ... glc slc` for cache-bypassing reads.
- Added a temporary env-gated K2 input-load experiment: `MEGAMOE_DCU_K2_GLC_SLC_LOAD=1` makes K2 read BF16 `l1_out` through inline asm `global_load_ushort ... glc slc`; default behavior remains normal loads.
- Remote source pytest passed (`11 passed in 7.36s`), K2 extension rebuilt, fresh `.so` was copied into the source tree, and `dccobjdump` confirmed the K2 code object contains `global_load_ushort ... glc slc`.
- Pro EP16 normal 2048 with `MEGAMOE_DCU_K2_GLC_SLC_LOAD=1` still failed. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k2_glc_2048/normal_ep16_2048_k2_glc_correctness.log`; failure `max_abs=0.1263427734375`, route `(rank=6,row=51,col=7152)`.
- Next step: run combine-only versus K1+combine for the original route `(7,1030,6286)` to verify whether K1 readback changes that route's actual combine slots, then decide whether the issue is K1 row visibility, K2 output visibility, or nondeterministic route timing.

## 2026-07-02 04:55:00 +08:00 - Combine-Only Versus K1+Combine Route Evidence
- Ran two selective route-stage tests in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_route_stage_7_1030_6286/` for original failing route `(source_rank=7, token=1030, col=6286)`.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=combine` did not fix the original route. The logged combine slot sum was `0.04688525199890137`, slots `[0.037353515625, 0.0047607421875, -0.0035247802734375, 4.076957702636719e-05, 0.0028228759765625, 0.00543212890625]`; the run still failed elsewhere at `(rank=4,row=1550,col=6879)`.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k1,combine` made the same original route match the earlier baseline within BF16 rounding. The logged combine slot sum was `-0.003143310546875`, slots `[-0.021240234375, 0.0047607421875, 0.0030670166015625, 0.00201416015625, 0.0028228759765625, 0.00543212890625]`; the run still failed elsewhere at `(rank=6,row=142,col=4245)`.
- Conclusion: K1 readback is the decisive perturbation for the selected route, and it specifically changes slot0 from wrong-positive to correct-negative. K2 `glc/slc` input loads, K1/K3 tail publish fences, and plain synchronization are already negative, so the next single-variable experiment should target K1 output store-side visibility/cache behavior.

## 2026-07-02 05:10:00 +08:00 - K1 Store-Side GLC Experiment Negative
- Applied a temporary K1 output-store experiment to both normal K1 ASM sources: all 640 PACK5 and all 640 UNIFIED `buffer_store_short ... // store D` instructions now use `offset:0 glc // store D`. Added a source guard expecting 1280 K1 store-D `glc` markers.
- Local checks passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`, `git diff --check`, and local marker counts of 640 per K1 ASM file.
- Checked `10.17.151.1` before remote work: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Synced the two K1 ASM files and `test_dcu_megamoe_v3.py`, then remote source checks passed in docker `sglang_megamoe`: `11 passed in 7.56s`.
- Remote build completed and fresh K1 `.co` plus runtime `.so` files were copied from `build/lib.../megamoe/` into the source-tree import path. `dccobjdump` confirmed 640 `buffer_store_short ... glc` instructions in each runtime K1 code object.
- Rechecked cards before correctness: all 16 HCUs idle, no KFD PIDs.
- Pro EP16 normal 2048 correctness-only still failed. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1_store_glc_2048/normal_ep16_2048_correctness.log`; failure `max_abs=0.10791015625`, route `(rank=14,row=1043,col=5051)`, fused `0.0888671875`, baseline `-0.01904296875`.
- Post-failure card check again showed all 16 HCUs idle and no KFD PIDs. Conclusion: K1 output-store `glc` alone is not sufficient; next minimal test is K1 store `glc` plus env-gated K2 `glc/slc` input loads.
- Rechecked cards before the combined run: all 16 HCUs idle, no KFD PIDs.
- Pro EP16 normal 2048 with K1 store `glc` plus `MEGAMOE_DCU_K2_GLC_SLC_LOAD=1` still failed. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1_store_glc_k2_load_glc_2048/normal_ep16_2048_correctness.log`; failure `max_abs=0.115234375`, route `(rank=4,row=1637,col=3261)`, fused `-0.0927734375`, baseline `0.0224609375`.
- Post-run card check showed all 16 HCUs idle and no KFD PIDs. Conclusion: the simple coherent store+load hypothesis is negative. Clean up K1/K3 publish and K1 store `glc` patches before further diagnosis.

## 2026-07-02 03:35:10 +08:00 - Clean Hot-Path Source Restored Locally
- Removed the negative K1 publish snippets from both normal K1 ASM sources and the negative K3 direct-combine publish snippets from both direct K3 ASM sources.
- Removed the temporary source guards that expected K1/K3 publish markers and K1 `offset:0 glc // store D` stores.
- Removed the env-gated K2 `MEGAMOE_DCU_K2_GLC_SLC_LOAD` diagnostic path, including the inline `global_load_ushort ... glc slc` helper and propagated `coherent_x_loads` launch arguments. This restores the default K2 read hot path instead of carrying a negative experiment.
- Local verification passed: marker search for `publish l1_out`, `publish direct combine`, `offset:0 glc // store D`, `MEGAMOE_DCU_K2_GLC_SLC_LOAD`, `global_load_u16_glc_slc`, and `coherent_x_loads` returned no matches; `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py` passed; `git diff --check` passed.
- Next step: sync cleaned sources to `10.17.151.1`, rebuild/copy fresh `.co` and `.so` artifacts into the source-tree runtime path, ISA-check that no negative experiment remains, then rerun Pro EP16 normal 2048 correctness on the clean baseline.

## 2026-07-02 03:38:00 +08:00 - Clean Hot-Path Remote Rebuild Complete
- Checked `10.17.151.1` before remote work: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Synced six cleaned source files to `/root/yuguo/DeepGEMM`: two K1 normal ASM sources, K2 fused extension source, two K3 direct-combine ASM sources, and `test_dcu_megamoe_v3.py`.
- First remote marker grep failed because stale generated artifacts from the previous K2 experiment remained (`k2_fused_ext.hip`, old K2 `.so`, and pycache/binary matches), while the real `.cu` source had already been cleaned. Removed only the stale K2 generated `.hip` and old K2 `.so` inside the repo path, then reran the source-only marker check.
- Remote source checks passed: `PYTHONPATH=. python3 -m py_compile ...` and `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.45s`.
- Remote `python3 setup.py bdist_wheel` completed; fresh K1/K3 `.co` files and all built `.so` runtime extensions were copied from `build/lib.linux-x86_64-cpython-310/megamoe/` into the source tree.
- Runtime artifact check passed: source negative markers absent, K1 runtime ISA has zero `buffer_store_short.*glc`, K2 runtime `.so` has zero `MEGAMOE_DCU_K2_GLC_SLC_LOAD` strings, and regenerated K2 `.hip` has no `glc/slc` diagnostic markers.
- Note: K1 runtime ISA still contains two `buffer_wbinvl1_vol` instructions from the pre-existing source `buffer_wbinvl1` path, not from the removed publish patch. The removed publish marker and store-side `glc` experiment are absent.
- Next step: recheck cards, then run Pro EP16 normal 2048 correctness-only on the clean rebuilt runtime.

## 2026-07-02 03:42:00 +08:00 - Clean Pro EP16 Normal 2048 Failure Reproduced
- Rechecked cards before the clean correctness run: all 16 HCUs idle, no KFD PIDs.
- Ran Pro EP16 normal 2048 correctness-only with clean rebuilt runtime, no `MEGAMOE_DCU_DEBUG_ROUTE`, no `MEGAMOE_DCU_DEBUG_ROUTE_STAGE`, no `MEGAMOE_DCU_DEBUG_SYNC_STAGE`, and no `MEGAMOE_DCU_K2_GLC_SLC_LOAD`.
- Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_clean_hotpath_2048/normal_ep16_2048_correctness.log`.
- Result: still failed, proving the remaining issue is in the actual Pro normal hot path rather than the discarded publish/glc experiments. Failure: `max_abs=0.12548828125`, route `(source_rank=10, token=559, col=3146)`, fused `0.078125`, baseline `-0.04736328125`, `stats_ok=True`.
- Failure combine slots: `[-0.0057373046875, -0.0038909912109375, 0.004638671875, 0.0771484375, -0.00250244140625, 0.00836181640625]`; slot3 dominates the wrong sign. Owners `[2,2,3,3,5,6]`, experts `[58,63,86,91,133,153]`.
- Post-run card check showed all 16 HCUs idle and no KFD PIDs.
- Next step: run selective route diagnostics for `(10,559,3146)`, starting with `combine` versus `k1,combine`, to see whether clean K1 readback still route-locally flips the bad slot.

## 2026-07-02 03:48:00 +08:00 - Clean Route Selective Readback Result
- Ran selective route diagnostics for `(source_rank=10, token=559, col=3146)` in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_clean_route_10_559_3146/`. Cards were idle before and after the batch; `hy-smi --showpids` reported no KFD PIDs.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=combine` changed the original route materially versus the no-debug failure: combine slot sum `-0.035327911376953125`; slots `[-0.000156402587890625, 0.0024871826171875, 0.00823974609375, -0.0517578125, -0.00250244140625, 0.00836181640625]`. The previous no-debug slot3 was `+0.0771484375`; combine readback sees slot3 as `-0.0517578125`.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k1,combine` moved the same route slightly closer to the no-debug baseline value: combine slot sum `-0.042999267578125`; slots `[-0.0057373046875, 0.00531005859375, 0.00823974609375, -0.0517578125, -0.007415771484375, 0.00836181640625]`. The no-debug baseline value was `-0.04736328125`.
- Both debug runs still failed elsewhere, so this is a route-local perturbation rather than a full fix.
- New interpretation: clean evidence now implicates the consumer/read-side of the combine buffer at least as strongly as K1. A destination-rank combine CPU readback after K3 can reveal/fix the large wrong slot before local reduce. Next minimal probe should target `reduce_local_combine` reading combine values, for example an env-gated coherent combine load, before more K1-side experiments.

## 2026-07-02 03:55:00 +08:00 - Reduce Combine Read-Side Probe Prepared Locally
- Read `reduce_local_combine_vec_kernel` in `megamoe/dcu_megamoe_opt/K3_fused/k3_fused_ext.cu`. It already issues `buffer_wbinvl1_vol` at kernel entry, but the per-slot combine payload read is a normal `uint4` load from `local_sections.combine`.
- Queried the DCU KB and local source for a source-backed vector coherent load form. Existing project code has `global_load_uint4_device`, and the Hygon KB microbenchmark examples show `global_load_dwordx4 ... off glc slc` with `s_waitcnt vmcnt(0)`.
- Added a temporary env-gated diagnostic `MEGAMOE_DCU_REDUCE_COMBINE_GLC_SLC_LOAD`: when set, `reduce_local_combine_vec_kernel` loads combine `uint4` values through `global_load_dwordx4 ... glc slc`; default behavior remains the normal `uint4` load.
- Local verification passed: `git diff --check` and `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`.
- Next step: sync `k3_fused_ext.cu`, force-regenerate remote `k3_fused_ext.hip`, rebuild/copy fresh `.so`, verify emitted `global_load_dwordx4 ... glc slc`, and run Pro EP16 normal 2048 with the env flag enabled.

## 2026-07-02 04:00:00 +08:00 - Reduce Combine Read-Side Probe Built Remotely
- Synced `megamoe/dcu_megamoe_opt/K3_fused/k3_fused_ext.cu` to `10.17.151.1`.
- Deleted stale generated `k3_fused_ext.hip`, old source-tree `k3_fused_ext*.so`, and old build object for `k3_fused_ext.o` inside `/root/yuguo/DeepGEMM` to force hipify and recompilation.
- Remote source checks passed: `PYTHONPATH=. python3 -m py_compile ...` and `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.30s`.
- Remote build regenerated `k3_fused_ext.hip`, recompiled `k3_fused_ext`, and copied fresh `.so` files from `build/lib.linux-x86_64-cpython-310/megamoe/` into the source-tree runtime path.
- Verification: regenerated `k3_fused_ext.hip` contains the env gate and `global_load_dwordx4 ... glc slc`; source-tree `k3_fused_ext*.so` contains `MEGAMOE_DCU_REDUCE_COMBINE_GLC_SLC_LOAD`; `dccobjdump` on the runtime `.so` reports one `global_load_dwordx4` with `glc` and `slc`.
- Next step: check card state, then run Pro EP16 normal 2048 correctness-only with `MEGAMOE_DCU_REDUCE_COMBINE_GLC_SLC_LOAD=1`.

## 2026-07-02 04:05:00 +08:00 - Reduce Combine Read-Side Probe Negative
- Checked cards before the test: all 16 HCUs idle, no KFD PIDs.
- Ran Pro EP16 normal 2048 correctness-only with `MEGAMOE_DCU_REDUCE_COMBINE_GLC_SLC_LOAD=1`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_reduce_glc_2048/normal_ep16_2048_reduce_glc_correctness.log`.
- Result: still failed. Failure `max_abs=0.121551513671875`, route `(source_rank=5, token=610, col=1383)`, fused `0.005401611328125`, baseline `0.126953125`, `stats_ok=True`.
- Failure combine slots under the reduce-load probe: `[0.000675201416015625, 0.002044677734375, 6.198883056640625e-05, 0.0048828125, -0.007293701171875, 0.005035400390625]`; slot4 has the dominant top-k weight (`0.5690420866012573`) but contributes only `-0.007293701171875`.
- Post-run `hy-smi --showpids` reported no KFD PIDs.
- Conclusion: making `reduce_local_combine` load combine through `global_load_dwordx4 ... glc slc` is not sufficient. The combine buffer values at failure time are already wrong/small, so the next diagnosis should compare K2/K3 intermediate values for route `(5,610,1383)` against the K3 Python reconstruction and, if needed, LL/baseline intermediate values.

## 2026-07-02 04:12:00 +08:00 - K2/K3 Intermediate Comparison For Route 5,610,1383
- Ran `MEGAMOE_DCU_DEBUG_ROUTE=5,610,1383` with `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k2,k3,combine`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k2_k3_route_5_610_1383/normal_ep16_2048_stage_k2_k3_combine.log`.
- The debugged original route became locally correct while the run failed elsewhere. Combine slot sum was `0.13124465942382812`, close to the reduce-probe failure baseline `0.126953125`.
- Slot4, the dominant missing slot in the reduce-load-probe failure, had visible K2 input `act_scale=0.00030274895834736526`, `act_dequant_absmax=0.1356315314769745`, route weight `0.5690420866012573`, and K3 Python reconstruction `0.11482040584087372`.
- The combine slot4 value under debug was `0.11474609375`, matching the K3 Python reconstruction within BF16 rounding. Other slots also matched their Python refs: slot0 `0.000675`, slot1 `0.004369`, slot2 `0.004860`, slot3 `0.004883`, slot5 `0.001725`.
- Interpretation: for this route, the visible K2 outputs and K3 math are correct under debug; the no-debug failure is not explained by a bad K3 formula or bad L2 weight addressing. The decisive side effect could still be K2 readback before K3, K3 debug readback, or combine readback before reduce.
- Next step: run a stage matrix for the same route with `combine`, `k3,combine`, and `k2,combine` to isolate which readback boundary makes the combine slot become correct.

## 2026-07-02 04:01:00 +08:00 - Stage Matrix For Route 5,610,1383 Logged
- Recorded the stage-matrix result from `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_stage_matrix_5_610_1383/` into planning so the next loop starts from the latest evidence.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=combine`: original route only partially improved. Slots were `[0.000675201416015625, 0.004364013671875, -0.000152587890625, 0.0048828125, 0.0211181640625, 0.003265380859375]`, sum `0.034152984619140625`; slot4 was still far below the Python K3 ref `0.11482040584087372`.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k3,combine`: original route became locally correct. K3 refs were `[0.004724027588963509, 0.0043694325722754, -0.003040947951376438, 0.0048825559206306934, 0.11482040584087372, -0.003940647933632135]`; combine slots were `[0.004730224609375, 0.004364013671875, -0.0030364990234375, 0.0048828125, 0.11474609375, -0.003936767578125]`, sum `0.1217498779296875`.
- `MEGAMOE_DCU_DEBUG_ROUTE_STAGE=k2,combine`: original route also had the dominant slot correct. Slots were `[0.000675201416015625, 0.001983642578125, 0.003570556640625, 0.0048828125, 0.11474609375, 0.005035400390625]`, sum `0.13089370727539062`.
- Interpretation update: K3 math and visible-input addressing are not the root cause. The next single-variable probe is a controlled host delay after the K3 launch and before the post-K3 rank barrier, because K3 debug readback sits in that window and fixes the dominant slot while plain synchronize alone did not.

## 2026-07-02 04:01:00 +08:00 - K3-Before-Barrier Delay Probe Prepared
- Added a temporary default-off Python diagnostic in `megamoe/opt.py`: `MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3_MS=<ms>` sleeps after the K3 launch/debug/sync hook and before the post-K3 rank barrier; `MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3_SYNC=1` first calls `torch.cuda.synchronize()`.
- This does not change the default hot path and does not touch K1/K2/K3 code objects. It is only to distinguish pure host-side timing from a missing memory-visibility/publish operation.
- Local checks passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py` and `git diff --check`.
- Synced `megamoe/opt.py` to 151.1 and remote source pytest passed: `11 passed in 7.32s`.
- Card check before GPU testing showed all 16 HCUs idle with no KFD PIDs.
- Pro EP16 normal 2048 with `MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3_MS=10` still failed: log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k3_delay_probe_2048/sleep10_nosync.log`, `max_abs=0.118896484375`, route `(rank=5,row=671,col=6217)`.
- Pro EP16 normal 2048 with `MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3_MS=10` plus `MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3_SYNC=1` also failed: log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k3_delay_probe_2048/sleep10_sync.log`, `max_abs=0.126220703125`, route `(rank=0,row=1596,col=7019)`.
- Post-run card check showed no KFD PIDs. Conclusion: the K3 debug/readback side effect is not explained by a simple host delay before the post-K3 barrier.

## 2026-07-02 04:01:00 +08:00 - K3 Store-Side GLC Probe Prepared Locally
- Tail-reduce control with `K3_USE_ASM_TAIL_REDUCE=1` still failed Pro EP16 normal 2048: log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_tailreduce_probe_2048/normal_ep16_2048_tailreduce_correctness.log`, `max_abs=0.116455078125`, route `(rank=13,row=1142,col=3835)`.
- Added a temporary direct-K3 ASM store-side probe in the two direct combine code objects only: `K3COMBINE_PACK5.s` and `K3COMBINE_UNIFIED_PACK5.s` now mark the K3 remote combine `global_store_short` and `global_store_dwordx4` writes with `glc`.
- Scope: direct normal K3 only; tail-reduce ASM was not changed by this probe.
- Local checks passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`, `git diff --check`, and marker grep found 10 `K3 combine store glc probe` sites per direct K3 ASM source.
- Remote source pytest passed (`11 passed in 7.27s`), build completed, fresh direct K3 `.co` files and runtime `.so` files were copied into the source tree.
- `dccobjdump` side-effect ISA files confirmed the direct PACK5 and UNIFIED code objects each have 136 `global_store_short/global_store_dwordx4 ... glc` instructions; tail-reduce code objects remained unchanged.
- Card check before correctness showed all 16 HCUs idle with no KFD PIDs.
- Pro EP16 normal 2048 with K3 direct combine store-side `glc` still failed: log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k3_store_glc_2048/normal_ep16_2048_k3_store_glc_correctness.log`, `max_abs=0.123809814453125`, route `(rank=10,row=1563,col=427)`.
- Conclusion: K3 remote combine store `glc` alone is not sufficient. This probe should be removed before final/performance unless a later combined fix proves it necessary.

## 2026-07-02 04:20:00 +08:00 - Planning Updated For Same-Input Stage Compare
- Re-read the active supernode planning files after compaction and updated `task_plan.md` so the Pro EP16 active work no longer points at the already-tested K3 delay/timing split.
- Added a findings entry recording that K3 delay, sleep+sync, tail-reduce control, and direct K3 store-side `glc` are all negative.
- Current next step: restore a clean K3 hot path, then add/run a same-input normal-vs-LL/baseline stage comparison for Pro EP16 2048 to locate the first divergent stage before making another correctness patch.

## 2026-07-02 04:35:00 +08:00 - Clean Hot-Path And Same-Input Diagnostic Prepared Locally
- Removed the negative direct-K3 store-side `glc` probe from both K3 direct-combine ASM sources.
- Removed the negative env-gated `reduce_local_combine` `global_load_dwordx4 ... glc slc` probe and restored normal `uint4` combine loads.
- Removed the negative K3 post-launch sleep diagnostic from `opt.py`; route-stage debug and explicit sync-stage debug remain available.
- Added `--debug-compare-backends-on-fail` to `test_mega_moe_dcu.py`. On a correctness failure, all ranks now gather the failing route, rerun normal and LL on the same generated input with `MEGAMOE_DCU_DEBUG_ROUTE` enabled, print `MEGAMOE_DCU_DEBUG_BACKEND_COMPARE`, and then exit together. This follows the same-input comparison suggested by the user without risking rank-barrier deadlock.
- Local verification passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_mega_moe_dcu.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`, `git diff --check`, and marker grep for the removed K3/reduce/sleep probes returned no matches.

## 2026-07-02 05:05:00 +08:00 - Same-Input Diagnostic Finds Normal K1 Source-X Mismatch
- Synced the clean K3/reduce/sleep cleanup and same-input compare diagnostic to `10.17.151.1`, rebuilt K3 artifacts, copied fresh `.co`/`.so`, and verified all K3 store-side `glc` counts are 0 in generated ISA.
- First same-input compare exposed a debug-helper layout bug: passing a bare unified weight tuple made the wrapper report `use_unified_weight_layout=false`. Fixed the helper to pass keyed `{"unified": ...}` weights and reran.
- Corrected same-input compare v2 selected route `(4,1544,6434)`: normal debug rerun value `-0.042724609375` vs baseline `-0.0478515625`, LL exact. Normal and LL differed in K2/K3 refs for slot0/slot5.
- Added debug-only `MEGAMOE_DCU_DEBUG_COMPARE_SOURCE_X=1`, which broadcasts the selected source rank's FP8 input row and compares it to normal K1 staged rows.
- v3 selected route `(1,1819,6306)` found the first bad boundary: normal K1 staged FP8 `x` bytes are wrong for some slots while `x_scale` is correct. Slot1/slot3 on rank7 had 13 sampled raw-byte mismatches and slot0 on rank4 had 12; unaffected slots had 0. Normal K1 still matches Python reconstruction from its own staged input, and normal K3 combine matches its own visible K2 input.
- Current conclusion: fix normal K1 input staging / remote source-x read path for Pro hidden=7168; K2/K3/reduce are downstream consumers of already-wrong staged x for the failing slots.

## 2026-07-02 04:39:47 +08:00 - K1 Rank-Local Source-Rank Patch Prepared Locally
- Re-read the active planning files and K1 normal source/ASM after compaction.
- Root-cause hypothesis: Pro hidden=7168 creates 224 32-byte source-x chunks per row, so a 64-lane wave can contain the tail of one staged row and the head of the next. The existing rank-local load path used `v_readfirstlane_b32 s63, v254` once, which can apply the previous row's source-rank peer base to the next row's low chunks.
- Patched both normal K1 ASM sources so the rank-local branch loops over source-rank groups inside the current wave: select one `v254` rank, load its peer base, issue the two dwordx4 loads under the matching exec mask, remove those lanes from the pending mask, and repeat until all valid lanes are loaded.
- Local verification passed: `git diff --check` and `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_mega_moe_dcu.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`.
- Next step: check 151.1 card state, sync the two K1 ASM files, rebuild/copy fresh `.co`/`.so`, verify the new `label_SymmStageRankLocalLoop` appears in generated ISA, then rerun the same-input source-x diagnostic and Pro EP16 normal 2048 correctness.

## 2026-07-02 04:48:47 +08:00 - K1 Source-X Patch Verified, Remaining K1 Output Mismatch
- Remote card check before the K1 test loop showed all 16 HCUs idle and no KFD PIDs.
- Synced the two K1 normal ASM files, rebuilt on `10.17.151.1`, copied fresh runtime `.so` artifacts from `build/lib.linux-x86_64-cpython-310/megamoe/` into the source-tree import path, and kept the fresh `.co` artifacts in `megamoe/dcu_megamoe_opt/K1_fused/`.
- Remote source verification passed: `PYTHONPATH=. python3 -m py_compile ...` and `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed`.
- Targeted same-input route diagnostic for `(source_rank=1, token=1819, col=6306)` now reports `x_mismatch=0` for all sampled K1 rows. This verifies the rank-local source-rank patch fixed the previously observed wrong FP8 source-x staging.
- Pro EP16 normal 2048 still fails overall: log `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1_ranklocal_patch_route_1_1819_6306/normal_ep16_2048_route_source_x.log`, failure `max_abs=0.1728515625` at route `(source_rank=1, token=569, col=541)`.
- Current boundary moved forward: normal K1 source-x bytes are now correct for the targeted route, but K1 output samples can still diverge from the Python K1 reference for low L1 output columns. First local follow-up is to audit the new rank-local loop's SGPR/mask safety, then inspect K1 packed A-weight and scale addressing.

## 2026-07-02 07:25:22 +08:00 - Same-Input Stage Order Locked Into Plan
- User corrected the diagnostic order: first compare K1 under the same input; if K1 does not pass, debug only K1. Only after K1 passes should K2 be compared, and only after K2 passes should K3/combine be compared.
- Updated `task_plan.md` with this gate order and the row-alignment rule: do not assume normal and LL physical scratch rows are identical until verified. If row ids match, compare directly; otherwise align by `(source_rank, token, topk slot, routed expert)`.
- Next action before more ASM changes: implement or run a same-input normal-vs-LL K1 output comparison that reports mapping equality, max abs delta, argmax column, and top delta columns per routed slot.

## 2026-07-02 07:37:22 +08:00 - K1-Only Same-Input Compare Instrumentation
- Implemented the first ordered diagnostic gate locally. `test_mega_moe_dcu.py --debug-compare-backends-on-fail` now defaults `--debug-compare-backend-stages=k1`, sets debug-only K1 capture, stops the compare reruns immediately after K1, and prints `MEGAMOE_DCU_DEBUG_K1_COMPARE` after normal and LL have run on the same input.
- Added debug-only K1 snapshots in `megamoe/opt.py`; default hot path is unchanged unless `MEGAMOE_DCU_DEBUG_CAPTURE_K1=1` or `MEGAMOE_DCU_DEBUG_STOP_AFTER_K1=1` is set.
- Local verification passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_mega_moe_dcu.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py` and `git diff --check`.
- Next step: check 151.1 card state, sync the two changed Python files, then run Pro EP16 normal 2048 with the K1-only compare to decide whether K1 output itself is the failing boundary.

## 2026-07-02 07:41:05 +08:00 - K1-Only Same-Input Compare Fails
- Checked 151.1 cards before the run: all 16 HCUs idle, no KFD PIDs.
- Synced `megamoe/opt.py` and `test_mega_moe_dcu.py`; remote `py_compile` passed and source pytest reported `11 passed`.
- Ran Pro EP16 normal 2048 correctness with `--debug-compare-backends-on-fail`, defaulting to K1-only compare. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1_only_compare_2048/normal_ep16_2048_k1_only_compare.log`.
- Final correctness still failed, but the ordered diagnostic produced the needed first-stage evidence: normal-vs-LL K1 full-row comparisons differ for all six selected slots even after aligning by route slot and route weight. Physical rows differ, as expected, so the comparison used slot/logical route alignment.
- K1 compare maxima: slot0 rank2 `1.10546875`, slot1 rank3 `1.091796875`, slot2 rank4 `1.158203125`, slot3 rank5 `0.998046875`, slot4 rank12 `1.125`, slot5 rank3 `0.96875`. Route weights matched between normal and LL.
- Current conclusion: K1 output itself is the first failing boundary. Continue only in K1, using the user's suggested targeted diff against `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` to find fused ASM modifications that retained Flash-specific constants or assumptions.

## 2026-07-02 07:55:16 +08:00 - K1 Low-Tile Staging Clobber Candidate Patched Locally
- Compared current K1 ASM against `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` instead of broad scanning.
- Rechecked 151.1 before the diagnostic: all 16 HCUs showed `0%` VRAM/HCU, and no matching python/pytest/torchrun processes were present.
- Ran Pro EP16 2048 K1-only same-input compare with `MEGAMOE_DCU_DEBUG_K1_FULL_ROW=1` and full source-x comparison. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1_fullrow_compare_2048/normal_ep16_2048_k1_fullrow_compare.log`.
- Evidence: source-x full raw mismatch count is `0`, LL and Python K1 refs match, and normal K1 errors concentrate in N tiles `0..7` (`col < 2048`). Later samples such as `3068..3075`, `4608`, and `6140..6143` match within BF16 tolerance.
- Root-cause candidate: the rank-local staging loop reuses `s62` for source-rank pointer slots, then the original B-address rebuild adds `s62` to `sgprAddressB`. Only `wg0 < 8` CTAs participate in staging, explaining why only low-column K1 tiles are wrong.
- Patched both K1 normal ASM variants (`PACK5` and `UNIFIED_PACK5`) to set `s62 = 0` immediately before adding the staged_x plane offset to `sgprAddressB`.
- Local verification passed: `git diff --check` and `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_mega_moe_dcu.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`.
- Next step: sync the two K1 ASM files to 151.1, rebuild/copy fresh K1 `.co` and runtime `.so`, verify the `s_mov_b32 s62, 0` marker in generated ISA/source, then rerun K1-only compare before touching K2/K3.

## 2026-07-02 08:08:00 +08:00 - K1 `s62` Restore Fix Verified On Pro EP16 2048
- Rechecked 151.1 card state before using the node: all 16 HCUs showed `0%` VRAM/HCU and `hy-smi --showpids` reported no KFD PIDs. Host-side `pgrep` still showed old `sglang` process names, but they were not occupying DCUs.
- Remote K1 code objects were fresh after the rebuild: `DISPATCH_PULL_L1_PACK5.co` and `DISPATCH_PULL_L1_UNIFIED_PACK5.co` both had timestamp `7月 2 07:56`.
- Ran Pro EP16 2048 with the failure-triggered K1-only compare flag after the `s62` restore. Because the main correctness path passed, the failure rerun did not trigger.
- Result: full correctness passed, `max_abs=0.000976562`, `mean_abs=1.22387e-05`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_k1_s62_restore_compare_2048/normal_ep16_2048_k1_s62_restore_compare.log`.
- Post-run `hy-smi --showpids` again reported no KFD PIDs.
- Interpretation: the first divergent K1 boundary is fixed for the active Pro EP16 2048 bucket. Since end-to-end correctness now passes, do not continue splitting K2/K3 for this bucket; resume ordered K2/K3 stage compare only if a later bucket fails.

## 2026-07-02 08:13:00 +08:00 - Pro EP16 2048 No-Debug Performance After K1 Fix
- Rechecked 151.1 cards at the start of the run: all 16 HCUs showed `0%` VRAM/HCU and no KFD PIDs.
- Ran no-debug Pro EP16 normal 2048 against `normal-contiguous` baseline. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_s62_restore_2048_perf/normal_ep16_2048_correctness_perf.log`.
- Correctness passed: `max_abs=0.000976562`, `mean_abs=1.22387e-05`.
- Performance: MegaMoE fused median average per rank `8.0324 ms`; normal-contiguous baseline median average per rank `13.2038 ms`; reported speedup `1.6438x`.
- Post-run `hy-smi --showpids` reported no KFD PIDs.
- Next step: run Flash EP8 4096 guardrail because the `s62` restore touches shared K1 normal ASM. If Flash regresses materially, split Pro-only normal ASM code objects.

## 2026-07-02 08:18:00 +08:00 - Flash EP8 Guardrail After K1 `s62` Fix
- Rechecked 151.1 cards before the run: all 16 HCUs showed `0%` VRAM/HCU and no KFD PIDs.
- Ran Flash EP8 normal 4096 with `MEGAMOE_DCU_PEER_MEMORY=rpc` against `normal-contiguous`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/flash_ep8_20260702_s62_guardrail_4096/normal_ep8_4096_correctness_perf.log`.
- Correctness passed: `max_abs=0.000671387`, `mean_abs=9.45305e-06`.
- Performance: MegaMoE fused median average per rank `5.8028 ms`; baseline `10.0556 ms`; speedup `1.7329x`.
- Prior same-node Flash reference after the earlier Pro changes was `5.784 ms`, so the K1 `s62` restore shows only a noise-level delta. No Flash performance reason currently exists to split a Pro-only K1 normal ASM.
- Post-run `hy-smi --showpids` reported no KFD PIDs.
- Next step: run one broader Pro EP16 4096 guardrail to make sure the 2048 fix is not bucket-specific.

## 2026-07-02 08:23:00 +08:00 - Pro EP16 4096 Guardrail After K1 Fix
- Rechecked 151.1 cards before the run: all 16 HCUs showed `0%` VRAM/HCU and no KFD PIDs.
- Ran no-debug Pro EP16 normal 4096 against `normal-contiguous`. Log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_s62_restore_4096_perf/normal_ep16_4096_correctness_perf.log`.
- Correctness passed: `max_abs=0.000991821`, `mean_abs=1.22245e-05`.
- Performance: MegaMoE fused median average per rank `15.7914 ms`; normal-contiguous baseline `25.3273 ms`; speedup `1.6039x`.
- Post-run `hy-smi --showpids` reported no KFD PIDs.
- Current validation status after the K1 `s62` restore: active Pro EP16 normal buckets `2048` and `4096` pass, and Flash EP8 4096 guardrail remains in the same performance band.

## 2026-07-02 08:45:00 +08:00 - Temporary Debug Code Cleaned
- Removed temporary Python route/stage debug code from `megamoe/opt.py`: the `MEGAMOE_DCU_DEBUG_*` hooks, K1 snapshot cache, same-input K1 capture, source-x compare, K2/K3/combine value dumps, stage sync hook, and stop-after-K1 early return.
- Removed failure-debug CLI plumbing from `test_mega_moe_dcu.py`: `--debug-combine-on-fail`, `--debug-compare-backends-on-fail`, backend rerun helpers, K1 snapshot compare output, and combine-slot failure dumps. The normal correctness failure still reports the argmax row/col, fused value, baseline value, and stats state.
- Added source-level regression assertions in `test_dcu_megamoe_v3.py` so the retired `MEGAMOE_DCU_DEBUG_*` hooks and debug CLI flags cannot reappear in production/test harness sources.
- Local verification passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_mega_moe_dcu.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py`, `git diff --check`, and `rg` found no remaining debug hooks in `megamoe/opt.py` or `test_mega_moe_dcu.py`.
- Remote source verification on `10.17.151.1` after syncing the three Python files passed: `python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.83s`, and the remote debug-residue grep reported clean.

## 2026-07-02 11:00:32 +08:00 - Pro Weight Layout Env Removed
- Removed the test-only `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT` override from `test_mega_moe_dcu.py`; it was only introduced to isolate whether Pro normal failures were unified-layout-specific during debugging.
- Restored the harness to the general V3 layout rule: `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` selects unified weights; otherwise normal backend uses normal ASM pack5 and LL uses unified pack5.
- Added `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT` to the retired-source assertions in `test_dcu_megamoe_v3.py`, so the Pro-specific override cannot silently return.
- Local verification passed: `python -m py_compile ...`, `git diff --check`, and residue grep confirmed the Pro layout env/override no longer appears in `test_mega_moe_dcu.py`. Local pytest is unavailable because the Windows Python environment has no `pytest`.
- Remote source verification on `10.17.151.1` passed after sourcing DTK and setting the DTK/amdsmi `LD_LIBRARY_PATH`: `python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed in 7.55s`, and the remote residue grep printed `PRO_LAYOUT_ENV_CLEANUP_OK`.
- Checked 151.1 card state before the default-layout smoke: all 16 HCUs were idle and `hy-smi --showpids` reported no KFD PIDs.
- Ran Pro EP16 normal 2048 correctness-only with the cleaned default harness layout. The run selected `weight_layout=normal` and passed with `max_abs=0.000976562`, `mean_abs=1.22387e-05`; log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/pro_ep16_20260702_cleanup_default_normal_2048/normal_ep16_2048_default_layout_correctness.log`.
- Post-run `hy-smi --showpids` again reported no KFD PIDs.
- Removed the extra eager correctness failure argmax/value formatting from `test_mega_moe_dcu.py` and restored the original one-line `max_abs` assertion. This keeps the harness cleanup strict now that the targeted debug reruns are gone. Local `py_compile`/`git diff --check` passed, and the synced 151.1 test file reported `EAGER_ASSERT_CLEANUP_OK`.

## 2026-07-02 13:03:07 +08:00 - Task Plan Status Audit
- Updated `.planning/dcu_megamoe_supernode/task_plan.md` to separate current Pro status into `✅`, `🚫`, and `[ ]` entries.
- Marked the completed Pro EP16 validation path as `✅`: source/dispatch gates, normal/LL EP16 correctness, 2048/4096 performance, Flash EP8 guardrail, debug cleanup, Pro layout override cleanup, and eager assertion cleanup.
- Marked negative diagnostic branches as `🚫`: K1/K3 publish fences, K1 store-side `glc`, K2/reduce `glc/slc`, K3 sleep/sync/tail-reduce/store probes, and the Pro-only layout override.
- Left only real open/deferred items as `[ ]`: packaging/install flow for stale source-tree `.so` imports, Pro EP8/EP32 runtime validation, and conditional Pro-only kernel split if a future Flash regression appears.

## 2026-07-02 14:37:33 +08:00 - Pod2 Environment Recorded And Jump Login Checked
- Read `dcu-supernode-workflow` and current `.planning/dcu_megamoe_supernode` context before touching the new environment.
- Recorded the Pod2 40-card profile in `task_plan.md` and `findings.md`: jump host, worker node map `c0..c9`, Docker container `lj_sgl_0512`, shared-storage note, and mount paths.
- Did not write the worker-node plaintext password into planning files or commands. Per the supernode workflow, compute password auth must be handled interactively or converted to key auth before automation.
- Verified jump host key auth from the local terminal: `simsadmin@10.2.208.215:51730` returned hostname `sims_508_tiaobj.localdomain`, user `simsadmin`, and current time.
- Tested local ProxyJump BatchMode auth to all 10 workers (`p_user@172.20.2.131` through `p_user@172.20.2.140`). Every worker was reachable but rejected non-interactive key auth with `Permission denied (...,password)`.
- Also tested jump-host-side BatchMode auth to `c0`; it failed the same way. Current blocker: compute-node terminal automation needs interactive password login or SSH public-key setup for `p_user`.
- 40-card state check is not yet run because worker login is not established. Once auth is available, run `docker exec lj_sgl_0512 bash -lc 'hy-smi; hy-smi --showpids || true'` on all `c0..c9`.

## 2026-07-02 14:48:55 +08:00 - Pod2 Key Auth, Container Startup, And DCU State
- Installed the local `id_rsa.pub` public key for `p_user` through the Pod2 jump host using an interactive password prompt, without writing the password to files or command logs.
- Verified ProxyJump BatchMode login for `c0..c9`. `c8` initially failed during the transition window, but later checks returned `p2c8`, `p_user`, and the expected `lj_sgl_0512` container status.
- Started the existing `lj_sgl_0512` container on every worker with `docker start lj_sgl_0512`; all `c0..c9` reported the container as `Up`.
- Verified shared mounts inside the running containers on all workers: `/module`, `/public/lijing`, and `/data_add/lizhg/lj` are present.
- Ran container-side DCU visibility checks on all workers. Current result: no `/dev/kfd`, no `/dev/dri/renderD*`, and `hy-smi` does not report usable cards. Most nodes show `Hfm Link Status: WAIT GLOBAL TOPO` plus `No hycu Driver loaded`; `c2` shows `Hfm Link Status: LINK FAILED`.
- Conclusion: Pod2 SSH and container access are now usable, but the 40-card GPU state is not usable/inspectable yet because DCU devices are not exposed or the platform driver/topology is not ready.

## 2026-07-02 14:59:56 +08:00 - Pod2 c0-c9 Card State Recheck
- Rechecked all Pod2 workers `c0..c9` through ProxyJump with `docker exec -i lj_sgl_0512 bash -s`; all containers are `running`.
- Container device visibility is still absent on every worker: `kfd=no` and `render_count=0` for `c0..c9`.
- `hy-smi` result:
  - `c0`, `c1`, `c3`, `c4`, `c5`, `c6`, `c7`, `c8`, `c9`: `Open mkfd failed` repeated, then only the HCU table header appears; `hy-smi --showpids` says `No KFD PIDs currently running`.
  - `c2`: `No hycu Driver loaded` for both `hy-smi` and `hy-smi --showpids`.
- Interpretation: no cards are currently usable from the running containers. This is a driver/device exposure issue, not a user-process occupancy issue.

## 2026-07-02 15:25:12 +08:00 - Pod2 c0-c9 Card State Recheck 2
- Rechecked host and container state for all Pod2 workers. All `lj_sgl_0512` containers are still `running`.
- Host-side device nodes mostly recovered: `c0`, `c1`, and `c3..c9` report `/dev/kfd` plus 4 render nodes; `c2` reports `/dev/kfd` but 0 render nodes.
- Container-side device nodes are still absent everywhere: `container_kfd=no` and `container_render_count=0` for `c0..c9`.
- Container `hy-smi` table rows:
  - `c0`: HCU0-1 only, VRAM `0%/0%`; `showpids` says no KFD PIDs.
  - `c1`: HCU0-3, all VRAM `0%`; `showpids` says no KFD PIDs.
  - `c2`: no HCU rows; `No device available, no device found or initialization failed`.
  - `c3`: HCU0-3, all VRAM `7%`; `showpids` cannot open process directory.
  - `c4`: HCU0-3, VRAM `0%/34%/34%/0%`; `showpids` cannot open process directory.
  - `c5`: HCU0-3, VRAM `35%/35%/35%/34%`; `showpids` cannot open process directory.
  - `c6`: HCU0-3, all VRAM `34%`; `showpids` cannot open process directory.
  - `c7`: HCU0-3, all VRAM `26%`; `showpids` cannot open process directory.
  - `c8`: HCU0-3, all VRAM `34%`; `showpids` cannot open process directory.
  - `c9`: HCU0-3, VRAM `11%/34%/11%/22%`; `showpids` cannot open process directory.
- Interpretation: Pod2 is not ready for MegaMoE GPU testing. The host has devices on most nodes, but the running container still lacks device nodes and `hy-smi --showpids` cannot attribute the nonzero VRAM on `c3..c9`.

## 2026-07-02 15:30:58 +08:00 - Pod2 Docker Restart And Card State
- Restarted the existing `lj_sgl_0512` container on every Pod2 worker `c0..c9` with `docker restart lj_sgl_0512`; each restart command returned the container name and `running=true`.
- After restart, container device nodes are fixed on `c1` and `c3..c9`: `container_kfd=yes` and `container_render_count=4`. `c2` still has `container_kfd=yes` but `container_render_count=0`.
- Post-restart `hy-smi`:
  - `c1`: HCU0-3 all idle, VRAM `0%`, HCU `0.0%`, no KFD PIDs.
  - `c2`: still no usable device, `No device available, no device found or initialization failed`.
  - `c3`: HCU0-3 VRAM `7%`, HCU `0.0%`; `showpids` cannot open process directory.
  - `c4`: HCU0-3 VRAM `34%`, HCU `50.0%`; `showpids` cannot open process directory.
  - `c5`: HCU0-3 VRAM `11%/34%/34%/11%`, HCU `0.0%`; `showpids` cannot open process directory.
  - `c6`: HCU0-3 VRAM `34%`, HCU `50.0%`; `showpids` cannot open process directory.
  - `c7`: HCU0-3 VRAM `26%`, HCU roughly `26.6%..50.0%`; `showpids` cannot open process directory.
  - `c8`: HCU0-3 VRAM `26%`, HCU `50.0%`; `showpids` cannot open process directory.
  - `c9`: HCU0-3 VRAM `35%/35%/35%/34%`, HCU `50.0%`; `showpids` cannot open process directory.
- `c0` restart succeeded and initially showed `container_kfd=yes`, `container_render_count=4`, but the final post-restart `hy-smi` retry could not reconnect: jump host can ping `172.20.2.131`, while SSH to port 22 returns `Connection refused`.
- Interpretation: restart fixed container device exposure on 9/10 nodes except `c2` render devices, but the Pod2 40-card pool is still not free/healthy for testing. Several nodes have nonzero VRAM/HCU utilization with `showpids` unable to attribute the owning process, and `c0` SSH is temporarily unavailable after the restart.

## 2026-07-02 16:51:04 +08:00 - Pod2 c0-c9 Card State Recheck 3
- Rechecked Pod2 workers `c0..c9` through the jump host after the container restart. The current status is still not suitable for 40-card MegaMoE testing.
- Unreachable/auth-failed nodes:
  - `c0`: SSH timed out during banner exchange, so the latest container/card state could not be read.
  - `c1`: SSH key auth now fails with `Permission denied`, so the latest container/card state could not be read.
  - `c2`: ProxyJump reports `No route to host`, so no card state could be read.
  - `c8`: ProxyJump reports `No route to host`, so no card state could be read.
- Reachable running containers:
  - `c3`: `lj_sgl_0512` running, `/dev/kfd=yes`, 4 render nodes, HCU0-3 at `7%` VRAM and `100.0%` HCU.
  - `c4`: running, `/dev/kfd=yes`, 4 render nodes, HCU0-3 at `34%` VRAM and `50.0%` HCU.
  - `c5`: running, `/dev/kfd=yes`, 4 render nodes, HCU0-2 at `35%` VRAM and `50.0%` HCU, HCU3 at `34%` VRAM and `50.0%` HCU.
  - `c6`: running, `/dev/kfd=yes`, 4 render nodes, HCU0-3 at `34%` VRAM and `50.0%` HCU.
  - `c7`: running, `/dev/kfd=yes`, 4 render nodes, HCU0-3 at `34%` VRAM and `50.0%` HCU.
  - `c9`: running, `/dev/kfd=yes`, 4 render nodes, HCU0-3 at `34%` VRAM and `50.0%` HCU.
- `hy-smi --showpids` still fails on reachable active nodes with `Unable to open process directory`, so the owning processes cannot be attributed from inside the container.
- Conclusion: Pod2 is still busy/unhealthy for this run. The usable-looking containers are occupied, and four workers are not reliably reachable through the current SSH path.

## 2026-07-02 16:58:00 +08:00 - Pod2 c0 Login Clarification
- User confirmed manual password login from the jump host to `p_user@172.20.2.131` succeeds.
- Rechecked from the jump host without using any password in commands: `172.20.2.131:22` is open.
- Jump-host-side `ssh -o BatchMode=yes p_user@172.20.2.131 ...` still fails with `Permission denied`, so the automation issue is key-based login, not basic reachability.
- User screenshot shows c0 login warning `Could not chdir to home directory /public/home/p_user: No such file or directory`; this likely explains unstable/missing key auth because the normal `~/.ssh/authorized_keys` location cannot be resolved.
- Corrected interpretation: c0 is manually reachable by password, but not currently usable by my non-interactive SSH checks until key auth/home handling is fixed or the user runs commands interactively from the c0 shell.

## 2026-07-02 17:02:00 +08:00 - Pod2 c1 Login Clarification
- User confirmed manual login to `p_user@172.20.2.132` succeeds.
- Rechecked from the jump host without using any password in commands: `172.20.2.132:22` is open.
- Jump-host-side `ssh -o BatchMode=yes p_user@172.20.2.132 ...` still fails with `Permission denied`.
- Corrected interpretation: c1 is also manually reachable by password, but not currently usable by my non-interactive SSH checks until key auth is fixed or commands are run interactively from the c1 shell.

## 2026-07-02 17:08:00 +08:00 - Pod2 Key Auth Root-Cause Check
- Compared key-auth-good nodes `c3` and `c4`: both have `p_user` home `/public/home/p_user`, current directory is that home, `.ssh` mode is `700`, `authorized_keys` mode is `600`, and the local public key is present in `authorized_keys`.
- This contrasts with the user's c0 screenshot, where login succeeds by password but reports `Could not chdir to home directory /public/home/p_user: No such file or directory` and lands in `/`.
- Current root-cause assessment: c0/c1 key auth is failing because the worker cannot resolve or access the configured home path used by sshd's default `AuthorizedKeysFile` lookup, not because the local key is missing.
- Fix direction: restore/create `/public/home/p_user` with the existing `.ssh/authorized_keys` content and correct ownership/modes, or repair the `/public/home` mount on c0/c1 if it is supposed to be shared storage. Without password-interactive access from my terminal, this needs either user-side manual commands on c0/c1 or an admin-side mount/home repair.

## 2026-07-02 17:15:00 +08:00 - Pod2 c1 Key Auth Recovered
- Rechecked c1 after the user noted that `/public/home/p_user` exists there.
- SSH debug now shows the target accepts the local RSA key, and `BatchMode` login through ProxyJump succeeds: `host=p2c1`, `user=p_user`, `HOME=/public/home/p_user`, `pwd=/public/home/p_user`.
- c1 path and permissions are valid: `/public/home/p_user` owner `p_user:p_user`, `.ssh` mode `700`, `authorized_keys` mode `600`, and the local public key is present.
- Corrected assessment: c1 is no longer a key-auth blocker. The earlier `Permission denied` was transient or observed while the shared home/auth state was not ready.
- Remaining c1 issue: `lj_sgl_0512` is currently `exited running=false`, so card checks on c1 require starting/restarting the container first.

## 2026-07-02 17:04:45 +08:00 - 151.1 Card State Check
- Checked `10.17.151.1` via `root` before any GPU work.
- `sglang_megamoe` is running.
- Host-side device nodes are present: `/dev/kfd` plus render nodes `renderD128..renderD143`.
- Container-side device nodes are also present: `/dev/kfd` plus all 16 render nodes.
- Host and container `hy-smi` both report all 16 HCUs at `0%` VRAM and `0.0%` HCU.
- `hy-smi --showpids` reports `No KFD PIDs currently running`.
- Host `ps` still shows stale defunct `sglang::schedul` zombie processes, but they have no KFD ownership and do not occupy DCU memory/utilization.
- Conclusion: `151.1` is currently idle and usable from the card-state perspective.

## 2026-07-02 17:45:00 +08:00 - 151.1 Flash/Pro Batch Started
- Added the requested `5120` normal bucket to the active 151.1 validation plan.
- Synced the current local code changes under `README.md` and `megamoe/` to `/root/yuguo/DeepGEMM` on `10.17.151.1`.
- Remote source checks passed after using the known amdsmi `LD_LIBRARY_PATH` override: `python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `11 passed`.
- Rebuilt MegaMoE in docker `sglang_megamoe`, copied fresh build output from `build/lib.../megamoe` into the source-tree import path, and confirmed fresh `_C`, K1/K2/K3 extension `.so`, and K1/K3 `.co` artifacts exist.
- Started the requested Flash RPC validation batch with normal buckets `512,1024,1025,2048,2050,4096,4097,5120,8192`, LL graph buckets `8,32,128,256,512`, and sampled uneven normal eager/graph buckets.
- Flash EP8 completed successfully through all scheduled uniform/uneven cases. During the first Flash EP16 normal eager `512` case, the SSH session lost liveness and subsequent simple SSH checks to `10.17.151.1` were closed by the server. Resume by reading `hygon_tmp/supernode_debug/flash_pro_batch_20260702/flash/summary.tsv` once SSH recovers; do not rerun already completed EP8 cases.

## 2026-07-02 18:55:00 +08:00 - 151.1 Flash EP16 Normal Eager Completed
- Resumed Flash EP16 with host-side background case runners to avoid SSH foreground disconnects killing the docker test.
- Flash EP16 normal eager passed correctness/performance for `512,1024,1025,2048,2050,4096,4097,5120`; all compared against `normal-contiguous` baseline and were faster than baseline in the collected run.
- Flash EP16 `8192` printed correctness pass (`max_abs=0.000488281`) but the full correctness+bench process hung before writing JSON, matching the known large-bucket teardown sensitivity. The hung child processes were cleaned up by exact PID because they belonged to this run and held `MASTER_PORT=8361`.
- Flash EP16 `8192` MegaMoE-only timing was collected separately with `correctness_iters=0`, `--skip-baseline-bench`, and `MASTER_PORT=18361`: fused median `12.4028 ms`; correctness evidence remains in the earlier `flash_ep16_normal_eager_8192/run.log`.
- Next: run Flash EP16 normal graph cap8192, LL graph cap512, and sampled uneven eager/graph cases with unique `MASTER_PORT` values; then compare Flash against historical V3/supernode data before starting Pro EP8/EP16.

## 2026-07-02 19:18:00 +08:00 - 151.1 Flash EP16 Graph/Uneven Summary
- Flash EP16 sampled uneven cases passed:
  - normal eager cap512 `1.2698 ms` vs baseline `2.1205 ms`; normal graph cap512 `128/256/512 -> 0.8963/0.9962/1.2209 ms`.
  - normal eager cap4096 `6.5289 ms` vs baseline `10.2621 ms`; normal graph cap4096 `1024/2048/4096 -> 1.9675/3.2432/6.0569 ms`.
- Flash EP16 LL graph passed after matching the earlier EP8 oracle (`baseline-kind normal-contiguous`): `8/32/128/256/512 -> 0.3972/0.4464/0.7503/1.2988/2.8057 ms`.
- Flash EP16 normal graph with tail-reduce1 passed and produced JSON through cap5120: `512/1024/1025/2048/2050/4096/4097/5120 -> 1.3884/1.9878/2.0118/3.2672/3.4190/6.2516/6.1880/7.8339 ms`.
- Flash EP16 normal graph `8192` correctness passed in the failed full-cap logs, but graph bench at 8192 repeatedly VMFaulted / timed out on the staged rank barrier. A `--cuda-graph-skip-baseline` 8192 replay-only run exited OK but does not produce timing rows by design.
- Flash performance guardrail readout so far: EP8 current results remain in the historical 151.1 band (`4096` current `5.765 ms` vs prior `5.764/5.803 ms`; `8192` current `10.714 ms` vs prior `10.770 ms`). EP16 eager is also close or slightly better than prior 151.1 data (`4096` current `6.688 ms` vs prior `6.574/6.684 ms`; `5120` current `8.290 ms` vs prior `8.174/8.306 ms`; `8192` current fused-only `12.403 ms` vs prior `12.604/12.819 ms`). No broad Flash regression is visible; only EP16 normal graph 8192 bench remains a separate stability issue.
- Next: start Pro EP8/EP16 using the same normal bucket list and sampled graph/uneven rules; keep Flash failed retry rows in `summary.tsv` but use the later OK retry rows for reporting.

## 2026-07-02 19:50:00 +08:00 - 151.1 Pro EP16 Normal Eager Collected
- Pro EP16 normal eager full fused/baseline timing passed for `512,1024,2048,2050,4096,4097,5120`; all were correct and faster than `normal-contiguous`.
- Pro EP16 `1025` initially coincided with a node/docker `exit 255` and empty log. A correctness-only retry passed, then timing-only with baseline bench passed. Treat the first interruption as node instability, not a 1025 correctness failure.
- Pro EP16 normal eager timings:
  - `512`: MegaMoE `3.1089 ms`, baseline `4.2739 ms`.
  - `1024`: MegaMoE `4.6881 ms`, baseline `7.2032 ms`.
  - `1025`: correctness passed separately; timing MegaMoE `4.7419 ms`, baseline `7.0390 ms`.
  - `2048`: MegaMoE `7.4311 ms`, baseline `13.1023 ms`.
  - `2050`: MegaMoE `7.4441 ms`, baseline `13.1279 ms`.
  - `4096`: MegaMoE `14.7188 ms`, baseline `25.2819 ms`.
  - `4097`: MegaMoE `14.6908 ms`, baseline `25.2925 ms`.
  - `5120`: MegaMoE `18.2497 ms`, baseline `31.3879 ms`.
  - `8192`: correctness passed and wrote JSON, but cleanup hung; exact-PID cleanup made the summary row show `fail`. Separate fused-only timing passed with MegaMoE `29.0623 ms`, baseline skipped.
- Next: collect Pro EP16 graph/LL/uneven samples, then Pro EP8 normal/graph/LL/uneven.

## 2026-07-02 20:08:00 +08:00 - Pro EP16 Normal Graph Tail-Reduce Root Cause
- Pro EP16 normal graph tail-reduce1 was minimized to cap5376/replay512 without graph bench; eager correctness still passed, but graph replay failed on non-rank0 processes with large max_abs.
- Same command with `K3_USE_ASM_TAIL_REDUCE=0` passed, isolating the failure to the K3 normal ASM tail-reduce graph path rather than K1/K2/plain K3 graph.
- Root cause: `K3_TAIL_APPLY_GRAPH_RUNTIME_STATE` computed graph runtime reduce vectors with `runtime_tokens << 9`, which is the Flash-only `4096 / 8 = 512` BF16-vector count per token. Pro hidden=7168 requires `7168 / 8 = 896`, so later token rows were only partially reduced.
- Local fix prepared: both normal K3 tail-reduce ASM sources now compute `runtime_tokens * (sgprSizeI >> 3)`, and source assertions reject the old `s_lshl_b32 s76, s76, 9` pattern. Remote rebuild and Pro graph retest are next.

## 2026-07-02 20:35:00 +08:00 - Pro Graph Fix Build Ready, 151.1 HCU Isolation Blocker
- 151.1 direct SSH on port 22 is usable; the earlier port `21520` path currently times out or closes during banner exchange, so the active route is `root@10.17.151.1:22`.
- The K3 tail-reduce graph fix was synced to 151.1 and remote source pytest passed (`11 passed`) before rebuild.
- The remote rebuild completed. Fresh artifacts from `build/lib.linux-x86_64-cpython-310/megamoe` were copied back into the source-tree import path after avoiding the older stale `build/lib` copy. The runtime tree now has fresh K3 tail-reduce `.co` files from `20:21` and `k3_fused_ext` from `20:19`.
- GPU verification is blocked by node state, not by source/build state. `dmesg` shows all 16 HCUs isolated at `2026-07-02 20:19:26` with `XID:51 ... HCU fault isolation success`, and host/container `hy-smi` now reports no available device.
- No current process holds `/dev/kfd` or `/dev/dri/renderD*`; the prior `wanghl_dev2` Pro SGLang server/bench process has exited. Do not run the Pro graph GPU validation until 151.1 is recovered, for example by an approved HCU reset or node-side recovery.

## 2026-07-02 21:22:00 +08:00 - Flash EP16 LL Graph Slowdown Triage
- Corrected assessment after re-reading the V3 LL records and the user's note: a cap mismatch invalidates a direct number-to-number comparison, but it is not a sufficient root-cause explanation. The old LL graph path was designed to make replay mostly runtime-token driven, and V3 records show capture512/capture8192 small-token replay stayed in the same performance band after active-row copy and K2/K3 runtime pruning.
- Historical 151.1 LL graph sample `hygon_tmp/supernode_debug/151_1_rpc/ep16_ll_graph_512_rpc_20260701_120729` requested `--num-max-tokens-per-rank 512`. Because `kTokenAlignment=384`, the runtime buffer printed `tokens=512/768`, but the CUDA graph capture capacity stayed at the requested `512`, so logs show `CUDA graph bucket token=.../512`.
- The current reported slow retry `flash_ep16_ll_graph_cap512_retry_normalbaseline` requested `--num-max-tokens-per-rank 768`, so the graph capture capacity became `768`, and logs show `CUDA graph bucket token=.../768`. This run is still not a fair historical cap512 comparison, but the slowdown must be treated as a possible current git-diff regression until proven otherwise.
- Static source diff review so far:
  - Flash `k3_out` aliases the old `l1_out` workspace because Flash has `hidden == 2 * intermediate`, so the separate Pro workspace is not a Flash workspace-pointer change.
  - Current K3 split-tail still reads `runtime_num_tokens` for reduce, and reads `m_indices[local_experts]` as the optional max-copy row slot.
  - The current repository's retained split-tail code only shrinks copy CTAs for tiny `max_copy_rows <= 16`; earlier V3 memory has a faster active-row-copy version, and later V3 memory records why several active-row-grid attempts were reverted. Therefore cap sensitivity can be exposed by this retained copy-grid design, but the user's point remains: the just-tested previous commit is the real baseline to compare.
- Next fair checks after 151.1 HCU recovery:
  1. First re-create the README `ll-masked` baseline environment for the actual 151.1 node: pick/verify active HCA names, generate or reuse a ROCSHMEM topology file from `/sys/class/infiniband/<dev>/device`, set `ROCSHMEM_ALLOWED_IBV_DEVICES`, `ROCSHMEM_TOPO_FILE_FORCE`, `DEEPEP_ENABLE_LL_DISPATCH_OPT=1`, `ROCSHMEM_DISABLE_HDP_FLUSH=1`, `ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288`, `ROCSHMEM_MAX_NUM_CONTEXTS=48`, and sufficiently large `ROCSHMEM_HEAP_SIZE` / `DUSHMEM_HEAP_SIZE`.
  2. Run the exact same Flash EP16 LL graph command on the current worktree and on the previous commit (`9607561`) with `--num-max-tokens-per-rank 512 --num-tokens 512 --baseline-kind ll-masked --correctness-iters 1 --skip-bench --cuda-graph --cuda-graph-test-tokens 8,32,128,256,512 --cuda-graph-bench`.
  3. If current is slower, bisect the current git diff by shared LL-risk groups first: K3 LL shape templating, tail done-counter expansion, route scratch sizing/`k3_out`, and K1 LL shape templating. Do not start from unrelated normal ASM Pro fixes.
  4. Add a narrow diagnostic dump around K1->K2->K3 for LL graph: `rows`, `m_per_expert`, `m_indices[0:local_experts]`, `m_indices[local_experts]`, copied runtime token, and K3 split-tail launch grid. This directly checks whether runtime replay work is being over-expanded.

## 2026-07-03 15:20:00 +08:00 - Flash EP8 Normal Graph Tail-Reduce Regression Patch
- Rechecked the recent Pro EP16 K3 tail-reduce graph fix after Flash EP8 normal graph cap8192 replay8192 failed correctness on 151.1. replay4096 passed with median `5607.7 us`; replay8192 failed with `max_abs=5.5331878662109375`.
- Root cause found in source: `K3_TAIL_APPLY_GRAPH_RUNTIME_STATE` reads `sgprSizeI`, but `sgprSizeI` is `s20`, and the extra-reducer WG mapping reuses `s20` before the macro runs.
- Patched both K3 tail-reduce ASM variants to read stable `hidden / 8` from `GpuProb+0xd8`; added `asm_reduce_hidden_vecs` to `k3_fused_ext.cu` and source guards to prevent graph-runtime width from using `sgprSizeI` again.
- Local verification: `git diff --check` passed; local `python -m pytest` is unavailable because the Windows Python environment lacks `pytest`. Next verification is remote source pytest/build on 151.1, then Flash EP8 normal graph replay8192 retest.

## 2026-07-03 16:25:00 +08:00 - Flash EP8 Normal Graph Hidden-Vec Fix Verified
- Synced the K3 hidden-vec patch to 151.1, restarted `sglang_megamoe` as requested, rebuilt MegaMoE, copied fresh `build/lib.linux-x86_64-cpython-310/megamoe` artifacts back into the source-tree import path, and ran source pytest in the container.
- Remote source pytest passed: `11 passed in 7.41s`.
- Flash EP8 normal graph cap8192 retest passed for replay4096 and replay8192 against `normal-contiguous`: correctness `max_abs=0.000488281`, graph replay medians `5636.4 us` and `11217.7 us`.
- Conclusion: the recent Pro tail-reduce hidden fix no longer breaks Flash EP8 normal graph replay8192; continue with the requested LL masked-baseline and Pro EP8 queue.

## 2026-07-03 16:35:00 +08:00 - LL Masked Baseline Environment Decision
- Re-read README `ll-masked` baseline notes and local historical LL records. The README HCA/topo block is reference material, not a hard requirement for 151.1.
- On 151.1, host and container `/sys/class/infiniband` are empty, so the next LL masked-baseline test should not force stale `mlx5_*` names or topology files from another node.
- Current execution is blocked by an active SGLang/DeepEP service occupying all 16 HCUs; do not kill it. When the cards are free, run Flash EP16 LL graph cap512 with the minimal 151.1 environment first, then add historical MNNVL/heap compatibility variables only if DeepEP LL init fails.

## 2026-07-03 16:36:00 +08:00 - Flash EP8 LL Graph cap512 ll-masked Result
- After the SGLang service released the cards, 151.1 still reported HCU14/HCU15 isolation, so the attempted Flash EP16 16-rank LL run was stopped and only its own spawned worker PIDs were cleaned up.
- Ran Flash EP8 LL graph cap512 replay512 on devices `0..7` with `--baseline-kind ll-masked` and no forced HCA/topo because `/sys/class/infiniband` is empty on 151.1.
- Result: correctness passed (`max_abs=0.000549316`, `mean_abs=2.29036e-05`). Graph replay median is `1.8144 ms`; `ll-masked` baseline graph median is `2.1164 ms`.
- Note: the wrapper script printed `RC=0` and wrote a valid result JSON; the shell returned nonzero only because a CRLF-contaminated `exit 0\r` line was parsed as a numeric-argument error. No KFD PIDs remained afterward.

## 2026-07-03 16:45:00 +08:00 - Active Plan Reordered Around Flash EP16 Graph Regression
- Updated `task_plan.md` so Flash EP16 graph performance regression is ahead of Pro EP8 broad data collection once 16 cards are healthy.
- Current EP16 blocker: 151.1 reports HCU14/HCU15 isolation, so Flash EP16 and Pro EP16 graph results cannot be judged until all 16 HCUs recover.
- Pro EP16 normal eager `512..8192` remains collected from the earlier run, but Pro EP16 normal graph hidden-vec fix is not runtime-verified yet. The fix is code/build-ready and Flash EP8 normal graph regression is fixed.
- Pro EP8 has no valid timing/correctness data yet. The first host-side batch failed because it ran outside docker and could not import `torch`; the docker run entered Pro EP8 normal 512 but hung before correctness output, and the later tail-reduce0 probe reached config before SSH was closed by the remote host. Treat this as an unresolved Pro EP8 512 bring-up issue, not a completed data point.
- Next operational order: recover/confirm 151.1 state, then fair Flash EP16 LL graph cap512 retest/A-B if needed, Flash EP16 normal graph `4096/5120/8192`, Pro EP16 graph verification, then Pro EP8 512 isolation or broad EP8 data only after the 512 issue is understood.

## 2026-07-03 17:20:00 +08:00 - 151.1 Execution State Recovered
- 151.1 host and `sglang_megamoe` container are usable again for EP16 runs.
- Host-side and container-side checks show all 16 HCUs visible, normal, idle, and no KFD PIDs.
- The existing `sglang_megamoe` container had exited with code 255 and was restarted. Repo path `/root/yuguo/DeepGEMM` and DTK env `/root/yuguo/dtk-26.04.1/env.sh` are present inside the container.
- Next action is the fair Flash EP16 LL graph cap512 retest with `--baseline-kind ll-masked`, replay `8,32,128,256,512`, and no forced stale HCA/topology variables because 151.1 exposes no InfiniBand devices.

## 2026-07-03 17:23:00 +08:00 - Flash EP16 LL Graph Fair Retest
- Ran Flash EP16 LL graph cap512 on 151.1 with `--baseline-kind ll-masked`, `--num-max-tokens-per-rank 512`, `--num-tokens 512`, replay `8,32,128,256,512`, and the node-actual ROCSHMEM environment.
- Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/flash_ep16_ll_masked_graph512_current_20260703_171943/result.json`.
- Current fused graph medians: `8/32/128/256/512 -> 0.3763/0.4113/0.5980/0.9915/1.9043 ms`.
- Historical cap512 medians from `ep16_ll_graph_512_rpc_20260701_120729`: `8/32/64/128/256/512 -> 0.3749/0.4069/0.4769/0.5977/0.9869/1.9228 ms`.
- Current-vs-history deltas for common buckets are `+0.38%/+1.08%/+0.05%/+0.47%/-0.96%`. The earlier Flash EP16 LL graph slowdown is not reproduced under the fair cap512/ll-masked command, so no current-vs-previous commit A/B is needed for this issue.
- Current ll-masked baseline graph medians: `8/32/128/256/512 -> 0.9274/0.9416/1.0221/1.3903/2.0442 ms`; fused remains faster than baseline at all replay buckets.

## 2026-07-03 17:36:00 +08:00 - Pro EP8 512 Initial Hang Isolation
- Started a Pro EP8 normal 512 correctness-only isolation matrix under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_512_isolation_20260703_172349`.
- `ipc_tail1` and `ipc_tail0` both reached Config with `peer_mode=ipc`, `local_experts=48`, `tokens=512/768`, then produced no correctness output before the 240s watchdog killed them (`rc=124`, no JSON).
- Because IPC already hangs with both tail-reduce settings, the current blocker is not specific to RPC/Fabric and not specific to K3 ASM tail-reduce1.
- Stopped the remaining RPC cases from the same matrix to avoid spending more card time on variables already ruled out by IPC. KFD PIDs were clear afterward.
- Next diagnostic is fused-only versus baseline separation for Pro EP8 normal 512.

## 2026-07-03 17:38:00 +08:00 - Pro EP8 Fused-Only Split
- Ran Pro EP8 normal 512 fused-only (`correctness_iters=0`, `skip-baseline-bench`, IPC, `K3_USE_ASM_TAIL_REDUCE=0`) under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_512_fused_only_20260703_173356`.
- The run again reached Config and timed out after 240s before any performance JSON (`rc=124`), so the Pro EP8 512 blocker is in MegaMoE normal fused execution, not in the normal-contiguous baseline or correctness comparison.
- User requested switching priority to Pro EP16 first. Next action: verify Pro EP16 can run, then rerun the previously failing/fixed Pro EP16 normal graph K3 tail-reduce replay512 scenario.

## 2026-07-03 17:42:00 +08:00 - Pro EP16 Graph Tail-Reduce Fix Verified
- Ran the previously failing/fixed Pro EP16 normal graph K3 tail-reduce scenario under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_graph_tailreduce_replay512_20260703_174033`.
- Command shape: EP16, Pro shape `hidden=7168`, `intermediate=3072`, `experts=384`, `topk=6`, `MEGAMOE_DCU_PEER_MEMORY=rpc`, `K3_USE_ASM_TAIL_REDUCE=1`, requested graph cap `5120` and replay `512`.
- Result: eager correctness passed with `max_abs=0.000488281`; graph replay512 passed with `max_abs=0.000976562`; process exited `rc=0` and left no KFD PIDs.
- Conclusion: the K3 tail-reduce graph hidden-vector fix is runtime-verified for the minimized Pro EP16 failure. Next action is Pro EP16 normal graph cap5120 full bucket timing.

## 2026-07-03 17:44:00 +08:00 - Pro EP16 Normal Graph cap5120 Buckets
- Ran Pro EP16 normal graph cap5120 full bucket timing under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_graph_cap5120_buckets_20260703_174224`.
- All replay buckets passed correctness with graph `max_abs=0.000976562`.
- Graph replay medians: `512/1024/1025/2048/2050/4096/4097/5120 -> 3.1346/4.5135/5.2051/8.2831/8.2907/14.5611/14.6161/17.6194 ms`.
- The run exited `rc=0` and left no KFD PIDs. Next action is a cap8192 Pro EP16 graph stability/performance point.

## 2026-07-03 17:51:00 +08:00 - Pro EP16 cap8192 Graph Boundary
- Pro EP16 normal graph cap8192 with replay `4096,8192` produced a mixed result under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_graph_cap8192_4096_8192_20260703_174450`.
- replay4096 correctness passed and graph bench reported `15.6721 ms`.
- replay8192 correctness passed (`max_abs=0.00146484`) before graph bench, but the 8192 graph benchmark replay triggered a K3COMBINE VMFault and rank-barrier timeout (`rc=1`).
- A follow-up cap8192 no-bench correctness run under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_graph_cap8192_correctness_20260703_174747` wrote a valid correctness JSON with replay8192 `max_abs=0.00146484`, then hung in teardown/cleanup. Exact PIDs from that run were killed after confirming they belonged to the run; KFD PIDs cleared afterward.
- Conclusion: Pro EP16 graph hidden-vector correctness is fixed through cap8192. Remaining cap8192 issue is graph-bench/cleanup stability, similar to the existing large-cap graph instability class, not the original hidden-size correctness bug.

## 2026-07-03 17:59:00 +08:00 - Pro EP8 K1 Boundary
- Pro EP8 normal 512 fused-only already ruled out the normal-contiguous baseline and correctness comparison: the run reached Config and timed out before JSON.
- Added temporary `MEGAMOE_DCU_STAGE_STOP` returns to isolate the first non-returning stage without mixing K2/K3/reduce work into the diagnosis.
- Stage-stop result under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_stage_stop_20260703_175353`: `start_barrier` returned cleanly, but `k1` timed out with no JSON.
- Current boundary: Pro EP8 512 hangs in normal K1 for `local_experts=48`. K2/K3/reduce have not been reached in this failure path, so the next source pass should focus on K1 normal ASM/launcher assumptions that still encode Flash/EP8 `local_experts<=32` or old route-scratch offsets.

## 2026-07-03 18:25:00 +08:00 - Pro EP8 K1 Compact-Prebuild Fix Verified
- Inspected K1 normal ASM around the route builder rather than doing a broad scan. The old in-ASM route path still encoded Flash EP8 `local_experts=32` assumptions: route base `rank * 32`, reset/publish loop bounds of 32, and meta flag coverage for only 32 local experts.
- Implemented the narrow launcher-side fix: force compact prebuild when `local_experts > 32`, in addition to the existing `num_ranks > 8` safe path. This routes Pro EP8 (`384 / 8 = 48` local experts) away from the old 32-entry in-ASM route builder while leaving Flash EP8/EP16/EP32 on their prior defaults.
- Synced the changed files to 151.1, rebuilt MegaMoE in `sglang_megamoe`, and confirmed source pytest after using base64 remote script execution to avoid PowerShell CRLF stdin pollution: `11 passed in 7.29s`.
- Pro EP8 512 K1 stage-stop now returns under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_k1_compact_fix_stage_stop_20260703_182035` with `rc=0`.
- Pro EP8 512 full normal fused-only also returns under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_512_full_fused_only_20260703_182153`, fused median `4.8504 ms`.
- Pro EP8 512 normal correctness against `normal-contiguous` passes under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_512_correctness_20260703_182306`: `max_abs=0.000976562`, fused median `4.7512 ms`, baseline bench skipped.
- Next action: remove the temporary `MEGAMOE_DCU_STAGE_STOP` code, rebuild/copy fresh artifacts, rerun Pro EP8 512 clean sanity, then proceed to the requested Pro EP8 bucket data.

## 2026-07-03 18:35:00 +08:00 - Pro EP8 Clean Source And RPC 512 Result
- Removed the temporary `MEGAMOE_DCU_STAGE_STOP` helper/early returns from `megamoe/opt.py`. Local checks passed: no stage-stop strings in `megamoe`, `python -m py_compile megamoe/opt.py`, and `git diff --check`.
- Synced the clean `opt.py` plus current K1/test source to 151.1. Remote source check confirms no stage-stop strings in source files; the only earlier hit was stale `__pycache__`.
- Remote source pytest passed again: `11 passed in 7.28s`.
- Clean Pro EP8 512 IPC sanity passed under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_512_clean_correctness_20260703_182653`: `max_abs=0.000976562`, fused median `4.7440 ms`, baseline bench skipped.
- Pro EP8 512 RPC normal eager with baseline bench passed under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_normal_eager_512_rpc_20260703_182832`: `max_abs=0.000976562`, fused `4.7955 ms`, normal-contiguous baseline `4.7731 ms`, speedup `0.995x`.
- Started the broad Pro EP8 RPC normal eager bucket loop, but the first bucket (`1024`) lost SSH before result JSON: connection closed by remote host shortly after config output. Follow-up SSH attempts to 151.1 also close or time out during banner exchange, so this is currently a host/SSH/node-state interruption rather than a recorded MegaMoE correctness failure.
- Do not launch more GPU work until 151.1 SSH recovers. First recovery steps: check host/container status, inspect exact residual `test_mega_moe_dcu.py` PIDs, run `hy-smi --showpids`, and only then resume from Pro EP8 normal eager 1024.

## 2026-07-03 18:55:00 +08:00 - Temporary Debug Cleanup Audit
- Scanned production source for retired temporary controls: `MEGAMOE_DCU_STAGE_STOP`, `_stage_stop_requested`, `MEGAMOE_DCU_DEBUG_*`, `MEGAMOE_DCU_PRO_WEIGHT_LAYOUT`, debug combine/backend CLI flags, K1/K2/K3 coherency probe envs, and temporary acquire/fence helper names.
- Production source no longer contains those temporary execution paths. The remaining retired names are intentionally present only in `test_dcu_megamoe_v3.py` negative assertions so they cannot re-enter production or the test harness.
- Cleaned wording-only leftovers: renamed graph correctness local variables from `*_debug` to value names and changed the K3 tail-reduce ASM comment from `Debug/safety variant` to a production tail-reduce description.
- Local checks passed after cleanup: `python -m py_compile megamoe/opt.py megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` and `git diff --check`.

## 2026-07-03 18:59:00 +08:00 - Plan Refresh After Status Questions
- Confirmed README already includes the Pro model shape and a Pro EP16 validation example. No additional README matrix is needed right now.
- Confirmed Flash EP16 LL graph with `ll-masked` baseline did run successfully under the fair cap512 command, using 151.1's actual environment rather than forcing README HCA/topology variables because `/sys/class/infiniband` is empty on that node.
- Clarified Pro EP16 8192 status: normal eager fused correctness/execution is stable; the unstable item is graph cap8192 replay8192 bench/cleanup, not eager.
- Checked 151.1 after the SSH interruption: host SSH is available again and all 16 HCUs are normal/idle with no KFD PIDs, but the `sglang_megamoe` container exited with code 255. Next GPU test requires restarting the existing container and rechecking `hy-smi --showpids`.
- Updated `task_plan.md` to mark completed Flash EP16 LL graph fair retest, mark the A/B/bisect as not pursued, record README Pro coverage, record Pro EP8 512 fix/clean sanity, and keep Pro EP8 broad buckets plus 8192 graph instability retest as pending.

## 2026-07-03 19:31:00 +08:00 - No-Copy Build Flow Fixed And Smoke-Tested
- Added hard verification to `megamoe/dcu_megamoe_opt/scripts/build_dcu_megamoe.sh`: after inplace build it verifies fresh source-tree runtime artifacts for `megamoe._C`, K1/K2/K3 extension `.so` files, and all six K1/K3 ASM `.co` files, then imports the runtime modules and rejects paths outside the repo or under `build/`.
- Restarted the existing `sglang_megamoe` container on 151.1 after confirming the host HCUs were idle and no KFD PIDs were present.
- Rebuilt inside `/root/yuguo/DeepGEMM` under DTK 26.04.1. The build completed and printed verified fresh artifacts plus import paths resolving to `/root/yuguo/DeepGEMM/megamoe/...`, so source-tree tests no longer need a manual copy from `build/lib...`.
- Post-build remote source pytest passed: `11 passed in 7.35s`.
- Pro EP8 512 normal eager RPC smoke passed after the no-copy rebuild: correctness `max_abs=0.000976562`, fused median `4.8080 ms`, baseline bench skipped. Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/no_copy_build_20260703_191544/pro_ep8_512_normal_eager_smoke.json`.
- Next action per user request: commit the current git changes, then continue Pro EP8 broad data collection from `1024`.

## 2026-07-03 19:41:00 +08:00 - Pro EP8 Normal Eager RPC Matrix
- Committed the current Pro support/no-copy-build changes before broad Pro EP8 data collection: `bf7b3fa Support DeepSeek V4 Pro MegaMoE shapes`.
- Ran Pro EP8 normal eager RPC matrix on 151.1 under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_normal_eager_rpc_current_20260703_193251`.
- Shape and mode: `num_ranks=8`, Pro shape `experts=384`, `topk=6`, `hidden=7168`, `intermediate=3072`, `local_experts=48`, `MEGAMOE_DCU_PEER_MEMORY=rpc`, `K3_USE_ASM_TAIL_REDUCE=0`, baseline `normal-contiguous`, warmup `5`, repeat `10`.
- All buckets passed correctness with `max_abs=0.000976562`.
- Results `tokens -> fused ms / baseline ms / speedup`: `512 -> 4.7537 / 4.8026 / 1.010x`; `1024 -> 5.2085 / 7.5860 / 1.456x`; `1025 -> 5.2166 / 7.6248 / 1.462x`; `2048 -> 8.0569 / 13.1070 / 1.627x`; `2050 -> 8.0917 / 13.1054 / 1.620x`; `4096 -> 15.4011 / 24.6854 / 1.603x`; `4097 -> 15.3239 / 24.7344 / 1.614x`; `5120 -> 17.2820 / 30.5266 / 1.766x`; `8192 -> 27.6906 / 48.5922 / 1.755x`.
- `hy-smi --showpids` was clear after each bucket.
- Next action: reproduce and isolate the large graph cap8192 instability class, starting with Flash EP16 normal graph retest and then Pro EP16 graph cap8192 replay8192.

## 2026-07-03 19:52:00 +08:00 - Graph8192 Instability Reproduced And Patch Hypothesis
- Reproduced Flash EP16 normal graph cap8192 instability with explicit `K3_USE_ASM_TAIL_REDUCE=1`, replay `4096,5120,8192`, graph bench enabled, and eager bench skipped.
- Eager correctness passed first (`max_abs=0.000488281`), then the run VMFaulted before printing the first graph bucket result. Log directory: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/graph8192_instability_flash_ep16_20260703_194417`.
- VMFault diagnostics repeatedly showed `rank_barrier_kernel` and `MegaMoE HIP staged rank barrier timeout` at slot `32`, e.g. `generation=2`, `arrival=28`, `release=1`. For EP16 dynamic signal layout, slot `32` is the normal start barrier.
- Interpretation: graph replay queues multiple replays back-to-back and only synchronizes after the replay loop. Normal tail-reduce graph had a start barrier but no end-of-replay rank barrier. A faster rank can enter the next replay start barrier and reset tail-reduce counters/signals while slower ranks are still finishing the previous replay's K3 tail-reduce, causing the generation-2 start barrier timeout and VMFault.
- Started a `--cuda-graph-skip-baseline` replay-only check after the first VMFault, but 151.1 disconnected at the host SSH level immediately after printing graph execution. Subsequent ping/SSH checks fail or close, so the node is currently unavailable.
- Local narrow patch prepared in `megamoe/opt.py`: for `normal` graph tail-reduce with `num_ranks > 8`, add a post-K3 rank barrier after K3 tail-reduce. This should serialize replay epochs before the next start barrier resets shared tail-reduce signals. The patch deliberately does not affect LL graph or EP8 normal graph.
- Local checks after the patch passed: `python -m py_compile megamoe/opt.py` and `git diff --check`.
- Pending validation when 151.1 recovers: sync `megamoe/opt.py`, rebuild with no-copy script, run source pytest, then rerun Flash EP16 graph cap8192 skip-baseline replay-only and full correctness+bench; finally rerun Pro EP16 graph cap8192 replay8192.
- Added a source-level guard in `test_dcu_megamoe_v3.py` so the normal-tail-reduce graph post-K3 barrier condition is checked by static tests.
- Rechecked 151.1 multiple times after the VMFault: ping gets 100% loss and SSH times out during banner exchange. No further remote validation has been run after the local patch.

## 2026-07-03 20:47:00 +08:00 - Pro EP8 LL 512 Attempt
- 151.1 recovered enough for SSH. The existing `sglang_megamoe` container was still exited with code 255 and was restarted.
- Pre-run card check showed all 16 HCUs normal/idle and `hy-smi --showpids` reported no KFD PIDs.
- Started Pro EP8 LL 512 on devices `0..7` with `MEGAMOE_DCU_PEER_MEMORY=rpc`, shape `experts=384`, `topk=6`, `hidden=7168`, `intermediate=3072`, backend `ll`, baseline `ll-masked`, warmup `5`, repeat `10`.
- The process printed `fused execution=v3_ll_eager` and `cuda graph execution=disabled`, then the remote host closed the SSH connection. No result JSON was produced under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_512_20260703_204440`.
- Immediate follow-up ping/SSH checks failed (`100%` ping loss and SSH banner timeout), so treat this as node/container-level interruption triggered during Pro EP8 LL bring-up, not as a recorded correctness mismatch.
- Next isolation after 151.1 recovers: do not repeat the full timing command first. Run minimal Pro EP8 LL 512 fused-only (`correctness_iters=0`, `skip-bench`, `skip-baseline-bench`, small warmup/repeat or no timing) to determine whether the failure is in MegaMoE LL fused execution before baseline/bench.

## 2026-07-03 21:35:00 +08:00 - Pro EP8 LL 512 Result
- Rechecked 151.1 before rerun: `sglang_megamoe` was up and `hy-smi --showpids` reported no KFD PIDs.
- Minimal fused-only isolation with `--baseline-kind normal-contiguous --correctness-iters 0 --skip-baseline-bench` passed. Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_512_isolation_20260703_212455/pro_ep8_ll_512_fused_only_normalbaseline.json`. Pro EP8 LL fused eager median was `16.7910 ms`; baseline timing was intentionally skipped.
- The first `ll-masked` correctness retry failed before comparison during DeepEP LL buffer init: `ROCSHMEM_HEAP_SIZE` was below the reported Pro requirement `11450456192` bytes.
- Retried with node-actual 151.1 environment, no forced HCA/topology, and Pro-sized DeepEP LL heap: `ROCSHMEM_HEAP_SIZE=12884901888`, `DUSHMEM_HEAP_SIZE=12884901888`, plus the README low-latency context variables. Correctness-only passed against `ll-masked`: `max_abs=0.000976562`, `mean_abs=4.86086e-05`.
- Full Pro EP8 LL 512 eager timing with `ll-masked` baseline passed. Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_512_isolation_20260703_212455/pro_ep8_ll_512_llmasked_timing_heap12g.json`. Fused eager median `16.6829 ms`, `ll-masked` baseline median `5.6826 ms`, speedup `0.3406x`.
- Pro EP8 LL graph cap512 replay512 also passed correctness. Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_512_isolation_20260703_212455/pro_ep8_ll_graph512_llmasked_timing_heap12g.json`. Graph replay median `16.6293 ms`; `ll-masked` baseline graph median `5.7480 ms`.
- `hy-smi --showpids` was clear after the LL runs.
- Conclusion: Pro EP8 LL 512 is correct with `ll-masked` after increasing DeepEP LL heap, but fused LL is much slower than the `ll-masked` baseline. This is a Pro LL performance issue, not a correctness issue and not only launch overhead.

## 2026-07-03 22:04:00 +08:00 - Pro LL Graph Performance Triage
- Ran Pro EP16 LL graph cap512 on 151.1 with `--baseline-kind ll-masked`, replay `8,32,128,256,512`, Pro shape `experts=384`, `topk=6`, `hidden=7168`, `intermediate=3072`, `num_ranks=16`, and 12GiB ROCSHMEM/DUSHMEM heaps.
- Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_graph512_masked_20260703_214635/result.json`.
- Correctness passed for eager and all graph buckets. Fused graph vs `ll-masked` baseline graph:
  - replay8: `3.2558 ms` vs `2.4204 ms`;
  - replay32: `3.3209 ms` vs `2.4802 ms`;
  - replay128: `5.1238 ms` vs `2.6445 ms`;
  - replay256: `8.6320 ms` vs `3.3827 ms`;
  - replay512: `15.2628 ms` vs `4.7778 ms`.
- This confirms Pro EP16 LL graph has the same abnormal performance pattern as Pro EP8 LL 512: correctness is clean, but fused LL is slower than the `ll-masked` baseline.
- Tested split-tail off for Pro EP8 LL graph512. Result: fused graph worsened from `16.6293 ms` to `17.1197 ms`, while baseline stayed about `5.7365 ms`. Split-tail is not the root cause.
- Temporarily changed Python `V3_LL_BLOCK_M` for ablation only, without rebuilding because the compiled K1/K3 extensions already contain 32/48/64 variants:
  - Pro EP16 replay512 default block32: `15.2628 ms`;
  - block64: `13.7521 ms`;
  - block48: `12.9954 ms`;
  - `ll-masked` baseline stayed about `4.78 ms`.
- Pro EP8 replay512 with block48 improved only modestly: fused graph `15.5261 ms` vs default `16.6293 ms`, baseline `5.7513 ms`.
- Restored local and remote `V3_LL_BLOCK_M = 32` after the ablation. 151.1 source check shows the default is back, and `hy-smi --showpids` is clear.
- Ran `hipprof --hip-trace --stats --show-pid --follow-fork` for Pro EP16 LL fused-only block48. Result path: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_blockm48_hipprof_20260703_215257/`.
- Profiling boundary: K1 dominates. Representative per-rank HIPOPS showed `V3_K1_LowLatencyMaskedGroupGemmKernel<24, 6144, 7168, ...>` around `11.5-11.8 ms` per call, while K3 LL group GEMM was about `0.8-0.9 ms`, K3 combine/reduce about `0.47-0.89 ms`, and K2 about `0.08 ms`.
- Current conclusion: Pro LL performance issue is rooted in the Pro K1 LL kernel. Tuning `ll_block_m` helps slightly but does not close the gap. Normal backend is currently the performance-safe Pro path; fixing LL properly requires a Pro-optimized K1 LL path or routing Pro small-token auto selection away from LL.

## 2026-07-03 22:16:00 +08:00 - Flash-vs-Pro K1 Profile Comparison
- User pointed out that K1 slowness should be checked from the Pro-vs-Flash code path differences, not explained away by model size.
- Checked 151.1 cards first; `hy-smi --showpids` was clear.
- Ran Flash EP16 LL fused-only hipprof under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/flash_ep16_ll_block32_hipprof_20260703_215928`.
- Flash EP16 token512 block32 result: fused eager median `1.8373 ms`; K1 median `1.3383 ms`, K3 GEMM median `0.3625 ms`, K3 combine median `0.6115 ms`, K2 median `0.0352 ms`.
- Ran Pro EP16 LL default block32 fused-only hipprof under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_block32_hipprof_20260703_220049`.
- Pro EP16 token512 block32 result: fused eager median `15.7026 ms`; K1 median `13.9043 ms`, K3 GEMM median `1.0253 ms`, K3 combine median `1.2074 ms`, K2 median `0.0823 ms`.
- K1 measured ratio is about `10.39x` versus the theoretical K1 work ratio `2.625x`, so Pro K1 has a real efficiency collapse.
- Diff/source check: Pro LL K1 support only routes the existing HIP C++ K1 template from Flash `<4096,4096>` to Pro `<6144,7168>` and enables local experts `24/48`; no dedicated Pro-optimized LL K1 kernel exists.
- Source check: K1 GEMM uses actual `cur_tokens` for `m_tiles`, so it is not doing full capacity rows; stage copy scales mainly with hidden and cannot explain the 10x K1 time.
- Next ablation: test whether the fully unrolled Pro `kKIterations=112` loop is the main codegen/runtime issue.

## 2026-07-03 22:23:00 +08:00 - K1 K-Loop Unroll Ablation
- Applied a temporary K1 LL ablation changing the main K loop pragma from full `#pragma unroll` to `#pragma unroll 1`.
- Synced the header to 151.1, rebuilt MegaMoE in-place, and source pytest passed (`11 passed`).
- Pro EP16 LL fused-only token512 timing with the ablation was worse: fused median `17.3007 ms` versus the default block32 fused-only `15.7026 ms`.
- Conclusion: simply disabling the Pro K-loop unroll is not the fix and does not explain the K1 efficiency collapse. The temporary local change was reverted after the measurement; remote source was also synced back to the default header for the next rebuild.

## 2026-07-03 22:40:00 +08:00 - K1 Stage-Copy Ablation
- Applied a temporary K1 LL launch ablation keeping blockM=32 but changing `kParallelStageCopy` from true to false for the block32 variants.
- Synced `k1_v3_fused_ext.cu` to 151.1, rebuilt MegaMoE in-place, and source pytest passed (`11 passed`).
- Pro EP16 LL fused-only token512 timing with stage-copy off was `15.8442 ms`, slightly worse than the default block32 `15.7026 ms`.
- Conclusion: the Pro K1 gap is not caused by the parallel stage-copy strategy. The next ablation should target grid/CU count or K1 tiling/occupancy rather than copy scheduling.

## 2026-07-03 23:58:00 +08:00 - K1 Dimension Ablation
- Added temporary non-production shape gates and K1/K3 LL dispatch instances to isolate Pro LL K1 cost by dimension. These gates were for measurement only.
- Mixed experts-only ablation passed on 151.1: `experts=384, hidden=4096, intermediate=2048, EP16 LL` under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/mixed_proexperts_flashdims_ep16_ll_20260703_231249`, fused `1.8772 ms`. This is essentially Flash-like, so `experts=384/local_experts=24` is not the cause.
- K/N split ablations under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_dim_ablation_runs_20260703_233714`:
  - `hidden=7168, intermediate=2048`: correct, fused `10.6547 ms`.
  - `hidden=4096, intermediate=3072`: correct, fused `2.5402 ms`.
- Hipprof confirms the K-only case is still K1-dominated:
  - K-only profile `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_konly_hipprof_20260703_234023`: fused `10.6549 ms`, K1 `<24,4096,7168>` average about `9.1-9.7 ms`; K3 GEMM about `0.70 ms`, K3 combine about `0.85 ms`.
  - N-only profile `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_nonly_hipprof_20260703_234147`: fused `2.5350 ms`, K1 `<24,6144,4096>` average about `1.56-2.09 ms`; K3 GEMM about `0.55 ms`, K3 combine about `0.61 ms`.
- Temporary-source caveat: source pytest failed while mixed-shape gates were open because `test_v3_model_shape_registry_covers_flash_and_pro_ep_sizes` intentionally asserts mixed shapes are unsupported. This was expected for the ablation.
- Cleaned all temporary mixed-shape gates from local and remote source. Remote default rebuild passed under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/default_rebuild_after_k1_dim_ablation_20260703_234453`, source pytest `11 passed`, grep confirmed temp strings absent, and `hy-smi --showpids` was clear.
- Conclusion: Pro LL K1 performance collapse is primarily triggered by `hidden/K=7168`; the Pro intermediate/N expansion is secondary, and expert count/local_experts is not the root cause.

## 2026-07-04 00:30:00 +08:00 - K1 Partial-Unroll Ablation
- Applied a temporary K1 LL codegen ablation changing only the main K loop from full `#pragma unroll` to `#pragma unroll 8`.
- Rebuilt on 151.1 and ran Pro EP16 LL fused-only token512 under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_kloop_unroll8_20260704_000153`.
- Result: correctness passed, fused `15.7133 ms`, effectively unchanged from the default block32 fused-only `15.7026 ms`.
- Interpretation: partial unroll 8 does not fix the Pro K1 `K=7168` collapse. Combined with the earlier `unroll 1` result (`17.3007 ms`), the issue is not solved by simple K-loop unroll-factor changes.
- Reverted the temporary pragma locally and on 151.1. Remote default rebuild passed under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/default_rebuild_after_unroll8_ablation_20260704_001643`, source pytest `11 passed`, no `#pragma unroll 8` remained, and `hy-smi --showpids` was clear.

## 2026-07-04 00:44:00 +08:00 - K1 Layout And Resource Check
- Re-read the K1/K2/K3 LL source contract before trying a hybrid backend experiment.
- LL K1 writes rows in per-local-expert contiguous layout (`row = local_expert * m_per_expert + row_in_expert`) and returns `ll_actual_m` as per-expert row counts plus a max-count slot. K2 and K3 both interpret `m_indices` as those per-expert counts.
- Normal ASM K1 returns row-wise `m_indices` for compact/non-deterministic row order; the ASM source explicitly documents "Row order is intentionally nondeterministic". Therefore normal ASM K1 cannot be directly spliced into the LL K2/K3 path without a row-layout rewrite or an explicit reorder/count bridge.
- Extracted the current 151.1 K1 extension fatbin and parsed gfx938 metadata through `llvm-readelf --notes`. For EP16 block32, Flash K1 `<24,4096,4096>` uses `vgpr=124`, `sgpr=100`, `sgpr_spill=6`, `private=0`; Pro K1 `<24,6144,7168>` uses `vgpr=132`, `sgpr=106`, `sgpr_spill=4`, `private=208`.
- Resource metadata does not show a catastrophic VGPR spill or LDS increase, so the Pro K1 gap is unlikely to be explained by simple register exhaustion. Next temporary ablation should test Pro-only K1 tile granularity, starting with `blockN=128`, then cleanly revert if it is not useful.

## 2026-07-04 01:04:00 +08:00 - Pro K1 Tile Granularity Ablation
- Verified local and 151.1 source hashes match for `megamoe/opt.py`, `k1_v3_fused_ext.cu`, and `k1_v3_pack5_groupgemm_impl.cuh`. Remote source contains the candidate Pro K1 LL `K=7168 -> blockN=128` selector and Pro `ll_block_m=48`; Flash shape still maps to blockM32/blockN256.
- Checked 151.1 before the run: `sglang_megamoe` is up, all 16 HCUs are Normal, and `hy-smi --showpids` reports no KFD PIDs.
- Ran Pro EP16 LL graph cap512 against `ll-masked` baseline under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_graph512_blockn128_blockm48_20260704_010417`.
- Correctness passed for eager and graph buckets. Graph medians for fused vs `ll-masked` baseline:
  - replay8: `1.3350 ms` vs `2.4166 ms`;
  - replay32: `1.4698 ms` vs `2.4787 ms`;
  - replay128: `1.6748 ms` vs `2.6485 ms`;
  - replay256: `3.0813 ms` vs `3.3804 ms`;
  - replay512: `5.0180 ms` vs `4.7777 ms`.
- Compared with the prior Pro EP16 LL graph result, replay512 improved from `15.2628 ms` to `5.0180 ms`. This makes the K1 performance issue a tile-granularity/codegen problem for the Pro `K=7168` instantiation, not a baseline or graph-capture issue.

## 2026-07-04 01:10:00 +08:00 - Flash And Pro EP8 LL Guardrails
- Ran Flash EP16 LL graph cap512 guardrail under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/flash_ep16_ll_graph512_guard_blockn128_candidate_20260704_010641`.
- Confirmed Flash still uses `_v3_ll_block_m(...Flash...) == 32`; Pro uses 48. Correctness passed. Flash graph medians replay `8/32/128/256/512` are `0.3774/0.4087/0.5981/0.9885/1.9169 ms`, matching the recent fair historical band (`0.3763/0.4113/0.5980/0.9915/1.9043 ms`).
- Ran Pro EP8 LL graph cap512 under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_graph512_blockn128_blockm48_20260704_010823`.
- Pro EP8 LL correctness passed. Eager fused is now `5.3814 ms` vs `ll-masked` baseline `5.7026 ms`; graph replay512 is now `5.3764 ms` vs baseline `5.7417 ms`.
- Compared with the prior Pro EP8 LL graph replay512 result (`16.6293 ms`), the Pro K1 tile change removes the 16ms-level performance cliff and makes Pro EP8 LL performance competitive with the masked baseline.
- Added static source tests for Pro/Flash LL blockM selection and K1 `K=7168 -> blockN=128`. Local `py_compile` passed; local pytest is unavailable because the local Python lacks pytest. Synced the test file to 151.1 and remote source pytest passed: `11 passed in 7.49s`.

## 2026-07-04 01:15:00 +08:00 - Post-K3 Barrier Hypothesis Rejected
- Tested the normal graph post-K3 barrier candidate on Flash EP16 normal graph cap8192 under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/flash_ep16_normal_graph8192_postk3barrier_probe_20260704_011221`.
- The run passed eager correctness first (`max_abs=0.000488281`), then VMFaulted during graph replay. The failing kernel was still `rank_barrier_kernel`, slot32, with generation/release mismatch (`generation=2`, `release=1`), and the process aborted with `SIGABRT`.
- This disproves the simple "add post-K3 barrier kernel" hypothesis. The extra barrier did not serialize the replay epoch safely and should not be kept.
- Checked 151.1 after the failure: container stayed up, all 16 HCUs are Normal, and `hy-smi --showpids` reports no KFD PIDs.
- Removed the failed tail-reduce graph post-K3 barrier branch and its static test assertions from local source, synced `opt.py` and the test file to 151.1, and reran remote source pytest: `11 passed in 7.51s`.

## 2026-07-04 01:25:00 +08:00 - Pro LL K1 Next Optimization Plan
- User requested continuing Pro LL K1 optimization by extracting a pure groupgemm skeleton, tuning it to the best achievable K1-only performance, and then folding proven structure back into the fused kernel.
- Added a Pro LL K1 optimization sub-plan to `task_plan.md`. The plan is to separate K1 compute from routing/K2/K3/combine costs, benchmark several tile/block/CU variants, compare against fused LL and `ll-masked` baseline over multiple token buckets, then only backport measured improvements.
- Initial acceptance criteria: keep Flash EP16 LL graph in the historical band, keep Pro correctness against `ll-masked`, and make Pro LL fused match or beat `ll-masked` on the tested small-token buckets.

## 2026-07-04 08:05:00 +08:00 - Pro LL K1-only Harness Validation
- Added a K1-only benchmark path to `test_mega_moe_dcu.py`: `--k1-only-bench` calls LL K1 directly using the fused route/stage contract and records K1-only median/min, rows per rank, max rows per expert, and effective LL blockM.
- Added `--k1-only-ll-block-m {0,32,48,64}` so Pro LL K1 blockM can be ablated without changing production selector code. `0` records the effective production selector value.
- Local checks passed: `python -m py_compile megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py megamoe/opt.py` and `git diff --check`.
- Synced `opt.py`, K1 LL source/header, and the two test files to 151.1. Remote source checks passed in `sglang_megamoe`: `11 passed in 7.26s`.
- 151.1 was idle before testing: `sglang_megamoe` up, all 16 HCUs normal, and `hy-smi --showpids` reported no KFD PIDs.
- Started Pro EP16 K1-only blockM ablation under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_only_blockm_ablation_20260704_075902`, beginning with blockM0/production selector. The command printed fused execution setup, then SSH was closed by the remote host before a JSON result was written.
- Follow-up SSH probes currently fail with connection close or banner timeout. Treat this as a node interruption; do not count the blockM0 attempt as valid K1-only performance data.
- Local safety follow-up: added a K1-only benchmark epoch guard. Each timed K1-only invocation now synchronizes the local DCU and enters a host `dist.barrier(group=group)` before the next warmup/repeat iteration. This protects the benchmark loop from re-entering the LL start barrier while a slower rank is still finishing the previous K1 GEMM. Local `py_compile` and `git diff --check` passed after the change.
- Tightened the static source assertion to check the dedicated `run_k1_only_epoch` helper and the guarded timing scope string. Local checks still pass.
- 151.1 remains unavailable after repeated probes (`Connection closed` or `banner timeout`), so the epoch-guarded harness has not yet been synced or remote-validated. First step after recovery: sync the two test files, rerun remote source pytest, then run one short Pro EP16 K1-only blockM0 smoke with low warmup/repeat before the full blockM matrix.

## 2026-07-04 08:35:00 +08:00 - Pro LL K1 Internal Ablation Direction
- User clarified that the intended ablation is to remove K1 internals such as kernel-side barriers or dispatch/stage work, not only sweep blockM.
- 151.1 recovered briefly. Restarted the existing `sglang_megamoe` container, confirmed all 16 HCUs were idle, synced the guarded K1-only harness, and reran remote source pytest: `11 passed in 7.89s`.
- Retried a short Pro EP16 K1-only full-mode smoke (`warmup=1`, `repeat=3`, production blockM selector). It again printed test setup and then SSH was closed by the remote host before any JSON result was written. Treat this as a reproduced full-K1-only instability, not a valid timing.
- Local code now adds two internal ablation modes:
  - `--k1-only-ablate-mode no-start-barrier`: uses a host rank barrier before launch and disables the K1 kernel start rank barrier.
  - `--k1-only-ablate-mode pure-gemm`: after one initialization K1, launches a C++ pure groupgemm path that skips dispatch/route scan/stage copy and reads the existing `actual_m`, `staged_x`, and `staged_x_scale`.
- Local checks after the ablation code passed: `python -m py_compile ...` and `git diff --check`.
- Remote sync/build for the pure-gemm C++ changes is still pending because 151.1 returned to SSH close/banner-timeout after the reproduced full-mode failure.

## 2026-07-04 11:20:00 +08:00 - Pro LL K1 Ablation Data And Current Blocker
- Synced the pure-gemm C++ path to 151.1 after recovery, rebuilt in-place with `build_dcu_megamoe.sh`, verified imports resolve to `/root/yuguo/DeepGEMM`, and reran remote source pytest: `11 passed`.
- Full K1-only mode with the in-kernel start rank barrier remains unsafe: the short Pro EP16 token512 run closed SSH before writing JSON. Do not use this mode for timing.
- `no-start-barrier` Pro EP16 token512 smoke produced valid JSON under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_ablate_no_start_20260704_110101/result.json`: median `4.7167 ms`, min `4.4850 ms`, rows/rank `3072`, max rows/expert average `149.375`.
- Matrix run under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_ablate_matrix_20260704_110224`:
  - `no-start-barrier` medians for tokens `8/32/128/256/512`: `1.4832/1.6491/1.7492/2.9272/4.5657 ms`.
  - `pure-gemm` medians for tokens `8/32/128/256/512`: `1.0704/1.1419/1.1690/2.1947/3.4260 ms`.
- Interpretation: dispatch/route scan/stage copy/start-barrier overhead is about `0.41/0.51/0.58/0.73/1.14 ms` across the same buckets, while the 512-token pure GEMM core itself is still about `3.43 ms`.
- BlockM core ablation started under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_core_blockm_20260704_111154`. Valid results before interruption:
  - pure-gemm token256 bm32 `2.1013 ms`, bm48 `2.5911 ms`, bm64 `10.1278 ms`.
  - pure-gemm token512 bm32 `3.8352 ms`, bm48 `3.4162 ms`.
- `ll_block_m=64` is rejected for Pro LL K1: it is already much slower at token256 and the token512 bm64 run interrupted SSH before writing JSON.
- Current 151.1 status after the bm64 interruption: TCP/SSH to `10.17.151.1:22` times out or closes during banner exchange, so no card/container state can be read yet.
- Next valid step after recovery: do not rerun bm64. Rerun bm32 vs bm48 for token256 and token512 with a larger repeat count, then decide whether another blockN/CU variant is justified.

## 2026-07-04 11:45:00 +08:00 - Local Pure-GEMM BlockN Diagnostic Prep
- While 151.1 remained unreachable by SSH banner timeout, added a local pure-gemm-only Pro K1 diagnostic knob: `--k1-only-ll-block-n {0,64,128,256}`.
- The new knob is guarded to `--k1-only-ablate-mode pure-gemm`; production fused K1 still uses the existing selector (`K=7168 -> blockN=128`, Flash `K=4096 -> blockN=256`) when `ll_pure_block_n=0`.
- C++ changes are limited to the pure groupgemm launcher and the shared kernel static assertion, adding Pro-only pure `blockN=64/128/256` instantiations. Non-Pro pure blockN override is rejected unless it is default/256.
- Local checks passed: `python -m py_compile megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_opt/K1_fused/k1_fused.py` and `git diff --check`.
- Local pytest is still unavailable (`No module named pytest`). Remote source pytest/build must be run after 151.1 recovers.

## 2026-07-05 08:54:36 +08:00 - Pro K1 Pure GroupGEMM vs DeepGEMM Masked Baseline
- Rechecked 151.1 before testing: `sglang_megamoe` was up, all 16 HCUs were idle, and `hy-smi --showpids` reported no KFD PIDs.
- Ran a token512 smoke under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_pure_vs_deepgemm_smoke_20260705_085305`. Result: pure K1 `3.7772 ms`, DeepGEMM masked `2.1582 ms`, ratio `1.750`, max_abs `0.00390625`.
- Ran the full Pro EP16 pure K1 vs same-input DeepGEMM masked matrix under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_pure_vs_deepgemm_matrix_20260705_085436`.
- Matrix summary:
  - tokens 8: pure `1.140250 ms`, DeepGEMM `0.891699 ms`, ratio `1.278738`, max_abs `0.00195312`.
  - tokens 32: pure `1.249630 ms`, DeepGEMM `0.962710 ms`, ratio `1.298034`, max_abs `0.00195312`.
  - tokens 128: pure `1.228529 ms`, DeepGEMM `0.967900 ms`, ratio `1.269273`, max_abs `0.00390625`.
  - tokens 256: pure `2.293589 ms`, DeepGEMM `1.405310 ms`, ratio `1.632088`, max_abs `0.00390625`.
  - tokens 512: pure `3.631959 ms`, DeepGEMM `2.162549 ms`, ratio `1.679481`, max_abs `0.00390625`.
- Post-run `hy-smi --showpids` again reported no KFD PIDs.
- Conclusion from user acceptance criterion: pure groupgemm is not达标. Continue by optimizing the pure K1 groupgemm core first; do not spend more time on dispatch/stage/fused overhead until the pure core is closer to same-shape DeepGEMM masked. If Flash's best kernel structure cannot be reused cleanly for Pro, split a Pro-only pure K1 kernel path.

## 2026-07-05 09:18:00 +08:00 - Pro K1 Pure BlockM/BlockN Diagnostic
- Ran Pro EP16 pure K1 same-input DeepGEMM masked comparisons for blockM `{32,48}` and blockN `{64,128,256}` at tokens `256` and `512` under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_k1_pure_bm_bn_deepgemm_20260705_090146`.
- token256 results:
  - bm32/bn64 pure `2.6990 ms`, DeepGEMM `1.4567 ms`, ratio `1.853`.
  - bm32/bn128 pure `2.0946 ms`, DeepGEMM `1.4369 ms`, ratio `1.458`.
  - bm32/bn256 pure `7.8040 ms`, DeepGEMM `1.4206 ms`, ratio `5.494`.
  - bm48/bn64 pure `3.2130 ms`, DeepGEMM `1.4233 ms`, ratio `2.257`.
  - bm48/bn128 pure `2.3095 ms`, DeepGEMM `1.4718 ms`, ratio `1.569`.
  - bm48/bn256 pure `7.1367 ms`, DeepGEMM `1.4788 ms`, ratio `4.826`.
- token512 results:
  - bm32/bn64 pure `4.5758 ms`, DeepGEMM `2.1659 ms`, ratio `2.113`.
  - bm32/bn128 pure `3.8220 ms`, DeepGEMM `2.1721 ms`, ratio `1.760`.
  - bm32/bn256 pure `13.7912 ms`, DeepGEMM `2.1544 ms`, ratio `6.401`.
  - bm48/bn64 pure `4.6950 ms`, DeepGEMM `2.1938 ms`, ratio `2.140`.
  - bm48/bn128 pure `3.6236 ms`, DeepGEMM `2.1727 ms`, ratio `1.668`.
  - bm48/bn256 pure `11.5711 ms`, DeepGEMM `2.1518 ms`, ratio `5.377`.
- Conclusion: blockN128 remains the only viable width, but the best pure K1 ratio is still `1.46x-1.67x` slower than DeepGEMM. Parameter tuning is not enough; next step is source/profile-level optimization or a Pro-only kernel structure.

## 2026-07-05 10:20:23 +08:00 - Pro K1 Reference Correction And MT256 Cleanup
- User challenged the earlier assumption that DeepGEMM masked uses `MT256x256x128`; rechecked local references under `hygon_tmp/K1_groupgemm_fp8`.
- Corrected finding: `ll-masked` should be compared against the masked small-token reference `deepgemm_groupgemm_masked_fp8_marlin_balanced_256x64x128_TN_BF16_WGM8.s` / `DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_MT256x64x128_WGM8_GROUPGEMM_MASKED`. The `MT256x256x128` reference is the normal/contiguous large-tile path.
- The Pro MT256 pure diagnostic branch is not a valid primary optimization direction for LL masked comparison. It also caused a 151.1 SSH/host interruption before producing a result after the pack5 retry.
- Cleaned local code so the unsafe `ll_block_m=256` / MT256 diagnostic path is no longer runnable and removed the unused MT256 Pro kernel/helper code from `k1_v3_pack5_groupgemm_impl.cuh`.
- Local checks after cleanup passed: `python -m py_compile ...` and `git diff --check`. A grep confirms no runnable MT256 diagnostic entry remains; only static tests assert the `ll_block_m == 256` entry stays absent.
- Next step after 151.1 recovers: rebuild, run source pytest, then continue Pro K1 pure optimization from the stable low-latency pure-gemm skeleton against the masked `256x64x128` reference.

## 2026-07-05 11:15:00 +08:00 - DeepGEMM 6a53e9c Masked Kernel Reference Read
- Cloned the requested DeepGEMM repo locally under `hygon_tmp/deepgemm_develop`, checked out `6a53e9c45c7d6b46395c3a85231d5f2322a36a2a`, matching the active installed `deepgemm` package build hash.
- Read the wrapper and dispatch path: `deepgemm/m_group_gemm_nt_masked.py` selects `mode=1002` for Pro K1 masked comparisons, and `csrc/py_itfs_cu/m_grouped_fp8_gemm_nt_masked.cu` maps that to the ASM code object `deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co`.
- Confirmed our test harness compares against `deepgemm.m_grouped_fp8_gemm_nt_masked` with marlin masked weights created only for the baseline. The fused/LL path still uses the existing pack5 layout, so the reference read does not justify another LL weight-layout change.
- Used the 151.1 container only for offline tool inspection, not GPU workload. `llvm-readelf` and `/opt/dtk/aillvm/bin/llvm-objdump` confirmed the installed gfx938 `256x64x128` code object has a single `WGM8_GROUPGEMM_MASKED` kernel and uses the expected `v_mmac`, LDS read, global load, and BF16 store instruction families.
- Current optimization direction: use DeepGEMM masked as a structural target for a Pro-only or Pro-specialized pure K1 GEMM core: 8 waves / 512 threads, row tile 64, N tile 256, and LDS-backed B-side pipeline. Keep Flash and the current LL weight layout stable while testing this.

## 2026-07-05 11:34:40 +08:00 - Local Diagnostic Cleanup
- Cleaned redundant local K1 diagnostic code before the next optimization pass. Removed the `--k1-only-ll-cus` CLI option/result field, the Python wrapper `ll_cus` override, and the unvalidated pure K1 128-CU / 8-wave instantiation.
- Restored the shared LL K1 template launch contract to 4 waves / 256 threads in source assertions. The stable pure-gemm harness remains available for blockN comparisons against same-input DeepGEMM masked.
- Kept production Pro fixes intact: Pro `K=7168 -> blockN=128`, Pro `ll_block_m=48`, no LL weight-layout change, and Flash `K=4096` remains on the original blockM32/blockN256 path.
- Local verification passed after this cleanup: `python -m py_compile ...`, `git diff --check`, and source greps confirm the removed diagnostic controls do not remain in code.

## 2026-07-05 12:12:45 +08:00 - Old C LL GroupGEMM vs Masked ASM Shape Check
- User asked to pause optimization and first answer why the old `C fp8 groupgemm` LL groupgemm used to track the masked ASM, while Pro K1 now has a large gap.
- Checked 151.1 before testing: `sglang_megamoe` was up, `hy-smi --showpids` reported no KFD PIDs, and all 16 HCUs were Normal/idle.
- Ran the old scratch harness `hygon_tmp/K1_groupgemm_fp8` on device 0. Flash `E=32,N=4096,K=4096` directly supports both `--mode c-ll` and `--mode balanced`.
- Flash paired results, using best-of-three within the run:
  - tokens 8: balanced `0.3044 ms`, C-LL `0.2984 ms`, ratio `0.98x`.
  - tokens 32: balanced `0.3044 ms`, C-LL `0.2986 ms`, ratio `0.98x`.
  - tokens 128: balanced `0.3116 ms`, C-LL `0.3064 ms`, ratio `0.98x`.
  - tokens 256: balanced `0.3214 ms`, C-LL `0.3193 ms`, ratio `0.99x`.
  - tokens 512: balanced `0.4267 ms`, C-LL `0.4885 ms`, ratio `1.15x`.
- Added scratch-only Pro template instantiations to the ignored `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` so the same old C-LL skeleton can be measured at Pro `E=24,N=6144,K=7168` with `blockN=256` and `blockN=128`. This does not touch tracked MegaMoE source.
- Pro E24 same-harness results:
  - tokens 8: balanced `0.5897 ms`, C-LL bm32/bn256 `2.7856 ms` (`4.72x`), C-LL bm48/bn128 `0.7932 ms` (`1.35x`).
  - tokens 32: balanced `0.5952 ms`, C-LL bm32/bn256 `2.7859 ms` (`4.68x`), C-LL bm48/bn128 `0.8001 ms` (`1.34x`).
  - tokens 128: balanced `0.6113 ms`, C-LL bm32/bn256 `2.7961 ms` (`4.57x`), C-LL bm48/bn128 `0.7977 ms` (`1.31x`).
  - tokens 256: balanced `0.6938 ms`, C-LL bm32/bn256 `5.5740 ms` (`8.04x`), C-LL bm48/bn128 `1.6100 ms` (`2.32x`).
  - tokens 512: balanced `1.1572 ms`, C-LL bm32/bn256 `11.1422 ms` (`9.63x`), C-LL bm48/bn128 `2.5276 ms` (`2.18x`).
- Attempted a Pro E48 scratch sample, but commands did not reach timing output and a leftover script/process was killed. `hy-smi --showpids` was clear afterward. Do not use E48 scratch data.
- Interpretation: the old C-LL path really was close for Flash, but it was never proven for Pro. The Pro gap is reproducible in the standalone old C harness without MegaMoE dispatch/K2/K3/fusion. The first cliff is `K=7168,N=6144` with old `blockN=256`; `blockN=128` removes most of the cliff but still leaves a meaningful pure GEMM core gap versus masked ASM.

## 2026-07-05 12:42:25 +08:00 - Pro C-LL Scratch Candidate Sweep
- User approved starting Pro pure groupgemm optimization while keeping LL weight layout stable.
- Added scratch-only candidates in ignored `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp`: Pro E24 `8 waves / 512 threads`, `CU128`, and `BM64/BN256` variants. These are not tracked MegaMoE source changes and do not alter weight layout.
- 151.1 was idle before and after the run (`hy-smi --showpids` clear).
- Candidate run directory: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_pro_candidate_20260705_122805`.
- Results versus masked ASM, best timing per command:
  - tokens128: masked `0.6132 ms`; base `4w/CU64/BM48/BN128` `0.7963 ms`; `4w/CU128/BM48/BN128` `0.9969 ms`; `8w/CU128/BM48/BN128` `1.5777 ms`; `8w/CU128/BM48/BN256` `0.9337 ms`; `8w/CU128/BM64/BN256` `5.7192 ms`.
  - tokens256: masked `0.6476 ms`; base `1.6008 ms`; `4w/CU128/BM48/BN128` `1.9821 ms`; `8w/CU128/BM48/BN128` `3.2499 ms`; `8w/CU128/BM48/BN256` `1.9314 ms`; `8w/CU128/BM64/BN256` `5.6905 ms`.
  - tokens512: masked `1.1883 ms`; base `2.4943 ms`; `4w/CU128/BM48/BN128` `3.0151 ms`; `8w/CU128/BM48/BN128` `5.0821 ms`; `8w/CU128/BM48/BN256` `2.9876 ms`; `8w/CU128/BM64/BN256` `11.4877 ms`.
- Conclusion: simply increasing CUs, using 8 waves, or copying the masked row/column tile shape into the direct-load C-LL skeleton is negative. The current `4w/CU64/BM48/BN128` remains the best measured direct-load skeleton. Stop sweeping these launch parameters; next step is profile/ISA comparison to explain the direct-load core gap versus masked ASM.

## 2026-07-05 13:40:31 +08:00 - Pro K1 Pure LDS-backed Pack5 Skeleton
- Profiled current Pro E24/N6144/K7168 pure direct C-LL versus masked ASM under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_pro_profile_20260705_124725`.
  - tokens512 direct C-LL `BM48/BN128`: hipprof kernel average `2.4815 ms`; harness `2.4788 ms`.
  - tokens512 masked ASM: hipprof kernel average `1.1830 ms`; harness `1.1834 ms`.
- ISA slice for the direct C-LL `E24,N6144,K7168,BM48,BN128` kernel shows the core is fully unrolled global-load plus `ds_bpermute`: 1344 static `v_mmac`, 566 `global_load`, 1344 `ds_bpermute`, and 1117 `s_waitcnt` in the slice. The masked ASM has 128 static `v_mmac`, LDS reads/writes, and a shorter WGM8 structure.
- Tested two direct-core structural ablations under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_pro_structural_ablation_20260705_125536`.
  - `K1_LL_MAIN_UNROLL=8` was neutral (`2.4847 ms` vs default `2.4827 ms` at tokens512).
  - Removing `__builtin_amdgcn_sched_barrier(0)` was faster (`2.2951 ms`) but failed correctness later with `max_abs=0.0610352`, so it is not valid.
- Added scratch-only Pro parameterization for the existing LDS-backed pack5 C skeleton, without changing LL weight layout or tracked MegaMoE production code.
  - Fixed `kProblemN/K` template params for Pro `6144/7168`.
  - Fixed lowlat-pack expert stride to `N*K` and K-outer stride to `N*64`.
  - Fixed a Flash-only stage ordering assumption: `stage_iter ^ 16` is valid for Flash K=4096 / 32 stages, but Pro K=7168 / 56 stages can map to out-of-range stages. Pro now uses linear stage order in the scratch LDS candidate.
- Correctness/performance for the Pro LDS-backed pack5 `row256` candidate:
  - Correctness passed against the ASM reference at tokens512: `max_abs=0`, `mean_abs=0`, bit/value mismatches `0`.
  - Matrix run `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/k1_pro_lds_pack_stagefix_matrix_20260705_133237`:
    - tokens128: masked `0.6142 ms`, direct C-LL `0.7964 ms`, LDS row256 `1.2018 ms`.
    - tokens256: masked `0.6432 ms`, direct C-LL `1.5931 ms`, LDS row256 `1.3151 ms`.
    - tokens512: masked `1.1948 ms`, direct C-LL `2.4797 ms`, LDS row256 `1.4948 ms`.
- Current conclusion: LDS row256 is not good for small Pro buckets because it pads to 256 rows per expert, but it is the first correct Pro pack5 skeleton that materially improves the large-bucket pure K1 core. Next step is to port it as a shape-gated Pro candidate for larger buckets while preserving the direct path for small buckets and Flash.

## 2026-07-05 15:05:00 +08:00 - Formal Pro K1 LDS Candidate Rejected Against LL Masked
- Synced the formal Pro K1 LDS row256 candidate and K1-only diagnostic harness to 151.1, rebuilt MegaMoE in `sglang_megamoe`, and verified imports resolve to source-tree `.so` files.
- Added a K1-only diagnostic compare that, only for `--k1-only-ll-block-n 512`, runs the LDS candidate and a direct blockN128 pure K1 on the same staged input/layout before comparing with DeepGEMM `ll-masked`.
- Result after rebuild: LDS vs direct128 mismatch equals LDS vs DeepGEMM masked mismatch. Example failures:
  - rank-local process 6 before rebuild: `max_abs=0.07421875`, argmax `(expert=0,row=5,col=5572)`, LDS `0.1689453125`, direct128/DeepGEMM `0.2431640625`.
  - process 3 after rebuild: `max_abs=0.06640625`, argmax `(expert=1,row=5,col=1017)`, LDS `-0.193359375`, direct128/DeepGEMM `-0.259765625`.
- Scratch `k1_gemm_lds_pro_stagefix` still passes against the normal MT256x256 ASM (`max_abs=0`) for Pro E24/N6144/K7168 token512, so the row256 skeleton is self-consistent with the normal/MT256 orientation, but it is not correct against the LL masked WGM8 baseline used by MegaMoE LL.
- Conclusion: do not connect the row256 LDS candidate to fused LL. The next Pro K1 core work must preserve the LL/masked/direct orientation. Treat the current formal row256 path as a rejected diagnostic unless it is removed or fully rewritten.

## 2026-07-05 15:07:59 +08:00 - Removed Rejected Row256 LDS Candidate From Formal Code
- Removed the tracked `V3_K1_LdsPack5PureGroupGemmKernel` and its LDS store helpers from `k1_v3_pack5_groupgemm_impl.cuh`.
- Removed the formal `block_n == 512` / `LDS256` launch branch and changed the K1 pure blockN control surface back to `{0,64,128,256}`.
- Removed the temporary K1-only lds-vs-direct128 diagnostic branch that only existed to prove the row256 mismatch. The same-input K1 pure vs DeepGEMM masked comparison remains available.
- Updated source static tests so `V3_K1_LdsPack5PureGroupGemmKernel` and `block_n == 512` are now rejected from production K1 code. Next checks: local py_compile, static pytest, diff check, then remote validation after checking card state.

## 2026-07-05 15:27:54 +08:00 - Post-cleanup Build Verification
- Local checks after removing the rejected LDS path passed: `python -m py_compile ...` and `git diff --check`.
- Local pytest could not run because this Windows Python does not have `pytest`; reran the source test in the 151.1 container instead.
- Synced the cleaned K1/Python/test files to 151.1. Container source pytest passed: `11 passed in 7.48s`.
- Rebuilt MegaMoE inside `sglang_megamoe`; build verified fresh source-tree `.so/.co` artifacts and import paths for `megamoe`, `_C`, K1, K2, and K3 extensions.
- Next runtime validation is the surviving K1 pure direct path (`blockN=128`) against same-input DeepGEMM `ll-masked`; do not rerun the removed `blockN=512` LDS path.

## 2026-07-05 16:00:12 +08:00 - New Pro K1 Acceptance Target
- User clarified that Pro LL K1 no longer needs to preserve the existing unified weight layout. A Pro-size independent layout/repack path is allowed as long as it is shape-gated and does not affect Flash or existing LL behavior.
- Updated `task_plan.md` acceptance: pure Pro K1 groupgemm must be `<= 1.05x` DeepGEMM `ll-masked` on tested buckets, with correctness against same-input `ll-masked`; only after that should the path be connected to fused LL and validated e2e.
- A just-started direct128 cleanup matrix was interrupted by the user. Checked 151.1, found the launched K1-only script still running, and stopped only that diagnostic process before proceeding. Do not treat the interrupted partial run as data.
- Next source task: read DeepGEMM masked packing/dispatch and MegaMoE's weight-layout selection so the Pro-only layout can be added without touching Flash.

## 2026-07-05 16:16:08 +08:00 - Preserve Pro Unified-Layout LL Path
- User clarified that the existing Flash-friendly LL kernel should remain available for Pro. It already supports the Pro shape through the `blockN=128` and `blockM=48` tuning and is useful as a unified-layout fused compatibility/fallback path.
- Planning implication: the new Pro masked-friendly K1 optimization must be introduced as an additional shape-gated layout/backend path, not by deleting or silently replacing the current Pro unified-layout LL fused route.

## 2026-07-05 17:07:30 +08:00 - Pro LL Masked-K1 Layout Connected To Fused
- Added a Pro-only `ll_pro_masked` L1 weight layout path. It uses MegaMoE K1 LL dispatch/stage-only to produce route metadata plus `staged_x/staged_x_scale`, then calls DeepGEMM masked FP8 groupgemm for L1; L2 remains the existing unified pack5 path. Flash and the existing Pro unified-layout LL fused path remain unchanged unless `MEGAMOE_DCU_PRO_LL_MASKED_K1=1` or the new public transform/dict key is used.
- Added a stage-only K1 LL launcher restricted to Pro shape `(hidden=7168,l1_rows=6144)` with `ll_block_m=48,ll_cus=64`, covering Pro EP8/EP16/EP32 local experts `{48,24,12}` without instantiating Flash stage-only variants.
- Verification completed so far:
  - Local `python -m py_compile` and `git diff --check` passed.
  - 151.1 source pytest passed: `12 passed`.
  - 151.1 full MegaMoE build passed with fresh `.so/.co` artifact verification and source-tree import checks.
  - Pro EP16 LL masked-layout correctness-only 512 passed against `ll-masked`: `max_abs=0.000976562`.
- Performance bring-up:
  - First e2e attempt was very slow (`61.2 ms`) because the fused helper called `deepgemm.m_grouped_fp8_gemm_nt_masked_impl(..., mode=1002)` directly. The same-input K1-only harness showed the wrapper path is fast (`DeepGEMM masked K1 2.16 ms` at 512), so the direct `_impl` call was replaced with the baseline wrapper `deepgemm.m_grouped_fp8_gemm_nt_masked(...)`.
  - After this fix, Pro EP16 LL masked-layout eager 512 passed correctness and measured `3.877 ms`, versus existing Pro unified-layout LL fused `5.233 ms` and `ll-masked` baseline `4.877 ms` on the same 151.1 setup.
- Next: validate graph/cap512 and additional Pro LL token buckets, then decide whether to make this Pro masked-K1 layout default or leave it explicit behind the layout key/env gate.

## 2026-07-05 17:42:00 +08:00 - Pro LL Masked-K1 Graph Validation
- Confirmed the "good pure groupgemm" path is connected to fused e2e as an additive Pro LL path, but not by merging it into the old monolithic K1 kernel. The active path is K1 LL stage-only route/stage packing, DeepGEMM masked K1 groupgemm on the staged input, then the existing K2/K3 fused flow.
- Existing Pro LL unified-layout fused route remains available and unchanged. Same setup comparison at 512:
  - Pro unified-layout LL fused eager: `5.233 ms`.
  - Pro `ll_pro_masked` LL fused eager: `3.877 ms`.
  - `ll-masked` baseline eager: `4.877 ms`.
- Pro EP16 `ll_pro_masked` graph cap512 passed correctness for replay `8/32/128/256/512` with MegaMoE graph medians `1.014/1.126/1.370/2.239/3.872 ms`; `ll-masked` baseline graph medians were `2.420/2.483/2.646/3.378/4.780 ms`.
- Correctness max_abs for graph replay buckets: `8/32 -> 0.000488281`, `128/256/512 -> 0.000976562`.
- Current status: EP16 512/cap512 proves the independent-layout fused path is numerically correct and materially faster than both the old Pro unified LL path and the `ll-masked` e2e baseline. Still open: Pro EP8 masked-K1 validation, more token buckets if needed, and whether to make the new layout default for Pro LL or keep it opt-in behind layout/env selection.

## 2026-07-05 17:18:00 +08:00 - Pro EP8 Masked-K1 And Unified EP16 A/B
- Checked 151.1 before the run: all 16 HCUs were Normal/idle and `hy-smi --showpids` reported no KFD PIDs.
- Pro EP8 `ll_pro_masked` eager 512 on devices `0..7` passed correctness against `ll-masked`: `max_abs=0.000976562`. Timing: fused `3.917 ms`, baseline `5.678 ms`, speedup `1.45x`.
- Pro EP8 `ll_pro_masked` graph cap512/replay512 also passed: replay `3.929 ms`, baseline graph `5.734 ms`, correctness `max_abs=0.000976562`.
- Pro EP8 unified-layout graph cap512/replay512 A/B, with `MEGAMOE_DCU_PRO_LL_MASKED_K1` unset, also passed correctness. Unified replay `5.314 ms`, baseline graph `5.735 ms`. Same-code EP8 512 comparison is therefore: old unified `5.314 ms`, new `ll_pro_masked` `3.929 ms`, baseline `5.734 ms`.
- Pro EP8 `ll_pro_masked` full graph cap512 matrix also passed. Replay `8/32/128/256/512` medians: MegaMoE `1.435/1.902/2.100/2.424/3.936 ms`; `ll-masked` baseline `3.325/3.744/3.913/4.147/5.729 ms`. Correctness max_abs was `0.000488281` for `8/32` and `0.000976562` for `128/256/512`.
- Current-code Pro EP16 unified-layout graph cap512 comparison, with `MEGAMOE_DCU_PRO_LL_MASKED_K1` unset, passed correctness. Unified graph replay `8/32/128/256/512` medians: `1.379/1.524/1.721/3.168/5.185 ms`; baseline graph medians: `2.415/2.478/2.643/3.380/4.790 ms`.
- Same-code EP16 A/B at replay512: old unified layout `5.185 ms`, new `ll_pro_masked` path `3.872 ms`, baseline `4.780-4.790 ms`. The independent layout is the only currently measured Pro LL path that is both faster than baseline and clearly better at 512.
- Added README usage notes for the Pro LL masked-K1 transform. The public example now shows `transform_fp8_weights_for_mega_moe_v3_pro_ll_masked_k1(...)` with `megamoe_backend="ll"` and explicitly documents that unified pack5 remains the Pro/Flash compatibility fallback.
- Verification after the documentation/source sync: local `python -m py_compile` passed for touched Python files, local `git diff --check` passed, local pytest is unavailable (`No module named pytest`), and 151.1 container source pytest passed: `12 passed in 7.45s`.

## 2026-07-05 17:58:00 +08:00 - K1-only Layout Diagnostic Cleanup
- Removed the stale `ll_asm_compatible_layout` pybind/Python parameter and K1-only harness trigger that belonged to the rejected blockN512/LDS diagnostic direction. The production K1 path now keeps the normal LL row tile path directly and the K1-only harness remains limited to the supported pure blockN `{64,128,256}` diagnostics.
- Added static coverage so `ll_asm_compatible_layout` is treated as a retired token in production/test harness sources, alongside the already rejected `block_n == 512` and `V3_K1_LdsPack5PureGroupGemmKernel` paths.
- First rebuild attempt caught the missing `use_ll` local after parameter removal; fixed locally by retaining `use_ll=true` for the LL pack5 wrapper while removing only the obsolete layout switch.
- Verification after cleanup:
  - Local `python -m py_compile` passed for touched Python files.
  - Local `git diff --check` passed.
  - 151.1 source pytest passed after sync: `12 passed in 7.49s`; after rebuild/smoke, it passed again: `12 passed in 7.29s`.
  - 151.1 rebuild completed and verified fresh source-tree `.so/.co` artifacts; the command's final exit was polluted only by a PowerShell/SSH Python here-doc delimiter issue after build completion, so a separate import check confirmed the rebuilt `k1_fused_ext` loads from the source tree.
  - Pro EP16 `ll_pro_masked` 512 correctness-only smoke passed after rebuild: `max_abs=0.000976562`.
  - Final `hy-smi --showpids` reported no KFD PIDs.

## 2026-07-05 18:00:00 +08:00 - Flash EP16 Guardrail After Cleanup
- Ran Flash EP16 LL graph cap512 after the K1 pybind cleanup/rebuild, with `MEGAMOE_DCU_PRO_LL_MASKED_K1` unset.
- Correctness passed. Graph replay `8/32/128/256/512` medians were `0.378/0.410/0.602/1.003/1.925 ms`; `ll-masked` baseline medians were `0.929/0.942/1.020/1.387/2.037 ms`.
- This matches the recent fair Flash EP16 cap512 band (`~1.904 ms` at replay512) within noise and confirms the Pro-only masked-K1 additions plus cleanup did not materially regress Flash LL graph.

## 2026-07-05 18:12:00 +08:00 - Next Pro LL K1 Fusion Plan
- User corrected the next-step scope: do not use old Pro unified fused K1 as a comparison target for the optimization loop. It remains only a compatibility fallback.
- Current `ll_pro_masked` split path is the interim performance oracle/fallback: K1 stage-only plus DeepGEMM masked L1 GEMM, then MegaMoE K2/K3.
- Next implementation work should target a Pro-specific C pure K1 groupgemm first, analogous to the Flash LL history where the C kernel reached copied masked-ASM performance before being fused.
- Concrete todo list:
  - Build a Pro-size C pure K1 groupgemm kernel for the masked-friendly layout and compare same-input output/timing against DeepGEMM `ll-masked`.
  - Borrow structure from the real masked reference (`256x64x128 WGM8 GROUPGEMM_MASKED`, row tile 64, N tile 256, K tile 128, LDS-backed B-side pipeline), not from the rejected normal/MT256 row256 path.
  - Keep the LL/masked output orientation and `actual_m` contract unchanged.
  - Once pure C K1 is within 5% of DeepGEMM masked, fuse it like Flash LL K1: start barrier, route/stage, actual_m, scale staging, and L1 GEMM in one Pro shape-gated K1 kernel.
  - Only after the fused C K1 candidate exists, run the unified final validation matrix. Until then, avoid spending cycles on more standalone validation beyond build/static sanity.

## 2026-07-05 19:41:50 +08:00 - Direct Masked-Layout Pure K1 Rejected
- Continued the Pro pure K1 effort from the new masked-friendly layout direction.
- The direct masked-layout prototype had one real correctness bug: it decoded the masked DeepGEMM weight layout without applying the physical N16 lane mapping used by the pack5/marlin family. After adding `physical_n16 -> logical_n16`, the large same-input K1 mismatch disappeared and the remaining diff versus DeepGEMM masked is BF16-level (`max_abs=0.00390625`, mean about `4e-09`) on the Pro EP16 token128 check.
- Performance is not close enough to keep this as the optimization path:
  - Pro EP16 token128, `BM64/BN256/8wave/CU128`: pure K1 `6.6811 ms`, DeepGEMM masked `0.9063 ms`, ratio `7.37x`.
  - Pro EP16 token128, `BM48/BN128/4wave/CU64`: pure K1 `5.3238 ms`, DeepGEMM masked `0.9129 ms`, ratio `5.83x`.
- Conclusion: the direct global-load/shuffle masked-layout prototype is useful as a correctness probe, but it is rejected for the 5% target. The next implementation direction is a Pro masked-orientation LDS-backed K1 skeleton that preserves the `ll-masked` output/`actual_m` contract instead of reviving the rejected normal/MT256 row256 orientation.

## 2026-07-05 20:20:00 +08:00 - Formal MegaMoE C K1 Backend Hook
- User asked to stop keeping the experiment only in scratch and instead make a C groupgemm optionally replace the DeepGEMM ASM K1 call inside the MegaMoE Pro masked-K1 path.
- Added an opt-in backend gate in `megamoe/opt.py`:
  - `MEGAMOE_DCU_PRO_LL_MASKED_K1_C_GROUPGEMM=1` switches `ll_pro_masked` L1 from DeepGEMM masked K1 to the MegaMoE C pure groupgemm extension after stage-only route/stage.
  - `MEGAMOE_DCU_PRO_LL_MASKED_K1_C_BLOCK_N={0,64,128,256}` controls the C pure K1 blockN diagnostic, defaulting to `128`.
  - Default behavior remains DeepGEMM masked ASM; Flash and the existing Pro default/fallback paths are unchanged.
- Local checks passed: `python -m py_compile megamoe/opt.py ...` and `git diff --check -- megamoe/opt.py`.
- Synced `megamoe/opt.py` to 151.1. Remote source pytest passed: `12 passed in 7.30s`.
- First formal e2e smoke with the C backend enabled passed correctness on 151.1:
  - Command shape: Pro EP16 LL eager, `tokens=128`, `hidden=7168`, `intermediate=3072`, `experts=384`, `topk=6`, `baseline-kind=ll-masked`.
  - Correctness: `max_abs=0.000488281`, `mean_abs=4.86055e-05`.
  - Performance is poor: fused `5.8634 ms`, short-run baseline timing `1.5520 ms`, speedup `0.265x`.
- Interpretation: the formal optional backend is now in place for iterative optimization, but the current C direct groupgemm is only a correctness-bearing scaffold. The next work is kernel optimization against the masked ASM structure, not more integration plumbing.

## 2026-07-05 21:08:00 +08:00 - Pro C K1 Formal LDS Candidate A/B
- Added a Pro-only masked-layout LDS C K1 candidate into the formal MegaMoE optional backend path. It is selected only when `MEGAMOE_DCU_PRO_LL_MASKED_K1_C_GROUPGEMM=1` and `MEGAMOE_DCU_PRO_LL_MASKED_K1_C_BLOCK_N=256`; the default Pro `ll_pro_masked` path still uses DeepGEMM masked ASM, and Flash paths are unchanged.
- Local checks passed before remote validation: `python -m py_compile` for touched Python/test files and `git diff --check` for the touched K1/opt files.
- 151.1 setup before the run: stale timed-out build process was stopped, process table was clean, and `hy-smi --showpids` reported no KFD PIDs.
- Same-command Pro EP16 LL token128 A/B:
  - Default `ll_pro_masked` path with DeepGEMM masked K1: correctness passed, fused `1.3832 ms`, `ll-masked` baseline `1.5496 ms`, speedup `1.12x`.
  - C LDS backend (`C_GROUPGEMM=1`, `C_BLOCK_N=256`): correctness passed, fused `2.2897 ms`, `ll-masked` baseline `1.5562 ms`, speedup `0.68x`.
- The LDS candidate improves materially over the earlier direct C formal backend (`5.86 ms` fused in the first token128 smoke), but it is still about `1.66x` slower than the default split path and about `1.47x` slower than the DeepGEMM masked timing in the same command.
- Resource metadata for `V3_K1_ProMaskedLdsGroupGemmKernel<24>` shows `group_segment_fixed_size=65536`, `vgpr_count=211`, `sgpr_count=30`, no spills. That is much heavier than the old direct LL template variants around `~100-127` VGPR, and it uses a full 64 KiB LDS double buffer.
- Local/remote source comparison shows the current formal LDS candidate is a 256-thread, 2-compute-wave + 2-loader-wave row64/N256 structure. The actual DeepGEMM masked Pro reference is `mode=1002`, `256x64x128`, 512-thread/WGM8. The next optimization should not be another blockN sweep; it should reduce the C kernel toward the masked ASM structure, likely by splitting the N256 work across more compute waves to reduce per-wave accumulator pressure while preserving the LL masked orientation.
- Attempted a K1-only same-input compare for the C LDS backend with `--k1-only-ablate-mode pure-gemm --k1-only-ll-block-n 256 --k1-only-compare-deepgemm`. The command printed setup, then SSH to 151.1 was closed; subsequent `ssh` and ping/TCP probes timed out. Do not count this as a timing or correctness result. Treat 151.1 as unavailable until it recovers, then first check/clean node state before rerunning K1-only diagnostics.

## 2026-07-05 21:07:07 +08:00 - Pro C K1 ASM-Guided Optimization Direction
- User requested the next Pro C K1 optimization to stay close to the copied masked ASM and to use `--save-temps` for generated device `.s` comparison.
- Adopted that as the next loop: generate the formal C LDS backend `.s`, compare against `hygon_tmp/K1_groupgemm_fp8/deepgemm_groupgemm_masked_fp8_marlin_balanced_256x64x128_TN_BF16_WGM8.s`, and only then change the C kernel.
- Initial expected deltas to confirm in generated `.s`: C candidate is currently `256-thread`, `64 KiB` LDS, `211` VGPR, while the reference masked ASM is `512-thread/WGM8`, `32 KiB` LDS, `191` VGPR. Optimization should target launch/wave ownership, accumulator pressure, LDS B-side read schedule, wait/barrier placement, and `v_mmac` grouping rather than another blind blockN sweep.

## 2026-07-05 23:25:00 +08:00 - Current C K1 Save-Temps Comparison
- Synced the cleaned K1 files to 151.1 and generated the current formal Pro masked LDS C backend device assembly with `hipcc -save-temps=obj`.
- Current C `<local_experts=24>` extracted assembly:
  - run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_save_temps_m32_current_20260705_224241`;
  - extracted file: `c_promasked24_m32_current.s`;
  - resource/count summary: `64 KiB` LDS, `211` VGPR, `26` SGPR, `128` `v_mmac`, `32` `ds_read_b128`, `24` `ds_read_b32`, `35` global-to-LDS loads, `36` `buffer_load_dwordx4`, `256` buffer stores, `61` barriers, `379` waitcnts.
- DeepGEMM masked object/source comparison confirms the main shape gap is not MMAC count. The reference has the same `128` `v_mmac`, but its WGM8 structure is much lighter around scheduling and epilogue: `32 KiB` LDS, `191` VGPR, 512-thread launch, compute waves `0..3`, loader waves `4..7`, wave id contributing to N offset (`NperWAVE=16`), and far fewer static barriers/waits/stores.
- Tried a targeted single-LDS-stage experiment to match ASM's `32 KiB` LDS footprint. It compiled and resource metadata dropped to `32 KiB`, but K1-only same-input correctness failed badly: `max_abs=0.9892578125`, example `pure=0.86328125` vs DeepGEMM `-0.1259765625`.
- Root cause of that failed experiment: the current C kernel's loader waves begin writing the next K-stage LDS buffer immediately after the stage barrier, while compute waves are still consuming the current stage. The double buffer is required unless the whole pipeline is rewritten like the ASM.
- Reverted the single-stage change locally and synced the restored header to 151.1. Rebuilt the in-place extension in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_restore_m32_build_20260705_232513`.
- Restoration K1-only smoke after rebuild returned to BF16-level agreement with DeepGEMM masked (`max_abs=0.00390625`, mean `4.3e-09`) but showed timing jitter: K1 median average `2.5683 ms`, min average `1.7723 ms`, DeepGEMM masked median `0.9598 ms`. Treat this as a restore/correctness check, not as a new performance baseline.
- Next viable C candidate should not split rows like the earlier bad 512-thread/M16 experiment; it should preserve the LL/masked output contract while moving wave ownership toward the ASM's N-split WGM8 mapping.

## 2026-07-05 23:50:00 +08:00 - Pro K1 Tune-Only Extraction Started
- User requested avoiding full fused-extension rebuilds and optimizing the C kernel directly against the DeepGEMM masked `.s`.
- Added a tune-only source file `megamoe/dcu_megamoe_opt/K1_fused/k1_pro_masked_lds_tune_ext.cu`. It exposes only `pro_masked_lds_groupgemm(out, staged_x, weight_masked, staged_x_scale, weight_scale, actual_m, rows_aligned_per_expert, local_experts)` and launches the current Pro masked LDS K1 kernel for local experts `12/24/48`.
- Added `megamoe/dcu_megamoe_opt/scripts/build_k1_pro_masked_lds_tune_ext.sh` so 151.1 can build just this extension, optionally with `MEGAMOE_DCU_K1_TUNE_SAVE_TEMPS=1` for generated device assembly.
- Added an env-gated K1-only harness path: `MEGAMOE_DCU_PRO_LL_MASKED_K1_TUNE_EXT=1` makes `--k1-only-ablate-mode pure-gemm --k1-only-compare-deepgemm` call the tune extension on the same staged inputs and compare against DeepGEMM masked.
- Local static checks passed: `python -m py_compile megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` and `git diff --check` for the touched tune/test files.

## 2026-07-05 23:58:00 +08:00 - Tune-Only Build And Smoke Passed
- Checked 151.1 before GPU work: `hy-smi --showpids` reported no KFD PIDs.
- Synced `k1_pro_masked_lds_tune_ext.cu`, `build_k1_pro_masked_lds_tune_ext.sh`, and the updated K1-only test harness to 151.1.
- Built only the tune extension with `MEGAMOE_DCU_K1_TUNE_SAVE_TEMPS=1`. Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_build_20260705_234136`; elapsed compile time was about `56s`, import path `/root/yuguo/DeepGEMM/megamoe/dcu_megamoe_opt/K1_fused/k1_pro_masked_lds_tune_ext.cpython-310-x86_64-linux-gnu.so`.
- Generated C device assembly: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_build_20260705_234136/build/megamoe/dcu_megamoe_opt/K1_fused/k1_pro_masked_lds_tune_ext-hip-amdgcn-amd-amdhsa-gfx938.s`.
- Pro EP16 token128 same-input smoke via the tune extension passed against DeepGEMM masked. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_smoke_t128_20260705_234317`.
  - Tune C K1 median: `1.8035 ms`; DeepGEMM masked median: `0.9566 ms`; ratio `1.885x`.
  - Diff: `max_abs=0.00390625`, `mean_abs=4.3035e-09`, rows aligned per expert `64`, expected_m `45`.
- The warning after JSON output was the usual process-group TCPStore shutdown noise after all useful results were printed; final `hy-smi --showpids` again reported no KFD PIDs.

## 2026-07-05 23:59:30 +08:00 - Scale-Order Precision Alignment
- Read the DeepGEMM masked `.s` `SMQUANT` path. It computes scale products first (`scale_b * scale_a` with packed FP32 multiplies) and then multiplies those scale products into the accumulated C values before BF16 conversion.
- Changed the Pro masked LDS C store helpers to compute `weight_scale * x_scale` first, then multiply by the accumulator. This is a narrow precision-alignment change for the opt-in C/tune kernel path.
- Rebuilt the tune-only extension with save-temps in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_scaleorder_build_20260705_234901`; compile time was about `57s`.
- Re-ran Pro EP16 token128 same-input K1-only smoke in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_scaleorder_smoke_t128_20260705_235025`.
  - C tune K1 median `1.8105 ms`, DeepGEMM masked median `0.9660 ms`, ratio `1.874x`.
  - Diff improved from BF16-level (`max_abs=0.00390625`) to bit-exact for the checked bucket: `max_abs=0`, `mean_abs=0`.
- Generated `.s` structure is otherwise basically unchanged: for `<24>` it still has `128 v_mmac`, `32 ds_read_b128`, `24 ds_read_b32`, `36 buffer_load_dwordx4`, `256 buffer_store`, `61 s_barrier`, `377 s_waitcnt`, and `2 s_setprio`. This confirms the scale-order patch is precision-only; performance still requires WGM8/N-split ownership changes.

## 2026-07-06 00:22:00 +08:00 - Pro K1 X-LDS WGM8 Candidate
- Added tune-only C candidates selected by `MEGAMOE_DCU_K1_TUNE_VARIANT`:
  - `wgm8`: 512-thread/N-split version that still stages weight in LDS.
  - `xlds_wgm8`: 512-thread/N-split version that stages input `x` in LDS and direct-loads masked weights, matching the DeepGEMM masked asm dataflow more closely.
- Results on Pro EP16 token128 same-input K1-only compare:
  - Existing 256-thread weight-LDS after scale-order: `1.8105 ms` vs DeepGEMM `0.9660 ms`, ratio `1.874x`, bit-exact.
  - `wgm8` weight-LDS: correct but slower, `2.3890 ms` vs `0.9759 ms`, ratio `2.448x`. Root cause: N-split compute waves duplicate global `x` loads because the dataflow is still weight-LDS.
  - `xlds_wgm8` input-LDS, contiguous N ownership: correct, `1.4220 ms` vs `0.9808 ms`, ratio `1.450x`, bit-exact.
  - `xlds_wgm8` plus masked-only small-token epilogue: correct, `1.4060 ms` vs `0.9746 ms`, ratio `1.443x`, bit-exact; generated `.s` for `<24>` has `16 KiB` LDS, `104` VGPR, `64 buffer_store`, and `155 s_waitcnt`.
  - `xlds_wgm8` plus asm-style interleaved N16 ownership: correct, `1.2900 ms` vs `0.9800 ms`, ratio `1.316x`, bit-exact. This is the current best C pure K1 candidate.
- Negative result: applying the asm row-based LDS swizzle directly to the current x-LDS layout failed correctness (`max_abs=1.67578125`), so that swizzle patch was reverted. Do not retry that exact swizzle without re-deriving the LDS layout.
- 151.1 was checked after the failed swizzle run; `hy-smi --showpids` reported no KFD PIDs.

## 2026-07-06 01:16:00 +08:00 - Pro K1 X-LDS Full-Tile Loader Optimization
- Applied a narrow ASM-guided C change to `xlds_wgm8`: the x-LDS loader now removes redundant `linear < 512` checks and uses a branchless x prefetch path for full 64-row tiles, while preserving guarded loads for partial tiles.
- Checked 151.1 before GPU work: `hy-smi --showpids` reported no KFD PIDs.
- Synced the tune header/source/build script/test harness to 151.1 and rebuilt only the tune extension with save-temps. Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_fulltile_build_20260706_010516`.
- Pro EP16 same-input K1-only matrix against DeepGEMM masked now passes bit-exact on all checked buckets:
  - tokens `8/32/128/256/512`
  - C tune K1 medians `1.0785/1.1309/1.1506/1.6135/2.5278 ms`
  - DeepGEMM masked medians `0.8675/0.9616/0.9898/1.4401/2.1596 ms`
  - ratios `1.243/1.176/1.162/1.120/1.170`
  - `max_abs=0`, `mean_abs=0` for every bucket.
- This improves the current-best C pure K1 substantially from the earlier xlds/preload/fullstore family, especially at token128 and token256, but it still misses the acceptance target of `<=1.05x` DeepGEMM masked. Do not fuse this C K1 yet.

## 2026-07-06 02:07:00 +08:00 - Pro K1 ASM-Guided Scheduling Attempts
- Continued from the tune-only `xlds_wgm8` C kernel and compared generated `.s` against the DeepGEMM masked `256x64x128 WGM8` ASM.
- Rebuilt and tested several narrow candidates on 151.1, checking `hy-smi --showpids` before GPU runs; the node reported no KFD PIDs before the measured runs.
- Rejected the "prefetch next half before phase4 wait" candidate. Generated `.s` showed the intended `buffer_load_dwordx4 -> s_waitcnt vmcnt(4) -> phase4` shape, but token128 same-input correctness failed with `max_abs=0.3076171875` and `pure=nan`, so the relaxed wait is unsafe under compiler/hardware load ordering.
- Rejected `#pragma unroll 2` on the x-LDS compute loop. It produced ASM-like static shape (`128 v_mmac`, `16 ds_read_b128`, `VGPR=128`, `LDS=16 KiB`) and stayed bit-exact, but performance did not improve enough and token512 regressed:
  - tokens `128/256/512`: C `1.1387/1.6079/2.6211 ms`, DeepGEMM `0.9598/1.4516/2.2052 ms`, ratios `1.186/1.108/1.189`.
- Rejected the full-tile/partial-tile store-branch split. Static control improved (`s_waitcnt 217 -> 202`, `s_cbranch 131 -> 101`, `v_cmp 101 -> 89`, `VGPR 128 -> 123`), but runtime was neutral-to-worse:
  - tokens `128/256/512`: C `1.1574/1.6076/2.4895 ms`, DeepGEMM `1.0184/1.4359/2.1494 ms`, ratios `1.136/1.120/1.158`.
- Restored the current best split-prefetch baseline on 151.1 in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_best_build_20260706_020211`. Static count is back to `64 v_mmac`, `22 buffer_load_dwordx4`, `128 buffer_store`, `6 s_barrier`, `217 s_waitcnt`, `131 s_cbranch`, `101 v_cmp`.
- Current retained best remains the split-prefetch x-LDS kernel from `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_wasm_splitpref_build_20260706_013607`: token `128/256/512` C medians `1.1491/1.5896/2.4878 ms`, all bit-exact, with full-tile matrix best record still `8/32/128/256/512 = 1.0785/1.1309/1.1506/1.6135/2.5278 ms` before the later scheduling refinements.

## 2026-07-06 02:16:00 +08:00 - Pro K1 Current Baseline Small Buckets And Priority Scope
- Filled in current split-prefetch baseline for small buckets after restoring the tune extension:
  - run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_splitpref_small_matrix_20260706_020446`;
  - token8: C `1.1040 ms`, DeepGEMM `0.8599 ms`, ratio `1.284`, bit-exact;
  - token32: C `1.1511 ms`, DeepGEMM `0.9678 ms`, ratio `1.189`, bit-exact.
- Tested a priority-scope candidate that moves `s_setprio 1` outside the K-stage loop and restores priority only after the loop. Static `.s` confirmed `s_setprio` at lines 10724 and 10988, but performance was worse despite bit-exact correctness:
  - run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_prioscope_matrix_20260706_020903`;
  - tokens `128/256/512`: C `1.1651/1.6854/2.5505 ms`, DeepGEMM `0.9780/1.4390/2.2085 ms`, ratios `1.191/1.171/1.155`.
- Reverted priority-scope and restored the split-prefetch baseline on 151.1 in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_best2_build_20260706_021212`.

## 2026-07-06 02:32:00 +08:00 - Pro K1 ASM-Guided Epilogue And LDS-Wait Attempts
- Tested a packed-epilogue candidate copied from the reference C/ASM style: `v_pk_mul_f32` for scale-pair and output-pair multiplication.
  - Build run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_pkmul_build_20260706_021846`.
  - Static `.s` looked better (`v_pk_mul_f32=128`, `s_waitcnt=107`, `VGPR=123` for `<24>`), but runtime was not better.
  - Pro EP16 same-input K1-only results: token128 `1.1573 ms` vs DeepGEMM `0.9863 ms`, token256 `1.6495 ms` vs `1.4520 ms`, token512 `2.5306 ms` vs `2.1868 ms`; all bit-exact.
  - Rejected and reverted because it regressed versus the retained split-prefetch baseline at 128/256/512 despite nicer static instruction shape.
- Tested an LDS-read overlap candidate: after four `ds_read_b128`, use `s_waitcnt lgkmcnt(2)` to compute the first two row groups before `lgkmcnt(0)` for the last two.
  - Build run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_lgkm2_build_20260706_022738`.
  - Generated `.s` matched the intended `ds_read -> lgkmcnt(2) -> half MMAC -> lgkmcnt(0) -> half MMAC` order.
  - Pro EP16 same-input K1-only results: token128 `1.1465 ms` vs DeepGEMM `0.9546 ms`, token256 `1.6482 ms` vs `1.4012 ms`, token512 `2.5152 ms` vs `2.1777 ms`; all bit-exact.
  - Rejected and reverted because only token128 improved marginally in absolute C time, while token256/token512 regressed and the ratio target remained far from `<=1.05x`.
- Tested a four-load inline asm group for the masked weight loads to better match the hand ASM's B-side load grouping.
  - First build `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_load4_build_20260706_023601` VMFaulted at token256 because the multi-output asm used normal `=v` constraints; generated `.s` allocated output `v[81:84]` on top of offset input `v81..v84`.
  - Fixed the asm outputs to early-clobber `=&v` and rebuilt in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_load4ec_build_20260706_023949`.
  - Corrected load4 results were bit-exact but not faster: token128 `1.1402 ms` vs DeepGEMM `0.9872 ms`, token256 `1.6133 ms` vs `1.4462 ms`, token512 `2.7117 ms` vs `2.1546 ms`.
  - Rejected and reverted because token512 regressed heavily. Lesson: multi-instruction inline asm outputs must use early-clobber when outputs can be written before all input operands are consumed.
- Tested removing the phase0/phase4 accumulator dependency `s_nop` block.
  - Build run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_nosnop_build_20260706_024721`.
  - Pro EP16 same-input K1-only results were bit-exact but mixed: token128 `1.1467 ms` vs DeepGEMM `0.9620 ms`, token256 `1.6651 ms` vs `1.4412 ms`, token512 `2.4719 ms` vs `2.1622 ms`.
  - Rejected and reverted. The tiny token512 gain does not justify a bucket-specific path, and token256 regresses relative to the retained split-prefetch baseline.
- Current retained source is back to the split-prefetch x-LDS baseline. Precision remains bit-exact (`max_abs=0`, `mean_abs=0`) on all checked candidates; remaining work is main-loop scheduling/ownership rather than epilogue precision.

## 2026-07-06 03:13:00 +08:00 - Pro K1 VMFault Experiment Reverted
- Re-read the active Pro C K1 optimization plan and restored the tune-only `xlds_wgm8` source after the aggressive `vmcnt(8)` double-buffer next-weight prefetch experiment VMFaulted.
- Removed the temporary `pro_masked_lds_wait_vmem_le8_device()` helper and reverted the loop-carried `nw00..nw43` pending-load schedule back to the retained split-prefetch order.
- Checked 151.1 before GPU work: `hy-smi --showpids` reported no KFD PIDs.
- Synced the restored header to 151.1 and rebuilt only the tune extension with `MEGAMOE_DCU_K1_TUNE_SAVE_TEMPS=1`. Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_vmcnt8_revert_20260706_031100`.
- Token128 same-input K1-only smoke passed after the revert:
  - run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_vmcnt8_revert_smoke_t128_20260706_031218`;
  - C tune K1 median `1.1661 ms`, DeepGEMM masked median `0.9573 ms`, ratio `1.218x`;
  - correctness remains bit-exact (`max_abs=0`, `mean_abs=0`).
- Interpretation: the restore is healthy. The next optimization loop should compare the regenerated C `.s` against the masked ASM and avoid loop-carried pending VMEM register schedules unless implemented with a stricter hand-asm block.

## 2026-07-06 03:22:00 +08:00 - Pro K1 Store-Control Ablation Rejected
- Compared the retained generated C `.s` with the actual DeepGEMM masked code object ISA. The reference object has `128 v_mmac`, `26 buffer_load_dwordx4`, `16 ds_read_b128`, `64 buffer_store_short`, `12 s_barrier`, `15 s_waitcnt`, and `8 s_setprio`; the retained C `<24>` x-LDS kernel has `64 v_mmac` in its loop body, `22 buffer_load_dwordx4`, `8 ds_read_b128`, `128 buffer_store_short`, `6 s_barrier`, `217 s_waitcnt`, `131 s_cbranch`, and `101 v_cmp`.
- Tested an unconditional padding-row store ablation for the `xlds_wgm8` store macro. Static `.s` improved sharply: `buffer_store_short 128->64`, `s_waitcnt 217->77`, `s_cbranch 131->19`, `v_cmp 101->5`, with VGPR still `123`.
- Pro EP16 same-input K1-only results for this ablation:
  - token128: C `1.1507 ms`, DeepGEMM `0.9673 ms`, ratio `1.190`, `max_abs=0`;
  - token256: C `1.6186 ms`, DeepGEMM `1.4465 ms`, ratio `1.119`, `max_abs=0`;
  - token512: C `2.5129 ms`, DeepGEMM `2.1987 ms`, ratio `1.143`, `max_abs=0`;
  - run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_unmasked_store_matrix_20260706_031722`.
- Rejected the ablation and reverted local source to the masked partial-tile store path. Static control reduction alone is not enough; extra padding-row writes are neutral-to-worse versus the retained split-prefetch baseline.

## 2026-07-06 03:48:00 +08:00 - Pro K1 Offset-Increment Rejected And Baseline Restored
- Reverted the offset-increment experiment in `k1_v3_pro_masked_lds_impl.cuh`: removed `K1_XLDS_W_OFFSET`, `K1_XLDS_LOAD_W_OFFSET`, `kWeightStageStride`, and carried `w_off*` state, restoring direct `K1_XLDS_LOAD_W_AT` recomputation for each prefetch.
- Previous offset-increment run was rejected: build `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_offsetinc_build_20260706_033232`, matrix `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_offsetinc_matrix_20260706_033350`, token `128/256/512` C medians `1.1564/1.6807/2.5308 ms` vs DeepGEMM `0.9736/1.4484/2.1583 ms`, all bit-exact but slower.
- Checked 151.1 before GPU work; no KFD PIDs were active.
- Synced the restored header and rebuilt only the tune extension with save-temps. Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_offsetinc_revert_20260706_034014`.
- Token128 same-input K1-only smoke passed after restore: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_offsetinc_revert_smoke_t128_20260706_034128`; C tune K1 `1.1435 ms`, DeepGEMM masked `0.9687 ms`, ratio `1.180x`, `max_abs=0`, `mean_abs=0`.
- Generated `.s` for `<local_experts=24>` is back to retained split-prefetch shape: `64 v_mmac`, `8 ds_read_b128`, `22 buffer_load_dwordx4`, `128 buffer_store_short`, `6 s_barrier`, `217 s_waitcnt`, `2 s_setprio`, `131 s_cbranch`, `101 v_cmp`, `v_lshl=70`, `s_mul_i32=10`, and no `vmcnt(8)`.
- Current conclusion: precision is already bit-exact on the checked restore path. To approach the hand ASM performance, next work should stop chasing cosmetic static-count edits and instead implement a larger ASM-guided main-loop schedule or re-derived LDS layout, then verify with save-temps and same-input DeepGEMM masked timing.

## 2026-07-06 03:57:00 +08:00 - Pro K1 N64 C Retile Rejected
- Built a tune-only `xlds_wgm8_n64` candidate to test whether matching the reference ASM's N64 tile direction helps when keeping the current M64 row ownership. This changed only the opt-in tune path and did not touch the default fused/Flash flow.
- Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_n64_build_20260706_034902`.
- Token128 same-input K1-only result: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_n64_smoke_t128_20260706_035019`; C `1.8438 ms`, DeepGEMM masked `0.9524 ms`, ratio `1.936x`, `max_abs=0`, `mean_abs=0`.
- Static count for `<local_experts=24>` improved per CTA (`16 v_mmac`, `8 ds_read_b128`, `10 buffer_load_dwordx4`, `32 buffer_store_short`, `62 s_waitcnt`, `46 s_cbranch`), but runtime became much worse. Conclusion: N64 alone increases CTA count by 4x and does not reproduce the reference's M256 ownership or hand-scheduled load/MMAC block.
- Removed the N64 candidate and its launcher branch from local source, synced the cleaned files back to 151.1, and rebuilt tune-only baseline in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_n64_reject_20260706_035300`.
- Restore smoke after cleanup passed: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_n64_reject_smoke_t128_20260706_035414`, repeat=1 C `1.1216 ms`, DeepGEMM `0.9719 ms`, ratio `1.154x`, `max_abs=0`. Treat this only as a restored `.so` sanity check.

## 2026-07-06 04:05:00 +08:00 - Pro K1 Next4 Relaxed-Wait Rejected
- Tried an ASM-guided phase retiming in the current `xlds_wgm8` C kernel: after phase0 MMAC, issue the next-stage phase0 weight loads with an early-clobber single-load helper, then use `s_waitcnt vmcnt(4)` before phase4. The intent was to reproduce the reference's pending-VMEM wait window without the previous load4 output/input aliasing issue.
- Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_next4ec_build_20260706_035733`.
- Generated `.s` partly matched the intended shape (`buffer_load_dwordx4` next4 followed by `s_waitcnt vmcnt(4)`), but compiler/control-flow lowering still inserted a later `s_waitcnt vmcnt(0)` before phase4 MMAC, so the wait window was not cleanly controlled.
- Token128 same-input K1-only failed correctness with NaN: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_next4ec_smoke_t128_20260706_0359*`; error `max_abs=0.40478515625`, `pure=nan`, `deepgemm=-0.040283203125`.
- Reverted the early-clobber helper and relaxed-wait schedule locally, confirmed no residual `pack_ec`, `LOAD_W_AT_EC`, `xlds_wgm8_n64`, or `K1_XLDS64` symbols, then synced and rebuilt tune-only baseline in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_next4ec_reject_20260706_040059`.
- Restore smoke passed: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_k1_tune_ext_xlds_restore_after_next4ec_reject_smoke_t128_20260706_040214`; repeat=1 C `1.1071 ms`, DeepGEMM `0.9871 ms`, ratio `1.122x`, `max_abs=0`.

## 2026-07-06 08:52:10 +08:00 - Pro LL K1 Fused-C Path Implemented
- Wired the retained Pro `xlds_wgm8` C groupgemm backbone into a Flash-style LL K1 fused path behind `MEGAMOE_DCU_PRO_LL_MASKED_K1_FUSED_C=1`.
- New path is Pro-shape gated (`hidden=7168`, `l1_rows=6144`, `ll_block_m=48`, `ll_cus=64`) and leaves the current split `stage-only + DeepGEMM masked K1` path as the default fallback/oracle.
- Added `V3_K1_ProMaskedXLdsWgm8FusedKernel` plus `pro_masked_xlds_wgm8_compute_tile_device`; the fused kernel runs the existing LL route/stage builder and then a persistent tile loop over the retained x-LDS/WGM8 compute tile.
- Added one block barrier after the x-LDS tile compute helper so loader waves cannot enter the next persistent tile and overwrite LDS while compute waves are still finishing stores/scale reads.
- Local checks passed: `python -m py_compile` for touched Python files and `git diff --check` for the touched source set. Local pytest is unavailable because the local Python environment lacks `pytest`.
- Synced the touched source files to 151.1 and completed a full remote MegaMoE rebuild in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_fused_c_build_20260706_082912`; fresh K1 artifacts were generated.
- Remote pytest path handling was anomalous: container-side `find/stat` and `py_compile` saw the test file, but `python3 -m pytest -q ./megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `file or directory not found`. Treat this as a test-runner/path issue until reproduced; it did not block the full compile.
- Next action: check 151.1 HCU state, then run Pro EP16 LL eager 128/256/512 A/B with default split DeepGEMM masked K1 versus the fused-C env gate.

## 2026-07-06 11:05:00 +08:00 - Pro LL K1 Fused-C Eager A/B
- Ran Pro EP16 LL eager A/B on 151.1 with `MEGAMOE_DCU_PRO_LL_MASKED_K1=1`, `--baseline-kind ll-masked`, and ROCSHMEM/DUSHMEM heap set to `12884901888`.
- Initial run without the larger heap failed before MegaMoE execution in DeepEP buffer initialization (`num_rdma_bytes=11450456192` exceeded the default heap). Re-ran with the previously used Pro LL heap setting.
- Default split path (`stage-only + DeepGEMM masked K1`) passed correctness:
  - token128: MegaMoE `1.3687 ms`, ll-masked baseline `2.6419 ms`, `max_abs=0.000488281`.
  - token256: MegaMoE `2.2304 ms`, ll-masked baseline `3.5123 ms`, `max_abs=0.000976562`.
  - token512: MegaMoE `3.8939 ms`, ll-masked baseline `4.8885 ms`, `max_abs=0.000976562`.
- Fused-C path passed correctness but is slower than the default split path:
  - token128: `1.6186 ms`, `max_abs=0.000488281`.
  - token256: `2.5056 ms`, `max_abs=0.000976562`.
  - token512: `4.2445 ms`, `max_abs=0.000976562`.
- Tried a narrow fused-context barrier retiming: move the final persistent-tile block barrier from after output stores to after LDS scale reads but before stores, so next-tile prefetch can overlap the previous tile stores while still protecting LDS reuse. It stayed correct and gave only a small speedup:
  - token128: `1.6108 ms`.
  - token512: `4.2252 ms`.
- Added a readlane-style fused tile dispatch tweak: load `actual_m` once per wave lane and broadcast with `__builtin_amdgcn_readlane`, passing `cur_tokens` into the x-LDS compute helper instead of reloading `actual_m` inside every tile. Remote full rebuild passed in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_fused_c_readlane_build_20260706_102653`.
- Runtime validation of the readlane tweak is pending because 151.1 was taken by an unrelated SGLang/prefix-cache benchmark after the rebuild.

## 2026-07-06 11:38:00 +08:00 - Pro LL K1 Fused-C Readlane Validation Blocked
- Rechecked 151.1 after the prefix-cache benchmark process exited. HCU utilization is idle, but the host `sglang.launch_server` process still holds about `97-98%` VRAM on all 16 cards.
- Do not run Pro EP16 LL validation in this state because the test is expected to fail allocation or perturb another user's resident server. Runtime validation remains pending until the server releases the cards or explicit permission is given to stop/restart it.

## 2026-07-06 13:55:00 +08:00 - Pro LL K1 Fused-C Readlane Eager A/B Complete
- Rechecked 151.1 and found all 16 cards idle before testing.
- Validated the readlane fused-C path with `MEGAMOE_DCU_PRO_LL_MASKED_K1=1`, `MEGAMOE_DCU_PRO_LL_MASKED_K1_FUSED_C=1`, `MEGAMOE_DCU_PEER_MEMORY=rpc`, `--baseline-kind ll-masked`, and ROCSHMEM/DUSHMEM heap `12884901888`.
- Pro EP16 LL eager readlane fused-C correctness passed:
  - token128: `1.6004 ms`, baseline `2.6409 ms`, `max_abs=0.000488281`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_fused_c_readlane_t128_20260706_134745`.
  - token256: `2.4841 ms`, baseline `3.5099 ms`, `max_abs=0.000976562`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_fused_c_readlane_t256_20260706_135012`.
  - token512: `4.2079 ms`, baseline `4.8629 ms`, `max_abs=0.000976562`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_fused_c_readlane_t512_20260706_134901`.
- Re-ran same-window default split DeepGEMM masked-K1 fallback with the fused-C env unset:
  - token128: `1.3710 ms`, baseline `2.6405 ms`, `max_abs=0.000488281`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_split_samewindow_t128_20260706_135153`.
  - token256: `2.2329 ms`, baseline `3.5117 ms`, `max_abs=0.000976562`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_split_samewindow_t256_20260706_135304`.
  - token512: `3.9052 ms`, baseline `4.8821 ms`, `max_abs=0.000976562`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_k1_split_samewindow_t512_20260706_135415`.
- Same-window conclusion: readlane fused-C is correct and slightly faster than the previous fused-C build, but still slower than the default split path by about `16.7%/11.2%/7.8%` at tokens `128/256/512`. Keep fused-C as an opt-in experiment; keep split DeepGEMM masked-K1 as the Pro LL default.

## 2026-07-06 15:10:00 +08:00 - Pro LL Split Finalization Source Cleanup
- User direction changed the final Pro LL plan: stop pursuing K1 fusion and keep only the high-performance split path.
- Integrated the DeepGEMM masked FP8 group GEMM ASM source into `megamoe/dcu_megamoe_opt/K1_fused` and added setup/build rules for the packaged `.co`.
- Added `k1_ll_masked_groupgemm_pack5` inside MegaMoE's K1 extension. `megamoe/opt.py` now runs Pro LL as `stage-only K1 + bundled masked ASM`, with no runtime `import deepgemm` or external `deepgemm.m_grouped_fp8_gemm_nt_masked` call in the MegaMoE execution path.
- Removed rejected Pro fused-C/tune code paths: the fused-C pybind/launcher, tune-only LDS extension, tune build script, and the pack5 include of the tune header.
- Updated test harness behavior: Pro shape with LL backend defaults to `ll_pro_masked`; only `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` forces the unified-layout compatibility path. Removed the old tune-extension runtime knob.
- Updated README to describe the final split behavior and the unified-layout override. Remote rebuild/validation on 151.1 is still pending.

## 2026-07-06 15:45:00 +08:00 - Pro LL Split Cleanup Continued
- Removed the remaining pure-K1 groupgemm C++ launcher and its pybind/Python arguments (`ll_pure_groupgemm`, `ll_pure_block_n`, `ll_pure_masked_weight_layout`).
- Removed the test harness `--k1-only-*` diagnostic surface and the staged-input/grouped-diff helpers that only supported pure-K1 same-input experiments.
- Updated source static tests so they now assert the old pure/fused/K1-only controls do not return, while preserving the `ll-masked` baseline helper for correctness/performance comparisons.
- Local checks passed: `python -m py_compile` on touched Python files and `git diff --check`. A source scan shows the retired Pro LL env knobs and pure-K1 symbols only appear in negative static-test assertions.
- Synced the final source set to 151.1 `/root/yuguo/DeepGEMM`, removed stale remote tune files, and ran remote `py_compile` plus a lightweight source-contract script in `sglang_megamoe`; both passed.
- Remote `python3 -m pytest` still reports `file or directory not found` for an existing test file, matching the earlier container pytest anomaly. Do not treat this as a source failure; use the lightweight contract script until the container pytest path issue is resolved.
- 151.1 runtime build/Pro LL sanity is blocked by an active SGLang DeepSeek-V4-Pro service plus prefix-cache benchmark. `hy-smi` shows 98-99% VRAM across all 16 HCUs, and host processes include `python -m sglang.launch_server ... --port 10015` plus `prefix_cache_benchmark_with_e2e.py`. No MegaMoE workload was launched.
- After two waiting polls, the same service and benchmark were still active, with several HCUs showing nonzero utilization. Leave full rebuild and Pro EP16 LL runtime validation pending until the node is released.
- Removed the remaining `kSkipDispatch` and `kMaskedWeightLayout` template parameters from `V3_K1_LowLatencyMaskedGroupGemmKernel`. They were only needed by the deleted pure-K1 / masked-layout C experiments; the current production instantiation always runs dispatch/stage and uses the unified pack5 LL layout. Local `py_compile`/`git diff --check` passed, and the cleaned header/test were synced to 151.1 with a remote lightweight check.

## 2026-07-06 15:12:27 +08:00 - Public Weight Transform API Cleanup
- Removed the legacy top-level `transform_fp8_weights_for_mega_moe` helper from `megamoe/__init__.py` and from `__all__`.
- Renamed the Pro LL masked-K1 helper from `transform_fp8_weights_for_mega_moe_v3_pro_ll_masked_k1` to `transform_fp8_weights_for_mega_moe_pro_ll_masked_k1`; updated README and source-contract assertions accordingly.
- Cleaned `v3_layout.py` by deleting the unused `unpack_pack5_weight`, `_cast_weight_to_fp8`, `_pack_fp8_weight_and_scale*`, and `transform_fp8_weights_for_mega_moe_v3_pack5*` helper cluster.
- Kept the remaining `v3_layout.py` helpers because they are either used by `flatten_pack5_weight*` or by static pack5 layout contract checks: `pack5_physical_to_logical_indices`, `pack5_logical_to_physical_ni`, `pack5_shape`, `pack5_flat_offset`, `pack5_weight`, `flatten_pack5_weight`, `pack5_weight_asm_normal`, and `flatten_pack5_weight_asm_normal`.
- Local verification passed: `python -m py_compile megamoe/__init__.py megamoe/dcu_megamoe_opt/v3_layout.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`, `git diff --check` on touched files, `rg` scan for retired helper names, and a lightweight Python source-contract script. Full local pytest is unavailable because this Python environment has no `pytest` module.

## 2026-07-06 16:11:38 +08:00 - Planning Status Cleanup
- Re-read `.planning/dcu_megamoe_supernode/task_plan.md`, `progress.md`, and `findings.md`, then updated stale `[]` entries to match the latest Pro LL split decision.
- Marked the Pro LL pure-C/fused-C optimization branch as completed or abandoned where appropriate: same-input C K1 work, save-temps/ASM comparison, fused-C e2e A/B, and cleanup are no longer active tasks because split masked ASM is the chosen production path.
- Kept true pending items as `[]`: final 151.1 rebuild/runtime sanity, Flash EP16 normal graph and large-cap graph instability retests, EP32 validation, Pod2 device exposure, and any future contingent debug branches.
- Updated `findings.md` top-level status so it no longer says the whole EP16 benchmark matrix is pending; the remaining open part is EP32 plus large normal graph-bench stability.

## 2026-07-06 16:18:00 +08:00 - Flash Regression Contingency Closed
- Reclassified the Pro-only normal ASM split item: Flash guardrails collected so far do not show a material performance regression, so this is not a current unfinished task.
- Keep it only as a future contingency if a fair Flash run later regresses.

## 2026-07-06 20:57:00 +08:00 - Pro LL Split Final 151.1 Sanity Complete
- Restarted `sglang_megamoe` on 151.1 after the container had exited; host/card check showed no KFD PIDs before running MegaMoE.
- Fixed the bundled Pro LL masked K1 source artifact before the run: the earlier scratch `.s` copy was not from the active DeepGEMM package/develop checkout. The production package now bundles the active develop `.co` from commit `6a53e9c45c7d6b46395c3a85231d5f2322a36a2a`, hash `73184662ec644cf9f4e9cfacec720a15428e84c5f84ad06e6e9e57bfa06543b4`, and removes the wrong scratch `.s`.
- Rebuilt on 151.1 and verified the prebuilt source `.co`, top-level source-tree `.co`, and build/lib `.co` all have hash `73184662ec644cf9f4e9cfacec720a15428e84c5f84ad06e6e9e57bfa06543b4`.
- Pro EP16 LL default split path correctness/performance run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_split_matrix_20260706_205021`.
- Eager Pro EP16 LL against `ll-masked` baseline passed for tokens `8/32/64/128/256/512`. MegaMoE medians were `0.9780/1.0747/1.2800/1.4883/2.3585/4.0353 ms`; baseline medians were `2.4149/2.4804/2.7022/2.8068/3.7240/5.2574 ms`; speedups were `2.47x/2.31x/2.11x/1.89x/1.58x/1.30x`.
- Graph cap512 Pro EP16 LL passed replay buckets `8/32/64/128/256/512`. MegaMoE graph medians were `1.1223/1.2266/1.3019/1.4499/2.3240/3.9464 ms`; `ll-masked` baseline graph medians were `2.5733/2.6482/2.6863/2.7920/3.6194/5.1602 ms`; speedups were `2.29x/2.16x/2.06x/1.93x/1.56x/1.31x`.
- The runtime path reported `weight_layout=ll_pro_masked`, `fused_execution=v3_ll_eager`, and `graph_execution=v3_ll_cuda_graph_replay`, confirming Pro LL defaults to the high-performance split layout rather than the unified-layout compatibility path.

## 2026-07-06 21:45:00 +08:00 - Pro EP8 LL Split Correctness Isolation
- Pro EP8 LL split token256 is currently not clean on the latest source. Two short runs failed against `ll-masked` with valid route stats and small but real output mismatch (`max_abs` about `0.05-0.07`).
- Added temporary assertion-side combine-slot readback. The latest failing token showed `combine_sum` matches fused output within BF16 rounding, so the final source-rank reduce is likely summing the visible slots correctly; the bad value is earlier, either per-route K1/K2/K3 output or split-tail copy into the per-slot combine buffer.
- A no-split-tail ablation was attempted but invalidated by an active SGLang DeepSeek-V4-Pro service occupying all cards; it failed by allocation pressure, not by a trusted MegaMoE result.
- Current node state blocks more DCU tests: host PIDs `79379..79394` plus parent `78899` hold about `86-89%` VRAM across all 16 HCUs. No new MegaMoE run should be launched until those cards are released.
- Next clean-card sequence: rerun Pro EP8 LL token256 default, rerun the same command with `MEGAMOE_DCU_LL_K3_SPLIT_TAIL=0`, then enable route-slot diagnostics for the first bad `(token, slot, col)` only if the failure persists.

## 2026-07-06 23:32:00 +08:00 - Pro LL Correctness Queue Restart
- User priority reset: fix Pro EP8 LL uniform correctness first, prove token256, then token512, then `8/32/64/128` if needed; after that run Pro LL uneven with EP16 first, then EP8; finally collect split-vs-`ll-masked` eager/graph/uneven data and Flash guardrails.
- Updated `task_plan.md` with the ordered queue and kept transient card-state checks out of the plan body.
- 151.1 is currently usable after the stale SGLang service was terminated: latest check showed all 16 HCUs at `0%` VRAM/HCU and no KFD PIDs.
- Immediate next action is root-cause reproduction, not a source patch: run Pro EP8 LL uniform token256 default against `ll-masked`, then split-tail-off on the same clean node if default fails.

## 2026-07-06 23:46:00 +08:00 - Pro EP8 LL 256 Boundary Isolated To Masked K1 Size
- Reproduced Pro EP8 LL token256 on clean 151.1 cards: default split path failed against `ll-masked` with valid stats and fused output matching the visible combine-slot sum.
- Re-ran token256 with `MEGAMOE_DCU_LL_K3_SPLIT_TAIL=0`; it still failed, so K3 split-tail copy/publish is not the first boundary.
- Route-slot diagnostics for the default failing point showed valid row metadata and nonzero staged scale, but some route rows had zero L1 output, default activation scale, and zero K3 output. This moves the first bad boundary to the Pro LL split K1 masked GEMM launch/input shape.
- Ran current-source Pro EP8 LL token512 before patch: correctness passed against `ll-masked` (`max_abs=0.000976562`) in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_uniform512_check_20260706_234239`.
- Prepared a narrow local fix: only Pro EP8 LL masked K1 is forced to use at least `128` rows/expert in Python scratch sizing, C API route_scratch sizing, and K1 stage-only launch sizing. Local `py_compile` and `git diff --check` passed.

## 2026-07-07 00:10:00 +08:00 - Pro EP8 LL Min-Rows Fix Correctness Pass
- Synced the Pro EP8 LL min-rows patch to 151.1 and rebuilt successfully in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_minrows_build_20260706_234708`; fresh source-tree artifacts were verified by the build script.
- Pro EP8 LL token256 now passes against `ll-masked`: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_uniform256_minrows_20260707_000131`, `max_abs=0.000976562`.
- Pro EP8 LL uniform sweep after the patch passed tokens `512/8/32/64/128`: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_uniform_sweep_minrows_20260707_000240`; max_abs values were `0.000976562/0.000488281/0.000488281/0.000976562/0.000976562`.
- Pro EP16 LL uneven correctness passed with local token list `512,385,257,128,64,32,16,8,7,0,0,0,0,0,0,0`: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep16_ll_uneven_minrows_20260707_000724`, `max_abs=0.000976562`.
- Pro EP8 LL uneven correctness passed with local token list `512,257,128,64,32,7,0,0`: run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_uneven_minrows_20260707_000839`, `max_abs=0.000976562`.

## 2026-07-07 00:25:00 +08:00 - Pro LL Performance And Flash Guardrail
- Pro EP8 LL eager split-vs-`ll-masked` data collected in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_perf_minrows_20260707_001031`.
  - tokens `8/32/64/128/256/512`
  - MegaMoE medians `1.4607/1.9197/1.9813/2.0970/2.4093/3.9271 ms`
  - `ll-masked` medians `1.4291/1.9572/2.1210/2.4853/3.1889/5.6766 ms`
  - speedups `0.978x/1.020x/1.071x/1.185x/1.324x/1.445x`
- Pro EP8 LL graph cap512 collected in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_graph_minrows_20260707_001535`.
  - replay tokens `8/32/64/128/256/512`
  - MegaMoE graph medians `1.4356/1.9034/1.9765/2.1167/2.4469/3.9259 ms`
  - `ll-masked` graph medians `3.3337/3.7530/3.8005/3.9095/4.1563/5.7333 ms`
- Pro LL uneven performance collected in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ll_uneven_perf_minrows_20260707_001644`.
  - EP16 local token list `512,385,257,128,64,32,16,8,7,0,0,0,0,0,0,0`: MegaMoE `1.5895 ms`, `ll-masked` `3.0464 ms`, speedup `1.917x`, avg received tokens/rank `528.375`.
  - EP8 local token list `512,257,128,64,32,7,0,0`: MegaMoE `2.3849 ms`, `ll-masked` `4.3379 ms`, speedup `1.819x`, avg received tokens/rank `750.0`.
- Flash EP8 LL graph guardrail passed after the Pro EP8 min-rows patch in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/flash_ep8_ll_guard_minrows_20260707_001854`.
  - replay tokens `8/32/128/256/512`
  - MegaMoE graph medians `0.5302/0.6394/0.7418/1.0349/1.8153 ms`
  - `ll-masked` graph medians `0.9908/1.1145/1.1942/1.2889/1.9799 ms`
  - Flash replay512 remains in the same band as the earlier guardrail (`~1.814 ms`), so no Flash LL regression is observed.
- Removed temporary Pro EP8 LL debug hooks from `opt.py` and the combine-slot assertion readback from `test_mega_moe_dcu.py`; synced the cleanup to 151.1. Post-cleanup Pro EP8 LL token256 smoke passed in `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_256_after_debug_cleanup_20260707_002226`.
- Local checks after cleanup: `py_compile` and `git diff --check` passed. Local pytest remains unavailable because the Windows Python environment has no `pytest` module.

## 2026-07-07 08:35:00 +08:00 - Runtime-Token-Aligned LL Baseline Fix
- User flagged that `ll-masked` baseline graph must be measured as fairly as MegaMoE graph, using the runtime token bucket rather than graph allocation cap.
- Confirmed the old `run_selected_baseline()` passed `ll_baseline_capacity_tokens` from `sym_buffer.cuda_graph_max_tokens_per_rank` into DeepEP `low_latency_dispatch()`. This made Pro EP8 graph cap512 small buckets dispatch with the larger allocation cap, inflating the baseline graph replay medians.
- Patched `test_mega_moe_dcu.py` so `ll-masked` baseline dispatch capacity is `max(expected_tokens_per_rank, x_bf16_arg.size(0))` for each captured bucket. Added a static contract assertion in `test_dcu_megamoe_v3.py`.
- Local checks passed: `python -m py_compile` for both touched tests and `git diff --check`. Local pytest remains unavailable because the Windows Python environment has no `pytest` module.
- Synced the two test files to 151.1. Remote `py_compile` passed. Remote static pytest did not fully pass because other remote source files were not fully synced with local static-test expectations (`setup.py` prebuilt path and `_v3_ll_block_m` assertion), not because of the new runtime-token baseline assertion.
- Re-ran Pro EP8 LL graph cap512 on 151.1 with `--baseline-kind ll-masked`, replay tokens `8/32/64/128/256/512`, run dir `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/pro_ep8_ll_graph_runtime_baseline_20260707_082358`.
  - MegaMoE graph medians: `1.4314/1.9004/1.9800/2.1053/2.4702/3.9245 ms`.
  - Runtime-token-aligned `ll-masked` graph medians: `1.4417/1.9496/2.1295/2.4835/3.1822/5.7530 ms`.
  - This supersedes the earlier unfair Pro EP8 graph baseline readout `3.3337/3.7530/3.8005/3.9095/4.1563/5.7333 ms` for small buckets.
- Attempted a same-token eager sweep for confirmation, but token8 hit a VMFault during the benchmark loop after correctness. Host inspection showed the node was taken by an unrelated `sglang.launch_server` started at 08:24, with scheduler PIDs `196517..196532` holding 79-82% VRAM on all 16 HCUs. No MegaMoE cleanup/kill was performed.

## 2026-07-07 08:45:00 +08:00 - LL Capacity Overflow Guard
- Reviewed Flash LL, Pro LL, Flash normal, and Pro normal capacity behavior after the user asked about extreme skew. Normal Pro EP8 has stronger compact-prebuild fixed tile-pool protection because `local_experts > 32`; LL still uses fixed `m_per_expert` per expert and can overflow under adversarial routing.
- Added a low-cost LL guard in the shared LL stage builder: when `row_in_expert >= m_per_expert`, set `symm_counts[kExperts + 1]` as a capacity-overflow flag before dropping that route row.
- Changed LL cumulative local expert stats to use the raw route count instead of the clipped count. In correctness harnesses that compare stats, an LL capacity overflow should no longer look like a clean route-stat match.
- Exposed the overflow flag by returning `m_indices` with `local_experts + 2` entries. K2/K3 still consume the original first `local_experts` counts plus the max slot, so normal no-overflow performance behavior should be unchanged.
- This is not a full worst-case-skew compute fix. Fully computing a case such as one expert receiving all routes would require a much larger per-expert capacity or a fallback path, which would affect LL small-bucket performance.
- Local verification passed: `python -m py_compile` for touched Python files and `git diff --check` for the touched K1/test files. Remote compile/runtime validation is pending because 151.1 is occupied by SGLang.

## 2026-07-07 10:03:25 +08:00 - Unified LL Skew Guard Source Patch
- User asked whether increasing `cap_per_expert` can be made safer by letting kernels skip invalid rows. Confirmed K1 unified LL already computes tile count from `actual_m`/`m_indices`, and K2/K3 also compact around `actual_m`, so increasing unified LL capacity mainly increases scratch/init/headroom rather than forcing full GEMM over every padded row.
- Added a source-side unified LL skew guard: workspace sizing now reserves up to 256 rows/expert, bounded by `num_ranks * num_max_tokens`, and the actual K1 launch applies this 256-row guard only when `ll_stage_only` is false. This covers Flash LL and Pro unified-layout LL while keeping the Pro default split masked-K1 path on its existing 128-row EP8 guard.
- Added static contract checks for the new guard in `test_dcu_megamoe_v3.py`.
- Local verification passed: `python -m py_compile megamoe\opt.py megamoe\dcu_megamoe_opt\tests\test_dcu_megamoe_v3.py` and `git diff --check` on the touched source/test files. Local pytest remains unavailable because Windows Python has no `pytest` module.
- Remote rebuild/runtime smoke is still pending; do not claim the Flash LL or Pro unified-layout guard is hardware-validated until 151.1 is free and rebuilt.

## 2026-07-07 10:41:43 +08:00 - Pro LL Graph And Flash Guardrail Retest
- Synced the current MegaMoE source/prebuilt subset to 151.1, rebuilt inside `sglang_megamoe`, and verified fresh import artifacts. Build log is under `hygon_tmp/supernode_debug/151_1_rpc/ll_guard_rebuild_*`.
- Remote `py_compile` passed. Remote static pytest still has two pre-existing source-contract mismatches (`setup.py` prebuilt path assertion and `_v3_ll_block_m` assertion); runtime validation continued because compiled artifacts were fresh and imported from the source tree.
- Runtime run root: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/ll_graph_flash_guard_20260707_103302`.
- Pro EP8 LL graph cap512 passed replay `8/32/64/128/256/512` against runtime-token-aligned `ll-masked`; MegaMoE graph medians `1.4357/1.9045/1.9807/2.1098/2.4462/3.9351 ms`, baseline medians `1.4405/2.1707/2.2157/2.5796/3.1846/5.7378 ms`.
- Pro EP16 LL graph cap512 passed replay `8/32/64/128/256/512` against `ll-masked`; MegaMoE graph medians `1.0103/1.1178/1.1885/1.3690/2.2363/3.8821 ms`, baseline medians `0.9977/1.1304/1.2654/1.5638/2.6578/4.7812 ms`.
- Pro uneven smoke passed: EP16 list `512,385,257,128,64,32,16,8,7,0,0,0,0,0,0,0` measured MegaMoE `1.7253 ms`, baseline `3.0445 ms`, speedup `1.765x`; EP8 list `512,257,128,64,32,7,0,0` measured MegaMoE `2.3481 ms`, baseline `4.2974 ms`, speedup `1.830x`.
- Flash guardrail passed after the unified LL skew guard. Flash EP8 LL graph cap512 replay `8/32/64/128/256/512` medians were `0.5483/0.6529/0.6983/0.7588/1.0272/1.8229 ms`; Flash EP16 LL graph medians were `0.3732/0.4078/0.4771/0.5932/0.9963/1.8817 ms`. Correctness max_abs stayed within `0.00055`.
- Pro EP8 unified-layout LL graph smoke with `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` passed replay `8/32/128/256`; MegaMoE medians `1.6754/2.3483/2.5788/2.7217 ms`. This validates the non-performance compatibility path after the 256-row skew guard.
- Final card check after runs showed no KFD PIDs and all 16 HCUs at 0% VRAM/HCU.

## 2026-07-07 13:41:27 +08:00 - Flash EP8 LL Small-Cap Baseline Retest
- User requested direct Flash EP8 LL graph comparisons for cap8/replay8 and cap32/replay32 rather than only cap512 replay buckets.
- Ran on 151.1 devices `0..7` with node-actual LL env, `--megamoe-backend ll`, `--baseline-kind ll-masked`, runtime-token-aligned baseline dispatch, `--cuda-graph-bench`, `--cuda-graph-replays 20`.
- Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/flash_ep8_ll_cap8_32_baseline_20260707_133854`.
- cap8/replay8 passed correctness (`max_abs=0.000488281`; graph bucket `max_abs=0.000244141`): MegaMoE graph `0.5446 ms`, `ll-masked` graph `0.5452 ms`; eager-timing summary `0.5491 ms` vs baseline `0.5459 ms`.
- cap32/replay32 passed correctness (`max_abs=0.000488281`; graph bucket `max_abs=0.000244141`): MegaMoE graph `0.6402 ms`, `ll-masked` graph `0.6545 ms`; eager-timing summary `0.6555 ms` vs baseline `0.6564 ms`.
- Interpretation: at exact small capture caps, Flash EP8 LL is effectively tied with `ll-masked` at token8 and slightly faster at token32. This is consistent with the prior cap512 replay8/replay32 guardrail, not a regression.
- Final card check showed no KFD PIDs and all 16 HCUs idle.

## 2026-07-07 14:15:11 +08:00 - Redundant Debug Cleanup Pass
- Reviewed the current source diff for leftover debug/probe/pure-K1/fused-C environment knobs and runnable branches. No active runtime entry remains for `MEGAMOE_DCU_PRO_LL_MASKED*`, `STAGE_STOP`, `--k1-only-*`, pure groupgemm, or fused-C Pro LL paths; the remaining matches are source-contract negative assertions in `test_dcu_megamoe_v3.py`.
- Cleaned two non-functional residues: README Pro LL example now matches the actual `fp8_w8a8_mega_moe(...)` signature, and the Pro masked ASM argument struct no longer exposes a local `debugBuffer` member name. Also clarified the LL overflow slot comment as production overflow status rather than tests/debug.
- Verified `prebuilt/` contains only the packaged `.co` code object and no temporary `.s`, logs, or object intermediates.
- Local verification passed: `python -m py_compile` for touched Python sources and `git diff --check` on the touched runtime/test/docs files. Local `pytest` is unavailable in the Windows Python environment (`No module named pytest`).

## 2026-07-07 14:47:14 +08:00 - Incremental ASM Code Object Build Flow
- User requested faster compile iteration: keep generated `.co` files after the first build, rebuild a code object only when its matching `.s` changes or the `.co` is missing, and avoid reassembling `.s` for ordinary C/C++ extension edits.
- Updated `setup.py` so staged asm code objects use the source-tree `.co` as a cache. `build_opt_asm_code_objects()` now checks `.co` mtime against `.s`, prints `Skipping up-to-date opt asm code object` for current outputs, and copies the cached `.co` into non-inplace build/wheel outputs instead of recompiling asm.
- Updated `build_dcu_megamoe.sh` so the default cleanup no longer deletes `.co`; it deletes only `.so`, `.o`, and generated `.hip` files. The first build compiles missing `.co` files normally, and later builds only rebuild stale or missing code objects.
- Changed build verification: `.so` artifacts still must be fresh from the current build, while `.co` artifacts only need to exist, be non-empty, and not be older than their `.s` source.
- Added README build notes and source-contract checks in `test_dcu_megamoe_v3.py` so the incremental `.co` behavior does not regress.
- Local verification: `python -m py_compile setup.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`, `git diff --check`, and a direct source-contract Python snippet passed. Local bash/pytest are unavailable (`bash` not found, `No module named pytest`).
- Synced the four touched files to 151.1 and verified in `sglang_megamoe`: remote `py_compile` passed, `bash -n build_dcu_megamoe.sh` passed, and a full build passed under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/incremental_co_build_20260707_143225`.
- Remote build evidence: log reported `SKIP_COUNT=12` (`6` staged asm `.co` entries skipped during source build plus the same `6` during wheel build), all seven runtime `.co` files verified, wheel was produced, source-tree `.co` mtimes before/after were unchanged, and source-tree imports resolved to the repo artifacts.
- Removed the redundant explicit `.co` clean environment switch after review; the build now has only the default incremental behavior. Re-synced the current files to 151.1 and re-ran remote `py_compile`, `bash -n`, and residue grep successfully.

## 2026-07-07 15:18:00 +08:00 - Incremental C/HIP Build Script Cleanup
- User pointed out that keeping `.co` incremental is not enough if every C/HIP iteration still deletes `build/`, source-tree `.so`, `.o`, and generated `.hip` files.
- Updated `build_dcu_megamoe.sh` so the default build preserves `build/temp`, generated `.hip`, object files, source-tree `.so`, and staged `.co` artifacts. The script now only clears transient packaging outputs (`dist`, top-level egg-info, `build/bdist`, and old wheels), then lets setuptools/ninja rebuild the changed translation units and dependent extension.
- Changed in-place `.so` verification from "fresh in this invocation" to "present and non-empty", while keeping import-path checks and `.co` freshness against `.s`. This is required for a no-op or unaffected extension to remain valid during incremental builds.
- Updated README and the source-contract test so default C/HIP incremental behavior is documented and protected against reintroducing broad delete logic.
- A first remote probe showed no-op builds drop to about `19s`, but a real K2 content change still compiled through two build directories because `bdist_wheel` rebuilt after `build_ext --inplace`. Tightened the script further to build once into `build/lib`, package the wheel with `bdist_wheel --skip-build`, and sync the resulting `.so` files back into the source tree for local imports.
- Remote validation on 151.1 after the single-build change: the first cache-fill run took `423s`; the following steady-state no-op build took `16s` with `0` ninja work lines and `6` asm `.co` skips. A temporary K2 source content change took `70s`, and restoring the file took `71s`; both had exactly `1` ninja compile work line and `6` asm `.co` skips. The temporary K2 edit was restored and the remote K2 source SHA256 matches the local file.
- A public-header probe on `include/mega_moe_dcu/layout.cuh` exposed that PyTorch's HIP ninja path did not automatically track `.cuh` dependencies for HIP objects: before the guard, only `_C` rebuilt. Added a MegaMoE header guard in `setup.py` that removes stale objects for known shared/stage headers before invoking ninja. Re-test on 151.1: temporary `layout.cuh` change and restore each removed `6` stale objects, produced `6` ninja work lines, skipped all `6` staged asm `.co` builds, and restored the header SHA256 to the local value.

## 2026-07-07 16:45:00 +08:00 - Incremental Build Matrix Verification
- User requested explicit validation of all incremental cases, not just the previously tested K2 `.cu` and public `layout.cuh`.
- Synced current `setup.py`, build script, README, and source-contract test to 151.1, then ran the matrix in `sglang_megamoe` under `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_rpc/incremental_matrix_20260707_161000`.
- Results from the matrix:
  - `.cu` (`K2_fused/k2_fused_ext.cu`): changed `71s`, restored `73s`; each had `1` ninja work line, `0` stale-header object removals, `0` asm rebuilds, `6` asm skips.
  - `csrc/apis/*.hpp` (`csrc/apis/mega_dcu.hpp`): changed/restored `45s`; each removed `2` stale objects and produced `2` ninja work lines, with `0` asm rebuilds and `6` asm skips.
  - K1 stage `.cuh` (`K1_fused/k1_v3_pack5_groupgemm_impl.cuh`): changed `140s`, restored `138s`; each removed `2` stale objects and produced `2` ninja work lines, with `0` asm rebuilds and `6` asm skips.
  - K3 stage `.cuh` (`K3_fused/k3_v3_pack5_groupgemm_impl.cuh`): changed/restored `225s`; each removed `2` stale objects and produced `2` ninja work lines, with `0` asm rebuilds and `6` asm skips.
  - ASM `.s` (`K1_fused/...PACK5.s`): changed/restored `16s`; each rebuilt `1` asm `.co`, skipped the other `5`, and produced `0` ninja work lines.
- The first matrix `.cpp` row was polluted by a verification-script restore trap that copied headers/asm after a build and made their mtimes newer than objects. After a clean no-op cache leveling run (`16s`, `0` stale objects, `0` ninja work, `0` asm rebuilds, `6` asm skips), a clean `.cpp` probe on `csrc/python_api_hip.cpp` measured changed/restored `45s`; each had `1` ninja work line, `0` stale-header object removals, `0` asm rebuilds, and `6` asm skips.
- The earlier current-guard public-header probe remains valid for public `.cuh`: temporary `include/mega_moe_dcu/layout.cuh` change/restored each removed `6` stale objects, produced `6` ninja work lines, rebuilt `0` asm `.co`, and skipped all `6` asm `.co`.
- Verified all temporary probe files on 151.1 match local SHA256 after restoration: `python_api_hip.cpp`, `k2_fused_ext.cu`, `mega_dcu.hpp`, `k1_v3_pack5_groupgemm_impl.cuh`, `k3_v3_pack5_groupgemm_impl.cuh`, `layout.cuh`, and the tested K1 `.s`. Final no-op build after all restores: `16s`, `0` stale objects, `0` ninja work lines, `0` asm rebuilds, `6` asm skips.

## 2026-07-07 21:14:04 +08:00 - Skew-Safe Capacity Fix Started
- User requested a real fix for extreme expert-skew correctness risk, not an overflow fallback.
- Added a new `Skew-Safe Capacity Fix` active section to `task_plan.md`.
- Restored 151.1 execution profile from planning memory and rechecked node state before this work:
  - `root@10.17.151.1`, docker `sglang_megamoe`, repo `/root/yuguo/DeepGEMM`, DTK `/root/yuguo/dtk-26.04.1/env.sh`;
  - 16 HCUs visible and idle, no KFD PIDs, container mount `/root/yuguo -> /root/yuguo`;
  - `/sys/class/infiniband` is empty, so 151.1 LL runs should use node-actual environment rather than stale HCA/topology settings.
- Found current 151.1 Python runtime issue: default `LD_LIBRARY_PATH` hits `/opt/hyhal/lib/libamd_smi.so` and `import torch` fails with `undefined symbol: amdsmi_init`. A temporary run-local fix works: prepend `/root/yuguo/dtk-26.04.1/.hyhal/rocm_smi/lib` and filter `/opt/hyhal/lib` plus `/opt/hyhal/lib64`; then `torch 2.10.0`, HIP `6.3.26113`, and `device_count=16` import successfully.
- Attempted `dcu-rag-kb-optimize` for Hygon/DCU routing capacity guidance, but the CLI timed out after about 30 seconds with no usable result. Proceeding from local source contracts and hardware validation instead of retrying the same query.

## 2026-07-07 21:25:37 +08:00 - Skew-Safe Capacity Implementation Pass
- Implemented the first code pass for the true skew fix:
  - `megamoe/opt.py` staged scratch sizing now uses a skew-safe normal compact tile upper bound and LL per-expert rows aligned to `num_ranks * capacity_tokens`.
  - `K1_fused/k1_fused_ext.cu` now defaults normal routing to HIP compact prebuild and uses the same skew-safe compact tile formula for eager and graph layouts.
  - `csrc/apis/mega_dcu.hpp` route-scratch sizing now mirrors the Python/C++ runtime formulas instead of the old mean+headroom normal capacity and 256-row LL skew guard.
  - Normal non-graph K2/K3 calls now receive K1 `active_tiles`; K2 also receives `row_combine_ptrs` for normal compact rows so enlarged capacity does not force full invalid-row compute.
- The implementation intentionally keeps overflow flags as diagnostics only; the fix is to make declared-capacity legal route patterns fit without relying on overflow fallback.

## 2026-07-07 21:33:23 +08:00 - Local Static Validation
- Added a test harness route pattern `--route-pattern single-local-rank` that sends each rank's unique top-k routes to the selected target rank's first local experts. This is the adversarial legal top-k case used for the worst-case capacity validation.
- Added source-contract assertions that the old `K_K1_LL_SKEW_GUARD_ROWS` / `kLlSkewGuardRows = 256` guard is gone, normal uses skew-safe compact capacity, LL uses worst-case capacity rows, normal K2/K3 receive active tiles, and the adversarial route-pattern CLI remains available.
- Added LL scratch-cap layering so a normal large-token buffer is not forced to allocate LL worst-case rows for the full normal capacity. Default LL scratch cap follows the existing auto-LL threshold (`MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD`, default 512) unless the caller explicitly sets a larger `cuda_graph_max_tokens_per_rank`; forced LL beyond that cap raises a Python configuration error.
- Local validation passed: `python -m py_compile` for touched Python files, `git diff --check`, and a custom source-contract smoke script. Local `pytest` is unavailable (`No module named pytest`), so pytest and runtime validation are being moved to 151.1.

## 2026-07-07 21:47:36 +08:00 - 151.1 Skew-Safe Validation
- Synced the changed runtime/test files to `root@10.17.151.1:/root/yuguo/DeepGEMM` and used the 151.1 LD fix (`.hyhal/rocm_smi/lib` before DTK libs, filtered `/opt/hyhal/lib*`) for Python commands.
- Remote static validation passed:
  - `python3 -m py_compile` for `megamoe/__init__.py`, `megamoe/opt.py`, and the two touched test files.
  - Source-contract smoke passed.
  - `python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` passed: `13 passed in 7.07s`.
- Remote incremental build passed in docker `sglang_megamoe` with `build_dcu_megamoe.sh`. The build rebuilt the changed C++/HIP extension objects, skipped the six up-to-date staged ASM `.co` files, synced all five `.so` files back into the source tree, verified seven code objects, and imported `megamoe`, `_C`, K1, K2, K3, and K3 V3 extensions successfully.
- 151.1 HCU state before/after runtime tests: all 16 HCUs showed `0% VRAM` and `0.0% HCU`; no residual card allocation was visible after the runs.
- Flash EP8 normal adversarial correctness passed on devices `0..7`:
  - Command shape: `--num-processes 8 --num-max-tokens-per-rank 128 --num-tokens 128 --hidden 4096 --intermediate-hidden 2048 --num-experts 256 --num-topk 6 --megamoe-backend normal --route-pattern single-local-rank --route-target-rank 0 --correctness-iters 1 --skip-bench`.
  - Result: `max_abs=6.10352e-05`, `mean_abs=6.40284e-10`, stats matched the `normal-contiguous` baseline. Output JSON: `hygon_tmp/supernode_debug/151_1_rpc/skew_safe_normal_ep8_t128.json`.
- Flash EP8 LL adversarial correctness passed on devices `0..7` after setting the known DeepEP LL heap/context env (`ROCSHMEM_HEAP_SIZE=4737418240`, `DUSHMEM_HEAP_SIZE=4737418240`, `ROCSHMEM_IPC_MNVL=1`, `ROCSHMEM_GDR_DISABLE_XDP=1`, `DEEPEP_ENABLE_LL_DISPATCH_OPT=1`, `ROCSHMEM_DISABLE_HDP_FLUSH=1`, `ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288`, `ROCSHMEM_MAX_NUM_CONTEXTS=48`, `MEGAMOE_DCU_PEER_MEMORY=rpc`):
  - Same Flash EP8 shape and route pattern with `--megamoe-backend ll --baseline-kind ll-masked`.
  - First LL attempt without the heap env failed in DeepEP baseline init before MegaMoE comparison (`ROCSHMEM_HEAP_SIZE` below `num_rdma_bytes(1090523264)`), matching known 151.1 baseline-environment behavior.
  - Retried result: `max_abs=0.000244141`, `mean_abs=2.22794e-05`, stats matched the `ll-masked` baseline. Output JSON: `hygon_tmp/supernode_debug/151_1_rpc/skew_safe_ll_ep8_t128.json`.

## 2026-07-07 22:20:50 +08:00 - LL Active-Only Work Started
- User confirmed the next implementation route: normal exact compact capacity first, then LL worst-capacity plus active-only work, and only then consider compact LL if Pro LL still regresses.
- Updated `task_plan.md` so the skew-safe capacity section is no longer marked fully complete: capacity correctness is validated, while LL active-only work is now the active subphase.
- Current source audit found two low-intrusion LL performance hazards after worst-capacity sizing:
  - K1 LL stage builder still clears `kExperts * m_per_expert` row metadata and staged scales before routing. This should be reduced to counts plus actual/padded active rows.
  - K3 LL split-tail uses `max(actual_m)` only for the tiny `<=16` case; larger `max(actual_m)` values still leave copy block scheduling tied to the full `m_per_expert` capacity.
- K2 already receives `actual_m` plus the max-count slot and maps logical rows back to physical `[expert, m_per_expert]`, so it is not the first modification target in this pass.

## 2026-07-07 22:33:00 +08:00 - LL Active-Only Local Patch
- Implemented the first LL active-only patch:
  - `K1_fused/k1_v3_pack5_groupgemm_impl.cuh` no longer clears `kExperts * m_per_expert` worth of source pointers, route weights, row pointers, row-expert metadata, and staged scales before every LL route build.
  - K1 still clears per-expert counts and output-index routes. Valid routed rows write their own metadata, and active tile padding rows write zero staged data plus a safe default scale.
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` now derives split-tail copy row blocks from the `m_indices[local_experts]` max-count slot for all buckets, not only the `<=16` case; reduce blocks are indexed after `active_copy_blocks`.
- Added source-contract assertions in `test_dcu_megamoe_v3.py` so row-capacity K1 initialization and the old K3 split-tail full-copy condition do not return silently.
- Local checks passed: `python -m py_compile` for touched Python/test files, `git diff --check`, and a direct execution of `test_v3_capacity_contract_is_skew_safe_without_overflow_fallback()` with a minimal pytest stub. Local real pytest is still unavailable (`No module named pytest`).

## 2026-07-07 22:49:00 +08:00 - LL Active-Only Performance Regression Triage
- Rechecked 151.1 before runtime testing: all 16 HCUs were visible, `hy-smi --showpids` reported no KFD PIDs, and the run-local LD fix allowed `torch 2.10.0` / HIP `6.3.26113` to see 16 devices.
- Synced/rebuilt current sources from the previous active-only patch before this triage. Flash EP8 LL adversarial correctness with `--route-pattern single-local-rank`, tokens=128, passed again against `ll-masked` (`max_abs=0.000244141` in the log), confirming the K1 active-row initialization change did not reintroduce clipping.
- Flash EP8 LL graph cap512 correctness passed, but graph replay medians regressed to `1.0037/1.1033/1.1262/1.1996/1.4770/2.2573 ms` for replay `8/32/64/128/256/512`, versus the historical post-guard band around `0.5483/0.6529/0.6983/0.7588/1.0272/1.8229 ms`.
- Triage result: this is too large to accept. K3 split-tail host launch still used capacity-sized `copy_blocks` (`rows_per_expert` worst stride) and only early-returned inside the kernel; K1 unified LL uses a fixed 64-CTA persistent grid and actual per-expert counts, so it is not the first launch-inflation target.
- Implemented the next local patch: K3 LL split-tail now launches a bounded copy CTA pool (`256` rows/expert for Flash, `128` for Pro) and grid-stride loops over `active_copy_blocks`, preserving worst-skew correctness while avoiding full-capacity empty CTA launches for random graph buckets.
- Local checks after the K3 CTA-pool patch passed: `python -m py_compile`, `git diff --check`, and the direct capacity-contract source test.

## 2026-07-07 23:22:00 +08:00 - K3 CTA Pool Retest And Pro Compact Trigger
- Synced the K3 CTA-pool patch and source-contract test to 151.1, ran remote pytest (`13 passed`) and rebuilt in `sglang_megamoe`; incremental build completed successfully and imported all MegaMoE extensions from the source tree.
- Flash EP8 LL graph cap512 after the K3 pool patch passed correctness and returned to the historical performance band. Replay medians for `8/32/64/128/256/512` were `0.558279/0.668380/0.702880/0.779200/1.046179/1.841059 ms`, versus the earlier bad post-worst run `1.0037/1.1033/1.1262/1.1996/1.4770/2.2573 ms`.
- Flash EP8 LL adversarial skew correctness was rerun after the K3 pool patch and still passed against `ll-masked` (`max_abs=0.000244141`, `mean_abs=2.22794e-05`). This verifies the copy CTA pool does not merely skip skew rows; it grid-stride processes every active copy block.
- Pro EP8 LL graph cap512 with the default split path required larger DeepEP baseline heap (`ROCSHMEM_HEAP_SIZE=DUSHMEM_HEAP_SIZE=12884901888`) because the baseline reported `num_rdma_bytes(11450456192)`.
- Pro EP8 LL graph cap512 with the default split path passed correctness but regressed materially at small/mid replay buckets: MegaMoE `2.158579/2.624898/2.694338/2.854138/3.173478/4.651879 ms`, baseline `1.442240/1.954239/2.123119/2.475079/3.190199/5.727577 ms`. Historical pre-exact MegaMoE was `1.4357/1.9045/1.9807/2.1098/2.4462/3.9351 ms`.
- Pro EP8 unified-layout LL graph cap512 with the same 12 GiB heap was also correct but slower (`2.384559/3.091958/3.165439/3.264698/3.446359/6.061297 ms`), so it is not a performance replacement for the default split path.
- Root-cause direction: the remaining Pro regression is the masked K1 launch. Exact LL capacity makes `rows_per_expert = num_ranks * graph_cap = 4096`, and `k1_ll_masked_groupgemm_pack5` passes that as `size_m`, which sets `num_MBlocks=64` and the physical B/C expert stride. The pre-exact path used a small `size_m` such as `128`, so small replay buckets now pay many empty M-blocks.
- Next implementation target is compact active-K1 for Pro LL split. The fix should preserve worst-capacity correctness without relying on overflow fallback, while making the masked K1/K2/K3 chain operate on active compact rows for ordinary random buckets.

## 2026-07-07 23:42:00 +08:00 - Pro Compact-Head Result And Default Guard
- Implemented the first Pro compact-head experiment: compact the active head rows into a small masked-K1 layout, copy compact head output back to the original worst-capacity layout, and run the offset masked-K1 tail path for rows beyond the compact head. This is an exact head/tail algorithm rather than an overflow fallback.
- 151.1 Pro EP8 LL token128 smoke passed against `ll-masked` after the compact-head implementation (`max_abs=0.000976562`, `mean_abs=4.92294e-05`).
- 151.1 Pro EP8 LL graph cap512 also passed correctness, but performance was worse than the previous exact split run: MegaMoE replay `8/32/64/128/256/512` measured `2.2771/2.7466/2.8233/2.9797/3.2819/4.7923 ms`, while the exact split run before compact-head was `2.1586/2.6249/2.6943/2.8541/3.1735/4.6519 ms`.
- This regression is too large to accept. The likely costs are compact staged-input copy, compact L1-output copy-back, and the offset tail masked-ASM launch even when ordinary random buckets have no tail rows.
- Added a local default guard: `MEGAMOE_DCU_PRO_LL_COMPACT_HEAD=1` is now required to allocate/use the compact-head workspace. With the env unset, the default path returns to the known exact split behavior while the slower experiment remains reproducible for ablation.
- Local checks after the default guard passed: `python -m py_compile`, `git diff --check`, and the direct skew-capacity source-contract function using a local pytest stub. Full pytest and runtime retest are next on 151.1 after checking HCU state.

## 2026-07-08 00:19:09 +08:00 - Graph-Cap Fix And Normal Regression Triage
- Fixed Pro LL graph capture policy so the default captures MegaMoE per replay bucket instead of reusing one max-capacity graph for every bucket. The old max-cap behavior remains available through `--cuda-graph-single-capture`.
- Remote static validation on 151.1 passed after syncing the graph-cap patch: `py_compile` passed and `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` reported `13 passed`.
- Pro EP8 LL graph cap512 with per-bucket default passed correctness and recovered the small/mid bucket performance band. MegaMoE replay medians for `8/32/64/128/256/512` were `1.4381/1.9296/2.0488/2.2953/2.7740/4.6565 ms`; the aligned `ll-masked` baseline was `1.4446/1.9547/2.1209/2.4758/3.1942/5.7281 ms`.
- Rechecked Flash EP8 normal eager 4096 after the exact compact-capacity change and found a serious performance regression despite correctness passing. Before any active-launch follow-up, MegaMoE measured `17.2707 ms` versus `normal-contiguous` baseline `9.8965 ms`; the historical same-node MegaMoE reference was `5.7636 ms`.
- Added a first Normal eager K1 active-launch patch: for non-captured compact K1, the host reads the single `active_tiles` count and launches only `active_tiles * 256` K1 rows while keeping the worst-case workspace capacity. Remote source pytest and rebuild passed.
- K1 active launch alone is not enough. Flash EP8 normal eager 4096 after that patch passed correctness but still measured `16.1966 ms` versus baseline `9.8811 ms`, far from the historical `5.7636 ms`.
- Source triage now points to Normal K3 as the next likely launch-inflation site: `K3_fused/k3_fused_ext.cu` still sets `prob.n` and `wg_n` from `total_rows`, and `opt.py` still passes `asm_done_target=((rows+255)//256)*((hidden+255)//256)` from capacity rows. The ASM receives `active_tiles`, but host launch and tail-reduce done target still scale with exact capacity.
- Do not treat Normal exact capacity as performance-complete until K3 active launch/done-target handling is fixed and Flash EP8 normal eager 4096 is remeasured on clean cards.

## 2026-07-08 01:09:00 +08:00 - Normal Exact Performance Recovery
- Added the Normal eager follow-up patches after the K1-only result:
  - K3 normal eager now reads `active_tiles` on the host outside graph capture, launches only active K3 row tiles, and sets the tail-reduce done target to the actually launched GEMM workgroups.
  - Normal eager K2 now uses the existing active-row CTA pool (`K_K2_GRAPH_ROW_BLOCKS`) when compact `active_tiles` metadata is available, instead of launching one CTA per worst-capacity row block.
- Remote source-contract pytest passed after syncing these patches to 151.1 (`13 passed`), and the 151.1 incremental rebuild completed with fresh K1/K3 runtime artifacts.
- Flash EP8 normal eager 4096 progression on 151.1:
  - exact compact capacity before active follow-up: `17.2707 ms` fused vs `9.8965 ms` baseline, correctness passed;
  - K1 active launch only: `16.1966 ms` fused vs `9.8811 ms` baseline, correctness passed;
  - K3 active launch/done-target after K1: `6.6010 ms` fused vs `9.6815 ms` baseline, correctness passed;
  - K2 active CTA pool after K1/K3: `5.9675 ms` fused vs `9.7207 ms` baseline, correctness passed;
  - final threshold-reverted recheck: `5.9164 ms` fused vs `9.7064 ms` baseline, correctness passed with `max_abs=0.000671387`, speedup `1.6406x`.
- Historical same-node Flash EP8 normal eager 4096 reference before exact capacity was `5.7636 ms`; the final exact-capacity result is about `+2.7%`, which is within the current acceptable guardrail and no longer a large regression.
- Small-bucket evidence after the active-launch work: 512 recheck after reverting the threshold experiment measured `2.0019 ms`; the same active K1/K3/K2 code before the rejected threshold experiment measured `1024/2048 = 2.1515/3.5004 ms`, all with correctness passing.
- Rejected the host-readback threshold experiment (`active launch only when capacity_tiles >= 512`). It worsened small/mid buckets (`512 -> 3.327 ms`, `1024 -> 5.086 ms`) and the 2048 matrix run ended with an empty log while the container exited `255`; after confirming no KFD PIDs, the existing `sglang_megamoe` container was restarted and the threshold constants were removed.
- Before the final 4096 run, local and remote SHA256 matched for `megamoe/opt.py`, K1/K3 normal extension sources, and the two test files. After the run, `hy-smi --showpids` again reported no KFD PIDs.
- Fresh remote source-contract pytest after the final runtime check passed on 151.1: `13 passed in 7.28s`.

## 2026-07-08 03:10:00 +08:00 - Pro LL Compact-Active And Skew Triage
- Implemented an eager-only Pro LL compact-active path behind the default-on `MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE` gate. The route/stage builder still keeps exact worst-capacity storage, but ordinary eager buckets compact active rows into a smaller `[expert, compact_rows]` layout before bundled masked K1/K2/K3.
- Local validation passed after the compact-active patch: `python -m py_compile`, `git diff --check`, and the source-capacity contract smoke.
- Synced current local sources to 151.1, ran remote source-contract pytest (`13 passed`), and rebuilt in `sglang_megamoe` under `hygon_tmp/supernode_debug/151_1_rpc/compact_active_build_20260708_0215`; the build synced fresh `.so` artifacts and imports resolved from the source tree.
- Pro EP8 LL eager default compact-active passed random-token correctness and improved the exact worst-capacity eager path:
  - token256: `2.6022 ms` fused vs `3.1798 ms` baseline, compared with the prior exact-worst `2.7449 ms`;
  - token512: `4.0952 ms` fused vs `5.6718 ms` baseline, compared with the prior exact-worst `4.6322 ms`.
- Pro EP8 LL graph cap512 remains intentionally on the exact per-bucket path (`allow_compact_active=False` during capture). It passed correctness with replay medians `1.4345/1.9274/2.0519/2.2848/2.7786/4.6458 ms`, in the same band as the earlier per-bucket graph run.
- New correctness blocker found: Pro EP8 LL adversarial `single-local-rank` skew fails against `ll-masked` even with compact-active disabled. token128 reports `max_abs ~= 0.107`, so this is not a compact-active-only bug.
- Ablations collected so far:
  - `MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE=0` still fails token128 with the same error scale.
  - `MEGAMOE_DCU_LL_K3_SPLIT_TAIL=0` still fails token128, so split-tail copy is not the first boundary.
  - `K3_USE_ASM_TAIL_REDUCE=0` still fails token64, so the final tail reducer is not the root cause.
  - Pro unified LL (`MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1`) also fails token64, pointing to common LL skew handling rather than only the Pro split masked-K1 wrapper.
- Threshold sweeps show the failure is row-count sensitive and not monotonic across scratch layouts: compact-active token8/HOT_M64 passes, token16/HOT_M128 fails, token32/HOT_M256 passes, token64/HOT_M512 fails; exact active-off token16 passes but token32/token64 fail. Next work is stage-boundary tracing on the first failing route, not another blind performance patch.

## 2026-07-08 09:55:18 +08:00 - Pro LL Skew K1 Mapping Triage
- Answered two graph/eager questions during the run:
  - Normal graph does not use the eager D2H `active_tiles_host` readback. Eager can shrink host launch after synchronizing the metadata kernel; graph capture/replay must keep fixed host launch dimensions and rely on device-side runtime tokens / active-tile early exit for correctness.
  - Capacity correctness for graph is still protected by the same worst-capacity rows/tiles. The remaining question for Normal graph is performance under a measured graph command, not overflow/cropping correctness.
- Rechecked 151.1 before the Pro tests; `hy-smi --showpids` was clean before launching. Rebuilt after restoring the Pro masked-K1 wrapper launch.
- A first attempted Pro masked-K1 wrapper change to launch `num_NBlocks x experts*num_MBlocks` was wrong. Disassembly of the active default masked `.co` (`73184662...`) showed it derives `expert = workgroup_x / num_NBlocks`, `n_tile = workgroup_x % num_NBlocks`, and uses persistent `workgroup_x += 128`; setting x only to `num_NBlocks` forces expert0 only. That change was reverted locally and remotely.
- Disassembled the scratch balanced `.co` (`0c721483...`) and found a different active-tile scheduler: it linearly maps workgroup_x over actual `masked_m` tiles. It writes all six skewed slots in the trace, but its numeric output is not compatible with the current packaged masked weight storage, so it cannot be used as a drop-in replacement.
- With the default packaged `.co` restored and persistent launch restored, token64 `single-local-rank` stage tracing shows rank0 writes all six route slots for token44:
  - `rows_per_expert=512`, counts `[512,512,512,512,512,512,0,...]`.
  - slot rows `350/862/1376/1888/2428/2940` all have nonzero L1, K3, and combine-slot abs values.
  - Correctness is still not acceptable: `max_abs=0.0980835`, `mean_abs=0.00504374` against `ll-masked`.
- Current boundary has therefore moved from "contribution dropped" to "Pro LL skew numeric/layout mismatch versus ll-masked". Next evidence needed when 151.1 is responsive again:
  - Compare default Pro split against `normal-contiguous` baseline for the same route pattern to check whether the `ll-masked` oracle path has a skew-specific ordering/layout difference.
  - Add a narrow K1 oracle outside production source, or in the test harness only, to compare the exact staged rows against `deepgemm.m_grouped_fp8_gemm_nt_masked` without introducing the banned production `deepgemm` call.
  - After resolving correctness, run Normal graph perf (`normal`, cap/replay 4096 or the historical graph bucket) because only Normal eager 4096 has been performance-validated after exact capacity.
- A follow-up failure-detail run timed out and 151.1 SSH stopped responding immediately afterward. Do not launch additional GPU tests until SSH and `hy-smi --showpids` confirm a clean node.

## 2026-07-08 11:35:08 +08:00 - 151.1 Recovery And Normal Graph Measurement
- Rechecked 151.1 after the container exit. Host `hy-smi` listed all 16 HCUs normally with VRAM/HCU usage at 0%, and `hy-smi --showpids` reported no KFD PIDs.
- Restarted the existing `sglang_megamoe` container only. Container device nodes `/dev/kfd`, `/dev/mkfd`, and `/dev/dri/renderD128..143` were present, `hy-smi --showpids` was clean, a minimal Torch HIP kernel passed on `HIP_VISIBLE_DEVICES=0..7`, and `deepgemm` import again reported `DEEPGEMM_GPU_CUS=64` with `gfx938` ASM.
- Ran the pending Flash EP8 Normal graph check after exact compact capacity:
  - Command shape: `num_processes=8`, `tokens=4096`, `hidden=4096`, `intermediate=2048`, `experts=256`, `topk=6`, `megamoe_backend=normal`, `baseline_kind=normal-contiguous`, `cuda_graph_test_tokens=4096`, `cuda_graph_replays=20`.
  - Correctness passed: eager check `max_abs=0.000671387`, graph bucket `max_abs=0.000488281`, `mean_abs=9.37182e-06`.
  - Graph replay-only timing was `median=6.9654 ms`, `min=6.8656 ms`.
- Interpretation: Normal graph is correctness-safe after the exact-capacity fix, but performance is not yet recovered to the Normal eager band. The current Normal eager reference is `5.9164 ms`, so graph replay is about `+17.7%` slower even though it excludes input updates.
- Root cause direction is consistent with the implementation contract: graph capture cannot use the eager `active_tiles_host` D2H shrink, so it still launches a worst-capacity compact grid and relies on device-side active-tile/runtime early exits. This is correct but leaves too many empty CTAs in graph replay.
- Next Normal performance fix should be capture-compatible active work: a fixed CTA pool or device-side active-tile consumer that decouples graph launch dimensions from worst capacity. Do not add D2H/synchronize inside graph capture.

## 2026-07-08 12:53:52 +08:00 - Pro LL Skew K1 Stage Visibility Work
- Current LL status: Flash LL worst-capacity plus active-only is correct and back in the expected performance band; Pro LL random graph/eager is correct and performance-acceptable. The active blocker is Pro EP8 LL adversarial `single-local-rank` skew.
- Latest traces moved the boundary back into K1 stage construction. For traced failing tokens, `row_combine_ptrs` and route slot mapping are valid, but some active route rows have `staged_x=0`, default `staged_x_scale=1.0e-4/448`, zero L1, zero K3, and zero combine contribution.
- Negative/partial fixes recorded before this entry:
  - K3 split-tail `glc` load/store and system-scope signal did not fix the skew failure.
  - Adding a system-scope K1 grid barrier after max-count publication helped one traced token but did not eliminate failures.
  - Switching per-expert route counts to `atomicAdd_system` plus system-scope acquire count loads recovered more rows, but another token still reproduced the same default-stage-row symptom.
- Next patch is targeted at K1 per-row source metadata visibility: store `symm_src_x_ptrs[row]` with system-scope release and load it in the stage-copy loops with system-scope acquire. This is a correctness synchronization fix, not an overflow fallback or capacity-policy change.

## 2026-07-08 13:25:00 +08:00 - Pro LL Skew Root Cause Fixed
- The source-pointer release/acquire experiment compiled but did not fix Pro EP8 LL skew by itself. token64 still failed with only slot0 contributing. Tracing then showed token28 had all slots nonzero, while token18 still had slot1..5 zero/default staged rows.
- Root cause found: Pro LL stage-only uses `kBlockM=48`, while exact worst rows can be `m_per_expert=512`. K1 stage copy rounded `expert_count=512` up to `stage_rows=528`; for expert0, padding rows 512..527 are physically expert1 rows 0..15, so padding zero writes clobbered real rows in the next expert. This exactly matched failing rows such as expert1 row9 and expert4 row2.
- Final retained patch clamps stage-copy padding to `m_per_expert` with `v3_k1_stage_rows_for_count_device()`. The heavier diagnostic changes were removed again: K1 source-pointer release/acquire, system-scope count/barrier experiments, and K3 glc split-tail load/store.
- 151.1 final-light rebuild passed and imports resolved from the source tree for K1/K3. Correctness guardrails after rebuild:
  - Pro EP8 LL skew token64 with `MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE=0`: passed, `max_abs=0.000488281`, `mean_abs=4.77837e-05`.
  - Pro EP8 LL skew token128 default compact-active: passed, `max_abs=0.000488281`, `mean_abs=4.77848e-05`.
  - Flash EP8 LL skew token128: passed, `max_abs=0.000244141`, `mean_abs=2.22794e-05`.
- Next step is performance validation on random Pro LL eager/graph plus Flash LL graph guardrails. The expected performance impact of the retained root fix should be negligible because it reduces out-of-stride padding work rather than adding work.

## 2026-07-08 13:45:00 +08:00 - LL Final Performance Guardrails Passed
- Pro EP8 LL eager random with default compact-active remains in the expected band after the final-light stage-clamp patch:
  - token256: fused `2.5905 ms`, baseline `3.1838 ms`, correctness `max_abs=0.000976562`.
  - token512: fused `4.1065 ms`, baseline `5.6630 ms`, correctness `max_abs=0.000976562`.
  - These match the earlier compact-active references (`2.6022/4.0952 ms`) within noise.
- Pro EP8 LL graph cap512 per-bucket remains in the recovered band. Replay medians for tokens `8/32/64/128/256/512` are `1.4481/1.9328/2.0492/2.2838/2.7810/4.6494 ms`; correctness passed every bucket.
- Flash EP8 LL graph cap512 guardrail passed. Replay medians for tokens `8/32/64/128/256/512` are `0.4989/0.6451/0.6853/0.7688/1.0380/1.8168 ms`; correctness passed every bucket. This is at least as good as the prior K3-pool guardrail band.
- Remote source-contract pytest passed after the final rebuild: `13 passed in 7.55s`. Final card check reported no KFD PIDs.
- Next cleanup: remove production-only Pro skew trace hooks from `opt.py`; keep the test-only route pattern because it is now the regression trigger for this precision/correctness class.

## 2026-07-08 13:55:00 +08:00 - LL Cleanup Verification
- Removed the production `MEGAMOE_DCU_TRACE_PRO_LL_SKEW` trace helper and call from `megamoe/opt.py`. The test-only `--route-pattern single-local-rank` support remains because it is the regression harness for the exact skew-capacity correctness class.
- Local checks after cleanup passed: `python -m py_compile` for `megamoe/opt.py` and the two touched test files, plus `git diff --check`.
- Synced the cleaned `opt.py` to 151.1. Remote verification passed: Python compile, `PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` (`13 passed in 7.24s`), and `hy-smi --showpids` reported no KFD PIDs.

## 2026-07-08 14:37:00 +08:00 - Temporary Debug And Ablation Cleanup
- Removed the remaining test-harness failure dump path: `--trace-failure-detail`, `TRACE_FAILURE`, the pre-baseline combine clone, combine slot value dumps, and top-column dumps are gone from `test_mega_moe_dcu.py`.
- Removed the rejected Pro LL compact-head ablation from production sources. `MEGAMOE_DCU_PRO_LL_COMPACT_HEAD`, compact-head staging/copy-back wrappers, and offset masked-K1 pybind exports are no longer present. The retained default Pro LL performance path is compact-active only.
- Simplified compact-active bookkeeping after the compact-head removal: the extra tail count array was deleted, `pro_compact_head_m/pro_compact_tail_m` became a single `pro_compact_m`, and route_scratch sizing now reserves one compact count array instead of two.
- Added/updated source-contract guards so retired failure dumps and compact-head entry points cannot silently re-enter: `--trace-failure-detail`, `TRACE_FAILURE`, `MEGAMOE_DCU_PRO_LL_COMPACT_HEAD`, `k1_ll_masked_prepare_compact_head`, `k1_ll_masked_copy_compact_head`, and `k1_ll_masked_groupgemm_pack5_offset` are asserted absent.
- Local checks after the cleanup passed: `python -m py_compile` for the touched Python files and `git diff --check`. Local `pytest` is unavailable on the Windows Python (`No module named pytest`), so contract pytest was run on 151.1 instead.
- 151.1 state and verification:
  - `hy-smi --showpids` was clean before and after GPU work.
  - Remote source pytest passed after fixing the container library path by prepending `/usr/local/lib/python3.10/dist-packages/amdsmi` to `LD_LIBRARY_PATH`: `13 passed in 7.42s`.
  - Full remote build passed and synced fresh source-tree `.so` artifacts; build log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_cleanup/cleanup_build_20260708_142145/build.log`.
  - Pro EP8 LL adversarial `single-local-rank` token128 passed against `ll-masked`: `max_abs=0.000488281`, `mean_abs=4.77848e-05`.
  - Pro EP8 LL random token256 performance remained in the pre-cleanup band: fused `2.5879 ms`, baseline `3.1855 ms`, correctness `max_abs=0.000976562`.

## 2026-07-08 15:04:00 +08:00 - EP16 Final LL Retest After Cleanup
- Rechecked 151.1 before the EP16 runs. Container `torch` import passed after the `amdsmi` library-path prepend, and `hy-smi --showpids` reported no KFD PIDs.
- Pro EP16 LL adversarial skew is clean after the final stage-copy clamp and cleanup:
  - token64 `single-local-rank`, target rank0, Pro shape `experts=384 topk=6 hidden=7168 intermediate=3072`, `ll-masked` baseline: passed with `max_abs=0.000488281`, `mean_abs=4.78497e-05`. Log/result dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_ep16_final/ep16_ll_skew_t64_20260708_145950`.
  - token128 same setup: passed with `max_abs=0.000488281`, `mean_abs=4.79524e-05`. Log/result dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_ep16_final/ep16_ll_skew_t128_20260708_150109`.
- Pro EP16 LL graph cap512 random route passed against `ll-masked` for all replay buckets. Eager correctness max was `0.000976562`; graph bucket maxes were `0.000488281/0.000488281/0.000488281/0.000976562/0.000976562/0.000976562` for tokens `8/32/64/128/256/512`.
- EP16 graph replay medians stayed better than the aligned baseline: MegaMoE `1.0938/1.2286/1.3612/1.6362/2.7352/4.8378 ms` versus baseline `1.1465/1.2819/1.4086/1.7139/2.8351/5.0976 ms` for tokens `8/32/64/128/256/512`. Full run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_ep16_final/ep16_ll_graph512_20260708_150225`.
- The same graph command also reported token512 eager-main performance `4.2887 ms` fused versus `5.2144 ms` baseline, speedup `1.216x`, with correctness passing.
- Post-run card checks again reported no KFD PIDs.
- Source scan of runnable code confirmed the then-retained Pro LL addition was `MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE`, default enabled, while `MEGAMOE_DCU_PRO_LL_COMPACT_HEAD` had no source/API residue. This was later simplified further by removing the compact-active env entirely and making compact-active the fixed eager Pro LL path.

## 2026-07-08 15:20:00 +08:00 - Remove Pro LL Compact-Active Env
- Removed `MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE` from the runnable Python/C++ implementation. Pro LL eager compact-active is now fixed-on when the Pro masked shape has compact scratch available; graph remains on the exact per-bucket non-compact-active path via `allow_compact_active=False`.
- Deleted the Python `pro_ll_compact_active_enabled()` helper and the C++ API-side env parser. Route-scratch sizing now allocates Pro compact scratch solely from the Pro masked shape predicate.
- Updated source-contract tests to assert the compact-active env/helper stay absent while the compact-active kernels and wrappers remain present.
- Local verification passed: `python -m py_compile megamoe/opt.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`.

## 2026-07-08 15:32:00 +08:00 - EP16 Single-Capture Graph Reality Check
- Rechecked 151.1 before the run; `torch` import passed and `hy-smi --showpids` was clean.
- Ran Pro EP16 LL graph cap512 with `--cuda-graph-single-capture`, so one capture capacity `512` graph replayed runtime tokens `8/32/64/128/256/512`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_ep16_final/ep16_ll_graph512_single_20260708_153055`.
- Correctness passed for every bucket. Bucket max errors were `0.000488281/0.000488281/0.000488281/0.000976562/0.000976562/0.000976562`.
- Single-capture graph replay medians were `1.8298/1.9214/2.0112/2.1938/3.1099/4.8275 ms` for tokens `8/32/64/128/256/512`. The aligned `ll-masked` baseline medians in the same run were `1.1478/1.2832/1.4014/1.7135/2.8365/5.1106 ms`.
- Interpretation: single cap512 graph is correctness-safe, but it does not match the pre-exact-capacity small/medium bucket performance. The per-bucket capture mode remains the current fair graph performance mode; true single-capture recovery would need a graph-compatible active work consumer or compact graph design.

## 2026-07-08 15:45:00 +08:00 - Baseline Cap512 Single-Capture Attempt Hung Node
- User correctly pointed out that a fair single-capture comparison should make the `ll-masked` baseline use the same cap512 dispatch capacity. The harness was adjusted locally so `--cuda-graph-single-capture` passes `graph_capture_tokens` to the baseline graph path; source-contract coverage was updated and local `py_compile` plus `git diff --check` passed.
- Synced the updated Python harness to 151.1 and rechecked cards; `hy-smi --showpids` was clean before launch.
- Attempted the same Pro EP16 LL `--cuda-graph-single-capture` run with baseline dispatch cap now aligned to 512. The run did not return; after the user interrupted the local command, 151.1 was no longer reachable by SSH or ping from the workstation. Two follow-up probes showed `ssh: connect to host 10.17.151.1 port 22: Connection timed out`, `PingSucceeded=False`, and 100% packet loss.
- No result numbers should be inferred from this aborted run. Once 151.1 recovers, first action must be host/container/KFD state inspection before any new GPU tests. Treat baseline-cap512 single-capture as unstable until proven otherwise.

## 2026-07-08 16:05:00 +08:00 - README Capacity Token Contract Update
- Updated `README.md` near the eager uneven-token `capacity_num_tokens` section to document graph-mode behavior explicitly.
- Documented that `capacity_num_tokens` is accepted by both LL and normal CUDA Graph captures. In graph mode it is the capture capacity, independent of replay-time `sym_buffer.cuda_graph_num_tokens`; when omitted it defaults to `sym_buffer.cuda_graph_max_tokens_per_rank`.
- Updated the CUDA Graph mode section and test-harness option list:
  - `--num-max-tokens-per-rank` is now described as the graph capacity upper bound, not necessarily every capture's size.
  - Default graph testing is documented as per-bucket capture, passing each listed replay token count as graph `capacity_num_tokens`.
  - `--cuda-graph-single-capture` is documented as one max-capacity capture for all replay buckets, with MegaMoE and `ll-masked` baseline capacities both using the max bucket.

## 2026-07-08 16:15:00 +08:00 - 151.1 Still Unreachable
- Rechecked 151.1 after the baseline-cap512 single-capture hang. The node is still unreachable from the workstation: ping timed out with 100% packet loss, TCP port 22 failed, and `ssh -F NUL -o ConnectTimeout=10 root@10.17.151.1` timed out.
- Because host SSH is down, no container `hy-smi --showpids` or KFD PID inspection is currently possible. First action after network/host recovery remains host login, Docker/container status, and DCU PID/state inspection before any further GPU test.

## 2026-07-08 16:13:49 +08:00 - Graph Performance Recheck After Recovery
- 151.1 recovered. Host SSH works, `sglang_megamoe` had exited with code 255 and was restarted, and host/container `hy-smi --showpids` reported no KFD PIDs before and after the rechecks.
- Synced the current local runnable sources to `/root/yuguo/DeepGEMM`, ran local `py_compile` plus `git diff --check`, remote source-contract pytest (`13 passed`), and rebuilt the DCU MegaMoE artifacts. Build/import verification passed; build log: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/graph_perf_build_20260708_155729/build.log`.
- Flash EP8 Normal graph token4096 reproduced the known graph/eager gap after the rebuild. Correctness passed against `normal-contiguous`; eager main-call median was `6.0118 ms`, while graph replay-only median was `6.9821 ms`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/normal_graph4096_20260708_160508`.
- Normal graph with `K3_USE_ASM_TAIL_REDUCE=0` stayed correct but was slower: graph replay token4096 `7.0714 ms`, eager main-call `6.0637 ms`. This rejects "disable normal ASM tail-reduce" as the graph recovery fix for this size. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/normal_graph4096_tail0_20260708_160839`.
- Pro EP8 LL graph cap512 per-bucket also reproduced the post-fix graph/eager gap, but remains faster than the fair `ll-masked` graph baseline. Correctness passed for all replay buckets. Eager token512 was `4.1925 ms`; graph replay medians for `8/32/64/128/256/512` were `1.4391/1.9300/2.0497/2.2888/2.7879/4.6485 ms`, while `ll-masked` baseline graph medians were `1.4463/2.1219/2.2071/2.5399/3.1942/5.7411 ms`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/pro_ep8_ll_graph512_20260708_160649`.
- Pro EP8 LL graph with `MEGAMOE_DCU_LL_K3_SPLIT_TAIL=0` was slower: graph replay512 `5.2302 ms`, eager main-call `4.8066 ms`. This keeps split-tail enabled as the correct LL graph direction. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/pro_ep8_ll_graph512_split0_20260708_161028`.
- Source inspection of the remaining Pro LL graph gap found a precise unsafe shortcut: `launch_pro_ll_masked_groupgemm_asm()` currently derives `num_MBlocks` from physical `size_m`. In exact graph cap512, Pro EP8 has `size_m=4096` and `num_MBlocks=64`; eager compact-active uses a smaller compact stride such as `128` rows/expert and `num_MBlocks=2`. Changing graph to use `expected_m_per_group` directly would be faster for random routes but is not correctness-safe under legal skew, because masked K1 would skip rows above the chosen M-block count while K2/K3 still consume `actual_m`.
- Current safe conclusion: no low-risk one-line performance patch should be applied for LL graph. Real graph recovery needs either masked K1 ASM support for device-side max-count scheduling, or true compact LL where K1/K2/K3 consume a compact active layout without host D2H and without dropping skew rows.

## 2026-07-08 16:29:00 +08:00 - Graph Gap Profiling
- Checked 151.1 before profiling: `sglang_megamoe` was up, Torch saw 16 devices, and `hy-smi --showpids` reported no KFD PIDs. Post-profile card check was also clean.
- Ran short `hipprof --stats --hip-trace --follow-fork --devices 0` probes with graph baseline skipped, so the stats focus on MegaMoE graph capture/replay kernels rather than DeepEP baseline work.
- Normal Flash EP8 graph token4096 profile run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/normal_graph4096_hipprof_20260708_162238`.
  - Main MegaMoE kernels in HIPOPS stats were K1 normal ASM at about `3.745 ms/call` and K3 normal ASM at about `2.798 ms/call`.
  - Compact route init/count/build/emit kernels were only tens of microseconds each. The remaining Normal graph gap is therefore dominated by K1/K3 captured capacity-grid work/early-exit overhead, not by the compact route builder.
- Pro EP8 LL graph token512 profile run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graph_perf/pro_ep8_ll_graph512_hipprof_20260708_162358`.
  - MegaMoE-side heavy kernels were K1 stage-only (`~3.701 ms/call`), K3 split/combine (`~2.699 ms/call`), Pro masked K1 (`~1.731 ms/call`), K3 LL masked GEMM (`~1.042 ms/call`), and K2 SwiGLU/quant (`~0.800 ms/call`) in the captured profile window.
  - This matches the source analysis: the graph/eager delta is most plausibly from Pro masked K1 still using the exact physical stride, while K2/K3 already consume actual rows / split-tail active copy blocks.
- No performance patch was applied from these profiles. The safe optimization paths remain larger changes: Normal K1/K3 graph active-work CTA pooling, or Pro LL masked K1 dynamic M-block scheduling / true compact LL graph. Fixed expected-M clamping is explicitly rejected because it would reintroduce legal-skew row loss.

## 2026-07-08 16:42:00 +08:00 - Baseline Cap512 Single-Capture Retest
- Rechecked 151.1 before rerunning the previously failed baseline-cap512 test. `sglang_megamoe` was up, Torch saw 16 devices, and `hy-smi --showpids` reported no KFD PIDs.
- Verified the remote test harness contains the intended fair single-capture baseline logic: under `--cuda-graph-single-capture`, `ll-masked` baseline graph capture uses `graph_capture_tokens` rather than the replay token.
- First ran a short EP16 token512 sanity with baseline cap512. It passed correctness and benchmarked MegaMoE graph replay `4.8772 ms` versus `ll-masked` baseline graph replay `5.1075 ms`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_baseline_cap512/ep16_ll_single_cap512_t512_20260708_163225`.
- Then ran the full EP16 single-capture cap512 bucket list with `tokens=8,32,64,128,256,512`, `num_processes=16`, Pro shape, `--megamoe-backend ll`, `--baseline-kind ll-masked`, `--cuda-graph-single-capture`, baseline cap fixed to 512, `repeat=10`. Python reported `STATUS=0`; the outer SSH wrapper ended with a harmless script-tail parse error after printing the complete result.
- Correctness passed for every replay bucket. MegaMoE graph medians were `1.8216/1.9240/2.0016/2.2017/3.1074/4.8633 ms` for `8/32/64/128/256/512`.
- The fair baseline-cap512 graph medians were `2.5676/2.6352/2.6780/2.8035/3.5800/5.1097 ms` for `8/32/64/128/256/512`.
- Full run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_baseline_cap512/ep16_ll_single_cap512_full_20260708_163424`.
- Post-run `hy-smi --showpids` was clean. This retest did not reproduce the previous node hang.

## 2026-07-08 17:10:00 +08:00 - Graph Optimization Direction Correction
- Committed and pushed the stable skew-safe capacity work before starting larger graph optimization exploration: `a9b8598 Fix MegaMoE skew-safe capacity` on `origin/supernode`.
- Briefly inspected a historical scratch masked K1 `.s` from 151.1, then discarded that direction after the user pointed out it was previously known to use a different weight layout. No production source change from that `.s` is retained.
- Verified the packaged Pro masked K1 `.co` hash is the current production object on both local and 151.1: `73184662ec644cf9f4e9cfacec720a15428e84c5f84ad06e6e9e57bfa06543b4`.
- Disassembly of the current packaged `.co` confirms it already uses device-side `masked_m` to bound M-block work. The kernel loads the `masked_m` pointer from the existing kernarg block, compares `m_block * 64 >= masked_m[expert]`, branches to `BlockEnd`, then continues persistent scheduling with `workgroup_x += 0x80`.
- This invalidates the earlier simplified hypothesis that Pro LL graph masked K1 is doing all fixed `size_m` M-blocks. The remaining graph/eager gap is more accurately a physical-stride / compact-layout issue: `size_m` still defines the per-expert memory stride and output layout, even though scheduled M-blocks are bounded by `masked_m`.
- A focused masked-K1 microbench with fixed `actual_m=128` found little sensitivity to the physical `size_m` stride itself. For local experts `E=12`, median K1 time was about `0.727/0.734/0.739 ms` for `size_m=128/512/4096`; for `E=48`, `size_m=128` was `2.9086 ms` and `size_m=4096` was `2.9275 ms`. This makes a simple masked-K1 stride patch unlikely to recover the full graph/eager delta.
- Safe next directions are therefore narrowed:
  - Normal graph: still needs a capture-compatible K1/K3 active CTA pool or device-side active-tile consumer if graph performance is prioritized.
  - Pro LL graph: do not replace the packaged `.co` with the scratch balanced `.s`; first isolate the remaining fixed-capture work outside masked-K1 stride alone, then either obtain/modify a layout-compatible masked-K1 ABI that consumes compact rows or implement a true compact LL graph where masked K1, K2, and K3 all consume compact active rows.

## 2026-07-08 18:22:00 +08:00 - Pro LL Graph K2 CTA Pool Fix
- Re-profiled Pro EP8 LL graph versus eager with a cleaner graph-only profile. The corrected evidence showed masked K1 and K3 local GEMM are close between graph and eager, while K2 SwiGLU/quant was the large graph-only outlier: graph `~0.797 ms/call` versus eager `~0.085 ms/call`.
- Root cause: K2's 2048/4096 register kernels already respect `max_row_blocks` with a grid-stride loop, but the generic hidden path used by Pro `hidden=3072` ignored the computed `launch_blocks` and launched `dim3(rows)`. In exact graph cap512, `rows = local_experts * rows_per_expert = 48 * 4096 = 196608`, so graph paid a capacity-sized empty CTA launch. Eager compact-active only had `48 * 128 = 6144` rows.
- Implemented a low-intrusion K2 fix: the generic K2 kernel now grid-strides over `effective_rows`, and the generic launch branch uses `dim3(launch_blocks)`. This preserves exact worst-capacity correctness because if legal skew makes `effective_rows > launch_blocks`, each CTA loops by `gridDim.x` until all active logical rows are covered.
- Verification:
  - Local `py_compile` and `git diff --check` passed; local pytest was unavailable in the Windows Python.
  - 151.1 source-contract pytest passed: `13 passed`.
  - 151.1 rebuild/import passed; K2 extension rebuilt from the modified source.
  - Pro EP8 LL graph random buckets passed against `ll-masked`. Replay medians improved to `1.444/1.916/1.993/2.134/2.447/3.956 ms` for tokens `8/32/64/128/256/512`, versus prior post-fix `1.439/1.930/2.050/2.289/2.788/4.649 ms`.
  - Same run reported eager token512 `4.0986 ms`, so graph token512 is now slightly faster than eager on this measurement.
  - Pro EP8 LL graph `single-local-rank` token128 passed with `max_abs=0.000488281`.
  - Post-fix graph profile shows K2 down to `~0.099 ms/call`; remaining Pro graph costs are mainly K1 stage-only, masked K1, K3 local GEMM, and K3 combine.
  - Pro EP16 token512 graph sanity passed using IPC peer mode after an unrelated fabric attach failure. Graph replay was `4.0927 ms` versus `ll-masked` baseline `5.0378 ms`, with `max_abs=0.000976562`.
  - Pro EP16 full graph bucket run also passed against `ll-masked` with medians `1.103/1.216/1.301/1.495/2.406/4.178 ms` versus baseline `1.147/1.278/1.405/1.709/2.840/5.118 ms` for tokens `8/32/64/128/256/512`.

## 2026-07-08 18:20:00 +08:00 - Pro LL Graph K2 CTA-Pool Fix
- Re-profiled Pro EP8 LL graph versus eager after the masked-K1 `.co` correction. The packaged masked K1 itself was not the main graph/eager delta: a clean graph profile showed K2 `swiglu_quant_channelwise_kernel` at about `0.797 ms/call`, while eager compact-active K2 was about `0.085 ms/call`.
- Root cause: the generic K2 path used by Pro hidden `3072` computed `launch_blocks` but still launched `dim3(rows)`. In graph exact-capacity mode this meant `48 * 4096 = 196608` CTAs for Pro EP8 cap512, even though the kernel already had actual-M logic to process active rows.
- Applied a low-intrusion fix in K2 generic kernel: it now loops `logical_row += gridDim.x`, and the generic launch uses `dim3(launch_blocks)`. This matches the existing 2048/4096 reg-kernel CTA-pool behavior and preserves legal skew coverage by grid-striding through all `effective_rows`.
- Verification:
  - Local `py_compile` and `git diff --check` passed; local pytest is unavailable on the workstation.
  - 151.1 remote `python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`: `13 passed`.
  - Rebuild/import passed; K2 extension was recompiled and synced. Build run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graphopt/k2_pool_build_20260708_181154`.
  - Pro EP8 LL cap512 graph buckets passed correctness. Graph medians improved to `1.444/1.916/1.993/2.134/2.447/3.956 ms` for tokens `8/32/64/128/256/512`, versus the previous `1.439/1.930/2.050/2.289/2.788/4.648 ms`. Token512 graph is now slightly faster than same-run eager `4.099 ms`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graphopt/pro_ep8_ll_graph512_k2pool_20260708_181412`.
  - Pro EP8 LL adversarial `single-local-rank` graph token128 passed with `max_abs=0.000488281`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graphopt/pro_ep8_ll_skew_graph_k2pool_20260708_181646`.
  - Post-fix graph profile confirmed K2 dropped to about `0.098 ms/call`; remaining large kernels are masked K1, K3 LL GEMM, K3 combine, and K1 stage-only. Profile dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graphopt/pro_ep8_ll_graph_only_hipprof_k2pool_20260708_181752`.
  - Pro EP16 LL single-capture cap512 improved substantially and passed all buckets: MegaMoE graph medians `1.132/1.224/1.309/1.498/2.406/4.144 ms` versus fair `ll-masked` cap512 baseline `2.563/2.633/2.673/2.804/3.578/5.118 ms` for tokens `8/32/64/128/256/512`. Run dir: `/root/yuguo/DeepGEMM/hygon_tmp/supernode_debug/151_1_graphopt/pro_ep16_ll_single_cap512_k2pool_20260708_181907`.
