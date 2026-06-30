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
