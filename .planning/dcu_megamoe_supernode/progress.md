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
