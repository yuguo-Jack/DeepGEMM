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
