# DCU MegaMoE V2 Findings

See `.planning/dcu_megamoe_v2/findings.md` for the full working notes.

Key current findings:

- The retained C groupgemm layout for V2 is pack5.
- Small-token K1 baseline uses `V2_K1_LowLatencyMaskedGroupGemmKernel` in the
  independent V2 harness, mirroring the best current `c-ll` path.
- Large-token K1 baseline uses
  `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<256,256,true>`
  with pack5 and aicc.
- Existing K2 `swiglu_quant_channelwise_out` is the first reuse target.
- Existing K1/K3 large-opt ASM wrappers are useful references, but accepted V2
  K3 must be rebuilt on the C groupgemm skeleton with pack5.
- V2 K1 `c-ll-symm-pull` now exercises the DCU MegaMoE sym-buffer peer pointer
  layout in a single fused C kernel. It is still a standalone harness path, not
  the final distributed Python API, but it is no longer the earlier local
  source-row imitation.
- Current K1 symm-pull quick degradation is about +11% at 32/128 tokens versus
  pure V2 K1, within the <=20% K1 target.
- With `--symm-devices 2`, the same K1 kernel performs real cross-device peer
  reads and remains correct, but degradation rises to +31.70% at 32 tokens and
  +40.73% at 128 tokens. Direct remote A loads in the compute loop are not
  sufficiently hidden; the next K1 step should introduce same-kernel
  staged/overlapped remote pull rather than treating peer memory like local HBM.
- `c-ll-symm-stage` stages peer token rows into local scratch inside the same
  K1 kernel before groupgemm reads local staged A. After vectorized 16B staging,
  two-DCU degradation drops to +16.95% at 32 tokens and +16.69% at 128 tokens,
  satisfying the K1 <=20% target for the quick small-token cases.
- K1 small-token 8-rank acceptance is now clean: 32 tokens are +17.46% and 128
  tokens are +18.21% versus pure K1, with correctness max_abs 0.000244141 and
  no value mismatches. hipprof shows only the fused K1 kernel in HIPOPS for the
  timed loop.
- V2 K2 currently reuses the existing optimized
  `megamoe/dcu_megamoe_large_opt/K2_fused/k2_fused_ext.cu` implementation
  through an isolated wrapper. In repo-local test environments without
  `megamoe._C`, the wrapper JIT-builds only that K2 extension into `hygon_tmp`.
- Unified V2 pack5 is now verified for both L1 `N=4096,K=4096` and L2
  `N=4096,K=2048` fixtures. Python and C++ offset helpers match on selected
  probes.
- The previous V2 K3 ASM/kpack2 prototype is rejected for acceptance. It used a
  K3-specific kpack2 layout, an ASM code object, and only large-token coverage.
- K3 ASM/kpack2 timing data is useful only as failure history. It must not be
  used as the K3 pure denominator, and it is not evidence that K3 communication
  has been hidden in the required C pack5 pipeline.
- The next valid K3 work is to adapt the C pack5 groupgemm skeleton for L2
  shape `N=4096, K=2048`, validate pure timings for 32/128/1024/4096, and then
  fuse combine in the C epilogue.
- Two rejected K3 shortcuts still matter for the rebuild: the unadapted V2 C
  pack5 harness VMFaulted at `N=4096,K=2048`, and the existing `K3COMBINE` ASM
  code object VMFaulted with null row-combine pointers.
- Current K3 large C pack5 row-combine is correctness-clean but still not
  performance-accepted. The best correct large path is the same-kernel
  copy-stage with 16 front-scheduled copy workers; 1024/4096 tokens remain
  above the <=25% degradation target.
- The K3 local-rank tail-reduce prototype is correctness-clean but too slow:
  1024 tokens are +75.49% and 4096 tokens are +88.30% versus pure K3 C pack5.
  It is a correctness checkpoint, not final combine-reduce acceptance.
- Two K3 copy-stage variants are rejected:
  - Moving copy workers after the compute grid regressed 1024-token timing to
    about 0.794 ms.
  - Same-block self-copy with `K3_COPY_WORKERS=0` had promising raw timing but
    first failed correctness with missing combine rows; after repairing wave
    participation it passed correctness but still slowed 1024 tokens to
    0.665637 ms, so it was reverted.
- K3/L2 `--c-tile-n 64` pack5 is rejected for large tokens: pure 1024 was
  correct but slowed to 0.934237 ms.
- Sorting standard combine rows by source rank/partial row is rejected because
  it regressed 4096-token copy-stage timing to 3.50465 ms.
- Device-scope ready fencing is rejected for K3 copy-stage: replacing
  `__threadfence_system()` with `__threadfence()` looked faster but later
  failed repeated correctness with NaN/mismatch after a forced rebuild.
- Copy-worker counts above 16 are rejected: 24 workers slowed 4096 tokens to
  2.31437 ms, and 32 workers slowed 4096 tokens to 3.74851 ms.
- Current restored K3 large copy-stage baseline is correctness-clean but still
  above target: 1024 tokens 0.613866 ms (+39.31%) and 4096 tokens 1.7904 ms
  (+37.82%).
- Save-temps show the K3 copy-stage specialization has no private scratch, but
  copy-worker blocks still reserve the GEMM kernel's 64 KiB LDS, which limits
  occupancy overlap.
- K3 small same-kernel local-rank tail-reduce is correctness-clean and within
  target: 32 tokens 0.164469 ms (+5.66%, max_abs 0.000244141), 128 tokens
  0.184448 ms (+12.19%, max_abs 0.000488281), both with value_mismatch 0.
  hipprof shows only the fused low-latency C pack5 kernel in HIPOPS.
- The small tail-reduce checkpoint is still local-rank scoped; full all-rank
  combine-reduce and end-to-end correctness remain pending.
- K3 large PMC confirms that direct row-combine and copy-stage fail for
  different reasons. Direct remote row-combine has high write/TCP stalls
  (`TCC_EA_WRREQ_STALL` about 4.50M, TCP data stall about 15.07M at 1024),
  while copy-stage reduces those write stalls but adds local reads and keeps
  copy-worker blocks at the full 64 KiB LDS footprint.
- A 32-bit adjacent-hidden pair-store epilogue for direct row-combine is
  rejected. It passed 1024 correctness but slowed 1024 to 0.711039 ms and 4096
  to 2.38252 ms, so the experiment was reverted.
- Large tail-reduce local-copy filtering is correct but not enough. It improves
  1024 from 0.773269 ms to 0.764443 ms and 4096 from 2.44608 ms to 2.39046 ms,
  but both remain far above target. The dominant large tail-reduce cost is not
  just copying non-local rows.
- Large local-rank tail-reduce topk-slot skipping improves the prototype to
  0.744538 ms at 1024 and 2.34016 ms at 4096, with max_abs 0 for both. This is
  still far above target and is not the final all-rank reduce semantics.
- K3 large copy-stage flag synchronization should stay on the restored
  `__threadfence_system()` plus ordinary flag store path for now. An
  `atomicExch` ready flag passed correctness but was slower, and a raw-buffer
  GLC flag variant hung during 8-rank correctness and caused the runtime to
  exit. Do not retry either version without first building a small standalone
  synchronization microbench.
- K3 large copy-worker row-pointer broadcast is rejected. Each output row has
  32 `uint4` copy chunks, so the attempted patch loaded `row_output_ptrs[row]`
  once from the chunk-0 lane and broadcast it across the half-wave. It passed
  8-rank correctness at 1024/4096 with `max_abs=0`, but timing regressed to
  0.625124 ms at 1024 and 1.81698 ms at 4096. The shuffle/address-broadcast
  overhead outweighed the metadata-load reduction.
- K3 large copy-worker row-tile scheduling is rejected. Assigning each worker a
  subset of row tiles and sweeping hidden tiles in order preserved 8-rank
  correctness (`max_abs=0` for 1024 and 4096), but timing regressed to
  0.755052 ms at 1024 and 2.02361 ms at 4096. The restored linear
  `tile += worker_count` order better matches tile-ready publication and
  remains the copy-stage baseline.
- Retired ASM/balanced-ASM host plumbing is not needed for accepted V2 because
  K1/K3 must use the C pack5 implementation.
- Retired `small-pull`, `small-symm-pull`, and `large-symm-pull` modes are
  historical only; active K1 fused work now goes through staged variants.
- `--symm-ranks` is the logical rank count for symmetric-buffer layout and
  route mapping. `--symm-devices` is the number of visible DCUs used by the
  standalone harness to place those logical rank buffers. They match in normal
  8-rank/8-HCU acceptance, but can differ for local simulations such as 8
  logical ranks over 4 visible devices.
- K3 large copy-stage tile-ready publication must wait for every compute
  thread's output stores, not only thread 0's stores. A thread-0-only VMEM wait
  before `__threadfence_system()` allowed copy workers to observe ready flags
  before all lanes' `out` stores were visible, causing sparse 4096-token
  mismatches on 8 ranks. The corrected all-thread wait restores correctness but
  raises overhead to about +41%, so optimization should continue from this
  correctness-safe baseline.
- K3 large corrected copy-stage profiling shows the current gap is memory-side:
  VMEM reads rise about 1.73x and TCP data stalls about 1.83x versus pure K3 at
  1024 tokens, while LDS wait is unchanged and LDS bank conflicts remain zero.
  Pair-copy coarsening of copy-worker tasks passed correctness but slowed 1024
  and 4096, so rowptr-load frequency alone is not the dominant bottleneck.
- A self-copy epilogue, where compute blocks copy their own `out` tile into
  combine targets instead of using copy-worker blocks, is rejected. It was
  faster on invalid timing but produced many zero rows in the combine buffer,
  and adding a device-scope fence before the self-copy did not fix correctness.
- Device-scope ready fencing after the all-thread VMEM wait is correctness-clean
  in one 1024/4096 8-rank run, but it only moved 1024 by about 1 us and slightly
  regressed 4096, so it was reverted to the safer system-scope publication.
- K3 large direct remote-store remains open but not accepted. The normal direct
  rowptr path is correctness-clean but slower than corrected copy-stage; sorting
  tasks by combine row (`--k3-sort-rows 1`) gives a small 4096-token win while
  preserving correctness. Linear combine-row remapping is rejected because 4096
  correctness fails, and row-resource / buffer-store variants are rejected
  because they pass correctness but are materially slower.
- After the all-thread wait fix, sorted routing is also mildly beneficial
  for corrected copy-stage large tokens and remains correctness-clean in real
  8-rank tests. It is not enough to meet target. The temporary `K3_SORT_ROWS`
  script switch was removed during cleanup; copy-stage uses sorted routing by
  default in the active V2 source.
- Direct rowptr 4096 without sorting can still show sparse correctness failures,
  so future direct remote-store experiments should use sorted rows unless they
  explicitly prove a new ordering is correct.
- `global_store_short ... glc slc` is compile-proven on gfx938 but rejected for
  K3 direct remote rowptr stores: correctness passed, yet 1024 and 4096 timing
  regressed dramatically. Do not use cache-bypass global short stores for this
  combine path without a new microbenchmark reason.
## 2026-06-02 - K3 Large Tail Reduce Findings

- Accepted K3 large tail-reduce experiments remain on the C pack5 implementation. No ASM/kpack2 path is used.
- Host-side mask/list preparation is outside the timed benchmark path and feeds the same fused compute/copy/reduce kernel.
- The useful local-rank tail-reduce optimizations are:
  - topk slot mask precomputation;
  - separate final tail output to skip zero-mask token stores;
  - active token list to skip reduce work for tokens with no local contribution.
- A local tail-row copy list was correctness-clean but only a micro-optimization; it was removed during cleanup because its repeated ready-flag waits offset most row-scan savings.
- Current 8-rank K3 large tail-reduce result:
  - 1024 tokens: fused `0.636959 ms` versus pure `0.444784 ms`, `+43.21%`, correctness `max_abs=0`.
  - 4096 tokens: fused `1.88045 ms` versus pure `1.30267 ms`, `+44.35%`, correctness `max_abs=0`.
- hipprof for `hygon_tmp/dcu_megamoe_v2/hipprof_k3_large_tail_reduce_active4096` shows only the V2 large C fused kernel in the timed HIPOPS path. There is no standalone reduce kernel.
- Remaining K3 large gap is dominated by copy-stage/synchronization/communication scheduling rather than the local-rank reduce math alone.
