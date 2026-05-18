# DCU MegaMoE K3 Fused

This folder isolates the large-shape K3 path. The prefix remains the baseline
DeepEP + DeepGEMM flow through dispatch, L1, and SwiGLU/quant. Only K3 is
swapped between the baseline and fused variants.

## Baseline

Baseline K3 is:

```text
DeepGEMM L2 -> DeepEP gather/postprocess -> DeepEP combine
```

## Fused Paths

The default fused path is the production candidate:

```text
ASM L2 + direct combine store -> rank_barrier -> reduce_local_combine
```

It launches three kernels after the common prefix. Additional paths are kept for
A/B and integration experiments:

| Path | Selector | Kernel shape |
| --- | --- | --- |
| `default` | no env | `ASM combine -> rank_barrier -> reduce` |
| `tail-reduce` | `K3_USE_ASM_TAIL_REDUCE=1` | one ASM launch with appended reducer WGs |
| `asm-scatter` | `--k3-mode asm-scatter` | original ASM L2 -> HIP scatter -> barrier -> reduce |

`tail-reduce` is correctness-clean, but it is not the default because the
extra reducer WGs run inside the heavy GEMM ASM code object and were slower than
the default separate reduce path at 2048 tokens in local testing.

The older `barrier-reduce` path was removed because it did not beat the default
path and carried a `4096/topk6`-only HIP kernel.
The standalone `tail-signal` path was also removed after intermittent hangs in
the wait-reduce benchmark path; `tail-reduce` keeps only the signal pieces it
needs internally.

## Shape Constraints

Token count is intended to vary through `NUM_TOKENS` and
`NUM_MAX_TOKENS_PER_RANK`. Non-token dimensions are much more constrained in
the current large-shape implementation:

| Path | Non-token constraints |
| --- | --- |
| `default` | The L2 ASM is specialized for `hidden=4096` and `intermediate_hidden=2048`. The current test/integration envelope also asserts `num_experts=256` and `num_topk=6`; `num_experts` must be divisible by the rank count. |
| `tail-reduce` | Same L2 ASM constraints as `default`. In addition, the ASM tail reducer itself is specialized for `hidden=4096` and `topk=6`, and the signal path assumes at most 8 local ranks. Current validation is for `num_experts=256`. |

The default reduce kernel is written with `num_topk`/`num_experts` parameters,
but this folder should still be treated as a DeepSeek-V4 large-shape path until
those dimensions are retested and the harness assertions are relaxed. Grouped
rows still need to be padded to the 256-row GEMM tile expected by the ASM path.

## Run

Use the wrapper from inside the remote container:

```bash
cd /workspace/DeepGEMM
K3_PATH=default NUM_PROCESSES=8 NUM_TOKENS=2048 \
  bash dcu_megamoe_large_opt/K3_fused/run_k3_fused_test.sh
```

Available `K3_PATH` values are `default`, `tail-reduce`, and `asm-scatter`.

Correctness-only:

```bash
K3_PATH=default NUM_PROCESSES=8 NUM_TOKENS=512 SKIP_BENCH=1 \
  bash dcu_megamoe_large_opt/K3_fused/run_k3_fused_test.sh
```

Stage timing:

```bash
K3_PATH=default NUM_PROCESSES=4 NUM_TOKENS=2048 \
  bash dcu_megamoe_large_opt/K3_fused/run_k3_fused_test.sh --profile-k3-stages
```

Scratch build products are written under `hygon_tmp/large_opt/K3_fused`.

## Tuning Knobs

- `K3_REDUCE_THREADS`: threads for reduce kernels, default `128`.
- `K3_ASM_TAIL_REDUCE_BLOCKS`: appended reducer WGs for `tail-reduce`, default `128`.
