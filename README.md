# DeepGEMM

DeepGEMM is a unified, high-performance tensor core kernel library that brings together the key computation primitives of modern large language models — GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more — into a single, cohesive CUDA codebase. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

DeepGEMM leverages some concepts from [CUTLASS](https://github.com/nvidia/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute), but avoids heavy reliance on their templates or algebras. The library is designed for simplicity, with only a limited number of core kernel functions, making it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes.

## News

- 2026.04.16: Mega MoE, FP8xFP4 GEMM, FP4 Indexer, PDL, faster JIT compilation and more.
    - Please see [#304](https://github.com/deepseek-ai/DeepGEMM/pull/304) for more details.
    - For Mega MoE benchmarks, refer to [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316).
- 2025.09.28: DeepGEMM now supports scoring kernels (weighted ReLU MQA logits) for the lightning indexer for DeepSeek v3.2.
    - Please see [#200](https://github.com/deepseek-ai/DeepGEMM/pull/200) for more details.
- 2025.07.20: DeepGEMM now supports both SM90/SM100, and has a full refactor with a low-CPU-overhead JIT CPP module.
    - NVRTC and post-compilation SASS optimization are all disabled.
    - NVRTC will be supported later.
    - As NVCC 12.9 will automatically do the FFMA interleaving, all post optimizations will be no longer supported.
    - Please see [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112) for more details.
- 2025.05.14: DeepGEMM now offers weight gradient kernels for dense and MoE backward! See [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95) for details.
- 2025.05.07: DeepGEMM now supports NVRTC with up to 10x compilation speedup! See [#94](https://github.com/deepseek-ai/DeepGEMM/pull/94) for details. Please use `DG_JIT_USE_NVRTC=1` to enable it (may have performance loss with some cases).
- 2025.04.18: DeepGEMM now achieves up to **1550 TFLOPS** on H800! See [#74](https://github.com/deepseek-ai/DeepGEMM/pull/74), [#78](https://github.com/deepseek-ai/DeepGEMM/pull/78), [#81](https://github.com/deepseek-ai/DeepGEMM/pull/81), [#86](https://github.com/deepseek-ai/DeepGEMM/pull/86) and [340d988](https://github.com/deepseek-ai/DeepGEMM/commit/340d9880f4a418d943d34260d20a79f41f4c0526) for details.

## Quick start

### Requirements

- NVIDIA SM90 or SM100 architecture GPU
- Python 3.8 or higher
- Compilers with C++20 support
- CUDA Toolkit:
    - CUDA 12.3 or higher for SM90
        - **We highly recommend 12.9 or higher for the best performance**
    - CUDA 12.9 or higher for SM100
- PyTorch 2.1 or higher
- CUTLASS 4.0 or higher (could be cloned by Git submodule)
- `{fmt}` library (could be cloned by Git submodule)

### Development

```bash
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Link some essential includes and build the CPP JIT module
cat develop.sh
./develop.sh
```

### Installation

```bash
cat install.sh
./install.sh
```

Then, import `deep_gemm` in your Python project, and enjoy!

## Interfaces

#### Notices

This library provides optimized GEMM kernels for NVIDIA GPUs with a naming convention: `D = C + A @ B`. The input shape layout is NT (non-transposed A, transposed B). While the SM90 implementation supports only the NT memory layout (row-major, col-major), the SM100 implementation supports all memory layouts (NT, TN, NN, TT). For example, `fp8_gemm_nt` will do a `D = C + A @ B.T`

For both architectures, the LHS scaling factor is required to have a TMA-aligned and transposed layout. And the data format for the scaling factor of SM90 and SM100 is different:

- SM90 requires scaling factors in FP32 format.
- SM100 requires scaling factors in packed [UE8M0](https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats) format, which packs 4 UE8M0 into a single `torch.int`.

Please note that operations like input transposition or FP8 casting must be handled separately by the user, please implement or fuse them into prior kernels independently. While the library provides some simple PyTorch utility functions, these may result in slower performance, but our primary focus is on optimizing the GEMM kernels themselves.

#### Normal dense GEMMs (non-grouped)

To perform a basic non-grouped FP8 GEMM, call the `fp8_gemm_{nt, nn, tn, tt}` function. For more details, please refer to the function documentation.

#### Grouped GEMMs (contiguous layout)

Unlike traditional grouped GEMMs in CUTLASS, DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for scenarios where experts in an MoE model share the same shape. For training forward passes or inference prefilling, where each expert may process a varying number of tokens, we concatenate these tokens into a single tensor, referred to as the "contiguous" layout. Note that each expert segment must be aligned to the GEMM M block size (`get_mk_alignment_for_contiguous_layout()`).  For more information, please refer to the `m_grouped_fp8_gemm_{nt, nn}_contiguous` function documentation.

We also provide a K-axis-grouped API for MoE weight backward (with M and N must remain fixed), please refer to `k_grouped_fp8_gemm_tn_contiguous` for more information.

#### Grouped GEMMs (masked layout)

During the inference decoding phase, when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives, we support masked grouped GEMMs. By providing a mask tensor, the kernel computes only the valid portions.

Use `m_grouped_fp8_gemm_nt_masked` for this purpose and consult the relevant documentation. An example usage is to use the output of low-latency kernels from [DeepEP](https://github.com/deepseek-ai/DeepEP) as input.

#### V3.2 MQA kernels for the indexer

The kernel family has two versions, non-paged (for prefilling) and paged (for decoding).
Take the non-paged version `fp8_mqa_logits` as an example. It has 6 inputs:

- `q`, E4M3 tensor with shape `[seq_len, num_heads, head_dim]`
- `kv`, E4M3 tensor (shaped as `[seq_len_kv, head_dim]`) with float SF (shaped as `[seq_len_kv]`)
- `weights`, float tensor with shape `[seq_len, num_heads]`
- `cu_seq_len_k_start` and `cu_seq_len_k_end`, int tensor with shape `[seq_len]`
- `clean_logits`, whether to clean the unfilled logits into `-inf`

The output tensor is shaped as `[seq_len, seq_len_kv]`, indicating token-to-token logits.
For each token `i` in `q`, it will iterate all tokens `j` from `[cu_seq_len_k_start[i], cu_seq_len_k_end[i])`,
and calculate the logit `out[i, j]` as:

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)  # [head_dim]
out_ij = q[i, :, :] @ kv_j  # [num_heads]
out_ij = out_ij.relu() * weights[i, :]  # [num_heads]
out_ij = out_ij.sum()  # Scalar
```

For more details and the paged version `fp8_paged_mqa_logits`, please refer to `tests/test_attention.py`.

#### Mega MoE

Mega MoE fuses and overlaps EP dispatch, linear 1 (FP8xFP4), SwiGLU, linear 2 (FP8xFP4), and EP combine into a single mega-kernel, overlapping NVLink communication and tensor core computation. It requires multi-process launch with symmetric memory. Usage:

```python
# Allocate symmetric memory buffer
# NOTES: requires PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden
)

# Transform weights (FP4 with UE8M0 SF) into the required layout
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

# Copy inputs into the buffer before each call
# You may fuse these into previous kernels
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# Run the fused mega MoE kernel
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

For the full example with multi-process setup and benchmarking, please refer to `tests/test_mega_moe.py`.

#### DCU/HIP W8A8 Mega MoE

The DCU path builds a standalone `megamoe` HIP extension for Hygon `gfx938`.  The
current fused W8A8 FP8 channelwise kernel is specialized for the DSV4-Flash
MegaMoE size used by the benchmark below:

- EP size: 8 ranks
- Experts: 256 total, 32 per rank
- Top-K: 6
- Hidden size: 4096
- Intermediate hidden size: 2048
- Maximum tokens per rank: 2048 for the staged large-shape path in this tree

Build on a DTK 26.04 environment:

```bash
source /opt/dtk-26.04/env.sh
./build_dcu_megamoe.sh
```

The build script keeps intermediate files under `build/` and writes the wheel to
`build/whl/`.  It also builds the extension in place, so the local checkout can
run the tests directly.
If you do not install the wheel but want to import `megamoe` from another
directory, either set `PYTHONPATH` to this repository root or install the source
tree in editable mode after building:

```bash
PYTHONPATH=/workspace/DeepGEMM python your_script.py
# or
pip install -e .
```

Editable installs point Python back to this checkout, so they use the in-place
`megamoe/_C*.so`, staged `k1/k2/k3_fused_ext*.so`, and staged `.co` files
created by `build_dcu_megamoe.sh`.  Python-only edits are picked up directly;
after changing HIP, asm, or `setup.py`, rerun `build_dcu_megamoe.sh`.  If you
run from an installed wheel instead, reinstall the newly generated wheel after
rebuilding.
The optional large-token staged path is built ahead of time as part of the
`megamoe` wheel.  Wheel installation places the staged extension modules and asm
code objects under the Python package directory, alongside the original
MegaMoE fused extension:

- `megamoe/_C*.so`
- `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext*.so`
- `megamoe/dcu_megamoe_large_opt/K2_fused/k2_fused_ext*.so`
- `megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused_ext*.so`
- `megamoe/dcu_megamoe_large_opt/K1_fused/*.co`
- `megamoe/dcu_megamoe_large_opt/K3_fused/*.co`

If any staged HIP or asm source changes, rebuild and reinstall the wheel.  The
`hygon_tmp` directory is only used by test scripts for temporary reports or
scratch files; it is not required for installed kernel binaries.

Run the DSV4-Flash correctness and performance check:

```bash
source /opt/dtk-26.04/env.sh
python tests/test_mega_moe_dcu.py \
  --num-processes 8 \
  --num-max-tokens-per-rank 2048 \
  --num-tokens 512 \
  --hidden 4096 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 6 \
  --correctness-iters 1 \
  --warmup 3 \
  --repeat 8 \
  --out hygon_tmp/megamoe_dcu_dsv4_flash_512.json
```

The default DCU execution path remains the original persistent fused kernel.
Set `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1` to route the same public
`megamoe.fp8_w8a8_mega_moe` API through the large-token staged path:

- K1: dispatch pull + L1 FP8 grouped GEMM
- K2: SwiGLU + channelwise FP8 quant
- K3: L2 FP8 grouped GEMM + combine reduce

The staged path is intended for the DeepSeek-V4-Flash EP8 shape above and keeps
all K1/K2/K3 implementation files under `megamoe.dcu_megamoe_large_opt`, so the
public and optional large-token paths stay inside the `megamoe` package.
Its large temporary activations reuse the original DCU MegaMoE `route_scratch`
allocation; the integration does not allocate a second persistent L1/K2/K3
activation workspace.
By default K3 uses the integrated `ASM combine -> barrier -> reduce` path.
Set `K3_USE_ASM_TAIL_REDUCE=1` together with
`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1`, or run the sweep script with
`K3_PATH=tail-reduce`, to test the K3 tail-reduce variant.
For `num_max_tokens_per_rank <= 2048`, the tail reducer defaults to 64
reducer workgroups; larger max-token buffers keep the previous 128-workgroup
default.
The staged path keeps the tail-reduce signal state in `route_scratch`; when the
large-opt environment is enabled before creating the symmetric buffer, this
state is prepared during buffer initialization rather than the timed execution
path.

One host-side tuning knob is available for the staged path and does not add
device kernels:

- `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS` controls when K2 consumes K1's
  `row_combine_ptrs` validity metadata. The default is 1536, so larger token
  counts skip inactive-row activation work while smaller token counts keep the
  leaner K2 launch path.

Run the requested token-per-rank sweep. The default list includes compact-window
representatives around 1025..1441 as well as the main 512/1024/2048 sizes:

```bash
source /opt/dtk-26.04/env.sh
bash scripts/run_dcu_megamoe_large_opt.sh
```

For a correctness-only smoke run, set `SKIP_BENCH=1`.  The staged test keeps
the weight FP8 conversion chunked by default; tune
`MEGAMOE_DCU_WEIGHT_CAST_CHUNK_ROWS` if the random-weight setup needs a smaller
or larger temporary allocation.

Or run one size directly:

```bash
source /opt/dtk-26.04/env.sh
MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 python tests/test_mega_moe_dcu.py \
  --num-processes 8 \
  --num-max-tokens-per-rank 2048 \
  --num-tokens 1024 \
  --hidden 4096 \
  --intermediate-hidden 2048 \
  --num-experts 256 \
  --num-topk 6 \
  --correctness-iters 1 \
  --warmup 3 \
  --repeat 8 \
  --out hygon_tmp/large_opt/integrated/dsv4_flash_large_opt_1024.json
```

To sweep the original persistent fused path while keeping the symmetric buffer
capacity fixed at 2048 tokens per rank:

```bash
source /opt/dtk-26.04/env.sh
mkdir -p hygon_tmp/megamoe_dcu_dsv4_flash
for tokens in 512 1024 2048; do
  python tests/test_mega_moe_dcu.py \
    --num-processes 8 \
    --num-max-tokens-per-rank 2048 \
    --num-tokens "${tokens}" \
    --hidden 4096 \
    --intermediate-hidden 2048 \
    --num-experts 256 \
    --num-topk 6 \
    --correctness-iters 1 \
    --warmup 3 \
    --repeat 8 \
    --out "hygon_tmp/megamoe_dcu_dsv4_flash/bench_${tokens}.json"
done
```

#### Utilities

The library provides some utility functions besides the above kernels:

- `deep_gemm.set_num_sms` / `get_num_sms`: set/get the maximum SM count to use
- `deep_gemm.set_tc_util` / `get_tc_util`: set/get an approximated tensor core utilization ratio
- `deep_gemm.set_pdl` / `get_pdl`: enable/disable Programmatic Dependent Launch (PDL)
- `deep_gemm.set_mk_alignment_for_contiguous_layout` / `get_mk_alignment_for_contiguous_layout`: set/get the group-level M/K alignment for contiguous layout
- `deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout`: get the theoretical minimum M/K alignment
- `deep_gemm.set_ignore_compile_dims`: configure dimensions to ignore during JIT compilation
- `deep_gemm.set_block_size_multiple_of`: constrain block sizes to be multiples of a given value
- `deep_gemm.transform_sf_into_required_layout`: transform scaling factors into the required layout
- `deep_gemm.get_tma_aligned_size`: get the required TMA alignment size
- `deep_gemm.get_mn_major_tma_aligned_tensor`: get a MN-major TMA-aligned tensor
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`: get a MN-major TMA-aligned tensor (with packing FP32 into UE8M0)
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`: K-grouped GEMM packing kernel

The library also provides some environment variables, which may be useful:

- General
    - `DG_JIT_DEBUG`: `0` or `1`, print JIT debugging information, `0` by default
    - `DG_PRINT_CONFIGS`: `0` or `1`, print selected configs for each shape, `0` by default
- JIT cache
    - `DG_JIT_CACHE_DIR`: string, cache directory for compiled kernels, `$HOME/.deep_gemm` by default
- Compiler selection
    - `DG_JIT_USE_NVRTC`: `0` or `1`, use NVRTC instead of NVCC (faster compilation, may have lower performance for some cases), `0` by default
    - `DG_JIT_NVCC_COMPILER`: string, NVCC compiler path; defaults to `torch.utils.cpp_extension.CUDA_HOME`
    - `DG_JIT_CPP_STANDARD`: integer, C++ standard version, `20` by default
- Compiler output
    - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` or `1`, print compilation commands, `0` by default
    - `DG_JIT_PTXAS_VERBOSE`: `0` or `1`, show detailed PTXAS output, `0` by default
    - `DG_JIT_PTXAS_CHECK`: `0` or `1`, assert no local memory usage in compiled kernels, `0` by default
    - `DG_JIT_PRINT_LOAD_TIME`: `0` or `1`, print kernel load time, `0` by default
- Debug and profiling
    - `DG_JIT_WITH_LINEINFO`: `0` or `1`, embed source line info for profiling tools, `0` by default
    - `DG_JIT_DUMP_ASM`: `0` or `1`, dump both PTX and SASS, `0` by default
    - `DG_JIT_DUMP_PTX`: `0` or `1`, dump PTX output, `0` by default
    - `DG_JIT_DUMP_SASS`: `0` or `1`, dump SASS output, `0` by default
    - `DG_COMM_KERNEL_DEBUG`: `0` or `1`, zero symmetric buffer before each Mega MoE call for debugging, `0` by default
    - `DG_USE_NVIDIA_TOOLS`: `0` or `1`, skip internal profiling when running under external NVIDIA tools, `0` by default
- Build options
    - `DG_SKIP_CUDA_BUILD`: `0` or `1`, skip CUDA extension build during installation, `0` by default
    - `DG_FORCE_BUILD`: `0` or `1`, force local build instead of downloading pre-built wheels, `0` by default
    - `DG_JIT_USE_RUNTIME_API`: `0` or `1`, use CUDA Runtime API for kernel loading (requires CUDA runtime >= 12.8), `0` by default

For additional examples and details, please refer to [the test code](tests/test_core.py) or review the corresponding Python documentation.

## Acknowledgement

DeepGEMM is inspired by the [CUTLASS](https://github.com/nvidia/cutlass) project. Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient BLAS kernel library on GPU}, 
      author={Chenggang Zhao and Zhean Xu and Liang Zhao and Jiashi Li and Chenhao Xu and Anyi Xu and Shengyu Liu and Kexing Zhou and Kuai Yu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
