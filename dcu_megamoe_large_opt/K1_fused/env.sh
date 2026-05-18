#!/usr/bin/env bash
set -euo pipefail

export DTK_ROOT="${DTK_ROOT:-/opt/dtk}"
export ROCM_HOME="${ROCM_HOME:-${DTK_ROOT}}"
export ROCM_PATH="${ROCM_PATH:-${DTK_ROOT}}"
export HIP_PATH="${HIP_PATH:-${DTK_ROOT}/hip}"
export PATH="${DTK_ROOT}/bin:${DTK_ROOT}/hip/bin:${DTK_ROOT}/aillvm/bin:${PATH}"
export LD_LIBRARY_PATH="/opt/hyhal/lib:${DTK_ROOT}/.hyhal/rocm_smi/lib:${DTK_ROOT}/lib:${DTK_ROOT}/lib64:${DTK_ROOT}/hip/lib:${DTK_ROOT}/hip/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/hygon_tmp/large_opt/K1_fused/torch_extensions}"
