#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DeepGEMM
source dcu_megamoe_large_opt/K1_fused/env.sh

extra_args=()
case "${K3_PATH:-default}" in
  default)
    ;;
  tail-reduce)
    export K3_USE_ASM_TAIL_REDUCE=1
    ;;
  asm-scatter)
    extra_args+=(--k3-mode asm-scatter)
    ;;
  *)
    echo "unknown K3_PATH=${K3_PATH}; expected default, tail-reduce, or asm-scatter" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_BENCH:-0}" == "1" ]]; then
  extra_args+=(--skip-bench)
fi

python3 dcu_megamoe_large_opt/K3_fused/test_k3_fused.py \
  --num-processes "${NUM_PROCESSES:-8}" \
  --num-tokens "${NUM_TOKENS:-512}" \
  --num-max-tokens-per-rank "${NUM_MAX_TOKENS_PER_RANK:-2048}" \
  --correctness-iters "${CORRECTNESS_ITERS:-1}" \
  --warmup "${WARMUP:-2}" \
  --repeat "${REPEAT:-5}" \
  "${extra_args[@]}" \
  "$@"
