#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source dcu_megamoe_large_opt/K1_fused/env.sh

tokens=("$@")
if [ "${#tokens[@]}" -eq 0 ]; then
  tokens=(512 1024 1536 2048)
fi

nprocs="${K1_NPROCS:-8}"
devices="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

for t in "${tokens[@]}"; do
  max_tokens="${K1_MAX_TOKENS_PER_RANK:-${t}}"
  echo "K1_FUSED_SWEEP_TOKEN=${t}"
  HIP_VISIBLE_DEVICES="${devices}" \
    bash dcu_megamoe_large_opt/K1_fused/run_k1_fused_test.sh \
      --num-processes "${nprocs}" \
      --num-tokens "${t}" \
      --num-max-tokens-per-rank "${max_tokens}" \
      --skip-bench \
      --out "hygon_tmp/large_opt/K1_fused/k1_fused_l1_swiglu_result_${t}.json"
done
