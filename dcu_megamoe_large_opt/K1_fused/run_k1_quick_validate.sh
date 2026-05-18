#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source dcu_megamoe_large_opt/K1_fused/env.sh

devices="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
nprocs="${K1_NPROCS:-8}"
tokens="${K1_TOKENS:-2048}"
max_tokens="${K1_MAX_TOKENS_PER_RANK:-${tokens}}"
port="${MASTER_PORT:-${K1_MASTER_PORT:-8680}}"
idle_timeout="${K1_IDLE_TIMEOUT_SECONDS:-180}"
warmup="${K1_WARMUP:-2}"
repeat="${K1_REPEAT:-5}"
skip_bench="${K1_SKIP_BENCH:-0}"
fused_only="${K1_FUSED_ONLY:-0}"
bench_scope="${K1_BENCH_SCOPE:-k1_swiglu}"
out="${K1_OUT:-hygon_tmp/large_opt/K1_fused/k1_fused_l1_quick_${tokens}.json}"

args=(
  --num-processes "${nprocs}"
  --num-tokens "${tokens}"
  --num-max-tokens-per-rank "${max_tokens}"
  --bench-scope "${bench_scope}"
  --out "${out}"
)

if [ "${skip_bench}" = "1" ]; then
  args+=(--skip-bench)
else
  args+=(--warmup "${warmup}" --repeat "${repeat}")
fi

if [ "${fused_only}" = "1" ]; then
  args+=(--fused-only)
fi

echo "[K1 quick] tokens=${tokens} max_tokens=${max_tokens} bench_scope=${bench_scope} skip_bench=${skip_bench} fused_only=${fused_only}"
HIP_VISIBLE_DEVICES="${devices}" \
MASTER_PORT="${port}" \
K1_IDLE_TIMEOUT_SECONDS="${idle_timeout}" \
  bash dcu_megamoe_large_opt/K1_fused/run_k1_fused_test.sh "${args[@]}"
