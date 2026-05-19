#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

num_processes="${NUM_PROCESSES:-8}"
max_tokens="${NUM_MAX_TOKENS_PER_RANK:-2048}"
warmup="${WARMUP:-3}"
repeat="${REPEAT:-8}"
correctness_iters="${CORRECTNESS_ITERS:-1}"
out_dir="${OUT_DIR:-hygon_tmp/large_opt/integrated}"
tokens_list="${TOKENS_LIST:-512 1024 1025 1280 1441 1442 2048}"
skip_bench="${SKIP_BENCH:-0}"

case "${K3_PATH:-default}" in
    default)
        ;;
    tail-reduce)
        export K3_USE_ASM_TAIL_REDUCE=1
        ;;
    *)
        echo "unknown K3_PATH=${K3_PATH}; expected default or tail-reduce" >&2
        exit 2
        ;;
esac

mkdir -p "$out_dir"

for tokens in $tokens_list; do
    extra_args=()
    if [[ "$skip_bench" == "1" || "$skip_bench" == "true" || "$skip_bench" == "TRUE" ]]; then
        extra_args+=(--skip-bench)
    fi

    MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 \
    python tests/test_mega_moe_dcu.py \
        --num-processes "$num_processes" \
        --num-max-tokens-per-rank "$max_tokens" \
        --num-tokens "$tokens" \
        --hidden 4096 \
        --intermediate-hidden 2048 \
        --num-experts 256 \
        --num-topk 6 \
        --correctness-iters "$correctness_iters" \
        --warmup "$warmup" \
        --repeat "$repeat" \
        --out "$out_dir/dsv4_flash_large_opt_${tokens}.json" \
        "${extra_args[@]}"
done
