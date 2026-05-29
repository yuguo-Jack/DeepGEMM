#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2_DIR="${ROOT_DIR}/csrc/kernels/dcu_megamoe_v2"

MODE="${MODE:-build}"
DEVICE="${DEVICE:-0}"
SYMM_RANKS="${SYMM_RANKS:-8}"
SYMM_DEVICES="${SYMM_DEVICES:-1}"
RANK_IDX="${RANK_IDX:-0}"
C_STAGE_N_GROUP="${C_STAGE_N_GROUP:-4}"
C_ROW_STAGE="${C_ROW_STAGE:-1}"
K3_COPY_STAGE="${K3_COPY_STAGE:-0}"
K3_COPY_WORKERS="${K3_COPY_WORKERS:-16}"
K3_TAIL_REDUCE="${K3_TAIL_REDUCE:-0}"
CHECK="${CHECK:-1}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-30}"
MEASURE_ROUNDS="${MEASURE_ROUNDS:-3}"
ALLOWED_MAX_ABS="${ALLOWED_MAX_ABS:-0.001}"

COMMON_ARGS=(
  --topk 6
  --n 4096
  --experts 32
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --measure-rounds "${MEASURE_ROUNDS}"
  --check "${CHECK}"
  --realistic-values 1
  --input-value-scale 0.02
  --weight-value-scale 0.02
  --allowed-max-abs "${ALLOWED_MAX_ABS}"
)

run_small_token() {
  local tokens="$1"
  local kernel_mode="${2:-c-ll}"
  local problem_k="${3:-4096}"
  local k3_rowptr="${4:-0}"
  local k3_combine="${5:-0}"
  local block_m
  local symm_args=()
  local k3_args=()
  case "${tokens}" in
    32) block_m=32 ;;
    64) block_m=16 ;;
    128) block_m=32 ;;
    256) block_m=48 ;;
    *)
      echo "No tuned V2 small-token preset for tokens=${tokens}" >&2
      return 2
      ;;
  esac
  case "${kernel_mode}" in
    c-ll-symm-stage)
      symm_args=(
        --symm-ranks "${SYMM_RANKS}"
        --symm-devices "${SYMM_DEVICES}"
        --rank-idx "${RANK_IDX}"
      )
      ;;
  esac
  if [[ "${k3_combine}" == "1" ]]; then
    symm_args=(
      --symm-ranks "${SYMM_RANKS}"
      --symm-devices "${SYMM_DEVICES}"
      --rank-idx "${RANK_IDX}"
    )
    k3_args+=(--k3-combine 1)
    if [[ "${K3_TAIL_REDUCE}" == "1" ]]; then
      k3_args+=(--k3-tail-reduce 1)
    fi
  fi
  if [[ "${k3_rowptr}" == "1" ]]; then
    k3_args+=(--k3-rowptr 1)
  fi
  make -C "${V2_DIR}" hipcc
  HIP_VISIBLE_DEVICES="${DEVICE}" "${V2_DIR}/k1_groupgemm_v2_hipcc" \
    --mode "${kernel_mode}" \
    --k "${problem_k}" \
    --tokens "${tokens}" \
    --ll-block-m "${block_m}" \
    --ll-cus 64 \
    "${symm_args[@]}" \
    "${k3_args[@]}" \
    "${COMMON_ARGS[@]}"
}

run_large_token() {
  local tokens="$1"
  local kernel_mode="${2:-c}"
  local problem_k="${3:-4096}"
  local k3_rowptr="${4:-0}"
  local k3_combine="${5:-0}"
  local k3_copy_stage="${6:-0}"
  local symm_args=()
  local k3_args=()
  case "${kernel_mode}" in
    c-symm-stage)
      symm_args=(
        --symm-ranks "${SYMM_RANKS}"
        --symm-devices "${SYMM_DEVICES}"
        --rank-idx "${RANK_IDX}"
      )
      if [[ "${kernel_mode}" == "c-symm-stage" ]]; then
        symm_args+=(--c-stage-n-group "${C_STAGE_N_GROUP}")
        symm_args+=(--c-row-stage "${C_ROW_STAGE}")
      fi
      ;;
  esac
  if [[ "${k3_combine}" == "1" ]]; then
    symm_args=(
      --symm-ranks "${SYMM_RANKS}"
      --symm-devices "${SYMM_DEVICES}"
      --rank-idx "${RANK_IDX}"
    )
    k3_args+=(--k3-combine 1)
  fi
  if [[ "${k3_copy_stage}" == "1" ]]; then
    k3_args+=(--k3-copy-stage 1)
    k3_args+=(--k3-copy-workers "${K3_COPY_WORKERS}")
    if [[ "${K3_TAIL_REDUCE}" == "1" ]]; then
      k3_args+=(--k3-tail-reduce 1)
    fi
  fi
  if [[ "${k3_rowptr}" == "1" ]]; then
    k3_args+=(--k3-rowptr 1)
  fi
  make -C "${V2_DIR}" aicc
  HIP_VISIBLE_DEVICES="${DEVICE}" "${V2_DIR}/k1_groupgemm_v2_aicc" \
    --mode "${kernel_mode}" \
    --c-lowlat-pack 1 \
    --k "${problem_k}" \
    --tokens "${tokens}" \
    "${symm_args[@]}" \
    "${k3_args[@]}" \
    "${COMMON_ARGS[@]}"
}

case "${MODE}" in
  build)
    make -C "${V2_DIR}" all
    ;;
  small)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll
    done
    ;;
  small-symm-stage)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll-symm-stage
    done
    ;;
  large)
    for tokens in ${TOKENS:-1024 4096}; do
      run_large_token "${tokens}"
    done
    ;;
  large-symm-stage)
    for tokens in ${TOKENS:-1024 4096}; do
      run_large_token "${tokens}" c-symm-stage
    done
    ;;
  k3-small)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll 2048
    done
    ;;
  k3-small-rowptr)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll 2048 1
    done
    ;;
  k3-small-combine)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll 2048 0 1
    done
    ;;
  k3-large)
    for tokens in ${TOKENS:-1024 4096}; do
      run_large_token "${tokens}" c 2048
    done
    ;;
  k3-large-rowptr)
    for tokens in ${TOKENS:-1024 4096}; do
      run_large_token "${tokens}" c 2048 1
    done
    ;;
  k3-large-combine)
    for tokens in ${TOKENS:-1024 4096}; do
      run_large_token "${tokens}" c 2048 0 1 "${K3_COPY_STAGE}"
    done
    ;;
  k3-large-copy-stage)
    for tokens in ${TOKENS:-1024 4096}; do
      run_large_token "${tokens}" c 2048 0 1 1
    done
    ;;
  k3)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll 2048
    done
    for tokens in 1024 4096; do
      run_large_token "${tokens}" c 2048
    done
    ;;
  k3-rowptr)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll 2048 1
    done
    for tokens in 1024 4096; do
      run_large_token "${tokens}" c 2048 1
    done
    ;;
  k3-combine)
    for tokens in ${TOKENS:-32 128}; do
      run_small_token "${tokens}" c-ll 2048 0 1
    done
    for tokens in 1024 4096; do
      run_large_token "${tokens}" c 2048 0 1 "${K3_COPY_STAGE}"
    done
    ;;
  layout-check)
    make -C "${V2_DIR}" layout-check
    "${V2_DIR}/pack5_layout_check" --experts 2 --n 4096 --k 4096 --expert 1 --row 3073 --col 777
    "${V2_DIR}/pack5_layout_check" --experts 2 --n 4096 --k 2048 --expert 1 --row 2049 --col 1537
    ;;
  clean)
    make -C "${V2_DIR}" clean
    ;;
  *)
    echo "Expected MODE=build, small, small-symm-stage, large, large-symm-stage, k3-small, k3-small-rowptr, k3-small-combine, k3-large, k3-large-rowptr, k3-large-combine, k3-large-copy-stage, k3, k3-rowptr, k3-combine, layout-check, or clean; got ${MODE}" >&2
    exit 2
    ;;
esac
