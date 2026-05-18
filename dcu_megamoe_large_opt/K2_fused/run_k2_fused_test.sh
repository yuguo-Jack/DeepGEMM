#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DeepGEMM
source dcu_megamoe_large_opt/K1_fused/env.sh

python3 dcu_megamoe_large_opt/K2_fused/test_k2_fused.py \
  --rows "${ROWS:-16384}" \
  --hidden "${HIDDEN:-2048}" \
  --warmup "${WARMUP:-5}" \
  --repeat "${REPEAT:-20}" \
  "$@"
