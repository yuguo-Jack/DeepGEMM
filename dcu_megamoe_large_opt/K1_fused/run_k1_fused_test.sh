#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

cd "$(dirname "$0")/../.."
source dcu_megamoe_large_opt/K1_fused/env.sh

python3 dcu_megamoe_large_opt/K1_fused/run_with_idle_timeout.py \
  --idle-timeout "${K1_IDLE_TIMEOUT_SECONDS:-180}" \
  -- python3 dcu_megamoe_large_opt/K1_fused/test_k1_symm_fused.py "$@"
