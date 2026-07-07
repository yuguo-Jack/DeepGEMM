#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"

python_bin="python"
if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
fi

build_dir="$repo_dir/build"
wheel_dir="$build_dir/whl"
build_epoch="$(date +%s)"

rm -rf "$build_dir" dist ./*.egg-info
rm -f megamoe/_C*.so deep_gemm/_C*.so
find megamoe/dcu_megamoe_opt \
    -path '*/prebuilt/*' -prune -o \
    -type f \( -name '*_ext*.so' -o -name '*.co' -o -name '*.o' -o -name '*.hip' \) \
    -delete 2>/dev/null || true
mkdir -p "$wheel_dir"

"$python_bin" setup.py \
    egg_info --egg-base "$build_dir" \
    build --build-base "$build_dir" \
    build_ext --build-temp "$build_dir/temp" --build-lib "$build_dir/lib" --inplace \
    bdist_wheel --bdist-dir "$build_dir/bdist" --dist-dir "$wheel_dir"

rm -rf dist ./*.egg-info

verify_fresh_artifact() {
    local pattern="$1"
    local matches=()
    shopt -s nullglob
    matches=( $pattern )
    shopt -u nullglob
    if [ "${#matches[@]}" -eq 0 ]; then
        echo "missing in-place build artifact: $pattern" >&2
        exit 1
    fi
    local artifact
    for artifact in "${matches[@]}"; do
        if [ ! -s "$artifact" ]; then
            echo "empty in-place build artifact: $artifact" >&2
            exit 1
        fi
        local mtime
        mtime="$(stat -c '%Y' "$artifact")"
        if [ "$mtime" -lt "$build_epoch" ]; then
            echo "stale in-place build artifact: $artifact" >&2
            exit 1
        fi
        echo "verified fresh artifact: $artifact"
    done
}

verify_fresh_artifact "megamoe/_C*.so"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K1_fused/k1_fused_ext*.so"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K2_fused/k2_fused_ext*.so"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K3_fused/k3_fused_ext*.so"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K3_fused/k3_v3_fused_ext*.so"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.co"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_UNIFIED_PACK5.co"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K1_fused/deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_UNIFIED_PACK5.co"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.co"
verify_fresh_artifact "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_UNIFIED_PACK5.co"

"$python_bin" - <<'PY'
import importlib
import pathlib

repo = pathlib.Path.cwd().resolve()
modules = (
    "megamoe",
    "megamoe._C",
    "megamoe.dcu_megamoe_opt.K1_fused.k1_fused_ext",
    "megamoe.dcu_megamoe_opt.K2_fused.k2_fused_ext",
    "megamoe.dcu_megamoe_opt.K3_fused.k3_fused_ext",
    "megamoe.dcu_megamoe_opt.K3_fused.k3_v3_fused_ext",
)
for name in modules:
    mod = importlib.import_module(name)
    path_text = getattr(mod, "__file__", "")
    if not path_text:
        raise SystemExit(f"{name} has no import path")
    path = pathlib.Path(path_text).resolve()
    try:
        rel = path.relative_to(repo)
    except ValueError:
        raise SystemExit(f"{name} imported from outside repo: {path}") from None
    if rel.parts and rel.parts[0] == "build":
        raise SystemExit(f"{name} imported from build directory: {path}")
    print(f"import ok: {name} -> {path}")
PY

echo "wheel output:"
ls -1 "$wheel_dir"/*.whl
