#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_dir"

export MEGAMOE_DCU_ARCH="${MEGAMOE_DCU_ARCH:-gfx938}"
case "$MEGAMOE_DCU_ARCH" in
    gfx936|gfx938) ;;
    *)
        echo "MEGAMOE_DCU_ARCH must be gfx936 or gfx938, got: $MEGAMOE_DCU_ARCH" >&2
        exit 2
        ;;
esac
echo "MegaMoE DCU build target: $MEGAMOE_DCU_ARCH"

python_bin="python"
if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
fi

build_dir="$repo_dir/build"
wheel_dir="$build_dir/whl"

rm -rf dist ./*.egg-info "$build_dir/bdist"
mkdir -p "$wheel_dir"
rm -f "$wheel_dir"/*.whl

"$python_bin" setup.py \
    egg_info --egg-base "$build_dir" \
    build --build-base "$build_dir" --build-lib "$build_dir/lib" \
    bdist_wheel --skip-build --bdist-dir "$build_dir/bdist" --dist-dir "$wheel_dir"

rm -rf dist ./*.egg-info

sync_built_shared_objects() {
    local artifact rel_path target
    shopt -s nullglob
    for artifact in \
        "$build_dir"/lib/megamoe/_C*.so \
        "$build_dir"/lib/megamoe/dcu_megamoe_opt/K1_fused/k1_fused_ext*.so \
        "$build_dir"/lib/megamoe/dcu_megamoe_opt/K2_fused/k2_fused_ext*.so \
        "$build_dir"/lib/megamoe/dcu_megamoe_opt/K3_fused/k3_fused_ext*.so \
        "$build_dir"/lib/megamoe/dcu_megamoe_opt/K3_fused/k3_v3_fused_ext*.so; do
        rel_path="${artifact#"$build_dir"/lib/}"
        target="$repo_dir/$rel_path"
        mkdir -p "$(dirname "$target")"
        if [ ! -e "$target" ] || [ "$artifact" -nt "$target" ]; then
            cp -p "$artifact" "$target"
            echo "synced shared object: $target"
        else
            echo "kept current shared object: $target"
        fi
    done
    shopt -u nullglob
}

sync_built_shared_objects

verify_shared_object() {
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
        echo "verified shared object: $artifact"
    done
}

verify_code_object() {
    local pattern="$1"
    local matches=()
    shopt -s nullglob
    matches=( $pattern )
    shopt -u nullglob
    if [ "${#matches[@]}" -eq 0 ]; then
        echo "missing code object: $pattern" >&2
        exit 1
    fi
    local artifact
    for artifact in "${matches[@]}"; do
        if [ ! -s "$artifact" ]; then
            echo "empty code object: $artifact" >&2
            exit 1
        fi
        local asm_source="${artifact%.co}.s"
        if [ -f "$asm_source" ] && [ "$artifact" -ot "$asm_source" ]; then
            echo "stale code object: $artifact is older than $asm_source" >&2
            exit 1
        fi
        echo "verified code object: $artifact"
    done
}

verify_shared_object "megamoe/_C*.so"
verify_shared_object "megamoe/dcu_megamoe_opt/K1_fused/k1_fused_ext*.so"
verify_shared_object "megamoe/dcu_megamoe_opt/K2_fused/k2_fused_ext*.so"
verify_shared_object "megamoe/dcu_megamoe_opt/K3_fused/k3_fused_ext*.so"
verify_shared_object "megamoe/dcu_megamoe_opt/K3_fused/k3_v3_fused_ext*.so"
verify_code_object "megamoe/dcu_megamoe_opt/K1_fused/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.co"
verify_code_object "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_I8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co"
if [[ "$MEGAMOE_DCU_ARCH" == "gfx938" ]]; then
    verify_code_object "megamoe/dcu_megamoe_opt/K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.co"
    verify_code_object "megamoe/dcu_megamoe_opt/K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_UNIFIED_PACK5.co"
    verify_code_object "megamoe/dcu_megamoe_opt/K1_fused/deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co"
    verify_code_object "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co"
    verify_code_object "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_UNIFIED_PACK5.co"
    verify_code_object "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.co"
    verify_code_object "megamoe/dcu_megamoe_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_UNIFIED_PACK5.co"
fi

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
