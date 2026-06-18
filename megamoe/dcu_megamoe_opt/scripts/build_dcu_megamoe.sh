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

rm -rf "$build_dir" dist ./*.egg-info
rm -f megamoe/_C*.so deep_gemm/_C*.so
find megamoe/dcu_megamoe_opt -type f \( -name '*_ext*.so' -o -name '*.co' -o -name '*.o' -o -name '*.hip' \) -delete 2>/dev/null || true
mkdir -p "$wheel_dir"

"$python_bin" setup.py \
    egg_info --egg-base "$build_dir" \
    build --build-base "$build_dir" \
    build_ext --build-temp "$build_dir/temp" --build-lib "$build_dir/lib" --inplace \
    bdist_wheel --bdist-dir "$build_dir/bdist" --dist-dir "$wheel_dir"

rm -rf dist ./*.egg-info

echo "wheel output:"
ls -1 "$wheel_dir"/*.whl
