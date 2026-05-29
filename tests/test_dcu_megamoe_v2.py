from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "csrc" / "kernels" / "dcu_megamoe_v2"
LAYOUT_PATH = V2_DIR / "layout.py"
STAGES_PATH = V2_DIR / "stages.py"

spec = importlib.util.spec_from_file_location("dcu_megamoe_v2_layout", LAYOUT_PATH)
assert spec is not None and spec.loader is not None
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)

stages_spec = importlib.util.spec_from_file_location("dcu_megamoe_v2_stages", STAGES_PATH)
assert stages_spec is not None and stages_spec.loader is not None
stages = importlib.util.module_from_spec(stages_spec)
stages_spec.loader.exec_module(stages)


def test_pack5_physical_to_logical_order():
    expected = torch.tensor([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
    torch.testing.assert_close(layout.pack5_physical_to_logical_indices(), expected)


def test_pack5_layout_mapping_is_explicit():
    experts, n, k = 2, 512, 128
    weight = torch.arange(experts * n * k, dtype=torch.int32).reshape(experts, n, k)
    packed = layout.pack5_weight(weight)

    assert packed.shape == (experts, 2, 2, 16, 4, 16, 16)
    for e in (0, 1):
        for ko in (0, 1):
            for no in (0, 1):
                for ni16 in (0, 7, 15):
                    for ks in (0, 3):
                        for physical_ni in (0, 5, 15):
                            for ki in (0, 9, 15):
                                logical_ni = (physical_ni & 3) * 4 + (physical_ni >> 2)
                                src_n = no * 256 + ni16 * 16 + logical_ni
                                src_k = ko * 64 + ks * 16 + ki
                                assert packed[e, ko, no, ni16, ks, physical_ni, ki].item() == weight[
                                    e, src_n, src_k
                                ].item()


def test_pack5_roundtrip_cpu():
    weight = torch.arange(3 * 256 * 64, dtype=torch.uint8).reshape(3, 256, 64)
    packed = layout.pack5_weight(weight)
    unpacked = layout.unpack_pack5_weight(packed, n=256, k=64)
    torch.testing.assert_close(unpacked, weight)


def test_pack5_l1_l2_shape_fixtures_use_same_layout():
    probes = {
        (4096, 4096): [(0, 0, 0), (0, 15, 63), (1, 3073, 777)],
        (4096, 2048): [(0, 0, 0), (0, 255, 127), (1, 2049, 1537)],
    }
    for (n, k), coords in probes.items():
        experts = 2
        weight = torch.zeros((experts, n, k), dtype=torch.uint8)
        for value, (expert, row, col) in enumerate(coords, start=1):
            weight[expert, row, col] = value

        packed = layout.pack5_weight(weight)
        assert tuple(packed.shape) == layout.pack5_shape(experts, n, k)
        flat = packed.reshape(-1)
        for value, (expert, row, col) in enumerate(coords, start=1):
            offset = layout.pack5_flat_offset(expert=expert, n=n, k=k, row=row, col=col)
            assert flat[offset].item() == value

        unpacked = layout.unpack_pack5_weight(packed, n=n, k=k)
        for value, (expert, row, col) in enumerate(coords, start=1):
            assert unpacked[expert, row, col].item() == value


def test_pack5_python_mapping_matches_c_helper():
    if shutil.which("make") is None:
        pytest.skip("make is required to build the C pack5 layout helper")
    build = subprocess.run(
        ["make", "-C", str(V2_DIR), "layout-check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if build.returncode != 0:
        pytest.fail(build.stdout + build.stderr)

    helper = V2_DIR / "pack5_layout_check"
    if not helper.exists():
        pytest.skip("C pack5 layout helper was not built")

    probes = [
        (2, 4096, 4096, 1, 3073, 777),
        (2, 4096, 2048, 1, 2049, 1537),
        (1, 256, 64, 0, 15, 63),
    ]
    for experts, n, k, expert, row, col in probes:
        run = subprocess.run(
            [
                str(helper),
                "--experts",
                str(experts),
                "--n",
                str(n),
                "--k",
                str(k),
                "--expert",
                str(expert),
                "--row",
                str(row),
                "--col",
                str(col),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        c_offset = int(run.stdout.strip())
        py_offset = layout.pack5_flat_offset(expert=expert, n=n, k=k, row=row, col=col)
        assert c_offset == py_offset


def test_pack5_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="n divisible by 256"):
        layout.pack5_weight(torch.empty((1, 255, 64), dtype=torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DCU/CUDA device is required")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_k2_swiglu_quant_reuse_matches_bf16_reference():
    torch.manual_seed(20260528)
    rows, hidden = 32, 128
    l1_out = (torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    route_weights = torch.rand((rows,), device="cuda", dtype=torch.float32).contiguous()
    act_fp8 = torch.empty((rows, hidden), device="cuda", dtype=torch.float8_e4m3fn)
    act_scale = torch.empty((rows,), device="cuda", dtype=torch.float32)
    out_bf16 = torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16)

    try:
        stages.swiglu_quant_channelwise_out_v2(
            l1_out,
            route_weights,
            act_fp8,
            act_scale,
            out_bf16,
            num_per_channels=hidden,
            output_bf16=True,
            clamp_value=10.0,
        )
    except Exception as exc:  # pragma: no cover - depends on HIP extension toolchain
        pytest.skip(f"DCU K2 extension is unavailable: {exc}")
    torch.cuda.synchronize()

    ref = stages.swiglu_reference(
        l1_out,
        route_weights,
        num_per_channels=hidden,
        clamp_value=10.0,
    ).to(torch.bfloat16)
    diff = (out_bf16.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    assert max_abs <= 1.0e-3, (max_abs, mean_abs)
    assert torch.isfinite(act_scale).all()
    assert (act_scale > 0).all()
