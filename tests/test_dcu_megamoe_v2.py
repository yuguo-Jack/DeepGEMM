from __future__ import annotations

import importlib.util
import os
import random
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "csrc" / "kernels" / "dcu_megamoe_v2"
LAYOUT_PATH = V2_DIR / "layout.py"
STAGES_PATH = V2_DIR / "stages.py"
V2_PACKAGE_LAYOUT_PATH = ROOT / "megamoe" / "dcu_megamoe_v2" / "layout.py"
SETUP_PATH = ROOT / "setup.py"

spec = importlib.util.spec_from_file_location("dcu_megamoe_v2_layout", LAYOUT_PATH)
assert spec is not None and spec.loader is not None
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)

stages_spec = importlib.util.spec_from_file_location("dcu_megamoe_v2_stages", STAGES_PATH)
assert stages_spec is not None and stages_spec.loader is not None
stages = importlib.util.module_from_spec(stages_spec)
stages_spec.loader.exec_module(stages)

v2_layout_spec = importlib.util.spec_from_file_location(
    "dcu_megamoe_v2_package_layout",
    V2_PACKAGE_LAYOUT_PATH,
)
assert v2_layout_spec is not None and v2_layout_spec.loader is not None
v2_layout = importlib.util.module_from_spec(v2_layout_spec)
v2_layout_spec.loader.exec_module(v2_layout)


def import_v2_package_or_skip():
    try:
        import megamoe.dcu_megamoe_v2 as v2
    except Exception as exc:
        pytest.skip(f"megamoe.dcu_megamoe_v2 is unavailable in this environment: {exc}")
    return v2


def load_v2_runtime_module():
    from importlib.util import module_from_spec, spec_from_file_location

    runtime_path = ROOT / "megamoe" / "dcu_megamoe_v2" / "runtime.py"
    runtime_spec = spec_from_file_location("dcu_megamoe_v2_runtime", runtime_path)
    assert runtime_spec is not None and runtime_spec.loader is not None
    runtime = module_from_spec(runtime_spec)
    sys.modules[runtime_spec.name] = runtime
    runtime_spec.loader.exec_module(runtime)
    return runtime


def _load_dcu_megamoe_baseline_test_module():
    module_path = ROOT / "tests" / "test_mega_moe_dcu.py"
    spec = importlib.util.spec_from_file_location("dcu_megamoe_baseline_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item for item in raw.replace(",", " ").split() if item]


def _reserve_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cuda_event_bench_ms(fn, *, warmup: int, repeat: int) -> dict[str, float]:
    if warmup < 0 or repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    median = _median_float(times)
    ordered = sorted(times)
    return {
        "median_ms": median,
        "min_ms": min(ordered),
        "max_ms": max(ordered),
    }


def _median_float(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


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


def test_v2_package_backend_contract(monkeypatch):
    v2 = import_v2_package_or_skip()

    monkeypatch.delenv("MEGAMOE_DCU_V2_BACKEND", raising=False)
    assert v2.get_v2_backend() == "ll"
    monkeypatch.setenv("MEGAMOE_DCU_V2_BACKEND", "normal")
    assert v2.get_v2_backend() == "normal"
    monkeypatch.setenv("MEGAMOE_DCU_V2_BACKEND", "ll")
    assert v2.get_v2_backend() == "ll"
    with pytest.raises(ValueError, match="MEGAMOE_DCU_V2_BACKEND"):
        v2.normalize_v2_backend("auto")


def test_v2_package_pack5_matches_prototype_layout():
    v2 = import_v2_package_or_skip()
    weight = torch.arange(2 * 512 * 128, dtype=torch.int32).reshape(2, 512, 128)

    torch.testing.assert_close(v2.pack5_weight(weight), layout.pack5_weight(weight))
    assert v2.pack5_shape(2, 512, 128) == layout.pack5_shape(2, 512, 128)
    assert v2.pack5_flat_offset(expert=1, n=512, k=128, row=257, col=65) == (
        layout.pack5_flat_offset(expert=1, n=512, k=128, row=257, col=65)
    )


def test_v2_package_layout_file_matches_prototype_layout():
    weight = torch.arange(2 * 512 * 128, dtype=torch.int32).reshape(2, 512, 128)

    torch.testing.assert_close(v2_layout.pack5_weight(weight), layout.pack5_weight(weight))
    assert v2_layout.pack5_shape(2, 512, 128) == layout.pack5_shape(2, 512, 128)
    assert v2_layout.pack5_flat_offset(expert=1, n=512, k=128, row=257, col=65) == (
        layout.pack5_flat_offset(expert=1, n=512, k=128, row=257, col=65)
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_weight_transform_api_returns_flattened_pack5():
    torch.manual_seed(20260602)
    l1 = (torch.randn((2, 512, 128), dtype=torch.bfloat16) * 0.02).contiguous()
    l2 = (torch.randn((2, 256, 64), dtype=torch.bfloat16) * 0.02).contiguous()

    (l1_weight, l1_scale), (l2_weight, l2_scale) = (
        v2_layout.transform_fp8_weights_for_mega_moe_v2_pack5(l1, l2)
    )

    assert l1_weight.dtype == torch.float8_e4m3fn
    assert l2_weight.dtype == torch.float8_e4m3fn
    assert l1_weight.shape == (2, 512 * 128)
    assert l2_weight.shape == (2, 256 * 64)
    assert l1_scale.shape == (2, 512)
    assert l2_scale.shape == (2, 256)
    assert l1_weight.is_contiguous()
    assert l2_weight.is_contiguous()
    assert l1_scale.is_contiguous()
    assert l2_scale.is_contiguous()
    l1_packed = l1_weight.reshape(v2_layout.pack5_shape(2, 512, 128))
    l1_repacked = v2_layout.pack5_weight(
        v2_layout.unpack_pack5_weight(l1_packed, n=512, k=128)
    )
    assert torch.equal(l1_packed, l1_repacked)


def test_v2_package_runtime_refuses_unconnected_backend():
    v2 = import_v2_package_or_skip()
    y = torch.empty((1, 1), dtype=torch.bfloat16)
    l1_weights = (torch.empty((1,), dtype=torch.uint8), torch.empty((1,), dtype=torch.float32))
    l2_weights = (torch.empty((1,), dtype=torch.uint8), torch.empty((1,), dtype=torch.float32))

    with pytest.raises(TypeError, match="distributed group"):
        v2.fp8_w8a8_mega_moe_v2(
            y,
            l1_weights,
            l2_weights,
            sym_buffer=object(),
            backend="ll",
        )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_runtime_workspace_views_are_sliced_from_route_scratch():
    runtime = load_v2_runtime_module()
    num_ranks = 8
    num_experts = 256
    num_max_tokens = 512
    num_topk = 6
    hidden = 512
    intermediate_hidden = 256
    route_scratch_bytes = runtime._v2_route_scratch_min_bytes(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )
    route_scratch = torch.empty((route_scratch_bytes,), dtype=torch.int8)
    sym_buffer = SimpleNamespace(
        group=SimpleNamespace(size=lambda: num_ranks, rank=lambda: 0),
        buffer=torch.empty((4096,), dtype=torch.int8),
        route_scratch=route_scratch,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )

    state = runtime._v2_state(
        sym_buffer,
        backend="normal",
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )
    cached = runtime._v2_state(
        sym_buffer,
        backend="normal",
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )
    views = state.scratch

    assert cached is state
    assert views.local_experts == 32
    assert views.valid_rows_per_expert == 96
    assert views.rows_aligned_per_expert == 256
    assert views.launch_rows == 8192
    assert views.grid_barrier_ints == 16 * ((views.launch_rows + 255) // 256) + 2
    assert views.route_scratch_bytes <= route_scratch.numel()
    assert views.l2_workspace.data_ptr() == views.l1_out.data_ptr()
    assert views.m_indices.data_ptr() == views.row_expert.data_ptr()
    assert views.row_combine_ptrs.data_ptr() == views.row_output_ptrs.data_ptr()

    expected = {
        "staged_x": (torch.float8_e4m3fn, (views.capacity_rows, hidden)),
        "staged_x_scale": (torch.float32, (views.capacity_rows,)),
        "route_weights": (torch.float32, (views.capacity_rows,)),
        "l1_out": (torch.bfloat16, (views.l1_out.shape[0], intermediate_hidden * 2)),
        "act_fp8": (torch.float8_e4m3fn, (views.capacity_rows, intermediate_hidden)),
        "act_scale": (torch.float32, (views.capacity_rows,)),
        "problem_size": (torch.int32, (views.local_experts,)),
        "route_scratch_i32": (
            torch.int32,
            (views.local_experts + 2 * views.local_experts * views.rows_aligned_per_expert,),
        ),
        "grid_barrier": (torch.int32, (views.grid_barrier_ints,)),
        "row_expert": (torch.int32, (views.launch_rows,)),
        "m_indices": (torch.int32, (views.launch_rows,)),
        "row_output_ptrs": (torch.int64, (views.launch_rows,)),
        "row_combine_ptrs": (torch.int64, (views.launch_rows,)),
        "output_index": (torch.int32, (num_ranks * num_max_tokens, num_topk)),
        "local_topk_mask": (torch.uint8, (num_max_tokens * num_topk,)),
        "tail_tokens": (torch.int32, (num_max_tokens,)),
    }
    base = route_scratch.data_ptr()
    end = base + route_scratch.numel()
    for name, (dtype, shape) in expected.items():
        tensor = getattr(views, name)
        assert tensor.dtype == dtype
        assert tuple(tensor.shape) == shape
        assert tensor.is_contiguous()
        assert base <= tensor.data_ptr() < end


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_runtime_normal_small_workspace_covers_padded_rows():
    runtime = load_v2_runtime_module()
    num_ranks = 1
    num_experts = 32
    num_max_tokens = 32
    num_topk = 1
    hidden = 4096
    intermediate_hidden = 2048
    route_scratch = torch.empty(
        (
            runtime._v2_route_scratch_min_bytes(
                num_ranks=num_ranks,
                num_experts=num_experts,
                num_max_tokens=num_max_tokens,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                backend="normal",
            ),
        ),
        dtype=torch.int8,
    )
    sym_buffer = SimpleNamespace(
        group=SimpleNamespace(size=lambda: num_ranks, rank=lambda: 0),
        buffer=torch.empty((4096,), dtype=torch.int8),
        route_scratch=route_scratch,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )

    state = runtime._v2_state(
        sym_buffer,
        backend="normal",
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
    )

    views = state.scratch
    assert views.valid_rows_per_expert == 1
    assert views.rows_aligned_per_expert == 256
    assert views.launch_rows == 8192
    assert views.capacity_rows >= 2 * views.launch_rows
    assert views.l1_out.shape[0] >= views.launch_rows
    assert views.act_fp8.shape[0] >= views.launch_rows


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_runtime_calls_k1_k2_k3_launchers():
    runtime = load_v2_runtime_module()
    num_ranks = 8
    num_experts = 256
    num_max_tokens = 512
    num_topk = 6
    hidden = 512
    intermediate_hidden = 256
    route_scratch = torch.empty(
        (
            runtime._v2_route_scratch_min_bytes(
                num_ranks=num_ranks,
                num_experts=num_experts,
                num_max_tokens=num_max_tokens,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
            ),
        ),
        dtype=torch.int8,
    )
    sym_buffer = SimpleNamespace(
        group=SimpleNamespace(size=lambda: num_ranks, rank=lambda: 0),
        buffer=torch.empty((4096,), dtype=torch.int8),
        route_scratch=route_scratch,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        cuda_graph_num_tokens=torch.empty((1,), dtype=torch.int32),
    )
    y = torch.empty((32, hidden), dtype=torch.bfloat16)
    l1_weights = (
        torch.empty((1,), dtype=torch.float8_e4m3fn),
        torch.empty((32, intermediate_hidden * 2), dtype=torch.float32),
    )
    l2_weights = (
        torch.empty((1,), dtype=torch.float8_e4m3fn),
        torch.empty((32, hidden), dtype=torch.float32),
    )

    k1_calls = []
    k2_calls = []
    k3_calls = []

    def fake_k1_launcher(*args, **kwargs):
        k1_calls.append((args, kwargs))
        return args[0]

    def fake_k2_launcher(*args, **kwargs):
        k2_calls.append((args, kwargs))
        return None

    def fake_k3_launcher(*args, **kwargs):
        k3_calls.append((args, kwargs))
        return None

    runtime._K1_STAGE_LAUNCHER = fake_k1_launcher
    runtime._K2_STAGE_LAUNCHER = fake_k2_launcher
    runtime._K3_STAGE_LAUNCHER = fake_k3_launcher

    runtime.run_stages_fused_v2(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        cumulative_local_expert_recv_stats=None,
        activation_clamp=10.0,
        fast_math=True,
        dispatch_num_tokens=None,
        backend="normal",
    )
    assert len(k1_calls) == 1
    assert len(k2_calls) == 1
    assert len(k3_calls) == 1
    _, kwargs = k1_calls[0]
    state = sym_buffer._dcu_megamoe_v2_state[1]
    views = state.scratch
    assert kwargs["backend"] == "normal"
    assert kwargs["staged_x"].data_ptr() == views.staged_x.data_ptr()
    assert kwargs["staged_x_scale"].data_ptr() == views.staged_x_scale.data_ptr()
    assert kwargs["route_weights"].data_ptr() == views.route_weights.data_ptr()
    assert kwargs["row_expert"].data_ptr() == views.row_expert.data_ptr()
    assert kwargs["output_index"].data_ptr() == views.output_index.data_ptr()
    assert kwargs["row_combine_ptrs"].data_ptr() == views.row_combine_ptrs.data_ptr()
    assert kwargs["route_scratch_i32"].data_ptr() == views.route_scratch_i32.data_ptr()
    assert kwargs["grid_barrier"].data_ptr() == views.grid_barrier.data_ptr()
    assert kwargs["rows_aligned_per_expert"] == 256
    assert kwargs["valid_rows_per_expert"] == 96
    assert kwargs["num_ranks"] == num_ranks
    assert kwargs["rank_idx"] == 0
    assert kwargs["epoch"] == 1
    assert kwargs["num_tokens"] == 32

    k2_args, k2_kwargs = k2_calls[0]
    assert k2_args[0].data_ptr() == views.l1_out.data_ptr()
    assert k2_args[0].shape == (views.launch_rows, intermediate_hidden * 2)
    assert k2_args[1].data_ptr() == views.route_weights.data_ptr()
    assert k2_args[2].data_ptr() == views.act_fp8.data_ptr()
    assert k2_args[2].shape == (views.launch_rows, intermediate_hidden)
    assert k2_args[3].data_ptr() == views.act_scale.data_ptr()
    assert k2_args[4].numel() == 0
    assert k2_kwargs["num_per_channels"] == intermediate_hidden
    assert not k2_kwargs["output_bf16"]
    assert k2_kwargs["clamp_value"] == 10.0
    assert k2_kwargs["row_combine_ptrs"].data_ptr() == views.row_combine_ptrs.data_ptr()

    k3_args, k3_kwargs = k3_calls[0]
    assert k3_args[0].data_ptr() == y.data_ptr()
    assert k3_args[1] is l2_weights
    assert k3_args[2].data_ptr() == views.act_fp8.data_ptr()
    assert k3_args[2].shape == (views.launch_rows, intermediate_hidden)
    assert k3_args[3].data_ptr() == views.act_scale.data_ptr()
    assert k3_args[4] is sym_buffer
    assert k3_kwargs["route_scratch"].data_ptr() == route_scratch.data_ptr()
    assert k3_kwargs["l2_workspace"].data_ptr() == views.l2_workspace.data_ptr()
    assert k3_kwargs["problem_size"].data_ptr() == views.route_scratch_i32.data_ptr()
    assert k3_kwargs["row_expert"].data_ptr() == views.row_expert.data_ptr()
    assert k3_kwargs["grid_barrier"].data_ptr() == views.grid_barrier.data_ptr()
    assert k3_kwargs["row_output_ptrs"].data_ptr() == views.row_output_ptrs.data_ptr()
    assert k3_kwargs["local_topk_mask"].data_ptr() == views.local_topk_mask.data_ptr()
    assert k3_kwargs["tail_tokens"].data_ptr() == views.tail_tokens.data_ptr()
    assert k3_kwargs["backend"] == "normal"
    assert k3_kwargs["epoch"] == 2
    assert k3_kwargs["k3_copy_workers"] == runtime.K3_COPY_WORKERS
    assert k3_kwargs["num_tokens"] == 32
    assert state.epoch == 3

    k1_calls.clear()
    k2_calls.clear()
    k3_calls.clear()
    runtime.run_stages_fused_v2(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        cumulative_local_expert_recv_stats=None,
        activation_clamp=10.0,
        fast_math=True,
        dispatch_num_tokens=-1,
        backend="normal",
    )
    _, uneven_k1_kwargs = k1_calls[0]
    _, uneven_k3_kwargs = k3_calls[0]
    assert uneven_k1_kwargs["num_tokens"] == -1
    assert uneven_k3_kwargs["num_tokens"] == -1


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _dcu_input_token_offset(num_ranks: int) -> int:
    signal_ptrs_offset = _align(num_ranks * 8, 16)
    workspace_offset = _align(signal_ptrs_offset + num_ranks * 8, 16)
    runtime_num_tokens_offset = workspace_offset
    uniform_num_tokens_offset = runtime_num_tokens_offset + 4
    return _align(uniform_num_tokens_offset + 4, 16)


def _combine_token_offset(
    *,
    num_ranks: int,
    num_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
) -> int:
    input_token_offset = _dcu_input_token_offset(num_ranks)
    input_sf_offset = input_token_offset + num_max_tokens * hidden
    topk_idx_offset = input_sf_offset + num_max_tokens * 4
    topk_weights_offset = topk_idx_offset + num_max_tokens * num_topk * 8
    return topk_weights_offset + num_max_tokens * num_topk * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DCU/CUDA device is required")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
@pytest.mark.parametrize(
    ("backend", "rows_aligned_per_expert"),
    [("ll", 64), ("normal", 256)],
)
def test_v2_k1_writes_route_metadata_from_sym_buffer(backend, rows_aligned_per_expert):
    from megamoe.dcu_megamoe_v2.K1_fused.k1_fused import (
        k1_dispatch_pull_l1_fused_v2,
    )

    num_ranks = 1
    rank_idx = 0
    local_experts = 32
    num_global_experts = 32
    num_max_tokens = 32
    num_topk = 1
    hidden = 4096
    l1_rows = 4096
    valid_rows_per_expert = 1
    launch_rows = local_experts * rows_aligned_per_expert
    combine_offset = _combine_token_offset(
        num_ranks=num_ranks,
        num_experts=num_global_experts,
        num_max_tokens=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
    )
    sym_buffer_bytes = _align(
        combine_offset + num_topk * num_max_tokens * hidden * 2,
        16,
    )

    sym_storage = torch.empty((sym_buffer_bytes,), dtype=torch.int8, device="cuda")
    sym_storage.zero_()
    sym_storage[:8].view(torch.int64).copy_(
        torch.tensor([sym_storage.data_ptr()], dtype=torch.int64, device="cuda")
    )
    runtime_offset = _dcu_input_token_offset(num_ranks) - 16
    sym_storage[runtime_offset:runtime_offset + 4].view(torch.int32).fill_(num_max_tokens)
    sym_storage[runtime_offset + 4:runtime_offset + 8].view(torch.int32).fill_(num_max_tokens)

    input_offset = _dcu_input_token_offset(num_ranks)
    x = sym_storage[input_offset:input_offset + num_max_tokens * hidden].view(
        torch.float8_e4m3fn
    ).view(num_max_tokens, hidden)
    x.copy_(torch.ones_like(x))
    x_scale_offset = input_offset + num_max_tokens * hidden
    x_scale = sym_storage[x_scale_offset:x_scale_offset + num_max_tokens * 4].view(
        torch.float32
    )
    x_scale.copy_(torch.linspace(0.01, 0.02, num_max_tokens, device="cuda"))
    topk_idx_offset = x_scale_offset + num_max_tokens * 4
    topk_idx = sym_storage[
        topk_idx_offset:topk_idx_offset + num_max_tokens * num_topk * 8
    ].view(torch.int64)
    topk_idx.copy_(torch.arange(num_max_tokens, dtype=torch.int64, device="cuda"))
    topk_weights_offset = topk_idx_offset + num_max_tokens * num_topk * 8
    topk_weights = sym_storage[
        topk_weights_offset:topk_weights_offset + num_max_tokens * num_topk * 4
    ].view(torch.float32)
    expected_weights = torch.linspace(0.125, 0.875, num_max_tokens, device="cuda")
    topk_weights.copy_(expected_weights)

    fake_sym_buffer = SimpleNamespace(buffer=sym_storage)
    l1_out = torch.empty((launch_rows, l1_rows), dtype=torch.bfloat16, device="cuda")
    staged_x = torch.empty((launch_rows, hidden), dtype=torch.float8_e4m3fn, device="cuda")
    staged_x_scale = torch.empty((launch_rows,), dtype=torch.float32, device="cuda")
    weight = torch.zeros(
        (local_experts, l1_rows * hidden),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    weight_scale = torch.ones((local_experts, l1_rows), dtype=torch.float32, device="cuda")
    problem_size = torch.full((local_experts,), valid_rows_per_expert, dtype=torch.int32, device="cuda")
    route_scratch_i32 = torch.empty(
        (local_experts + 2 * local_experts * rows_aligned_per_expert,),
        dtype=torch.int32,
        device="cuda",
    )
    grid_barrier_ints = (
        2
        if backend == "ll"
        else 2 + 2 * ((launch_rows + 255) // 256)
    )
    grid_barrier = torch.zeros((grid_barrier_ints,), dtype=torch.int32, device="cuda")
    route_weights = torch.empty((launch_rows,), dtype=torch.float32, device="cuda")
    row_expert = torch.empty((launch_rows,), dtype=torch.int32, device="cuda")
    output_index = torch.empty((num_ranks * num_max_tokens, num_topk), dtype=torch.int32, device="cuda")
    row_combine_ptrs = torch.empty((launch_rows,), dtype=torch.int64, device="cuda")
    local_topk_mask = torch.empty((num_max_tokens * num_topk,), dtype=torch.uint8, device="cuda")
    tail_tokens = torch.empty((num_max_tokens,), dtype=torch.int32, device="cuda")
    stats_initial = torch.arange(local_experts, dtype=torch.int32, device="cuda")
    cumulative_stats = stats_initial.clone()

    k1_dispatch_pull_l1_fused_v2(
        l1_out,
        (weight, weight_scale),
        fake_sym_buffer,
        route_scratch=sym_storage,
        staged_x=staged_x,
        staged_x_scale=staged_x_scale,
        problem_size=problem_size,
        row_expert=row_expert,
        route_weights=route_weights,
        output_index=output_index,
        row_combine_ptrs=row_combine_ptrs,
        local_topk_mask=local_topk_mask,
        tail_tokens=tail_tokens,
        grid_barrier=grid_barrier,
        route_scratch_i32=route_scratch_i32,
        cumulative_local_expert_recv_stats=cumulative_stats,
        num_tokens=num_max_tokens,
        num_ranks=num_ranks,
        num_global_experts=num_global_experts,
        num_max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        rank_idx=rank_idx,
        rows_aligned_per_expert=rows_aligned_per_expert,
        valid_rows_per_expert=valid_rows_per_expert,
        backend=backend,
    )
    torch.cuda.synchronize()

    rows = torch.arange(num_max_tokens, dtype=torch.long) * rows_aligned_per_expert
    torch.testing.assert_close(route_weights[rows].cpu(), expected_weights.cpu())
    torch.testing.assert_close(row_expert[rows].cpu(), torch.arange(num_max_tokens, dtype=torch.int32))
    torch.testing.assert_close(output_index.flatten().cpu(), rows.to(torch.int32).cpu())
    torch.testing.assert_close(local_topk_mask[:num_max_tokens].cpu(), torch.ones((num_max_tokens,), dtype=torch.uint8))
    torch.testing.assert_close(tail_tokens.cpu(), torch.arange(num_max_tokens, dtype=torch.int32))
    torch.testing.assert_close(
        cumulative_stats.cpu(),
        (stats_initial + 1).cpu(),
    )

    expected_ptrs = (
        sym_storage.data_ptr()
        + combine_offset
        + torch.arange(num_max_tokens, dtype=torch.int64) * hidden * 2
    )
    torch.testing.assert_close(row_combine_ptrs[rows].cpu(), expected_ptrs)


def test_v2_extension_sources_export_c_pack5_launchers():
    k1_source = (ROOT / "megamoe" / "dcu_megamoe_v2" / "K1_fused" / "k1_fused_ext.cu").read_text()
    k3_source = (ROOT / "megamoe" / "dcu_megamoe_v2" / "K3_fused" / "k3_fused_ext.cu").read_text()
    k1_pybind = (ROOT / "megamoe" / "dcu_megamoe_v2" / "K1_fused" / "k1_fused_pybind.cpp").read_text()
    k3_pybind = (ROOT / "megamoe" / "dcu_megamoe_v2" / "K3_fused" / "k3_fused_pybind.cpp").read_text()
    k1_python = (ROOT / "megamoe" / "dcu_megamoe_v2" / "K1_fused" / "k1_fused.py").read_text()
    k3_python = (ROOT / "megamoe" / "dcu_megamoe_v2" / "K3_fused" / "k3_fused.py").read_text()
    k1_kernel_source = (ROOT / "csrc" / "kernels" / "dcu_megamoe_v2" / "k1_groupgemm_v2.cpp").read_text()

    assert "launch_k1_ll_symm_stage" in k1_source
    assert "launch_k1_normal_symm_stage" in k1_source
    assert "launch_k3_ll_rowptr_tail_reduce" in k3_source
    assert "launch_k3_normal_copy_stage_tail_reduce" in k3_source
    assert "PYBIND11_MODULE" in k1_pybind
    assert "PYBIND11_MODULE" in k3_pybind
    assert "torch/extension.h" not in k1_source
    assert "torch/extension.h" not in k3_source
    assert "k1_not_connected" not in k1_source
    assert "k3_not_connected" not in k3_source
    assert "DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN" in k1_source
    assert "DCU_MEGAMOE_V2_DISABLE_STANDALONE_MAIN" in k3_source
    assert "DCU_MEGAMOE_V2_KERNEL_ONLY" in k1_source
    assert "DCU_MEGAMOE_V2_KERNEL_ONLY" in k3_source
    assert "torch.utils.cpp_extension" not in k1_python
    assert "torch.utils.cpp_extension" not in k3_python
    assert "route_weights_out" in k1_kernel_source
    assert "row_combine_ptrs_out" in k1_kernel_source
    assert "local_topk_mask" in k1_pybind
    assert "tail_tokens" in k1_pybind
    assert "runtime_num_tokens" in k3_pybind
    assert "tail_token_count" in k3_pybind
    assert "route_weights, row_expert, output_index" in k1_python
    assert "route_scratch_i32 is too small for K1 normal metadata" in k1_pybind


def test_v2_extensions_are_registered_in_setup_build():
    setup_source = SETUP_PATH.read_text()

    assert "megamoe.dcu_megamoe_v2" in setup_source
    assert "megamoe.dcu_megamoe_v2.K1_fused" in setup_source
    assert "megamoe.dcu_megamoe_v2.K2_fused" in setup_source
    assert "megamoe.dcu_megamoe_v2.K3_fused" in setup_source
    assert "megamoe.dcu_megamoe_v2.K1_fused.k1_fused_ext" in setup_source
    assert "megamoe.dcu_megamoe_v2.K3_fused.k3_fused_ext" in setup_source
    assert "k1_fused_pybind.cpp" in setup_source
    assert "k3_fused_pybind.cpp" in setup_source


def test_v2_runtime_stage_plan_mapping():
    runtime = load_v2_runtime_module()

    ll = runtime.get_v2_stage_plan("ll")
    assert ll.k1_kernel_family == "low_latency_c_pack5"
    assert ll.k1_problem_k == 4096
    assert ll.k3_kernel_family == "low_latency_c_pack5"
    assert ll.k3_problem_k == 2048
    assert not ll.k3_uses_copy_stage
    assert ll.k3_uses_tail_reduce

    normal = runtime.get_v2_stage_plan("normal")
    assert normal.k1_kernel_family == "normal_c_pack5"
    assert normal.k1_problem_k == 4096
    assert normal.k3_kernel_family == "normal_c_pack5"
    assert normal.k3_problem_k == 2048
    assert normal.k3_uses_copy_stage
    assert normal.k3_uses_tail_reduce


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


def _ensure_v2_route_scratch_capacity(sym_buffer, backend: str) -> None:
    from megamoe.dcu_megamoe_v2 import runtime as v2_runtime

    required = v2_runtime._v2_route_scratch_min_bytes(
        num_ranks=int(sym_buffer.group.size()),
        num_experts=int(sym_buffer.num_experts),
        num_max_tokens=int(sym_buffer.num_max_tokens_per_rank),
        num_topk=int(sym_buffer.num_topk),
        hidden=int(sym_buffer.hidden),
        intermediate_hidden=int(sym_buffer.intermediate_hidden),
        backend=backend,
    )
    if sym_buffer.route_scratch.numel() < required:
        sym_buffer.route_scratch = torch.empty(
            (required,),
            dtype=torch.int8,
            device=sym_buffer.route_scratch.device,
        )


def _make_local_only_topk(
    *,
    num_tokens: int,
    num_topk: int,
    rank: int,
    local_experts: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.arange(num_tokens * num_topk, device=device, dtype=torch.int64)
    local = (local % local_experts).view(num_tokens, num_topk)
    topk_idx = local + rank * local_experts
    raw = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device) * 0.1
    return topk_idx.contiguous(), torch.softmax(raw, dim=-1).contiguous()


def _make_real_flow_topk(
    *,
    num_tokens: int,
    num_topk: int,
    rank: int,
    num_ranks: int,
    local_experts: int,
    route_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_mode == "local_only":
        return _make_local_only_topk(
            num_tokens=num_tokens,
            num_topk=num_topk,
            rank=rank,
            local_experts=local_experts,
            device=device,
        )
    if route_mode != "cross_rank":
        raise ValueError(f"unknown V2 real-flow route_mode: {route_mode}")
    token = torch.arange(num_tokens, device=device, dtype=torch.int64).view(-1, 1)
    slot = torch.arange(num_topk, device=device, dtype=torch.int64).view(1, -1)
    target_rank = (rank + slot + 1) % num_ranks
    expert_in_rank = (token * num_topk + slot) % local_experts
    topk_idx = target_rank * local_experts + expert_in_rank
    raw = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device) * 0.1
    return topk_idx.contiguous(), torch.softmax(raw, dim=-1).contiguous()


def _diff_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | int]:
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": diff.max().item() if diff.numel() else 0.0,
        "mean_abs": diff.mean().item() if diff.numel() else 0.0,
        "mismatch": int((diff > 1.0e-3).sum().item()) if diff.numel() else 0,
    }


def _v2_real_flow_stage_metrics(
    *,
    baseline,
    y_v2: torch.Tensor,
    sym_buffer,
    baseline_l1,
    baseline_l2,
    token_count: int,
    num_ranks: int,
    rank: int,
    num_topk: int,
    intermediate_hidden: int,
    hidden: int,
    activation_clamp: float,
) -> dict[str, dict[str, float | int]]:
    cached = getattr(sym_buffer, "_dcu_megamoe_v2_state", None)
    if cached is None:
        raise RuntimeError("V2 state is missing after real-flow execution")
    scratch = cached[1].scratch
    rows = int(scratch.launch_rows)
    active_rows = scratch.row_combine_ptrs[:rows] != 0
    if not bool(active_rows.any().item()):
        raise RuntimeError("V2 stage metrics found no active rows")
    m_indices = scratch.row_expert[:rows].contiguous()

    ref_l1 = torch.empty_like(scratch.l1_out[:rows])
    baseline.deepgemm.m_grouped_fp8_gemm_nt_contiguous(
        (
            scratch.staged_x[:rows],
            scratch.staged_x_scale[:rows],
        ),
        baseline_l1,
        ref_l1,
        m_indices,
    )

    from megamoe.dcu_megamoe_v2.K2_fused import (
        swiglu_quant_channelwise_out_v2,
        swiglu_reference,
    )

    k2_ref = swiglu_reference(
        scratch.l1_out[:rows],
        scratch.route_weights[:rows],
        num_per_channels=intermediate_hidden,
        clamp_value=activation_clamp,
    ).to(torch.bfloat16)
    k2_bf16 = torch.empty_like(k2_ref)
    tmp_act_fp8 = torch.empty_like(scratch.act_fp8[:rows])
    tmp_act_scale = torch.empty_like(scratch.act_scale[:rows])
    swiglu_quant_channelwise_out_v2(
        scratch.l1_out[:rows],
        scratch.route_weights[:rows],
        tmp_act_fp8,
        tmp_act_scale,
        k2_bf16,
        num_per_channels=intermediate_hidden,
        output_bf16=True,
        clamp_value=activation_clamp,
        row_combine_ptrs=None,
    )

    ref_l2 = torch.empty((rows, hidden), dtype=torch.bfloat16, device=y_v2.device)
    baseline.deepgemm.m_grouped_fp8_gemm_nt_contiguous(
        (
            scratch.act_fp8[:rows],
            scratch.act_scale[:rows],
        ),
        baseline_l2,
        ref_l2,
        m_indices,
    )
    output_index = scratch.output_index.view(
        num_ranks,
        int(sym_buffer.num_max_tokens_per_rank),
        num_topk,
    )[rank, :token_count]
    valid = output_index >= 0
    ref_y = torch.zeros((token_count, hidden), dtype=torch.bfloat16, device=y_v2.device)
    if bool(valid.any().item()):
        token_ids = torch.arange(token_count, device=y_v2.device).view(-1, 1)
        token_ids = token_ids.expand(-1, num_topk)
        ref_y.index_add_(
            0,
            token_ids[valid].reshape(-1),
            ref_l2[output_index[valid].to(torch.long)],
        )

    return {
        "k1_l1_diagnostic": _diff_metrics(
            scratch.l1_out[:rows][active_rows],
            ref_l1[active_rows],
        ),
        "k2_swiglu": _diff_metrics(k2_bf16[active_rows], k2_ref[active_rows]),
        "k3_l2_reduce": _diff_metrics(y_v2[:token_count], ref_y),
    }


def _v2_real_flow_worker(
    local_rank: int,
    num_local_ranks: int,
    token_count: int,
    backend: str,
    token_counts_per_rank: tuple[int, ...] | None = None,
    route_mode: str = "local_only",
) -> None:
    baseline = _load_dcu_megamoe_baseline_test_module()
    import megamoe
    import megamoe.dcu_megamoe_v2 as v2

    os.environ["MEGAMOE_DCU_USE_LARGE_OPT_3STAGE"] = "0"
    rank, num_ranks, group = baseline.init_dist(local_rank, num_local_ranks)
    torch.manual_seed(20260603 + rank)
    random.seed(20260603 + rank)
    uneven_tokens = token_counts_per_rank is not None
    if uneven_tokens:
        if len(token_counts_per_rank) != num_ranks:
            raise RuntimeError("token_counts_per_rank must match num_ranks")
        local_token_count = int(token_counts_per_rank[rank])
        max_token_count = max(int(item) for item in token_counts_per_rank)
    else:
        local_token_count = int(token_count)
        max_token_count = int(token_count)

    hidden = 4096
    intermediate_hidden = 2048
    local_experts = 32
    num_experts = num_ranks * local_experts
    num_topk = 6
    activation_clamp = 10.0
    sym_buffer = None
    ep_buffer = None
    try:
        sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            max_token_count,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        _ensure_v2_route_scratch_capacity(sym_buffer, backend)
        if uneven_tokens:
            sym_buffer.cuda_graph_num_tokens.fill_(local_token_count)
        ep_buffer = baseline.deep_ep.Buffer(
            group,
            baseline.DEEPEP_BUFFER_BYTES,
            0,
            explicitly_destroy=True,
        )
        ep_config = baseline.deep_ep.Config(*baseline.DEEPEP_CONFIG)

        x_bf16 = (
            torch.randn((local_token_count, hidden), dtype=torch.bfloat16, device="cuda")
            * 0.02
        ).contiguous()
        l1_bf16 = (
            torch.randn(
                (local_experts, intermediate_hidden * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        ).contiguous()
        l2_bf16 = (
            torch.randn(
                (local_experts, hidden, intermediate_hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        ).contiguous()
        topk_idx, topk_weights = _make_real_flow_topk(
            num_tokens=local_token_count,
            num_topk=num_topk,
            rank=rank,
            num_ranks=num_ranks,
            local_experts=local_experts,
            route_mode=route_mode,
            device=x_bf16.device,
        )
        x_fp8, x_scale = megamoe.cast_to_fp8_channelwise(x_bf16)
        baseline_l1, baseline_l2 = megamoe.transform_fp8_weights_for_mega_moe(
            l1_bf16, l2_bf16
        )
        v2_l1, v2_l2 = v2.transform_fp8_weights_for_mega_moe_v2_pack5(
            l1_bf16, l2_bf16
        )

        sym_buffer.x[:local_token_count].copy_(x_fp8)
        sym_buffer.x_sf[:local_token_count].copy_(x_scale)
        sym_buffer.topk_idx[:local_token_count].copy_(topk_idx)
        sym_buffer.topk_weights[:local_token_count].copy_(topk_weights)

        y_v2 = torch.empty(
            (int(sym_buffer.num_max_tokens_per_rank), hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        stats_initial = torch.arange(local_experts, dtype=torch.int32, device="cuda")
        stats_v2 = stats_initial.clone()
        v2.fp8_w8a8_mega_moe_v2(
            y_v2,
            v2_l1,
            v2_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=stats_v2,
            activation_clamp=activation_clamp,
            fast_math=True,
            dispatch_num_tokens=-1 if uneven_tokens else local_token_count,
            backend=backend,
        )
        stage_metrics = {}
        if local_token_count <= 32 and route_mode == "local_only":
            stage_metrics = _v2_real_flow_stage_metrics(
                baseline=baseline,
                y_v2=y_v2,
                sym_buffer=sym_buffer,
                baseline_l1=baseline_l1,
                baseline_l2=baseline_l2,
                token_count=local_token_count,
                num_ranks=num_ranks,
                rank=rank,
                num_topk=num_topk,
                intermediate_hidden=intermediate_hidden,
                hidden=hidden,
                activation_clamp=activation_clamp,
            )
        baseline_y, baseline_counts, meta = baseline.run_deepgemm_megamoe_baseline(
            ep_buffer,
            ep_config,
            x_fp8,
            x_scale,
            topk_idx,
            topk_weights,
            baseline_l1,
            baseline_l2,
            num_experts,
            local_experts,
            intermediate_hidden,
            hidden,
            activation_clamp,
            baseline.DEEPEP_EXPERT_ALIGNMENT,
            "hip",
            layout_cache=None,
            return_stats=True,
        )
        torch.cuda.synchronize()
        expected_stats = stats_initial + baseline_counts
        stats_ok = bool(torch.equal(stats_v2, expected_stats))

        diff = (y_v2[:local_token_count].float() - baseline_y.float()).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        mismatch = int((diff > 1.0e-3).sum().item()) if diff.numel() else 0
        nan_count = int(torch.isnan(diff).sum().item()) if diff.numel() else 0
        reliable_stage_metrics = {
            name: metrics
            for name, metrics in stage_metrics.items()
            if name != "k1_l1_diagnostic"
        }
        stage_ok = all(
            float(metrics["max_abs"]) <= 1.0e-3 and int(metrics["mismatch"]) == 0
            for metrics in reliable_stage_metrics.values()
        )
        if rank == 0:
            print(
                {
                    "backend": backend,
                    "tokens": local_token_count,
                    "max_tokens": max_token_count,
                    "tokens_per_rank": list(token_counts_per_rank) if uneven_tokens else None,
                    "ranks": num_ranks,
                    "route_mode": route_mode,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "mismatch": mismatch,
                    "stats_ok": stats_ok,
                    "stage_ok": stage_ok,
                    "stage_metrics": stage_metrics,
                    "baseline_meta": meta,
                },
                flush=True,
            )
        if not (max_abs <= 1.0e-3 and stats_ok and stage_ok):
            row_max = diff.max(dim=1).values if diff.numel() else torch.empty((0,), device="cuda")
            bad_token_idx = torch.nonzero(row_max > 1.0e-3, as_tuple=False).flatten()
            cached = getattr(sym_buffer, "_dcu_megamoe_v2_state", None)
            k1_meta = {}
            if cached is not None:
                scratch = cached[1].scratch
                rows = int(scratch.launch_rows)
                counts = scratch.route_scratch_i32[: scratch.local_experts].detach().cpu()
                row_ptrs = scratch.row_combine_ptrs[:rows].detach().cpu()
                route_weights = scratch.route_weights[:rows].detach().cpu()
                row_expert = scratch.row_expert[:rows].detach().cpu()
                output_index = scratch.output_index.detach().cpu()
                local_topk_mask = scratch.local_topk_mask[:local_token_count].detach().cpu()
                k1_meta = {
                    "launch_rows": rows,
                    "rows_aligned_per_expert": int(scratch.rows_aligned_per_expert),
                    "valid_rows_per_expert": int(scratch.valid_rows_per_expert),
                    "count_min": int(counts.min().item()) if counts.numel() else 0,
                    "count_max": int(counts.max().item()) if counts.numel() else 0,
                    "count_sum": int(counts.sum().item()) if counts.numel() else 0,
                    "count_head": [int(x) for x in counts[:8].tolist()],
                    "nonnull_row_ptrs": int((row_ptrs != 0).sum().item()),
                    "nonzero_route_weights": int((route_weights != 0).sum().item()),
                    "output_index_nonneg": int((output_index >= 0).sum().item()),
                    "row_expert_min": int(row_expert.min().item()) if row_expert.numel() else -1,
                    "row_expert_max": int(row_expert.max().item()) if row_expert.numel() else -1,
                    "local_mask_nonzero": int((local_topk_mask != 0).sum().item()),
                    "local_mask_unique": [
                        int(x) for x in torch.unique(local_topk_mask).detach().cpu().tolist()
                    ],
                }
            bad_head = bad_token_idx[:16].detach().cpu().to(torch.int64).tolist()
            bad_topk = []
            bad_routes = []
            if bad_head:
                bad_topk = topk_idx[bad_token_idx[:8]].detach().cpu().to(torch.int64).tolist()
            if cached is not None:
                scratch = cached[1].scratch
                num_max_tokens = int(sym_buffer.num_max_tokens_per_rank)
                rows_aligned = int(scratch.rows_aligned_per_expert)
                src_rank_offset = scratch.local_experts
                src_token_offset = src_rank_offset + scratch.local_experts * rows_aligned
                output_index_cpu = scratch.output_index.detach().cpu()
                src_ranks_cpu = scratch.route_scratch_i32[
                    src_rank_offset : src_rank_offset + scratch.local_experts * rows_aligned
                ].detach().cpu()
                src_tokens_cpu = scratch.route_scratch_i32[
                    src_token_offset : src_token_offset + scratch.local_experts * rows_aligned
                ].detach().cpu()
                route_weights_cpu = scratch.route_weights[: scratch.launch_rows].detach().cpu()
                row_ptrs_cpu = scratch.row_combine_ptrs[: scratch.launch_rows].detach().cpu()
                staged_scale_cpu = scratch.staged_x_scale[: scratch.launch_rows].detach().cpu()
                l1_abs_cpu = (
                    scratch.l1_out[: scratch.launch_rows].float().abs().amax(dim=1).detach().cpu()
                )
                for token_idx in bad_head[:4]:
                    for topk_slot in range(num_topk):
                        row = int(output_index_cpu[rank * num_max_tokens + token_idx, topk_slot].item())
                        entry = {
                            "token": int(token_idx),
                            "slot": int(topk_slot),
                            "expert": int(topk_idx[token_idx, topk_slot].detach().cpu().item()),
                            "row": row,
                        }
                        if 0 <= row < int(scratch.launch_rows):
                            entry.update(
                                {
                                    "row_in_expert": int(row % rows_aligned),
                                    "src_rank": int(src_ranks_cpu[row].item()),
                                    "src_token": int(src_tokens_cpu[row].item()),
                                    "route_weight": float(route_weights_cpu[row].item()),
                                    "row_ptr_nonzero": bool(row_ptrs_cpu[row].item() != 0),
                                    "staged_scale": float(staged_scale_cpu[row].item()),
                                    "l1_absmax": float(l1_abs_cpu[row].item()),
                                }
                            )
                        bad_routes.append(entry)
            print(
                {
                    "backend": backend,
                    "tokens": local_token_count,
                    "max_tokens": max_token_count,
                    "tokens_per_rank": list(token_counts_per_rank) if uneven_tokens else None,
                    "rank": rank,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "mismatch": mismatch,
                    "nan_count": nan_count,
                    "stats_ok": stats_ok,
                    "stage_ok": stage_ok,
                    "stage_metrics": stage_metrics,
                    "stats_v2_head": [
                        int(x) for x in stats_v2.detach().cpu()[:8].tolist()
                    ],
                    "expected_stats_head": [
                        int(x) for x in expected_stats.detach().cpu()[:8].tolist()
                    ],
                    "bad_tokens": int((row_max > 1.0e-3).sum().item()) if row_max.numel() else 0,
                    "bad_token_head": [int(x) for x in bad_head],
                    "bad_topk_head": bad_topk,
                    "bad_routes": bad_routes,
                    "baseline_meta": meta,
                    "k1_meta": k1_meta,
                    "route_mode": route_mode,
                },
                flush=True,
            )
        assert max_abs <= 1.0e-3, (
            backend,
            local_token_count,
            max_token_count,
            max_abs,
            mean_abs,
            mismatch,
        )
        assert stats_ok, (
            backend,
            local_token_count,
            max_token_count,
            stats_v2.detach().cpu().tolist(),
            expected_stats.detach().cpu().tolist(),
        )
        assert stage_ok, (backend, local_token_count, max_token_count, stage_metrics)
    finally:
        if sym_buffer is not None:
            sym_buffer.destroy()
        if ep_buffer is not None:
            ep_buffer.destroy()
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier(group=group)
            dist.destroy_process_group()


def _v2_real_flow_perf_worker(
    local_rank: int,
    num_local_ranks: int,
    token_count: int,
    backend: str,
    warmup: int,
    repeat: int,
    route_mode: str,
) -> None:
    baseline = _load_dcu_megamoe_baseline_test_module()
    import megamoe
    import megamoe.dcu_megamoe_v2 as v2
    import torch.distributed as dist

    os.environ["MEGAMOE_DCU_USE_LARGE_OPT_3STAGE"] = "0"
    rank, num_ranks, group = baseline.init_dist(local_rank, num_local_ranks)
    torch.manual_seed(20260603 + rank)
    random.seed(20260603 + rank)

    hidden = 4096
    intermediate_hidden = 2048
    local_experts = 32
    num_experts = num_ranks * local_experts
    num_topk = 6
    activation_clamp = 10.0
    sym_buffer = None
    ep_buffer = None
    try:
        sym_buffer = megamoe.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            token_count,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        _ensure_v2_route_scratch_capacity(sym_buffer, backend)
        ep_buffer = baseline.deep_ep.Buffer(
            group,
            baseline.DEEPEP_BUFFER_BYTES,
            0,
            explicitly_destroy=True,
        )
        ep_config = baseline.deep_ep.Config(*baseline.DEEPEP_CONFIG)

        x_bf16 = (
            torch.randn((token_count, hidden), dtype=torch.bfloat16, device="cuda") * 0.02
        ).contiguous()
        l1_bf16 = (
            torch.randn(
                (local_experts, intermediate_hidden * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        ).contiguous()
        l2_bf16 = (
            torch.randn(
                (local_experts, hidden, intermediate_hidden),
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.02
        ).contiguous()
        topk_idx, topk_weights = _make_real_flow_topk(
            num_tokens=token_count,
            num_topk=num_topk,
            rank=rank,
            num_ranks=num_ranks,
            local_experts=local_experts,
            route_mode=route_mode,
            device=x_bf16.device,
        )
        x_fp8, x_scale = megamoe.cast_to_fp8_channelwise(x_bf16)
        baseline_l1, baseline_l2 = megamoe.transform_fp8_weights_for_mega_moe(
            l1_bf16, l2_bf16
        )
        v2_l1, v2_l2 = v2.transform_fp8_weights_for_mega_moe_v2_pack5(
            l1_bf16, l2_bf16
        )

        sym_buffer.x[:token_count].copy_(x_fp8)
        sym_buffer.x_sf[:token_count].copy_(x_scale)
        sym_buffer.topk_idx[:token_count].copy_(topk_idx)
        sym_buffer.topk_weights[:token_count].copy_(topk_weights)

        y_v2 = torch.empty(
            (int(sym_buffer.num_max_tokens_per_rank), hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        baseline_layout_cache = None

        def get_baseline_layout_cache():
            nonlocal baseline_layout_cache
            if baseline_layout_cache is None:
                layout_result = ep_buffer.get_dispatch_layout(topk_idx, num_experts)
                if len(layout_result) != 5:
                    raise RuntimeError(f"unexpected DeepEP layout return arity: {len(layout_result)}")
                num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, event = layout_result
                if hasattr(event, "current_stream_wait") and getattr(event, "event", None) is not None:
                    event.current_stream_wait()
                baseline_layout_cache = (
                    num_tokens_per_rank,
                    num_tokens_per_expert,
                    is_token_in_rank,
                )
            return baseline_layout_cache

        def run_v2_once():
            v2.fp8_w8a8_mega_moe_v2(
                y_v2,
                v2_l1,
                v2_l2,
                sym_buffer,
                cumulative_local_expert_recv_stats=None,
                activation_clamp=activation_clamp,
                fast_math=True,
                dispatch_num_tokens=token_count,
                backend=backend,
            )
            return y_v2

        def run_baseline_once():
            baseline_y, _, _ = baseline.run_deepgemm_megamoe_baseline(
                ep_buffer,
                ep_config,
                x_fp8,
                x_scale,
                topk_idx,
                topk_weights,
                baseline_l1,
                baseline_l2,
                num_experts,
                local_experts,
                intermediate_hidden,
                hidden,
                activation_clamp,
                baseline.DEEPEP_EXPERT_ALIGNMENT,
                "hip",
                layout_cache=get_baseline_layout_cache(),
                return_stats=False,
            )
            return baseline_y

        run_v2_once()
        baseline_y = run_baseline_once()
        torch.cuda.synchronize()
        diff = (y_v2[:token_count].float() - baseline_y.float()).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        mismatch = int((diff > 1.0e-3).sum().item()) if diff.numel() else 0
        assert max_abs <= 1.0e-3, (backend, token_count, max_abs, mean_abs, mismatch)

        v2_time = _cuda_event_bench_ms(run_v2_once, warmup=warmup, repeat=repeat)
        baseline_time = _cuda_event_bench_ms(run_baseline_once, warmup=warmup, repeat=repeat)
        stage_breakdown = os.getenv(
            "MEGAMOE_DCU_V2_REAL_FLOW_PERF_STAGE_BREAKDOWN",
            "0",
        ) == "1"
        stage_medians: dict[str, float] = {}
        if stage_breakdown:
            for _ in range(warmup):
                stage_times: dict[str, float] = {}
                v2.fp8_w8a8_mega_moe_v2(
                    y_v2,
                    v2_l1,
                    v2_l2,
                    sym_buffer,
                    cumulative_local_expert_recv_stats=None,
                    activation_clamp=activation_clamp,
                    fast_math=True,
                    dispatch_num_tokens=token_count,
                    backend=backend,
                    profile_stages=stage_times,
                )
            stage_samples: dict[str, list[float]] = {
                "k1_ms": [],
                "k2_ms": [],
                "k3_ms": [],
            }
            for _ in range(repeat):
                stage_times = {}
                v2.fp8_w8a8_mega_moe_v2(
                    y_v2,
                    v2_l1,
                    v2_l2,
                    sym_buffer,
                    cumulative_local_expert_recv_stats=None,
                    activation_clamp=activation_clamp,
                    fast_math=True,
                    dispatch_num_tokens=token_count,
                    backend=backend,
                    profile_stages=stage_times,
                )
                for key in stage_samples:
                    stage_samples[key].append(float(stage_times[key]))
            stage_medians = {
                key: _median_float(values) for key, values in stage_samples.items()
            }

        metric_values = [
            v2_time["median_ms"],
            v2_time["min_ms"],
            v2_time["max_ms"],
            baseline_time["median_ms"],
            baseline_time["min_ms"],
            baseline_time["max_ms"],
        ]
        if stage_breakdown:
            metric_values.extend(
                [
                    stage_medians["k1_ms"],
                    stage_medians["k2_ms"],
                    stage_medians["k3_ms"],
                ]
            )
        gathered = baseline.gather_rank_metrics(
            metric_values,
            x_fp8.device,
            group,
        ).cpu()
        if rank == 0:
            v2_median = float(gathered[:, 0].max().item())
            v2_min = float(gathered[:, 1].max().item())
            v2_max = float(gathered[:, 2].max().item())
            baseline_median = float(gathered[:, 3].max().item())
            baseline_min = float(gathered[:, 4].max().item())
            baseline_max = float(gathered[:, 5].max().item())
            degradation = (
                float("nan")
                if baseline_median == 0.0
                else (v2_median / baseline_median - 1.0) * 100.0
            )
            print(
                {
                    "backend": backend,
                    "tokens": token_count,
                    "ranks": num_ranks,
                    "route_mode": route_mode,
                    "warmup": warmup,
                    "repeat": repeat,
                    "v2_staged_rankmax_median_ms": v2_median,
                    "v2_staged_rankmax_min_ms": v2_min,
                    "v2_staged_rankmax_max_ms": v2_max,
                    "baseline_e2e_rankmax_median_ms": baseline_median,
                    "baseline_e2e_rankmax_min_ms": baseline_min,
                    "baseline_e2e_rankmax_max_ms": baseline_max,
                    "degradation_pct_vs_baseline_e2e": degradation,
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "mismatch": mismatch,
                    **(
                        {
                            "stage_rankmax_median_ms": {
                                "k1": float(gathered[:, 6].max().item()),
                                "k2": float(gathered[:, 7].max().item()),
                                "k3": float(gathered[:, 8].max().item()),
                            }
                        }
                        if stage_breakdown
                        else {}
                    ),
                },
                flush=True,
            )
    finally:
        if sym_buffer is not None:
            sym_buffer.destroy()
        if ep_buffer is not None:
            ep_buffer.destroy()
        if dist.is_initialized():
            dist.barrier(group=group)
            dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("MEGAMOE_DCU_V2_REAL_FLOW", "0") != "1",
    reason="set MEGAMOE_DCU_V2_REAL_FLOW=1 to run distributed real-flow correctness",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="DCU/CUDA device is required")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_real_flow_correctness_against_deepep_deepgemm_baseline():
    tokens = [int(item) for item in _parse_env_list("MEGAMOE_DCU_V2_REAL_FLOW_TOKENS", "32")]
    backends = _parse_env_list(
        "MEGAMOE_DCU_V2_REAL_FLOW_BACKENDS",
        os.getenv("MEGAMOE_DCU_V2_BACKEND", "ll"),
    )
    route_mode = os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_ROUTE_MODE", "local_only").strip()
    if route_mode not in {"local_only", "cross_rank"}:
        raise ValueError("MEGAMOE_DCU_V2_REAL_FLOW_ROUTE_MODE must be local_only or cross_rank")
    num_processes = int(os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_RANKS", "4"))
    if num_processes not in {4, 8}:
        pytest.skip("V2 real-flow baseline validation requires 4 or 8 ranks")
    uneven_items = [
        int(item)
        for item in _parse_env_list("MEGAMOE_DCU_V2_REAL_FLOW_UNEVEN_TOKENS", "")
    ]
    if uneven_items:
        if len(uneven_items) != num_processes:
            raise ValueError(
                "MEGAMOE_DCU_V2_REAL_FLOW_UNEVEN_TOKENS must contain one "
                "token count per rank"
            )
        if any(item < 0 for item in uneven_items):
            raise ValueError("uneven token counts must be non-negative")
        tokens_per_rank = tuple(uneven_items)
        max_token_count = max(tokens_per_rank)
    else:
        tokens_per_rank = None
        max_token_count = 0
    import torch.multiprocessing as mp

    old_env = {name: os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK")}
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        for backend in backends:
            if tokens_per_rank is not None:
                os.environ["MASTER_PORT"] = str(_reserve_free_tcp_port())
                mp.spawn(
                    _v2_real_flow_worker,
                    args=(num_processes, max_token_count, backend, tokens_per_rank, route_mode),
                    nprocs=num_processes,
                    join=True,
                )
                continue
            for token_count in tokens:
                os.environ["MASTER_PORT"] = str(_reserve_free_tcp_port())
                mp.spawn(
                    _v2_real_flow_worker,
                    args=(num_processes, token_count, backend, None, route_mode),
                    nprocs=num_processes,
                    join=True,
                )
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.mark.skipif(
    os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_PERF", "0") != "1",
    reason="set MEGAMOE_DCU_V2_REAL_FLOW_PERF=1 to run distributed real-flow performance",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="DCU/CUDA device is required")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch FP8 dtype is required")
def test_v2_real_flow_integrated_performance_against_deepep_deepgemm_baseline():
    tokens = [
        int(item)
        for item in _parse_env_list(
            "MEGAMOE_DCU_V2_REAL_FLOW_PERF_TOKENS",
            os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_TOKENS", "32,128,1024,4096"),
        )
    ]
    backends = _parse_env_list(
        "MEGAMOE_DCU_V2_REAL_FLOW_PERF_BACKENDS",
        os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_BACKENDS", "ll,normal"),
    )
    num_processes = int(
        os.getenv(
            "MEGAMOE_DCU_V2_REAL_FLOW_PERF_RANKS",
            os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_RANKS", "4"),
        )
    )
    warmup = int(os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_PERF_WARMUP", "3"))
    repeat = int(os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_PERF_REPEAT", "5"))
    route_mode = os.getenv(
        "MEGAMOE_DCU_V2_REAL_FLOW_PERF_ROUTE_MODE",
        os.getenv("MEGAMOE_DCU_V2_REAL_FLOW_ROUTE_MODE", "local_only"),
    ).strip()
    if route_mode not in {"local_only", "cross_rank"}:
        raise ValueError(
            "MEGAMOE_DCU_V2_REAL_FLOW_PERF_ROUTE_MODE must be local_only or cross_rank"
        )
    if num_processes not in {4, 8}:
        pytest.skip("V2 real-flow performance validation requires 4 or 8 ranks")
    import torch.multiprocessing as mp

    old_env = {name: os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK")}
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        for backend in backends:
            for token_count in tokens:
                os.environ["MASTER_PORT"] = str(_reserve_free_tcp_port())
                mp.spawn(
                    _v2_real_flow_perf_worker,
                    args=(num_processes, token_count, backend, warmup, repeat, route_mode),
                    nprocs=num_processes,
                    join=True,
                )
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
