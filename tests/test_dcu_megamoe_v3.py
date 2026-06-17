from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG_PATH = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "v3_config.py"
V3_LAYOUT_PATH = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "v3_layout.py"
LARGE_OPT_PATH = ROOT / "megamoe" / "large_opt.py"
SETUP_PATH = ROOT / "setup.py"
MEGA_DCU_API_PATH = ROOT / "csrc" / "apis" / "mega_dcu.hpp"
K1_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K1_fused"
K2_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K2_fused"
K3_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K3_fused"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_int(source: str, name: str) -> int:
    match = re.search(rf"\b{name}\s*=\s*([0-9]+)\b", source)
    assert match is not None, f"missing integer constant {name}"
    return int(match.group(1))


def test_v3_backend_auto_policy(monkeypatch):
    config = load_module("dcu_megamoe_v3_config", V3_CONFIG_PATH)
    monkeypatch.delenv("MEGAMOE_DCU_BACKEND", raising=False)
    monkeypatch.delenv("MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD", raising=False)

    assert config.BACKEND_ENV == "MEGAMOE_DCU_BACKEND"
    assert config.NORMAL_LL_TOKEN_THRESHOLD_ENV == "MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD"
    assert config.DEFAULT_NORMAL_LL_TOKEN_THRESHOLD == 256
    for tokens in (0, 1, 8, 32, 128, 256):
        assert config.select_v3_backend(tokens) == "ll"
    for tokens in (257, 512, 1024, 4096, 8192):
        assert config.select_v3_backend(tokens) == "normal"
    assert config.v3_backend_mode("auto") == "auto"
    assert config.v3_backend_mode("ll") == "ll"
    assert config.v3_backend_mode("normal") == "normal"
    assert config.select_v3_backend(8, "ll") == "ll"
    assert config.select_v3_backend(8, "normal") == "normal"
    assert config.normal_ll_token_threshold("512") == 512
    monkeypatch.setenv("MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD", "512")
    assert config.select_v3_backend(512) == "ll"
    assert config.select_v3_backend(513) == "normal"
    monkeypatch.setenv("MEGAMOE_DCU_BACKEND", "normal")
    assert config.select_v3_backend(8) == "normal"
    with pytest.raises(ValueError, match="non-negative"):
        config.select_v3_backend(-1)
    with pytest.raises(ValueError, match="non-negative"):
        config.normal_ll_token_threshold("-1")
    with pytest.raises(ValueError, match="MEGAMOE_DCU_BACKEND"):
        config.select_v3_backend(8, "bogus")


def reference_pack5_weight(weight: torch.Tensor) -> torch.Tensor:
    experts, n, k = weight.shape
    assert n % 256 == 0 and k % 64 == 0
    out = torch.empty_like(weight).flatten()
    dst = 0
    for expert in range(experts):
        for ko in range(k // 64):
            for no in range(n // 256):
                for ni16 in range(16):
                    for ks in range(4):
                        for ni in range(16):
                            src_ni = (ni & 3) * 4 + (ni >> 2)
                            for ki in range(16):
                                row = no * 256 + ni16 * 16 + src_ni
                                col = ko * 64 + ks * 16 + ki
                                out[dst] = weight[expert, row, col]
                                dst += 1
    return out.reshape(experts, k // 64, n // 256, 16, 4, 16, 16)


def reference_pack5_weight_asm_normal(weight: torch.Tensor) -> torch.Tensor:
    experts, n, k = weight.shape
    assert n % 256 == 0 and k % 64 == 0
    out = torch.empty_like(weight).flatten()
    dst = 0
    for expert in range(experts):
        for ko in range(k // 64):
            for no in range(n // 256):
                for ni16 in range(16):
                    for ks in range(4):
                        for ni in range(16):
                            for ki in range(16):
                                row = no * 256 + ni16 * 16 + ni
                                col = ko * 64 + ks * 16 + ki
                                out[dst] = weight[expert, row, col]
                                dst += 1
    return out.reshape(experts, k // 64, n // 256, 16, 4, 16, 16)


def test_v3_pack5_layout_matches_reference():
    v3_layout = load_module("dcu_megamoe_v3_layout", V3_LAYOUT_PATH)
    weight = torch.arange(2 * 512 * 128, dtype=torch.uint8).reshape(2, 512, 128)

    torch.testing.assert_close(
        v3_layout.pack5_weight(weight), reference_pack5_weight(weight)
    )
    torch.testing.assert_close(
        v3_layout.pack5_weight_asm_normal(weight),
        reference_pack5_weight_asm_normal(weight),
    )
    assert not torch.equal(
        v3_layout.pack5_weight(weight), v3_layout.pack5_weight_asm_normal(weight)
    )
    assert v3_layout.flatten_pack5_weight(weight).shape == (2, 512 * 128)
    assert v3_layout.flatten_pack5_weight_asm_normal(weight).shape == (2, 512 * 128)
    assert v3_layout.pack5_shape(2, 512, 128) == (2, 2, 2, 16, 4, 16, 16)
    offset = v3_layout.pack5_flat_offset(expert=1, n=512, k=128, row=257, col=65)
    assert reference_pack5_weight(weight).flatten()[offset] == weight[1, 257, 65]


def test_v3_build_surface_is_minimal_and_explicit():
    setup_source = SETUP_PATH.read_text(encoding="utf-8")

    for required in (
        "k1_v3_fused_ext.cu",
        "k3_v3_fused_ext.cu",
        "MEGAMOE_DISPATCH_PULL_L1_PACK5.s",
        "K3COMBINE_PACK5.s",
        "K3COMBINE_TAILREDUCE_PACK5.s",
        "'*.cuh'",
    ):
        assert required in setup_source

    for retired in (
        "DG_BUILD_MEGAMOE_V2_EXT",
        "dcu_megamoe_v2",
        "mega_moe_fused_hip.cu",
        "k1_v3_stub_ext.cu",
        "k3_v3_stub_ext.cu",
        "DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS",
        "DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS",
        "large_opt_k1_sources",
        "large_opt_k3_sources",
        "large_opt_k3_v3_ext",
    ):
        assert retired not in setup_source


def test_v3_runtime_sources_have_clear_backend_boundaries():
    k1_py = (K1_FUSED_DIR / "k1_fused.py").read_text(encoding="utf-8")
    k1_asm_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k1_asm_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            K1_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s",
        )
    )
    k1_ext = (K1_FUSED_DIR / "k1_v3_fused_ext.cu").read_text(encoding="utf-8")
    k1_header = (K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k3_py = (K3_FUSED_DIR / "k3_fused.py").read_text(encoding="utf-8")
    k3_asm_ext = (K3_FUSED_DIR / "k3_fused_ext.cu").read_text(encoding="utf-8")
    k3_ext = (K3_FUSED_DIR / "k3_v3_fused_ext.cu").read_text(encoding="utf-8")
    k3_header = (K3_FUSED_DIR / "k3_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )

    assert "FUSED_L1_ASM_PACK5_CO" in k1_py
    assert "def k1_symm_fused_l1_asm(" not in k1_py
    assert "def k1_symm_fused_l1_asm_graph(" not in k1_py
    assert "ensure_fused_l1_asm_code_object" not in k1_py
    assert "def k1_symm_fused_l1_v3_asm_pack5(" in k1_py
    assert "def k1_symm_fused_l1_v3_asm_pack5_graph(" in k1_py
    assert "def k1_symm_fused_l1_v3(" in k1_py
    assert "def k1_symm_fused_l1_v3_graph(" in k1_py
    assert "ext.k1_symm_fused_l1_v3_asm_pack5" in k1_py
    assert "k1_symm_fused_l1_v3_pack5" in k1_py
    assert "k1_symm_fused_l1_v3_asm_pack5" in k1_asm_ext
    assert 'm.def("k1_symm_fused_l1"' not in k1_asm_ext
    assert "k1_symm_fused_l1_asm_impl" in k1_asm_ext
    assert "use_absolute_x_ptrs" not in k1_asm_ext
    assert "prob.reserved_c0 |= 4u" in k1_asm_ext
    assert "use_compact_prebuild ? 2u : 4u" not in k1_asm_ext
    assert "kK1AutoCompactMinLocalTileSaving" in k1_asm_ext
    assert "kK1AutoCompactLargeTilesPerExpert" in k1_asm_ext
    assert "label_SymmRouteStoreAbsolutePtr" not in k1_asm_sources
    assert "label_SymmStageLoadAbsolutePtr" not in k1_asm_sources
    assert "absolute 64-bit" not in k1_asm_sources
    assert "dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5" in k1_ext
    assert "V3_K1_LowLatencyMaskedGroupGemmKernel" in k1_header
    assert "V3_K1_Fused_DeepGemm" not in k1_header
    assert "v3_k1_build_fixed_route_tile_device" not in k1_header
    assert "V3_K1_Pure" not in k1_header

    assert "K3_COMBINE_PACK5_ASM_CO" in k3_py
    assert "K3_COMBINE_TAIL_REDUCE_PACK5_ASM_CO" in k3_py
    assert "k3_l2_combine_asm_pack5_out" in k3_py
    assert "k3_l2_combine_asm_tail_reduce_pack5_out" in k3_py
    assert "k3_l2_combine_asm_out" not in k3_asm_ext
    assert "k3_l2_combine_asm_tail_reduce_out" not in k3_asm_ext
    assert "k3_l2_combine_asm_pack5_out" in k3_asm_ext
    assert "k3_l2_combine_asm_tail_reduce_pack5_out" in k3_asm_ext
    assert "debug_d" not in k3_asm_ext
    assert "row_combine_ptrs_i32" in k3_asm_ext
    assert "def k3_l2_fused_asm_to_combine(" not in k3_py
    assert "ensure_k3_combine_asm_code_object" not in k3_py
    assert "ensure_k3_combine_tail_reduce_asm_code_object" not in k3_py
    assert "k3_v3_ll_combine(" in k3_py
    assert "k3_v3_ll_combine_tail(" in k3_py
    assert "graph_runtime_num_tokens" in k3_py
    v3_k3_signature = k3_py.split("def k3_l2_fused_v3_to_combine(", 1)[1].split(
        ") -> torch.Tensor | None:",
        1,
    )[0]
    assert "graph_runtime_num_tokens: torch.Tensor | None = None" in v3_k3_signature
    assert "runtime_num_tokens_tensor" in k3_ext
    assert "runtime_num_tokens_ptr" in k3_ext
    assert "num_tokens, runtime_num_tokens, num_topk" in k3_ext
    assert "effective_num_tokens" in k3_header
    assert "static_cast<int64_t>(effective_num_tokens) * vecs_per_token" in k3_header
    assert "k3_v3_ll_reference" not in k3_ext
    assert "V3_K3_LowLatencyMaskedGroupGemmKernel" in k3_header
    assert "V3_K3_Fused_DeepGemm" not in k3_header
    assert "kContiguousOutput" not in k3_header
    assert "V3_K3_Pure" not in k3_header


def test_retired_v3_debug_and_dormant_api_are_absent_from_production_sources():
    production_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            LARGE_OPT_PATH,
            K1_FUSED_DIR / "k1_v3_fused_ext.cu",
            K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh",
            K2_FUSED_DIR / "k2_fused.py",
            K2_FUSED_DIR / "k2_fused_ext.cu",
            K3_FUSED_DIR / "k3_fused.py",
            K3_FUSED_DIR / "k3_fused_ext.cu",
            K3_FUSED_DIR / "k3_v3_fused_ext.cu",
        )
    )

    for retired in (
        "MEGAMOE_DCU_K3_DEBUG_LAUNCH",
        "MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC",
        "MEGAMOE_DCU_V3_STAGE_TIMING",
        "MEGAMOE_DCU_V3_LL_BLOCK_M",
        "MEGAMOE_DCU_V3_K2_SYSTEM_FENCE",
        "MEGAMOE_DCU_V3_REDUCE_ACQUIRE",
        "MEGAMOE_DCU_V3_BARRIER_ACQUIRE",
        "MEGAMOE_DCU_V3_LL_K1_PARALLEL_STAGE_COPY",
        "system_fence_after_write",
        "invalidate_before_read",
        "acquire_after_wait",
        "V3 K3 LL no-tail signal path is not wired yet",
    ):
        assert retired not in production_sources

    assert not (ROOT / "megamoe" / "dcu_megamoe_v2").exists()


def test_v3_ll_capacity_headroom_covers_256_and_512_exact_buckets():
    k1_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")

    assert "kLlHeadroomExpectedRowsThreshold = 48" in k1_ext
    assert "kLlHeadroomRows = 64" in k1_ext
    assert "ll_expected_rows_per_expert >= kLlHeadroomExpectedRowsThreshold" in k1_ext
    assert "ll_expected_rows_per_expert >= 160 ? 64 : 0" not in k1_ext


def test_v3_normal_graph_runtime_work_is_limited_without_d2h():
    k1_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k2_ext = (ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K2_fused" / "k2_fused_ext.cu").read_text(encoding="utf-8")
    large_opt_source = LARGE_OPT_PATH.read_text(encoding="utf-8")

    assert "runtime_limited_init" in k1_ext
    assert "const int init_rows = runtime_limited_init != 0 ? active_rows : capacity_rows;" in k1_ext
    assert "runtime_num_tokens == nullptr ? 0 : 1" in k1_ext
    assert "runtime_num_tokens != nullptr\n            ? 12" in k1_ext
    assert "if (!has_actual_m && active_tiles != nullptr && active_tile_m > 0)" in k2_ext
    assert "K_K2_GRAPH_ROW_BLOCKS = 8192" in large_opt_source
    assert "active_tiles=k2_active_tiles" in large_opt_source
    assert "state.scratch.k1_active_tiles if v3_backend != V3_BACKEND_LL else None" in large_opt_source


def test_public_capacity_token_and_graph_backend_contract_is_explicit():
    api_source = (ROOT / "megamoe" / "__init__.py").read_text(encoding="utf-8")
    c_api_source = MEGA_DCU_API_PATH.read_text(encoding="utf-8")
    test_source = (ROOT / "tests" / "test_mega_moe_dcu.py").read_text(encoding="utf-8")
    large_opt_source = LARGE_OPT_PATH.read_text(encoding="utf-8")

    assert "megamoe_backend: str = V3_BACKEND_NORMAL" in api_source
    assert "graph: bool = False" in api_source
    assert "capacity_num_tokens: Optional[int] = None" in api_source
    assert "dispatch_num_tokens: Optional[int] = None" not in api_source
    assert "ll_cuda_graph" not in api_source
    assert "normal_cuda_graph" not in api_source
    assert "big_fused_cuda_graph" not in api_source
    assert "stages_fused_cuda_graph" not in api_source
    assert "v3_shape" not in api_source
    assert "v3_backend = normalize_v3_backend(megamoe_backend)" in api_source
    assert "select_v3_backend(" not in api_source
    assert "_C.fp8_mega_moe" not in api_source
    assert "fp8_mega_moe_with_graph_tokens" not in c_api_source
    assert "launch_mega_moe_multirank_persistent" not in c_api_source

    graph_signature = large_opt_source.split("def _run_large_opt_3stage_graph(", 1)[1].split(
        ") -> None:",
        1,
    )[0]
    assert "dispatch_num_tokens" not in graph_signature
    assert "capacity_num_tokens" not in graph_signature
    assert "v3_backend: str" in graph_signature

    assert "--megamoe-backend" in test_source
    assert "--cuda-graph" in test_source
    assert "--ll-cuda-graph" not in test_source
    assert "--normal-cuda-graph" not in test_source
    assert "--big-fused-cuda-graph" not in test_source
    assert "--stages-fused-cuda-graph" not in test_source
    assert "--dispatch-num-tokens" not in test_source
    run_fused_call = test_source.split("def run_fused(", 1)[1].split(
        "def fill_graph_inputs",
        1,
    )[0]
    graph_call = test_source.split("def run_graph_bucket_once(", 1)[1].split(
        "fill_graph_inputs(capture_tokens)",
        1,
    )[0]
    assert "megamoe_backend=v3_backend" in run_fused_call
    assert "capacity_num_tokens=backend_selector_tokens" in run_fused_call
    assert "dispatch_num_tokens=" not in run_fused_call
    assert "megamoe_backend=v3_backend" in graph_call
    assert "graph=True" in graph_call
    assert "dispatch_num_tokens=" not in graph_call
    assert "capacity_num_tokens=" not in graph_call


def test_v3_staged_route_scratch_size_uses_ll_normal_layout():
    api_source = MEGA_DCU_API_PATH.read_text(encoding="utf-8")
    large_opt_source = LARGE_OPT_PATH.read_text(encoding="utf-8")

    assert "staged_pack5_shape" in api_source
    assert "num_ranks == 8 && num_experts == 256 && num_topk == 6" in api_source
    assert "hidden == 4096 && intermediate_hidden == 2048" in api_source
    assert "return legacy_route_scratch_bytes();" not in api_source
    assert "dcu_route_scratch_bytes(" not in api_source
    assert "MEGAMOE_DCU_NORMAL" not in api_source
    assert "get_normal_token_threshold_for_mega_moe()" not in api_source
    assert "kK1AsmLaunchArgsBytes = 256" in api_source
    assert "capacity_rows" in api_source
    assert "static_cast<int64_t>(intermediate_hidden) * 2" in api_source
    assert "kProbStorageBytes = 256" in api_source
    assert "tail_signal_addrs_offset" in api_source
    assert "dcu_route_tile_scratch_layout(" not in api_source
    assert "std::max(v3_staged" not in api_source

    assert "def _v3_staged_capacity_rows(" in large_opt_source
    assert "normal_token_threshold()" not in large_opt_source
    assert "normal_backend_forced()" not in large_opt_source
    assert "K_K1_ASM_LAUNCH_ARGS_BYTES = 256" in large_opt_source
    assert "capacity_rows * intermediate_hidden * 2" in large_opt_source

    for py_name, cpp_name in (
        ("K_K1_ROUTE_TILE_M", "kK1RouteTileM"),
        ("K_K1_ALIGNMENT", "kK1Alignment"),
        ("K_K1_ROUTE_CAPACITY_SLACK", "kK1RouteCapacitySlack"),
        ("K_K1_LL_ROW_TILE", "kK1LlRowTile"),
        (
            "K_K1_LL_HEADROOM_EXPECTED_ROWS_THRESHOLD",
            "kK1LlHeadroomExpectedRowsThreshold",
        ),
        ("K_K1_LL_HEADROOM_ROWS", "kK1LlHeadroomRows"),
        ("K_K1_ASM_LAUNCH_ARGS_BYTES", "kK1AsmLaunchArgsBytes"),
        ("K_PROB_STORAGE_BYTES", "kProbStorageBytes"),
        ("K_TAIL_DONE_COUNTER_RING_SLOTS", "kTailDoneCounterRingSlots"),
    ):
        assert source_int(large_opt_source, py_name) == source_int(api_source, cpp_name)

    for mirrored_name in (
        "ll_capacity_rows",
        "normal_capacity_rows",
        "capacity_rows",
        "route_task_workspace_bytes",
        "tail_signal_addrs_offset",
    ):
        assert mirrored_name in api_source
        assert mirrored_name in large_opt_source
    assert "capacity_rows * static_cast<int64_t>(hidden)" in api_source
    assert "capacity_rows * hidden" in large_opt_source
