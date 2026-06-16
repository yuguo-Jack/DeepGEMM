from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
V3_CONFIG_PATH = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "v3_config.py"
V3_LAYOUT_PATH = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "v3_layout.py"
LARGE_OPT_PATH = ROOT / "megamoe" / "large_opt.py"
SETUP_PATH = ROOT / "setup.py"
K1_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K1_fused"
K2_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K2_fused"
K3_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_large_opt" / "K3_fused"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_gate_requires_forced_large_opt(monkeypatch):
    config = load_module("dcu_megamoe_v3_config", V3_CONFIG_PATH)

    monkeypatch.setenv("USE_MEGAMOE_V3", "1")
    monkeypatch.delenv("MEGAMOE_DCU_USE_LARGE_OPT_3STAGE", raising=False)
    assert not config.v3_enabled()

    for value in ("auto", "threshold", "adaptive", "0", "false", "off"):
        monkeypatch.setenv("MEGAMOE_DCU_USE_LARGE_OPT_3STAGE", value)
        assert not config.v3_enabled()

    for value in ("1", "true", "yes", "on", "large_opt", "3stage"):
        monkeypatch.setenv("MEGAMOE_DCU_USE_LARGE_OPT_3STAGE", value)
        assert config.v3_enabled()

    monkeypatch.setenv("USE_MEGAMOE_V3", "0")
    monkeypatch.setenv("MEGAMOE_DCU_USE_LARGE_OPT_3STAGE", "1")
    assert not config.v3_enabled()


def test_v3_backend_contract(monkeypatch):
    config = load_module("dcu_megamoe_v3_config_backend", V3_CONFIG_PATH)

    monkeypatch.delenv("MEGAMOE_DCU_V3_BACKEND", raising=False)
    assert config.get_v3_backend() == "normal"
    monkeypatch.setenv("MEGAMOE_DCU_V3_BACKEND", "ll")
    assert config.get_v3_backend() == "ll"
    monkeypatch.setenv("MEGAMOE_DCU_V3_BACKEND", "NORMAL")
    assert config.get_v3_backend() == "normal"
    with pytest.raises(ValueError, match="MEGAMOE_DCU_V3_BACKEND"):
        config.normalize_v3_backend("auto")


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
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1.s",
            K1_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s",
        )
    )
    k1_ext = (K1_FUSED_DIR / "k1_v3_fused_ext.cu").read_text(encoding="utf-8")
    k1_header = (K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k3_py = (K3_FUSED_DIR / "k3_fused.py").read_text(encoding="utf-8")
    k3_ext = (K3_FUSED_DIR / "k3_v3_fused_ext.cu").read_text(encoding="utf-8")
    k3_header = (K3_FUSED_DIR / "k3_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )

    assert "FUSED_L1_ASM_PACK5_CO" in k1_py
    assert "def k1_symm_fused_l1_v3_asm_pack5(" in k1_py
    assert "def k1_symm_fused_l1_v3_asm_pack5_graph(" in k1_py
    assert "def k1_symm_fused_l1_v3(" in k1_py
    assert "def k1_symm_fused_l1_v3_graph(" in k1_py
    assert "ext.k1_symm_fused_l1_v3_asm_pack5" in k1_py
    assert "k1_symm_fused_l1_v3_pack5" in k1_py
    assert "k1_symm_fused_l1_v3_asm_pack5" in k1_asm_ext
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
    asm_k3_signature = k3_py.split("def k3_l2_fused_asm_to_combine(", 1)[1].split(
        ") -> torch.Tensor | None:",
        1,
    )[0]
    assert "ll_block_m" not in asm_k3_signature
    assert "graph_runtime_num_tokens" not in asm_k3_signature
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
