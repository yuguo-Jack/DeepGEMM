from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
V3_CONFIG_PATH = ROOT / "megamoe" / "dcu_megamoe_opt" / "v3_config.py"
V3_LAYOUT_PATH = ROOT / "megamoe" / "dcu_megamoe_opt" / "v3_layout.py"
OPT_PATH = ROOT / "megamoe" / "opt.py"
SETUP_PATH = ROOT / "setup.py"
BUILD_SCRIPT_PATH = (
    ROOT / "megamoe" / "dcu_megamoe_opt" / "scripts" / "build_dcu_megamoe.sh"
)
MEGA_DCU_API_PATH = ROOT / "megamoe" / "dcu_megamoe_opt" / "csrc" / "apis" / "mega_dcu.hpp"
MEGA_DCU_KERNEL_PATH = (
    ROOT / "megamoe" / "dcu_megamoe_opt" / "csrc" / "kernels" / "mega_moe_baseline_hip.cu"
)
K1_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_opt" / "K1_fused"
K2_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_opt" / "K2_fused"
K3_FUSED_DIR = ROOT / "megamoe" / "dcu_megamoe_opt" / "K3_fused"


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
    assert config.DEFAULT_NORMAL_LL_TOKEN_THRESHOLD == 512
    for tokens in (0, 1, 8, 32, 128, 256, 512):
        assert config.select_v3_backend(tokens) == "ll"
    for tokens in (513, 1024, 4096, 8192):
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


def test_v3_model_shape_registry_covers_flash_and_pro_ep_sizes():
    config = load_module("dcu_megamoe_v3_config_shapes", V3_CONFIG_PATH)

    for shape, local_experts_by_rank in (
        ((256, 6, 4096, 2048), {8: 32, 16: 16, 32: 8}),
        ((384, 6, 7168, 3072), {8: 48, 16: 24, 32: 12}),
    ):
        num_experts, num_topk, hidden, intermediate_hidden = shape
        for num_ranks, expected_local_experts in local_experts_by_rank.items():
            assert config.staged_pack5_shape_supported(
                num_ranks=num_ranks,
                num_experts=num_experts,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
            )
            assert (
                config.staged_pack5_local_experts(
                    num_ranks=num_ranks,
                    num_experts=num_experts,
                    num_topk=num_topk,
                    hidden=hidden,
                    intermediate_hidden=intermediate_hidden,
                )
                == expected_local_experts
            )

    assert not config.staged_pack5_shape_supported(
        num_ranks=16,
        num_experts=384,
        num_topk=6,
        hidden=4096,
        intermediate_hidden=3072,
    )
    assert "DeepSeek-V4-Pro" in config.STAGED_PACK5_SHAPE_CONTRACT


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


def reference_plain_pack5_weight(weight: torch.Tensor) -> torch.Tensor:
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
        v3_layout.pack5_weight_asm_normal(weight), reference_plain_pack5_weight(weight)
    )
    assert v3_layout.flatten_pack5_weight(weight).shape == (2, 512 * 128)
    assert v3_layout.flatten_pack5_weight_asm_normal(weight).shape == (2, 512 * 128)
    assert v3_layout.pack5_shape(2, 512, 128) == (2, 2, 2, 16, 4, 16, 16)
    offset = v3_layout.pack5_flat_offset(expert=1, n=512, k=128, row=257, col=65)
    assert reference_pack5_weight(weight).flatten()[offset] == weight[1, 257, 65]
    assert not torch.equal(v3_layout.pack5_weight(weight), reference_plain_pack5_weight(weight))


def test_v3_build_surface_is_minimal_and_explicit():
    setup_source = SETUP_PATH.read_text(encoding="utf-8")

    for required in (
        "k1_v3_fused_ext.cu",
        "k3_v3_fused_ext.cu",
        "MEGAMOE_DISPATCH_PULL_L1_PACK5.s",
        "MEGAMOE_DISPATCH_PULL_L1_UNIFIED_PACK5.s",
        "K3COMBINE_PACK5.s",
        "K3COMBINE_UNIFIED_PACK5.s",
        "K3COMBINE_TAILREDUCE_PACK5.s",
        "K3COMBINE_TAILREDUCE_UNIFIED_PACK5.s",
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
        "opt_k1_sources",
        "opt_k3_sources",
        "opt_k3_v3_ext",
    ):
        assert retired not in setup_source


def test_v3_runtime_sources_have_clear_backend_boundaries():
    opt_py = OPT_PATH.read_text(encoding="utf-8")
    k1_py = (K1_FUSED_DIR / "k1_fused.py").read_text(encoding="utf-8")
    k1_asm_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k1_asm_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            K1_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s",
            K1_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_UNIFIED_PACK5.s",
        )
    )
    k1_ext = (K1_FUSED_DIR / "k1_v3_fused_ext.cu").read_text(encoding="utf-8")
    k1_header = (K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k2_py = (K2_FUSED_DIR / "k2_fused.py").read_text(encoding="utf-8")
    k2_ext = (K2_FUSED_DIR / "k2_fused_ext.cu").read_text(encoding="utf-8")
    k3_py = (K3_FUSED_DIR / "k3_fused.py").read_text(encoding="utf-8")
    k3_asm_ext = (K3_FUSED_DIR / "k3_fused_ext.cu").read_text(encoding="utf-8")
    k3_asm_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            K3_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.s",
            K3_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_UNIFIED_PACK5.s",
            K3_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_PACK5.s",
            K3_FUSED_DIR
            / "DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE_UNIFIED_PACK5.s",
        )
    )
    k3_ext = (K3_FUSED_DIR / "k3_v3_fused_ext.cu").read_text(encoding="utf-8")
    k3_header = (K3_FUSED_DIR / "k3_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )

    assert "FUSED_L1_ASM_PACK5_CO" in k1_py
    assert "FUSED_L1_ASM_UNIFIED_PACK5_CO" in k1_py
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
    assert "compact_active_tiles_offset" in k1_asm_ext
    assert "static_cast<uint32_t>(2 * local_experts)" in k1_asm_ext
    assert "(static_cast<uint32_t>(local_experts) << 16)" in k1_asm_ext
    assert "use_compact_prebuild ? 2u : 4u" not in k1_asm_ext
    assert "kK1AutoCompactMinLocalTileSaving" in k1_asm_ext
    assert "kK1AutoCompactHighTilesPerExpert" in k1_asm_ext
    assert "const bool default_compact_prebuild = true" in k1_asm_ext
    assert "K1_PREBUILD_MODE" in k1_asm_ext
    assert "num_ranks <= 8 && local_experts <= 32" in k1_asm_ext
    assert "label_SymmRouteStoreAbsolutePtr" not in k1_asm_sources
    assert "label_SymmStageLoadAbsolutePtr" not in k1_asm_sources
    assert "absolute 64-bit" not in k1_asm_sources
    assert "ko64 * (4096 * 64)" not in k1_asm_sources
    assert "v_lshlrev_b32 v[vgprPack5OffsetBaseA], 18" not in k1_asm_sources
    assert "s_mov_b32 s[sgprGlobalReadIncsA+0], 0x80000" not in k1_asm_sources
    assert "s_mov_b32 s60, 0x8000" not in k1_asm_sources
    assert "s_mul_i32 s71, s91, 0x1000" not in k1_asm_sources
    assert "no256 * (4096 / 4)" not in k1_asm_sources
    assert "SizeI / 4" not in k1_asm_sources
    assert "v_mov_b32 v253, 64                                 // route_scratch total active compact tiles" not in k1_asm_sources
    assert "s_cmp_ge_u32 s[sgprScaleFlag], 32" not in k1_asm_sources
    assert k1_asm_sources.count("packed compact metadata") == 2
    assert k1_asm_sources.count("route_scratch active_tiles i32 offset") == 2
    assert "m_indices[compact tile * 256]" not in k1_asm_sources
    assert k1_asm_sources.count("tile_experts offset") == 2
    assert (
        k1_asm_sources.count("route_scratch_i32[tile_experts + compact tile]")
        == 2
    )
    assert "DEBUG_FORCE_EXPERT1" not in k1_asm_sources
    assert "make HIP-built compact metadata visible" not in k1_asm_sources
    assert k1_asm_sources.count("s_lshr_b32 s62, s61, 16") == 2
    assert k1_asm_sources.count("pack5 ni16 * (4 * 256)") == 2
    assert k1_asm_sources.count(
        "v_mul_lo_u32 v[vgprPack5OffsetBaseA], s[sgprSizeI]"
    ) == 2
    assert k1_asm_sources.count(
        "v_lshlrev_b32 v[\\vgprTmp+0], 10, v[\\vgprOffset0I]"
    ) == 2
    assert k1_asm_sources.count(
        "s_lshl_b32 s[sgprGlobalReadIncsA+0], s[sgprSizeI], 7"
    ) == 4
    assert k1_asm_sources.count("s_mul_i32 s71, s91, s[sgprSizeL]") == 2
    assert k1_asm_sources.count("s_lshl_b32 s60, s[sgprSizeL], 3") == 2
    assert k1_asm_sources.count("label_SymmStageProIndex") == 4
    assert "v_mul_lo_u32 v251, 9363, v251" not in k1_asm_sources
    assert "v_mul_lo_u32 v252, 224, v251" not in k1_asm_sources
    assert k1_asm_sources.count("s_mov_b32 s63, 9363") == 2
    assert k1_asm_sources.count("v_mul_lo_u32 v251, s63, v251") == 2
    assert k1_asm_sources.count("s_mov_b32 s63, 224") == 2
    assert k1_asm_sources.count("v_mul_lo_u32 v252, s63, v251") == 2
    assert k1_asm_sources.count("label_SymmStageStoreDynamicStride") == 4
    assert k1_asm_sources.count("global staged row * hidden") == 2
    assert k1_asm_sources.count("v_mul_lo_u32 v251, s[sgprSizeL], v251") == 2
    assert k1_asm_sources.count("v_lshlrev_b32 v251, 12, v251") == 2
    assert "dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5" in k1_ext
    assert "V3_K1_LowLatencyMaskedGroupGemmKernel" in k1_header
    assert "tail_chunk_expected" not in k1_py
    assert "publish_tail_chunk_expected" not in k1_ext
    assert "v3_k1_publish_tail_chunk_expected_device" not in k1_header
    assert "kV3K1TailChunkSignalSlotBase" in k1_header
    assert "kV3K1TailCopyExpertDoneOffset" in k1_header
    assert "kDcuMegaMoeTailDoneCounterInts" in k1_asm_ext
    assert "V3_K1_Fused_DeepGemm" not in k1_header
    assert "v3_k1_build_fixed_route_tile_device" not in k1_header
    assert "V3_K1_Pure" not in k1_header

    assert "K3_COMBINE_PACK5_ASM_CO" in k3_py
    assert "K3_COMBINE_TAIL_REDUCE_PACK5_ASM_CO" in k3_py
    assert "K3_COMBINE_UNIFIED_PACK5_ASM_CO" in k3_py
    assert "K3_COMBINE_TAIL_REDUCE_UNIFIED_PACK5_ASM_CO" in k3_py
    assert "k3_l2_combine_asm_pack5_out" in k3_py
    assert "k3_l2_combine_asm_tail_reduce_pack5_out" in k3_py
    assert "k3_l2_combine_asm_out" not in k3_asm_ext
    assert "k3_l2_combine_asm_tail_reduce_out" not in k3_asm_ext
    assert "k3_l2_combine_asm_pack5_out" in k3_asm_ext
    assert "k3_l2_combine_asm_tail_reduce_pack5_out" in k3_asm_ext
    assert "ko64 * (4096 * 64)" not in k3_asm_sources
    assert "v_lshlrev_b32 v[vgprPack5OffsetBaseA], 18" not in k3_asm_sources
    assert "s_mov_b32 s[sgprGlobalReadIncsA+0], 0x80000" not in k3_asm_sources
    assert "v_lshlrev_b32 v153, 13, v152" not in k3_asm_sources
    assert "no256 * (4096 / 4)" not in k3_asm_sources
    assert "SizeI / 4" not in k3_asm_sources
    assert k3_asm_sources.count("pack5 ni16 * (4 * 256)") == 4
    assert k3_asm_sources.count(
        "v_mul_lo_u32 v[vgprPack5OffsetBaseA], s[sgprSizeI]"
    ) == 4
    assert k3_asm_sources.count(
        "v_lshlrev_b32 v[\\vgprTmp+0], 10, v[\\vgprOffset0I]"
    ) == 4
    assert k3_asm_sources.count(
        "s_lshl_b32 s[sgprGlobalReadIncsA+0], s[sgprSizeI], 7"
    ) == 8
    assert k3_asm_sources.count("v_mov_b32 v149, s[sgprSizeI]") == 0
    assert k3_asm_sources.count("v_mov_b32 v149, 4096") == 4
    assert k3_asm_sources.count("v_mul_lo_u32 v153, s[sgprSizeI], v152") == 4
    assert k3_asm_sources.count("v_lshlrev_b32 v153, 1, v153") == 4
    assert "s_lshl_b32 s76, s76, 9" not in k3_asm_sources
    assert "s_lshr_b32 s86, s[sgprSizeI], 3" not in k3_asm_sources
    assert k3_asm_sources.count(
        "s_load_dword s86, s[sgprExternalArgAddress:sgprExternalArgAddress+1], 0xd8"
    ) == 2
    assert k3_asm_sources.count("s_mul_i32 s76, s76, s86") == 2
    assert "asm_reduce_hidden_vecs" in k3_asm_ext
    assert "offsetof(GpuProb, asm_reduce_hidden_vecs) == 0xd8" in k3_asm_ext
    assert "sizeof(GpuProb) <= 256" in k3_asm_ext
    assert "debug_d" not in k3_asm_ext
    assert "row_combine_ptrs_i32" in k3_asm_ext
    assert "def k3_l2_fused_asm_to_combine(" not in k3_py
    assert "ensure_k3_combine_asm_code_object" not in k3_py
    assert "ensure_k3_combine_tail_reduce_asm_code_object" not in k3_py
    assert "ext.k3_v3_ll_combine(" not in k3_py
    assert "ext.k3_v3_ll_combine_tail" in k3_py
    assert "ext.k3_v3_ll_combine_tail_split" in k3_py
    assert "void k3_v3_ll_combine(" not in k3_ext
    assert 'm.def("k3_v3_ll_combine",' not in k3_ext
    assert "k3_v3_ll_combine_tail_split" in k3_ext
    assert "MEGAMOE_DCU_LL_K3_SPLIT_TAIL" in opt_py
    assert 'os.getenv("MEGAMOE_DCU_LL_K3_SPLIT_TAIL", "1")' in opt_py
    assert "ll_k3_split_tail_enabled" in opt_py
    assert "K_LL_SPLIT_TAIL_MAX_TOKENS = 512" in opt_py
    assert "kV3K1TailChunkSignalSlots = 8" in k1_header
    assert "kV3K3TailChunkSignalSlots = 8" in k3_header
    assert "kSplitTailChunkSignalSlots = 8" in k3_asm_ext
    assert "LL split-tail chunk signal capacity must cover 512 tokens" in k3_ext
    assert "ll_split_tail: bool = False" in k3_py
    assert "def _tail_reduce_enabled_for_backend" in opt_py
    assert "if v3_backend == V3_BACKEND_LL:\n        return True" in opt_py
    tail_reduce_gate = opt_py.split("def _tail_reduce_enabled_for_backend", 1)[1].split(
        "\n\n",
        1,
    )[0]
    assert "k3_tail_reduce_enabled(default=int(num_ranks) <= 8)" in tail_reduce_gate
    assert "LL no-tail / tail-reduce-0 path has been retired" not in opt_py
    assert "DCU MegaMoE LL no-tail path has been retired" not in opt_py
    assert "MEGAMOE_DCU_LL_K3_SPLIT_TAIL_CHUNK_READY" not in opt_py
    assert "ll_split_tail_chunk_ready" not in opt_py
    assert "use_chunk_ready" not in k3_py
    assert "use_chunk_ready" not in k3_ext
    assert "graph_runtime_num_tokens" in k3_py
    v3_k3_signature = k3_py.split("def k3_l2_fused_v3_to_combine(", 1)[1].split(
        ") -> torch.Tensor | None:",
        1,
    )[0]
    assert "graph_runtime_num_tokens: torch.Tensor | None = None" in v3_k3_signature
    assert "ll_split_tail: bool = False" in v3_k3_signature
    assert "ll_split_copy_shrink" not in v3_k3_signature
    assert "runtime_num_tokens_tensor" in k3_ext
    assert "runtime_num_tokens_ptr" in k3_ext
    assert "split_copy_shrink" not in k3_ext
    assert "split_copy_shrink" not in k3_py
    assert "split_copy_shrink" not in opt_py
    assert "num_tokens,\n            runtime_num_tokens,\n            num_topk" in k3_ext
    assert "max_copy_rows_ptr" in k3_ext
    assert "effective_num_tokens" in k3_header
    assert "static_cast<int64_t>(effective_num_tokens) * vecs_per_token" in k3_header
    assert "V3_K3_LowLatencyCombineReduceKernel" in k3_header
    assert "v3_k3_split_reduce_chunk_tile_device" in k3_header
    assert "v3_k3_tail_wait_gemm_expert_ready_device" not in k3_header
    assert "kV3K3TailGemmExpertDoneOffset" not in k3_header
    assert "kV3K3TailCopyExpertDoneOffset" in k3_header
    assert "publish_gemm_expert_done" not in k3_header
    assert "requires fast_math" not in opt_py
    assert "fast_math: bool = True" in k2_py
    assert "bool(fast_math)" in k2_py
    assert "template <bool kFastMath>" in k2_ext
    assert "swiglu_gate<kFastMath>" in k2_ext
    assert "launch_swiglu_quant_channelwise_auto<true>" in k2_ext
    assert "launch_swiglu_quant_channelwise_auto<false>" in k2_ext
    assert 'pybind11::arg("fast_math") = true' in k2_ext
    assert "const int expected_count = (token_end - token_start) * num_topk" in k3_header
    assert "max_copy_rows <= kCopyRows" not in k3_header
    assert "copy_row_blocks_per_expert =\n            (max_copy_rows + kCopyRows - 1) / kCopyRows" in k3_header
    assert "const int launched_copy_blocks" in k3_header
    assert "copy_idx += launched_copy_blocks" in k3_header
    assert "const int reduce_idx = static_cast<int>(blockIdx.x) - launched_copy_blocks;" in k3_header
    assert "graph_ll_split_copy_shrink_out" not in k1_header
    assert "graph_ll_split_copy_shrink" not in opt_py
    assert "expected=%d seen=%d all_done=%d" not in k3_header
    assert "at::cuda::getStreamFromPool" not in k3_ext
    assert "hipEventRecord" not in k3_ext
    assert "hipStreamWaitEvent" not in k3_ext
    assert "3 * kTailDoneCounterRingSlots + 2 * 32" not in k3_ext
    assert "kV3K3TailCopyExpertDoneCount" in k3_ext
    assert "kSplitTailDoneCounterInts" in k3_ext
    assert "k3_v3_ll_reference" not in k3_ext
    assert "V3_K3_LowLatencyMaskedGroupGemmKernel" in k3_header
    assert "V3_K3_Fused_DeepGemm" not in k3_header
    assert "kContiguousOutput" not in k3_header
    assert "V3_K3_Pure" not in k3_header


def test_retired_v3_debug_and_dormant_api_are_absent_from_production_sources():
    test_harness_source = (
        ROOT / "megamoe" / "dcu_megamoe_opt" / "tests" / "test_mega_moe_dcu.py"
    ).read_text(encoding="utf-8")
    production_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            OPT_PATH,
            K1_FUSED_DIR / "k1_fused.py",
            K1_FUSED_DIR / "k1_fused_ext.cu",
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
        "MEGAMOE_DCU_DEBUG_ROUTE",
        "MEGAMOE_DCU_DEBUG_ROUTE_STAGE",
        "MEGAMOE_DCU_DEBUG_CAPTURE_K1",
        "MEGAMOE_DCU_DEBUG_STOP_AFTER_K1",
        "MEGAMOE_DCU_DEBUG_COMPARE_SOURCE_X",
        "MEGAMOE_DCU_DEBUG_K1_COMPARE",
        "MEGAMOE_DCU_DEBUG_BACKEND_COMPARE",
        "MEGAMOE_DCU_DEBUG_COMBINE",
        "MEGAMOE_DCU_DEBUG_SLEEP_AFTER_K3",
        "MEGAMOE_DCU_K2_GLC_SLC_LOAD",
        "MEGAMOE_DCU_REDUCE_COMBINE_GLC_SLC_LOAD",
        "--debug-combine-on-fail",
        "--debug-compare-backends-on-fail",
        "--trace-failure-detail",
        "TRACE_FAILURE",
        "pre_baseline_combine",
        "system_fence_after_write",
        "invalidate_before_read",
        "ll_asm_compatible_layout",
        "acquire_after_wait",
        "V3 K3 LL no-tail signal path is not wired yet",
    ):
        assert retired not in production_sources
        assert retired not in test_harness_source

    assert not (ROOT / "megamoe" / "dcu_megamoe_v2").exists()


def test_v3_capacity_contract_is_skew_safe_without_overflow_fallback():
    k1_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k1_header = (K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k3_header = (K3_FUSED_DIR / "k3_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k3_normal_ext = (K3_FUSED_DIR / "k3_fused_ext.cu").read_text(
        encoding="utf-8"
    )
    k3_ext = (K3_FUSED_DIR / "k3_v3_fused_ext.cu").read_text(encoding="utf-8")
    k1_py = (K1_FUSED_DIR / "k1_fused.py").read_text(encoding="utf-8")
    api_source = MEGA_DCU_API_PATH.read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")
    init_source = (ROOT / "megamoe" / "__init__.py").read_text(encoding="utf-8")
    test_harness_source = (
        ROOT / "megamoe" / "dcu_megamoe_opt" / "tests" / "test_mega_moe_dcu.py"
    ).read_text(encoding="utf-8")

    assert "kLlHeadroomExpectedRowsThreshold = 48" in k1_ext
    assert "kLlHeadroomRows = 64" in k1_ext
    assert "ll_expected_rows_per_expert >= kLlHeadroomExpectedRowsThreshold" in k1_ext
    assert "ll_expected_rows_per_expert >= 160 ? 64 : 0" not in k1_ext
    assert "kLlSkewGuardRows = 256" not in k1_ext
    assert "K_K1_LL_SKEW_GUARD_ROWS" not in opt_source
    assert "kK1LlSkewGuardRows" not in api_source
    assert "ll_worst_rows_per_expert" in k1_ext
    assert "ll_worst_rows_per_expert" in opt_source
    assert "ll_worst_rows_per_expert" in api_source
    assert "route_capacity_tokens_per_rank * static_cast<int64_t>(num_ranks)" in k1_ext
    assert "int(num_ranks) * int(num_max_tokens)" in opt_source
    assert "static_cast<int64_t>(num_ranks) * num_max_tokens_per_rank" in api_source
    assert "_v3_normal_skew_safe_compact_tiles" in opt_source
    assert "normal_skew_safe_compact_tiles" in api_source
    assert "skew_safe_compact_capacity_tiles" in k1_ext
    assert "const bool default_compact_prebuild = true" in k1_ext
    assert "int launch_wg_n = wg_n" in k1_ext
    assert "hipMemcpyAsync(\n                &active_tiles_host" in k1_ext
    assert "hipStreamSynchronize(stream)" in k1_ext
    assert "prob.n = static_cast<uint32_t>(launch_rows)" in k1_ext
    assert "local_work_size) * wg_m * launch_wg_n" in k1_ext
    assert "int launch_wg_n = wg_n" in k3_normal_ext
    assert "int active_tiles_host = wg_n" in k3_normal_ext
    assert "hipMemcpyAsync(\n                &active_tiles_host" in k3_normal_ext
    assert "hipStreamSynchronize(stream)" in k3_normal_ext
    assert "const int launch_rows = launch_wg_n * 256" in k3_normal_ext
    assert "prob.n = static_cast<uint32_t>(launch_rows)" in k3_normal_ext
    assert "const int gemm_workgroups = wg_m * launch_wg_n" in k3_normal_ext
    assert "effective_asm_done_target" in k3_normal_ext
    assert "max_row_blocks=K_K2_GRAPH_ROW_BLOCKS if k2_active_tiles is not None else None" in opt_source
    assert "active_tiles=k2_active_tiles" in opt_source
    assert "v3_backend != V3_BACKEND_LL" in opt_source
    assert "ll_scratch_capacity_tokens_per_rank" in init_source
    assert "normal_ll_token_threshold()" in init_source
    assert "ll_capacity_tokens_per_rank" in api_source
    assert 'pybind11::arg("ll_capacity_tokens_per_rank") = -1' in api_source
    assert "ll_capacity_tokens: int | None = None" in opt_source
    assert "_check_ll_scratch_capacity(" in init_source
    assert "kProEp8LlMaskedK1MinRowsPerExpert = 128" in k1_ext
    assert "pro_ep8_ll_masked_k1" in k1_ext
    assert "K_PRO_EP8_LL_MASKED_K1_MIN_ROWS_PER_EXPERT = 128" in opt_source
    assert "kProEp8LlMaskedK1MinRowsPerExpert = 128" in api_source
    assert "MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE" not in opt_source
    assert "MEGAMOE_DCU_PRO_LL_COMPACT_ACTIVE" not in api_source
    assert "pro_ll_compact_active_enabled" not in opt_source
    assert "pro_ll_compact_active_enabled" not in api_source
    assert "MEGAMOE_DCU_PRO_LL_COMPACT_HEAD" not in opt_source
    assert "MEGAMOE_DCU_PRO_LL_COMPACT_HEAD" not in api_source
    assert "pro_ll_compact_head_enabled" not in opt_source
    assert "pro_ll_compact_head_enabled" not in api_source
    assert "allow_compact_active=True" in opt_source
    assert "allow_compact_active=False" in opt_source
    assert "actual_max_m <= int(compact_rows_per_expert)" in opt_source
    assert "return (\n                compact_l1_out" in opt_source
    assert "pro_compact_route_weights" in opt_source
    assert "pro_compact_row_combine_ptrs" in opt_source
    assert "compact_route_weights" in opt_source
    assert "compact_row_combine_ptrs" in opt_source
    assert "def k1_ll_masked_prepare_compact_active(" in k1_py
    assert "k1_ll_masked_prepare_compact_active_pack5" in k1_py
    assert "void k1_ll_masked_prepare_compact_active_pack5(" in k1_ext
    assert "pro_ll_masked_compact_stage_active_kernel" in k1_ext
    assert "compact_route_weights.data_ptr<float>()" in k1_ext
    assert "compact_row_combine_ptrs.data_ptr<int64_t>()" in k1_ext
    assert '"k1_ll_masked_prepare_compact_active_pack5"' in k1_ext
    assert "k1_ll_masked_prepare_compact_head" not in k1_py
    assert "k1_ll_masked_prepare_compact_head" not in k1_ext
    assert "k1_ll_masked_copy_compact_head" not in k1_py
    assert "k1_ll_masked_copy_compact_head" not in k1_ext
    assert "k1_ll_masked_groupgemm_pack5_offset" not in k1_ext
    assert "atomicExch(symm_counts + kExperts + 1, 1)" in k1_header
    assert "make_i32_view(route_scratch_i32_offset, {local_experts + 2})" in k1_ext
    assert "symm_counts[expert] > 0 ? symm_counts[expert] : 0" in k1_header
    assert "const int row_capacity = kExperts * m_per_expert" not in k1_header
    assert "for (int idx = global_tid; idx < row_capacity" not in k1_header
    assert "staged_x_scale[row] = kDefaultStagedScale" in k1_header
    assert "if (max_copy_rows_ptr != nullptr)" in k3_header
    assert "if (max_copy_rows <= kCopyRows)" not in k3_header
    assert "const int launched_copy_blocks" in k3_header
    assert "copy_idx += launched_copy_blocks" in k3_header
    assert "const int reduce_idx = static_cast<int>(blockIdx.x) - launched_copy_blocks;" in k3_header
    assert "copy_launch_rows_per_expert = hidden == 7168 ? 128 : 256" in k3_ext
    assert "std::min(full_copy_blocks, copy_blocks) + reduce_blocks" in k3_ext
    assert 'ROUTE_PATTERN_SINGLE_LOCAL_RANK = "single-local-rank"' in test_harness_source
    assert "--route-pattern" in test_harness_source
    assert "topk_idx.copy_(experts.view(1, -1).expand_as(topk_idx))" in test_harness_source


def test_v3_pro_ll_masked_k1_stage_only_path_is_additive():
    k1_header = (K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k1_ext = (K1_FUSED_DIR / "k1_v3_fused_ext.cu").read_text(encoding="utf-8")
    k1_pybind = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k1_py = (K1_FUSED_DIR / "k1_fused.py").read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "V3_K1_LowLatencyStageOnlyKernel" in k1_header
    assert "dcu_megamoe_v3_launch_k1_ll_stage_only_pack5" in k1_ext
    assert "ll_stage_only" in k1_pybind
    assert "ll_return_staged" in k1_pybind
    assert "DCU_MEGAMOE_V3_LAUNCH_K1_LL_PURE_FOR_PRO_SHAPE" not in k1_ext
    assert "ll_pure_masked_weight_layout" not in k1_ext
    assert "ll_pure_groupgemm" not in k1_pybind
    assert "ll_pure_block_n" not in k1_pybind
    assert "hidden == 7168 && l1_rows == 6144" in k1_pybind
    assert (
        "DCU_MEGAMOE_V3_LAUNCH_K1_LL(EXPERTS, N, K, 48, 64, false, false)"
        in k1_ext
    )
    assert "PRO_LL_MASKED_L1_ASM_CO" in k1_py
    assert "deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.co" in k1_py
    assert "PREBUILT_CODE_OBJECTS" in setup_source
    assert "'prebuilt'," in setup_source
    assert "'gfx938'," in setup_source
    assert "deepgemm_groupgemm_masked_fp8_marlin_256x64x128_TN_BF16_WGM8.s" not in setup_source
    assert "k1_ll_masked_groupgemm_pack5" in k1_pybind
    assert "kProLlMaskedL1AsmKernelName" in k1_pybind
    assert "DEEPGEMM_FP8_FP8_BF16_PERCHANNEL_MARLIN_ASM_TN_" in k1_pybind
    assert "MT256x64x128_WGM8_GROUPGEMM_MASKED" in k1_pybind
    assert "k1_ll_masked_groupgemm(" in opt_source
    assert "import deepgemm" not in opt_source
    assert "m_grouped_fp8_gemm_nt_masked" not in opt_source
    assert "ll_pro_masked_fused_groupgemm" not in k1_pybind
    assert "ll_pro_masked_fused_groupgemm" not in k1_py


def test_dcu_megamoe_build_is_incremental_by_default():
    setup_source = SETUP_PATH.read_text(encoding="utf-8")
    build_script = BUILD_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "def _generated_file_current(" in setup_source
    assert "cached_dst = project_path(rel_dst)" in setup_source
    assert "Skipping up-to-date opt asm code object" in setup_source
    assert "_copy_file_if_needed(cached_dst, dst)" in setup_source
    assert "def _remove_objects_stale_against_headers(" in setup_source
    assert "Removing stale object after header update" in setup_source
    assert "def build_extensions(self):" in setup_source
    assert "_remove_objects_stale_against_headers(self.build_temp, self.extensions)" in setup_source
    assert 'rm -rf "$build_dir"' not in build_script
    assert "-delete" not in build_script
    assert "-name '*.co'" not in build_script
    assert "-name '*.o'" not in build_script
    assert "-name '*.hip'" not in build_script
    assert 'rm -f megamoe/_C*.so' not in build_script
    assert "verify_fresh_artifact" not in build_script
    assert "build_epoch" not in build_script
    assert "sync_built_shared_objects" in build_script
    assert "bdist_wheel --skip-build" in build_script
    assert "--inplace" not in build_script
    assert "verify_shared_object" in build_script
    assert "verify_code_object" in build_script


def test_v3_normal_graph_runtime_work_is_limited_without_d2h():
    k1_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k2_ext = (ROOT / "megamoe" / "dcu_megamoe_opt" / "K2_fused" / "k2_fused_ext.cu").read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")

    assert "runtime_limited_init" in k1_ext
    assert "const int init_rows = runtime_limited_init != 0 ? active_rows : capacity_rows;" in k1_ext
    assert "runtime_num_tokens == nullptr ? 0 : 1" in k1_ext
    assert "runtime_num_tokens != nullptr\n            ? 12" in k1_ext
    assert "if (!has_actual_m && active_tiles != nullptr && active_tile_m > 0)" in k2_ext
    assert "logical_row += static_cast<int>(gridDim.x)" in k2_ext
    assert "dim3(launch_blocks)" in k2_ext
    assert "K_K2_GRAPH_ROW_BLOCKS = 8192" in opt_source
    assert "active_tiles=k2_active_tiles" in opt_source
    assert "state.scratch.k1_active_tiles if v3_backend != V3_BACKEND_LL else None" in opt_source


def test_public_capacity_token_and_graph_backend_contract_is_explicit():
    api_source = (ROOT / "megamoe" / "__init__.py").read_text(encoding="utf-8")
    c_api_source = MEGA_DCU_API_PATH.read_text(encoding="utf-8")
    c_kernel_source = MEGA_DCU_KERNEL_PATH.read_text(encoding="utf-8")
    test_source = (
        ROOT / "megamoe" / "dcu_megamoe_opt" / "tests" / "test_mega_moe_dcu.py"
    ).read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")

    assert "megamoe_backend: str = V3_BACKEND_NORMAL" in api_source
    assert "graph: bool = False" in api_source
    assert "capacity_num_tokens: Optional[int] = None" in api_source
    assert "from .dcu_megamoe_opt.v3_layout import" in api_source
    assert "flatten_pack5_weight" in api_source
    assert "flatten_pack5_weight_asm_normal" in api_source
    assert "def mega_moe_pre_dispatch(" in api_source
    assert "num_tokens: int" in api_source
    assert 'm.def("mega_moe_pre_dispatch"' in c_api_source
    assert 'pybind11::arg("num_tokens"))' in c_api_source
    assert "mega_moe_pre_dispatch_fp8_channelwise_kernel" in c_kernel_source
    assert "stage_topk_route<TopkIdxI64, TopkWeightsBf16>" in c_kernel_source
    assert "mega_moe_pre_dispatch_fp8_channelwise_vec16_4096_kernel" in c_kernel_source
    assert "mega_moe_pre_dispatch_fp8_channelwise_wave4_4096_kernel" in c_kernel_source
    assert "dim3((rows + 3) / 4)" in c_kernel_source
    assert "__builtin_hcu_cvt_pk_fp8_f32" in c_kernel_source
    assert "__builtin_hcu_cvt_f32_bf16" in c_kernel_source
    assert "lightop" not in api_source
    assert "weight8bit_nt_kpack2_marlin_masked" in api_source
    assert "transform_fp8_weights_for_mega_moe_pro_ll_masked_k1" in api_source
    old_pro_helper = (
        "transform_fp8_weights_for_mega_moe_v3_" "pro_ll_masked_k1"
    )
    old_legacy_def = "def " "transform_fp8_weights_for_mega_moe("
    old_legacy_export = '"transform_fp8_weights_for_mega_moe"' + ","
    assert old_pro_helper not in api_source
    assert old_legacy_def not in api_source
    assert old_legacy_export not in api_source
    assert '"ll_pro_masked"' in api_source
    assert "l1_layout != l2_layout and not use_pro_ll_masked_k1" in api_source
    assert 'use_unified_weight_layout = l2_layout == "unified"' in api_source
    assert "from megamoe.dcu_megamoe_opt import v3_layout" not in test_source
    assert "megamoe.flatten_pack5_weight_asm_normal" in test_source
    assert "megamoe.flatten_pack5_weight(l" in test_source
    assert "megamoe.weight8bit_nt_kpack2_marlin_masked" in test_source
    assert "force_unified_layout = env_flag_enabled(UNIFIED_WEIGHT_LAYOUT_ENV)" in test_source
    assert 'if v3_backend == "ll" and pro_shape:' in test_source
    assert 'weight_layout = "unified" if force_unified_layout else "ll_pro_masked"' in test_source
    assert '"ll_pro_masked"' in test_source
    assert 'BASELINE_AUTO = "auto"' in test_source
    assert 'default=BASELINE_AUTO' in test_source
    assert "fused_quantizes_input" not in test_source
    assert "fused_input_quant_in_timed_path" not in test_source
    assert "includes_input_quantization" not in test_source
    assert "megamoe.mega_moe_pre_dispatch(" in test_source
    assert "num_tokens=num_tokens" in test_source
    assert "num_tokens=graph_capture_tokens" in test_source
    assert "sym_buffer.x[:num_tokens]" not in test_source
    assert "sym_buffer.x[:capture_tokens]" not in test_source
    assert "normal_baseline_predispatch_buffers = {}" in test_source
    assert "\n    baseline_predispatch_buffers = {}" not in test_source
    assert "cast_input_for_test_baseline" not in test_source
    assert 'hasattr(ep_buffer, "low_latency_dispatch_fp8")' not in test_source
    assert "dispatch_quantization" not in test_source
    assert "low_latency_dispatch_fp8" not in test_source
    assert "ep_buffer.low_latency_dispatch(" in test_source
    assert "quant_type=2" in test_source
    assert "quant_group_size=0" in test_source
    assert "fp8_round_scale=False" in test_source
    assert "normal_baseline_graph_cache" not in test_source
    assert "def get_ll_masked_baseline_graph(" in test_source
    assert "if baseline_kind == BASELINE_LL_MASKED and not args.cuda_graph_skip_baseline" in test_source
    assert 'fused_execution = f"v3_{v3_backend}_eager"' in test_source
    assert 'graph_execution = f"v3_{v3_backend}_cuda_graph_replay" if args.cuda_graph else "disabled"' in test_source
    assert '"fused_timing_scope": "eager_main_call"' in test_source
    assert '"baseline_timing_scope": baseline_execution' in test_source
    assert '"cuda_graph_requested": bool(args.cuda_graph)' in test_source
    assert '"graph_execution": graph_execution' in test_source
    assert '"includes_host_input_update": False' in test_source
    assert '"baseline_graph_kind": BASELINE_LL_MASKED' in test_source
    assert '"baseline_graph_kind": BASELINE_NORMAL_CONTIGUOUS' not in test_source
    assert "run_selected_baseline(\n                    graph_x_bf16[:local_token]" in test_source
    assert "baseline graph is captured per runtime bucket" in test_source
    assert "baseline_dispatch_tokens = max(" in test_source
    assert "baseline_dispatch_tokens,\n                expected_tokens_per_rank," in test_source
    assert "use_layout_cache=False" in test_source
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

    graph_signature = opt_source.split("def _run_opt_3stage_graph(", 1)[1].split(
        ") -> None:",
        1,
    )[0]
    assert "dispatch_num_tokens" not in graph_signature
    assert "capacity_num_tokens: Optional[int] = None" in graph_signature
    assert "v3_backend: str" in graph_signature

    assert "--megamoe-backend" in test_source
    assert "--cuda-graph" in test_source
    assert "--cuda-graph-single-capture" in test_source
    assert "--k1-only-bench" not in test_source
    assert "--k1-only-ll-block-m" not in test_source
    assert "--k1-only-ll-block-n" not in test_source
    assert "--k1-only-compare-deepgemm" not in test_source
    assert "--k1-only-ablate-mode" not in test_source
    assert "def k1_ll_staged_input_views(" not in test_source
    assert "deepgemm_masked_fp8_gemm(" in test_source
    assert "k1_pure_vs_deepgemm_masked_median_ratio" not in test_source
    assert "ll_pure_masked_weight_layout" not in test_source
    assert "k1_only_effective_ll_block_n" not in test_source
    assert "ll_stage_only" in opt_source
    assert "_run_pro_ll_masked_k1_groupgemm" in opt_source
    assert "def run_k1_only(" not in test_source
    assert "def k1_only_ll_block_m(" not in test_source
    assert "def k1_only_ll_block_n(" not in test_source
    assert "def run_k1_only_epoch(" not in test_source
    assert "k1_only_median_ms_avg_per_rank" not in test_source
    assert "ll_k1_main_call_with_host_epoch_barrier" not in test_source
    assert "ll_k1_pure_groupgemm_call_with_host_epoch_barrier" not in test_source
    assert "dist.barrier(group=group)" in test_source
    assert "ll_block_m = _v3_ll_block_m(" in opt_source
    assert "--opt-3stage" not in test_source
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
        "fill_graph_inputs(graph_capture_tokens)",
        1,
    )[0]
    assert "megamoe_backend=v3_backend" in run_fused_call
    assert "capacity_num_tokens=backend_selector_tokens" in run_fused_call
    assert "dispatch_num_tokens=" not in run_fused_call
    assert "megamoe_backend=v3_backend" in graph_call
    assert "graph=True" in graph_call
    assert "capacity_num_tokens=graph_capture_tokens" in graph_call
    assert "dispatch_num_tokens=" not in graph_call
    assert "max_capture_tokens if args.cuda_graph_single_capture else token" in test_source
    assert "baseline_capacity_token = (" in test_source
    assert "graph_capture_tokens if args.cuda_graph_single_capture else token" in test_source
    assert '"graph_capture_tokens_per_rank": graph_capture_tokens' in test_source
    assert "pro_model_shape" not in test_source
    assert "UNIFIED_WEIGHT_LAYOUT_ENV" in test_source
    assert '"normal" if v3_backend == "normal" else "unified"' in test_source


def test_v3_staged_route_scratch_size_uses_ll_normal_layout():
    init_source = (ROOT / "megamoe" / "__init__.py").read_text(encoding="utf-8")
    api_source = MEGA_DCU_API_PATH.read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")

    assert "dcu_supported_staged_pack5_shape" in api_source
    assert "DeepSeek-V4-Flash" in api_source
    assert "DeepSeek-V4-Pro" in api_source
    assert "hidden=7168" in api_source
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

    assert "def _v3_staged_capacity_rows(" in opt_source
    assert "normal_token_threshold()" not in opt_source
    assert "normal_backend_forced()" not in opt_source
    assert "K_K1_ASM_LAUNCH_ARGS_BYTES = 256" in opt_source
    assert "l1_cols = intermediate_hidden * 2" in opt_source
    assert "k3_out_offset" in opt_source
    assert "k3_out = state.scratch.k3_out[:rows]" in opt_source
    assert "capacity_rows * hidden * K_DTYPE_SIZES[torch.bfloat16]" in opt_source

    for py_name, cpp_name in (
        ("K_K1_ROUTE_TILE_M", "kK1RouteTileM"),
        ("K_K1_LL_ROW_TILE", "kK1LlRowTile"),
        (
            "K_K1_LL_HEADROOM_EXPECTED_ROWS_THRESHOLD",
            "kK1LlHeadroomExpectedRowsThreshold",
        ),
        ("K_K1_LL_HEADROOM_ROWS", "kK1LlHeadroomRows"),
        ("K_K1_ASM_LAUNCH_ARGS_BYTES", "kK1AsmLaunchArgsBytes"),
        ("K_PROB_STORAGE_BYTES", "kProbStorageBytes"),
    ):
        assert source_int(opt_source, py_name) == source_int(api_source, cpp_name)
    assert "K_TAIL_DONE_COUNTER_RING_SLOTS = 16" in opt_source
    assert "kTailDoneCounterRingSlots = kDcuMegaMoeTailDoneCounterRingSlots" in api_source
    assert (
        "K_TAIL_DONE_COUNTER_INTS = 3 * K_TAIL_DONE_COUNTER_RING_SLOTS + 64"
        in opt_source
    )
    assert "kDcuMegaMoeTailDoneCounterInts" in api_source

    for mirrored_name in (
        "ll_capacity_rows",
        "normal_capacity_rows",
        "capacity_rows",
        "route_task_workspace_bytes",
        "tail_signal_addrs_offset",
        "ll_capacity_tokens",
    ):
        assert mirrored_name in api_source
        assert mirrored_name in opt_source
    assert "ll_scratch_capacity_tokens_per_rank" in init_source
    assert "self.ll_scratch_capacity_tokens_per_rank" in init_source
    assert "ll_capacity_tokens_per_rank" in api_source
    assert "ll_capacity_tokens" in opt_source
    assert "capacity_rows * static_cast<int64_t>(hidden)" in api_source
    assert "capacity_rows * hidden" in opt_source


def test_v3_supernode_source_support():
    init_source = (ROOT / "megamoe" / "__init__.py").read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")
    setup_source = SETUP_PATH.read_text(encoding="utf-8")
    python_api_source = (
        ROOT / "megamoe" / "dcu_megamoe_opt" / "csrc" / "python_api_hip.cpp"
    ).read_text(encoding="utf-8")
    layout_source = (
        ROOT / "megamoe" / "dcu_megamoe_opt" / "include" / "mega_moe_dcu" / "layout.cuh"
    ).read_text(encoding="utf-8")
    k1_py = (K1_FUSED_DIR / "k1_fused.py").read_text(encoding="utf-8")
    k1_asm_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k1_ext = (K1_FUSED_DIR / "k1_v3_fused_ext.cu").read_text(encoding="utf-8")
    k1_header = (K1_FUSED_DIR / "k1_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k3_ext = (K3_FUSED_DIR / "k3_v3_fused_ext.cu").read_text(encoding="utf-8")
    k3_header = (K3_FUSED_DIR / "k3_v3_pack5_groupgemm_impl.cuh").read_text(
        encoding="utf-8"
    )
    k3_py = (K3_FUSED_DIR / "k3_fused.py").read_text(encoding="utf-8")

    assert "staged_pack5_shape_supported" in init_source
    assert "K_DSV4_SUPPORTED_EP_RANKS = SUPPORTED_STAGED_EP_RANKS" in opt_source
    assert "V3_LL_DEFAULT_BLOCK_M = 32" in opt_source
    assert "V3_LL_PRO_BLOCK_M = 48" in opt_source
    assert "def _v3_ll_block_m(" in opt_source
    assert ") == (384, 6, 7168, 3072):" in opt_source
    assert "return V3_LL_PRO_BLOCK_M" in opt_source
    assert "return V3_LL_DEFAULT_BLOCK_M" in opt_source
    assert "K1_SUPPORTED_RANKS = SUPPORTED_STAGED_EP_RANKS" in k1_py
    assert "counts[local_experts]" in k1_asm_ext
    assert "const bool default_compact_prebuild = true" in k1_asm_ext
    assert "legacy in-ASM route layout is kept as an" in k1_asm_ext
    assert "? fixed_capacity_tiles" not in k1_asm_ext
    assert "dcu_required_signal_slots" in layout_source
    assert "dcu_split_tail_chunk_signal_slot_base(num_ranks)" in k1_header
    assert "dcu_split_tail_chunk_signal_slot_base(num_ranks)" in k3_header
    assert "signal_addrs[num_ranks + rank]" in k3_header
    assert "V3_K1_LowLatencyMaskedGroupGemmKernel<" in k1_ext
    assert "ll_block_m == 256" not in k1_ext
    assert "pure MT256" not in k1_asm_ext
    assert "dcu_megamoe_v3_launch_k1_ll_pure_groupgemm_pack5" not in k1_ext
    assert "DCU_MEGAMOE_V3_LAUNCH_K1_LL_PURE_FOR_PRO_SHAPE" not in k1_ext
    assert "block_n == 512" not in k1_ext
    assert "V3_K1_LdsPack5PureGroupGemmKernel" not in k1_header
    assert "DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_SHAPE(EXPERTS, 6144, 7168)" in k1_ext
    assert "DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(48)" in k1_ext
    assert "static_assert(kBlockN == 64 || kBlockN == 128 || kBlockN == 256" in k1_header
    assert "static_assert(kNumWarps == 4 || kNumWarps == 8" in k1_header
    assert "__launch_bounds__(kNumWarps * 64, 1)" in k1_header
    assert "bool kSkipDispatch" not in k1_header
    assert "bool kMaskedWeightLayout" not in k1_header
    assert "if constexpr (!kSkipDispatch)" not in k1_header
    assert "if constexpr (kMaskedWeightLayout)" not in k1_header
    assert "static_assert(kExperts <= 64" in k1_header
    assert "V3_K3_LowLatencyMaskedGroupGemmKernel<" in k3_ext
    assert "DCU_MEGAMOE_V3_LAUNCH_LL_FOR_SHAPE(EXPERTS, 7168, 3072)" in k3_ext
    assert "DCU_MEGAMOE_V3_LAUNCH_LL_FOR_EXPERTS(48)" in k3_ext
    assert "static_assert(kExperts <= 64" in k3_header
    assert "open_hip_fabric_handles" in python_api_source
    assert "hsa_ext_rpc_memory_attach" in python_api_source
    assert 'os.getenv("MEGAMOE_DCU_PEER_MEMORY", "ipc")' in init_source
    assert 'peer_memory_mode == "fabric"' in init_source
    assert "build_libraries = ['hsa-runtime64']" in setup_source
    assert "addrs = [0] * (2 * int(num_ranks))" in k3_py
