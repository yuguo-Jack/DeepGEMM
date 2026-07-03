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
    assert "force_compact_prebuild || num_ranks > 8 || local_experts > 32" in k1_asm_ext
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
    assert "max_copy_rows <= kCopyRows" in k3_header
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
        "MEGAMOE_DCU_PRO_WEIGHT_LAYOUT",
        "MEGAMOE_DCU_K2_GLC_SLC_LOAD",
        "MEGAMOE_DCU_REDUCE_COMBINE_GLC_SLC_LOAD",
        "--debug-combine-on-fail",
        "--debug-compare-backends-on-fail",
        "system_fence_after_write",
        "invalidate_before_read",
        "acquire_after_wait",
        "V3 K3 LL no-tail signal path is not wired yet",
    ):
        assert retired not in production_sources
        assert retired not in test_harness_source

    assert not (ROOT / "megamoe" / "dcu_megamoe_v2").exists()


def test_v3_ll_capacity_headroom_covers_256_and_512_exact_buckets():
    k1_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")

    assert "kLlHeadroomExpectedRowsThreshold = 48" in k1_ext
    assert "kLlHeadroomRows = 64" in k1_ext
    assert "ll_expected_rows_per_expert >= kLlHeadroomExpectedRowsThreshold" in k1_ext
    assert "ll_expected_rows_per_expert >= 160 ? 64 : 0" not in k1_ext


def test_v3_normal_graph_runtime_work_is_limited_without_d2h():
    k1_ext = (K1_FUSED_DIR / "k1_fused_ext.cu").read_text(encoding="utf-8")
    k2_ext = (ROOT / "megamoe" / "dcu_megamoe_opt" / "K2_fused" / "k2_fused_ext.cu").read_text(encoding="utf-8")
    opt_source = OPT_PATH.read_text(encoding="utf-8")

    assert "runtime_limited_init" in k1_ext
    assert "const int init_rows = runtime_limited_init != 0 ? active_rows : capacity_rows;" in k1_ext
    assert "runtime_num_tokens == nullptr ? 0 : 1" in k1_ext
    assert "runtime_num_tokens != nullptr\n            ? 12" in k1_ext
    assert "if (!has_actual_m && active_tiles != nullptr && active_tile_m > 0)" in k2_ext
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
    assert "from megamoe.dcu_megamoe_opt import v3_layout" not in test_source
    assert "megamoe.flatten_pack5_weight_asm_normal" in test_source
    assert "megamoe.flatten_pack5_weight(l" in test_source
    assert "megamoe.weight8bit_nt_kpack2_marlin_masked" in test_source
    assert 'BASELINE_AUTO = "auto"' in test_source
    assert 'default=BASELINE_AUTO' in test_source
    assert "fused_quantizes_input" not in test_source
    assert "fused_input_quant_in_timed_path" not in test_source
    assert "includes_input_quantization" not in test_source
    assert "megamoe.mega_moe_pre_dispatch(" in test_source
    assert "num_tokens=num_tokens" in test_source
    assert "num_tokens=capture_tokens" in test_source
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
    assert "capacity_num_tokens" not in graph_signature
    assert "v3_backend: str" in graph_signature

    assert "--megamoe-backend" in test_source
    assert "--cuda-graph" in test_source
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
    assert "pro_model_shape" not in test_source
    assert "MEGAMOE_DCU_PRO_WEIGHT_LAYOUT" not in test_source
    assert "UNIFIED_WEIGHT_LAYOUT_ENV" in test_source
    assert '"normal" if v3_backend == "normal" else "unified"' in test_source


def test_v3_staged_route_scratch_size_uses_ll_normal_layout():
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
    ):
        assert mirrored_name in api_source
        assert mirrored_name in opt_source
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
    assert "K1_SUPPORTED_RANKS = SUPPORTED_STAGED_EP_RANKS" in k1_py
    assert "counts[local_experts]" in k1_asm_ext
    assert "num_ranks > 8 || local_experts > 32" in k1_asm_ext
    assert "? fixed_capacity_tiles" in k1_asm_ext
    assert "dcu_required_signal_slots" in layout_source
    assert "dcu_split_tail_chunk_signal_slot_base(num_ranks)" in k1_header
    assert "dcu_split_tail_chunk_signal_slot_base(num_ranks)" in k3_header
    assert "signal_addrs[num_ranks + rank]" in k3_header
    assert "V3_K1_LowLatencyMaskedGroupGemmKernel<" in k1_ext
    assert "DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_SHAPE(EXPERTS, 6144, 7168)" in k1_ext
    assert "DCU_MEGAMOE_V3_LAUNCH_K1_LL_FOR_EXPERTS(48)" in k1_ext
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
