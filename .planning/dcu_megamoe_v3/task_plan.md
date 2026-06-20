# DCU MegaMoE V3 迁移计划

## 目标
DCU MegaMoE V3 已转为主路径；当前目标是把 DCU MegaMoE 生产代码、pybind/API、baseline preprocess/postprocess、测试、脚本和相关资源集中到 `megamoe/dcu_megamoe_opt/`，并把旧 `large_opt` 命名收敛为 `opt`。构建不再依赖 `csrc/` 或 `deep_gemm/include/` 下的 DCU MegaMoE 专属文件。

当前生产路径：

- K1: `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1`
- K3: `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE`
- K3 tail reduce 版: `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE`

生产 API 不再读取 V3/backend env；测试脚本用 `MEGAMOE_DCU_BACKEND=auto|ll|normal` 与 `MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD` 模拟框架侧选择。历史 `USE_MEGAMOE_V3` / V3 backend gate 已退役。

## 2026-06-18 opt 自包含规整目标

- ✅ 将 `megamoe/dcu_megamoe_large_opt/` 重命名为 `megamoe/dcu_megamoe_opt/`，`megamoe/large_opt.py` 重命名为 `megamoe/opt.py`，并统一函数/缓存/env 命名。
- ✅ 将 `csrc/apis/mega_dcu.hpp`、`csrc/python_api_hip.cpp`、`csrc/kernels/mega_moe_baseline_hip.cu` 迁入 `megamoe/dcu_megamoe_opt/csrc/`。
- ✅ 将 `deep_gemm/include/deep_gemm/{layout,comm,common,mma}` 下 DCU MegaMoE 专属头迁入 `megamoe/dcu_megamoe_opt/include/mega_moe_dcu/`，生产 include 改为 `<mega_moe_dcu/...>`。
- ✅ 将 DCU MegaMoE 测试、脚本迁入 `megamoe/dcu_megamoe_opt/{tests,scripts}/`，外层 DCU 专属残留已清理；两张过时 pipeline PNG 由用户确认删除，不再迁入 assets。
- [ ] 更新 README、setup package data、source guard，并通过本地 compile/pytest source guard、远端 `build_ext --inplace`、代表性 smoke 与 README smoke。

## 2026-06-20 LL overlap / K1+K2 fusion active items

- ✅ K3 LL tail reduce peer-wait sharing：只让一个 reducer block 等跨 rank tail signal，其他 reducer block 等本地 peer-ready ring；不新增 kernel，不改 normal ASM done-counter 前两个槽 ABI。
- ✅ K2 compact-row fast path：当 K1 LL 提供 `actual_m` 时，K2 跳过 padded/legacy row 防御检查，复用 K1 compact 合同减少 metadata load/branch；normal/legacy 路径保持原逻辑。
- ✅ 已完成远端 build/source guard/LL tail+no-tail eager+graph smoke；更激进的 per-chunk readiness 或 K1+K2 真融合继续作为证据驱动 backlog，不能在没有 profiler/正确性闭环时直接进生产。
- ✅ per-row chunk-ready 与 naive K1+K2 fused-K2 原型已反证并撤出生产代码；后续若重启，必须改成 per-expert/per-token-chunk readiness 或 paired-N/row-wise amax 合同级设计。

## 2026-06-16 V3 转正目标更新

本节 supersedes 上述历史 gate 目标：V3 不再作为 `USE_MEGAMOE_V3` / `MEGAMOE_DCU_V3_BACKEND` 隔离实验路径，而要转为 DCU MegaMoE 主路径。

- retained production route：对外 DCU MegaMoE 接口和单测参数尽量沿用原有使用方法；除权重必须使用 V3 pack5 layout 外，框架侧不应因为 V3 转正而新增必需调用参数。
- backend auto policy：按实际运行 tokens per rank 分流，`<=256` 走 V3 LL，`>256` 走 V3 normal。graph capture bucket 可以大于实际 replay tokens，backend 选择不能简单使用 capture 上限；uneven 场景必须按全局/dispatch token bucket 保证所有 rank 选同一 backend。
- legacy route retirement：之前 big fused 主路径退役并由 LL 覆盖；之前 legacy staged fused 主路径退役并由 V3 normal 覆盖。旧路径不再作为生产 fallback、默认 benchmark 对照或性能优化对象。
- env cleanup：V3 转正后删除 `USE_MEGAMOE_V3` 与 `MEGAMOE_DCU_V3_BACKEND` 对生产路径的影响；README 中移除对应实验开关说明；生产 API 不再读取 backend env，测试脚本用 `MEGAMOE_DCU_BACKEND=auto|ll|normal` 与 `MEGAMOE_DCU_NORMAL_LL_TOKEN_THRESHOLD` 模拟框架层选择。
- memory cleanup：big fused 删除后，Phase 8 的 `route_scratch` / symm buffer 空间优化从 optional backlog 升级为必做，实现 V3-only staged scratch layout，减少历史兼容包袱。
- final gate：完成后必须重跑 README 覆盖命令、source tests、远端 build、V3 LL/normal 功能/精度/性能矩阵，并确认各 size 性能符合历史预期。

## 当前状态
- V2 独立路径只保留为历史证据；`DG_BUILD_MEGAMOE_V2_EXT`、`megamoe/dcu_megamoe_v2/`、`csrc/kernels/dcu_megamoe_v2/`、`tests/test_dcu_megamoe_v2.py` 和相关 setup/package/test 引用已清理，V3 决策不再依赖 V2 代码路径。
- K1/K3 V3 LL compute core 仍来自 stage-owned C pack5 fused 实现；K1/K3 V3 normal 生产路径已切到 isolated ASM-pack5 pair（no-tail 与 tail-reduce），不再依赖 normal C/aicc kernel。
- 长期 staged fused 保留边界进一步收敛：后续预计只保留 V3 实现，LL 负责小 token，normal 负责大 token；非 V3 legacy staged fused 仅作为当前兼容/回归观察对象，不再作为长期优化方向。
- V3 normal no-tail/tail、uneven 和 cuda graph 已完成覆盖点验证；2026-06-16 发现的 normal graph multi-token replay 间歇性 correctness 问题已定位为 V3 normal K1 ASM-pack5 graph flag/meta reset 遗漏，现已修复并通过 no-tail replay=5、tail repeat 4/4 和 LL graph 回归验证。
- K3 normal C/aicc raw 路线已正式遗弃为生产实现：no-tail C scalar/store 路线被 ASM-pack5 取代，tail C in-kernel signal/reducer 路线停止修复；相关代码、编译开关、env gate 和临时脚本只保留历史记录或 retired stub。
- 当前 fused LL staged correctness 已完成固定 32/128、uneven、小 bucket graph 与 Phase 10 1024 capture graph sweep；V3 LL C pack5 extension 默认编译。Phase 10 必做刷数中 normal eager 全矩阵已通过；LL capture/eager 1024 correctness bug 已定位为 K1 LL per-expert row capacity 过紧并修复；LL graph capture=1024 的 replay 性能问题已进一步定位为 K3 LL graph 误按 capture rows 做 GEMM work，现已改为消费 K1 runtime per-expert row counts，完整 `8,32,64,128,256,512,1024` no-tail/tail graph sweep 已补测通过且小 token replay 恢复到预期区间。
- 2026-06-15 清理后远端验证已完成：卡空闲、同步成功、`build_ext --inplace` 通过、source pytest 6/6 通过，8 卡矩阵覆盖 V3 normal tail/no-tail、normal uneven graph、LL tail/no-tail、以及非 V3 legacy opt smoke。
- 现有 staged fused 主入口是 `megamoe/opt.py`，K2 和外部编排逻辑尽量复用。
- V3 normal backend 编译策略：normal 不再编译 C/aicc extension；normal K1/K3 使用 ASM-pack5 code object，LL C pack5 extension 默认按 hipcc/现有 setup 逻辑编译。
- 现有 K1/K3 staged fused Python wrapper 分别在：
  - `megamoe/dcu_megamoe_opt/K1_fused/k1_fused.py`
  - `megamoe/dcu_megamoe_opt/K3_fused/k3_fused.py`
- 现有 ASM host launch / pybind 分别在：
  - `megamoe/dcu_megamoe_opt/K1_fused/k1_fused_ext.cu`
  - `megamoe/dcu_megamoe_opt/K3_fused/k3_fused_ext.cu`

## 已确认设计点
1. V3 门控采用两层条件：`MEGAMOE_DCU_USE_OPT_3STAGE=1` 先强制进入现有 staged fused opt，`USE_MEGAMOE_V3=1` 再按 backend 替换 K1/K3：LL 走 V3 C pack5 fused kernel，normal 走 V3 isolated ASM-pack5 code object。
2. LL/normal 后端选择使用新环境变量 `MEGAMOE_DCU_V3_BACKEND=ll|normal`，不再沿用 `MEGAMOE_DCU_V2_BACKEND`。
3. V3 直接集成到 `megamoe/dcu_megamoe_opt`，kernel 代码剥离到已有 `K1_fused` / `K3_fused` 文件夹；K1 不使用 `dcu_megamoe_v2` 下的实现作为 V3 core，而是从 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 的原始 pure kernel 迁入并扩通信。
3a. `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` 是原始 DeepGEMM 汇编文件，也是原 DCU MegaMoE K1/K3 fused ASM 修改的基础；V3 扩通信时可参考它和现有 fused ASM 的通信语义、插入点与 wait/signal 处理，但 GEMM 主干仍以 pure C 5pack kernel 为主体，不能为了照搬 ASM 包装破坏 pure C 的 load/compute/store 流水。
3b. V3 K1/K3 fused normal/LL 主 kernel 的改动路径必须优先来自“原始 groupgemm ASM vs 原 DCU MegaMoE K1/K3 fused ASM”的差异图，再映射到 pure C normal/LL 5pack 主体上；V2 不能作为主 kernel 结构来源，只能作为接口/行为的次级参考。除 weight layout 差异和通信语义注入外，尽量复用现有 DCU MegaMoE 周边实现，只有个别无法复用的边界代码才重写。
3c. 当 V3 K1/K3 fused normal/LL 主 kernel 的实现选择不确定时，默认判定顺序固定为：先看原始 groupgemm ASM 与原 DCU MegaMoE K1/K3 fused ASM 的差异，再映射到 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` / pure C 5pack 主体；若该路径和 V2 fused/real-flow 结构冲突，以 ASM-diff + pure C 主体为准，V2 实现不参与生产 kernel 结构决策。
3d. CUDA MegaMoE 参考优先使用本工程内实现，而不是外部弱匹配资料：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`、`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`。借鉴点应转译为 DCU staged fused 合同中的 epilogue remote combine、dispatch metadata、scheduler/count 缓存和 overlap 思路，不能直接照搬 CUDA/TMA/SM100 机制。
4. 外部单测尽量复用test_mega_moe_dcu.py，命令行参数保持兼容；只在必要处增加 V3 env 组合和性能对照。
5. V3 权重 layout 与原始 DCU MegaMoE 不同，layout/transform 相关逻辑必须增量区分，不能让默认路径继承 V3 layout 假设。
6. `MEGAMOE_DCU_V3_BACKEND` 未设置但 V3 开启时，默认使用 `normal`；显式设置 `ll` 时才走 LL backend。
6a. backend 与 token 档位绑定：LL 骨架面向 tokens per rank `<512`，normal 骨架面向 tokens per rank `>=512`。
7. V3 权重 layout 差异在单测/离线准备阶段提前转好；DCU MegaMoE 执行和 bench 流程不引入权重处理 kernel 或额外 runtime launch。
8. K1 前的 rank barrier kernel 第一版 V3 先保留，功能和 correctness 跑通后再作为独立 A/B 项尝试移除；若证据不足，保留原 barrier。
9. uneven tokens 和 cuda graph 功能最终需要对齐现有 DCU MegaMoE，不作为可永久缺失项。
10. 除替换 K1/K3 计算 kernel 本身外，不在 `dcu_megamoe_opt` 集成中引入额外 kernel；其他单测和参数尽量复用现有入口。
11. 实验首选 8 卡；如果部分卡显存被占但仍可继续跑，就继续执行；如果所有卡都不可用，则持续监控显存和进程状态，卡空后按计划恢复工作。
12. 工作流必须持续更新 `.planning/dcu_megamoe_v3/task_plan.md`：完成项用 ✅，未完成必做项用 `[ ]`，遗弃项用 🚫，可选 backlog 用 🧭；当前不保留无明确 owner 的悬挂进行中项。
13. pure 5pack C normal/LL 实现本质上不带通信；V3 不是直接搬 pure，而是先分析现有 DCU MegaMoE 主要通信/metadata kernel，再把对应 pure groupgemm core 扩成带 dispatch-pull、combine/tail-reduce 语义的 fused kernel，并控制相对 pure 的性能劣化。
14. 首选实现策略是把通信操作彻底隐藏到 GEMM pipeline 内：K1 将 dispatch-pull/route metadata 融进 global load 与 tile scheduling，K3 将 combine/tail-reduce 融进 epilogue/store/reduce 阶段，避免在 groupgemm 前后形成独立通信步骤。
15. 通信融合不能破坏原 5pack C kernel 内已有 GEMM 流水；优先保持原 load/compute/store pipeline 和 tile 组织，只在自然插入点嵌入通信语义，目标是融合完通信后性能尽可能少劣化。
15a. V3 的实际 staged 通信链路耗时必须按 K1/K3、LL/normal 分项持续对照对应原始 C 5pack groupgemm/pure kernel；优化目标不是只过 e2e baseline gate，而是让 fused 通信链路尽可能逼近对应 pure C groupgemm 的耗时，所有明显 delta 都要记录、解释并优先优化。
16. correctness 接入按定位友好的顺序推进：先只替换 V3 K1 fused，K2 复用原逻辑且 K3 仍走原 ASM；该阶段只称为 K1-only staged correctness/probe，用于隔离 K1 dispatch-pull、metadata、row layout 与后续 K2/K3 消费合同。只有 K1/K2/K3 V3 功能链路都接入并覆盖 tail-reduce/no-tail、eager/graph、uneven tokens 后，才称为完整 e2e correctness。
17. K1-only staged gate 必须显式处理权重 layout 差异：L1 权重使用 V3 pack5 layout，K3 仍走原 ASM layout 的 L2 权重；不能把整套权重一次性转成 pack5 后喂给原 K3 ASM。接入 K3 V3 no-tail 时，才把 L2 切到 V3 pack5 layout。
18. K1-only staged correctness 只有在 V3 K1 主 kernel 本身已融合 dispatch-pull、route metadata、row_combine_ptrs、output_index 和 stats 语义后才有效；当前 raw rowptr GEMM + 独立 route prebuild kernels 只能做低层 smoke，不能用于宣称功能对齐。
19. normal backend 的生产性能路径采用隔离 ASM-pack5 code object：K1/K3 no-tail 和 tail-reduce 均默认使用 ASM-pack5；normal C/aicc raw K1/K3 不再作为生产、fallback 或性能对照计划推进。
20. 长期 staged fused 路径按 V3-only 收敛：LL backend 面向小 token，normal backend 面向大 token；legacy staged fused 非 V3 实现后续进入退役/删除评估，当前只保留兼容性和过渡期回归验证。

## 成功标准
- `USE_MEGAMOE_V3` 未设置、为 false，或 `MEGAMOE_DCU_USE_OPT_3STAGE` 不是强制 `1` 时：
  - `MEGAMOE_DCU_USE_OPT_3STAGE` 原始代码逻辑不变，单测参数功能都不受影响；
  - K1/K3 仍加载原 ASM code object；
  - 现有 correctness/perf 测试行为不变。
- `MEGAMOE_DCU_USE_OPT_3STAGE=1` 且 `USE_MEGAMOE_V3=1` 时：
  - K1 LL 使用 V3 C pack5 dispatch-pull + L1 groupgemm；K1 normal 使用 isolated ASM-pack5 dispatch-pull + L1 groupgemm；
  - K2 继续复用现有 `swiglu_quant_channelwise_out`；
  - K3 LL 使用 V3 C pack5 L2 groupgemm + combine/tail-reduce；K3 normal 使用 isolated ASM-pack5 L2 groupgemm + combine/tail-reduce；
  - `K3_USE_ASM_TAIL_REDUCE=1` 对齐 fused tail reduce 功能；
  - `K3_USE_ASM_TAIL_REDUCE=0` 对齐原 K3 combine + 外部 `reduce_local_combine` 功能；
  - 输出 correctness 与原 baseline 对齐，`max_abs <= 1e-3`，统计项不回退。
- baseline 只作为正确性验证 oracle；V3 相对 baseline 性能更好不具有决定意义，也不作为最终成功标准。
- 外部 staged fused 编排尽可能复用 `megamoe/opt.py`，只在必要处做 V3 分流。
- 权重 layout 分流只影响 V3 测试/离线准备；原始 DCU MegaMoE layout、CLI 参数和单测入口保持兼容。
- 执行路径和 benchmark 路径只消费已经是 V3 pack5 layout 的权重，不新增权重转换 kernel、权重重排 kernel 或额外 runtime launch。
- 不改 V2 pure denominator / 不继续调 V2 性能；V2 只作为接口行为参考，不能作为 V3 K1 compute core。
- 不引入默认路径行为变化，不修改现有大 fused / baseline path。
- V3 fused 相对 5pack C pure normal/LL 的劣化要尽量小：优先保留 pure groupgemm 内层结构，只在必要边界补通信、metadata 和 combine 语义。
- V3 fused 实际通信链路的 K1/K3 分段耗时必须尽可能逼近对应原始 C 5pack groupgemm/pure kernel；Phase 6 需要记录 K1/K3 LL/normal 的 pure-vs-fused delta，并把明显差距作为优先优化项，而不是只用 e2e baseline 结果判断。
- 通信融合应尽量 pipeline-internal：不能把 dispatch/combine 当成额外前处理/后处理阶段去堆 launch 或大规模内存 pass。
- 通信融合不能打乱原 5pack C GEMM 流水：load/compute/store 的核心调度、tile 组织和数据复用路径应尽量保持，通信只作为 pipeline 内的附加语义。
- 如果 K1/K3 pure-vs-fused 劣化明显，不等到最终性能阶段，提前进入局部优化；但最终验收仍以 Phase 6 Performance gate 为准。
- 性能门槛：
  - tokens per rank `<512` 时，V3 LL 必须快于 `MEGAMOE_DCU_USE_OPT_3STAGE=0`；
  - tokens per rank `>=512` 时，V3 normal 必须快于 `MEGAMOE_DCU_USE_OPT_3STAGE=1` 但 `USE_MEGAMOE_V3=0` 的原 staged fused 路径。
  - Phase 6 benchmark 固定档位：32/128 使用 LL，1024/4096 使用 normal。
  - baseline 性能只记录作参考，不参与是否通过 Phase 6 的判断。

## 遗弃/退役范围
- [🚫] V3 normal C/aicc raw K1/K3 不再作为生产路径、默认 fallback、性能优化目标或后续计划项；历史 Phase 2/3/6 中 normal C/aicc 相关 ✅ 项仅表示曾经完成过 bring-up、诊断或反证。
- [🚫] `DG_BUILD_MEGAMOE_V3_NORMAL_AICC`、normal raw backend build、`MEGAMOE_DCU_V3_K1_ASM_PACK5`/`MEGAMOE_DCU_V3_K3_ASM_PACK5` opt-in/out、`MEGAMOE_DCU_V3_NO_TAIL_SIGNAL`/`NO_TAIL_SYNC`/`K2_SYSTEM_FENCE`/`REDUCE_ACQUIRE`/`BARRIER_ACQUIRE` 等诊断 env 已退役；当前代码不再消费这些 env。
- [🚫] `DG_BUILD_MEGAMOE_V2_EXT` 与 `megamoe/dcu_megamoe_v2/` 整体进入退役范围；后续实现清理时应删除 V2 extension build、package data、导入/测试/脚本引用，只保留 planning 中的历史记录。
- [✅] 当前保留的 C extension 范围是 V3 LL pack5，且默认编译；normal 的验证和性能回归以 ASM-pack5 eager/graph、no-tail/tail、uniform/uneven 为准。

## 用户确认的后续清理边界
- [✅] V2 整体已清：删除 `DG_BUILD_MEGAMOE_V2_EXT`、`megamoe/dcu_megamoe_v2/`、`csrc/kernels/dcu_megamoe_v2/`、setup/package/test/script 中的 V2 build/package 引用；历史结论只留在 planning。
- [✅] V3 normal C/raw 残留已清：K1/K3 pack5 header 中不会再 launch 的 normal C kernel body、fixed-route bring-up helper，以及 raw/stub/availability 相关断言和脚本已删除或退役。
- [✅] 诊断 pybind 已清：删除 `k3_v3_ll_reference` 及调用它的 debug scripts/source guard。
- [✅] 调试 env 已清：删除 `MEGAMOE_DCU_K3_DEBUG_LAUNCH`、`MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC*`、`MEGAMOE_DCU_V3_STAGE_TIMING`；`MEGAMOE_DCU_V3_LL_BLOCK_M` 已在固定 block M=32 后移除。
- [✅] dormant API 参数已清：删除 K3 `invalidate_before_read`、K3 `acquire_after_wait`、K2 `system_fence_after_write`，它们不属于当前 V3 生产路径。
- [✅] K1 V3 normal ASM 边界已重整到 Python wrapper + pybind entry 层：新增 `k1_symm_fused_l1_v3_asm_pack5()` / graph wrapper 和 C++ pybind `k1_symm_fused_l1_v3_asm_pack5(...)`，调用点不再通过原 K1 ASM pybind 或泛化 backend 名称表达 V3 normal；底层 ASM launch 机制通过内部 helper 复用以避免行为和性能漂移。
- [✅] `setup.py` 已回到最小 delta：保留 V3 必需源文件、pack5 ASM code object、`*.cuh` package data；清掉历史 source-list 临时变量和 V2/raw gate。
- [✅] `opt.py` 已清掉开发期 sync/timing，只保留清晰 V3 backend 分流：normal ASM-pack5，LL C pack5。
- [✅] `megamoe/__init__.py` 已检查：`_v3_normal_symm_warmup_enabled` 属于受 shape/backend/env 严格限制的 warmup alloc 逻辑，不是调试路径；为避免 first-run/perf 回归暂时保留。
- [✅] `megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` 已缩成轻量合同测试：V3 gate/backend、pack5 layout helper、setup surface、关键 env/retired symbol 默认不误触发。
- [✅] `hygon_tmp/debug` 已清理为当前生产路径脚本为主；normal C/raw、reference/pure、旧 A/B 方向脚本已删除或改到当前 pybind/env。

## 当前验收快照（2026-06-15）
- [✅] 远端环境：`hg@10.17.176.11` / `sglang_megamoe` 可用，8 张 HCU 空闲；验证后无 KFD 进程残留，显存/计算均为 0%。
- [✅] 同步：本地 `setup.py`、`megamoe`、`tests` 已同步到 `/workspace/DeepGEMM`；远端旧 V2、旧 V3 stub、旧非 pack5 header 残留已删除。
- [✅] 构建：容器内 `python3 -m compileall setup.py megamoe megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` 通过；`PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` 6/6 通过；清理 build 后 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过。
- [✅] 源码残留：生产源码和主测试中 `DG_BUILD_MEGAMOE_V2_EXT`、`dcu_megamoe_v2`、`k3_v3_ll_reference`、调试 env、旧 raw/stub/Pure/非 pack5 命名均无残留；剩余历史字符串仅在轻量 source-guard 负向断言中。
- [✅] 8 卡功能/性能 smoke：结果保存在远端 `hygon_tmp/debug/full_validation_20260615_161234/`。
  - V3 normal tail 32: correct, fused `1.526879 ms`, baseline `2.805279 ms`, speedup `1.8373x`。
  - V3 normal no-tail 32: correct, fused `1.504639 ms`, baseline `2.814119 ms`, speedup `1.8703x`。
  - V3 normal tail uneven `17,31,48,64,79,96,112,127`: correct, fused `1.541459 ms`, baseline `2.823078 ms`, speedup `1.8314x`。
  - V3 normal tail graph uneven: correct, eager `1.539139 ms`; graph replay 32/96/128 token 分别 `1.413819/1.428639/1.436719 ms`，graph 不劣化。
  - V3 LL no-tail graph uneven: correct, eager `0.774360 ms`; graph replay 32/96/128 token 分别 `0.661760/0.735480/0.738820 ms`，graph 不劣化。
  - V3 LL tail graph uneven: correct, eager `0.819420 ms`; graph replay 32/96/128 token 分别 `0.721340/0.790440/0.806820 ms`，graph 不劣化。
  - V3 LL tail 32: correct, fused `0.709419 ms`, baseline `2.759899 ms`, speedup `3.8904x`。
  - V3 LL no-tail 32: correct, fused `0.689540 ms`, baseline `2.761178 ms`, speedup `4.0044x`。
  - V3 normal no-tail 4096: correct, fused `5.808258 ms`, baseline `9.460317 ms`, speedup `1.6288x`。
  - V3 normal tail 4096: correct, fused `5.838198 ms`, baseline `9.513697 ms`, speedup `1.6296x`。
  - 非 V3 legacy opt tail 32: correct, fused `1.931739 ms`, baseline `2.785199 ms`, speedup `1.4418x`。
- [✅] 本轮结论：V3 production cleanup 没有伤及主要生产路径或 legacy opt smoke；normal/LL graph replay 相比 eager 无劣化；normal 4096 tail/no-tail 和 LL tail/no-tail 均正常。后续只保留性能余量类 backlog，不再把 V2/normal C/raw/stub/诊断 env 作为计划项推进。

## 计划收敛规则（2026-06-15）
- `✅ complete for production gate` 表示当前保留生产路径已实现并完成本轮远端验证；历史更大矩阵或纯性能优化不再阻塞该阶段。
- `🚫 历史项/遗弃` 表示曾经用于 bring-up 或反证的 V2、V3 normal C/aicc/raw、stub、debug env、diagnostic pybind 路线；不再作为后续计划推进。
- `[ ]` 表示用户明确点名但尚未完成的必做验证/实现项；完成前不能降级成 optional backlog，也不能用已有 smoke 结果替代。
- `🧭 optional backlog` 表示只在后续有明确性能证据或用户重新点名时才恢复的诊断/优化想法；不是当前未完成任务。
- 当前 active 生产边界固定为：V3 normal = isolated ASM-pack5 K1/K3 no-tail/tail eager/graph；V3 LL = stage-owned C pack5 K1/K3 no-tail/tail eager/graph。
- 长期 retained staged fused 边界进一步收敛为 V3-only：LL 负责小 token，normal 负责大 token；legacy staged fused 非 V3 路径后续随 big fused 一并评估退役。

## 分阶段计划

### Phase 0: 代码梳理与接口对齐
- [✅] 逐段梳理 `opt.py` 的 eager 与 graph staged flow：K1 输入输出、K2 依赖、K3 tail reduce 两种分支。
- [✅] 梳理 K1 ASM wrapper/ext 的真实接口：route_scratch 切片、`row_combine_ptrs`、`route_weights`、`m_indices`、`output_index`、`staged_x`、stats。
- [✅] 梳理 K3 ASM wrapper/ext 的真实接口：`row_combine_ptrs`、`output_workspace`、`prob_storage`、tail signal、tail reduce、active_tiles/graph 参数。
- [✅] 梳理 V2 C pack5 LL/normal K1/K3 已有能力与缺口，形成接口对照表。
- 状态：✅ complete

### Phase 1: V3 开关与后端选择设计
- [✅] 增加 `USE_MEGAMOE_V3` 解析函数，并确保只有 `MEGAMOE_DCU_USE_OPT_3STAGE=1` 时才允许进入 V3 分支。
- [✅] 实现 `MEGAMOE_DCU_V3_BACKEND=ll|normal`；未设置时默认 `normal`，显式 `ll` 才走 LL。
- [✅] 明确 V3 weight contract：runtime/bench 只接收已提前转换好的 V3 pack5 权重，V3 分支不做权重处理 kernel。
- [✅] 单测 fixture/setup 侧提前准备 V3 layout 权重，防止默认路径误用 V3 pack5 layout。
- [✅] 加 source-level 测试保证默认路径仍引用原 ASM wrapper，V3 开启才引用新实现。
- 状态：✅ complete

### Phase 2: K1 V3 功能迁移
- [✅] 纠偏并移除 V2-derived K1 V3 core 依赖：`K1_fused` 下 V3 compute core 不得包含 `V2_` kernel 或 include/copy `dcu_megamoe_v2` 实现。
- [✅] 将 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 的原始 pure K1 normal/LL 5pack C kernel 迁入 `megamoe/dcu_megamoe_opt/K1_fused` 的 stage-owned 代码。
- [🚫] 历史项：K1 V3 raw pure kernel 显式 build gate 与 backend-scoped probe 曾用于 bring-up；normal raw backend 现已遗弃，raw build 仅保留 LL。
- [✅] 复核历史 session `019e6ecc-aaed-74e1-aa6e-78b8ee3133f3` 的 V2 编译超时处理，并将 V3 raw K1 显式编译路径对齐 V2 的 VGPR codegen flag。
- [🚫] 历史项：V3 normal backend 的 aicc 编译策略已退役；normal C/aicc raw 不再编译，normal 生产路径使用 ASM-pack5。
- [✅] 跑通 K1 V3 normal low-level aicc smoke：fake symmetric buffer dispatch-pull metadata 有效，all-ones 输入/权重输出 sample 为 4096。
- [✅] 收敛 K1 V3 normal low-level 实现为单一 fused normal kernel：删除 `pure_ext` 命名、删除 `k1_v3_stage_rows_from_ptrs_kernel`，normal GEMM 直接从 `row_x_ptrs/row_x_scales` load。
- [✅] 删除 K1 V3 normal low-level bring-up 的 `hipMemsetAsync(grid_barrier)`，normal fused kernel 不再依赖额外 grid-barrier 初始化 op。
- [✅] low-level smoke 复用原始 K1 ASM 路径已有 compact prebuild route kernel 与 scratch header/capacity/route-grid 策略，V3 low-level 不再维护独立 route header 布局；该项不代表 production fused 通信已完成。
- [✅] low-level smoke 让 K1 V3 normal GEMM 消费 compact prebuild 产出的 `m_indices/row_x_ptrs/row_x_scales`：weight expert 来自 `m_indices`，padding mask 来自 `row_x_ptrs`，不再假设固定 per-expert row stride；该 rowptr GEMM 路径不得作为 e2e correctness。
- [✅] 撤回 raw rowptr K1 从 public staged/e2e wrapper 的接入：`USE_MEGAMOE_V3=1` 在 K1 真正融合通信语义前 fail-fast，不再跑无效 K1-only e2e。
- [✅] 建立 K1 V3 normal single-kernel fixed-route fused 骨架：blockIdx.x==0 在同一 K1 main kernel 内写 `row_x_ptrs/row_x_scales/m_indices/route_weights/output_index/row_combine_ptrs/stats`，其他 GEMM CTA 等 per-tile flag 后进入 pure 5pack GEMM 主体；aicc 编译通过，low-level ones/expert_ramp smoke 通过。
- [✅] LL 继续以 stage-owned C pack5 K1 groupgemm core 为主体，维护 dispatch-pull 通信输入、route metadata、row combine pointer 和 stats 语义；normal 对应合同由 ASM-pack5 覆盖。
- [✅] LL K1 dispatch-pull 已保留在当前 C pack5 staged kernel 内；normal 不再推进 C/aicc 版本。
- [✅] 当前保留路径未引入独立 K1 通信前处理；LL 的 pure-vs-fused delta 已在 Phase 6/6b 记录并收口，normal 由原 K1 ASM-pack5 合同覆盖。
- [✅] 已参考原始 DeepGEMM ASM 与现有 K1 fused ASM 的 dispatch-pull 语义和 wait/signal 位置；当前 normal 生产实现直接复用 isolated ASM-pack5 边界。
- [✅] 建立 K1 ASM 差异图：`hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` vs 原 K1 fused dispatch-pull ASM，先定位通信插入点、metadata 访问和可见性语义，再映射到 V3 pure C normal/LL 主体。
- [✅] 对齐现有 K1 ASM 输出合同：`l1_out`、`route_weights`、`m_indices`、`output_index`、`row_combine_ptrs`、可选 expert stats。
- [✅] 对齐原 K1 的 shape contract：EP8、experts=256、local_experts=32、topk=6、hidden=4096、alignment=256。
- [✅] K1 pure-vs-fused 劣化检查点已在 Phase 6/6b 建立；LL residual 已收口为性能余量，normal 生产路径切到 ASM-pack5。
- [✅] K1 实际 staged 通信链路 vs 原始 C 5pack groupgemm/pure kernel 的分项耗时已记录；后续不再把 normal C/aicc raw 作为优化目标。
- [✅] K1 fused 相对 pure 的主要额外成本已通过 A/B/PMC 归因；无当前阻塞项。
- [🚫] K1 前 rank barrier remove A/B 已反证：LL no-tail skip barrier correctness 失败，当前保留 rank barrier，不再作为待办。
- [✅] 跑通 K1 normal metadata/unit correctness：以原 K1 ASM compact metadata 作为 oracle，验证 V3 fixed-route 在容量内场景的 `row_combine_ptrs/m_indices/route_weights/stats` 语义。
- [✅] 修复 K1 V3 normal rowptr A-load 数值问题：确认 metadata 对齐但输出不一致时，根因是 divergent row pointer 被错误包装成 raw-buffer resource；已改为普通 global vector load，并通过 ASM/V3 K1 output compare 验证 `max_abs=0`。
- [✅] 将 K1 V3 normal fixed-capacity route 生成从“每 tile 全量重扫 route”收敛为更接近原 K1 fused ASM 的 scanner 结构：前 4 个 row tiles 按 source rank 分片扫描，同一 K1 kernel 内写 row-side metadata 并发布 tile ready flag。
- [✅] 跑通显式 fixed-route staged bring-up 定位验证：`MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 下，V3 K1 normal + 原 K2 + 原 K3 ASM 在 1024 tokens tail-reduce on/off 均完成 debug 链路定位；该结果只能说明容量内 fixed-route launch/metadata/K2/K3 链路没有直接故障，不算 V3 e2e correctness，也不解除默认 fail-fast。
- [✅] 对齐 K1 V3 normal compact-capacity tile list：同一 K1 kernel 内完成 count/build/emit compact rows，4096 random compare 中 `asm_rows == v3_rows == 29696`、active/common=24446、K1 output `max_abs=0`；1024 random compare 仍保持 rows=8192、active/common=6091、`max_abs=0`。
- [✅] 跑通显式 compact-capacity staged debug：`MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 下，V3 K1 normal + 原 K2 + 原 K3 ASM 在 1024/4096 tokens、tail-reduce on/off 均完成 debug 链路定位；该结果只能说明 normal compact-capacity K1 进入后续 staged 链路没有直接故障，不算最终 V3 e2e correctness，也不解除默认 fail-fast。
- [✅] 恢复 normal K1-only correctness gate：public eager `USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal` 不再依赖 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP`，K3 保持原 ASM，1024/4096 tokens tail-reduce on/off 均通过 baseline correctness。
- [✅] K1 LL low-level fused helper 已接入同一 K1 launch，并通过 fake symmetric-buffer raw smoke、K1/K2/K3 单卡逐级诊断和 8 卡 K1-only staged gate；当前 K1-only staged 使用 K3-ASM-compatible 256-row/expert stride，最终 V3 K3 LL 接入后再保留/恢复 64-row 真 LL layout 做性能路径。
- [✅] uneven tokens 与 cuda graph 已通过当前 production gate 覆盖；overflow/zero-weight 专门边界改为后续可选 edge-case guard，不阻塞当前 V3 K1 功能闭环。
- 状态：✅ complete for production gate（normal C/aicc K1 历史路线已遗弃；后续仅保留可选 edge-case/perf backlog）

### Phase 3: K3 V3 功能迁移
- [✅] 建立 K3 V3 stage-owned build/source 边界：V3 LL pack5 extension 默认编译，不再保留 stub/raw build gate；生产 V3 代码不得 include `dcu_megamoe_v2` 实现。
- [✅] 建立 K3 ASM 差异图：`hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` vs 原 K3 combine / tail-reduce fused ASM，先定位 combine/tail-reduce 插入点、signal/fence/wait 语义，再映射到 V3 pure C normal/LL 主体；V2 只作为接口/行为参考。
- [✅] K3 当前生产边界已收敛：LL 以 C pack5 groupgemm core 补齐 combine/tail 语义；normal 以 isolated ASM-pack5 补齐 no-tail/tail 语义，normal C/aicc 不再推进。
- [🚫] 历史项：K3 V3 normal raw aicc 编译边界已退役；normal raw extension 不再编译。
- [✅] 跑通 K3 V3 normal no-tail raw low-level smoke：all-ones 和两 expert/tile-homogeneous pack5 pattern 均经 `row_combine_ptrs` scatter 写 combine buffer，sample max error=0。
- [✅] 跑通 K3 V3 normal no-tail 与原 K3 ASM 的 8 卡同输入分布式对比：4096 requested tokens、actual capacity 4224、K2 传真实 `row_combine_ptrs` skip inactive rows 时，combine 与 reduce 后 `y` 均为 `max_abs=0`。
- [🚫] 历史项：K3 V3 normal no-tail raw combine 曾接入 eager staged no-tail 分支；当前已被 normal ASM-pack5 替代，不再作为 production 或 fallback。
- [✅] K2->K3 可见性诊断：`MEGAMOE_DCU_V3_K2_SYSTEM_FENCE=1` 已验证不能单独修复 official 4096 no-tail；K2 save-temps 显示 fence path 生成 `s_waitcnt vmcnt(0) lgkmcnt(0)` + `buffer_wbinvl1_vol`，排除“K2 fence 未等 store 完成”假设。
- [✅] K3->reduce rowptr store 形式诊断：V3 K3 pack5 combine BF16 rowptr store 已改为显式 `global_store_short`，aicc save-temps 确认 `global_store_short=384`、`flat_store_short=0`，epilogue 有 `s_waitcnt vmcnt(0)` + `buffer_wbinvl1_vol`；official 4096 no-tail 仍失败，说明 store family 不是唯一缺口。
- [✅] 撤回默认 host sync / print 通过路径：`MEGAMOE_DCU_V3_NO_TAIL_SYNC` 默认关闭，显式诊断打开时也不打印；print 让 4096 no-tail 通过只能作为时序证据，不能作为修复。
- [✅] 对齐 K3 no-tail 后的非 final barrier/rowptr 可见性语义：参考 Hygon allreduce release/acquire 和原 K3 ASM diff 后，legacy peer barrier / reduce acquire / K2 fence / store-only rowptr glc 均被反证；V3 K3 active-row mask 与 epilogue store 的 `row_combine_ptrs` load 已改为 `global_load_dwordx2 ... glc`，K1 scanner CTAs 的 metadata publish 使用 `__threadfence_system()` 后 clean 1024/4096 no-tail 三轮稳定通过；跳过 baseline oracle sync 的残差由 clone/原 ASM A/B 归类为测试 oracle 异步窗口，不作为生产同步修复。
- [✅] 当前保留 K3 路径没有新增 runtime 通信 launch：normal 使用原 K3 fused ASM-pack5 epilogue/store/reduce 合同，LL 使用 stage-owned C pack5 combine/tail kernel。
- [✅] LL K3 的 pure-vs-fused delta 已在 Phase 6/6b 记录并收口；normal C/aicc pure 流水保护计划已被 ASM-pack5 生产路径替代。
- [✅] 已参考原始 DeepGEMM ASM 与现有 K3 fused ASM 的 combine/tail-reduce 语义和 wait/signal 位置；normal 生产实现直接使用 isolated ASM-pack5。
- [✅] 对齐 K3 normal no-tail-reduce：写 combine buffer，外部 `rank_barrier + reduce_local_combine` 仍可复用；1024/4096 clean 三轮通过。LL no-tail 后续按 LL backend 接入单独覆盖。
- [✅] 对齐 K3 normal tail-reduce：V3 normal 使用 isolated `K3COMBINE_TAILREDUCE_PACK5` ASM code object；LL 使用 C pack5 tail-reduce kernel。
- [✅] 跑通 K3 V3 normal tail raw single-rank smoke：topk=1 与 topk=6 均验证 combine scatter、done owner slot、self signal wait、local reduce。
- [🚫] 历史项：normal C raw tail stage compare / worker-load / done-signal 定位链条已停止，生产 tail 迁移到 ASM-pack5。
- [🚫] 历史项：normal C raw tail reduce worker 读取语义不再继续修；相关证据保留在 progress/findings，后续不作为计划项推进。
- [✅] 反证 tail direct-signal A/B：把 signal 改成 `atomicAdd_system(done_counter, 1)` 判定出的最后 GEMM CTA thread0 直接发布 peer signal 后，8 卡 `v3_only` 第 1 轮即出现多 rank nonfinite，比旧 owner-slot 路径更差；该补丁已撤回，不作为修复方向。
- [✅] 完成 tail signal slots/generation 诊断：旧 owner-slot 路径下 8 卡 `v3_only` / `asm_first` stage compare 看到 local signal slots reset 前为 0、V3 后 send/recv generation 均为 1，done counter 也到达目标；当前证据不支持“signal slot reset 或 generation 未到达”作为根因。
- [🚫] 历史项：K3 normal C/raw tail combine/row mapping Python 诊断不再继续；生产 tail 已迁移到 ASM-pack5，旧诊断读法只作为历史证据保留。
- [🚫] 历史项：V3 normal C/raw tail reducer/combine 可见性与 active-row 覆盖语义不再作为后续修复路线；当前生产 tail 由 ASM-pack5 覆盖。
- [✅] 按用户要求暂缓 normal tail 深挖，转入 K1/K3 fused LL：已恢复 true LL 64-row layout 并完成 K3 LL no-tail/tail staged correctness。
- [🚫] 历史项：normal C/raw tail-reduce 定位停止恢复；无需再修旧 device post-hoc reduce / stage compare `rank_stats` 链条。
- [✅] 对齐 K3 V3 LL no-tail-reduce：使用 LL pure 5pack 主体和 V3 pack5 L2 layout，写 `row_combine_ptrs` combine buffer，外部 `rank_barrier + reduce_local_combine` 继续复用；固定验证 32/128 tokens。
- [✅] 对齐 K3 V3 LL tail-reduce：在 LL kernel 内完成 combine 写入后的本地 reduce 和跨 rank signal；32/128 tokens 三轮 8 卡 correctness 通过。
- [✅] K3 pure-vs-fused 劣化检查点已建立：LL 记录 pure/local/staged rowptr delta，normal ASM-pack5 记录 pure/direct/remote-store ceiling。
- [✅] K3 实际 staged 通信链路 vs 原始 C 5pack groupgemm/pure kernel 的分项耗时对照已记录；normal C/aicc raw 对照只保留为历史。
- [✅] K3 fused residual 已归因为 remote/scattered combine store 数据通路；低风险 store/wait/locality A/B 已反证或收口，剩余为 optional backlog。
- [✅] 明确跨 rank store 完成语义，保持与现有 tail signal / barrier 语义等价或更保守。
- 状态：✅ complete for production gate（normal C/raw K3 历史路线已遗弃；remote-store/contract 大改仅保留 optional backlog）

### Phase 4: Eager staged fused 集成
- [✅] 在 `opt.py` 或 K1/K3 wrapper 层做最小分流：`MEGAMOE_DCU_USE_OPT_3STAGE=1 && USE_MEGAMOE_V3=1` 才调用 V3；若 3-stage 未 forced，`USE_MEGAMOE_V3` 不影响原路径。
- [✅] normal K1-only staged correctness gate：V3 K1 fused normal + 原 K2 + 原 K3 ASM 用于隔离 K1 correctness 和 layout/metadata 问题，L1 使用 V3 pack5，L2 保持原 ASM layout；1024/4096 tokens tail-reduce on/off 均通过 baseline correctness。
- [✅] LL K1-only staged correctness gate：V3 K1 fused LL + 原 K2 + 原 K3 ASM，L1 使用 V3 pack5，L2 保持原 ASM layout；32/128 tokens、tail-reduce on/off 均通过 baseline correctness。注意：该阶段为 K1-only staged gate，不是完整 V3 e2e；为兼容原 K3 ASM 使用 256-row/expert stride。
- [✅] K1-only staged gate 后已接 K3 V3 staged，no-tail/tail 均已进入当前 V3 production gate。
- [✅] K2、workspace/state/cache 外部逻辑保持复用；只保留 V3 backend/layout 必要分流。
- [✅] 执行流程和 benchmark 流程不插入权重处理 kernel、权重重排 kernel 或额外 launch。
- [✅] 四种 eager 组合已覆盖：LL/normal × tail_reduce on/off。
- 状态：✅ complete for production gate

### Phase 5: Correctness、CLI 与默认路径回归
- [✅] 默认路径回归：`USE_MEGAMOE_V3=0` 的 legacy opt tail 32 通过 baseline correctness 和性能 smoke。
- [✅] 兼容性回归：`megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` 继续作为主入口，V3 通过 env/backend/layout 分流，不新增测试专用热路径。
- [✅] V3 单测只在 setup/fixture 阶段提前转好权重；执行、bench 参数和主流程继续复用。
- [✅] V3 correctness：本轮远端矩阵用 DeepEP/DeepGEMM baseline 作为 oracle，覆盖 normal tail/no-tail、normal uneven、normal graph、LL tail/no-tail；所有 JSON 结果 `correct=True`。
- [✅] tail reduce on/off 均已覆盖：normal 32/4096 tail/no-tail、LL 32 tail/no-tail 均通过；normal/LL graph replay 在 uneven bucket 下无性能劣化。
- 状态：✅ complete for cleanup validation（更大 token 档位和长期性能优化记录继续保留在 Phase 6/6b 历史与 backlog）

### Phase 6: Performance gate
- [✅] 建立 retained production gate 的性能对照：V3 normal/LL 均已完成 8 卡 correctness/perf 验收，normal 覆盖 32、4096、tail/no-tail、uneven graph，LL 覆盖 32、tail/no-tail、uneven graph；4096 normal no-tail/tail 分别为 `5.808258/5.838198 ms`，相对 baseline `9.460317/9.513697 ms`，约 `1.63x`。
- [✅] normal no-tail performance 收口：当前 retained champion 为 K1 dynamic/parallel staged producer + K3 isolated ASM-pack5 no-tail；旧 normal C/aicc K3 scalar/rowptr/store 调参链路已停止作为生产计划推进。
- [✅] normal tail ASM-pack5 迁移收口：V3 normal plain-pack5 tail code object、Python/C++ wrapper、signal generation 协议和 default gate 已接入；cleanup refresh 已补 4096 formal repeat，tail fused `5.838198 ms`，baseline `9.513697 ms`，未复现早期 baseline VMFault/卡住。
- [✅] LL residual refresh 收口：保留 LL C pack5 K1/K3 路线，32/128 历史 formal、cleanup 32 tail/no-tail、LL uneven graph no-tail/tail 均通过；当前低风险微调暂无新的必做候选。
- [🚫] 历史项/遗弃：normal C/aicc no-tail stage-delta 优化、normal C/raw tail-reduce、normal K3 C scalar rowptr/store 主线、fixed-route/raw/stub/availability 相关 bring-up helper 均不再作为 active 计划；相关反证只作为历史依据保留在 findings/progress。
- [🚫] 历史项/遗弃：K3 rowptr4/rowptr8 batch-load、staged-vector-store、rowaddr reuse、resource/load-family 摆动、no-tail internal signal、device-scope fence 等 C/aicc A/B 已反证或被 ASM-pack5 生产路径替代，不再重复。
- 状态：✅ complete for production gate（Phase 6 不再保留性能余量类 optional backlog；后续若 Phase 10 sweep 暴露明确回归，再按新证据重新立项）

### Phase 6b: LL 进一步优化
- [✅] LL retained path PMC/ISA attribution：当前默认 LL K1 block32 无 spill，block64 有 `private_segment=272` 并解释 block64 回退；K3 block32 no-tail 无 spill，128 residual 仍指向 remote rowptr combine 数据通路。
- [✅] LL K1 parallel stage-copy 与 active-row/padded-zero 收口：block32/CUS64 固定保留 parallel stage-copy；32/128 correctness 与 formal 性能均通过，K2/blockM 变体复测后继续保留 default。
- [✅] LL K3 remote combine microtuning 收口：store modifier、deferred wait、CUS、dest-sort、rowaddr wave-shuffle、store-burst、compact mapped-reduce 和 current split 复核均无稳定正收益，当前不再推进本地 epilogue/store-order 小改。
- [✅] LL graph/uneven 覆盖补齐：cleanup refresh 已补 LL no-tail/tail uneven graph，graph replay 均快于 eager，不劣化。
- [🧭] optional backlog：若后续要继续压 LL 128，优先从真实 remote/scattered combine 数据通路、合同级 reduce/chunk-readiness 或新 profiling 证据切入；不基于已反证的 rank-bucket stride mismatch artifact 规划大改。
- 状态：✅ complete for current retained LL（remaining items are optional backlog）

### Phase 7: Uneven tokens 与 cuda graph 对齐
- [✅] 准备 LL uneven correctness 快速入口：`hygon_tmp/debug/run_v3_ll_correctness_matrix.sh` 固定 V3 LL no-tail/tail uneven token 列表，等待 8 卡空闲后执行。
- [✅] 收紧 V3 graph 边界：normal backend graph 仍显式 guard；LL backend 已接入 V3 K1/K3 graph wrapper，避免 V3 pack5 L2 误落到原 K3 ASM graph fallback。
- [✅] 对齐 V3 LL uneven tokens：rank 间 token 数不均时 metadata、tail reduce、local combine 和输出顺序一致；K1 LL 已按 per-source `sections.num_tokens` 扫描 route，no-tail/tail uneven matrix 通过。
- [✅] 对齐 V3 LL cuda graph：runtime token tensor、capture-safe launch arg storage、tail signal/reset 覆盖 uniform 32/128 和 uneven 64 bucket，no-tail/tail graph replay 均通过。
- [✅] 图模式保持 V3 门控和 backend/layout 分流：V3 LL graph 生效，V3 normal graph 继续 fail-fast，不影响原 graph path。
- [✅] V3 LL 功能补齐后性能哨兵：默认 eager 32/128 no-tail `0.8276/1.0278 ms`，tail `0.8305/1.0400 ms`，相对最近 retained 无明显劣化。
- [✅] V3 LL graph-vs-eager 性能哨兵：修复 graph K1 容量误按 `sym_buffer.num_max_tokens_per_rank=384` 放大的问题；after-fix graph replay 32/128 no-tail `0.6860/0.8723 ms`、tail `0.7293/0.9198 ms`，均快于对应 eager `0.8276/1.0278 ms`、`0.8305/1.0400 ms`。
- [✅] V3 normal uneven eager parity：normal ASM-pack5 no-tail/tail 在 token list `1024,768,512,896,640,384,256,128` 下均通过，`max_abs=0.000488281`。
- [✅] V3 normal cuda graph 基础支持：normal graph 不再 fail-fast；K1 normal graph 走 ASM-pack5，K3 no-tail ASM-pack5 透传 active_tiles，tail ASM-pack5 复用 generation/runtime active-tile 合同；uniform 1024 bucket replay 512/1024、uneven 1024 bucket no-tail/tail 曾通过基础 smoke。
- [✅] 必做/BUG：V3 normal ASM graph multi-token replay correctness 已修复。根因是 V3 normal graph 走 K1 ASM-pack5，但 K1 graph `staged_flags/meta_flags` reset layout 只覆盖 legacy backend，导致 captured `flag_generation` 复用时读到上一轮 ready flag。修复后 no-tail `512,1024` replay=5 通过，tail `512,1024` repeat 4/4 通过，512/1024 均 `max_abs=0.000488281`。
- [✅] V3 normal graph-vs-eager 性能哨兵：uniform 1024 bucket replay 512/1024 no-tail `1.616/1.983 ms` vs eager `1.692/2.075 ms`，tail `1.658/1.954 ms` vs eager `1.747/1.994 ms`；uneven no-tail graph `1.663 ms` vs eager `1.787 ms`，tail graph `1.724 ms` vs eager `1.825 ms`，均无劣化。
- [✅] baseline timing VMFault/卡住排查：当前 V3 normal no-tail/tail 1024/4096、graph 前置 tail/no-tail 1024、以及旧 no-large 128 复现矩阵均可完成 `before_baseline_timing -> after_baseline_timing`；旧 VMFault 日志归类为早期 K3 raw/persistent/环境污染或并发端口冲突，不是当前 baseline oracle/timing 的稳定回归。
- [✅] 2026-06-15 cleanup validation refresh：normal tail uneven `17,31,48,64,79,96,112,127` 通过；同一 uneven bucket 的 normal graph replay 32/96/128 token 均通过且快于 eager。
- [✅] 2026-06-15 LL graph uneven 补测：同一 uneven token list 下 LL no-tail/tail graph replay 32/96/128 token 均通过且快于 eager。
- 状态：✅ complete for graph correctness refresh（normal graph multi-token replay bug 已关闭；若后续 full sweep 出现新回归，再按新证据重新立项）

### Phase 8: Symm buffer / route_scratch 显存审计
- [✅] symm buffer 结构已复核：DCU `get_symm_buffer_size_for_mega_moe()` 只分配 peer pointer header、runtime token、输入 `x/x_sf/topk_idx/topk_weights` 和 BF16 combine 区；`l1_acts/l1_acts_sf/l2_acts/l2_acts_sf` 在 DCU API 中是空 view，未继承 CUDA 原版的大中间激活存储。
- [✅] combine 区当前不是冗余：big fused、legacy staged、V3 normal ASM-pack5 和 V3 LL 都通过 K1 生成的 `row_combine_ptrs` 写入同一 slot-major combine layout；no-tail 外部 reduce 和 tail K3 in-kernel reduce 都依赖该合同。
- [✅] route_scratch 结构已复核：全局分配采用 big-fused `dcu_route_scratch_bytes()`，前段为 K1 route task workspace，后段为通用 route tile scratch；staged fused 复用后段 `x_fp8/act_bf16/act_fp8/act_scale` 区域作为 `staged_x/l1_out/act_fp8/act_scale`，没有额外再分一套 Python tensor workspace。
- [✅] 主要冗余来源已定位：当前 `SymmBuffer` 同时服务 big fused、legacy staged、V3 normal、V3 LL，因此 route_scratch 按 big-fused 通用 layout 和 384 token 对齐容量一次性分配。小 token LL 实际 rows 远少于通用 layout，4096 normal 也远少于 route tile worst-case rows。
- [🧭] planned direction：big fused 后续进入退役/删除范围后，`route_scratch` 不再需要兼容 full `dcu_route_tile_scratch_layout`；届时把显存优化从 optional backlog 提升为 staged/V3 路径清理项。
- [🧭] planned direction：legacy staged fused 非 V3 路径后续也进入退役评估后，route_scratch 可从 staged-only 继续收敛为 V3-only staged layout；LL 按小 token bucket rows 分配，normal 按大 token ASM-pack5 capacity rows 分配。
- [🧭] planned direction：新增 V3-only/staged-only route_scratch size/layout 公式，按 retained V3 staged fused 真实合同分配：K1 metadata/task workspace、`staged_x`、`l1_out`、`act_fp8`、`act_scale`、K3 `prob_storage`、graph runtime token、tail done counter/signal addrs、K1 graph flags/meta flags；不再预留 big fused 的 L2 queue、tile pull/done counters、tile rowptr arrays 等 persistent kernel scratch。
- [🧭] planned direction：V3 LL 小 token 用 backend/token-bucket rows 分配 scratch，避免 32/128 档按 384-token big-fused route layout 分配；V3 normal ASM-pack5 用 K1 actual capacity rows 分配 staged scratch，而不是按 route tile worst-case rows。
- [🧭] planned verification：route_scratch 缩容实现必须覆盖 V3 normal/LL、tail/no-tail、eager/graph、uniform/uneven、32/128/1024/4096；同时确认 big fused 入口、legacy staged fused 非 V3 入口、graph flag reset/fallback 的删除或 fail-fast 语义清晰。
- 状态：✅ audit complete（big fused DSV4 主路径退役后，route_scratch 缩容已进入 Phase 12 生产实现）

### Phase 9: K1 source-rank address contract / >4GB span 优化
- [✅] 当前主动项：借 8192 放大 `symm_x_span > UINT32_MAX` 问题，拆清 K1 normal ASM 的 fast uint32-offset、absolute pointer、rank-local bit2 三类路径耗时，优先找通用优化，不做 8192 特化。
- [✅] 诊断：4096 warmup=1 可把 `v3_symm_span` 压到 `3781689344 <= UINT32_MAX`，K1 median 约 `2.70-2.77 ms`；warmup=0 时 `v3_symm_span=5683806208`，K1 median 约 `3.33-3.38 ms`，说明 4GB span 对 K1 有明确影响。
- [✅] 诊断：8192 warmup=0/1 都仍 `symm_x_span > UINT32_MAX`，K1 median 都约 `5.36-5.41 ms`，warmup 无法解决 8192；K3 warmup A/B 对 remote combine 基本无正收益，4GB span 主要伤 K1 地址路径而非 K3 peer-store 本体。
- [✅] 历史诊断：8192 旧 `K1_PREBUILD_MODE=auto/asm/compact` A/B 显示当时 auto 与 asm-route 等价，K1 约 `5.37 ms`；旧 compact 因为 `symm_x_span > UINT32_MAX` 时走 absolute pointer，虽把 rows 从 `57344` 降到 `54272`，但 K1 变慢到约 `6.50 ms`。该结论只适用于 absolute compact 旧实现。
- [✅] 诊断：已隔离 asm-route bit2 `{rank-local x offset, source rank}` 与 asm-route absolute pointer；同为 8192/rows `57344` 时 bit2 约 `5.33 ms`，absolute 约 `6.40 ms`，因此不能把 `>4GB` 路径简单切到 absolute。
- [✅] 诊断：K3 remote combine locality / peer-write A/B 已覆盖 2048/4096/8192。简单 dest-sort 在 8192 从约 `5.98 ms` 退化到约 `8.07 ms`，rank-bucket compact 在 2048/4096 仅噪声级小幅变化、8192 两轮一正一负；当前没有可直接生产化的 K3 rowptr locality 改动。
- [✅] 生产收口：K1 compact/graph prebuild 在 `symm_x_span > UINT32_MAX` 时不再写 absolute x pointer，也不再设置 `reserved_c0` bit1；row_x_ptrs 改写为 bit2 `{rank-local x offset, source rank}`，graph/compact 与 asm-route 使用同一 MUBUF source-load 合同。
- [✅] 验证：远端 source pytest 6/6、K1 object 强制重编通过；`K1_PREBUILD_MODE=compact` 8192 no-tail e2e correctness 通过，fused `10.667 ms`；K1-only compact 8192 `v3_symm_span_gt_u32=true`、rows `54272`、median `5.156 ms`，相较旧 compact absolute `~6.50 ms` 明显恢复；normal graph no-tail/tail replay 512/1024 均 correct 且无性能劣化。
- [✅] 生产 auto 收口：降低 auto compact 门槛后，8192 `K1_PREBUILD_MODE=auto` 已选择 compact/bit2，K1-only rows/active `54272/54272`、median `5.158 ms`，与强制 compact `5.187 ms` 等价；full e2e no-tail auto correctness 通过，fused `10.792 ms`，强制 compact `10.859 ms`。
- [✅] 12ms+ 复现与归因：强制旧 `K1_PREBUILD_MODE=asm` full e2e no-tail 可复现 `12.712 ms`，说明当前 auto 不复现 12ms+ 是因为默认路径改变；K3 拆分显示 rows `57344 -> 54272` 后 `staged_rowptr` 从 `5.943 ms` 降到 `4.671 ms`，full e2e 约 `1.9 ms` 改善主要来自 K3/rowptr 下游压力下降，而不是 K1 kernel 本体单独贡献。
- [✅] 生产清理：K1 ASM `reserved_c0` bit1 absolute pointer 死分支已从 route emit 和 stage source-load 中移除；bit2 `{rank-local offset, source_rank}` 与默认 uint32-offset 路径保留，source guard 防止 absolute label/旧 host bitfield 回流。
- [✅] auto 规则细化：auto 现在同时要求 per-expert fractional saving `>=5%` 且 local tile saving `>=8 tiles`，并额外把 `fixed_capacity_tiles_per_expert >= 7` 的大 rowptr 档切到 compact；后者覆盖 8192/8448 这类 K1 不明显变快但 K3 rowptr/remote combine 明显受益的场景。
- [✅] 8448 复核：初版 auto 因 compact capacity rows 与 fixed rows 同为 `57344` 而保持 asm，K1-only `5.390 ms`、e2e `12.864 ms`；强制 compact 虽未降低 capacity rows，但 prebuild 后 K3 `staged_rowptr` 从约 `6.22 ms` 降到 `4.78 ms`，e2e `11.248 ms`；加入大 rowptr 规则后 auto 选择 compact，K1 rows/active `57344/57344`、e2e correct 且 fused `11.202 ms`。
- [✅] 代表 size e2e 矩阵：normal eager no-tail 覆盖 `256,512,1024,1025,2048,2050,4096,4097,8192,8448` 的 `asm/compact/auto`，30/30 correctness 通过；auto 对齐当前强制最优或在噪声容限内。2048 首轮 compact 轻微领先但三轮复核 asm 更优，因此规则保持 `2048/2050 -> asm`。
- [✅] tail-reduce 哨兵已补：后续 latest matrix 与 normal graph/eager 回归已覆盖 4096/8192 tail/no-tail、eager/graph、uniform/uneven；未见 correctness 回退，性能处于历史预期区间。
- [🚫] 当前不继续推进：ASM hot loop 内减少 `source_rank -> peer pointer table -> s_load_dwordx2`、SRD 缓存、source-rank grouped row emission 都属于高风险合同级改动；在 compact/bit2 + auto 规则后，8192/8448 e2e 已恢复到目标区间，暂无足够收益证据支撑继续改生产路径。
- [🧭] optional backlog：短期继续保留 `_v3_normal_symm_warmup_enabled` 作为低风险生产保护；它只提高 first production `symm_x_span <= UINT32_MAX` 的概率，不作为长期地址合同。
- [🧭] optional backlog：只有未来 profiler 再证明 K1 `source_rank`/peer pointer lookup 是 4096/8192/8448 的主瓶颈时，才重新评估 per-rank pointer table、显式 `{source_rank, rank_local_offset}` row metadata、source-rank grouped row emission 等合同级方案；不得作为小 ASM patch 直接推进。
- [🧭] optional backlog：更大改造方向是 DeepEP/Flux 风格的 rank/channel staging：先按 source rank/channel 把 remote x 拉到 local `staged_x`，GEMM 主体只消费 local staged input；该方案会触及 K1 route emission、staged_x layout、graph/uneven、tail/no-tail 合同，必须等 V3-only scratch/route_scratch 收敛后再评估。
- [🧭] optional verification：任何去 4GB span 依赖的方案都必须满足 correctness 覆盖 V3 normal no-tail/tail、eager/graph、uniform/uneven，并且 4096/8192 K1 与 e2e 不劣化到当前 warmup fast path 之外；4096 参考 guard 以当前 cleanup refresh `~5.81/5.84 ms` e2e 和历史 K1 `~2.94-2.95 ms` fast path 为准。
- 状态：✅ production optimization complete（absolute pointer 已移除，compact/bit2 与 auto 规则已通过代表矩阵和 tail-reduce 哨兵验证；剩余仅长期合同级 backlog）

### Phase 10: Required V3 performance sweep
- [✅] 必做：运行前检查远端 8 卡状态，确保无残留 KFD 进程；同步当前 workspace 到 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`，并重跑 `compileall`、`megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` 和 `build_ext --inplace` 确认产物新鲜。
- [✅] 必做/BUG：V3 LL row-capacity bug 已修复。根因是 K1 LL host 侧 `rows_per_expert` 只按平均路由数对齐，随机 routing 中部分 local expert 超过容量后被截断，导致 fused stats mismatch 或较大 numerical diff；1024 capture 失败后，exact 256 eager 也暴露同类 overflow，因此当前策略改为 `expected rows/expert >= 48` 时补 64 rows，保留 32/128 tiny bucket 原容量，同时保证 LL 实际执行至少支持到 512 tokens/rank。graph capture bucket 可以更大，但不能替代 exact eager bucket 的容量安全验证。
- [✅] 必做：V3 LL 使用 graph 模式完整刷 uniform tokens per rank `8,32,64,128,256,512,1024`；graph capture bucket 固定按 `1024`，已记录 replay 性能表。旧 `phase10_ll_graph_20260615_232857` 慢表仅代表 K1 row-capacity correctness 修复后的中间状态，已被 runtime-row K3 graph 修复后的新结果覆盖。
- [✅] 必做：V3 LL graph sweep 同时覆盖 `K3_USE_ASM_TAIL_REDUCE=0` no-tail 与 `K3_USE_ASM_TAIL_REDUCE=1` tail；K1 row-capacity correctness 和 K3 graph fixed-row 性能问题均已修复，全档位均通过，graph replay 性能已落盘。
- [✅] 必做/BUG：V3 LL tail reduce/signal graph replay 已改为 runtime-token 化。K3 tail reducer 现在从 graph runtime token tensor 读取 replay tokens 并 clamp 到 capture bucket，capture1024/replay8..512 不再按 1024 token 做 reduce；capture8192 下 K2 graph 固定-row block 开销已通过现有 K2 reg kernel 的 grid-stride row loop + 内部 `max_row_blocks=2048` 收敛，不新增 kernel、不引入 D2H sync，LL replay 8 token 已恢复到 `~0.56 ms`。
- [✅] 2026-06-17 cleanup follow-up：LL tail eager 不再误传 graph runtime token output；eager 128/512 with `num_max_tokens_per_rank=1024` tail correctness 通过，graph capture1024 replay 32/96/128/256/512 tail correctness 通过。确认 eager 有效 rows 不被 max 放大，max 只作为 symm combine stride / scratch capacity 合同。
- [✅] 2026-06-20 latency follow-up：LL tail-reduce=1 的 K1 前 standalone `rank_barrier` launch 已内联进 K1 C pack5 start path；同步语义仍保留在 K1 block0 + existing local grid barrier 中。LL tail eager 8/32/128/256、LL tail graph capture256 replay 8/32/128/256、LL no-tail eager 32 smoke 均通过。normal/no-tail/post-K3 external reduce 不在本轮移除范围内。
- [✅] 2026-06-20 LL overlap follow-up：K1/K2 已完成 metadata 级融合，K1 在 `actual_m` 后写 `max(actual_m)`，K2 直接消费并统一 clamp actual rows；K3 tail reducer 增加 runtime-token active reducer worker 收缩，capture8192 replay `<=256` 使用 64 worker、512+ 保持 128 worker。LL tail/no-tail eager/graph representative smoke 均通过，未新增 kernel/D2H/H2D。
- [✅] 必做：V3 normal 使用 eager 模式刷 uniform tokens per rank `256,512,1024,1025,2048,2050,4096,4097,8192`；不把 graph 结果代替本轮 normal eager sweep。
- [✅] 必做：V3 normal eager sweep 同时覆盖 no-tail/tail；所有档位 correctness 通过，1024/1025、2048/2050、4096/4097 边界点未见 correctness 异常或明显性能台阶。
- [✅] 必做：normal eager 结果落盘到 `hygon_tmp/debug/phase10_v3_sweep_20260615_196S/`；最终 LL graph 补测结果落盘到 `hygon_tmp/debug/ll_graph_dynamic_verify_20260615_235731/`，包含日志、case status、JSON 和 `summary.csv`；已回写 `progress.md` / `findings.md` / 本 Phase 状态。
- [🧭] optional backlog：K2 计算级融入 K1 只在后续 profiler 证明 K2 + `l1_out` 中间读写占 LL graph replay 的显著比例时再恢复；当前保留 metadata 级融合。直接把 SwiGLU/row-wise scale/FP8 quant 塞进 K1 epilogue 有较高 occupancy 和跨 N tile reduction 风险，暂不作为生产优化。
- 状态：✅ complete for required sweep（normal eager sweep、LL 1024 row-capacity bug fix、LL graph `8..1024` no-tail/tail runtime-row replay sweep 均已完成；LL 1024 仍只作为 graph capture/correctness 覆盖，不作为大 token 生产推荐路径）

### Phase 11: V3 转正为 DCU MegaMoE 主路径
- [✅] 梳理当前 public DCU MegaMoE 入口、`opt.py`、`__init__.py`、tests 和 README 中的 big fused / legacy staged / V3 gate 分支，确认哪些是生产入口、哪些只剩历史引用。
- [✅] 将默认 DCU MegaMoE 主路径切到 V3 staged dispatcher：实际 tokens per rank `<=256` 选择 LL，`>256` 选择 normal；保留外部调用参数和单测使用方式，权重 layout 差异仍由 V3 pack5 helper/fixture 处理。
- [✅] 处理 graph 与 uneven backend 选择：graph 不用 capture max tokens 误判 backend；uneven 使用 dispatch/global bucket，确保 8 rank backend 一致。
- [✅] 移除 `USE_MEGAMOE_V3` 和 `MEGAMOE_DCU_V3_BACKEND` 对生产路径的控制；同步删除 source guards、测试 env 组合和 README 实验开关说明。
- [✅] 清理之前 big fused 主路径相关代码/默认入口/测试引用，以 V3 LL 覆盖小 token 主路径；DSV4 public eager/graph 已默认进入 V3 staged，public API 改为显式 `megamoe_backend="ll"|"normal"` 加 `graph=True|False`，不再保留旧 `big_fused_cuda_graph` / `stages_fused_cuda_graph` / `ll_cuda_graph` / `normal_cuda_graph` 语义。
- [✅] 清理之前 legacy staged fused 主路径相关代码/默认入口/测试引用，以 V3 normal 覆盖大 token 主路径；Python 旧 K1/K3 非-pack5 staged wrapper 已删除，V3 normal 只通过 isolated ASM-pack5 entry 表达。
- [✅] 保留并验证仍有意义的非 V3 参数/env：tail/no-tail、K1 auto/compact 策略、`MEGAMOE_DCU_OPT_VERBOSE_BUILD` 仍保留；`USE_MEGAMOE_V3` / `MEGAMOE_DCU_V3_BACKEND` 等实验 env 已不再被生产路径消费。
- [🧭] retained compatibility：底层 `_C.fp8_mega_moe*` 暂保留为非 DSV4 shape fallback；DSV4 route_scratch size 已按 V3-only staged layout 缩容，不能再把旧 persistent big-fused 当作 DSV4 私有直调用路径。
- 状态：✅ complete for public V3 main path（底层非 DSV4 `_C` fallback 作为兼容保留，不再列为 V3 主路径阻塞项）

### Phase 12: V3-only route_scratch / symm footprint 优化
- [✅] 在 DSV4 V3 staged 主路径上重新定义 retained route_scratch 合同：K1 metadata workspace、`staged_x`、normal ASM K1 graph flags/meta/GpuProb、`l1_out`、`act_fp8`、`act_scale`、K3 `prob_storage`、graph runtime token、tail done/signal addrs。
- [✅] DSV4 size API 不再按 legacy big-fused `DcuRouteTileScratchLayout` full layout 分配，不再预留 L2 queue、tile pull/done counters、tile rowptr arrays、persistent kernel scratch 等空间；非 DSV4 shape 暂留 legacy fallback。
- [✅] 按 V3 backend/token bucket 缩容：小 buffer 按 LL rows/headroom；大 buffer 按 normal fixed-capacity 上界覆盖 auto compact/graph compact；实测 8192 requested max 对齐 8448 后 route_scratch 为 `0.830 GiB`，旧约 `4.013 GiB`。
- [✅] 校验 symm buffer 的 combine 区、runtime token、peer pointer header 是否仍是 retained V3 path 必需；combine 区仍被 K3 no-tail 外部 reduce 与 tail in-kernel reduce 共同消费，runtime token/peer header 仍是 graph/跨 rank 合同的一部分，当前不做高风险 shrink。
- [✅] 覆盖完整 eager/graph、uniform/uneven、tail/no-tail 的 route_scratch 缩容验证；Phase 13 矩阵 40/40 case pass，LL/normal eager、graph replay、uneven eager/graph 均通过。
- 状态：✅ complete（DSV4 route_scratch shrink 已落地，symm buffer retained 区域暂无可安全删除项）

### Phase 13: README、测试与最终回归矩阵
- [✅] 更新工程 README 的 DCU MegaMoE 部分：V3 成为主路径、权重 pack5 layout 要求、框架/测试层 `<=256` LL / `>256` normal 策略、保留参数/env、移除 V3 gate/backend env；graph 示例使用 `megamoe_backend` + `graph=True`，不再沿用旧 graph flag 名称。
- [✅] 更新 `megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` 和 `megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py`：尽量复用原参数和使用方法，删除 V3 env-gated 专用断言，新增默认主路径、auto backend、retired symbol 和 route_scratch shrink 合同测试。
- [✅] 本地静态验证：`compileall`、`git diff --check` 通过；本地无 pytest 环境，source pytest 在远端容器执行。
- [✅] 远端构建验证：同步到 `/workspace/DeepGEMM`，`compileall`、`PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` 7/7 通过，`MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过。
- [✅] 远端功能/精度验证：LL/normal、tail/no-tail、eager/graph、uniform/uneven 均通过；结果目录 `hygon_tmp/debug/phase13_final_20260616_181534/`。
- [✅] 远端性能 sweep：LL 刷 `8,32,64,128,256`；normal main path 刷 `512,1024,1025,2048,2050,4096,4097,8192`，另补 normal actual256/dispatch512 boundary；对照历史 progress 表，未见功能回退，8192 当前稳定在 `~10.9-11.0 ms`，相对最近 `~10.75-10.79 ms` 有约 1-2% 漂移但仍优于旧 `12ms+` 档。
- [✅] README 命令验证：`scripts/run_dcu_megamoe_opt.sh` 以 `TOKENS_LIST=512 SKIP_BENCH=1` smoke 通过，验证 README/script 入口可执行。
- [✅] 2026-06-17 follow-up：上层 API 再收敛为 `megamoe_backend` + `graph`；auto/threshold/env 只存在于 `megamoe/dcu_megamoe_opt/tests/test_mega_moe_dcu.py` 作为框架选择模拟；旧 `dispatch_num_tokens` public/CLI 参数删除，eager-only 容量上界改名为 `capacity_num_tokens`，不参与分化、不进入 graph。远端 build、source tests、LL/normal eager/graph smoke 已通过。
- [✅] 2026-06-17 latest matrix：K2 actual_m replay pruning 与 LL 8192 headroom 修复后，重刷 LL `8,32,33,64,128,129,256,257,512,513` eager/graph、normal `256,512,1024,1025,2048,2050,3072,4096,4097,8192` eager/graph，graph 均 capture8192；LL/normal eager/graph uneven smoke 均 `correct=True`，README smoke v2 通过。结果落盘 `hygon_tmp/debug/full_matrix_latest_20260617_191221`，表格已写入 `progress.md`。
- 状态：✅ complete（最终回归矩阵通过，8192 轻微性能漂移已记录为后续观察项）

### Phase 14: V3 pack5 weight layout policy
- [✅] 已验证：single/unified ABI（LL transposed layout + normal ASM `logical ni -> physical ni` 映射）功能完整，但 normal 4096/8192 相比历史双-layout 最优有约 `8%` 回退；该路径不再作为默认性能路径。
- [✅] 默认策略改为 dual layout：LL 继续使用 transposed pack5，normal ASM 使用 plain pack5；public API 可接收 `{ "ll": ..., "normal": ... }` 权重 dict，由 `megamoe_backend` 选择对应 layout。框架 PD 分离或显存允许时用该默认路径追极致性能。
- [✅] 兼容策略：新增 `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1`，打开后使用 `{ "unified": ... }` 或旧 tuple 权重，并加载 `_UNIFIED_PACK5` code object，保留单 ABI/单权重占用路径；该开关不是默认性能路径。
- [✅] 构建面：新增 K1/K3 no-tail/tail `_UNIFIED_PACK5.s` 和对应 setup code-object build；默认 PACK5 ASM 恢复 plain `ni`，unified PACK5 ASM 保留 transposed physical `ni` 映射。
- [✅] 代表验证：远端 `build_ext --inplace`、source pytest `9/9` 通过；默认 dual layout eager LL 32/512 为 `0.641/2.297 ms`，normal 512/4096/5120/8192 为 `1.753/5.798/7.403/10.886 ms`；unified normal4096 为 `6.274 ms`。
- [✅] graph 代表验证：默认 dual layout graph capture8192 下 LL replay 32/512 为 `0.661/2.258 ms`，normal replay 4096/8192 为 `5.687/10.485 ms`；unified normal4096 replay 为 `6.180 ms`。
- [✅] 第一轮重新挖掘结论：DCU KB 与本工程 CUDA MegaMoE 参考均支持“layout 是 global-load/LDS/MMAC fragment 合同”的判断；当前 LL transposed 与 normal plain 分别匹配各自 kernel 热路径，是默认双 layout 下的最佳已知安全基线。暂未找到可低风险替换的第三种共同 layout 或单侧新 layout。
- [✅] source-backed 参考已读：`ds_read_m32x16_b16_{normalxalt,altxalt,swizzle}.cpp` 可作为后续 normal 新 code object 的 LDS/matrix-read 合同参考，但它要求同步改 LDS write/read/MMAC/store，不能作为默认 ASM 的几行局部 remap。
- [🚫] 已证伪方向仍不重复：plain layout + LL direct mapped load、plain layout + LL B-side register shuffle、plain layout + 不改 LL 合同、transposed layout + normal local-read/LDS fragment remap、normal store-side remap。
- [ ] 后续必做：在 dual layout 默认策略下继续重新挖掘 LL/normal 各自最佳 layout/kernel 组合；允许隔离两套 kernel/code object，不强求复用。下一轮只进入有 profiler/ISA/source-backed 证据的分支，例如 LL B-load/MMAC 合同保持不变的 layout 变体、normal 连续 global load 与 LDS/MMAC 合同同时闭合的变体。任一新候选必须同时覆盖 LL `32/512` 与 normal `512/4096/8192/5120`，并不得低于当前默认 dual-layout 代表性能。
- [🧭] 可选长期方向：如果双 layout 仍有显存压力或 PD 分离之外也需要单权重，可重新设计第三种共同 layout；进入条件是有 profiler/ISA/source-backed 证据，不再靠单点 permutation 猜测。
- 状态：[ ] active（默认 dual layout 已恢复历史性能档位；unified 作为兼容开关保留。下一步围绕双 layout 各自最佳性能继续挖掘，而不是强行单 ABI。）

## 讨论用接口对照草案

| 项目 | 原 staged fused ASM | V2/V3 C pack5 参考 | 迁移关注点 |
| --- | --- | --- | --- |
| K1 输入 | sym_buffer, route_scratch, L1 weight/scale, code object | sym_buffer, route_scratch views, pack5 L1 weight/scale | 是否复用现有 route_scratch layout，避免外部重排 |
| K1 输出 | l1_out, route_weights, m_indices, output_index, row_combine_ptrs | V2 已能输出同类 metadata | shape/capacity/compact-prebuild 行为要对齐 |
| K3 no-tail | act_fp8, act_scale, m_indices, L2 weight/scale, row_combine_ptrs | V2 normal copy-stage/LL rowptr combine | 写 combine buffer 后继续复用外部 reduce |
| K3 tail | K3COMBINE_TAILREDUCE code object + signal addrs | V2 K3 内部 tail reduce | 需要对齐跨 rank 完成/本地 reduce 时序 |
| 通信语义 | K1 dispatch-pull + K3 combine/tail reduce | pure normal/LL 本身不带通信 | 在通信外壳中保留 5pack C groupgemm core 性能 |
| pipeline 融合 | ASM 把通信与 GEMM 调度绑定 | C pure 更容易改 pipeline | K1 藏到 load/route，K3 藏到 epilogue/store/reduce |
| GEMM 流水保护 | 原 ASM 自带融合流水 | pure 5pack C 已有高性能流水 | 不破坏 load/compute/store 核心调度 |
| 后端选择 | 原 ASM 单一路径 | LL C pack5 / normal ASM-pack5 | public API 执行显式 `megamoe_backend`；测试/框架层按 selector tokens 与 threshold 自动选择 |
| 隔离 | 历史 env-gated 实验路径 | DSV4 shape 默认 V3 staged；非 DSV4 暂留 `_C` fallback | 不再消费 `USE_MEGAMOE_V3` / `MEGAMOE_DCU_V3_BACKEND` |
| layout | 原始 DCU MegaMoE layout | V3 pack5 layout | 单测/离线侧提前转好，runtime/bench 不做权重处理 kernel |

## 风险
- V2 C pack5 real-flow 包装路径性能已知不适合作为 V3 core；若继续复用会偏离保护原始 pure kernel 流水的目标，必须移除 V2-derived K1 core 依赖。
- K3 tail reduce 的跨 rank 同步语义比普通 combine 更敏感，不能只看本地 correctness。
- K1 前 rank barrier 若直接消掉，可能破坏 sym buffer 输入 copy 可见性；LL tail path 2026-06-20 的安全做法是把 start barrier/reset 内联进 K1 C pack5，而不是删除同步语义。normal ASM 和 no-tail post-K3 外部 reduce 仍不能直接消。
- graph 支持需要额外处理 capture-safe launch argument、runtime token、active_tiles 和 signal reset，不建议在第一步和 eager 混做。
- V3 layout 与原始 DCU MegaMoE layout 不同，最容易在测试准备和 wrapper 参数复用处产生隐性错配；不能通过 runtime 权重处理 kernel 掩盖 layout 问题。
- pure normal/LL 的性能优势来自 5pack C groupgemm 本体；扩成 fused 后新增的 dispatch-pull、row pointer、combine/tail-reduce 逻辑如果耦合过重，可能稀释 pure 优势，需要按 K1/K3 分项量化劣化来源。
- 中途 pure-vs-fused 劣化过大时必须提前优化，不能只把问题留到 Phase 6；不过所有提前优化都服务于 Phase 6 的最终 tokens 分段性能门槛。
- 如果通信逻辑没有藏进 pipeline，而是退化为独立前后处理，既容易引入额外 launch/内存 pass，也会直接破坏 pure 5pack C 的性能优势。
- 如果为了通信融合重写或打乱原 GEMM pipeline，风险比通信逻辑本身更高；这类改动必须有明确性能证据，否则应回到保护原流水的方案。
- 如果从 V2 C fused/real-flow 实现机械抽取 K3/K1 主 kernel，容易继承 V2 copy-stage、tail-reduce 和调度结构，偏离“original groupgemm ASM diff -> pure C 5pack 主体”的干净路径；这类 V2-derived core 必须撤出生产路径。
- 当已有 DCU MegaMoE 周边实现可复用时，V3 不应为了规避接口适配而重写整段周边流程；优先复用原 K1 compact prebuild、K3 row-combine/tail-signal 合同和 staged wrapper，只重写无法承载 pack5/pure C 主体的边界。

## 实验运行策略
- 首选远端 8 卡验证和 benchmark，运行前检查 `hy-smi`/进程/显存状态。
- 如果有人占用部分显存但测试仍能在 8 卡或可用卡上推进，优先继续跑 correctness/compile/smoke，不因非致命占用停下。
- 如果 8 卡全部不可用，保持监控而不是中断工作；期间继续做本地代码梳理、实现、静态检查和计划更新，卡空后恢复远端验证。
- 每个验证命令都记录当前生产相关参数/环境：`megamoe_backend`、`graph`、selector tokens、`K3_USE_ASM_TAIL_REDUCE`、`K1_PREBUILD_MODE`、tokens/rank、tail/no-tail；旧 `USE_MEGAMOE_V3` / `MEGAMOE_DCU_V3_BACKEND` 不再作为生产验证项。
- 远端 build/test/profiler/debug 的临时日志、status 和产物统一放到 repo 内 `hygon_tmp/`，默认使用 `hygon_tmp/debug/`；不要把项目调试产物散落在 `/tmp`。
- 当前 tail-reduce 验证优先级：normal tail 走 isolated ASM-pack5 production path，LL tail 走 retained C pack5 path；不再恢复 normal C/raw tail-reduce 定位链条。`megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py` 只用于 env gate、wrapper/source boundary、V2 隔离和源码 guard，不作为 GPU correctness。历史 `global_load_dwordx4 off`、stage compare、direct-signal、signal-slot/generation 等 C/raw tail A/B 已反证，后续不重复同一方向；新的 tail 变更必须直接用 8 卡 eager/graph correctness 与性能 gate 验收。
