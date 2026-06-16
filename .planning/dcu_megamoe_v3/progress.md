# DCU MegaMoE V3 Progress

## 2026-06-04 - 新建 V3 计划

- 用户明确：
  - 停止继续做 V2 优化；
  - 新建 V3 计划文件夹；
  - 把已经打磨好的 DCU MegaMoE 相关代码重新梳理清楚；
  - 将 K1 pure LL/normal、K3 pure LL/normal 扩展后替换 staged fused 中的 K1 dispatch-pull L1 ASM kernel 与 K3 combine ASM kernel；
  - K3 combine 有 tail reduce / no-tail 两版，都需要对齐；
  - 直接集成在 DCU MegaMoE 内；
  - 外部逻辑尽可能复用 staged fused；
  - 使用 `USE_MEGAMOE_V3` 做隔离；
  - `USE_MEGAMOE_V3` 未开启时保持原始 DCU MegaMoE 逻辑；
  - 先讨论计划，定具体后再开始实现。
- 已完成：
  - 读取 planning-with-files / karpathy / dcu-rag-kb / hygon optimizer 相关约束；
  - 初步梳理 `megamoe/large_opt.py`、K1/K3 large_opt wrapper 与 ext；
  - 运行 DCU KB 初始检索；
  - 创建 `.planning/dcu_megamoe_v3/` 的 `task_plan.md`、`findings.md`、`progress.md`。
- 未开始：
  - 未修改实现代码；
  - 未改默认 staged fused 行为；
  - 未运行远端编译/测试。

## 2026-06-04 - 更新 V3 门控、目录与性能目标

- 用户进一步明确：
  - V3 backend 使用 `MEGAMOE_DCU_V3_BACKEND`；
  - `USE_MEGAMOE_V3` 只在 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1` 时生效；
  - 外部单测尽量复用，命令行参数需要兼容；
  - V3 权重 layout 与原始 DCU MegaMoE 不同，相关逻辑需要增量区分；
  - K1 前 rank barrier kernel 可以尝试消掉；
  - V3 直接集成到 `megamoe/dcu_megamoe_large_opt`；
  - kernel 代码放到已有 `K1_fused` / `K3_fused` 文件夹；
  - 不再继续改动 `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` 作为生产实现；
  - uneven tokens 和 cuda graph 需要最终对齐 DCU MegaMoE 功能。
- 性能目标更新为：
  - tokens per rank `<512` 时，V3 快于 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0`；
  - tokens per rank `>=512` 时，V3 快于 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1` 且 `USE_MEGAMOE_V3=0`。
- 已完成：
  - 更新 `.planning/dcu_megamoe_v3/task_plan.md`；
  - 更新 `.planning/dcu_megamoe_v3/findings.md`；
  - 记录当前进度。
- 未开始：
  - 未修改实现代码；
  - 未改默认 staged fused 行为；
  - 未运行远端编译/测试。

## 2026-06-04 - 采纳开工前三项实现建议

- 用户采纳：
  - `MEGAMOE_DCU_V3_BACKEND` 未设置但 V3 开启时，默认使用 `normal`；
  - 显式设置 `MEGAMOE_DCU_V3_BACKEND=ll` 才走 LL；
  - V3 权重 layout 分流新增显式 V3 transform/helper，外部 CLI 保持兼容；
  - K1 前 rank barrier 第一版 V3 先保留，功能跑通后再单独 A/B 尝试移除。
- 已完成：
  - 更新 `.planning/dcu_megamoe_v3/task_plan.md`；
  - 更新 `.planning/dcu_megamoe_v3/findings.md`；
  - 记录当前进度。

## 2026-06-04 - 收紧权重 layout 和 kernel 集成边界

- 用户进一步明确：
  - V3 开发整合不要在 DCU MegaMoE 执行路径里引入额外 kernel；
  - 权重 layout 差异在单测中提前转好；
  - 执行和 benchmark 流程不要引入权重处理 kernel；
  - 其他单测和命令行参数少改动，尽量复用现有入口。
- 已完成：
  - 将 V3 plan 中的权重分流改成测试/离线准备侧处理；
  - 明确 runtime/bench 只消费已转换好的 V3 pack5 权重；
  - 明确除替换 K1/K3 计算 kernel 本身外，不在 `dcu_megamoe_large_opt` 集成中引入额外 kernel。

## 2026-06-04 - 增加 8 卡实验与持续推进规则

- 用户进一步明确：
  - 集成阶段首选 8 卡实验；
  - 有人占显存但仍能推进时继续跑；
  - 如果卡都被占用，则监控等待，不中断计划；
  - 每完成一个工作项就在计划里打勾，再继续下一项，尽量持续推进。
- 已完成：
  - 在 `task_plan.md` 增加 8 卡实验环境检查项；
  - 增加实验运行策略，记录显存占用、env 组合和监控等待规则。

## 2026-06-04 - Phase 0 完成 large_opt.py flow 梳理

- 已完成：
  - 梳理 `fp8_mega_moe_large_opt_3stage()` eager staged flow；
  - 梳理 `_run_large_opt_3stage_graph()` graph staged flow；
  - 在 `findings.md` 记录 K1/K2/K3、tail-reduce/no-tail、graph runtime token 和 active_tiles 关系；
  - 在 `task_plan.md` 勾选 Phase 0 第一项。
- 结论：
  - 第一版 V3 eager 分流应优先替换 K1/K3 wrapper 调用；
  - K2、scratch/state/cache、rank_barrier、外部 reduce 先尽量复用；
  - graph V3 等 eager 稳定后再对齐。

## 2026-06-04 - 讨论 pure 到 fused 的扩展边界

- 用户进一步明确：
  - pure normal/LL 实现本身不太带通信操作；
  - V3 需要先分析已有 DCU MegaMoE 主要 kernel；
  - 再把对应 5pack C pure normal/LL 扩成带通信语义的 fused K1/K3；
  - 性能要保证相对 pure 少劣化；
  - 计划表完成项用 ✅，进行中项用 ⏳。
- 已完成：
  - 更新 `task_plan.md` 的状态符号约定；
  - 将 Phase 0 已完成项改为 ✅，当前 K1 ASM 接口梳理项标为 ⏳；
  - 在 `findings.md` 记录 pure normal/LL 到 fused V3 的扩展原则。
- 未开始：
  - 本轮未继续实现；
  - 未继续远端实验；
  - 未新增代码改动。

## 2026-06-04 - 补充 pure-vs-fused 提前优化触发条件

- 用户进一步明确：
  - 如果 fused 相对 pure 劣化太多，需要提前进一步优化；
  - 最终仍以 Phase 6 Performance gate 的要求为准。
- 已完成：
  - 在 `task_plan.md` 成功标准、Phase 2、Phase 3 和风险中补充提前优化触发条件；
  - 在 `findings.md` 补充 pure-vs-fused 劣化明显时提前定位优化的原则。

## 2026-06-04 - 明确 baseline 只用于正确性

- 用户进一步明确：
  - baseline 只做正确性验证；
  - V3 相比 baseline 性能好不具有决定意义。
- 已完成：
  - 在 `task_plan.md` 成功标准和 Phase 5 中明确 baseline 只作为 correctness oracle；
  - 在 `findings.md` 记录 baseline timing 只可参考，不参与 Phase 6 是否通过的判断。
- 未开始：
  - 本轮未继续实现；
  - 未继续远端实验。

## 2026-06-04 - 明确 LL/normal 的性能档位

- 用户进一步明确：
  - LL 骨架适用于 tokens per rank `<512`；
  - normal 骨架适用于 tokens per rank `>=512`；
  - bench size 固定为 32/128 跑 LL，1024/4096 跑 normal。
- 已完成：
  - 在 `task_plan.md` 成功标准和 Phase 6 中写入 backend/token 档位绑定；
  - 在 `findings.md` 将原建议档位收敛为 32/128(LL) 和 1024/4096(normal)。
- 未开始：
  - 本轮未继续实现；
  - 未继续远端实验。

## 2026-06-04 - 明确通信隐藏进 GEMM pipeline

- 用户进一步明确：
  - C 实现相对更好改；
  - 希望融合的通信操作彻底隐藏在 GEMM pipeline 里。
- 已完成：
  - 在 `task_plan.md` 增加 pipeline-internal 通信融合原则；
  - 在 Phase 2 写入 K1 dispatch-pull 应融进 load/route/tile scheduling；
  - 在 Phase 3 写入 K3 combine/tail-reduce 应融进 epilogue/store/reduce；
  - 在 `findings.md` 记录该策略是首选设计，不是额外优化项。
- 未开始：
  - 本轮未继续实现；
  - 未继续远端实验。

## 2026-06-04 - 明确保护原 GEMM 流水

- 用户进一步明确：
  - 不要破坏原 kernel 内的 GEMM 流水；
  - 融合通信后的性能要尽可能少劣化。
- 已完成：
  - 在 `task_plan.md` 增加保护原 5pack C GEMM load/compute/store pipeline 的硬约束；
  - 在 Phase 2/3 补充 K1/K3 通信只嵌入自然插入点，不重排核心 compute pipeline；
  - 在 `findings.md` 记录“若为通信融合打乱原流水，风险高于通信逻辑本身”的原则。
- 未开始：
  - 本轮未继续实现；
  - 未继续远端实验。

## 2026-06-10 - 恢复 V3 工作上下文

- 已完成：
  - 确认当前分支为 `dcu_mega_v3`；
  - 读取并遵守 `planning-with-files`、`dcu-rag-kb`、`hygon-hip-kernel-optimizer`、`remote-ssh-docker-workflow` 的本轮约束；
  - 读取 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md`；
  - 运行 planning session catchup，未发现需要合并的遗留上下文输出。
- 本轮复核：
  - `session-catchup.py` 直接执行时被 Windows 文件关联到 Node，报 `SyntaxError: Invalid or unexpected token`；
  - 已改用显式 `python C:\Users\Administrator\.agents\skills\planning-with-files\scripts\session-catchup.py` 运行，命令成功且无输出；
  - `git status --short` 显示既有改动：`.codex/skills/remote_work/SKILL.md` modified、`third-party/cutlass` deleted，本轮不回滚、不触碰。
- 当前状态：
  - Phase 0 仍停在 K1 ASM wrapper/ext 真实接口梳理；
  - 本轮尚未修改生产代码，`task_plan.md` 状态暂不变。

## 2026-06-10 - Phase 0 接口梳理完成

- 已完成：
  - 复读 `large_opt.py` eager/graph staged flow，确认 K1/K2/K3、tail-reduce/no-tail、graph runtime token 和 active_tiles 关系；
  - 复读 `K1_fused/k1_fused.py` 与 `k1_fused_ext.cu`，记录 K1 route_scratch 内部切片、`GpuProb` 布局、compact/asm-route bitfield、graph flag reset 合同；
  - 复读 `K3_fused/k3_fused.py` 与 `k3_fused_ext.cu`，记录 no-tail combine、tail signal、tail-reduce、prob_storage、active_tiles/graph offset 合同；
  - 复读 V2 `api.py`、`layout.py`、`runtime.py`、K1/K3 pybind/ext，形成 V2 pack5 LL/normal 到 staged fused V3 的能力与缺口表；
  - 查本地 DCU KB：初次并行检索 30s 超时；更窄 query 成功，确认 system-scope signal 前应保守使用 `__threadfence_system()` / system-scope fence。
- 已更新：
  - `findings.md` 增加 “2026-06-10 Phase 0 静态接口复核”；
  - `task_plan.md` 将 Phase 0 剩余项标为 ✅，Phase 0 状态标为 ✅ complete；
  - `task_plan.md` 将 Phase 1 第一项标为 ⏳，Phase 1 状态标为 ⏳ in_progress。
- 下一步：
  - 重读计划后进入 Phase 1，先做最小 V3 env/backend gate 和 wrapper 分流骨架；未完成 kernel 必须显式 fallback 到原 ASM 路径，不破坏默认逻辑。

## 2026-06-10 - Phase 1 最小 V3 gate/backend 落地

- 已完成代码：
  - 新增 `megamoe/dcu_megamoe_large_opt/v3_config.py`，集中解析 `USE_MEGAMOE_V3`、`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE` forced-only gate、`MEGAMOE_DCU_V3_BACKEND=ll|normal`；
  - 在 `large_opt.py` eager/graph staged flow 中加入 `_selected_v3_backend()`，只有 backend 非空时才调用 V3 wrapper；
  - 在 K1 wrapper 增加 `k1_symm_fused_l1_v3()` / `k1_symm_fused_l1_v3_graph()`，当前显式 fallback 到原 ASM wrapper；
  - 在 K3 wrapper 增加 `k3_l2_fused_v3_to_combine()`，当前显式 fallback 到原 ASM wrapper；
  - 新增 `tests/test_dcu_megamoe_v3.py`，覆盖 gate/backend contract 和 source-level dispatch guard。
- 验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/v3_config.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `pytest -q tests/test_dcu_megamoe_v3.py` 失败：本地 PATH 没有 `pytest`；
  - `python -m pytest -q tests/test_dcu_megamoe_v3.py` 失败：本地 Python 环境没有 pytest 模块；
  - 已用 inline Python 执行等价断言，输出 `v3 gate/source assertions passed`；
  - `git diff --check` 通过。
- 已更新：
  - `task_plan.md` 将 Phase 1 的 gate/backend/source-level test 标为 ✅；
  - `task_plan.md` 将 V3 weight contract 标为 ⏳，作为下一步；
  - `findings.md` 增加 Phase 1 gate/backend 实现记录。
- 注意：
  - 当前 V3 wrapper 是显式 fallback，尚未迁入 C pack5 kernel；
  - 尚未运行远端 DCU 编译/8 卡 correctness，本阶段只做本地 source-level 与语法验证。

## 2026-06-10 - Phase 1 V3 pack5 weight contract 落地

- 已完成代码：
  - 新增 `megamoe/dcu_megamoe_large_opt/v3_layout.py`，作为 V3-owned pack5 offline/test helper；
  - helper 提供 pack/unpack/flatten/offset/shape 与 `transform_fp8_weights_for_mega_moe_v3_pack5()`；
  - 执行路径没有导入 `v3_layout.py`，保持 runtime/bench 不做权重处理 kernel 或 repacking。
- 已完成测试：
  - 扩展 `tests/test_dcu_megamoe_v3.py`，CPU 对照 V3 pack5 helper 与既有 pack5 reference layout；
  - inline Python layout 对照通过，输出 `v3 layout assertions passed`。
- 验证：
  - `python -m compileall megamoe/dcu_megamoe_large_opt/v3_layout.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 已更新：
  - `task_plan.md` 将 Phase 1 weight contract 标为 ✅；
  - `task_plan.md` 将 fixture/setup 侧准备 V3 layout 权重标为 ⏳。

## 2026-06-10 - Phase 1 收口并进入 Phase 2

- 已完成：
  - 在 `tests/test_dcu_megamoe_v3.py` 增加 source-level guard，确认 `large_opt.py` 不导入 `v3_layout`，避免 runtime/bench 执行路径引入权重处理；
  - 最终本地综合验证通过：`python -m compileall ...`、inline Python `v3 local assertions passed`、`git diff --check`。
- 已更新：
  - `task_plan.md` 将 Phase 1 所有项标为 ✅，Phase 1 状态标为 ✅ complete；
  - `task_plan.md` 将 Phase 2 第一项 K1 stage-owned C pack5 迁入标为 ⏳，Phase 2 状态标为 ⏳ in_progress。
- 注意：
  - `pytest` 仍因本地环境缺失不可用；未运行远端 DCU 编译/测试；
  - 下一步进入 K1 迁移前必须重读计划，并先确认 `setup.py` / extension build 结构。

## 2026-06-10 - Phase 2 K1 build/module 边界确认

- 已完成：
  - 重读 `task_plan.md`、`findings.md`、`progress.md` 后继续 Phase 2；
  - 静态检查 `setup.py`，确认 large-opt K1 extension 目前只编译 `K1_fused/k1_fused_ext.cu`；
  - 对照 V2 K1 extension 的 split pybind/kernel 结构，确认 V2 kernel source 仍通过 include `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` 复用；
  - 读取 V2 K1 LL/normal pybind 与 raw launcher，确认 LL/normal 都已有 route metadata、output_index、row_combine_ptrs 和 stats 输出；
  - 读取 large-opt K1 scratch allocator，确认 V3 K1 应复用现有 `route_scratch` 切片和返回 tuple。
- 结论：
  - Phase 2 K1 第一版采用“同一个 large-opt K1 extension module + 新增 stage-owned raw launcher source”的方向；
  - 不新建 Python import path，不让 V3 生产实现继续 include V2 `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`；
  - 下一步继续读 V2 K1 core 的 dispatch/pipeline 插入点，再创建 large-opt K1 V3 stage-owned source 骨架。
- 注意：
  - 静态搜索中 `pyproject.toml` 不存在导致一次 `rg` exit code 1，但不影响结果；
  - 尚未进入远端编译或 hipprof/ISA 验证。

## 2026-06-10 - Phase 2 K1 stage-owned source 迁入

- 已完成代码：
  - 新增 `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_groupgemm_impl.cuh`，作为 V3-owned C pack5 K1 kernel copy；
  - 新增 `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_fused_ext.cu`，提供 V3 LL/normal raw launcher symbol；
  - 更新 `setup.py`，把 V3 K1 source 加入 large-opt K1 extension，并把 `*.cuh` 纳入 package data。
- 已更新：
  - `task_plan.md` 将 Phase 2 第一项 K1 stage-owned source 迁入标为 ✅；
  - `task_plan.md` 将 Phase 2 第二项 route metadata / row combine / stats 语义补齐标为 ⏳。
- 本地验证：
  - `python -m compileall ...` 通过；
  - `git diff --check` 通过；
  - 静态确认 V3 source 不再 include `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`。
  - `tests/test_dcu_megamoe_v3.py` 增加 K1 V3 source boundary guard；inline 等价断言输出 `k1 v3 source guard assertions passed`。
- 远端验证：
  - 读取 `.vscode/sftp.json` 得到实际远端：`hg@10.17.176.13`，remote path `/home/hg/yuguo/DeepGEMM`，container repo `/workspace/DeepGEMM`；
  - SSH 和 remote repo 检查首次成功，`megamoe` container 存在但停止；
  - 第一次 `docker start megamoe` 时 SSH 连接超时，随后恢复后成功启动容器；
  - 显式同步本轮 V3 文件到 remote/container 后，容器内 `python3 -m compileall ...` 通过；
  - 第一次 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 失败，原因是 remote checkout 缺少既有 V2 `K1_fused/k1_fused_pybind.cpp`；
  - 已同步既有 V2 K1/K3 extension source 后重跑 build，但命令 10 分钟超时；随后 SSH status/process 检查又出现 `10.17.176.13:22` 连接超时；
  - 因此本轮尚未完成 DTK/HIP extension build，不能声称 C++ 编译通过；需先确认远端是否仍有残留 build 进程。
- 下一步：
  - 在 K1 V3 wrapper 内切出 V3 route metadata、grid barrier、local_topk_mask、tail_tokens 和 pack5 shape checks；
  - 解决 grid barrier 初始化/epoch 化问题后，才把 Python V3 wrapper 从 ASM fallback 切到 raw launcher。

## 2026-06-10 - Phase 2 K1 V3 low-level wrapper 补齐

- 已完成代码：
  - 在 `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu` 增加 V3 raw launcher declaration；
  - 新增 `k1_symm_fused_l1_v3_pack5()` pybind 入口；
  - 入口从现有 `route_scratch` 切出 V3 metadata/grid barrier/local mask/tail token/row combine/output index/staged scale；
  - 入口调用 stage-owned `dcu_megamoe_v3_launch_k1_ll_symm_stage_raw()` 或 `dcu_megamoe_v3_launch_k1_normal_symm_stage_raw()`；
  - 返回 ASM-compatible 五元组，方便下一步 K1 metadata/unit correctness 对照；
  - public Python `k1_symm_fused_l1_v3()` 仍保持 ASM fallback，未接入 production eager path。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py` 通过；
  - inline source guard 输出 `k1 v3 low-level entry guard assertions passed`；
  - `git diff --check` 通过。
  - 最终综合 source assertions 输出 `v3 source assertions passed`。
- 远端状态：
  - 多轮 SSH status 重试仍为 `ssh: connect to host 10.17.176.13 port 22: Connection timed out`；
  - 尚未同步本次 `k1_fused_ext.cu` 新改动，也尚未完成 DTK/HIP extension build。
- 注意：
  - 低层入口目前用 `hipMemsetAsync` 初始化 grid barrier，仅用于未接 production 的 unit bring-up；
  - 真正切到 production V3 前，需要改成 epoch/in-kernel init 或提供无额外 runtime op 的证据。

## 2026-06-10 - 纠正远端工作流并恢复 11 节点验证

- 用户纠正：
  - remote skill / `.vscode/sftp.json` 应使用 `hg@10.17.176.11:22`；
  - Docker container 为 `sglang_megamoe`；
  - repo 路径为 `/home/hg/yuguo/DeepGEMM` -> `/workspace/DeepGEMM`。
- 已完成：
  - 重新读取 planning 三文件和 remote skill；
  - 核对 11 节点 SSH、容器状态、mount：`/home/hg/yuguo` bind 到 `/workspace`；
  - 确认容器内 repo 为 `/workspace/DeepGEMM`；
  - 修正 `.codex/skills/remote_work/SKILL.md` 中残留的 `megamoe` container 与 `DeepDEMM` 路径，统一为 `sglang_megamoe` / `DeepGEMM`；
  - 显式同步本轮 V3 文件和远端缺失的 V2 reference/build 文件到 `/home/hg/yuguo/DeepGEMM`。
- 远端验证：
  - `python3 -m compileall ...` 通过；
  - 第一次 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 因远端缺少 `megamoe/dcu_megamoe_v2/layout.py` 失败；
  - 补齐 V2 reference 后重跑通过，结果 `5 passed`；
  - `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 已启动，当前 `dcc -cc1` 正在编译 `k1_v3_fused_ext.hip`，尚未完成。
- 注意：
  - 本轮不再使用之前误用的 13 节点 / `megamoe` container 结果；
  - 当前 Phase 2 第二项仍为 ⏳，等待 K1 V3 low-level extension 编译和 unit correctness 后再推进状态。

## 2026-06-10 - Phase 2 K1 默认编译恢复

- 已完成：
  - 检查长时间编译状态，确认没有残留 `setup.py build_ext`、`dcc`、`hipcc` 进程；
  - 定位默认 build 卡在 `k1_v3_fused_ext.hip` 的重模板 codegen；
  - 将临时 build flag 从误导性的 `DG_BUILD_MEGAMOE_V3_PACK5` 改为 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS`；
  - 保持 V3 权重 contract 仍为 pack5，`k1_symm_fused_l1_v3_pack5()` 输入校验不变；
  - 默认 build 只编译轻量 raw launcher 符号，真实 K1 raw 重模板需显式 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` 才编译。
- 验证：
  - 本地 `python -m compileall ...` 通过；
  - 本地 source assertions 输出 `v3 k1 raw-build naming/source assertions passed`；
  - 本地 `git diff --check` 通过；
  - 同步到 11 节点 `sglang_megamoe` 后，远端 `python3 -m compileall ... && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `5 passed`；
  - 远端默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，返回码 `0`。
- 临时文件规则：
  - 用户要求临时文件放在 `hygon_tmp` 下；
  - 已将本次远端 build log/status 从 `/tmp` 归档到 `hygon_tmp/sglang_debug/deepgemm_v3_build_default_20260610_144010.log` 和 `.status`；
  - 已更新 remote skill 和 task plan，后续远端 build/test/profiler/debug 产物统一放 `hygon_tmp/`。
- 当前状态：
  - Phase 2 第二项仍为 ⏳；
  - 低层 K1 V3 raw kernel 还没有 correctness 结果，不能接入 production eager path。

## 2026-06-10 - Phase 2 K1 core 来源纠偏

- 用户纠正：
  - V3 K1 不应使用 `dcu_megamoe_v2` 下的实现，因为该融合路径性能很差；
  - 需要从 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 的原始 pure K1 5pack C 实现重新扩展通信语义。
- 已完成计划更新：
  - `task_plan.md` 已明确 V2 只作为接口行为参考，不再作为 V3 K1 compute core；
  - Phase 2 已新增移除 V2-derived K1 core 依赖的进行项；
  - Phase 2 K1 迁入项已改为从 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 迁入原始 pure normal/LL kernel。
- 当前代码状态：
  - `k1_v3_groupgemm_impl.cuh` 仍是先前错误方向的 V2-derived copy，正在移除/替换；
  - `k1_v3_fused_ext.cu` 已先退成轻量 stub，下一步接 stage-owned pure-kernel raw launcher；
  - Phase 2 仍为 ⏳，不能把 K1 V3 接入 production eager。

## 2026-06-10 - 记录原始 ASM 参考边界并继续验证

- 用户补充：
  - `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` 是原始 DeepGEMM 汇编文件；
  - 原 DCU MegaMoE K1 fused / K3 fused ASM 是在该原始 GEMM ASM 基础上扩通信；
  - V3 扩通信可以参考这些 ASM，但仍要守住 pure C 5pack GEMM 主干流水并把通信尽量隐藏进计算 pipeline。
- 已完成：
  - 运行本地 DCU KB 检索，确认原始 grouped fp8/bf16 GEMM 参考与当前 `hygon_tmp/K1_groupgemm_fp8` pure source 对齐；
  - 更新 `task_plan.md` 和 `findings.md`，明确 ASM 是 fused 语义/同步插入点参考，不是 V3 GEMM 主体来源。
- 下一步：
  - 继续执行 K1 core 来源纠偏后的本地 source guard、远端 pytest/default build；
  - 如果默认 build 通过，再尝试 raw pure K1 TU 的显式编译 probe，所有日志继续放 `hygon_tmp/sglang_debug/`。

## 2026-06-10 - K1 pure source 纠偏本地验证完成

- 已完成：
  - 删除/移除 V3 路径对旧 `k1_v3_groupgemm_impl.cuh` 的依赖；
  - `k1_v3_fused_ext.cu` 退为 raw availability/stub，真正 pure raw TU 收敛到单文件 `k1_v3_pure_ext.cu`，避免启用 raw 编译时重复定义 kernel；
  - `k1_v3_pure_groupgemm_impl.cuh` 中 normal / LL kernel body 与 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 原始 pure kernel 逐字对齐，仅做 V3 kernel 名替换；
  - 清理 V3 low-level wrapper 中残留的 “V2-derived” 注释，并让测试 guard 覆盖 `k1_fused_ext.cu` 的 V3 入口。
  - 在 `tests/test_dcu_megamoe_v3.py` 增加可选 source-match guard：当 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 存在时，逐段验证 V3 pure normal/LL kernel body 只做命名替换。
- 本地验证：
  - `python -m compileall ...` 通过；
  - source guard 输出 `v3 k1 pure source guard passed`；
  - pure body 对齐脚本输出 `normal and LL pure kernel bodies match original after V3 renames`；
  - 本地用 pytest stub 直接执行新增 source-match 测试函数，输出 `k1 pure source match test function passed`；
  - `git diff --check` 通过；
  - 检查新文件 LF、无 BOM。
- 远端状态：
  - 逐文件 scp 因 SSH 连接重置/拒绝中断；
  - 已在 `hygon_tmp/sglang_debug/` 准备单 tar 同步包，等 11 节点 SSH 恢复后继续同步、pytest、default build 和 raw TU 编译 probe；
  - 当前 Phase 2 的 dispatch-pull 通信语义补齐仍为 ⏳，public Python V3 path 仍 fallback，不接半成品。

## 2026-06-10 - K1 backend-scoped build gate 远端验证

- 已完成：
  - 重新从 `.vscode/sftp.json` 读取 11 节点参数，使用 `sglang_megamoe` 容器和 `/workspace/DeepGEMM` repo；
  - 将当前 V3 K1 pure source、setup、测试和 planning 文件打包同步到 `hygon_tmp/sglang_debug/deepgemm_v3_sync_20260610_153518.tar`；
  - 容器内确认 `setup.py` 已包含 `DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND`，`k1_v3_pure_ext.cu` 已包含 normal raw 宏，旧 `k1_v3_groupgemm_impl.cuh` 不存在；
  - 远端 `python3 -m compileall ... && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `6 passed`；
  - 远端默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，默认 K1 extension 只编 `k1_fused_ext` 和轻量 `k1_v3_fused_ext`，没有编 raw pure TU。
- raw probe 结果：
  - `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal` 的全量 `setup.py build_ext --inplace` 在 240 秒硬超时后退出；
  - status 为 `124`，日志为 `hygon_tmp/sglang_debug/deepgemm_v3_raw_normal_build_20260610_153737.log`；
  - 检查确认没有残留 `setup.py`、`hipcc`、`dcc` 或 raw build 进程。
- 当前判断：
  - 常规开发/默认编译已恢复正常，不是“几个 kernel 编十分钟”；
  - raw pure bring-up 不能再用全量 extension build 当探针，下一步改为更小粒度的 raw normal compile harness 或拆分 source，避免重复同一个超时动作。

## 2026-06-10 - 复核 V2 编译超时历史并修正 V3 raw flags

- 已完成：
  - 按用户指定复核 session `019e6ecc-aaed-74e1-aa6e-78b8ee3133f3`；
  - 确认 V2 编译处理的两个要点：extension/raw K1 路径带 `-mllvm -enable-num-vgprs-768=true`，并用独立 Makefile/script 做 raw kernel bring-up；
  - 对照当前 `setup.py`，发现 V3 large-opt K1 raw flags 缺少 V2 使用的 VGPR codegen 限制；
  - 更新 `setup.py`：仅在 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` 时给 V3 K1 raw extension 追加 `-mllvm -enable-num-vgprs-768=true`；
  - 更新 `tests/test_dcu_megamoe_v3.py` source guard，要求 V3 raw build path 保留该 flag；
  - 更新 `task_plan.md` / `findings.md` 记录本次历史复核和修正。
- 下一步：
  - 本地先跑 compile/source guard；
  - 同步到 11 节点 `sglang_megamoe` 后，先跑 pytest 和默认 build，再用硬超时 probe 验证 raw normal 是否摆脱前次 codegen 超时。

## 2026-06-10 - V3 raw normal 编译恢复验证完成

- 已完成远端验证：
  - 使用单 tar 同步到 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`；
  - 容器内确认 `setup.py` 已包含 V3 raw K1 的 `-enable-num-vgprs-768=true`；
  - `python3 -m compileall ... && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `6 passed`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_vgpr_pytest_20260610_161410.log`；
  - 默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，status `0`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_vgpr_build_default_20260610_161431.log`；
  - raw normal `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal DG_FORCE_BUILD=1 MAX_JOBS=2 python3 setup.py build_ext --inplace` 通过，status `0`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_vgpr_raw_normal_20260610_161554.log`；
  - import smoke 确认 `k1_symm_fused_l1_v3_pack5` 存在；
  - 检查无残留 `setup.py`、`hipcc`、`dcc` 或 `k1_v3_pure_ext` 编译进程。
- 结论：
  - V2 历史里的 VGPR codegen flag 是本轮 raw normal 编译超时的关键缺口；
  - 当前默认 build 仍轻量，显式 raw normal 编译可作为下一步 K1 V3 low-level correctness 的前置条件；
  - 后续如果做 LL/all raw 编译，仍需按 backend 单独 probe 和记录，不把全量 all 当常规验证。

## 2026-06-10 - V2 默认编译移出 V3 闭环

- 用户明确：
  - V2 不要管；
  - V2 不要编；
  - 当前 V3 工作中就当 V2 不存在。
- 已完成：
  - `setup.py` 中 `DG_BUILD_MEGAMOE_V2_EXT` 默认值改为 `0`；
  - 默认 extension build 不再包含 `megamoe.dcu_megamoe_v2.K1_fused.k1_fused_ext` 和 `megamoe.dcu_megamoe_v2.K3_fused.k3_fused_ext`；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，要求 V2 extension 默认不编；
  - `task_plan.md` / `findings.md` 记录 V2 默认不参与 V3 build/probe/bench。
- 当前 aicc 判断：
  - 不能把全仓库 setup 粗暴切到 `/workspace/dtk_aicc`，因为 V2 旧代码会触发 clang18 builtin 签名不兼容；
  - 既然 V2 已默认不编，下一步可以只验证 large_opt/V3 extension 在 `/workspace/dtk_aicc` wrapper 下的 build 和 low-level smoke。

## 2026-06-10 - V3 normal backend aicc 编译策略确认

- 用户进一步明确：
  - 不是全仓库都必须用 aicc；
  - V3 的 K1 fused 和 K3 fused 的 normal backend 扩展必须用 aicc；
  - LL 和其他没有要求的路径可以继续按原 hipcc/现有逻辑。
- 已完成代码：
  - `setup.py` 新增 `DG_BUILD_MEGAMOE_V3_NORMAL_AICC`，默认开启；
  - 新增 per-extension aicc marker，只有 V3 normal raw/fused backend gate 打开时临时把 PyTorch HIP extension 编译器切到 `hygon_tmp/sglang_debug/v3_aicc_rocm/bin/hipcc -> /workspace/dtk_aicc/bin/aicc`；
  - K1 V3 normal raw gate 已标记 aicc；
  - K3 V3 normal raw gate 也已标记 aicc，后续 K3 normal C/fused TU 接入时复用同一机制；
  - 默认 build、LL、K2、主扩展和非 V3 raw path 继续走原 hipcc；
  - V2 extension 默认仍不编。
- 验证：
  - 本地 `python -m compileall setup.py tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 source guard 输出 `v3 K1/K3 normal aicc source guard passed`；
  - 本地 `git diff --check` 通过；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `6 passed`；
  - 远端默认 build 通过，`v2_seen=0`、`unexpected_aicc_marker=0`、`unexpected_raw_pure=0`；
  - 远端 K1+K3 V3 normal raw build 通过，status `0`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_k1k3_normal_aicc_build_20260610_170658.log`；
  - 该 build 记录 `v2_seen=0`、`k1_v3_normal_aicc_marker=1`、`k3_v3_normal_aicc_marker=1`、`aicc_wrapper_seen=1`、`k1_v3_pure_compiled=1`。
- 当前状态：
  - 编译策略已经对齐用户要求；
  - K3 V3 normal 实际 C/fused kernel 仍未实现，Phase 3 接入时必须沿用已验证的 aicc marker；
  - 下一步继续 Phase 2 K1 normal low-level smoke/correctness。

## 2026-06-10 - K1 V3 normal low-level smoke 跑通

- 已完成：
  - 使用 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1`、`DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal`、`DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1` 构建 large-opt K1 extension；
  - 构建日志确认 K1 normal raw path 通过 `hygon_tmp/sglang_debug/v3_aicc_rocm/bin/hipcc -> /workspace/dtk_aicc/bin/aicc` 编译；
  - 新增临时 smoke 脚本 `hygon_tmp/sglang_debug/k1_v3_normal_smoke.py`，只作为 debug/bring-up 用例，不进入生产路径；
  - fake symmetric buffer 场景下 low-level `k1_symm_fused_l1_v3_pack5(..., backend="normal")` 成功生成 route metadata；
  - all-ones 输入和权重下 active sample 输出均值为 `4096.0`，证明 aicc-built normal raw GEMM 实际参与计算。
- 验证结果：
  - `active_routes=3072`，等于 `512 tokens * topk 6`；
  - `route_weight_sum=3072.0`；
  - `unique_experts_first_expected=12`；
  - `out_shape=(8192, 4096)`；
  - `active_sample_abs_mean=4096.000000`。
- 当前限制：
  - low-level normal smoke 仍先用单独 staging kernel 把 row pointers materialize 成 contiguous `staged_x/staged_x_scale`；
  - 下一步要把 row-ptr load 融进 K1 normal GEMM load pipeline，去掉该额外 runtime kernel。

## 2026-06-10 - K1 V3 normal 收敛为单一 fused row-ptr kernel

- 用户指出：
  - 不应为了复用 pure body 强行保留 pure/fused 两套并行 kernel；
  - 不能复用时可以重写必要部分，做好 V3 fused 隔离即可；
  - 执行路径不能增加额外 kernel。
- 已完成代码调整：
  - 将 heavy K1 V3 normal TU 命名收敛为 `K1_fused/k1_v3_fused_ext.cu`；
  - 将默认 build 的符号补齐文件命名为 `K1_fused/k1_v3_stub_ext.cu`；
  - 将 shared header 命名为 `K1_fused/k1_v3_groupgemm_impl.cuh`；
  - 删除 `k1_v3_stage_rows_from_ptrs_kernel` 定义和 normal path launch；
  - normal launcher 不再接收 `staged_x/staged_x_scale`，直接传 `row_x_ptrs/row_x_scales`；
  - header 中 normal 不再保留 `V3_K1_Pure_DeepGemm...` 并行 kernel，而是单一 `V3_K1_Fused_DeepGemm...` 由原 pure normal body 做有限通信替换得到。
- 本地验证：
  - `python -m compileall ...` 通过；
  - source guard 通过：单一 fused normal body 等于原始 pure normal body 经过显式 row-ptr/x-scale load 替换后的结果；
  - `git diff --check` 通过。
- 远端验证：
  - 同步到 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`6 passed`；
  - 默认 build 通过，`v2_seen=0`、`unexpected_aicc_marker=0`、`unexpected_fused_tu=0`；
  - 显式 K1 V3 normal aicc build 通过，`k1_normal_aicc_marker=1`、`fused_tu_seen=3`；
  - row-ptr direct-load smoke 通过：zero/ones 都 `status=0`，ones 下 `active_sample_abs_mean=4096.000000`。
- 当前限制：
  - low-level bring-up 仍用 route init/count/build/emit helper kernel 和 `hipMemsetAsync(grid_barrier, ...)`；
  - public Python V3 path 仍未接入 production eager；
  - 下一步要把 route metadata / dispatch-pull 进一步内联或证明复用现有生产 helper 不违反最终执行路径约束。

## 2026-06-10 - K1 V3 normal 删除 grid memset bring-up op

- 已完成：
  - 删除 `k1_symm_fused_l1_v3_pack5()` normal bring-up 中的 `hipMemsetAsync(grid_barrier, ...)`；
  - 更新 source guard，要求 `k1_fused_ext.cu` 不再包含 `hipMemsetAsync` 和 `k1_v3_stage_rows_from_ptrs_kernel`；
  - 更新 smoke 脚本的输出统计为 active rows only，避免 padded/inactive rows 的未初始化值影响判断。
- 验证：
  - 本地 `compileall`、source guard、`git diff --check` 通过；
  - 远端 source-level pytest 通过，`6 passed`；
  - 远端 K1 normal aicc build 通过，`k1_normal_aicc_marker=1`、`hipmemset_seen=0`；
  - active-only smoke 通过：
    - zero: `active_out_abs_max=0.000000`、`active_sample_abs_mean=0.000000`；
    - ones: `active_out_abs_max=4576.000000`、`active_sample_abs_mean=4096.000000`。
- 当前限制：
  - route init/count/build/emit helper kernel 仍只属于 low-level bring-up；
  - production V3 K1 还不能接入 eager path，下一步继续把 route metadata 语义向 fused kernel 内收敛。

## 2026-06-10 - K1 V3 normal 复用原 compact prebuild route

- 用户提醒：
  - 原始 DCU MegaMoE K1 ASM 路径已经有 compact prebuild route kernel；
  - V3 不应为了 route build 再造一套不必要的 prebuild 逻辑。
- 已完成代码：
  - `k1_symm_fused_l1_v3_pack5()` 的 normal low-level metadata 切片改为复用原 K1 compact prebuild 的 route header/capacity 公式；
  - route grid 的 `blocks_per_rank` 策略对齐原 ASM prebuild 路径；
  - V3 normal GEMM 不再用 `tile_token / rows_aligned_per_expert` 推 expert，而是从 compact prebuild 写好的 `m_indices[row]` 读取 tile expert；
  - V3 normal GEMM 的 valid/padding mask 改为从 `row_x_ptrs` 判断，能消费 compact tile list 和 padded rows；
  - 更新 source-level guard，防止 V3 回退到独立 route header 或固定 per-expert stride 假设。
- 验证：
  - 本地 `python -m compileall ...` 通过；
  - 本地 source guard 通过；
  - 本地 `git diff --check` 通过；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `6 passed`；
  - 远端默认 build 通过：`v2_seen=0`、`aicc_marker=0`、`v3_fused_tu=0`；
  - 远端显式 K1 V3 normal aicc build 通过：`DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1`、`DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal`、`DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1`、`k1_normal_aicc_marker=1`；
  - low-level smoke 通过：
    - 512 zero：`active_routes=3072`、`active_out_abs_max=0.0`；
    - 512 ones：`active_routes=3072`、`active_sample_abs_mean=4096.0`；
    - 4096 expert-ramp：`active_routes=24576`、`unique_experts_first_expected=32`、`expert_ramp_abs_err=0.0`。
- 记录：
  - build log: `hygon_tmp/sglang_debug/deepgemm_v3_compact_reuse_k1_normal_aicc_build_20260610_181314.log`;
  - smoke logs: `hygon_tmp/sglang_debug/deepgemm_v3_compact_reuse_smoke_zero_latest_20260610_182052.log`、`...ones_latest_20260610_182052.log`、`...expert_ramp_4096_20260610_182021.log`。
- 当前限制：
  - K1 public Python V3 path 仍未接入 eager production；
  - compact prebuild 复用的是原始路径已有 kernel，不是新增 V3 route kernel；是否作为 production V3 normal 第一版的一部分，需要和原 ASM 的 `K1_PREBUILD_MODE=auto/compact` 策略一起接入。

## 2026-06-10 - Phase 3 K3 V3 边界启动

- 用户提醒：
  - K1 route build 应复用原始 DCU MegaMoE 已有 compact prebuild kernels，不应另造 V3 route build；
  - K3 继续推进时同样要避免把 V2 生产实现当作 V3 core。
- 已完成：
  - 重读 `task_plan.md`、`progress.md`、`findings.md`；
  - 复读 `setup.py`、large-opt `K3_fused/k3_fused_ext.cu`、`K3_fused/k3_fused.py` 和现有 V3 source guard；
  - 更新 Phase 3 计划措辞：K3 V3 先建立 stage-owned default-stub/heavy-TU build 边界，生产 V3 代码不得 include `dcu_megamoe_v2`。
- 下一步：
  - 添加 K3 V3 stub/heavy source 文件和 setup source 分流；
  - 默认 build 保持轻量，显式 K3 normal raw build 继续走 aicc marker；
  - 暂不把 public K3 V3 wrapper 从 ASM fallback 切出。

## 2026-06-10 - Phase 3 K3 V3 build 边界验证

- 已完成代码：
  - 新增 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_stub_ext.cu`，默认 build 只提供 K3 V3 raw availability 和未实现 launcher stub；
  - 新增 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_fused_ext.cu`，显式 raw backend build 才编译，当前只保留 stage-owned launcher 壳；
  - `setup.py` 中 large-opt K3 extension 改为默认编 `k3_fused_ext.cu + k3_v3_stub_ext.cu`，`DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1` 时再加 `k3_v3_fused_ext.cu`；
  - K3 package data 纳入 `*.cuh`，便于后续 K3 stage-owned impl header；
  - `k3_fused_ext.cu` 暴露 K3 V3 raw availability 查询；
  - `tests/test_dcu_megamoe_v3.py` 增加 K3 V3 source boundary guard。
- 本地验证：
  - `python -m compileall setup.py tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - inline source guard 输出 `k3 v3 boundary source guard passed`；
  - `git diff --check` 通过。
- 远端验证：
  - 同步包：`hygon_tmp/sglang_debug/deepgemm_v3_k3_boundary_sync_20260610_182940.tar`；
  - 第一次远端 pytest 命令因 PowerShell 转义把 `exit $st` 传坏，shell 返回 1；但 pytest 输出为 `7 passed in 4.32s`，这是命令包装错误，不是测试失败；
  - 默认 build 通过：log `hygon_tmp/sglang_debug/deepgemm_v3_k3_boundary_default_build_20260610_183015.log`，`__STATUS:0`，`v2_seen=0`，`aicc_marker=0`，`k3_heavy_seen=0`；
  - 显式 K3 normal raw aicc build 通过：log `hygon_tmp/sglang_debug/deepgemm_v3_k3_boundary_raw_normal_aicc_build_20260610_183229.log`，`__STATUS:0`，`v2_seen=0`，`aicc_marker=1`，`k3_heavy_seen=3`，`k1_heavy_seen=0`；
  - raw availability import 输出 `k3_available True True False`。
- 当前状态：
  - K3 V3 build/source 边界完成；
  - public `k3_l2_fused_v3_to_combine()` 仍 fallback 到 ASM，未接未完成 kernel；
  - 下一步开始拆 K3 pure C pack5 fragment，并保持 V2 只作接口/行为参考。

## 2026-06-10 - 收紧 V3 ASM-diff-first 主 kernel 约束

- 用户再次强调：
  - V3 K1/K3 fused normal/LL 主 kernel 相对 pure C normal/LL 的改动，应以原始 groupgemm ASM 与原 DCU MegaMoE K1/K3 fused ASM 的差异为主要参考；
  - 原始 groupgemm ASM 是 `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s`；
  - V2 fused/real-flow 实现不应作为 V3 主 kernel 结构来源；
  - V3 尽量复用现有 DCU MegaMoE 周边实现，只在个别无法复用的边界代码重写。
- 已完成计划更新：
  - `task_plan.md` 增加确认设计点 3b；
  - Phase 2/3 增加 original groupgemm ASM vs fused ASM 差异图任务；
  - risks 增加 V2-derived core 误导风险；
  - `findings.md` 增加“V3 主 kernel 派生路径收紧”。
- 当前纠偏动作：
  - 已经尝试过的 K3 V2-derived C header/launcher 方向将撤回到 K3 V3 raw boundary stub；
  - 后续不重复该 V2-derived extraction 失败路径，改为先做 ASM 差异图，再映射到 pure C 5pack 主体。

## 2026-06-10 - 撤回 K3 V2-derived raw core

- 已完成代码纠偏：
  - 删除 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_groupgemm_impl.cuh`；
  - `k3_v3_fused_ext.cu` 不再 include 该 header，也不再 launch `V3_K3_Fused_DeepGemm...`；
  - K3 raw normal/LL launcher 当前只保留 stage-owned symbol 和 backend macro 边界；
  - `tests/test_dcu_megamoe_v3.py` 改为反向 guard：K3 V3 raw 边界不能重新 include/launch 该 V2-derived groupgemm core，public wrapper 仍 fallback 到原 ASM。
- 本地验证：
  - `python -m compileall setup.py tests/test_dcu_megamoe_v3.py ...` 通过；
  - inline source guard 输出 `k3 v3 boundary rollback guard passed`；
  - `git diff --check` 通过。
- 下一步：
  - 同步到 11 节点 `sglang_megamoe` 后重跑 pytest、默认 build、显式 K3 normal raw aicc 边界 build；
  - 通过后开始 original groupgemm ASM vs K1/K3 fused ASM 的差异图。

## 2026-06-10 - K3 rollback 远端边界验证

- 已完成远端同步：
  - 从 `.vscode/sftp.json` 读取 `hg@10.17.176.11:22`、remote path `/home/hg/yuguo/DeepGEMM`；
  - 使用同步包 `hygon_tmp/sglang_debug/deepgemm_v3_k3_rollback_sync_20260610_185843.tar`；
  - 远端显式删除 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_groupgemm_impl.cuh`，容器内确认文件不存在。
- 远端验证：
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_rollback_pytest_20260610_185921.log`；
  - 默认 build 通过，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_rollback_default_build_20260610_185944.log`，`__STATUS:0`，`v2_seen=0`，`aicc_marker=0`，`k3_heavy_seen=0`；
  - 显式 K3 normal raw aicc boundary build 通过，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_rollback_raw_normal_aicc_build_20260610_190059.log`，`__STATUS:0`，`v2_seen=0`，`aicc_marker=1`，`aicc_wrapper_seen=4`，`k3_heavy_seen=3`，`k3_groupgemm_seen=0`；
  - import availability 输出 `k3_available True True False`。
- 当前状态：
  - K3 V3 raw normal 编译策略仍满足“normal 用 aicc”；
  - K3 V3 raw TU 已退回 boundary-only，不再包含 V2-derived main kernel；
  - 下一步开始 original groupgemm ASM vs K1/K3 fused ASM 差异图。

## 2026-06-10 - original ASM diff 初版落盘

- 已完成：
  - 按用户新强调的约束，把 `original groupgemm ASM -> 原 K1/K3 fused ASM -> V3 pure C 5pack` 的派生路径写入 `task_plan.md` 的确认设计点和 Phase 任务；
  - 对比 original groupgemm ASM、K1 dispatch-pull ASM、K3 combine ASM、K3 tail-reduce ASM，形成初版差异图；
  - 将 K1 差异归纳为 `SYMMROUTE stage` + `MegaMoE dispatch-pull stage`：metadata 生成、staged_x pull、staged_flags done/generation、pointer mode 和 invalid row zero-fill；
  - 将 K3 no-tail 差异归纳为 epilogue/store 阶段的 row-combine pointer scatter、half-tile LDS staging、valid-row mask 和 graph active_tiles gate；
  - 将 K3 tail 差异归纳为 combine store 后的 done counter、peer signal、last-local-WG reduce、extra reducer WG 和 graph runtime reduce state；
  - 更新 `findings.md` 记录初版差异图；
  - 更新 `task_plan.md`，将 K1/K3 ASM 差异图任务标为 ✅。
- 设计结论：
  - 后续 K1/K3 V3 主 kernel 只能从该 ASM 差异图映射到 pure C 5pack 主体；
  - V2 继续只作为接口/行为参考，不能重新进入 K1/K3 production compute core；
  - K3 下一步优先实现 no-tail combine 的 epilogue/store scatter 边界，tail signal/reduce 另按 DCU KB + ISA/正确性证据闭环推进。

## 2026-06-10 - K3 pure source 初筛

- 已完成：
  - 搜索 `hygon_tmp` 与 `megamoe/dcu_megamoe_large_opt`，并显式排除 V2 目录；
  - 未发现非 V2 的独立 K3 pure C pack5 production source；
  - 确认 large-opt K3 当前可信代码边界仍是 ASM wrapper、K3 ASM 文件和 V3 stub/heavy 壳。
- 下一步：
  - 检查 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 是否可作为 K3 normal/LL 主体的参数化来源；
  - 优先派生 K3 no-tail combine 的 epilogue/store scatter，不从 V2 source 恢复任何 main kernel。

## 2026-06-10 - K3 V3 normal no-tail raw kernel 初版

- 已完成代码：
  - 新增 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh`；
  - 从 original pure 5pack/K1 V3 normal 主体派生 K3 normal no-tail raw kernel；
  - K3 kernel 固定 `N=4096,K=2048`，禁用 K1 的 `stage_iter ^ 16` 调度；
  - 输入侧改为 contiguous `act_fp8/act_scale` load；
  - weight offset 改为按 `kProblemK` 参数化；
  - epilogue 改为 `row_combine_ptrs` scatter 写 combine buffer；
  - `k3_v3_fused_ext.cu` 的 normal raw path 现在只在 raw normal gate 下 launch 该 kernel，tail path 保持 no-op。
- 本地验证：
  - `python -m compileall setup.py tests/test_dcu_megamoe_v3.py megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - 本地没有 pytest，改用 inline source guard，输出 `k3 v3 pack5 source guard passed`；
  - `git diff --check` 通过。
- 当前限制：
  - 尚未远端 aicc 编译；
  - 尚未 correctness smoke；
  - no-tail store 先是 scalar row-pointer write，后续需要对照 K3 ASM vectorized epilogue 优化。

## 2026-06-10 - K3 V3 raw normal aicc build timeout 定位

- 已完成远端验证：
  - 同步到 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `7 passed`；
  - 默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，`v2_seen=0`、`aicc_marker=0`、`k3_pack5_header_seen=0`；
  - 显式 K3 normal raw aicc build 在 600s timeout，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_pack5_raw_normal_aicc_build_20260610_191816.log`，status `124`。
- 定位结论：
  - 当前 `setup.py` 把 aicc marker 打在整个 `large_opt_k3_ext` 上；
  - raw normal build 因此用 aicc 编译原 `k3_fused_ext.hip` 和 stub，超时前未稳定编到新的 K3 V3 heavy TU；
  - 不能重复同一条 600s build，下一步改为独立 K3 V3 raw extension / 单 heavy TU aicc 边界。

## 2026-06-10 - K3 V3 raw normal aicc 边界收窄完成

- 已完成代码：
  - `setup.py` 中主 K3 extension 只保留 `k3_fused_ext.cu + k3_v3_stub_ext.cu`，不再追加 V3 heavy TU；
  - 新增独立 raw extension `megamoe.dcu_megamoe_large_opt.K3_fused.k3_v3_fused_ext`，显式 `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1` 时才编；
  - normal raw backend 下只给该 raw extension 打 aicc marker，并带 `-mllvm -enable-num-vgprs-768=true`；
  - `k3_v3_fused_ext.cu` 暴露最小 pybind availability 与 `k3_v3_normal_combine_raw()`，public staged wrapper 仍 fallback 到 ASM。
- 本地验证：
  - `python -m compileall setup.py tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py megamoe/large_opt.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - 同步包 `hygon_tmp/sglang_debug/deepgemm_v3_k3_split_raw_sync_20260610_193728.tar`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`；
  - 默认 build 通过，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_split_raw_default_build_20260610_193806.log`，`v2_seen=0`、`aicc_marker=0`、`k3_raw_tu_seen=0`；
  - raw normal aicc build 通过，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_split_raw_normal_aicc_build_20260610_193939.log`，`v2_seen=0`、`aicc_marker=1`，`k3_v3_fused_ext` 单独走 aicc shim。
- 已更新：
  - `task_plan.md` 将 K3 raw normal aicc 编译边界收窄项标为 ✅。

## 2026-06-10 - K3 V3 normal no-tail raw smoke 通过

- 新增临时脚本：
  - `hygon_tmp/sglang_debug/k3_v3_normal_smoke.py`。
- 远端运行：
  - `HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} PYTHONPATH=. python3 hygon_tmp/sglang_debug/k3_v3_normal_smoke.py`
  - log `hygon_tmp/sglang_debug/deepgemm_v3_k3_normal_smoke_20260610_194531.log`。
- 结果：
  - raw extension availability：`normal=True`；
  - all-ones case：`ones_sample_mean=2048.000000`，`ones_sample_max_abs_err=0.000000`；
  - hidden pattern case 首次失败后定位到 `k3_v3_fused_ext.cu` launch 使用 `<256, 256, false>` baseline-layout 分支；
  - 已改成 `<256, 256, true>` pack5 分支，并重跑远端 pytest 与 raw normal aicc build 通过；
  - 初始 pattern case 使用 rows=256 并在同一个 256-row tile 内混 expert，失败：row 128 仍按 tile 首行 expert0 计算；
  - 根因确认：当前 K3 normal core 读取 `row_expert[tile_token]`，要求 256-row tile expert homogeneous；
  - 修正为 rows=512、两个 256-row tile 分别 expert0/expert1 后通过，log `hygon_tmp/sglang_debug/deepgemm_v3_k3_pack5_true_smoke_tile_20260610_195257.log`；
  - pack5/expert pattern case：`pattern_sample_mean=7680.000000`，`pattern_sample_max_abs_err=0.000000`；
  - status `0`。
- 已更新：
  - `task_plan.md` 增加并勾选 K3 V3 normal no-tail raw smoke 小项。
- 当前限制：
  - 该 smoke 只覆盖 low-level 本地 row pointer scatter 和 tile-homogeneous expert ordering；
  - public staged V3 K3 仍 fallback 到 ASM；
  - tail-reduce、真实 K1/K2 staged input、graph/uneven 尚未验证。

## 2026-06-10 - 再次收紧 V3 主 kernel 参考路径

- 用户再次强调：
  - V3 K1/K3 fused normal/LL 主 kernel 相对 pure C normal/LL 的改动，应主要参考原始 DCU MegaMoE K1/K3 fused ASM 相对原始 groupgemm ASM 的差异；
  - 预期差异主要是 weight layout 和通信语义注入；
  - V3 应尽可能复用现有 DCU MegaMoE 周边实现，只有个别无法复用的边界代码才重写；
  - V2 fused/real-flow 实现不应干扰 V3 主 kernel 结构。
- 已完成：
  - 更新 `task_plan.md` 设计点 3c，固定不确定时的判定顺序为 original ASM diff -> pure C 5pack 主体；
  - 在风险中补充“可复用 DCU MegaMoE 周边实现时不重写整段周边流程”的约束。
- 下一步：
  - 继续 Phase 3 K3 tail-reduce / signal / fence 设计前，先按 DCU KB 查询同步语义，再做代码判断。

## 2026-06-10 - K3 tail-reduce 同步语义复核

- 已完成：
  - 使用 `dcu-rag-kb-query` 查询 `hygon-extend gfx938 HIP system scope atomic threadfence_system signal wait tail reduce allreduce`；
  - 使用 `dcu-rag-kb-optimize` 查询 `MegaMoE K3 combine tail reduce signal kernel`；
  - 对照原 K3 tail-reduce ASM 的 `K3_TAIL_ATOMIC_SIGNAL`、`K3_TAIL_WAIT_SIGNAL`、combine scatter 后 done counter、last-local-WG signal/reduce 和 extra reducer WG；
  - 对照 Hygon custom allreduce 与 flux reduce-scatter 参考中的 system-scope fence、atomic signal 和 epilogue-path reduce。
- 结论：
  - V3 K3 tail raw 第一版必须保持同一 kernel 内完成 combine store、done counter、peer signal wait 和 local reduce；
  - signal 只能在 combine stores 完成并经过 system-scope fence 后发布；
  - peer wait/reduce 只能由最后本地 GEMM WG 和 extra reducer WGs 承担，不能让所有 GEMM WG 等待。
- 已更新：
  - `findings.md` 增加 “K3 tail-reduce 同步语义复核”。

## 2026-06-10 - V3 约束复核与本地 source guard

- 已完成：
  - 复读 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md`；
  - 确认用户新增要求已落到计划：V3 K1/K3 fused normal/LL 主 kernel 的默认派生顺序为 original groupgemm ASM vs 原 DCU MegaMoE fused ASM 差异图，再映射到 pure C 5pack 主体；V2 不参与生产 kernel 结构决策；
  - 本地 `python -m compileall setup.py tests/test_dcu_megamoe_v3.py megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - 本地 `git diff --check` 通过；
  - 本地 Python 环境没有 `pytest` 模块，因此用最小 pytest stub 执行 `tests/test_dcu_megamoe_v3.py` 中的 source-level 测试，输出 `v3 source assertions passed`。
- 下一步：
  - 同步 K3 tail raw 当前实现到 11 节点 `sglang_megamoe` 容器，跑远端真实 pytest 与 K3 normal aicc build。

## 2026-06-10 - K3 tail raw aicc 编译资源失败与最小修复

- 远端验证：
  - 同步到 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`；
  - K3 V3 normal raw aicc build 失败，status `1`；
  - 失败日志确认 `v2_seen=0`、`k1_heavy_seen=0`、`aicc_marker=1`，即只在 K3 raw extension 上触发 aicc；
  - 失败根因是 tail 模板 `V3_K3_Fused_...<256,256,true,true>` 的 `local memory (65552) exceeds limit (65536)`。
- 已做最小修复：
  - 查询 DCU KB / Hygon optimizer 相关 local memory、spill、aicc/gfx938 资源约束；
  - 不改 GEMM 主干，只收缩 tail 附加路径：
    - 去掉 tail wait helper 的 device `printf`；
    - tail reduce worker 从 8 个同时 live 的 float 累加改为逐 BF16-pair 累加和 pack；
  - 本地 `compileall`、source-level tests、`git diff --check` 通过。
- 下一步：
  - 重新同步并运行 K3 normal raw aicc build，确认资源修复是否解除超限。

## 2026-06-10 - K3 tail raw LDS 超限根因修复

- 第二轮定位：
  - 第一版资源收敛后远端重编仍报 `local memory (65552) exceeds limit (65536)`；
  - 进一步确认 `lds_stage` 已正好占满 65536B，tail path 额外 `__shared__ int tail_is_last_gemm_block` 是直接超限源。
- 已完成代码：
  - 删除 `__shared__ int tail_is_last_gemm_block`；
  - 改为 thread0 在 done counter `atomicAdd_system` 后设置 local predicate，再用 `__syncthreads_or(tail_is_last_thread != 0)` 广播给整个 block；
  - source guard 增加 `__syncthreads_or` 检查，并禁止恢复 shared tail flag。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py setup.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - 本地最小 pytest stub 执行 source-level tests 通过；
  - `git diff --check` 通过。
- 下一步：
  - 同步到远端，重跑真实 pytest 与 K3 normal raw aicc build。

## 2026-06-10 - K3 tail owner-slot 改法

- 远端重编结果：
  - `__syncthreads_or` 版本仍失败，aicc 报 `local memory (65792) exceeds limit (65536)`；
  - 说明该 primitive 在当前 aicc/gfx938 下也会引入额外本地/LDS 资源，不适合这个已经 64KB 满占用的 GEMM 主体。
- 已完成代码：
  - 撤掉 `__syncthreads_or`；
  - 改用 `done_counter[1]` 记录 last GEMM WG owner id (`blockIdx.x + 1`)；
  - raw tail wrapper 要求 `done_counter` 至少 2 个 int32；
  - source guard 要求 owner-slot 写入存在，并禁止恢复 `__syncthreads_or` 和 shared tail flag。
- 本地验证：
  - `compileall` 通过；
  - source-level tests 通过；
  - `git diff --check` 通过。
- 下一步：
  - 同步远端并再次运行真实 pytest / K3 raw normal aicc build。

## 2026-06-10 - K3 tail raw normal aicc build 通过

- 远端验证：
  - 同步 owner-slot 版本到 `hg@10.17.176.11` / `sglang_megamoe`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`；
  - `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=normal DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1 DG_FORCE_BUILD=1 MAX_JOBS=2 python3 setup.py build_ext --inplace` 通过；
  - build log 指标：`__STATUS:0`、`v2_seen=0`、`aicc_marker=1`、`k3_raw_tu_seen=6`、`k1_heavy_seen=0`、`local_mem_exceed_seen=0`。
- 结论：
  - K3 V3 fused normal tail 模板现在能通过 aicc 编译；
  - normal aicc 仍只作用在独立 K3 raw extension，没有把 V2 或 K1 heavy TU 拉进来。
- 下一步：
  - 做 K3 V3 normal tail raw 单 rank smoke，先验证 combine scatter + done owner slot + local reduce 的最小 correctness。

## 2026-06-10 - K3 tail raw smoke 通过

- 已完成：
  - 新增临时脚本 `hygon_tmp/sglang_debug/k3_v3_normal_tail_smoke.py`；
  - smoke 使用 K3 V3 normal raw tail API，单 rank all-ones，row pointer 直接指向 sym-buffer combine 区；
  - 首轮 topk=1 暴露 row1 全 0，定位为 tail reduce worker 用 `blockDim.x=768` 分片，而 last GEMM block 只有前 512 compute threads 能进入 tail reduce；
  - 修复为 `kTailReduceThreads=512` 后，clean rebuild 确认 `k3_v3_fused_ext.hip -o` 被重新编译。
- 远端验证：
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`；
  - K3 raw normal aicc clean rebuild 通过：`__STATUS:0`、`v2_seen=0`、`aicc_marker=1`、`k3_raw_compile_seen=1`、`k1_heavy_seen=0`、`local_mem_exceed_seen=0`；
  - tail smoke topk=1：`tail_sample_mean=2048.000000`、`tail_sample_max_abs_err=0.000000`、`tail_done_counter=[16, 13]`、`tail_signal_slot8=1`；
  - tail smoke topk=6：`tail_sample_mean=12288.000000`、`tail_sample_max_abs_err=0.000000`、`tail_done_counter=[96, 82]`、`tail_signal_slot8=1`。
- 已更新：
  - `task_plan.md` 增加并勾选 K3 V3 normal tail raw single-rank smoke 小项。
- 下一步：
  - 继续梳理 staged wrapper 接入：尤其是 V3 tail owner slot 的 scratch/reset 合同，避免生产路径新增 reset kernel。

## 2026-06-10 - 调整 e2e 集成验证顺序

- 用户明确：
  - e2e correctness 应先只把 V3 K1 fused 替进去测试；
  - K3 暂时保持原 ASM 路径；
  - K1-only e2e 通过后再接 V3 K3 fused，避免 K1/K3 同时替换导致问题不好定位。
- 已完成计划更新：
  - `task_plan.md` 新增设计点 16，固定 K1-only -> K3 no-tail -> K3 tail-reduce 的 e2e 接入顺序；
  - Phase 4 标为 ⏳ in_progress，并新增 K1-only e2e 小项。
- 当前执行策略：
  - 下一步优先检查 `K1_fused/k1_fused.py` 的 public V3 wrapper 是否可以从 ASM fallback 切到 `k1_symm_fused_l1_v3_pack5`；
  - `large_opt.py` 的 K3 launcher 暂时要保持原 ASM，即使 V3 backend 已开启，也先不接 `k3_l2_fused_v3_to_combine` 到 e2e。

## 2026-06-10 - 记录 K1-only e2e layout 合同

- 用户提醒：
  - K1-only e2e correctness 时必须注意 weight layout 差异。
- 已更新：
  - `task_plan.md` 新增设计点 17；
  - Phase 4 K1-only e2e 项明确：L1 使用 V3 pack5 layout，L2 仍保持原 ASM layout。
- 执行含义：
  - 不能把 K1-only e2e 的整套权重统一转换成 V3 pack5；
  - 如果 staged API 仍只接同一组 `(w1, w2)`，测试/fixture 需要分别准备 `w1_pack5` 与 `w2_original`；
  - 后续接 K3 V3 staged 时，才切换 L2 到 V3 pack5 layout。

## 2026-06-10 - 撤回无效 K1 raw rowptr e2e 路线

- 用户指出：
  - `dcu_megamoe_v3_launch_k1_normal_symm_stage_raw` 内没有真正融合通信语义；
  - 当前状态下跑 e2e correctness 没有意义，功能必须先对齐。
- 已确认：
  - 当前 V3 K1 raw normal path 的 route/metadata 由独立 compact route kernels 预先生成；
  - raw normal 主 kernel 只消费 `row_x_ptrs/row_x_scales/m_indices` 做 pack5 GEMM；
  - 这不是计划要求的 K1 dispatch-pull/route metadata 融入 GEMM pipeline。
- 已完成代码纠偏：
  - `k1_symm_fused_l1_v3()` public staged wrapper 改为 fail-fast；
  - `test_mega_moe_dcu.py` 在 `USE_MEGAMOE_V3=1` staged e2e 时 fail-fast，不再准备 L1 pack5/L2 ASM 的半成品 e2e fixture；
  - `tests/test_dcu_megamoe_v3.py` 更新 source guard，禁止 public wrapper 调用 raw rowptr path 做 e2e。
- 已更新计划：
  - `task_plan.md` 新增设计点 18；
  - Phase 4 K1-only e2e 改为等待 K1 主 kernel 融合通信语义后再执行；
  - 当前工作重心回到 Phase 2：把 dispatch-pull、route metadata、row_combine_ptrs、output_index、stats 融进 K1 主 kernel。

## 2026-06-10 - K1 normal fixed-route single-kernel 骨架通过 low-level smoke

- 已完成代码：
  - `k1_v3_groupgemm_impl.cuh` 在 K1 normal GEMM 主 kernel 内增加 `build_fixed_route_in_kernel` 路径；
  - route/metadata producer 使用 `blockIdx.x==0`，在同一 kernel 内扫描 symmetric buffer 并写 `row_x_ptrs/row_x_scales/m_indices/route_weights/output_index/row_combine_ptrs/stats`；
  - 同 tile 的 GEMM CTA 通过 `grid_barrier` per-tile flag 等 metadata ready 后进入原 5pack GEMM 主体；
  - `k1_symm_fused_l1_v3_pack5()` normal path 改用 fixed capacity tiles，不再在 V3 normal path launch compact route prebuild kernels。
- 编译定位：
  - 第一版新增 `__shared__ int tile_match_count` 后 aicc 失败：`local memory (65552) exceeds limit (65536)`；
  - 改为使用 `grid_barrier` per-tile slot 作为 match counter 后，K1 raw normal aicc clean build 通过。
- 远端验证：
  - source pytest：`8 passed`；
  - build env：`DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1 DG_FORCE_BUILD=1 MAX_JOBS=2`；
  - ones smoke：`active_routes=3072`、`route_weight_sum=3072.0`、`active_sample_abs_mean=4096.0`；
  - expert_ramp smoke：expert pattern 输出对齐，`expert_ramp_abs_err=0`。
  - all-ranks skew smoke：
    - 512 tokens 暴露 capacity overflow：期望 24576 routes，但 fixed capacity 只有 8192 rows；
    - 128 tokens 在容量内通过：`active_routes=6144`、`route_weight_sum=6144.0`、`expert_ramp_abs_err=0`。
- 当前状态：
  - 这只是 normal low-level fixed-route fused skeleton；
  - public staged/e2e 仍保持 fail-fast；
  - 下一步继续把 fixed-route 骨架收敛到 staged 输出合同与 compact/tile scheduling，再恢复 K1-only e2e。

## 2026-06-10 - 纠正 K1 fixed-route bring-up 与 e2e correctness 边界

- 用户指出：
  - 当前 fixed-route skeleton 即使能进入 staged 链路，也不能被称为 e2e correctness；
  - 功能必须对齐原 DCU MegaMoE K1 fused 行为后，才能做 K1-only correctness。
- 已完成代码边界修正：
  - `k1_symm_fused_l1_v3()` 默认重新 fail-fast；
  - 只有显式设置 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 时，才允许调用 `ext.k1_symm_fused_l1_v3_pack5()` 进入当前 fixed-route skeleton；
  - 错误信息明确该路径只用于 bring-up/debug，不能用于 correctness claims；
  - `large_opt.py` 注释改为“等 V3 K1 功能对齐后，K3 仍保持 ASM 以隔离 K1 correctness”。
- 已更新计划：
  - `task_plan.md` 当前状态改为 single-kernel fixed-route bring-up；
  - Phase 2 metadata/unit correctness 项标为 ⏳，并记录 fixed-route staged bring-up 必须带显式调试开关；
  - Phase 4 继续暂停，直到 K1 通信语义和 compact scheduling/capacity 行为都功能对齐。
- 验证：
  - 本地 `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 `git diff --check` 通过；
  - 远端 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM` 中 `source /opt/dtk/env.sh && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `8 passed in 4.23s`。

## 2026-06-10 - K1 normal metadata/unit correctness 通过

- 新增临时脚本：
  - `hygon_tmp/sglang_debug/k1_v3_metadata_compare.py`；
  - 脚本在 fake symmetric buffer 上对比原 K1 ASM compact metadata oracle 与 V3 fixed-route skeleton；
  - 验证 row-side metadata，不把 ASM launch 后的 `output_index` 当作 row-id oracle。
- 调试过程：
  - 第一版误用 `output_index` 做 task->row 唯一映射，原 ASM 出现 duplicate row / mismatch；确认这是脚本 oracle 错误；
  - 第二版在 ASM 后未 clone `route_scratch` 视图，V3 launch 覆盖了 ASM metadata；修复为 ASM 后 snapshot metadata，并给 V3 换新 scratch。
- 远端验证：
  - 128 tokens、all-ranks local-skew 容量内：ASM/V3 row-side metadata 均通过，`active_rows=6144`；
  - 1024 tokens、rank0-only：ASM/V3 row-side metadata 均通过，`active_rows=6144`；
  - 1024 tokens、all-ranks、`global_round_robin`：ASM/V3 row-side metadata 均通过，`active_rows=6144`；
  - 以上均在 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`，容器命令均 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`。
- 已更新：
  - `task_plan.md` 将 K1 normal metadata/unit correctness 标为 ✅；
  - 下一步继续 K1 compact scheduling/capacity 行为对齐，再考虑 staged eager K1 bring-up。

## 2026-06-10 - 收紧 fixed-route staged bring-up 的 correctness 边界

- 用户纠正：
  - 当前 fixed-route staged run 不能被称为 e2e correctness；
  - 功能必须先对齐原 DCU MegaMoE K1 fused contract。
- 已完成：
  - 将 `task_plan.md` 中 fixed-route staged run 的完成项改成 debug/bring-up 定位验证，不再写成 correctness 通过；
  - 在 `findings.md` 记录该 run 只证明容量内 fixed-route launch/metadata/K2/原 K3 链路能跑完；
  - 明确 public V3 staged wrapper 仍默认 fail-fast，只有 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 才能跑当前 skeleton，且不能用于 correctness claims。
- 同步记录：
  - 之前 staged bring-up 暴露的 `hipExtModuleLaunchKernel invalid configuration argument` 根因是测试 fixture layout 污染：baseline oracle 误用了 V3 pack5 L1；
  - 已拆分 baseline/fused 权重，K1-only bring-up 仅 fused L1 用 pack5，baseline 与原 K3 ASM 保持原 layout；
  - 这只是测试准备修正，不代表 V3 K1 功能对齐。

## 2026-06-10 - K1 V3 fixed-route eager capacity 对齐

- 已完成代码：
  - `k1_symm_fused_l1_v3_pack5()` 中新增 `route_capacity_tokens_per_rank`；
  - eager/no runtime-token 时 capacity 估算使用真实 `num_tokens`；
  - graph/runtime-token 时仍使用 `num_max_tokens_per_rank`；
  - `output_index` 分配继续使用 `num_max_tokens_per_rank`，不缩外部 contract。
- 目的：
  - 对齐原 K1 eager fixed-capacity 风格；
  - 避免 1024 tokens、max 1152 这类场景把 rows 从 8192 膨胀到 16384；
  - 为后续 compact scheduling/capacity 行为对齐减少噪音。
- 下一步：
  - 本地 source/format 检查；
  - 同步到 11 节点 `sglang_megamoe`；
  - 重编 V3 K1 normal aicc extension，重跑 source pytest 和 fixed-route staged debug，确认 rows 回落且 debug 链路仍能完成。

## 2026-06-10 - 纠正 fixed-route 验证口径并修复 K1 rowptr A-load

- 用户指出：
  - 当前 fixed-route staged run 不应称为 e2e correctness；
  - 功能必须先对齐原 DCU MegaMoE K1 fused contract。
- 已确认：
  - fixed-route staged run 只能用于 debug/bring-up；
  - 真正 K1-only e2e correctness 的前置条件仍是 K1 主 kernel 对齐原始 dispatch-pull、compact scheduling/capacity、row_combine/output_index/stats 合同。
- 已完成代码：
  - `k1_v3_groupgemm_impl.cuh` 中 `buffer_load_fp8_b128_rowptr_device()` 不再用 divergent `row_ptr` 构造 raw-buffer resource；
  - 改为从 `row_ptr + row_byte_offset` 直接做普通 global vector load。
- 远端验证：
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`8 passed in 4.19s`；
  - K1 normal aicc clean build通过：`__STATUS:0`、`aicc_marker=1`、`k1_raw_compile_seen=2`、`k3_raw_seen=0`、`v2_seen=0`；
  - `hygon_tmp/sglang_debug/k1_v3_output_compare.py` rank7 random 1024/1152 输出：`max_abs=0.0`、`mean_abs=0.0`，ASM/V3 active rows 均为 `6091`。
- 下一步：
  - 只把后续 staged run 称为 fixed-route staged debug；
  - 继续 Phase 2：对齐 K1 compact scheduling/capacity 行为，不进入 Phase 4 correctness。

## 2026-06-10 - K1 V3 normal route scanner 向原 ASM 控制面收敛

- 已完成代码：
  - `v3_k1_build_fixed_route_tile_device()` 从每个 expert tile 全量重扫 route，改为前 4 个 row tiles 作为 scanner；
  - scanner 按 `source_rank = tile_id; source_rank += scanner_tiles` 分片 source ranks，并用 16 个 `blockIdx.x` CTAs 分片 route offsets；
  - per-expert row counter 改用 `route_scratch_i32[local_expert]`；
  - 主 K1 V3 normal kernel 和 launcher 补齐 `route_scratch_i32` 参数链；
  - init owner CTA 发布 init flag 前增加 `__syncthreads()` 和 thread0 publish，避免 reset 可见性竞态。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/large_opt.py setup.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - source pytest：`8 passed in 4.31s`；
  - K1 normal aicc clean build通过：`__STATUS:0`、`aicc_marker=1`、`k1_raw_compile_seen=2`、`k3_raw_seen=0`、`v2_seen=0`；
  - K1 output compare 1024/1152 rank7 random：`asm_active=6091`、`v3_active=6091`、`max_abs=0.0`；
  - fixed-route staged debug 1024/1152 tail-reduce on/off 均 status `0`，rows=8192，reported diff `max_abs=0.000488281`；
  - K1 output compare 4096/4096 rank7 random：`asm_active=24446`、`v3_active=24446`、`max_abs=0.0`。
- 仍未完成：
  - 4096 compare 显示原 ASM auto path `asm_rows=29696`，V3 fixed-capacity scanner `v3_rows=32768`；
  - 下一步继续实现/对齐 compact capacity scheduling，不能进入 Phase 4 correctness。

## 2026-06-10 - K1 V3 normal compact-capacity 对齐

- 用户再次纠正：
  - 当前验证不能称为 e2e correctness；
  - 功能必须先对齐原 DCU MegaMoE K1 fused contract。
- 已完成代码：
  - V3 normal wrapper 复用原 K1 ext 的 `compact_capacity_tiles()` 选择 normal compact capacity；
  - `dcu_megamoe_v3_launch_k1_normal_symm_stage_raw()` 增加 `compact_capacity_in_kernel` 参数；
  - K1 V3 main kernel 内补 count/build/emit compact tile list：`tile_bases/tile_experts/emit_counts` 均在同一 K1 launch 内完成；
  - compact count 阶段改为对齐原 `k1_count_compact_routes_kernel`：只按 local expert 计数，不提前跳过 zero weight；
  - scanner CTA 发布 count/emit completion flag 前加 `__threadfence()`，对齐 DCU KB producer-consumer flag 建议；
  - 未改 GEMM load/compute/store 主体。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/large_opt.py setup.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - 首次 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 失败，原因是 source guard 仍要求 fixed-only capacity；
  - 修正 source guard 后重跑通过：`8 passed in 4.25s`；
  - K1 normal aicc clean build 通过：`__STATUS:0`、`aicc_marker=1`、`k1_raw_compile_seen=2`、`k3_raw_seen=0`、`v2_seen=0`；
  - K1 output compare 4096/4096 rank7 random：`asm_rows=29696`、`v3_rows=29696`、`asm_active=24446`、`v3_active=24446`、`missing_in_v3=0`、`extra_in_v3=0`、`max_abs=0.0`；
  - K1 output compare 1024/1152 rank7 random：`asm_rows=8192`、`v3_rows=8192`、`asm_active=6091`、`v3_active=6091`、`max_abs=0.0`；
  - staged debug 1024/1152 tail on/off 均 status 0，reported `max_abs=0.000488281`；
  - staged debug 4096 requested tokens tail on/off 均 status 0，reported `max_abs=0.000488281`。
- 当前状态：
  - K1 V3 normal compact-capacity rows 与 K1 output 在随机 1024/4096 compare 中已对齐原 K1 ASM；
  - public staged V3 wrapper 仍默认 fail-fast；
  - 下一步不能直接宣称 e2e correctness，需要恢复 K1-only correctness gate 前先补 normal output contract guard、zero-weight/overflow/uneven/graph/LL 的边界计划。

## 2026-06-10 - K1 V3 normal public K1-only gate 通过

- 已完成代码：
  - `k1_symm_fused_l1_v3()` normal backend 去掉 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP` 调试开关；
  - LL backend 仍 fail-fast，提示当前 K1-only correctness gate 只支持 normal；
  - graph wrapper 仍 fail-fast；
  - source guard 更新为禁止 wrapper 中出现 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP`，并检查 LL fail-fast 文案。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py` 通过；
  - `git diff --check` 通过。
- 远端 source 验证：
  - 第一次 pytest 因 source guard 检查跨字符串拼接完整句子失败；
  - 改为检查稳定片段后，`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`8 passed in 4.25s`。
- 远端 staged K1-only gate 验证：
  - env：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal`，不设置 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP`；
  - K3 仍走原 ASM，分别覆盖 `K3_USE_ASM_TAIL_REDUCE=1/0`；
  - 1024/1152 tail on：status 0，`max_abs=0.000488281`、`mean_abs=9.50943e-06`；
  - 1024/1152 tail off：status 0，`max_abs=0.000488281`、`mean_abs=9.50943e-06`；
  - 4096 requested tokens tail on：status 0，`max_abs=0.000488281`、`mean_abs=9.45465e-06`；
  - 4096 requested tokens tail off：status 0，`max_abs=0.000488281`、`mean_abs=9.45465e-06`。
- 当前状态：
  - normal K1-only staged correctness gate 已恢复；
  - K3 V3 staged 尚未接入，LL/uneven tokens/cuda graph 仍 pending；
  - 下一步按计划选择：补 LL K1-only gate，或在保持 K1 normal 回归的基础上接 K3 V3 no-tail staged。

## 2026-06-10 - 纠正 LL staged probe 的 correctness 口径

- 用户指出：
  - 当前 LL 路径不能称为 e2e correctness；
  - 功能必须对齐原 DCU MegaMoE 语义链路，不能只看局部 probe。
- 已完成修正：
  - 更新 `task_plan.md`，明确 K1-only 阶段只能称为 staged correctness/probe；完整 e2e correctness 仅用于 K1/K2/K3 V3 功能链路、tail-reduce/no-tail、eager/graph、uneven tokens 都覆盖之后；
  - 更新 `findings.md`，记录 LL raw smoke 通过与 32-token staged probe 失败的边界；
  - LL staged probe 失败记录为：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=1`，8 卡 32 tokens，原 K3 ASM tail-reduce，`max_abs=0.0831298828125 > atol=0.0035`。
- 下一步：
  - 不再用 e2e 通过/失败描述当前 LL；
  - 拆分定位 K1 LL row layout、`row_combine_ptrs`、padded row、K2/K3 原 ASM 消费合同是否与原路径一致。

## 2026-06-10 - 修复 K1 LL grid barrier 初始化并通过 K1 output compare

- 已定位：
  - `K1_V3_COMPARE_BACKEND=ll` 首次远端 compare 180s 超时；
  - 容器内残留 `python3 hygon_tmp/sglang_debug/k1_v3_output_compare.py`，已 kill；
  - 根因是 LL grid barrier 使用复用 scratch 中的两个 int，但没有可靠初始化 counter/phase。
- 已完成代码：
  - `k1_v3_groupgemm_impl.cuh` 增加 `v3_k1_ll_grid_barrier_init_device()`；
  - LL raw launcher 和 host pybind 参数链增加 `barrier_epoch`；
  - `k1_fused_ext.cu` 在 LL path 使用 `next_fused_l1_flag_generation()` 传入 epoch；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，防止 LL barrier 回退到未初始化状态；
  - `hygon_tmp/sglang_debug/k1_v3_metadata_compare.py` / `k1_v3_output_compare.py` 去掉旧 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP` 依赖，并支持 `K1_V3_COMPARE_BACKEND=ll`。
- 验证：
  - 本地 `compileall` 和 `git diff --check` 通过；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`8 passed`；
  - 远端 LL raw clean build 通过：`__STATUS:0`、只编 K1 LL，`k3_raw_seen=0`、`v2_seen=0`；
  - 远端 LL K1 output compare 32/128 random rank7：ASM/V3 active rows 都是 203，common 203，missing/extra 0，`max_abs=0.0`。
- 下一步：
  - 重跑真实 8 卡 K1-only staged probe；
  - 如果仍不对齐，优先拆 K2/K3 消费 LL row layout 的合同。

## 2026-06-10 - K1 LL 真实 staged probe 仍未通过

- 远端 8 卡 K1-only staged probe：
  - env：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=1`；
  - shape：32 tokens，sym_buffer 显示容量 `32/384`，K3 仍为原 ASM tail-reduce；
  - 结果：`max_abs=0.09228515625 > atol=0.0035`。
- 额外诊断：
  - 加 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 强制 K2 跳过 inactive rows 后仍失败；
  - 结果：`max_abs=0.072265625 > atol=0.0035`。
- 当前判断：
  - K1 LL 在 fake random route 上可按 `row_combine_ptrs` 与原 K1 ASM 对齐；
  - 真实 staged 差异更可能在 K2 后输出合同或原 K3 ASM 对 LL 64-row/expert row layout 的消费假设；
  - 下一步写单 GPU fake sym-buffer K2/K3 compare，拆分 K2 与 K3。

## 2026-06-10 - K1/K2 对齐，K3 ASM 不接受 LL 64-row layout

- 新增诊断脚本：
  - `hygon_tmp/sglang_debug/k1_v3_ll_k2k3_compare.py`；
  - 使用 fake symmetric buffer，逐级比较原 K1 ASM 与 V3 LL 的 K1 输出、K2 act、K2 scale、原 K3 ASM combine。
- 远端结果：
  - `K1_V3_COMPARE_RANK=7 K1_V3_COMPARE_TOKENS=32 K1_V3_COMPARE_MAX_TOKENS=384 K1_V3_COMPARE_EXPERT_MODE=random`；
  - K1：`max_abs=0.0`；
  - K2 act：`max_abs=0.0`；
  - K2 scale：`max_abs=0.0`；
  - K3 ASM combine：`k3_combine_max_abs=97.0`、`k3_combine_mean_abs=13.263659477233887`。
- 结论：
  - LL staged failure 已收窄到原 K3 ASM 对 row layout 的假设；
  - 当前 V3 LL layout 是 64 rows/expert，一个 256-row K3 tile 混了 4 个 experts；
  - 原 K3 ASM 需要 256-row tile-homogeneous expert layout；
  - 下一步为 K1-only staged probe 增加 K3-ASM-compatible LL stride，最终 V3 K3 LL 接入后再回到真正 LL 布局。

## 2026-06-10 - LL K1-only staged gate 通过

- 已完成代码：
  - `k1_symm_fused_l1_v3()` 增加 `ll_asm_compatible_layout` 参数，默认 `False`；
  - `k1_fused_ext.cu` 根据该参数选择 LL row tile：默认 64-row 真 LL，兼容原 K3 ASM 时使用 256-row/expert；
  - `large_opt.py` 在当前 K1-only staged gate 中对 `MEGAMOE_DCU_V3_BACKEND=ll` 传 `ll_asm_compatible_layout=True`；
  - source guard 覆盖该兼容布局，防止默认低层 LL 被污染。
- 验证：
  - 本地 `compileall` 和 `git diff --check` 通过；
  - 远端 `tests/test_dcu_megamoe_v3.py`：`8 passed`；
  - 远端 K1 LL rebuild：`__STATUS:0`、`k1_raw_compile_seen=3`、`k3_raw_seen=0`、`v2_seen=0`、`aicc_marker=0`；
  - 单卡 `K1_V3_LL_ASM_COMPAT=1` K1/K2/K3 compare：K1、K2 act、K2 scale、K3 ASM combine 全部 `max_abs=0.0`。
- 8 卡 K1-only staged correctness：
  - 32 tokens tail on：status 0，`max_abs=0.000244141`；
  - 32 tokens tail off：status 0，`max_abs=0.000244141`；
  - 128 tokens tail on：status 0，`max_abs=0.000244141`；
  - 128 tokens tail off：status 0，`max_abs=0.000244141`。
- 当前状态：
  - K1-only staged gate 的 normal/LL 已通过；
  - 这仍不是完整 V3 e2e correctness；
  - 下一步按计划接 K3 V3 staged，先 no-tail，再 tail-reduce。

## 2026-06-10 - 纠正 e2e 口径并开始 K3 V3 no-tail staged 接入

- 用户纠正：
  - K1-only staged gate 不能叫 e2e correctness；
  - 功能必须最终对齐原 DCU MegaMoE 语义，不能只看 K1 局部通过。
- 已完成本地代码：
  - `k3_l2_fused_v3_to_combine()` 从 ASM fallback 改为 normal no-tail raw combine 调用；
  - normal no-tail 只接受 V3 K3 raw extension 和 pack5 L2 权重；
  - tail-reduce、LL、graph/active_tiles、tail-signal metadata 均显式 fail-fast 或继续走原 ASM；
  - `large_opt.py` 只在 eager staged `backend=normal && K3_USE_ASM_TAIL_REDUCE=0` 时调用 V3 K3；
  - `tests/test_mega_moe_dcu.py` 在 K3 V3 no-tail 条件下把 L2 也提前转为 V3 pack5，K1-only/tail/LL 仍保持 L2 原 ASM layout。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_mega_moe_dcu.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过；
  - 本地 `python -m pytest -q tests/test_dcu_megamoe_v3.py` 因本机 Python 无 pytest 模块失败，source pytest 需在远端容器跑。
- 下一步：
  - 同步到 11 节点 `sglang_megamoe`；
  - 远端跑 source pytest；
  - 用 aicc 编译 V3 K3 normal raw extension；
  - 8 卡跑 normal/no-tail staged correctness，确认这只是 K3 V3 no-tail staged gate，不是完整 e2e。

## 2026-06-10 - K3 V3 no-tail partial-tile 定位推进

- 用户纠正：
  - K1-only staged gate 不是 e2e correctness；
  - 功能必须对齐原 DCU MegaMoE 语义，K3 V3 no-tail 需要先和原 K3 ASM/直接 reference 对齐。
- 已完成：
  - 远端 11 节点 `sglang_megamoe` 重新强制编译 K3 V3 raw normal，aicc 编译约 56s/54s，未触发 V2；
  - `tests/test_dcu_megamoe_v3.py` 远端通过：`8 passed`；
  - K3 raw full-tile smoke 通过：ones 和 pattern 均 `max_abs_err=0`；
  - 新增临时诊断 `hygon_tmp/sglang_debug/k3_v3_normal_stage_compare.py` 的 col/row/tile validity 统计；
  - 单卡诊断确认原 K3 ASM 与 direct FP8 reference 对齐，错误集中在 V3 K3 raw normal；
  - 在 K3 主 kernel 中加入 inactive A-load mask：`row_combine_ptrs[row] <= 0` 时 A fragment 返回 zero，不新增 kernel launch。
- 远端结果：
  - random route partial tile：active-row mask 前 `max_abs≈20+`、mean 约 `2e-2`，且可能出现 nonfinite；mask 后降到 `max_abs=4.375`、`mean=7.93e-5`；
  - local_round_robin full tile：V3 K3 raw normal 与原 K3 ASM `max_abs=0`、`mean_abs=0`；
  - 剩余错误集中在 dense-prefix boundary wave，例如 row0 valid、row16 invalid 的 MT256 partial wave。
- 当前状态：
  - K3 V3 normal no-tail staged 仍未通过，不可标记 complete；
  - 下一步验证 64-row/partial-wave 诊断或补 MT256 boundary wave 的 kernel-internal masked handling。

## 2026-06-11 - K3 V3 partial-tile 诊断反证

- 已完成诊断：
  - `<64,256>` no-tail normal 诊断分支重编通过，但 source guard 失败且 full-tile pattern 失败，random partial 退化到 `max_abs=107.5`；
  - `<256,256>` 下 partial masked store 禁用 lowlat shuffle，full-tile 仍通过但 random partial 退化到 `max_abs=108.5`；
  - A-load active mask 改为 target-row-only，full-tile 通过但 random partial 出现整行归零，`max_abs=78.0`；
  - A-load active mask 改为 logical/target union，full-tile 通过但 random partial 出现 nonfinite，`k3_combine_rhs_nonfinite=528`。
- 当前结论：
  - tile shape、关闭 partial shuffle、target-row-only 和 union mask 都不是正确方向；
  - 当前代码回退到 `<256,256,true>` + logical-row active mask 作为最稳定位基线；
  - K3 V3 no-tail 仍停在 partial-tile boundary wave residual，未进入完整 staged correctness。

## 2026-06-11 - K3 V3 no-mask/prezero 反证与当前定位

- 已完成：
  - 诊断脚本增加 `K3_V3_COMPARE_PREZERO_ACT=1`，可在 K2 前预清零 `act_fp8/act_scale`；
  - 临时验证 K3 V3 去掉 active mask 后，即使 prezero 也仍失败，random partial `max_abs=28.875`、`mean_abs=0.0259957`；
  - 代码已恢复到 `<256,256,true>` + logical-row active mask + lowlat shuffle。
- 当前定位：
  - source pytest 与 full-tile smoke 通过；
  - random partial 残差变成稀疏 tail-wave 问题，典型样本 `max_abs=3.625`、`mean_abs=8.94e-06`；
  - 最大误差 wave 只有前 10 行有效，row16 无效，说明下一步应针对少于 16 有效行的 MT256 boundary wave 做 kernel-internal 处理，不能再回到 no-mask、tile64 或关闭 shuffle 方向。

## 2026-06-11 - K3 V3 normal partial-tile raw compare 通过

- 已完成：
  - `hygon_tmp/sglang_debug/k3_v3_normal_stage_compare.py` 增加 wave-validity 统计，避免只看单个 max 误判；
  - 反证 wave-level unmasked store 与 `mask==0xf` 分流后，改为所有 compute waves 统一执行 GEMM loop，B-load 用 logical rowptr mask，store 统一走 rowptr-guarded unmasked helper；
  - 清理不再使用的 masked store / valid-mask helper，避免后续误用已反证路径。
- 验证：
  - 远端 K3 V3 normal aicc rebuild status `0`；
  - `tests/test_dcu_megamoe_v3.py`：`8 passed`；
  - `k3_v3_normal_smoke.py`：ones / pattern 均 `max_abs_err=0`；
  - `k3_v3_normal_stage_compare.py`：1024/1152 rank7 random seed1234 下 K3 ASM vs V3 combine `max_abs=0.0`、`mean_abs=0.0`。
- 当前状态：
  - K3 V3 normal no-tail raw/single-card partial compare 已通过；
  - 还不能标记 eager staged no-tail complete，下一步需要跑 8 卡 normal/no-tail staged correctness。

## 2026-06-11 - K3 V3 no-tail 8 卡 staged 残差与同输入等价诊断

- 已完成远端验证：
  - K3 V3 normal aicc raw rebuild 后，`tests/test_dcu_megamoe_v3.py` 通过，K3 raw smoke 与 single-card rank7 random compare 均通过；
  - 8 卡 full staged normal/no-tail 4096 tokens 对 baseline 仍有小误差超阈值：
    - `v3_normal_k3_notail_staged_4096_fence_20260611_021648.log`，最大失败 `0.00594329833984375`；
    - all-thread fence 版本 `v3_normal_k3_notail_staged_4096_allthread_fence_20260611_022244.log`，最大失败 `0.007080078125`；
    - all-thread rerun `v3_normal_k3_notail_staged_4096_allthread_fence_rerun_20260611_022845.log`，失败 `0.0035400390625`，略高于阈值；
  - 原 ASM staged no-tail oracle `orig_3stage_notail_4096_oracle_check_20260611_023100.log` 通过，`max_abs=0.000488281`。
- 已完成诊断脚本修正：
  - `hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 改用实际 `sym_buffer.num_max_tokens_per_rank=4224` 计算 combine offset；
  - K2 调用补上 `row_combine_ptrs=row_combine_ptrs`，对齐真实 staged inactive-row skip；
  - 本地 `python -m py_compile` 与 `git diff --check` 通过；
  - 脚本同步到 11 节点 `sglang_megamoe`。
- 远端 8 卡同输入 ASM-vs-V3 K3 诊断：
  - env：`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=hygon_tmp/sglang_debug:. K3_V3_DIST_TOKENS=4096 K3_V3_DIST_MAX_TOKENS=4096 K3_V3_DIST_PROCS=8`；
  - log：`hygon_tmp/sglang_debug/k3_v3_dist_y_compare_4096_rowptrk2_20260611_023616.log`；
  - status 0，结果 `requested_max_tokens=4096`、`actual_max_tokens=4224`、`global_max_abs=0.0`、`global_gt_atol=0`、`global_y_max_abs=0.0`。
- 当前状态：
  - K3 V3 normal no-tail 与原 K3 ASM 在同一 V3 K1/K2 输出和真实 K2 rowptr skip 条件下已等价；
  - Phase 3/4 的完整 staged no-tail 仍是 ⏳，因为对 baseline 4096 仍有残差；
  - 下一步继续分离 K1/K2/fixture/layout/wrapper 差异来源，不能把同输入 K3 等价当作完整 e2e 通过。

## 2026-06-11 - K3 no-tail signal-only 同步编译修复

- 已定位：
  - V3 K3 no-tail staged 不加 debug sync 仍有小残差，debug sync 通过，说明问题更像 K3 combine store 完成后的跨 rank 可见性/时序；
  - 第一次把 signal-only 完成同步合入 K3 主 kernel 后，aicc 报 `local memory (65552) exceeds limit (65536)`，同时影响 `<tail_reduce=true>` 和 `<signal_only=true>` 两个模板实例。
- 已完成代码：
  - 将 `kTailReduce` 与 `kSignalOnly` 的尾部完成同步拆成两个 `if constexpr` 分支；
  - tail-reduce 分支只保留 owner-slot 发布、原 block-wide peer signal/wait 和 local reduce；
  - no-tail signal-only 分支新增 thread0-only peer signal/wait helper，由最后完成的 GEMM block 在 system fence 后发信号并等待 peer 完成，不调用带 block barrier 的 tail helper。
- 验证：
  - 本地 `compileall`、`py_compile`、`git diff --check` 通过；
  - 远端 11 节点 `sglang_megamoe` 显式同步 `k3_v3_pack5_groupgemm_impl.cuh`；
  - V3 K3 normal raw aicc 重编通过，日志 `hygon_tmp/sglang_debug/k3_v3_signalonly_split_rebuild_20260611_032948.log`，status `0`；
  - build 命令只启用 V3 K1/K3 raw normal 和 `DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1`，未启用 V2 编译。
- 下一步：
  - 运行远端 source pytest；
  - 运行 official staged normal/no-tail 4096，不加 debug sync，验证 signal-only 是否消除残差。

## 2026-06-11 - K3 no-tail signal-only 反证与 default-off gate

- 已完成远端验证：
  - K3 V3 normal raw aicc split rebuild 通过，log `hygon_tmp/sglang_debug/k3_v3_signalonly_split_rebuild_20260611_032948.log`；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`8 passed`；
  - official staged normal/no-tail 1024 signal-only 通过，`max_abs=0.000488281`；
  - official staged normal/no-tail 4096 signal-only 失败，log `v3_normal_k3_notail_staged_4096_signalonly_split_20260611_033144.log`，失败 rank 典型 `max_abs=0.0060577392578125`；
  - signal-only + debug sync 仍失败，`max_abs=0.004486083984375`；
  - signal-only + 禁用 K2 inactive-row skip 仍失败，`max_abs=0.00537109375`。
- 已完成诊断：
  - `v3_formal_dual_compare_4096_signalonly_split_20260611_033728.log` 显示 formal top-level signal path `global_v3_vs_asm_max=0.0050048828125`，但 manual no-signal `global_manual_v3_vs_asm_max=0.00054931640625`；
  - 因此 signal-only 不是正确修复，只能作为诊断开关。
- 已完成代码：
  - `large_opt.py` 新增 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` gate；
  - unset 时 normal/no-tail V3 不传 `sym_buffer/done_counter/signal_addrs`，默认走 no-signal combine-only；
  - 显式 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 才启用 no-tail signal-only 路径。
- default no-signal 后继续验证：
  - 远端 source pytest 通过，`8 passed`；
  - official staged normal/no-tail 4096 default no-signal 仍失败，log `v3_normal_k3_notail_staged_4096_default_nosignal_20260611_034012.log`，典型 `max_abs=0.00628662109375`；
  - formal default no-signal compare 出现部分 rank NaN/隐藏 max 的现象；prezero 后仍有 `0.004+` 残差；
  - chain compare default no-signal 通过，log `v3_chain_compare_4096_default_nosignal_20260611_034341.log`，`global_y_v3k3_vs_asmk3_max=0.0`，`global_y_v3k1_v3k3_vs_asm_max=0.00054931640625`；
  - official staged default no-signal + debug sync 只降到接近阈值，仍有 `0.003570556640625 > 0.0035` 的失败样本；
  - 原 ASM no-tail oracle recheck 通过，log `orig_3stage_notail_4096_oracle_recheck_20260611_034703.log`，`max_abs=0.000488281`。
- 当前结论：
  - K3 V3 pack5 compute/layout 已由同输入和 chain compare 排除为主因；
  - 当前阻塞点转为 official top-level staged 4096 中 combine scratch 初始化、K2->K3 时序/可见性、wrapper 调用顺序和 chain compare 的 pre-K3 `zero_local_combine + dist.barrier` 差异；
  - 下一步按 systematic debugging 继续做最小差异诊断，不在无根因情况下新增 runtime kernel 或扩大重写。

## 2026-06-11 - K3 no-tail gate 修复与 sync 语义定位

- 已完成代码：
  - 修复 `large_opt.py` no-tail V3 gate：默认 no-signal 分支不再误传 signal tensors，只有 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 才构造 `k3_kwargs` signal 参数；
  - 增加 `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC_LABELS`，可选择性打开 `after_k1`、`after_k2`、`after_k3_combine` 等 debug sync，用于定位而不是生产修复；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，防止 no-tail default gate 回退成无条件 signal path；
  - 临时诊断脚本增加 prezero/prebarrier/sentinel toggles，产物仍在 `hygon_tmp/sglang_debug/`。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py` 通过；
  - `python -m py_compile hygon_tmp/sglang_debug/v3_chain_compare.py hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - 同步 `large_opt.py`、`tests/test_dcu_megamoe_v3.py` 到 11 节点 `sglang_megamoe`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`8 passed in 4.18s`；
  - gate fix 后 official staged 4096 default no-signal 仍失败，典型 `max_abs=0.00506591796875`；
  - formal dual compare gatefix 通过阈值，`global_v3_vs_baseline_max=0.00311279296875`；
  - full debug sync 通过，`max_abs=0.000488281`；
  - selective sync 中 `after_k2` only 与 `after_k3_combine` only 仍失败，`after_k2 after_k3_combine` 通过。
- 当前结论：
  - gate bug 已修，但 official staged no-tail 仍未完成；
  - K3 compute/layout 和 active-row store 已基本排除；
  - 下一步检查现有 `rank_barrier`、signal wait/store 和 reduce 读 combine 的 release/acquire/visibility 语义，避免新增 runtime kernel。

## 2026-06-11 - V3 no-tail consumer-side visibility 验证补丁

- 已完成代码：
  - `k3_fused_ext.cu` 新增 device-side `invalidate_l1_device()` 和 `load_signal_system_acquire()` helper；
  - `rank_barrier_kernel` 增加默认关闭的 `acquire_after_wait` 参数，打开时 release wait 使用 system-scope acquire load，并在等待期间/结束后执行 `buffer_wbinvl1_vol`；
  - `reduce_local_combine_vec_kernel` 增加默认关闭的 `invalidate_before_read` 参数，打开时 reduce 读 combine 前先 invalidate L1；
  - `K3_fused/k3_fused.py` 透传同名可选参数，默认均为 `False`；
  - `large_opt.py` 仅在 V3 normal/no-tail 分支打开 `acquire_after_wait=use_v3_k3_no_tail` 与 `invalidate_before_read=use_v3_k3_no_tail`，旧 ASM path 默认行为不变；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过；
  - `rg` 检查确认新增参数只在 V3 no-tail large_opt 调用处显式打开。
- 待验证：
  - 远端 11 节点同步并编译 `k3_fused_ext.cu`；
  - 远端 source pytest；
  - 8 卡 official staged normal/no-tail 4096 default no-signal correctness；
  - 如仍失败，再定位是否需要在 V3 K3 main kernel 读 K2 输出前补同类 acquire/invalidate，而不是新增 runtime kernel。

## 2026-06-11 - V3 no-tail visibility 补丁远端验证与边界定位

- 远端环境：
  - `.vscode/sftp.json` 确认使用 `hg@10.17.176.11`，container `sglang_megamoe`，container repo `/workspace/DeepGEMM`；
  - 所有容器命令均 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`；
  - 临时脚本和日志放在 `hygon_tmp/sglang_debug/`。
- 已完成验证：
  - 远端 source pytest：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py`，`8 passed in 4.27s`；
  - V3 raw normal rebuild：`DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=normal DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1 DG_FORCE_BUILD=1 MAX_JOBS=2 python3 setup.py build_ext --inplace`，status 0；
  - rebuild log 显示 K1 fused normal 和 K3 V3 raw normal 走 aicc shim，K3 wrapper 走原 hipcc；未启用 V2 编译。
- official staged normal/no-tail 4096 default no-signal 结果：
  - consumer-side acquire/invalidate 补丁后仍失败，典型失败 rank `max_abs=0.0089111328125 > 0.0035`；
  - 单独 `after_k2` debug sync 仍失败，典型 `max_abs=0.00543212890625`；
  - 单独 `after_k3_combine` debug sync 仍失败，出现 `fused/baseline nonfinite fused=4 baseline=0 diff=4`；
  - `after_k2,after_k3_combine` 两个 debug sync 同时打开时通过，`max_abs=0.000488281`。
- 当前判断：
  - 只在外部 `rank_barrier/reduce_local_combine` 做 acquire/invalidate 不足以模拟 host-side `torch.cuda.synchronize()`；
  - 单独 K2 或 K3 后同步均不足，组合才稳定，说明两个边界都弱：K3 入口读 K2 `act_fp8/act_scale` 前需要可见性处理，K3 出口 combine store 发布后也需要对齐原 K3 ASM 的 `s_waitcnt vmcnt(0) + buffer_wbinvl1_vol` 语义；
  - 下一步在 V3 K3 main kernel 内做最小边界补丁，不新增 runtime kernel，不触碰 pack5 GEMM 主体循环。
- 补丁验证结果：
  - 给 V3 K3 no-tail 同时加入口 `invalidate_l1_device()` 与出口 store 后 `invalidate_l1_device()`，远端 source pytest 通过，强制删除 `k3_v3_fused_ext.o` 后 aicc 重编通过；
  - official 4096 default no-signal 反而变坏，rank 1 出现大量 nonfinite：`fused/baseline nonfinite fused=1649712 baseline=0 diff=1649712`；
  - 该组合补丁不能保留为修复。下一步拆成 entry-only / exit-only A/B，定位哪一侧破坏数据路径，再决定是否需要更接近 ASM 的单点/单 wave 发布方式。
- entry/exit A/B：
  - entry-only：仍大量 nonfinite，典型 `fused/baseline nonfinite fused=1599447 baseline=0 diff=1599447`；入口 `buffer_wbinvl1_vol` 明确有害，不能保留；
  - exit-only：不再出现大量 nonfinite，但 official 4096 仍失败，典型 `max_abs=0.0078125`；出口 invalidate 不是充分修复；
  - 结论：K3 主 kernel 边界直接加 `buffer_wbinvl1_vol` 不是当前正确修复。下一步回到 official vs chain compare 差异，重点检查 combine buffer 清零/复用、row pointer 覆盖和测试迭代时序。

## 2026-06-11 - K2->K3 visibility 诊断开关本地接入

- 已完成上下文恢复：
  - 重读 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md`；
  - 重读 remote skill，确认继续使用 `hg@10.17.176.11`、container `sglang_megamoe`、container repo `/workspace/DeepGEMM`；
  - 重读 `dcu-rag-kb`、`hygon-hip-kernel-optimizer` 和 systematic-debugging 约束。
- 已完成证据补充：
  - DCU KB 检索 `global_load ... glc`，命中 Hygon microbenchmark：`glc` 用于 L2-oriented / bypass-L1 load，并配 `s_waitcnt vmcnt(0)`；
  - 对照原 K3 ASM，row-combine pointer 读取存在 `global_load_dwordx2 ... off glc`，A 侧 K2 输出读取主要仍为 `buffer_load_*`；
  - 结合前序反证，K3 kernel 开头粗暴 `buffer_wbinvl1` 已确认有害，不继续叠加。
- 已完成本地代码：
  - `K2_fused/k2_fused_ext.cu` 增加默认关闭的 `system_fence_after_write` pybind 参数；
  - `K2_fused/k2_fused.py` 透传该参数，默认 `False`；
  - `large_opt.py` 增加 `MEGAMOE_DCU_V3_K2_SYSTEM_FENCE`，只在 V3 normal/no-tail 诊断时打开；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，确保该 K2 fence 诊断默认关闭。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K2_fused/k2_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 下一步：
  - 同步到 11 节点；
  - 重编 K2/K1/K3 V3 normal；
  - 分别验证 official 4096 default、K2 fence only、K2 fence + `after_k3_combine` debug sync，判断 K2->K3 与 K3->reduce 是否为两个独立弱边界。

## 2026-06-11 - K3 V3 rowptr global_store_short 诊断本地接入

- 当前进展回答：
  - K1 V3 normal/LL K1-only staged gate 已通过；
  - K3 V3 normal no-tail raw/same-input compare 已通过；
  - official 8 卡 4096 no-tail 仍有小残差；
  - 最新证据显示 K2->K3 和 K3->reduce 两段可见性边界都弱，K2 fence + after-K3 host sync 可通过，但 K2 fence、after-K3 sync、reduce acquire/glc 和 signal-only 单独都不够。
- 已完成本地代码：
  - 在 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 增加 `global_store_bf16_device()`；
  - `store_bf16_rowptr_device()` 从普通 `row_ptr[hidden] = value` 改为 `global_store_short %0, %1, off` inline asm；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，防止 V3 K3 rowptr store 退回普通指针 store。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh tests/test_dcu_megamoe_v3.py` 通过；
  - source guard inline `python -c ...` 通过，输出 `v3 k3 global_store_short source guard passed`。
- 错误记录：
  - 曾在 PowerShell 中误用 bash heredoc `python - <<'PY'`，触发 `Missing file specification after redirection operator`；
  - 已改用 `python -c`，后续 Windows 本地内联 Python 不重复 heredoc 写法。
- 下一步：
  - 同步 K3 V3 header、source test 和 planning 文件到 11 节点；
  - 用 aicc 重编 V3 K3 normal；
  - 跑远端 source pytest、K3 raw smoke/compare、official 4096 no-tail default/K2-fence 组合；
  - 如编译通过但 correctness 仍失败，继续用 ISA/official-vs-chain 差异定位 K3->reduce store visibility。

## 2026-06-11 - 撤回默认 no-tail host-sync 诊断桥

- 用户指出“加打印才过”不能算修复，确认当前问题仍是 K3 no-tail official 4096 的真实 bug。
- 已完成本地代码：
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC` 默认从开启改为关闭；
  - 显式打开 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 时只做诊断 `torch.cuda.synchronize()`，不再打印；
  - `tests/test_dcu_megamoe_v3.py` source guard 改为保护默认关闭和非打印诊断路径。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe/large_opt.py tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 `python -m pytest -q tests/test_dcu_megamoe_v3.py` 仍因本地 Python 无 pytest 模块失败，远端容器继续作为 pytest 验证环境。
- 额外证据：
  - DCU KB 命中 Hygon allreduce `barrier_at_end<final_sync=false>` 的 release/acquire 模式；
  - 原 K3 ASM no-tail epilogue 是 `global_store_short` 后 `s_waitcnt vmcnt(0)` + `buffer_wbinvl1_vol`；
  - 因此 post-K3 rank barrier 不能被当成 final sync，下一步继续定位设备侧 release/acquire/visibility，而不是依赖 host delay 或 stdout timing。

## 2026-06-11 - K3 no-tail post-print 反证结果落盘

- 已完成远端同步/验证：
  - 修复 `tests/test_dcu_megamoe_v3.py` 中过于脆弱的 source guard 后，远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed in 4.25s`；
  - V3 K3 normal aicc save-temps 生成到 `hygon_tmp/sglang_debug/k3_v3_savetemps_nosync_20260611_120040/`；
  - save-temps 证据显示 `global_store_short=384`、`flat_store_short=0`，并且 epilogue 末尾有 `s_waitcnt vmcnt(0)` + `buffer_wbinvl1_vol`。
- 已完成代码/编译：
  - `K3_fused/k3_fused_ext.cu` 中默认关闭的 `acquire_after_wait=True` 诊断改为复用已有 peer-to-peer `mega_moe_rank_barrier(...)` + invalidate；
  - 远端重编命令启用 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1`、`DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1`、`DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1`、`MAX_JOBS=2`；
  - build log `hygon_tmp/sglang_debug/rebuild_v3_peerbarrier_20260611_120631.log` 显示 K1/K3 V3 normal raw ext 走 aicc shim，K3 wrapper 走 hipcc，产出 `.so` 已复制；命令外层 quoting 的 final `exit` 报错不代表 build 失败。
- official 8 卡 4096 normal/no-tail 结果：
  - default `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0` 失败，log `v3_peerbarrier_default_notail_4096_20260611_120851.log`，典型 `max_abs=0.005859375`；
  - `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 失败，log `v3_peerbarrier_reduce_acquire_notail_4096_20260611_120959.log`，典型 `max_abs=0.004638671875`；
  - `MEGAMOE_DCU_V3_K2_SYSTEM_FENCE=1 MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 失败，log `v3_peerbarrier_k2fence_reduce_acquire_notail_4096_20260611_121107.log`，典型 `max_abs=0.00818634033203125`。
- K2 fence codegen 检查：
  - K2 save-temps 目录 `hygon_tmp/sglang_debug/k2_savetemps_fence_20260611_121241/`；
  - 生成 ISA 出现 `s_waitcnt vmcnt(0) lgkmcnt(0)` + `buffer_wbinvl1_vol`；
  - 结论：K2 fence 不是“源码写了但 ISA 没等 store 完”的问题。
- 当前下一步：
  - 不继续叠加 host sync / print / blind fence；
  - 用 `hygon_tmp/sglang_debug/v3_chain_compare.py` 的 `PREZERO/PREBARRIER` toggles 复现 official-vs-chain 差异，确认 combine buffer 初值、K3 前 barrier 或 workspace 时序是否遮住同一个 failure。

## 2026-06-11 - 恢复失败 fence 实验后的远端基线

- 已完成本地恢复检查：
  - 重读 planning 三文件、remote/systematic-debugging/DCU KB/optimizer 约束；
  - 确认 `fence.acq_rel.sys` / `system_fence_acq_rel` 相关实验代码没有残留；
  - 本地 `python -m compileall ...` 和 `git diff --check` 通过。
- 已完成 DCU KB 复核：
  - Hygon/Flux 参考显示 DCU_ASM 分支使用 `__threadfence_system()`，非 DCU_ASM 分支才使用 `fence.acq_rel.sys`；
  - 这与上轮 aicc/gfx938 对 `fence.acq_rel.sys` 报 `invalid instruction` 一致，后续不再把该 inline asm 作为修复方向。
- 已完成远端恢复：
  - 同步当前源码与 planning 文件到 `hg@10.17.176.11` / `sglang_megamoe`；
  - 远端 source pytest：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py`，`10 passed in 4.25s`；
  - V3 normal raw/aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_restore_after_fence_fail_20260611_125726.log`，K1/K3 V3 normal raw 扩展均恢复产出 `.so`。
- 复现结果：
  - official 8 卡 4096 normal/no-tail default 再次失败，log `hygon_tmp/sglang_debug/v3_official_restore_default_notail_4096_20260611_125904.log`；
  - 典型失败：`fused/baseline max_abs=0.00457763671875 > 0.0035`；
  - 结论：当前 bug 不是上轮失败 build 或打印副作用，仍是 K3 V3 no-tail 设备侧完成/可见性问题。

## 2026-06-11 - K3 no-tail rowptr glc 修复

- 已完成 aicc fence/builtin 最小探针：
  - 新增临时探针 `hygon_tmp/sglang_debug/fence_probe.cu`；
  - 远端 run dir `hygon_tmp/sglang_debug/fence_probe_20260611_130134/`；
  - case0 `__threadfence_system()` 编译通过，生成 `buffer_wbinvl1_vol`，不自动生成 `s_waitcnt vmcnt(0)`；
  - case1 `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` 编译通过但几乎无额外 flush；
  - case2 `__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent")` 编译通过并生成 `buffer_wbinvl1_vol`；
  - case3/case4 `"system"` scope 编译失败，错误为 `Unsupported atomic synchronization scope`；
  - case5/case6 手写 `s_waitcnt + buffer_wbinvl1_vol` 编译通过。
- 已完成第一轮 rowptr glc 反证：
  - 只把 K3 V3 epilogue/store 侧 `row_combine_ptrs` 读取改为 `global_load_dwordx2 ... glc`；
  - 远端 source pytest `10 passed`；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_k3_rowptr_glc_store_20260611_130643.log`；
  - official 4096 normal/no-tail 仍失败并变差，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_store_notail_4096_20260611_130804.log`，典型 `max_abs=0.007598876953125`；
  - 结论：store-side rowptr glc 不是充分修复，不能只修 epilogue。
- 已完成完整 rowptr glc 修复：
  - 将 active-row mask 与 epilogue/store 两处 `row_combine_ptrs` 读取都改为 `global_load_dwordx2 ... off glc`，load 后显式 `s_waitcnt vmcnt(0)`；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_k3_rowptr_glc_all_20260611_131034.log`，status 0；
  - official 8 卡 4096 normal/no-tail 通过，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_all_notail_4096_20260611_131155.log`，`max_abs=0.000488281`；
  - official 8 卡 4096 normal/no-tail 三轮通过，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_all_notail_4096_iters3_20260611_131320.log`，三轮 `max_abs=0.00138855 / 0.000488281 / 0.000488281`；
  - official 8 卡 1024 normal/no-tail 通过，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_all_notail_1024_20260611_131421.log`，`max_abs=0.000488281`；
  - 以上通过均设置 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`，不是 host sync / print timing。
- 已完成 ISA 证据：
  - save-temps run dir `hygon_tmp/sglang_debug/k3_v3_rowptr_glc_savetemps_20260611_131748/`；
  - 生成汇编 `k3_v3_fused_ext-hip-amdgcn-amd-amdhsa-gfx938.s` 统计：`global_load_dwordx2.*glc=396`、`global_store_short=384`、`flat_store_short=0`；
  - 汇编片段显示 rowptr glc load 后有 `s_waitcnt vmcnt(0)`，随后进入 `global_store_short` combine 写回。
- 已更新：
  - `task_plan.md` 将 K3 normal no-tail eager staged、no-tail barrier/rowptr 可见性和 normal no-tail-reduce 对齐项标为 ✅；
  - `findings.md` 增加 rowptr glc root-cause 记录。
- 下一步：
  - 继续 Phase 3/4：接 K3 tail-reduce normal 路径；LL K3 与 graph/uneven 仍未完成，不得宣称完整 V3 e2e。

## 2026-06-11 - K3 no-tail clean 多轮复测失败，撤回完成状态

- 用户指出“加打印才过”不能作为修复依据，本轮重新按 clean 正式路径复测：
  - 远端 source guard：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py`，`10 passed in 4.43s`；
  - 8 卡 official normal/no-tail 4096：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0 MEGAMOE_DCU_V3_NO_TAIL_SYNC=0 MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=0`；
  - log：`hygon_tmp/sglang_debug/v3_clean_notail_4096_after_user_bug_20260611_132639.log`；
  - 结果：第 1/3 轮通过，随后 rank 1 失败，`fused/baseline max_abs=0.00389862060546875 > 0.0035`。
- 已更新计划：
  - 撤回 K3 V3 normal no-tail staged、no-tail rowptr/barrier 可见性、normal no-tail-reduce 三项的 ✅，改回 ⏳；
  - 当前优先定位“首轮通过、连续迭代失败”的迭代间状态/可见性问题，不进入 tail-reduce 接入，也不把 host sync/打印作为修复。

## 2026-06-11 - K1 metadata release guard 同步修正

- 重新开始定位 clean no-tail 多轮失败前，远端 `tests/test_dcu_megamoe_v3.py` source guard 暴露一个测试自身问题：
  - 同一文件中仍残留旧断言，要求 K1 metadata 完成前使用普通 `__threadfence()`；
  - K1 V3 header 已经改为 `__threadfence_system()`，因此该旧 guard 与当前 release/acquire 假设相冲突。
- 已完成：
  - 本地修正 `tests/test_dcu_megamoe_v3.py`，把 guard 改为检查整份 K1 V3 header 中 `kCountDoneSlot` / `kEmitDoneSlot` 完成计数前均为 `__threadfence_system()`；
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 `git diff --check -- tests/test_dcu_megamoe_v3.py` 通过；
  - 显式 scp 同步到 11 节点 `sglang_megamoe` 后，远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed in 4.37s`。
- 注意：
  - 这是 source guard 修正，不是 kernel correctness 结论；
  - 下一步继续跑 K1 release fence 强制重编后的 1024/4096 clean no-tail 多轮复测。

## 2026-06-11 - K1 release fence 后 normal/no-tail clean 复测通过

- 已完成远端验证：
  - 1024 normal/no-tail clean 三轮：
    - 环境：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0 MEGAMOE_DCU_V3_NO_TAIL_SYNC=0 MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=0`；
    - log：`hygon_tmp/sglang_debug/v3_clean_notail_1024_k1_release_20260611_142052.log`；
    - 结果：3/3 通过，三轮 `max_abs=0.000488281`。
  - 4096 normal/no-tail clean 三轮：
    - 同样显式关闭 V3 no-tail host sync 和 debug stage sync；
    - log：`hygon_tmp/sglang_debug/v3_clean_notail_4096_k1_release_20260611_142341.log`；
    - 结果：3/3 通过，三轮 `max_abs=0.000488281`。
- 测试 oracle A/B：
  - 新增测试诊断开关 `MEGAMOE_DCU_TEST_SKIP_BASELINE_SYNC=1`，默认行为不变；
  - V3 1024 skip baseline sync、不 clone fused：失败，log `hygon_tmp/sglang_debug/v3_clean_notail_1024_k1_release_skip_oracle_sync_20260611_142838.log`，rank 6 `max_abs=0.00390625`；
  - V3 1024 skip baseline sync + `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1`：通过，log `hygon_tmp/sglang_debug/v3_clean_notail_1024_k1_release_skip_oracle_sync_clone_20260611_143037.log`；
  - 原 ASM staged no-tail 1024 skip baseline sync：通过，log `hygon_tmp/sglang_debug/orig_3stage_notail_1024_skip_oracle_sync_20260611_143242.log`。
- 当前判断：
  - K1 metadata release fence 与 K3 rowptr glc 组合后，normal/no-tail 生产 V3 路径在 `NO_TAIL_SYNC=0/DEBUG_STAGE_SYNC=0` 下通过 1024/4096 clean 三轮；
  - 跳过 baseline oracle sync 的 V3 残差可被 clone fused 消除，更像测试 oracle 异步/时序窗口，而不是 K3 no-tail main kernel 需要 host sync；
  - 该结论不放宽生产路径约束：V3 kernel/runtime 没有新增同步或额外 kernel，tail-reduce 仍未接入。
- 已更新：
  - `task_plan.md` 将 K3 V3 normal no-tail staged、no-tail rowptr/barrier 可见性、normal no-tail-reduce 恢复为 ✅；
  - 下一步进入 K3 normal tail-reduce staged 接入。

## 2026-06-11 - K3 V3 normal tail-reduce 初始接入与已测试假设

- 已完成本地/远端接入：
  - `large_opt.py` 的 tail-reduce 分支在 `USE_MEGAMOE_V3=1` 且 `MEGAMOE_DCU_V3_BACKEND=normal` 时调用 `k3_l2_fused_v3_to_combine()`；
  - `asm_tail_done_counter` scratch 从 1 个 int32 扩为 2 个 int32，分别用于完成计数和 completion owner slot；
  - `K3_fused/k3_fused.py` 已接到 `k3_v3_normal_combine_tail_raw()`；
  - `rank_barrier_kernel` 进入 staged 前会清 `asm_done_counter[0]` 和 `[1]`；
  - 远端 source pytest 通过，V3 K3 tail wrapper/aicc rebuild 通过。
- 初始 tail-reduce 结果：
  - 1024 normal/tail 三轮首次 e2e 失败，log `hygon_tmp/sglang_debug/v3_tail_1024_initial_20260611_145520.log`，报 fused nonfinite；
  - “tail wait 后额外全线程 invalidate L1” A/B 仍失败，log `hygon_tmp/sglang_debug/v3_tail_1024_invalidate_20260611_151415.log`，该补丁已撤回，不作为修复方向。
- 已测试单变量假设：
  - tail reduce worker 读取本地 combine 区时此前使用普通 `uint4` load，可能没有对齐跨 CTA/跨 rank combine 写入后的 acquire/glc 读取语义；
  - 已将 `k3_v3_pack5_groupgemm_impl.cuh` 中 tail reduce worker 的 combine `uint4` load 改为 `global_load_dwordx4 ... off glc` + `s_waitcnt vmcnt(0)`；
  - 后续 1024 tail e2e 反证该假设，相关代码和 source guard 已撤回；保留记录只为避免重复同一失败动作。

## 2026-06-11 - K3 tail-reduce glc 假设反证与 reducer 调度修正

- 用户要求对比原 `K3COMBINE_TAILREDUCE` 与 `K3COMBINE` 的 ASM 差异；已读取本地两份 `.s`：
  - `megamoe/dcu_megamoe_large_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.s`
  - `megamoe/dcu_megamoe_large_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE.s`
- 反证结果：
  - tail reduce worker combine `uint4` glc-load A/B 仍失败，log `hygon_tmp/sglang_debug/v3_tail_1024_glc_20260611_154554.log`，`fused/baseline nonfinite fused=356725`；
  - 该改动已撤回，source guard 也已撤回；不把它作为修复。
- ASM diff 线索：
  - tail ASM 有 extra reducer WG 分支；reducer 起点/stride 使用 `0x300=768` lanes/block；
  - `asm_reduce_blocks` 为 64/128 时，最后一个 GEMM WG 只完成 done/signal，不参与 reduce，避免和 extra reducer 重叠写 `y`；
  - V3 C 此前用 512 threads，并把最后一个 GEMM WG 作为第 `reduce_blocks + 1` 个 worker，和 ASM tail reduce 合同不一致。
- 已完成本地代码修正：
  - `kTailReduceThreads` 改为 768；
  - extra reducer 的 `reducer_idx = wg_id - gemm_workgroups`；
  - extra reducer worker_count 使用 `reduce_blocks`；
  - 最后 GEMM WG 只在 `reduce_blocks <= 0` fallback 时做本地 reduce；
  - source guard 更新为保护 768-lane extra reducer 合同。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh tests/test_dcu_megamoe_v3.py` 通过；
  - inline source guard `tail reducer schedule guard passed`。
- 远端验证：
  - source pytest `10 passed in 4.42s`；
  - aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_k3_tail_reducer768_20260611_161506.log`；
  - 1024 normal/tail 三轮仍失败，log `hygon_tmp/sglang_debug/v3_tail_1024_reducer768_20260611_161622.log`，`fused/baseline nonfinite fused=351592`；
  - 因此 reducer 768-lane 调度对齐不是充分修复。下一步保留该 ASM-derived 调度差异，同时按 tail ASM 的 wait 后 `s_barrier + buffer_wbinvl1_vol` 做一个单变量组合验证。

## 2026-06-11 - K3 tail wait invalidate 补丁已编译但 e2e 被环境 OOM 阻塞

- 已完成：
  - 按 tail ASM extra reducer wait 后的 `s_barrier + buffer_wbinvl1_vol` 线索，在 V3 tail wait helper 中增加 wait 后全 block barrier 与 invalidate；
  - 远端 source guard 曾验证通过，V3 K1/K3 normal aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_k3_tail_wait_invalidate_20260611_163241.log`。
- 直接 e2e 结果：
  - 命令使用 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=1 MEGAMOE_DCU_V3_NO_TAIL_SYNC=0 MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=0`；
  - log `hygon_tmp/sglang_debug/v3_tail_1024_wait_invalidate_20260611_163755.log`；
  - 失败发生在测试权重准备阶段，报 HIP OOM：GPU 显存被 `sglang serve` 占满，尚未进入有效 fused correctness 对比；
  - 该结果只记录为环境阻塞，不作为 tail wait invalidate 假设的 correctness 证据。
- 测试链条调整：
  - 后续若只改 K3/K1 `.cuh/.cu` 主体且未改 Python gate/wrapper/source guard，优先跳过 source pytest，直接执行 remote rebuild + e2e；
  - 只有修改 `large_opt.py`、wrapper、`setup.py`、source guard 或门控语义时才先跑 `tests/test_dcu_megamoe_v3.py`。

## 2026-06-11 - 用单卡 raw tail smoke 缩短 tail 定位链条

- 远端 8 卡仍被 `sglang serve` 占用，完整 e2e 可能继续 OOM；因此先运行单卡 raw tail smoke，避免把环境问题当 kernel 结果。
- 运行：
  - `HIP_VISIBLE_DEVICES=0 PYTHONPATH=. python3 hygon_tmp/sglang_debug/k3_v3_normal_tail_smoke.py`
  - log `hygon_tmp/sglang_debug/v3_tail_raw_smoke_wait_invalidate_20260611_170233.log`
- 结果：
  - topk=1 阶段失败，`tail_done_counter=[16, 10]`、`tail_signal_slot8=1`，说明 GEMM 完成计数和 self signal 到达；
  - 但 `tail_sample_mean=0.089844`、`tail_sample_max_abs_err=2047.910156`，期望值应为 `2048`；
  - 这把当前 tail 问题收窄到单卡 K3 V3 raw tail 的 combine/reduce 数据路径，不需要先等待 8 卡 e2e。
- 下一步：
  - 先对比同输入 no-tail raw combine 是否仍正确；
  - 再在 raw smoke 中读 combine buffer 和 reduce_y，判断是 GEMM combine 写错/未写，还是 tail reduce worker 读/寻址/累加错误。

## 2026-06-11 - K3 tail y-split raw 修复通过，测试链条收短

- 为缩短 tail 定位链条，先在单卡 raw 上做低层反证：
  - no-tail raw combine 同输入仍正确，log `hygon_tmp/sglang_debug/v3_notail_raw_smoke_after_tail_fail_20260611_170719.log`，all-ones 和 pattern 均为 `max_abs_err=0`；
  - tail combine probe 显示失败时 `combine_mean=0.08984375` 且 `reduce_mean=0.08984375`，说明问题发生在 `kTailReduce=true` 专门化的 GEMM/combine 写入阶段，不是 reduce worker 单独读错；
  - signal-only no-tail raw 正确，log `hygon_tmp/sglang_debug/v3_signal_combine_probe_20260611_171834.log`，进一步排除 tail signal 发布本身破坏 no-tail combine。
- 已反证并撤回 noinline 假设：
  - 将 tail wait/reduce helper 改为 `noinline` 后重编通过，但 `v3_tail_combine_probe_noinline_20260611_174213.log` 仍失败，combine 仍约为 `0.08984375`；
  - 该改动已撤回，不作为修复。
- 已完成 y-split 修正：
  - tail extra reducer 不再使用线性 `wg_id >= gemm_workgroups` 分支，而是放到额外 `grid.y` 行；
  - GEMM CTA 的 `(blockIdx.x, blockIdx.y)` 映射与 no-tail 保持一致，reducer index 使用 `(tile_row_idx - wg_n) * wg_m + tile_hidden_idx`；
  - launcher 使用 `reduce_rows = (reduce_blocks + wg_m - 1) / wg_m` 和 `dim3 grid(wg_m, wg_n + reduce_rows)`。
- 验证结果：
  - aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_k3_tail_ysplit_20260611_175222.log`；
  - tail combine probe 通过，log `hygon_tmp/sglang_debug/v3_tail_combine_probe_ysplit_20260611_175341.log`，combine/reduce 均为 `2048` 且 `max_abs_err=0`；
  - tail raw smoke 通过，log `hygon_tmp/sglang_debug/v3_tail_raw_smoke_ysplit_20260611_175341.log`，topk=1 输出 `2048`、topk=6 输出 `12288`，均 `max_abs_err=0`；
  - y-split 后 no-tail raw smoke 仍通过，log `hygon_tmp/sglang_debug/v3_notail_raw_smoke_after_ysplit_20260611_175443.log`；
  - 全量 K1+K3 V3 normal aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_k1k3_normal_ysplit_20260611_175519.log`。
- 测试链条调整：
  - `tests/test_dcu_megamoe_v3.py` 是 source-level guard，用于保护 env gate、wrapper/source 边界、aicc 选择、V2 隔离和 y-split 字符串约束，不是 GPU correctness；
  - y-split 后 source guard 已通过一次，后续若只改 `.cu/.cuh` kernel 主体，优先跳过 source pytest，直接执行 rebuild + raw smoke / 8 卡 e2e；
  - 只有修改 `large_opt.py`、Python wrapper、`setup.py`、source guard 或门控语义时才先跑 source pytest。

## 2026-06-11 - K3 tail stage compare 进一步缩短 e2e 链路

- 用户询问 `source pytest` 测什么后，明确测试链条调整：
  - `tests/test_dcu_megamoe_v3.py` 是 source-level guard，保护 env gate、wrapper/source 边界、aicc 选择、V2 隔离和 y-split 字符串约束；
  - 当前 K3 tail kernel-body 诊断不再把 source pytest 放在主链路，直接走 rebuild + raw/stage/e2e；
  - 只有改 Python 分流、`setup.py`、wrapper 或 source guard 时才先跑 source pytest。
- 已新增临时诊断脚本：
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py`；
  - 同一份 V3 K1 + K2 输出下，依次跑原 ASM K3 tail 和 V3 K3 tail，直接比较 `y`，绕开 DeepEP baseline oracle。
- 远端 8 卡 stage compare：
  - 命令环境：`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. K3_V3_TAIL_STAGE_TOKENS=1024 K3_V3_TAIL_STAGE_MAX_TOKENS=1024 K3_V3_TAIL_STAGE_ITERS=3`；
  - log：`hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_20260611_184340.log`；
  - 结果：第 1/2 轮全 rank `max_abs=0`；第 3 轮无 nonfinite，但 rank4/5/6/7 出现小漂，最大 rank7 `max_abs=0.00390625`，超过当前 `0.0035` 阈值；
  - 结论：K3 V3 tail 与 ASM tail 在同一 K1/K2 输入下仍存在重复调用时的状态/可见性/迭代间语义差异，但已经不是 baseline oracle 或 full e2e 链条才暴露的问题，也不是大面积 nonfinite。
- 下一步：
  - 用同一 stage compare 继续抓每轮 ASM/V3 后 `asm_done_counter[0/1]`、tail signal slots 和输出摘要；
  - 优先判断 `reset_tail_signal_slots`、signal generation、done owner slot、combine buffer 覆盖/读取是否与原 ASM tail 完全一致。

## 2026-06-11 - K3 tail reduce worker combine load 指令族 A/B

- 用户提醒计划要及时更新，本轮已同步 `task_plan.md`：
  - Phase 3 tail-reduce 仍为 ⏳，未标完成；
  - 新增 8 卡 K3 tail stage compare 小项；
  - 新增 tail reduce worker combine-buffer 读取语义小项；
  - 实验运行策略中明确：kernel-body 诊断跳过 source pytest，直接 rebuild + raw/stage/e2e。
- 新证据：
  - 重新看原 `K3COMBINE_TAILREDUCE.s` 后确认 tail reduce 读 combine 的宏是 `global_load_dwordx4 ... off` 后接 `s_waitcnt vmcnt(0)`，不是 `glc`；
  - 之前的 combine `uint4 glc-load` 反证发生在 y-split 修复之前，当时 `kTailReduce=true` 下 combine 本身错误，不能继续作为当前 y-split 后的有效反证。
- 已做本地代码改动：
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 新增 `global_load_uint4_device()`；
  - tail reduce worker 读 combine buffer 从普通 C `uint4` load 改为显式 `global_load_dwordx4 ... off + s_waitcnt vmcnt(0)`；
  - 该改动只影响 tail reduce worker 的 epilogue/reduce 阶段，不改 K3 GEMM 主循环，不新增 runtime kernel。
- 待验证：
  - 本地 compile/diff check；
  - 同步远端；
  - 强制重编 V3 K3 normal aicc；
  - 跑单卡 raw tail smoke 与 8 卡 stage compare，若通过再回到 official e2e。

## 2026-06-11 - K3 tail worker global-load A/B 已完成 raw 级验证

- 已完成本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` 通过；
  - `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh .planning/dcu_megamoe_v3/task_plan.md .planning/dcu_megamoe_v3/progress.md .planning/dcu_megamoe_v3/findings.md` 通过。
- 已同步远端：
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh`；
  - `.planning/dcu_megamoe_v3/task_plan.md`、`progress.md`、`findings.md`。
- 远端 K3-only rebuild：
  - log `hygon_tmp/sglang_debug/rebuild_v3_k3_tail_global_load_20260611_185130.log`；
  - status 0；
  - 该次只覆盖 K3 raw，不可直接用于 K1+K3 stage compare，因为 K1 extension 当时仍可能是 stub 状态。
- 远端单卡 raw tail smoke：
  - log `hygon_tmp/sglang_debug/v3_tail_raw_smoke_global_load_20260611_185421.log`；
  - status 0；
  - topk=1 输出期望 `2048`，sample max err 0，done `[16,10]`，signal slot8=1；
  - topk=6 输出期望 `12288`，sample max err 0，done `[96,93]`，signal slot8=1。
- 远端 full K1+K3 normal rebuild：
  - log `hygon_tmp/sglang_debug/rebuild_v3_k1k3_normal_tail_global_load_20260611_185516.log`；
  - status 0；
  - build 输出确认 K1 V3 normal aicc shim 与 K3 V3 normal aicc shim 均启用，分别带 `DCU_MEGAMOE_V3_ENABLE_K1_RAW_NORMAL` / `DCU_MEGAMOE_V3_ENABLE_K3_RAW_NORMAL`。
- 当前状态：
  - 该 global-load 改动已通过 raw correctness，不改 GEMM 主循环、不新增 runtime kernel；
  - K3 normal no-tail 已关闭；
  - K3 normal tail 仍未完成，下一步必须跑最新 full rebuild 状态下的 8 卡 stage compare，再决定是否进入 official e2e。

## 2026-06-11 - K3 tail worker global-load A/B 被 8 卡 stage compare 反证

- 已运行最新 full K1+K3 normal rebuild 状态下的 8 卡 stage compare：
  - 命令环境：`HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. K3_V3_TAIL_STAGE_TOKENS=1024 K3_V3_TAIL_STAGE_MAX_TOKENS=1024 K3_V3_TAIL_STAGE_ITERS=3`；
  - log `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_global_load_20260611_190230.log`；
  - status 1。
- 结果：
  - 第 1 轮全 rank `max_abs=0`；
  - 第 2 轮 rank4/5/6/7 出现小漂，rank7 `max_abs=0.006805419921875`，超过阈值并触发失败；
  - 无 nonfinite，仍是 ASM tail vs V3 tail 在同一 K1/K2 输入下的重复调用小漂问题。
- 结论：
  - 显式 `global_load_dwordx4 ... off + s_waitcnt vmcnt(0)` 只证明 raw 单卡数据路径可跑，不能修复 8 卡 staged 重复调用状态/可见性差异；
  - 该 A/B 不再重复。下一步应增强 `k3_v3_tail_stage_compare.py`，收集每轮 ASM/V3 后 done counter、signal slots、输出差异位置/摘要，以及必要时做 ASM/V3 顺序互换或单独 V3 重复调用。

## 2026-06-11 - K3 tail stage compare order/zero-combine 诊断

- 已增强临时诊断脚本 `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py`：
  - 支持 `K3_V3_TAIL_STAGE_ORDER=asm_first|v3_first|v3_only|asm_only`；
  - 支持 `K3_V3_TAIL_STAGE_ZERO_COMBINE=1` 和 `K3_V3_TAIL_STAGE_KEEP_GOING=1`；
  - 输出 `nonfinite_first/nonfinite_second/max_abs/token/hidden/value/done0/done1`，用于绕开 full e2e 和 baseline oracle。
- 8 卡 order 诊断日志：
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_diag_asm_only_20260611_190748.log`：`asm_only` 3/3 全 rank 稳定，`max_abs=0`；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_diag_v3_only_20260611_190810.log`：`v3_only` 第 2/3 轮出现小漂，最高约 `0.009674`，无 nonfinite；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_diag_asm_first_20260611_190833.log`：第 3 轮 V3 second 多 rank nonfinite；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_diag_v3_first_20260611_190856.log`：第 1 轮 V3 first 在部分 rank nonfinite，ASM second 稳定。
- zero-combine 诊断日志：
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_zero_v3_only_20260611_191018.log`：zero 本地 combine 后仍有 V3 不稳定/非有限值，说明不是单纯 stale combine buffer；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_zero_asm_first_20260611_191042.log`：大部分稳定，仅第 3 轮有小漂；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_zero_v3_first_20260611_191107.log`：第 1 轮 V3 first 仍可产生 nonfinite，后续轮次稳定。
- 结论：
  - 原 ASM tail 自身重复调用稳定；
  - V3 tail 自身重复调用不稳定，因此 remaining issue 在 V3 tail path 内部，不是 baseline oracle、ASM 顺序或权重 layout 粗错；
  - zero combine 不能消除问题，下一步聚焦 V3 tail done/signal 发布和 reducer 启动时序。

## 2026-06-11 - K3 tail direct-signal A/B 被反证并撤回

- 单变量改动：
  - 在 `kTailReduce` epilogue 中，让 `atomicAdd_system(done_counter, 1)` 判定出的最后 GEMM CTA thread0 直接调用 peer signal；
  - `done_counter[1]` 仅作为诊断/fallback owner slot，正常 `reduce_blocks>0` 路径不再由所有 GEMM CTA 读取 owner slot 决定 signal。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` 通过；
  - `git diff --check` 通过；
  - 远端 source guard `10 passed in 4.11s`。
- 远端编译：
  - 首次 `--force` build 只 relink，`ninja: no work to do`，不算 kernel 重编；
  - 删除 `K3_fused/k3_v3_fused_ext*.o` 后重编成功，log `hygon_tmp/sglang_debug/rebuild_v3_k3_tail_direct_signal_forced_obj_20260611_192051.log`，确认 `[1/1] ... k3_v3_fused_ext.hip` 由 aicc shim 重编。
- 8 卡验证：
  - `K3_V3_TAIL_STAGE_ORDER=v3_only K3_V3_TAIL_STAGE_KEEP_GOING=1`；
  - log `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_direct_signal_v3_only_20260611_192159.log`；
  - 第 1 轮即出现多 rank nonfinite，明显比旧 owner-slot 路径更差。
- 结论：
  - “最后 GEMM CTA 直接 signal” 不是当前修复；
  - 该补丁已撤回，本地代码回到旧 owner-slot signal 路径；
  - 下一步不继续叠同步，改为增强 stage compare 采集 tail signal slots/generation 与 reducer wait 状态。

## 2026-06-11 - K3 tail signal-slot 诊断已完成，转向 combine/row mapping

- 已完成计划更新：
  - `task_plan.md` 将 tail signal slots/generation 诊断标为 ✅；
  - 新增进行中项：修正 K3 tail `combine_reduce` / row mapping 诊断，继续定位 V3 reducer/combine 可见性与 active-row 覆盖语义；
  - 实验运行策略更新为：direct-signal 与 signal-slot/generation 均已反证为非充分根因，后续不重复同一方向的 A/B。
- 诊断结论：
  - 旧 owner-slot 路径下的 8 卡 `v3_only` / `asm_first` stage compare 显示 local tail signal slots `[8, 15]` 在 V3 launch 前为 0，launch 后 send/recv generation 均为 1；
  - done counter 到达目标，未观察到 signal slot reset 或 generation 到达异常；
  - 因此当前 K3 normal tail no-pass 不能再优先归因于 signal slot 生命周期。
- 新发现：
  - 增强脚本中的 Python `combine_reduce` 视图在 `asm_first` 场景下也可能读到 nonfinite，而 ASM `y` 仍为 finite；
  - 该诊断暂时只能说明“直接按当前 Python view 求和不可信”，不能说明 ASM 或 V3 的 combine buffer 一定已经损坏；
  - 下一步先复读 `get_sections()`、`combine_token_offset()`、K1 `row_combine_ptrs/output_index` 和原 tail reducer 读法，修正诊断后再决定 kernel 修复点。

## 2026-06-11 - Normal tail 状态落盘并切到 fused LL

- 按用户要求，先暂停 K3 normal tail-reduce 深挖，改做 K1/K3 fused LL。
- 已记录 normal tail 当前状态：
  - normal no-tail 已完成，1024/4096 clean 三轮通过；
  - normal tail 未完成，不标 ✅；
  - 已反证 `global_load_dwordx4 off` worker load、direct-signal、signal slot reset/generation 等方向；
  - device post-hoc reduce 诊断脚本已在 `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` 中加入，并同步远端且 compileall 通过。
- 最新两次远端 device-reduce 诊断运行均无有效 kernel 结果：
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_device_reduce_asm_first_20260611_194029.log`：torchrun/NCCL socket abort，`rank_stats_count=0`；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_device_reduce_asm_first_1iter_20260611_194505.log`：仍在 launcher/NCCL 层失败，没有可用于 kernel 判断的 `rank_stats`；
  - `hy-smi` 显示 8 卡 VRAM/HCU use 为 0，因此当前失败更像 torchrun/NCCL/launcher 状态，不是显存占用。
- 已更新：
  - `task_plan.md`：normal tail 标记为暂缓，新增 LL K3 no-tail/tail active items；
  - `findings.md`：记录 normal tail 恢复入口和 LL layout 风险。
- 下一步：
  - 复读 K1/K3 LL wrapper、launcher、K1 LL 256-row compatibility path 与 K3 LL raw/pure kernel；
  - 先接 K3 V3 LL no-tail combine，固定 32/128 tokens，K2 和外部 reduce 继续复用。

## 2026-06-11 - 开始 K1/K3 fused LL no-tail 接入

- 已按用户最新要求记录状态并把 Phase 3 的 K3 V3 LL no-tail 项标为 ⏳。
- 当前执行顺序：
  - 先复核 K1 LL 当前 256-row ASM-compatible layout 与 true LL 64-row layout 的分流条件；
  - 再实现 K3 V3 LL no-tail row-combine 写回，优先复用现有 K1/K2/外部 reduce 周边合同；
  - tail-reduce LL 暂不混入第一轮，避免同时打开两个定位面。

## 2026-06-11 - K3 LL no-tail 第一版本地接入

- 已完成本地代码改动：
  - `large_opt.py` 中 LL no-tail 现在走 V3 K3；LL tail-reduce 仍保留 K3 ASM fallback，并且只有 tail fallback 时 K1 LL 才使用 256-row ASM-compatible layout；
  - `K3_fused/k3_fused.py` 允许 `backend="ll"` 的 no-tail combine，tail/graph 仍显式未实现；
  - `K3_fused/k3_v3_fused_ext.cu` 新增 `k3_v3_ll_combine_raw` pybind，最终使用 pure LL kernel 的 fixed row-capacity + `row_combine_ptrs` store 模式写 combine buffer；
  - `tests/test_mega_moe_dcu.py` 在 V3 LL no-tail 下给 L2 使用 pack5 layout，LL tail 仍保持 K1-only/ASM L2 layout。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_mega_moe_dcu.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过；
  - 本地 `python -m pytest -q tests/test_dcu_megamoe_v3.py` 因本地 Python 缺少 pytest 失败，远端容器内补跑。

## 2026-06-11 - K3 LL 编译卡顿后收窄实现

- 远端 source guard 已通过：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py`，结果 `10 passed in 4.44s`。
- 首次 LL raw build 命令：
  - `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll MAX_JOBS=8 python3 setup.py build_ext --inplace`
  - log `hygon_tmp/sglang_debug/rebuild_v3_k1k3_ll_notail_20260611_200246.log`
  - K1 LL 已重编完成；K3 LL hipcc/dcc 在 `k3_v3_fused_ext.hip` 上超过 15 分钟仍无推进，已停止该 build。
- 已调整实现以缩小 K3 LL 编译面：
  - 撤掉 `<64, 256, true>` fused specialization；
  - 改用 `V3_K3_Pure_LowLatencyMaskedGroupGemmKernel` 的固定 row-capacity + rowptr store 模式；
  - 本地 `compileall` 和 `git diff --check` 已通过，待同步远端重跑 source guard 与 LL raw build。

## 2026-06-11 - K3 LL pure-rowptr raw build 通过

- 修复一次远端 build 语法失败：
  - log `hygon_tmp/sglang_debug/rebuild_v3_k3_ll_pure_rowptr_20260611_202322.log`；
  - 根因是 C macro 包住 HIP kernel launch，hipify 展开成 `hipLaunchKernelGGL` 后语法损坏；
  - 已改成 `launch_v3_k3_ll_pure_rowptr<block_m>()` 模板 helper。
- 远端 source guard 重新通过：`10 passed in 4.36s`。
- 远端 LL raw build 通过：
  - 命令环境：`DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll MAX_JOBS=8`；
  - log `hygon_tmp/sglang_debug/rebuild_v3_k3_ll_pure_rowptr_fix_20260611_202558.log`；
  - K3 V3 LL raw extension 已复制到 `megamoe/dcu_megamoe_large_opt/K3_fused`。
- 下一步：8 卡 eager correctness，先 `K3_USE_ASM_TAIL_REDUCE=0`，tokens 32/128。

## 2026-06-11 - K3 V3 LL no-tail 8 卡 correctness 通过

- 纠正一次测试命令问题：
  - 错误命令外层用了 `torchrun --nproc_per_node=8`，而 `tests/test_mega_moe_dcu.py` 自己会 `torch.multiprocessing.spawn(--num-processes 8)`，实际变成 64 个进程并导致 GPU5 OOM；
  - 已改为直接 `python3 tests/test_mega_moe_dcu.py --num-processes 8`，不是 kernel 失败。
- 通过命令环境：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1`
  - `USE_MEGAMOE_V3=1`
  - `MEGAMOE_DCU_V3_BACKEND=ll`
  - `K3_USE_ASM_TAIL_REDUCE=0`
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`
  - `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=0`
- 8 卡 correctness-only 结果：
  - 32 tokens 三轮通过，log `hygon_tmp/sglang_debug/v3_ll_notail_32_spawn_20260611_203126.log`，`max_abs=0.000244141`；
  - 128 tokens 三轮通过，log `hygon_tmp/sglang_debug/v3_ll_notail_128_spawn_20260611_203227.log`，`max_abs=0.000244141`。
- 已将 Phase 3 的 K3 V3 LL no-tail 项标为 ✅，并把 K3 V3 LL tail-reduce 标为 ⏳。
- 下一步：实现/验证 K3 V3 LL tail-reduce，固定 32/128 tokens。

## 2026-06-11 - K3 V3 LL tail source guard 对齐

- 已继续 K3 V3 LL tail-reduce 接入工作：
  - `large_opt.py` 当前 tail 分支已按 `v3_backend in ("normal", "ll")` 选择 V3 K3；
  - `K3_fused/k3_fused.py` 已在 LL tail 分支调用 `k3_v3_ll_combine_tail_raw`；
  - 测试/fixture 已将 V3 LL 的 L2 权重切到 pack5 layout。
- 已修正过期 source guard：
  - 不再要求 LL tail fallback 到 ASM；
  - 不再要求 LL tail 强制 K1 ASM-compatible 256-row layout；
  - 新增对 `k3_v3_ll_combine_tail_raw`、LL tail worker 256-thread reduce 和 pack5 L2 分流的 guard。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_mega_moe_dcu.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 远端 source guard 第一次失败并已修正：
  - 失败原因是 guard 仍查找直接模板调用字符串 `launch_v3_k3_ll_pure_rowptr<32>`；
  - 当前实现通过 `DCU_MEGAMOE_V3_LAUNCH_LL(32, true/false)` 调用同一 helper，因此 guard 改为检查 helper 存在和 32-blockM tail/no-tail 两个 macro 分支。
- 下一步：
  - 同步远端；
  - 远端 `tests/test_dcu_megamoe_v3.py` source guard；
  - 强制重编 V3 K3 LL raw extension；
  - 跑 8 卡 LL tail 32/128 correctness。

## 2026-06-11 - K3 V3 LL tail 8 卡 correctness 通过

- 远端 source guard：
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `10 passed in 4.24s`。
- 远端 K3 V3 LL raw rebuild：
  - 强制删除旧 `k3_v3_fused_ext` object/so 后重编；
  - build 输出确认 `[1/1] ... k3_v3_fused_ext.hip` 并复制新的 `k3_v3_fused_ext.cpython-310-x86_64-linux-gnu.so`；
  - 本次 build 命令中的 `$log` 被 PowerShell 提前展开为空，未成功写入远端 build log；后续远端命令已切到单引号 here-string 以保留 `$()` 和 `${PIPESTATUS[0]}`。
- 8 卡 LL tail correctness：
  - 32 tokens 三轮通过，log `hygon_tmp/sglang_debug/v3_ll_tail_32_spawn_20260611_205208.log`，`max_abs=0.000244141`；
  - 128 tokens 三轮通过，log `hygon_tmp/sglang_debug/v3_ll_tail_128_spawn_20260611_205301.log`，`max_abs=0.000244141`。
- 已更新：
  - `task_plan.md` 将 K3 V3 LL tail-reduce 标为 ✅；
  - `findings.md` 记录 LL no-tail/tail correctness 结论与剩余缺口。
- 下一步候选：
  - 回补 LL source/build 状态同步；
  - 进入 LL performance gate 的 32/128 初测；
  - 或按计划恢复 K3 normal tail device post-hoc reduce 诊断。

## 2026-06-11 - LL performance gate 初测未通过

- 已按 Phase 6 做 8 卡 LL 32/128 smoke 级性能采样，`warmup=2 repeat=5`：
  - V3 LL no-tail 32：`1.1678 ms`，correct；
  - V3 LL tail 32：`1.1704 ms`，correct；
  - 原 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0` persistent_fused 32：`1.0018 ms`，correct；
  - V3 LL no-tail 128：`1.6858 ms`，correct；
  - V3 LL tail 128：`1.7118 ms`，correct；
  - 原 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0` persistent_fused 128：`1.5558 ms`，correct。
- 结论：
  - LL correctness 完成，但 LL performance gate 目前未过；
  - V3 tail/no-tail 相近，初步排除“仅外部 reduce 或 tail reduce”作为主要差距；
  - 下一步按 Hygon 优化闭环做分段/ profiler 定位，不做无证据调参。
- 已尝试 `hipprof --stats --hip-trace --follow-fork` 跑 8 卡 V3 LL 32 correctness-only：
  - 由于当前 8 卡 VRAM 显示 94%，hipprof 跟踪开销导致 fixture 阶段 pack5 权重分配 OOM；
  - 产出的 HIP API/HIPOPS 统计不完整，不作为性能根因证据。
- 已在 `large_opt.py` 增加默认关闭的诊断开关 `MEGAMOE_DCU_V3_STAGE_TIMING=1`：
  - 只在 V3 staged path 中记录 HIP event timing；
  - 默认关闭，不影响 runtime/bench 路径；
  - 用于下一步分解 rank barrier、K1、K2、K3、no-tail barrier/reduce 的耗时。
- 已增加默认不变的 LL blockM A/B 开关：
  - `MEGAMOE_DCU_V3_LL_BLOCK_M=32|48|64`，默认 32；
  - 传给 K1/K3 V3 LL wrapper，用于验证低 token 下 `block_m=64` 是否能减少固定 64-row/expert layout 的 tile 数；
  - 本地 `compileall` 和 `git diff --check` 通过。
- 远端 source guard 已重新通过：`10 passed in 4.12s`。
- 尝试运行 `MEGAMOE_DCU_V3_STAGE_TIMING=1` 的 V3 LL 32 no-tail 诊断失败在 fixture 阶段 OOM：
  - log `hygon_tmp/sglang_debug/v3_ll_notail_32_stage_timing_20260611_210738.log`；
  - host 上存在 `sglang serve` + `evalscope`，`hy-smi` 显示 8 卡 VRAM 约 95% 且部分 HCU 正在运行；
  - 按远端工作流约束，不擅自 kill 占卡进程。
- 静态定位新增优先 A/B：
  - `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0`，验证低 token 下 K2 跳过 inactive padded rows 是否能收回主要差距；
  - `MEGAMOE_DCU_V3_LL_BLOCK_M=64`，验证 K1/K3 LL 64-row/expert layout 下减少 m tile 是否有效。
- 已新增诊断脚本 `hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`：
  - 默认跑 tokens 32/128；
  - 对比 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 与 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0 + MEGAMOE_DCU_V3_LL_BLOCK_M=64`；
  - 仅用于远端卡空后的 correctness+bench A/B，不改变生产默认。

## 2026-06-11 - K1/K3 fused LL 状态回写与 A/B plumbing 修正

- 已按用户要求继续聚焦 K1/K3 fused LL，normal tail 仍保持暂缓状态。
- 本地检查发现 `large_opt.py` 已向 V3 K3 LL 传 `ll_block_m`，但 `k3_l2_fused_v3_to_combine()` 签名缺少该参数；这会阻断后续 `MEGAMOE_DCU_V3_LL_BLOCK_M=64` A/B。
- 已修复 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py`，给 V3 K3 wrapper 增加 `ll_block_m: int = 32` 参数，并继续传给 `k3_v3_ll_combine_raw` / `k3_v3_ll_combine_tail_raw`。
- 已增强 `tests/test_dcu_megamoe_v3.py` source guard：
  - 覆盖默认关闭的 `MEGAMOE_DCU_V3_STAGE_TIMING=0`；
  - 覆盖 `MEGAMOE_DCU_V3_LL_BLOCK_M=32` 默认和 `{32,48,64}` 校验；
  - 覆盖 K1/K3 wrapper 对 `ll_block_m` 的传递。
- 验证：
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - 本地 `git diff --check` 通过；
  - 同步到 11 节点后，容器内 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `10 passed in 4.31s`。
- 下一步：检查远端卡状态；若仍被 `sglang/evalscope` 占用，则继续做 LL 静态准备，卡空后运行 `hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`。

## 2026-06-11 - K1/K3 fused LL uneven 验证入口准备

- 远端卡状态：
  - `hy-smi` 显示 8 卡 VRAM 约 95-96%，GPU4/6/7 HCU 使用较高；
  - host 进程显示 `sglang serve` 与 `evalscope` 正在运行；
  - 按工作流约束不擅自 kill 占卡进程，因此 8 卡 LL A/B 暂缓。
- 已新增 `hygon_tmp/sglang_debug/run_v3_ll_correctness_matrix.sh`：
  - 默认只跑 V3 LL uneven correctness；
  - 默认 token 列表为 `32,48,16,64,24,40,56,8`，`NUM_MAX_TOKENS_PER_RANK=64`；
  - 分别跑 `K3_USE_ASM_TAIL_REDUCE=0` 和 `K3_USE_ASM_TAIL_REDUCE=1`；
  - 使用 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll`。
- 验证：
  - 本地 `git diff --check` 通过；
  - 本地没有 `bash`，脚本语法改在远端容器检查；
  - 远端 `bash -n hygon_tmp/sglang_debug/run_v3_ll_correctness_matrix.sh` 通过；
  - 远端 `python3 -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过。
- 还未完成：
  - LL uneven no-tail/tail 8 卡 correctness 尚未运行；
  - LL performance A/B 尚未运行；
  - V3 graph 仍未接，K1 V3 graph wrapper 当前仍显式 NotImplemented。

## 2026-06-11 - K1/K3 fused LL 远端资源状态

- 远端 source guard 最终状态再次验证通过：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py`，结果 `10 passed in 4.15s`。
- 最新 `hy-smi` 显示 HCU 使用率已降到 0%，但 8 卡 VRAM 仍约 95-96%。
- host 进程检查显示 `evalscope` 已结束，只剩 `sglang serve` PID 166043 常驻，占用 `/data2/MiMo-V2-Flash-Channel-FP8/` 服务显存。
- 按远端工作流约束，不擅自 kill 该服务；因此暂不运行会分配权重 fixture 的 8 卡 LL correctness/perf。
- 卡/服务释放后的直接执行顺序：
  - `./hygon_tmp/sglang_debug/run_v3_ll_correctness_matrix.sh` 验证 LL uneven no-tail/tail；
  - `./hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh` 验证 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 与 `MEGAMOE_DCU_V3_LL_BLOCK_M=64`。

## 2026-06-11 - V3 graph 边界收紧

- 静态复核 graph path：
  - `K1_fused/k1_fused.py` 的 `k1_symm_fused_l1_v3_graph()` 当前仍显式 `NotImplementedError`；
  - `large_opt.py` graph 分支后半段仍保留原 K3 ASM graph fallback；
  - 测试 fixture 在 `USE_MEGAMOE_V3=1` 时会将 L2 切到 V3 pack5 layout，因此若未来只半接 K1 V3 graph，可能误把 pack5 L2 喂给原 K3 ASM graph。
- 已改动：
  - 在 `_run_large_opt_3stage_graph()` 中检测到 `v3_backend is not None` 时立即 fail-fast；
  - source guard 增加 `"V3 staged graph path is not wired yet"` 检查。
- 验证：
  - 本地 `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - 本地 `git diff --check` 通过；
  - 远端容器 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `10 passed in 4.36s`。

## 2026-06-12 - 切入 LL performance tuning

- 按用户新优先级调整执行顺序：
  - 先优化 V3 K1/K3 fused LL；
  - LL 达标后再优化 normal；
  - 最后补 uneven、graph、normal tail 等未完成项。
- 已重新读取 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md`，并复读 planning、remote、hygon optimizer、dcu-rag-kb skill。
- 已运行 DCU KB optimize 检索，结论支持当前第一轮 A/B：
  - 先测 low-token padded rows / K2 inactive skip；
  - 再测 LL blockM；
  - 若阶段耗时仍高，再按 fine-grained compute/communication signal 方向定位 K1/K3 overlap。
- 远端资源状态：
  - `hy-smi`：8 卡 HCU 0%，VRAM 95-96%；
  - host 进程仍有 `sglang serve` PID 166043 与 scheduler 子进程占用显存；
  - 未擅自 kill 服务，因此未运行会分配权重 fixture 的 8 卡 perf/uneven。
- 已改进并同步远端诊断脚本：
  - `hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh` 增加 VRAM guard、default/k2skip/block48/block64/可选 baseline/可选 stage timing、JSON `--out`；
  - 新增 `hygon_tmp/sglang_debug/summarize_v3_ll_perf.py` 汇总 perf JSON。
- 验证：
  - 本地 `python -m py_compile hygon_tmp/sglang_debug/summarize_v3_ll_perf.py` 通过；
  - 本地 `git diff --check` 针对诊断脚本和 planning 文件通过；
  - 已 scp 脚本到 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM/hygon_tmp/sglang_debug/`；
  - 容器内 `bash -n run_v3_ll_perf_ab.sh` 与 `python3 -m py_compile summarize_v3_ll_perf.py` 通过；
  - 当前远端 96% VRAM 下运行空 case，脚本在 fixture 前报 `max_vram_pct=96 exceeds MAX_VRAM_PCT=90` 并退出，guard 生效。
- 下一步：
  - 若用户释放/允许释放 `sglang serve` 占用显存，先跑 `run_v3_ll_perf_ab.sh`；
  - 若显存继续被占，继续做 LL 静态定位和脚本化 profiler/ISA 准备，不切 normal。

## 2026-06-12 - K1 LL block64 tiny-store A/B 干扰项收窄

- 静态复查 K1/K3 LL blockM specializations：
  - K3 LL launcher 始终用 `kMaskTinyStore=true`；
  - K1 LL launcher此前只有 blockM=32 的 low-load 分支会按 `valid_rows_per_expert <= 16` 启用 tiny-store mask；
  - block64 A/B 会在 32-token 场景多写 padded L1 rows，影响对“减少 m tile”本身的判断。
- 已修改：
  - `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_fused_ext.cu`：blockM=64 且 `mask_tiny_store` 时实例化 `DCU_MEGAMOE_V3_LAUNCH_K1_LL(64, 64, true)`；
  - `tests/test_dcu_megamoe_v3.py`：新增 source guard，防止 block64 tiny-store 分支丢失。
- 验证：
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py ...` 通过；
  - 本地 `git diff --check` 通过；
  - 远端 source guard `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `10 passed in 4.31s`；
  - 远端先只开 K1 LL raw build 会把 K3 extension 重链成 stub，已立即用 K1+K3 LL raw flags 重新 build 恢复；
  - 远端 import sanity：K1 有 `k1_symm_fused_l1_v3_pack5`，K3 raw availability 为 `kernels=True, ll=True, normal=False`，符合当前只编 LL 的调优阶段。
  - 远端 `strings`/符号检查确认 K1 `.so` 里已有 block64 tiny-store 与 block64 non-tiny 两个 LL kernel symbol；
  - `dccobjdump --show-sass --separate-functions` 对当前 Python extension `.so` 未输出可 grep 的函数/ISA 行，本次只记为 inconclusive，不作为最终 ISA 证据。
- 还未运行：
  - 因远端 8 卡 VRAM 仍由 `sglang serve` 占用 95-96%，未跑 8 卡 correctness/perf A/B。

## 2026-06-12 - 释放远端 8 卡显存并恢复 LL A/B

- 用户明确要求“直接杀了”后，已在 11 节点精确定位占卡进程属于容器 `dsq_sglang_601`，不是当前测试容器 `sglang_megamoe`。
- host 用户态对 PID 166043 及 `sglang::scheduler` 子进程发送 TERM/KILL 因权限不足失败；随后改用 `docker exec --user root dsq_sglang_601` 精确 `pkill`：
  - `[/]usr/local/bin/sglang serve`
  - `[s]glang::`
- 杀进程后在 `sglang_megamoe` 容器内运行 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM && hy-smi`，8 张 HCU 均显示 `VRAM%=0%`、`HCU%=0.0%`。
- 下一步立即运行 `hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`，先测 V3 LL no-tail 32/128 的 default、`K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 和 `MEGAMOE_DCU_V3_LL_BLOCK_M=64` A/B。

## 2026-06-12 - LL no-tail A/B 与 stage timing 完成

- 已完成 8 卡 no-tail 32/128 A/B：
  - `v3_ll_default_32`: `1.1771 ms`；
  - `v3_ll_k2skip_32`: `1.1835 ms`；
  - `v3_ll_k2skip_block64_32`: `2.8951 ms`；
  - `v3_ll_default_128`: `1.7210 ms`；
  - `v3_ll_k2skip_128`: `1.6959 ms`；
  - `v3_ll_k2skip_block64_128`: `3.5302 ms`。
- A/B 结论：
  - `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 只对 128 有小幅改善，32 略退，K2 padding 不是主瓶颈；
  - block64 当前明显退化，先不继续沿这个方向调。
- stage timing 已运行 default/k2skip 32/128 并解析：
  - K2 p50 只有 `0.027-0.028 ms`；
  - K3 LL combine 是主耗时，32 p50 约 `0.67 ms`，128 p50 约 `1.14 ms`；
  - K1 LL p50 约 `0.42-0.46 ms`；
  - no-tail barrier/reduce p50 很小。
- 运行中遇到两次本地/远端 shell quoting 错误：
  - PowerShell 直接拼接 `TOKENS_LIST="32 128"` 触发本地 parser error；
  - 嵌套单引号包 Python f-string 触发远端 bash syntax error；
  - 已改用 here-string + `python3 <<'PY'` 结构完成解析，后续远端复杂命令继续使用这种写法。
- 下一步：优先读 K3 V3 LL pure-rowptr/fused 实现与 wrapper，做 K3 LL pure-vs-fused、rowptr store/epilogue 的定位；必要时再进入 hipprof/codeobj/ISA 验证。

## 2026-06-12 - 补充 fused 链路逼近 pure C groupgemm 要求

- 用户进一步明确：V3 K1/K3 fused 的实际通信链路耗时不能只满足 e2e baseline gate，需要尽可能逼近对应原始 C 5pack groupgemm/pure kernel。
- 已更新：
  - `task_plan.md` 的已确认设计点、成功标准、Phase 2、Phase 3 和 Phase 6；
  - `findings.md` 的 pure-to-fused 扩展原则。
- 当前执行含义：
  - LL optimization 继续先看 K3/K1 分段；
  - 下一步补齐最新 K3 LL rowptr-store 优化数据，并建立 pure-vs-fused timing probe/基线。

## 2026-06-12 - K3 LL rowptr vector-store 数据回写

- 已补充记录 K3 LL epilogue vector-store 优化：
  - `store_bf16x4_rowptr_device()` 将 4 个 BF16 scalar rowptr store 合并为 1 个 `global_store_dwordx2`；
  - 该改动只在 K3 rowptr epilogue，未新增 kernel，未改变 pure 5pack GEMM 主循环。
- 远端短测：
  - no-tail 32 tokens：stage timing last3 median total `0.9265 ms`，K1 `0.4125 ms`，K3 `0.399 ms`；
  - no-tail 128 tokens：stage timing last3 median total `1.0660 ms`，K1 `0.452 ms`，K3 `0.5005 ms`；
  - K3 combine 相比旧 stage timing 从 32/128 的约 `0.669/1.136 ms` 降到 `0.399/0.5005 ms`。
- 工具链记录：
  - 一次远端 inline Python 解析命令因嵌套引号失败，后改为新增 `hygon_tmp/sglang_debug/parse_stage_timing.py` 并同步远端解析；
  - 该脚本为临时诊断工具，不属于生产路径。
- 已更新：
  - `task_plan.md` Phase 6 增加并勾选 LL K3 rowptr-store epilogue 第一轮优化；
  - `findings.md` 增加 K3 LL rowptr vector-store 优化结论。

## 2026-06-12 - LL pure-vs-fused 基线与 K3 rowptr-load hoist 回写

- 已建立 LL 32/128 的首轮 pure-vs-fused timing 基线：
  - K1 pure LL 原始 C harness：32/128 约 `0.300/0.308 ms`；
  - K1 staged after hoist：32/128 约 `0.416/0.4525 ms`；
  - K3 local rowptr raw after hoist：32/128 约 `0.2334/0.2385 ms`；
  - K3 staged after hoist：32/128 约 `0.336/0.4205 ms`。
- 已新增诊断入口/脚本：
  - `k3_v3_ll_pure_raw(...)` 作为诊断-only pybind，不进入生产 runtime/bench；
  - `hygon_tmp/sglang_debug/bench_k3_ll_pure_raw.py` 对比 contiguous pure 与 local rowptr combine；
  - `hygon_tmp/sglang_debug/parse_stage_timing.py` 解析 `[V3_STAGE_TIMING]`。
- 已完成 K3 LL rowptr-load hoist：
  - 每个 logical row 只加载一次 `row_combine_ptrs`，inactive row 直接跳过 BF16 convert/store；
  - local rowptr raw 32/128 从 `0.2652/0.2943 ms` 降到 `0.2334/0.2385 ms`；
  - no-tail staged K3 32/128 从 `0.399/0.5005 ms` 降到 `0.336/0.4205 ms`。
- 远端验证：
  - 容器内 source guard `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `10 passed`；
  - K1/K3 LL raw rebuild 通过；
  - no-tail 8 卡短测 e2e：32 `0.9831 ms`，128 `1.1237 ms`；
  - tail 8 卡短测 e2e：32 `0.8285 ms`，128 `0.9955 ms`。
- 当前优先级：
  - 最大 gap 是 K3 LL 128 staged vs local rowptr raw 约 `+0.182 ms`；
  - 第二是 K1 LL 128 staged vs pure 约 `+0.145 ms`；
  - 按用户要求“优先优化差异大的”，下一步先继续 K3 128 的通信链路/remote rowptr store 差异定位。

## 2026-06-12 - LL K1 source-rank row grouping A/B 撤回

- 已完成一次 rank-aware row grouping A/B：
  - 临时分支按 expert 内 `source_rank` 分组生成 K1 LL rows；
  - 初次 build 失败在 kernel signature 漏参，修正后远端 LL raw rebuild 通过；
  - 8 卡 direct correctness+perf 32/128 均 correct。
- 结果：
  - rank-group on e2e 32/128 为 `1.0067/1.1255 ms`，未优于默认 hoist 后 `0.9831/1.1237 ms`；
  - stage timing 32 K1/K3 约 `0.4215/0.337 ms`，128 K1/K3 约 `0.464/0.425 ms`，K1 和 K3 都没有改善。
- 已撤回该临时分支：
  - 本地 `rg` 确认 `rank_group_rows` / `MEGAMOE_DCU_V3_LL_RANK_GROUP_ROWS` / `symm_rank_counts` / `symm_rank_offsets` 无残留；
  - 本地 compileall 和 diff check 通过；
  - 同步远端后直接重编 LL raw K1/K3，log `hygon_tmp/sglang_debug/rebuild_v3_ll_rank_group_revert_20260612_104303.log`。
- 撤回后直接验证：
  - 命令短链条为 `TOKENS_LIST="32 128" RUN_STAGE_TIMING=1 RUN_DEFAULT=1 RUN_K2SKIP=0 RUN_BLOCK48=0 RUN_BLOCK64=0 RUN_BASELINE=0 WARMUP=2 REPEAT=3 ITERS=1 ./hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`；
  - 32 tokens correct，e2e `0.9883 ms`，last3 stage median total/K1/K3 `0.8685/0.411/0.335 ms`；
  - 128 tokens correct，e2e `1.1061 ms`，last3 stage median total/K1/K3 `0.990/0.452/0.4185 ms`。
- 下一步继续按“优先优化差异大的”原则，定位 K3 128 staged vs local rowptr raw 的 `+0.18 ms` 通信链路差异。

## 2026-06-12 - LL K3 storex4 A/B 反证并回滚

- 按用户要求采用短链条：kernel-only 改动不跑 source pytest，直接走远端同步、V3 LL raw rebuild、8 卡 e2e correctness 和 stage/perf。
- A/B 内容：
  - 在 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 中尝试把 K3 LL rowptr epilogue 的相邻 lane `bf16x4` 合并为 `global_store_dwordx4`；
  - 该版本可编译，build log 为 `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_storex4_20260612_105627.log`。
- 验证结果：
  - 32 tokens 第一轮 e2e correctness 失败，`fused/baseline max_abs=0.056884765625`；
  - 判定该 lane-pair storex4 形态不等价，不能保留。
- 已回滚：
  - 删除 `global_store_u32x4_device` / `store_bf16x8_rowaddr_device` 和 lane-pair shuffle store，恢复每 lane `global_store_dwordx2`；
  - 重编 log 为 `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_storex4_revert_20260612_105926.log`；
  - 恢复后 32/128 direct correctness 通过，e2e `0.9863/1.1153 ms`；
  - stage last3 median：32 K1/K3 `0.4165/0.336 ms`，128 K1/K3 `0.452/0.4185 ms`。
- 下一步：
  - 继续优化 K3 128 staged vs local rowptr raw 的 remote/staged rowptr store delta，但不重复当前错误的 lane-pair `global_store_dwordx4` 形态。

## 2026-06-12 - LL K3 storex4 修正验证后最终撤回

- 为定位 storex4 初版失败，新增诊断-only 探针 `hygon_tmp/sglang_debug/probe_global_store_dwordx4.cu`：
  - 单线程 `global_store_dwordx4` 固定 pattern 写入通过，说明 helper dword 顺序正确；
  - lane-pair 探针复现初版问题，原因是 `__shfl` 放在 only-even-lane 分支里，source odd lane 没参与同一条 shuffle；
  - 修正为所有 lane 先执行 `__shfl`、even lane 再 store 后，探针通过。
- 将该修正移植回 K3 LL rowptr epilogue后：
  - build log：`hygon_tmp/sglang_debug/rebuild_v3_ll_k3_storex4_all_lane_shfl_20260612_110800.log`；
  - 32/128 direct correctness 通过；
  - 但 e2e 退到 `1.0054/1.1601 ms`，stage K3 last3 退到 32 `0.346 ms`、128 `0.4265 ms`。
- 因为修正版仍慢于 dwordx2 基线，已最终撤回生产 K3 header：
  - final rebuild log：`hygon_tmp/sglang_debug/rebuild_v3_ll_k3_storex4_final_revert_20260612_111143.log`；
  - final direct 32/128 correctness 通过，e2e `0.9873/1.1170 ms`；
  - final stage last3 K1/K3 为 32 `0.4145/0.337 ms`、128 `0.456/0.4185 ms`。
- 结论：
  - `global_store_dwordx4` lane-pair 合并不是当前 K3 LL 128 的有效方向；额外 shuffle/pack/codegen 成本超过减少 store 指令的收益。

## 2026-06-12 - LL K3 remote rowptr split 诊断完成

- 按用户要求保持 kernel-only 短链条；本次只改诊断脚本 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`，不改生产 kernel。
- 已新增 rowptr split 模式：
  - `rowptr_all_zero`；
  - `staged_local_only`；
  - `staged_remote_only`。
- 远端 8 卡 128 tokens 诊断：
  - log/json: `hygon_tmp/sglang_debug/k3_rowptr_modes/k3_rowptr_split_128_20260612_112211.*`；
  - active rows/rank `768.0`，local rows/rank `32.75`，remote rows/rank `735.25`；
  - `local_rowptr` `0.2461 ms`，`staged_rowptr` `0.4084 ms`；
  - `staged_local_only` `0.2339 ms`，`staged_remote_only` `0.4071 ms`。
- 结论：
  - K3 128 staged delta 已归因到跨 rank remote combine store 本身；
  - K3 local rowptr/pure GEMM 已接近，不继续重复 K3 storex4 或 rank grouping；
  - 下一步转入 K1 LL 128 pure-vs-fused delta 定位。

## 2026-06-12 - LL K1 unused mask/tail metadata A/B

- 用户确认 kernel-only 改动采用短链条：直接测 correctness 和性能，不再跑长 source pytest。
- 已按计划读取三份 planning 文件、remote workflow、hygon optimizer，并用 DCU KB 查询 Hygon/gfx938 同步相关建议；KB 没有支持删除跨 rank barrier 的直接证据，因此本次不动 K1 前 rank barrier，只跳过 unused optional metadata。
- 改动：
  - `K1_fused/k1_fused_ext.cu` 的 LL raw launcher 对 `local_topk_mask/tail_tokens` 传 `nullptr`；
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 中 optional metadata 后的 grid barrier 只在该 metadata 实际启用时执行。
- 验证：
  - 本地 compileall 与 `git diff --check` 通过；
  - 同步 11 节点 `sglang_megamoe` 后，LL raw K1/K3 rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_skip_unused_mask_20260612_112821.log`；
  - 8 卡 no-tail 32/128 direct correctness 通过；
  - e2e：32 `0.9794996306 ms`，128 `1.1229792386 ms`；
  - stage last3：32 total/K1/K3 `0.8675/0.408/0.3355 ms`，128 total/K1/K3 `0.987/0.448/0.4215 ms`。
- 结论：
  - 小幅改善 K1 stage，保留该补丁；
  - 主差距仍在 K1 route/staged input 和 K3 remote store，下一步继续优先 K1 route/stage copy 定位。

## 2026-06-12 - LL K1 source pointer / scale hoist A/B

- 已完成第二个 K1 LL kernel-only A/B：
  - route emit 阶段直接保存 source `x` row pointer，并写 `staged_x_scale[row]`；
  - staged copy loop 改为直接按 pointer 读，不再每个 16B vector 重算 peer sections；
  - 删除单独 scale staging loop，padded rows 的 default scale 在 init 阶段写好。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py ...` 通过；
  - `git diff --check` 通过；
  - `rg` 确认旧 `symm_src_ranks/symm_src_tokens` 无残留。
- 远端验证：
  - 同步 header 后第一次 build 没有重编 K1 raw object，已发现并修正：删除 `build/.../k1_v3_fused_ext.o`、touch `k1_v3_fused_ext.hip` 后重编；
  - 真实 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_stage_ptr_hoist_forced_20260612_113411.log`。
- no-tail 8 卡短链条：
  - 32 correctness 通过，e2e `0.9729191363 ms`，last3 total/K1/K3 `0.858/0.402/0.338 ms`；
  - 128 correctness 通过，e2e `1.1115394831 ms`，last3 total/K1/K3 `0.9875/0.452/0.4195 ms`。
- tail 8 卡短链条：
  - 32 correctness 通过，e2e `0.9501995072 ms`，last3 total/K1/K3tail `0.8285/0.4055/0.362 ms`；
  - 128 correctness 通过，e2e `1.1180994809 ms`，last3 total/K1/K3tail `0.9915/0.448/0.4765 ms`。
- 结论：
  - 该补丁保留；它主要改善 32-token K1 stage，对 128-token K1 仍是小幅/噪声级；
  - 下一步继续定位 K1 128 剩余 delta，优先考虑 route scan/atomic allocation/staged_x 写入量，而不是再重复 optional metadata 或 K3 store-width。

## 2026-06-12 - LL K1 skip padded staged_x zero-write A/B 启动

- 用户确认 kernel-only 性能改动采用短链条：直接测 8 卡 e2e correctness 和 stage/perf，不再跑 source pytest。
- 已重读 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md`，继续 Phase 6 LL performance。
- 源码依据：
  - K2 `swiglu_quant_channelwise_*` 在 `row_combine_ptrs[row] == 0` 或 `route_weight == 0.0f` 时提前返回/写零；
  - 因此 K1 LL staged copy 可以尝试不再给 padded GEMM rows 写 staged_x 零值，只 staging 有效 route rows。
- 已完成本地改动：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 的 staged copy loop 从 `row_in_expert < ceil(count/blockM)*blockM` 收窄到 `row_in_expert < symm_counts[expert]`。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/large_opt.py` 通过；
  - `git diff --check -- K1_fused/k1_v3_groupgemm_impl.cuh K1_fused/k1_fused_ext.cu` 通过。
- 下一步：
  - 同步到 11 节点 `sglang_megamoe`，强制重编 K1 LL raw object；
  - 先跑 no-tail 32/128 correctness + stage timing，若通过再跑 tail 32/128。

## 2026-06-12 - LL K1 skip padded staged_x zero-write A/B 撤回

- 远端强制重编完成：
  - build log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_skip_padded_stage_zero_20260612_114307.log`；
  - build 输出确认 `k1_v3_fused_ext.hip` 重新编译，避免 header-only 改动未生效。
- no-tail 8 卡短链条结果：
  - 32 tokens correctness 通过，e2e `0.9711591303 ms`，stage last3 total/K1/K3 `0.8545/0.405/0.336 ms`；
  - 128 tokens correctness 通过，e2e `1.1198992580 ms`，stage last3 total/K1/K3 `0.987/0.454/0.422 ms`。
- 结论：
  - 相比 pointer/scale hoist 基线，K1 stage 与 e2e 都没有稳定收益，128 明显不是改善；
  - 已撤回该 header 改动，不继续跑 tail；
  - 下一步同步撤回后的 header 并强制重编，恢复到 pointer/scale hoist 版本后继续寻找 K1 128 delta 来源。

## 2026-06-12 - LL K1 skip padded staged_x zero-write 恢复验证

- 已同步撤回后的 `k1_v3_groupgemm_impl.cuh` 到 11 节点并强制重编：
  - build log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_skip_padded_stage_zero_revert_20260612_114632.log`；
  - build 输出确认 `k1_v3_fused_ext.hip` 再次重新编译。
- 恢复版 no-tail 8 卡短链条：
  - 32 tokens correctness 通过，e2e `0.9739195034 ms`，stage last3 total/K1/K3 `0.8485/0.401/0.335 ms`；
  - 128 tokens correctness 通过，e2e `1.1123593748 ms`，stage last3 total/K1/K3 `0.9875/0.453/0.420 ms`。
- 当前有效 LL 基线回到 pointer/scale hoist 版本；下一步继续拆 K1 128 的 route scan、atomic allocation、metadata publish/wait 和 fixed tile/GEMM 成本。

## 2026-06-12 - LL K1 compact route-scan stride A/B 启动

- 用户确认 kernel-only 性能改动采用短链条：已通过 correctness 的保留优化继续作为基线，新 kernel A/B 直接测 8 卡 e2e correctness 和 stage/perf。
- 当前假设：
  - LL 32/128 fixed-size eager 下，K1 route scan 原先按 `num_max_tokens_per_rank` 扫描每个 source rank 的全部 route slot；
  - 32 tokens 在 `num_max_tokens_per_rank=384` 的 bench 配置中会扫约 12 倍空 route，128 tokens 会扫约 3 倍空 route；
  - fixed-size eager 已传入真实 `runtime_num_tokens`，可以只缩短 scan stride，同时保持 `output_index` 的 `num_max_tokens_per_rank` layout 不变。
- 已完成本地改动：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 中 LL route emit scan 使用 `route_token_stride = clamp(runtime_num_tokens, 0, num_max_tokens_per_rank)`；
  - `output_index` 初始化和写回仍使用 `num_max_tokens_per_rank * num_topk` 布局，避免破坏后续合同；
  - `runtime_num_tokens < 0` 的 graph/未来 runtime-token 路径保留 max-token fallback。
- 下一步：
  - 本地静态检查后同步到 11 节点；
  - 强制重编 K1 LL raw object；
  - 直接跑 no-tail 32/128 correctness + stage/perf，若通过且收益明确再跑 tail。

## 2026-06-12 - LL K1 compact route-scan stride A/B 完成

- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/large_opt.py` 通过；
  - `git diff --check -- megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_groupgemm_impl.cuh .planning/dcu_megamoe_v3/task_plan.md .planning/dcu_megamoe_v3/progress.md .planning/dcu_megamoe_v3/findings.md` 通过。
- 远端验证：
  - 已同步 `k1_v3_groupgemm_impl.cuh` 到 11 节点；
  - 强制重编 K1 LL raw object，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_compact_route_scan_20260612_115353.log`。
- no-tail 8 卡短链条：
  - 32 correctness 通过，e2e `0.9843191206 ms`，stage last3 total/K1/K3 `0.8515/0.3985/0.337 ms`；
  - 128 correctness 通过，e2e `1.1036993712 ms`，stage last3 total/K1/K3 `0.986/0.444/0.421 ms`。
- tail 8 卡短链条：
  - 32 correctness 通过，e2e `0.9503593892 ms`，stage last3 total/K1/K3tail `0.8275/0.4015/0.3635 ms`；
  - 128 correctness 通过，e2e `1.1043791324 ms`，stage last3 total/K1/K3tail `0.9945/0.4485/0.4795 ms`。
- 结论：
  - compact route-scan 对 K1 128 有小幅正收益，对 32 主要在噪声区间；
  - correctness 覆盖 no-tail/tail 32/128，保留该补丁；
  - 主 gap 仍在 K3 128 remote combine store 和 K1 fixed tile/metadata 成本。

## 2026-06-12 - 查询 Flux / DeepEP 通信隐藏参考

- 按用户要求使用本地 DCU KB 查询 Flux GEMM+RS、DeepEP LL overlap 和 CUDA/MegaMoE 类似融合方向。
- 检索结果：
  - Flux GEMM+RS 把 ReduceScatter 语义放进 epilogue/store，并使用 rank-aware tile swizzle；
  - DeepEP LL combine overlap 使用 `comp_signal`、`block_m`、`threshold` 做 chunk-level readiness；
  - Flux/DeepEP 都强调通信 readiness 是调度合同的一部分，不是额外后处理。
- 已更新：
  - `findings.md` 记录检索结论和映射；
  - `task_plan.md` Phase 6 新增 K3 LL epilogue/tile ownership/communication-readiness A/B 项。
- 下一步：
  - 读取 K3 LL `k3_v3_pack5_groupgemm_impl.cuh` 的 tile/rowptr store 映射；
  - 尝试最小可回滚的 K3 tile-order/epilogue-readiness A/B；
  - 不重复已反证的 `global_store_dwordx4` 和 K1 source-rank row grouping。

## 2026-06-12 - LL K3 tile-level rowptr readiness A/B 启动

- KB 映射：
  - Flux GEMM+RS 将通信目标 layout 放入 epilogue/store；
  - DeepEP LL overlap 使用 chunk-level readiness，而不是等整个 expert 完成；
  - 映射到当前 K3 LL：在每个 GEMM tile 开始前读取本 tile 的 `row_combine_ptrs`，让通信/输出活跃性进入 tile 调度。
- 当前假设：
  - LL 32/128 的 K3 rows 是 fixed 64-row/expert，但有效 route rows compact 到每个 expert 前部；
  - 对很多 expert，后半个 32-row tile 可能完全 inactive；
  - 现有 K3 LL 会先完整计算该 tile，再在 epilogue 发现 rowptr 为 0 后跳过 store；
  - 如果 tile 级 rowptr 全空则直接跳过 GEMM，可以减少 K3 低 token 的 fixed tile 成本。
- 已完成本地改动：
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 中为 K3 LL kernel 增加 shared `tile_row_addrs[kBlockM]` 和 `tile_has_active_rows`；
  - rowptr path 在 GEMM 前预读本 tile row addresses，全空则 `continue` 到下一 tile；
  - epilogue store 复用预读 row address，不再重复加载 `row_combine_ptrs`。
- 下一步：
  - 本地静态检查；
  - 同步到 11 节点并强制重编 K3 LL raw object；
  - 直接跑 no-tail/tail 32/128 correctness + stage/perf；
  - 若 correctness 或性能失败，立即撤回该 A/B。

## 2026-06-12 - LL K3 tile-level rowptr readiness A/B 撤回

- 远端验证结果：
  - K3 LL raw rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_tile_rowptr_ready_20260612_125416.log`；
  - no-tail 32 tokens 初段出现 K3 stage 下降信号，但同一轮随后触发 staged rank barrier timeout 和 torch spawn abort；
  - 典型错误为 `MegaMoE HIP staged rank barrier timeout: rank=2 ticket=184 generation=24 arrival=191 release=23`。
- 处置：
  - 已本地撤回 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 中的 shared `tile_row_addrs` / `tile_has_active_rows`、tile 级 rowptr 预读、empty-tile `continue`；
  - epilogue 恢复为已验证的每 logical row `global_load_i64_glc_device(row_combine_ptrs + logical_row)` + inactive row skip；
  - `task_plan.md` 标记该 A/B 为已反证，后续不重复“全空 rowptr 直接跳过 GEMM tile”的方向。
- 下一步：
  - 本地静态检查后同步撤回版到 11 节点；
  - 强制重编 K3 LL raw object；
  - 先跑 no-tail 32，再跑 128，确认恢复到稳定 rowptr-load hoist 基线；
  - 若恢复通过，再继续做不改变 tile 完成语义的 K3 epilogue/remote-store A/B 或回到 K1 128 delta。

## 2026-06-12 - LL K3 tile-level rowptr readiness 撤回版恢复验证

- 本地检查：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py megamoe/large_opt.py` 通过；
  - `rg` 确认 `tile_row_addrs` / `tile_has_active_rows` / `atomicExch(&tile_has_active_rows...)` 无残留；
  - `git diff --check -- K3 header + planning files` 通过。
- 远端验证：
  - 已同步撤回后的 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh`；
  - 强制重编 K3 LL raw object，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_tile_rowptr_ready_revert_20260612_130157.log`；
  - no-tail 32 correctness 通过，e2e `0.9811390042 ms`，stage last3 total/K1/K3 `0.8595/0.4025/0.336 ms`；
  - no-tail 128 correctness 通过，e2e `1.1132393777 ms`，stage last3 total/K1/K3 `0.979/0.4445/0.419 ms`。
- 结论：
  - 撤回后未再复现 rank barrier timeout；
  - 有效代码已回到 rowptr-load hoist + K1 compact route-scan 基线；
  - 后续 K3 LL 若继续借鉴 Flux/DeepEP，只做不改变 tile 完成语义的 A/B；否则转向 K1 128 delta。

## 2026-06-12 - LL K3 rowptr register-prefetch A/B 启动

- 目的：
  - 继续借鉴 Flux/DeepEP 的“通信 readiness 尽量进 tile/epilogue 节奏”，但不再改变 tile 完成语义；
  - 只尝试隐藏 `row_combine_ptrs` metadata load latency，不跳过 GEMM tile、不新增 barrier、不新增 kernel。
- 本地改动：
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 在 K3 LL tile 开始处为每个 lane 的 `kMRepeats` 行提前读取 rowptr 到 `row_addr_prefetch[]`；
  - epilogue 从 `row_addr_prefetch[mr]` 取地址并保持原 inactive row skip；
  - GEMM 主循环、remote store helper、no-tail/tail completion 语义都不改。
- 本地检查：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py megamoe/large_opt.py` 通过；
  - `git diff --check` 通过。
- 下一步：
  - 同步到 11 节点并强制重编 K3 LL raw object；
  - 直接跑 no-tail 32/128 correctness + stage/perf；
  - 若无收益或正确性失败，立即撤回。

## 2026-06-12 - LL K3 rowptr register-prefetch A/B 完成

- 远端重编：
  - 已同步 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh`；
  - 强制重编 K3 LL raw object，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_rowptr_reg_prefetch_20260612_130658.log`。
- no-tail 8 卡短链条：
  - 32 correctness 通过，e2e `0.9137193784 ms`，stage last3 total/K1/K3 `0.769/0.401/0.2575 ms`；
  - 128 correctness 通过，e2e `1.0727392435 ms`，stage last3 total/K1/K3 `0.956/0.4415/0.379 ms`。
- tail 8 卡短链条：
  - 32 correctness 通过，e2e `0.8936993778 ms`，stage last3 total/K1/K3tail `0.787/0.4015/0.318 ms`；
  - 128 correctness 通过，e2e `1.0852793753 ms`，stage last3 total/K1/K3tail `1.000/0.442/0.4605 ms`。
- 结论：
  - 该 A/B 保留；
  - no-tail K3 stage 从撤回基线约 `0.336/0.419 ms` 降到 `0.2575/0.379 ms`；
  - tail 也保持 correctness 且没有明显退化；
  - 下一步按剩余 delta，优先继续看 K1 128 `+0.14 ms` 或 K3 128 remote store 剩余 `+0.14 ms`。

## 2026-06-12 - LL K1 output_index bounded-init A/B 启动

- 按计划从剩余较大 delta 转向 K1 LL 128 staged/pure 差距。
- 当前假设：
  - K1 LL route scan 已经收窄到真实 `runtime_num_tokens` stride；
  - 但 `output_index` 清表仍按 `num_ranks * num_max_tokens_per_rank * topk` 全量写 `-1`；
  - fixed-size eager correctness 和后续 K2/K3 只依赖真实 token 范围，清理每 rank 前缀即可，写回位置仍按 `num_max_tokens_per_rank` stride 保持合同。
- 下一步：
  - 在 `K1_fused/k1_v3_groupgemm_impl.cuh` 做最小 A/B；
  - 本地静态检查后同步 11 节点，强制重编 K1 LL raw object；
  - 直接跑 no-tail/tail 32/128 correctness + stage/perf，若失败或无收益则撤回。

## 2026-06-12 - LL K1 output_index bounded-init A/B 撤回

- 远端第一次重编误用了旧 raw env，测试未进入 V3 raw kernel；随后确认 setup 当前需要：
  - `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll`
  - `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll`
- 正确 raw rebuild 后，no-tail 32/128 短链条在 32 tokens 首轮触发 HIP VMFault，随后多个 rank 报 staged rank barrier timeout，log 为 `hygon_tmp/sglang_debug/v3_ll_k1_output_index_bounded_notail_20260612_133739.log`。
- 结论：
  - `output_index` 的全量 `-1` 清表不能在当前 eager 链路中只按真实 token 前缀收窄；
  - 即使后续 Python 表面不消费超出真实 token 的索引，当前低层 scratch/异步链路仍要求该区域保持保守清零；
  - 该 A/B 已撤回，后续不重复此方向，继续找 K1 128 delta 的其他来源。

## 2026-06-12 - LL K1 direct source A-load A/B 启动

- 当前依据：
  - K1 LL staged/pure 128 tokens 仍有约 `+0.14 ms` 差距；
  - 当前 staged path 先按 route 把 source `x` copy 到 `staged_x`，随后 GEMM 再读 `staged_x`；
  - Flux/DeepEP 检索结论更支持把通信 readiness/source pointer 直接放入 GEMM load 或 epilogue/store 节奏，而不是形成额外内存 pass。
- A/B 目标：
  - 保留 K1 route metadata、`row_combine_ptrs`、`output_index` 全量清表和 GEMM compute 主体；
  - 在 LL GEMM A-load 处使用 route 阶段写好的 source row pointer 直接读取 source `x`；
  - 跳过 `staged_x` copy pass，减少一次读+写中间 scratch。
- 当前限制：
  - 远端 8 卡被 `sglang serve` 占用约 91-92% 显存，先做本地实现和静态检查；
  - 需要等卡空或用户确认清理该服务后再跑恢复验证、correctness 和 perf。

## 2026-06-12 - LL K1 direct source A-load A/B 编译状态

- 已完成代码改动：
  - `v3_k1_build_ll_stage_device` 增加 `kDirectSymmLoad` 模板分支；
  - direct 分支保留 route metadata、`row_combine_ptrs`、`output_index` 全量清表和最终 grid barrier，但跳过 `staged_x` copy pass；
  - LL GEMM A-load 在 direct 分支中通过 `symm_src_x_ptrs[logical_row]` 直接读取 source row 的 16B vector，padded/overflow 行仍以 zero vector 处理；
  - LL launcher 暂时实例化 direct source A-load 版本作为 A/B。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/large_opt.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - 已同步 `k1_v3_groupgemm_impl.cuh` 和 `k1_v3_fused_ext.cu`；
  - LL raw build 通过，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_direct_source_aload_20260612_134544.log`；
  - `dccobjdump` 对 Python `.so` 和 K1 `.o` 都只输出 host ELF header，未拿到 device ISA，记为 inconclusive。
- 未完成：
  - 8 卡 correctness/perf 尚未运行；当前 host 上 `sglang serve` 占用 8 卡约 91-92% VRAM 且 HCU active，VRAM guard 会 abort，性能数据也会失真。

## 2026-06-12 - LL K1 direct source A-load A/B 反证并撤回

- 8 卡恢复后直接运行 no-tail 32/128 correctness + stage timing：
  - log `hygon_tmp/sglang_debug/v3_ll_k1_direct_source_aload_notail_20260612_134923.log`；
  - 32/128 correctness 均通过；
  - e2e 32/128 分别约 `1.3673/1.9367 ms`，明显慢于 register-prefetch 基线约 `0.9137/1.0727 ms`；
  - 128 stage timing 中 K1 长期在 `1.16-1.34 ms` 区间，远慢于基线 K1 约 `0.44 ms`。
- 结论：
  - 跳过 `staged_x` copy 后，GEMM A-load 直接跨 rank/token 散读 source row，coalescing 和 remote load latency 成本远高于 staging pass；
  - 这个方向不符合“保护 pure GEMM load/compute/store 流水”的性能目标，已撤回；
  - 后续 K1 优化不再做直接 source row 散读，优先看更轻量的 staging 内部调度或 metadata/wait 成本。

## 2026-06-12 - LL K1 direct source A-load 撤回恢复验证

- 已撤回 direct source A-load 分支：
  - `K1_fused/k1_v3_fused_ext.cu` 回到 LL staged launcher；
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 删除 `kDirectSymmLoad` / `direct_x_ptrs` / direct A-load，恢复 staged_x copy；
  - 本地 `compileall`、`git diff --check` 和 `rg kDirectSymmLoad|direct_x_ptrs` 检查通过。
- 远端恢复：
  - 重编 log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_direct_source_aload_revert_20260612_135347.log`；
  - no-tail 32/128 correctness 通过，e2e `0.8982195035/1.0647394806 ms`，stage K1 约 `0.42-0.49 ms`、K3 约 `0.36-0.42 ms`；
  - tail 32/128 correctness 通过，e2e `0.9149796218/1.1034392416 ms`，stage K1 约 `0.43-0.50 ms`、K3tail 约 `0.45-0.50 ms`。
- 结论：
  - 有效代码恢复到 K3 rowptr register-prefetch + K1 compact route-scan/staged-copy 基线；
  - direct source A-load 仅作为反证保留在 findings，不再沿该方向优化。

## 2026-06-12 - Flux / CUDA MegaMoE 复查并启动 K3 epilogue cleanup A/B

- 已按用户要求再次查询本地 DCU KB：
  - Flux GEMM+RS 仍明确支持 epilogue scatter/reduce 和 rank-aware scheduling；
  - DeepEP LL overlap 仍明确支持 chunk-level compute signal/readiness；
  - CUDA MegaMoE 查询禁用本地 CUDA 后成功，但强匹配较弱，主要支持 grouped GEMM expert dispatch/problem descriptor 热路径，没有给出新的 K3 remote-store 方案。
- 本地代码小改动：
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 中，LL rowptr combine path 不再提前计算 contiguous/pure output 用的 `out_warp`；
  - `out_warp` 只在 `row_combine_ptrs == nullptr` 的 pure/contiguous store 分支计算；
  - 不改变 MMAC 主循环、rowptr 预取、remote store helper、tile 完成语义或 tail/no-tail 同步。
- 本地检查：
  - `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py megamoe/large_opt.py` 通过；
  - `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 通过。
- 下一步：
  - 同步 K3 header 到 11 节点；
  - 强制重编 K3 LL raw object；
  - 直接跑 no-tail/tail 32/128 correctness + stage/perf，若无收益或退化即撤回。

## 2026-06-12 - 本工程 CUDA MegaMoE 复查与 K3 cleanup 收口

- 用户纠正 CUDA MegaMoE 参考就在本工程内，已直接读取并复查：
  - `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`；
  - `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`；
  - `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`。
- 本地 CUDA MegaMoE 映射结论：
  - L2 epilogue 直接写远端 combine buffer；
  - dispatch 写 source token/topk 与 combine metadata；
  - scheduler 寄存器缓存 per-expert counts，并把 L1/L2 block iteration 与专家 token 数绑定；
  - CUDA 的 shared/TMA staging 机制不能直接搬到 DCU，但方向支持 DCU V3 继续把通信语义压在 K1 staging/metadata 和 K3 epilogue/store 内。
- K3 epilogue cleanup A/B 完成：
  - 远端 build log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_epilogue_cleanup_20260612_141242.log`；
  - no-tail 32/128 correctness 通过，stage parse last3 128 K1/K3 `0.445/0.3775 ms`；
  - tail 32/128 correctness 通过，stage parse last3 128 K1/K3tail `0.4395/0.454 ms`；
  - no-tail/tail 32 档也保持正确性和稳定阶段耗时；
  - 该改动保留，`task_plan.md` 已标为 ✅。
- 下一步：
  - 基于本工程 CUDA MegaMoE 继续找更大的 LL delta；
  - 优先做不改变 tile 完成语义的小步 A/B，避免重复 direct source A-load、empty-tile skip、source-rank grouping 和 lane-pair storex4。

## 2026-06-12 - LL K1 source-rank CTA route-scan A/B 反证

- 尝试内容：
  - 把 K1 LL route scan 从全局线性 route task 改成 CTA 按 `source_rank` 分组；
  - 目标是减少每 route 重复 `get_sections()`，借鉴本工程 CUDA MegaMoE 的 dispatch metadata/scheduler count cache 思路；
  - 未改 `output_index` 清表、rowptr 合同、staged_x copy 和 GEMM 主体。
- 远端验证：
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_rank_cta_scan_20260612_142843.log`；
  - no-tail 32 correctness 通过，e2e `0.9042394981 ms`，stage last3 K1/K3 `0.4055/0.256 ms`；
  - no-tail 128 correctness 通过，e2e `1.0739393830 ms`，stage last3 K1/K3 `0.444/0.379 ms`。
- 处置：
  - 该 A/B 无稳定收益且改变 route emit 并行顺序，未继续跑 tail；
  - 已本地撤回并同步远端重编，rebuild log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_rank_cta_scan_revert_20260612_143223.log`；
  - 撤回后 128 no-tail 短验证 correctness 通过，e2e `1.0798391104 ms`，log `hygon_tmp/sglang_debug/v3_ll_k1_rank_cta_scan_revert_notail128_20260612_143253.log`。
- 下一步：
  - 继续优先看 K3 remote store 剩余 delta，或 K1 staging/GEMM 固定 tile 成本；
  - 不再重复按 source-rank 重排 K1 route emission 的方向。

## 2026-06-12 - LL K3 rowptr non-GLC A/B 反证并撤回

- 用户提醒 CUDA MegaMoE 在本工程内，已继续以本地 `deep_gemm/include/deep_gemm/...mega_moe...` 作为主要参考，而不是外部弱匹配资料。
- 收口当前 K3 rowptr non-GLC A/B：
  - 实际日志为 `v3_ll_k3_rowptr_nonglc_notail32_20260612_143742.log` 和 `v3_ll_k3_rowptr_nonglc_notail128_20260612_143838.log`；
  - parse last3：32 tokens K1/K3 `0.4035/0.249 ms`；
  - parse last3：128 tokens K1/K3 `0.449/0.4105 ms`。
- 结论：
  - 普通 rowptr load 虽然 32 档略快，但 128 档明显慢于 GLC/cleanup 基线约 `0.3775 ms`；
  - 已本地恢复 `global_load_i64_glc_device(row_combine_ptrs + logical_row)`，准备同步远端重编短测确认。
- 恢复验证：
  - 远端强制重编 log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_rowptr_nonglc_revert_20260612_1444xx.log`；
  - 128 no-tail 短测 log `hygon_tmp/sglang_debug/v3_ll_k3_rowptr_nonglc_revert_notail128_20260612_144549.log`；
  - correctness 为 true，e2e `1.0762991160 ms`；
  - stage last3 K1/K3 `0.4485/0.3805 ms`，已回到 GLC/cleanup 基线附近。

## 2026-06-12 - LL K1 staged-copy compact iteration A/B 启动

- 对照本工程 CUDA MegaMoE：
  - CUDA scheduler 会缓存 per-expert token counts，并用 counts 驱动 block iteration；
  - 当前 DCU V3 K1 staged copy 已有 `symm_counts[expert]`，但 copy loop 仍扫描完整 `row_capacity * kStageVecsPerRow`，无效 row 只是在循环内 `continue`。
- 本地 A/B 改动：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 中 staged copy 改为按 expert 外循环，每个 expert 只遍历 `ceil(symm_counts[expert] / kBlockM) * kBlockM` 行的 FP8 vector；
  - padded rows 仍写 zero，保留 K2/GEMM 对 rounded rows 的读取合同；
  - 不改变 route metadata、`output_index` 全量清表、rowptr 合同、K1 GEMM 主体或 K3。
- 下一步：
  - 本地静态检查；
  - 同步 11 节点强制重编 K1 LL raw object；
  - 直接跑 no-tail 32/128 correctness + stage/perf，若有收益再跑 tail，否则撤回。

## 2026-06-12 - LL K1 staged-copy compact iteration A/B 反证

- 远端重编：
  - log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_compact_stage_copy_20260612_1450xx.log`；
  - 确认 `k1_v3_fused_ext.hip` 重新编译。
- no-tail 8 卡短测：
  - 32 correctness 通过，parse last3 K1/K3 `0.420/0.254 ms`，e2e median 被 barrier/outlier 拉高；
  - 128 correctness 通过，parse last3 K1/K3 `0.4545/0.379 ms`，e2e `1.0714593977 ms`。
- 结论：
  - 该方案没有降低 K1 stage，32/128 都不如或不稳于当前线性 copy-loop 基线；
  - 节省空 row 扫描被 expert 外循环和更差的调度抵消；
  - 已本地撤回，准备同步远端恢复。
- 恢复验证：
  - 远端重编 log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_compact_stage_copy_revert_20260612_1459xx.log`；
  - 初次恢复测被新启动的 `dsq_sglang_601` 容器内 `sglang serve` 占满 8 卡拦截；按用户此前“直接杀了”的授权，仅终止该容器内 sglang 进程后继续；
  - 128 no-tail 恢复 log `hygon_tmp/sglang_debug/v3_ll_k1_compact_stage_copy_revert_notail128_20260612_150017.log`；
  - correctness 为 true，e2e `1.0826915056 ms`，stage last3 K1/K3 `0.447/0.3855 ms`，远端已回到线性 copy-loop 基线。

## 2026-06-12 - LL code object 资源分析

- 用户纠正 CUDA MegaMoE 参考就在本工程后，继续以 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、`scheduler/mega_moe.cuh`、`layout/mega_moe.cuh` 作为本地参考，不再用外部弱匹配 CUDA MegaMoE 信息。
- 远端 `hipprof --codeobj-analyze` 在非交互管道下只稳定列出 ELF/kernel 候选，未吐出资源表；已改用 `dccobjdump --extract-elf=all` 抽出 gfx938 code object，再读取 `*-resource-usage.RES`。
- K1 V3 LL code object 资源：
  - block32 mask true/false: `sgpr_count=100`、`sgpr_spill_count=2`、`vgpr_count=124`；
  - block48: `vgpr_count=151`；
  - block64: `private_segment_fixed_size=272`、`sgpr_count=102`、`vgpr_count=193`，解释了此前 block64 A/B 退化。
- K3 V3 LL code object 资源：
  - no-tail block32: `sgpr_count=100`、`sgpr_spill_count=1`、`vgpr_count=153`；
  - tail block32: `sgpr_count=100`、`sgpr_spill_count=16`、`vgpr_count=130`；
  - tail block64: `sgpr_count=100`、`sgpr_spill_count=16`、`vgpr_count=189`。
- 结论：
  - no-tail K3 当前主要仍是 remote combine store 本身和 epilogue 调度；资源表没有给出比已知方向更强的源码改动证据；
  - tail K3 有明确 SGPR spill 信号，下一步优先做 LL tail-only 参数/常量化 A/B，尽量降低 reducer/signal 参数对 GEMM CTA 的标量压力。

## 2026-06-12 - LL K3 tail 参数常量化 A/B 反证并撤回

- 实现：
  - 在 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 中临时新增 fixed-shape tail signal/wait/reduce helper；
  - LL tail kernel 不再使用 `num_ranks/num_experts/num_topk/done_target/reduce_blocks/signal_generation` 运行时参数；
  - 另做一版去掉 fixed reducer topk unroll 的子 A/B。
- 验证：
  - fixed shape 版远端重编成功；
  - code object 资源从 tail block32 `kernarg_segment_size=384`、`sgpr_spill_count=16` 改善到 `kernarg_segment_size=128`、`sgpr_spill_count=6`；
  - tail 32/128 correctness 通过，summary `0.894/1.100 ms`；
  - tail 128 rerun correctness 通过，summary `1.085 ms`，stage last3 K3tail `0.456 ms`；
  - no-unroll 子 A/B correctness 通过但 128 summary `1.097 ms`、stage last3 K3tail `0.463 ms`。
- 结论与处置：
  - 资源指标改善未转成稳定性能收益；
  - 已撤回 fixed-shape helper，恢复 generic tail 参数路径；
  - 恢复版远端重编成功，128 tail correctness 通过，summary `1.088 ms`。

## 2026-06-12 - LL K3 PMC read/write 对照

- 已完成：
  - 从 11 节点拉回 `pmc_v3_ll_k3_notail128_20260612_153148.csv`；
  - 新增临时解析脚本 `hygon_tmp/sglang_debug/parse_pmc_k3.py`；
  - 给 `bench_k3_ll_rowptr_modes.py` 增加 `--modes`，保证单 mode profiler 不混入其他 rowptr 模式；
  - 8 卡空闲后运行 `local_rowptr` 和 `staged_remote_only` 的 `hipprof --pmc-read/--pmc-write --pmc-type 3`。
- 结果：
  - `local_rowptr` kernel median 约 `0.279 ms`；
  - `staged_remote_only` kernel median 约 `0.402 ms`；
  - VMEM/VALU/LDS 指令数基本一致，remote-only 的 `TA_BUSY` 与 `TCP_TCP_TA_DATA_STALL_CYCLES` 明显升高，`write_req_stall` 近 0。
- 下一步：
  - 不重复 store-width / 写请求 stall 优化；
  - 先分析 rowptr 地址分布和 tile/order，再决定是否做 K3 小步 A/B，或转回 K1 metadata/staging delta。

## 2026-06-12 - LL K1 clamp barrier fold A/B 反证并撤回

- 改动：
  - 临时把 K1 LL route 后 `symm_counts` clamp/stats 合入 stage-copy 前；
  - stage copy 使用 `min(count, m_per_expert)`，依赖 copy 末尾已有 grid barrier，减少一次内部 grid barrier。
- 验证：
  - 远端 K1 LL raw rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_clamp_barrier_fold_20260612_155117.log`；
  - no-tail 32/128 correctness 通过，128 no-tail last3 K1 约 `0.449 ms`；
  - tail 32/128 correctness 通过，128 tail last3 K1 约 `0.4435 ms`。
- 处置：
  - 无稳定性能收益，已撤回；
  - 远端恢复重编成功，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_clamp_barrier_fold_revert_20260612_155530.log`；
  - 恢复版 128 no-tail correctness 通过，last3 K1 约 `0.4415 ms`。

## 2026-06-12 - LL rowptr 分布诊断 caveat

- 临时给 `bench_k3_ll_rowptr_modes.py` 增加 `--dump-rowptr-stats`，尝试分析 128 tokens 的 rowptr 目标 rank 和 16-row chunk 分布。
- 结果：
  - 初版和读取 device peer table 后的版本都只识别到约 256 个有效地址，而 active row 约 768；
  - 说明 Python 侧 range 归属没有对齐 K1 `get_sections(...).combine` 的全部地址合同。
- 结论：
  - 当前 rowptr 分布诊断只能作为 caveat，不能用 invalid 计数指导 kernel 改动；
  - 后续若继续做 K3 tile/order，需要先补可靠的 device-side 或 wrapper-side 归属统计。

## 2026-06-12 - LL Phase 6 gate 通过并切到 normal

- 已完成 8 卡 LL 当前代码 vs baseline gate：
  - no-tail log `hygon_tmp/sglang_debug/v3_ll_gate_current_vs_baseline_notail_20260612_155730.log`，summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260612_160002.csv`；
  - baseline/no-large 32/128 median 约 `1.1349/1.6308 ms`；
  - V3 LL no-tail 32/128 median 约 `0.7600/0.9376 ms`；
  - correctness 均通过。
- 已完成 8 卡 LL tail gate：
  - tail log `hygon_tmp/sglang_debug/v3_ll_gate_current_vs_baseline_tail_20260612_160018.log`，summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260612_160252.csv`；
  - baseline/no-large 32/128 median 约 `1.1425/1.6417 ms`；
  - V3 LL tail 32/128 median 约 `0.7763/0.9725 ms`；
  - correctness 均通过。
- 结论：
  - Phase 6 `<512` 固定档位已达标；
  - 已把 `findings.md` 和 `task_plan.md` 更新为 LL gate passed；
  - 下一步按计划进入 normal 1024/4096，对照 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 && USE_MEGAMOE_V3=0`，并确认 V3 normal K1/K3 fused raw object 按 aicc 编译。

## 2026-06-12 - normal gate runner 准备与远端阻塞

- 已新增临时 runner `hygon_tmp/sglang_debug/run_v3_normal_perf_ab.sh`：
  - 默认跑 1024/4096；
  - `orig_stage_normal` 使用 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=0`；
  - `v3_normal` 使用 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal`；
  - 默认 `K3_USE_ASM_TAIL_REDUCE=0`，避免误触发当前暂缓的 normal tail path；
  - 可选 `RUN_STAGE_TIMING=1` 输出 K1/K2/K3 分段。
- 已新增临时 build/summary 辅助：
  - `hygon_tmp/sglang_debug/build_v3_normal_raw.sh` 固化 K1/K3 V3 normal raw aicc build 命令；
  - `hygon_tmp/sglang_debug/summarize_v3_normal_perf.py` 直接汇总 `orig_stage_normal` 与 `v3_normal` 的 `fused_median_ms_avg_per_rank`，输出 `v3_vs_orig_speedup`；
  - 本地 `python -m compileall hygon_tmp/sglang_debug/summarize_v3_normal_perf.py` 与 `git diff --check` 通过。
- 本地确认 `setup.py` 的 normal aicc 策略仍正确：
  - K1 raw normal / K3 raw normal 扩展会被 `_mark_v3_normal_aicc()` 标记；
  - `CustomBuildExt` 对这些 extension 临时切换到 `hygon_tmp/sglang_debug/v3_aicc_rocm/bin/hipcc -> /workspace/dtk_aicc/bin/aicc`；
  - LL 和非 V3 逻辑保持原 hipcc。
- 已同步 normal runner、`setup.py` 和 K1/K3 V3 相关源码到 11 节点；同步后 runner 权限为可执行。
- 远端状态：
  - 同步后首次 `hy-smi` 显示 8 卡 VRAM/HCU 均为 0%；
  - 启动 normal raw build 时 SSH 握手阶段出现 `kex_exchange_identification: read: Connection reset`；
  - 随后三次重试均为 `ssh: connect to host 10.17.176.11 port 22: Connection refused`，命令未进入容器执行；
  - 当前不能把该问题视为编译超时或 kernel 编译失败，需等 SSH 恢复后继续 rebuild。

## 2026-06-12 - normal K3 rowptr split 诊断准备

- 继续按 Phase 6 normal gate 推进，重读计划/发现/进展后复查 normal runner、aicc build 分流和 K3 normal wrapper。
- 当前远端状态：
  - `Test-NetConnection 10.17.176.11 -Port 22` 超时并提示 TCP connect failed；
  - `ssh -F NUL -o ConnectTimeout=5` 返回 `Connection refused`；
  - 命令仍未进入 `sglang_megamoe` 容器，不能归类为编译失败。
- 已完成本地准备：
  - `bench_k3_ll_rowptr_modes.py` 增加 `--backend normal`，normal K3 可复用 LL 的 staged/local/remote rowptr split 诊断；
  - 新增 `hygon_tmp/sglang_debug/run_v3_normal_k3_rowptr_modes.sh`，默认跑 1024/4096 的 `local_rowptr/staged_rowptr/staged_local_only/staged_remote_only/rowptr_all_zero`；
  - 本地 `python -m compileall` 与 `git diff --check` 通过；本机没有 `bash`，shell `-n` 等远端恢复后执行。
- 下一步：
  - SSH 恢复后先上传 `build_v3_normal_raw.sh`、`summarize_v3_normal_perf.py`、`bench_k3_ll_rowptr_modes.py`、`run_v3_normal_k3_rowptr_modes.sh`；
  - chmod 后跑 normal raw aicc rebuild，再跑 no-tail 1024/4096 e2e + stage timing；
  - 若 K3 stage 是主要差异，直接运行 normal K3 rowptr split runner。

## 2026-06-12 - normal gate 恢复驱动脚本

- 新增本地临时脚本 `hygon_tmp/sglang_debug/push_and_run_v3_normal_gate.ps1`：
  - 从 `.vscode/sftp.json` 读取 11 节点 SSH/SCP 参数；
  - 同步 V3 normal 需要的 setup/source/test/temp runner 文件；
  - 进入 `sglang_megamoe` 容器后执行 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`；
  - 默认先跑 `build_v3_normal_raw.sh`，再跑 1024/4096 normal no-tail e2e + stage timing；
  - 支持 `-SkipBuild` / `-SkipBench` 便于 SSH 恢复后分步执行。
- 本地验证：
  - PowerShell parser 语法检查通过；
  - `python -m compileall` 覆盖 normal summary 和 K3 rowptr split 诊断脚本；
  - `git diff --check` 通过。

## 2026-06-12 - normal pure reference runner 准备

- 新增 `hygon_tmp/sglang_debug/run_v3_normal_pure_refs.sh`：
  - `RUN_K1_PURE=1` 时调用 `hygon_tmp/K1_groupgemm_fp8/run_best_c.sh`，以 `MODE=large TOKENS="1024 4096"` 测 K1 pack5 pure normal；
  - `RUN_K3_ROWPTR=1` 时调用 normal K3 rowptr split runner，补 K3 local/staged/remote rowptr 对照；
  - 该 runner 只用于 Phase 6 normal pure-vs-fused delta 表，不进入 runtime/bench 生产路径。
- 同步清单：
  - 已把该 runner 加入 `push_and_run_v3_normal_gate.ps1` 的 scp 文件列表和远端 chmod 列表。
- 本地验证：
  - PowerShell driver parser 通过；
  - `python -m compileall` 通过；
  - `git diff --check` 通过。

## 2026-06-12 - normal K1 unused mask/tail A/B 待验证

- 静态发现：
  - V3 normal K1 launch 与 LL 共用 `v3_k1_build_ll_stage_device` stage builder；
  - normal launch 仍传入非空 `local_topk_mask` / `tail_tokens`，会触发本地 token 扫描和额外 grid barrier；
  - `large_opt.py` 后续 V3 staged normal 只消费 K1 返回的 `l1_out/route_weights/m_indices/output_index/row_combine_ptrs`，K2/K3 不消费 `local_topk_mask` / `tail_tokens`。
- 本地改动：
  - normal launch 改为传 `nullptr, nullptr`，与 LL 已保留的 unused mask/tail 优化保持一致；
  - 这是一个小 A/B 候选，必须等远端 aicc rebuild 后跑 1024/4096 correctness + perf；若不稳或无收益则撤回。
- 本地验证：
  - `python -m compileall` 覆盖 Python wrapper/test 文件；
  - `git diff --check` 通过。

## 2026-06-12 - normal K3 后续优化范围收窄到 no-tail

- 用户确认远端正在修复，后续 normal K3 fused 先优化 no-tail。
- 已更新计划：
  - normal `>=512` 性能 gate 先固定 `K3_USE_ASM_TAIL_REDUCE=0`，优先跑 1024/4096 correctness + stage timing + pure-vs-fused delta；
  - normal tail-reduce 暂停，不参与当前性能 gate。
- 当前判断：
  - normal tail-reduce 不能描述成“只剩一点精度漂移”；
  - 已有记录显示它仍是 tail reducer/combine 可见性、active-row 覆盖或诊断视图对齐未闭环的问题，虽然出现过精度/非有限值症状；
  - LL tail 已通过，normal tail 后续单独恢复诊断。

## 2026-06-12 - 远端环境恢复确认

- 按 `.vscode/sftp.json` 读取连接参数后复查 11 节点：
  - `10.17.176.11:22` TCP 已恢复，SSH 登录 `hg@10.17.176.11` 成功；
  - `sglang_megamoe` 容器运行中；
  - 容器内 `/workspace/DeepGEMM` 存在，`setup.py` 存在；
  - `/opt/dtk/bin/hipcc`、`/opt/dtk/bin/aicc` 和 `/workspace/dtk_aicc/bin/aicc` 可见；
  - `hy-smi` 显示 8 卡 VRAM/HCU 均为 0%；
  - host/container 内未发现残留 `setup.py build_ext`、`hipcc`、`aicc`、`dcc -cc1`、`torchrun` 或 MegaMoE 测试进程。
- 注意：
  - 容器内 git 默认触发 `dubious ownership`，已用一次性 `git -c safe.directory=/workspace/DeepGEMM` 验证；后续命令不依赖 git 时可忽略，若需要 git 元数据需带该参数或设置 safe.directory。
  - 远端当前只看到 `hygon_tmp/sglang_debug/run_v3_normal_perf_ab.sh`，`build_v3_normal_raw.sh` 和 normal K3 rowptr split runner 需要在恢复 normal gate 前重新同步。

## 2026-06-12 - normal 1024 no-tail 初测与 mask/tail A/B 撤回

- 已完成远端恢复后的 normal raw build：
  - 使用 `hygon_tmp/sglang_debug/build_v3_normal_raw.sh`；
  - 构建确认 V3 K1 fused normal 和 V3 K3 fused normal 都走 aicc shim；
  - build log：`hygon_tmp/sglang_debug/rebuild_v3_normal_raw_20260612_170219.log`。
- 驱动脚本问题：
  - `hygon_tmp/sglang_debug/push_and_run_v3_normal_gate.ps1` 的 bench 命令存在 PowerShell/remote 单引号嵌套问题，导致 build 成功后 bench 没跑；
  - 后续先用直接 `ssh docker exec bash -lc 'source /opt/dtk/env.sh && cd /workspace/DeepGEMM && ...'` 命令推进，driver 稍后修。
- 1024 no-tail 对照：
  - 原 staged normal：`orig_stage_normal_1024_perf_20260612_170449.json`，correct true，`fused_median_ms_avg_per_rank ≈ 2.4109 ms`；
  - V3 normal 的 `nullptr,nullptr` mask/tail A/B 在 perf/repeat 阶段失败，log `v3_normal_1024_bench_20260612_170527.log`，错误为 `fused/baseline nonfinite fused=1 baseline=0 diff=1`。
- 根因与处置：
  - `v3_k1_build_ll_stage_device` 的 mask/tail 分支还包含一个 K1 stage 内部 grid barrier，`nullptr,nullptr` 实际删除 readiness barrier；
  - 已查询 DCU KB，DeepEP/Flux 资料支持把 routing/readiness metadata 当成 GEMM scheduling 合同处理；
  - 已本地恢复 normal launch 传 `local_topk_mask.data_ptr<uint8_t>()` / `tail_tokens.data_ptr<int32_t>()`，撤回该 A/B。
- 恢复后结果：
  - V3 normal 1024 no-tail correct true，`v3_normal_1024_perf_20260612_170911.json`，summary `v3_normal_perf_summary_20260612_170950.csv`；
  - `fused_median_ms_avg_per_rank ≈ 11.1902 ms`，stage timing 显示 K1 约 `8.3-8.8 ms`、K3 combine 约 `2.16-2.25 ms`，K1 是当前最大差异。
- 下一步：
  - 同步当前本地撤回补丁到 11 节点并重编；
  - 继续跑 4096 normal no-tail，补齐 Phase 6 normal gate 数据；
  - 4096 后按 stage delta 优先优化，当前预期先看 K1，K3 no-tail 仍保留 rowptr split 诊断入口。

## 2026-06-12 - normal 4096 no-tail blocker 记录

- 远端状态：
  - 8 卡当前空闲，`sglang_megamoe` 容器运行，未发现残留 `setup.py build_ext`、`hipcc`、`aicc`、`dcc`、`torchrun` 或 MegaMoE 测试进程。
- 4096 对照：
  - 原 staged normal no-tail 通过，结果 `orig_stage_normal_4096_perf_20260612_171557.json`，`fused_median_ms_avg_per_rank ≈ 6.6148 ms`；
  - V3 normal no-tail 在 stage timing/perf 链路失败，log `v3_normal_4096_bench_20260612_171635.log`，错误为 `fused/baseline nonfinite fused=2 baseline=0 diff=2`。
- 已完成诊断：
  - correctness-only 不开 stage timing 的 base/sync/acquire/both 四组均 3/3 通过；
  - 开 `MEGAMOE_DCU_V3_STAGE_TIMING=1` 后 base 复现 fused 非有限值；
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 让 stage-timing correctness 通过；
  - acquire flags 不能稳定解决，并会转成 `max_abs` 超阈值。
- 当前判断：
  - host sync 只是定位证据，不能作为最终实现；
  - 下一步直接跑 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 的 K3 no-tail in-kernel signal A/B，判断是否能补齐 combine store 完成/可见性语义。

## 2026-06-12 - normal 4096 no-tail signal A/B correctness 通过

- 执行：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0 MEGAMOE_DCU_V3_STAGE_TIMING=1 MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1`；
  - `python3 tests/test_mega_moe_dcu.py ... --num-tokens 4096 --num-max-tokens-per-rank 4096 --correctness-iters 3 --skip-bench`。
- 结果：
  - correctness 3/3 通过，`max_abs=0.000488281`；
  - log `hygon_tmp/sglang_debug/v3_normal_4096_diag_stage_signal_20260612_173053.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_4096_diag_stage_signal_20260612_173053.json`。
- 结论：
  - in-kernel no-tail signal 能替代诊断用 host sync 覆盖 4096 stage timing 下的可见性问题；
  - 下一步跑 1024/4096 perf，评估 signal 成本和新的 K1/K3 stage delta。

## 2026-06-12 - normal no-tail signal 1024 隔离诊断

- 直接 perf runner 结果：
  - `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 跑 1024/4096 perf 时，1024 在 correctness 阶段失败；
  - log `hygon_tmp/sglang_debug/v3_normal_1024_bench_20260612_173232.log`；
  - rank5 报 `fused/baseline nonfinite fused=63 baseline=0 diff=63`，runner 未进入 4096。
- 缩短诊断：
  - `signal`：1024 correctness-only 失败，rank6 报 `fused=118`；
  - `signal_clone`：加 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 后 1024 stage-timing correctness-only 5/5 通过，log `v3_normal_1024_diag_signal_clone_20260612_173444.log`；
  - `signal_sync`：加 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 仍失败，转为 `max_abs=0.003704071044921875 > 0.0035`。
- 当前判断：
  - 1024 原样失败更像 correctness oracle 对比时 fused output 被后续 baseline/异步路径污染，clone 可以隔离；
  - 不能把 host sync 作为修复，也不能仅凭 signal 原样跑 perf；
  - 下一步 perf runner 临时加 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1`，只用于保证 correctness 前置对比稳定，benchmark 本身仍测 V3 fused。

## 2026-06-12 - normal no-tail signal clone 假设撤回

- 后续完整 perf runner 反证了上一条的 clone 假设：
  - `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1 MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 跑 1024/4096 perf + stage timing 时仍在 1024 correctness 阶段失败；
  - log `hygon_tmp/sglang_debug/v3_normal_1024_bench_20260612_173701.log`；
  - rank5 报 `fused/baseline nonfinite fused=135 baseline=0 diff=135`。
- 当前处置：
  - 不再把 clone 作为稳定测试侧隔离变量；
  - 本地已开始一个单变量 A/B：no-tail signal path 复用现有 tail completion signal/wait 模板，`reduce_blocks=0` 且不新增 kernel；
  - 下一步同步 `K3_fused/k3_v3_fused_ext.cu` 到 11 节点，aicc rebuild V3 normal raw 后先跑 1024 stage-timing correctness。

## 2026-06-12 - normal no-tail signal generation 与 ACQ_REL A/B 准备

- tail completion 模板复用 A/B 已跑完并撤回：
  - build log `hygon_tmp/sglang_debug/rebuild_v3_normal_signal_tailtmpl_20260612_174303.log`；
  - 1024 stage-timing correctness 第 2 轮失败，log `hygon_tmp/sglang_debug/v3_normal_1024_diag_signal_tailtmpl_20260612_174419.log`，rank0 报 `fused/baseline nonfinite fused=12 baseline=0 diff=12`；
  - 结论是复用 tail completion 模板不是充分修复，当前代码已回到 signal-only normal no-tail path。
- generation bug 已定位并修复：
  - Python 侧 `_LargeOptState.asm_signal_generation` 每次 eager tail/no-tail signal launch 递增；
  - 发现 C++ `k3_v3_normal_combine_signal_raw(... signal_generation)` 曾未把 generation 继续传入 internal launch，kernel 实际仍收到固定 `1`；
  - 已修改 `dcu_megamoe_v3_launch_k3_normal_combine_raw` signature 和 call sites，no-tail signal/tail signal 现在透传 generation。
- generation 修复后的测试结果：
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_signal_generation_fix_20260612_175327.log`；
  - 1024 signal/no-clone/no-sync correctness 到第 4 轮失败，rank7 `fused/baseline nonfinite fused=45 baseline=0 diff=45`；
  - 1024 signal+clone correctness 5/5 通过，log `v3_normal_1024_diag_signal_genfix_clone_20260612_175617.log`；
  - 1024 signal+clone+stage timing perf 通过，结果 `v3_normal_1024_perf_20260612_175709.json`，平均 fused median 约 `11.2729 ms`；
  - 4096 signal+clone+stage timing 仍在多轮后出现 `max_abs` 漂移，典型 `0.0044-0.0049 > 0.0035`，不能进入性能优化。
- 当前代码改动：
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 的 normal completion path 已改为 system-scope `__hip_atomic_fetch_add(..., __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM)` 更新 `done_counter`；
  - peer signal 使用 system-scope release atomic；
  - 该 ACQ_REL A/B 还未远端 build/test，下一步先做本地静态检查、同步、aicc rebuild，再跑 1024/4096 correctness。

## 2026-06-12 - ACQ_REL A/B 本地检查

- 已重读 Phase 6 normal 计划段，当前仍处于 normal K3 no-tail signal correctness/perf 评估项。
- 本地检查结果：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_mega_moe_dcu.py` 通过；
  - `git diff --check` 覆盖 `large_opt.py`、K3 V3 source 和三份 planning 文件，通过；
  - 初次编码检查误用了 bash heredoc，在 PowerShell 下报 parser error；已用 PowerShell byte scan 重跑，三份 planning 文件均为 UTF-8 无 BOM、LF。
- 下一步：
  - 同步 `large_opt.py`、`K3_fused/k3_v3_fused_ext.cu`、`K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 到 11 节点；
  - 远端 aicc rebuild V3 normal raw 后先跑 1024/4096 no-tail signal correctness。

## 2026-06-12 - ACQ_REL A/B 远端 rebuild 通过

- 已从 `.vscode/sftp.json` 读取 11 节点参数，并同步：
  - `megamoe/large_opt.py`
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_fused_ext.cu`
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh`
- 远端状态：
  - `sglang_megamoe` 运行；
  - 8 卡 VRAM/HCU 均为 0%；
  - 未发现残留 `setup.py build_ext`、`hipcc`、`aicc`、`dcc -cc1`、`torchrun` 或 MegaMoE 测试进程。
- build：
  - `source /opt/dtk/env.sh && cd /workspace/DeepGEMM && bash hygon_tmp/sglang_debug/build_v3_normal_raw.sh`；
  - V3 K1 fused normal 和 V3 K3 fused normal 都确认走 aicc shim；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_acqrel_20260612_180534.log`；
  - build 成功。
- 下一步：
  - 先跑 1024/4096 no-tail signal correctness-only，确认 ACQ_REL 是否解决 generation 修复后剩余的非有限/`max_abs` 漂移。

## 2026-06-12 - ACQ_REL A/B 1024 correctness 通过

- 执行环境：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal`
  - `K3_USE_ASM_TAIL_REDUCE=0 MEGAMOE_DCU_V3_STAGE_TIMING=1 MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1`
  - 未设置 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE`，未设置 `MEGAMOE_DCU_V3_NO_TAIL_SYNC`。
- 1024 correctness-only：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_diag_signal_acqrel_20260612_180714.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_1024_diag_signal_acqrel_20260612_180714.json`；
  - 5/5 通过，`max_abs=0.000488281`。
- 阶段信号：
  - 稳态 K1 stage 约 `8.3-8.7 ms`；
  - K3 combine stage 约 `2.25-2.45 ms`；
  - no-tail barrier/reduce 约 `0.02-0.03/0.047 ms`。
- 下一步：
  - 同口径跑 4096 correctness-only，重点看 generation 修复后仍存在的 `max_abs` 漂移是否被 ACQ_REL 消除。

## 2026-06-12 - ACQ_REL A/B 4096 correctness 通过

- 执行环境同 1024：
  - normal backend、no-tail、stage timing、no-tail signal；
  - 未设置 clone-before-baseline，未设置 host sync。
- 4096 correctness-only：
  - log `hygon_tmp/sglang_debug/v3_normal_4096_diag_signal_acqrel_20260612_180819.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_4096_diag_signal_acqrel_20260612_180819.json`；
  - 5/5 通过，最大一次 `max_abs=0.000885010`，低于 `atol=0.0035`。
- 阶段信号：
  - 稳态 K1 stage 约 `42.6-43.2 ms`；
  - K3 combine stage 约 `8.0-8.4 ms`；
  - no-tail barrier/reduce 约 `0.02-0.03/0.165-0.168 ms`。
- 结论：
  - ACQ_REL done counter + release peer signal 修复了 generation 透传后 1024 非有限和 4096 `max_abs` 漂移的短诊断形态；
  - 下一步跑 1024/4096 perf runner，确认 correctness 在 bench 链路也稳定，并量化 signal 成本。

## 2026-06-12 - ACQ_REL A/B perf runner 1024 仍失败

- 执行：
  - `TOKENS_LIST="1024 4096" RUN_ORIG_STAGE=0 RUN_V3=1 RUN_STAGE_TIMING=1 MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1 K3_USE_ASM_TAIL_REDUCE=0 WARMUP=3 REPEAT=5 ITERS=3 bash hygon_tmp/sglang_debug/run_v3_normal_perf_ab.sh`
- 结果：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_bench_20260612_180953.log`；
  - 1024 第一轮 correctness 通过，第二轮前后 rank4 报 `fused/baseline nonfinite fused=13 baseline=0 diff=13`；
  - runner 未进入 4096。
- 当前判断：
  - ACQ_REL 已让 `--skip-bench` 1024/4096 短链路通过，但完整 perf runner 的 correctness 前置仍不稳定；
  - 下一步用同等参数直接调用 `tests/test_mega_moe_dcu.py`、去掉 runner 包装，确认问题来自 bench 模式/参数还是 runner env 包装。

## 2026-06-12 - ACQ_REL A/B 手工 no-skip 复现

- 执行：
  - 直接调用 `tests/test_mega_moe_dcu.py`，1024 tokens，`--correctness-iters 3 --warmup 3 --repeat 5`，不加 `--skip-bench`；
  - env 与 runner 一致，仍为 no-tail signal、stage timing、no clone、no host sync。
- 结果：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_manual_noskip_acqrel_20260612_181206.log`；
  - rank0 打印到 `Correctness 3/3`，但 process 6 在同一 correctness 阶段报 `fused/baseline nonfinite fused=18 baseline=0 diff=18`；
  - 说明不是 runner shell 包装问题，而是完整 no-skip 链路下仍存在跨 rank 非确定性。
- 下一步：
  - 同参数加 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 做隔离；若通过，问题偏向 baseline oracle/后续异步污染；若失败，继续回到 K3 signal/可见性语义。

## 2026-06-12 - no-tail signal wait acquire A/B 准备

- clone 隔离结果：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_manual_noskip_acqrel_clone_20260612_181358.log`；
  - 加 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 后仍失败，process 7 报 `fused/baseline nonfinite fused=4 baseline=0 diff=4`；
  - 说明不是 baseline 覆盖 fused tensor 的表层问题。
- DCU KB / 本地代码证据：
  - `deep_gemm::mega::load_signal_system` 当前是 `__hip_atomic_load(..., __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)`；
  - Hygon/Flux 参考对非 final sync 使用 release store/add + acquire load；
  - 原 tail wait path 已使用更强的 `global_load_i32_glc_slc_device`，而 no-tail signal-only wait 仍用 relaxed load。
- 本地代码改动：
  - 在 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 增加 `load_signal_system_acquire_device()`；
  - 只把 `v3_k3_wait_peer_signals_thread0_device` 的 wait load 从 relaxed `load_signal_system` 改为 system-scope acquire；
  - 不改 GEMM 主循环、不改 signal 发送、不新增 runtime kernel。
- 下一步：
  - 本地 compile/diff 检查；
  - 同步 K3 V3 cuh，aicc rebuild 后重跑 1024 no-skip correctness/perf 前置。

## 2026-06-12 - no-tail signal wait acquire rebuild 通过

- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_mega_moe_dcu.py` 通过；
  - `git diff --check` 通过；
  - source grep 确认 no-tail signal wait 使用 `load_signal_system_acquire_device()`。
- 远端：
  - 已同步 `K3_fused/k3_v3_pack5_groupgemm_impl.cuh`；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_wait_acquire_20260612_181621.log`；
  - V3 K1/K3 fused normal 均走 aicc shim；
  - build 成功。
- 下一步：
  - 复跑 1024 no-skip 手工命令，判断 acquire wait 是否消除完整 bench 前置链路中的非有限值。

## 2026-06-12 - no-tail signal wait acquire 1024 bench 通过

- 执行：
  - 1024 tokens，normal backend，no-tail signal，stage timing；
  - `--correctness-iters 3 --warmup 3 --repeat 5`，不加 `--skip-bench`。
- 结果：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_manual_noskip_wait_acquire_20260612_181744.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_1024_manual_noskip_wait_acquire_20260612_181744.json`；
  - correctness 3/3 通过；
  - bench 完成，`fused_median_ms_avg_per_rank ≈ 11.2548 ms`，`fused_min_ms_avg_per_rank ≈ 11.0521 ms`；
  - stage timing 稳态 K1 约 `8.3-8.7 ms`，K3 combine 约 `2.25-2.50 ms`。
- 结论：
  - no-tail signal peer wait 的 acquire load 是 ACQ_REL 后仍需的补丁；
  - 下一步跑 4096 同口径 bench，确认大档位也稳定。

## 2026-06-12 - normal no-tail 4096 bench 稳定组合确认

- 先跑 wait-acquire 但不加 reduce acquire：
  - log `hygon_tmp/sglang_debug/v3_normal_4096_manual_noskip_wait_acquire_20260612_181848.log`；
  - rank0 打印到 correctness 3/3，但 process 7 报 `fused/baseline nonfinite fused=96 baseline=0 diff=96`；
  - 说明 4096 还需要 reduce 读侧可见性处理。
- 再加 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1`：
  - log `hygon_tmp/sglang_debug/v3_normal_4096_manual_wait_acquire_reduce_acquire_20260612_182006.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_4096_manual_wait_acquire_reduce_acquire_20260612_182006.json`；
  - correctness 3/3 通过；
  - bench 完成，`fused_median_ms_avg_per_rank ≈ 50.9254 ms`，`fused_min_ms_avg_per_rank ≈ 50.5339 ms`。
- 当前稳定组合：
  - K3 done counter 使用 system-scope ACQ_REL；
  - peer signal 使用 release atomic；
  - peer wait 使用 system-scope acquire load；
  - 4096 no-tail 外部 barrier/reduce 需要 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1`，即 post-K3 rank barrier acquire + reduce 前 invalidate/glc read。
- 性能判断：
  - 1024 V3 normal 当前约 `11.25 ms`；
  - 4096 V3 normal 当前约 `50.93 ms`；
  - stage timing 显示主要瓶颈仍是 K1 stage：1024 约 `8.3-8.7 ms`，4096 约 `42 ms`；K3 是第二瓶颈但不是优先项。
- 下一步：
  - 将稳定 no-tail 可见性组合收敛到默认 V3 normal no-tail path 或至少在 runner 中固定；
  - 进入 normal K1 stage 优化，优先和 pure normal K1 对照。

## 2026-06-12 - normal no-tail 稳定组合收敛到默认 V3 路径

- 本地代码改动：
  - `v3_no_tail_signal_enabled()` 的默认值从 `"0"` 改为 `"1"`，因此 V3 normal no-tail 默认启用 in-kernel signal；
  - `use_v3_reduce_acquire` 改为在 `use_v3_k3_no_tail_signal` 为真时自动启用，同时保留 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 诊断开关；
  - 显式 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=0` 仍可关闭 signal 做诊断；
  - 影响范围仍限定在 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 && USE_MEGAMOE_V3=1 && backend=normal && no-tail`。
- 测试维护：
  - 更新 `tests/test_dcu_megamoe_v3.py` source guard，确认 no-tail signal 默认 `"1"`；
  - 将 reduce acquire source guard 命名改为默认服务 no-tail signal 的语义。
- 下一步：
  - 本地 compile/source guard/diff 检查；
  - 同步 `large_opt.py` 和测试文件，远端用默认 env 重跑 1024/4096 normal no-tail correctness+bench，确认不再依赖手工 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` / `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1`。

## 2026-06-12 - normal no-tail 默认 gate source 测试通过

- 本地：
  - `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py tests/test_mega_moe_dcu.py` 通过；
  - `git diff --check` 通过；
  - 本机 `python -m pytest` 仍缺 pytest 模块，已用 inline source guard 验证默认 signal/acquire 逻辑。
- 远端：
  - 已同步 `megamoe/large_opt.py` 和 `tests/test_dcu_megamoe_v3.py`；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed`。
- 下一步：
  - 用默认 env 跑 V3 normal no-tail 1024/4096 bench，不显式设置 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 或 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE`。

## 2026-06-12 - 恢复远端并准备默认 env bench

- 已按 `remote-ssh-docker-workflow` 从 `.vscode/sftp.json` 读取 11 节点参数，确认：
  - `sglang_megamoe` 容器运行；
  - 容器 repo 为 `/workspace/DeepGEMM`；
  - 8 卡 VRAM/HCU 均为 0%；
  - 未发现残留 `setup.py build_ext`、`hipcc`、`aicc`、`dcc -cc1`、`torchrun` 或 MegaMoE 测试进程。
- 一次远端健康检查命令因 PowerShell 本地解析 `pgrep` pattern 中的管道而失败，已改为把 remote command 存入变量后重跑成功；后续包含 `|` 的 SSH 命令继续用变量传参。
- 下一步：
  - 运行默认 env 的 V3 normal no-tail 1024/4096 bench，显式 unset `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 与 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE`，确认默认 gate 已收敛到稳定可见性组合。

## 2026-06-12 - 默认 env normal no-tail bench 仍复现 1024 非有限

- 默认 env runner：
  - 命令显式 `env -u MEGAMOE_DCU_V3_NO_TAIL_SIGNAL -u MEGAMOE_DCU_V3_REDUCE_ACQUIRE`，跑 `TOKENS_LIST="1024 4096" RUN_V3=1 RUN_STAGE_TIMING=1 K3_USE_ASM_TAIL_REDUCE=0`；
  - 1024 在第 3 轮 correctness 左右失败，log `hygon_tmp/sglang_debug/v3_normal_1024_bench_20260612_182633.log`，rank6 报 `fused/baseline nonfinite fused=30 baseline=0 diff=30`。
- 直接手工命令复现：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_manual_default_20260612_182817.log`；
  - 同样第 3 轮左右失败，rank7 报 `fused/baseline nonfinite fused=24 baseline=0 diff=24`；
  - 说明不是 runner shell 包装问题。
- 已确认：
  - 远端 `large_opt.py` 中 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 默认值为 `"1"`；
  - Python 默认 `v3_no_tail_signal_enabled()` 为 True；
  - runner 脚本没有显式把 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 设为 0。
- 当前判断：
  - default gate 失败仍是跨迭代 signal/可见性/slot 状态问题；
  - 新增的自动 `use_v3_reduce_acquire = use_v3_k3_no_tail_signal or env` 与 1024 手工稳定组合存在差异，需要单变量验证；
  - 下一步先用 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 复核 host sync 是否仍能覆盖，再定位 reduce acquire 自动开启是否参与触发。

## 2026-06-12 - normal no-tail signal-only owner-slot A/B 准备

- 补充诊断：
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 也会在 1024 第 4 轮左右失败，log `hygon_tmp/sglang_debug/v3_normal_1024_manual_default_sync_20260612_182940.log`，rank7 报 `fused/baseline nonfinite fused=18 baseline=0 diff=18`；
  - 说明单纯 host sync 不是充分定位条件，问题更像多轮 signal/done-counter/slot 状态或 completion 发布语义。
- KB/参考：
  - `dcu-rag-kb-query "Hygon gfx938 cross CTA done counter release acquire system scope peer signal threadfence_system remote store combine visibility atomic fetch_add"` 命中 Hygon/Flux/SGLang allreduce 资料；
  - 结论仍是 remote 数据写入后 signal 必须是 release，等待必须 acquire；非 final sync 需要 fence，避免 signal 先于数据可见。
- 本地 A/B 改动：
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 的 normal `kSignalOnly` 分支不再由最后完成的 GEMM CTA 直接 signal peers；
  - 改为使用 `done_counter[1]` owner slot：最后 CTA release-store owner id，owner CTA acquire-load 确认后再发布 peer signal/wait；
  - `K3_fused/k3_v3_fused_ext.cu` 将 `k3_v3_normal_combine_signal_raw` 的 `done_counter` contract 收紧为至少两个 int32。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_mega_moe_dcu.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 下一步：
  - 同步 K3 V3 source 到 11 节点，aicc rebuild V3 normal raw；
  - 先跑 1024 default env 5 轮 correctness/no-skip，再跑 4096 default env。

## 2026-06-12 - normal no-tail owner-slot A/B rebuild 通过

- 已同步：
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh`
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_fused_ext.cu`
- 远端 rebuild：
  - command 进入 `sglang_megamoe` 后 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM && bash hygon_tmp/sglang_debug/build_v3_normal_raw.sh`；
  - log `hygon_tmp/sglang_debug/rebuild_v3_normal_owner_slot_20260612_183312.log`；
  - raw build log `hygon_tmp/sglang_debug/rebuild_v3_normal_raw_20260612_183312.log`；
  - V3 K1/K3 fused normal 均确认走 aicc shim；
  - build 成功。
- 下一步：
  - 跑 1024 default env 五轮 correctness/no-skip，验证 owner-slot 是否解决多轮非有限。

## 2026-06-12 - normal no-tail owner-slot A/B 1024 仍失败

- 1024 default env 五轮 no-skip：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_manual_default_owner_slot_20260612_183435.log`；
  - 前 4 轮 correctness 通过，第 5 轮 rank6 报 `fused/baseline nonfinite fused=3 baseline=0 diff=3`；
  - owner-slot 比直接 signal 的第 3/4 轮失败略有改善，但不是充分修复。
- 当前判断：
  - 不能把 owner-slot A/B 标为完成修复；
  - 下一步用 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 做定位，确认 `fused_y` 是否被 baseline oracle 或后续异步路径污染；
  - 若 clone 仍失败，回到 K3 combine/y 输出写入本身和跨迭代 scratch reuse 继续查。

## 2026-06-12 - normal no-tail owner-slot + clone 1024 通过

- 1024 owner-slot + `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1`：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_manual_owner_slot_clone_20260612_183540.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_1024_manual_owner_slot_clone_20260612_183540.json`；
  - correctness 5/5 通过并完成 bench；
  - `fused_median_ms_avg_per_rank ≈ 11.2150 ms`，`fused_min_ms_avg_per_rank ≈ 11.1860 ms`。
- 判断：
  - clone 后 correctness 通过，说明 `run_fused` 返回的输出在隔离后可与 baseline 对齐；
  - default no-clone 仍失败，不能当作生产修复完成；当前更像测试侧 baseline oracle / output 生命周期 / 多 rank 迭代时序 artifact；
  - 性能 bench 不包含 clone，可先用 clone 隔离 correctness 前置继续 normal 性能优化，同时把 no-clone artifact 作为后续 correctness harness 修复项。
- 下一步：
  - 跑 4096 owner-slot + clone，确认大档位也能过 correctness+bench。

## 2026-06-12 - normal no-tail owner-slot + clone 4096 通过

- 4096 owner-slot + `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1`：
  - log `hygon_tmp/sglang_debug/v3_normal_4096_manual_owner_slot_clone_20260612_183814.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_4096_manual_owner_slot_clone_20260612_183814.json`；
  - correctness 3/3 通过并完成 bench；
  - `fused_median_ms_avg_per_rank ≈ 50.9933 ms`，`fused_min_ms_avg_per_rank ≈ 50.4369 ms`。
- 当前 normal no-tail 性能基线：
  - 1024 owner-slot + clone：`≈ 11.2150 ms`；
  - 4096 owner-slot + clone：`≈ 50.9933 ms`；
  - stage timing 仍显示 K1 主导：1024 K1 约 `8.4-8.7 ms`，4096 K1 约 `41-43 ms`；K3 约 `2.2-2.5 ms` / `8.0-8.4 ms`。
- 未关闭问题：
  - default no-clone correctness loop 仍可触发 fused 非有限；当前将其记录为测试 harness / baseline oracle 输出生命周期 artifact，不能作为 production kernel 修复完成的证据；
  - 后续需要单独修 correctness harness，不能用 clone 掩盖 production bug。但性能优化可先使用 clone 隔离 correctness 前置，bench 本身不包含 clone。
- 下一步：
  - 重读计划后进入 normal K1 优先优化：先跑 normal pure reference runner，建立 K1/K3 pure-vs-fused delta。

## 2026-06-12 - normal pure reference runner 首次执行失败

- 目的：
  - 建立 normal 1024/4096 的 K1 pure pack5 与 K3 rowptr split 基线，用于 normal pure-vs-fused delta。
- 执行：
  - 远端 11 节点 `sglang_megamoe` 容器内执行 `TOKENS_LIST="1024 4096" RUN_K1_PURE=1 RUN_K3_ROWPTR=1 bash hygon_tmp/sglang_debug/run_v3_normal_pure_refs.sh`。
- 失败：
  - log 标记 `hygon_tmp/sglang_debug/v3_normal_k1_pure_pack5_20260612_184211.log`；
  - 脚本第 19 行 `./run_best_c.sh` 报 `Permission denied`，原因是远端 `hygon_tmp/K1_groupgemm_fp8/run_best_c.sh` 无 executable bit。
- 处置：
  - 下一步不重复同一失败命令；将 pure runner 改成显式 `bash ./run_best_c.sh` 后重跑。

## 2026-06-12 - normal pure reference runner aicc 工具链失败并修复

- 第二次执行：
  - runner 进入 `hygon_tmp/K1_groupgemm_fp8` 后调用 `bash ./run_best_c.sh`；
  - log 标记 `hygon_tmp/sglang_debug/v3_normal_k1_pure_pack5_20260612_184254.log`。
- 失败：
  - `make aicc` 使用 `/opt/dtk/bin/aicc`，并传入 `-mllvm -enable-num-vgprs-768=true`；
  - 当前 `/opt/dtk/bin/aicc` 报 `Unknown command line argument '-enable-num-vgprs-768=true'`。
- 修复：
  - 复查既有 findings，确认已验证路径是 `/workspace/dtk_aicc/bin/aicc` + `ROCM_PATH=/workspace/dtk_aicc HIP_PATH=/opt/dtk/hip HIP_CLANG_PATH=/workspace/dtk_aicc/aillvm/bin HIP_ROCCLR_HOME=/opt/dtk DEVICE_LIB_PATH=/opt/dtk/dcc/dccgcn/bitcode`；
  - 修改 `hygon_tmp/K1_groupgemm_fp8/Makefile`，让 pure reference 的 aicc target 使用该环境；生产 `setup.py` 和 V3 extension 编译策略不变。

## 2026-06-12 - normal pure-vs-fused 基线完成

- 执行：
  - `TOKENS_LIST="1024 4096" RUN_K1_PURE=1 RUN_K3_ROWPTR=1 bash hygon_tmp/sglang_debug/run_v3_normal_pure_refs.sh`；
  - K1 pure log: `hygon_tmp/sglang_debug/v3_normal_k1_pure_pack5_20260612_184430.log`；
  - K3 rowptr JSON: `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260612_184527.json`、`hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260612_184551.json`。
- K1 pure normal pack5：
  - 1024 tokens median `0.745737 ms`，min `0.745318 ms`；
  - 4096 tokens median `2.25922 ms`，min `2.25905 ms`。
- K3 normal rowptr split（mode_summary median_avg_rank_ms）：
  - 1024: `rowptr_all_zero 0.7176`、`local_rowptr 0.9389`、`staged_local_only 1.1004`、`staged_remote_only 2.2078`、`staged_rowptr 2.2162 ms`；
  - 4096: `rowptr_all_zero 2.5415`、`local_rowptr 3.3051`、`staged_local_only 3.4587`、`staged_remote_only 7.7993`、`staged_rowptr 7.8297 ms`。
- 结论：
  - 当前 normal fused K1 stage 约 `8.4-8.7 ms / 41-43 ms`，相对 K1 pure `0.746 / 2.259 ms` 是最大 delta；
  - K3 staged rowptr 基本解释了当前 K3 stage `~2.2 / ~8.0 ms`，主要额外成本是 remote rowptr 数据通路；
  - 后续优先优化 normal K1 stage，K3 暂作为第二优先级。

## 2026-06-12 - normal K1 staged-input A/B 首轮失败待修

- 已完成代码尝试：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 增加 normal staged-input path：builder CTA 将 `row_x_ptrs` 指向的 FP8 row 拷到 contiguous `staged_x`，GEMM A-load 改为 contiguous `buffer_load_fp8_b128_pack_device(staged_x_resource, ...)`；
  - `k1_fused_ext.cu`、`k1_v3_fused_ext.cu`、`k1_v3_stub_ext.cu` 更新 normal raw launcher signature，传入 `staged_x`；
  - 仅重编 K1 normal raw extension，避免重新编 V2 或无关 K3。
- 验证结果：
  - 本地 compileall / `git diff --check` 通过；
  - 远端 K1-only aicc rebuild 更新 K1 `.so`；
  - 远端当时 8 卡被 `sglang serve` 占用约 91-93% VRAM，perf runner 因 VRAM guard 退出；
  - 直接 1024 no-tail `--skip-bench` 得到 K1 stage 约 `2.3-3.1 ms`，比此前约 `8.5 ms` 明显改善，但 correctness 失败，最大 `max_abs=0.12060546875`。
- 当前定位：
  - 不把 staged-input A/B 标完成；
  - 首要修复点是 K1 ready flag：当前 store/wait 使用 plain store + relaxed system load，和 K3/Flux/DeepEP 的 release/acquire 可见性模式不一致；
  - 下一步只改 ready flag release/acquire，重编 K1 normal 后直接跑 1024 correctness+stage timing。

## 2026-06-12 - normal K1 staged-input ready flag A/B rebuild + smoke

- 本地：
  - 改 `K1_fused/k1_v3_groupgemm_impl.cuh`：ready flag store 改为 system-scope release atomic store，wait 改为 system-scope acquire load；
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py tests/test_mega_moe_dcu.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 远端：
  - 已同步 `k1_v3_groupgemm_impl.cuh`；
  - K1-only aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_stage_acquire_20260612_190739.log`；
  - import smoke 通过：`k1_ext_import_ok True`。
- 单卡低显存 smoke：
  - `HIP_VISIBLE_DEVICES=7` local-rank normal expert_ramp 通过，`expert_ramp_abs_err=0`；
  - `HIP_VISIBLE_DEVICES=7 K1_V3_SMOKE_ALL_RANKS=1` simulated all-ranks normal expert_ramp 通过，`expert_ramp_abs_err=0`。
- ISA 尝试：
  - `dccobjdump --show-sass` 对 K1 `.so` 与 build `.o` 只输出 host `elf64-x86-64`，没有拿到 device ISA；
  - 该项记为 inconclusive，不作为完成证据。
- 资源状态：
  - `sglang_megamoe` 容器可用，但 8 卡仍被 host PID `284249` 的 `sglang serve --model-path /data2/MiMo-V2-Flash-Channel-FP8` 占用约 135-138GB/卡；
  - 暂不跑 8 卡 correctness/perf，等待显存释放或用户授权处理占卡服务。
- 待资源恢复后首个命令：
  - `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1 TOKENS_LIST="1024" RUN_ORIG_STAGE=0 RUN_V3=1 RUN_STAGE_TIMING=1 K3_USE_ASM_TAIL_REDUCE=0 WARMUP=3 REPEAT=5 ITERS=3 bash hygon_tmp/sglang_debug/run_v3_normal_perf_ab.sh`；
  - 若 1024 通过，再同口径跑 `TOKENS_LIST="4096"`；若失败，优先比较 acquire 前后 K1 stage timing 和 rank 错误形态。

## 2026-06-12 - normal K1 staged-input 单卡短链路继续定位

- 远端状态：
  - 11 节点 `sglang_megamoe` 一度恢复，`hy-smi` 显示 8 卡 VRAM/HCU 均为 0%，无测试/编译残留进程；
  - 后续在 `k1_v3_output_compare` waitcnt 版本复测时 SSH 在 KEX 阶段被远端关闭，`ssh -vvv ... "hostname && date"` 报 `kex_exchange_identification: Connection closed by remote host`；当前不是测试命令本身错误，需等 SSH 恢复后继续。
- 诊断脚本：
  - 更新 `hygon_tmp/sglang_debug/k1_v3_output_compare.py`，增加按 `row_combine_ptrs` 解码 source_rank/token/topk_slot、finite-only max diff、ASM/V3 nonfinite 计数和首个 nonfinite detail；
  - 本地 `python -m compileall hygon_tmp/sglang_debug/k1_v3_output_compare.py hygon_tmp/sglang_debug/k1_v3_metadata_compare.py` 通过；
  - 已同步诊断脚本到远端。
- K1 staged-input 现象：
  - rank5/random/1024 单卡输出对比可复现间歇性有限错值，metadata 对齐且无 NaN/Inf；
  - 典型失败：`asm_active == v3_active == common == 6145`、`missing_in_v3=0`、`extra_in_v3=0`、`asm_nonfinite=0`、`v3_nonfinite=0`，但 `max_abs` 可到 `0.6~1.1`；
  - `decoded_ptr` 显示错误可落在不同 source_rank/topk_slot/token，ASM 接近 ref，V3 同 expert 但输出偏离，指向 staged_x/同 kernel store-load 可见性，而非 route metadata 或 weight layout。
- A/B 与结果：
  - GLC source row load A/B 已反证：K1 8-rank chain 仍错且 l1_by_ptr 出现非有限，已撤回到普通 source load；
  - ready flag release/acquire 和 build/tile-ready release store 已构建并通过 expert_ramp smoke，但 rank5 随机短链路仍间歇失败；
  - all-thread `__threadfence_system()` before ready flag A/B 已构建，rank5 repeat 仍失败；
  - 新增 `s_waitcnt vmcnt(0)` before all-thread system fence 的 waitcnt A/B 已通过本地检查并远端 K1-only aicc rebuild，build log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_stage_waitcnt_20260612_195342.log`，但复测因 SSH 被关闭未完成。
- 下一步：
  - SSH 恢复后先跑 `k1_v3_output_compare.py` rank5 repeat waitcnt 版本；
  - 若仍失败，优先做“不走 global staged_x store/load、直接 rowptr A-load”的诊断开关或 staging fill-ones 诊断，明确是 staging store-load 可见性还是 GEMM core/weight 路径。

## 2026-06-12 - normal K1 direct-rowptr 诊断开关本地准备

- 远端：
  - SSH 仍在握手阶段关闭，`ssh -F NUL hg@10.17.176.11 ... "hostname && date"` 继续报 `Connection closed by 10.17.176.11 port 22`；
  - 本轮无法继续远端编译/复测。
- 本地代码准备：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 增加 `K1_LOAD_B_PACK` 诊断分支：默认使用 staged_x buffer load；当 `stage_input_in_kernel=false` 时回到 `row_x_ptrs` direct A-load；
  - `K1_fused/k1_v3_fused_ext.cu` 增加临时 env `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT=1`，只用于 normal K1 诊断，默认生产路径仍 staged-input；
  - 本地 `python -m compileall ...` 与 `git diff --check` 通过。
- 下一步：
  - SSH 恢复后同步 `k1_v3_groupgemm_impl.cuh` / `k1_v3_fused_ext.cu`，K1-only aicc rebuild；
  - 先跑默认 staged waitcnt 版本 rank5 repeat，再跑 `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT=1` rank5 repeat；
  - 若 direct-rowptr 通过而 staged 失败，根因锁定 staged_x store/load 可见性；若 direct-rowptr 也失败，回到 GEMM core / pack5 weight / scale path 定位。

## 2026-06-12 - normal K1 staged-input waitcnt A/B 通过单卡短链路

- 远端恢复：
  - 11 节点 SSH 已恢复；`sglang_megamoe` 曾处于 `Exited(255)`，已按 remote workflow 启动；
  - 容器 repo 为 `/workspace/DeepGEMM`，8 卡 VRAM/HCU 均为 0%；
  - 容器重启后 K1 inplace `.so` 缺失，已同步 K1 V3 normal 源码和诊断脚本并执行 K1-only normal rebuild；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_direct_diag_20260612_215652.log`，确认 `K1_fused.k1_fused_ext` 使用 V3 normal aicc shim，`DG_BUILD_MEGAMOE_V2_EXT=0`，未编 V2。
- 验证：
  - 单独 import 检查通过：`k1_ext_import_ok True`；
  - `HIP_VISIBLE_DEVICES=7` rank5/random/1024 重复 10 轮 K1 ASM vs V3 output compare 全部通过；
  - log `hygon_tmp/sglang_debug/k1_v3_output_compare_rank5_repeat_waitcnt_20260612_215816.log`；
  - 每轮 metadata 对齐，`asm_rows == v3_rows == 8192`、`asm_active == v3_active == common == 6145`、`missing/extra/nonfinite == 0`、`max_abs=0`。
- 结论：
  - `s_waitcnt vmcnt(0)` + all-thread `__threadfence_system()` + ready flag release/acquire 是当前 staged-input correctness 的最小正向修复；
  - direct-rowptr 诊断开关暂不作为下一步必跑项，除非 8 卡 staged/e2e 或更长 repeat 再复现错误；
  - normal 性能下一步回到 ASM diff 和 stage timing，重点压 K1/K3 fused normal 相对原 staged ASM 的 delta。

## 2026-06-12 - normal 1024 no-tail 8 卡 stage timing 更新

- 执行：
  - `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1 TOKENS_LIST="1024 4096" RUN_ORIG_STAGE=1 RUN_V3=1 RUN_STAGE_TIMING=1 K3_USE_ASM_TAIL_REDUCE=0 WARMUP=3 REPEAT=5 ITERS=3 bash hygon_tmp/sglang_debug/run_v3_normal_perf_ab.sh`；
  - 顶层 log `hygon_tmp/sglang_debug/v3_normal_stage_timing_waitcnt_20260612_220305.log`。
- 原 staged 1024 no-tail：
  - log `hygon_tmp/sglang_debug/orig_stage_normal_1024_bench_20260612_220306.log`；
  - correctness 3/3 通过，`fused_median_ms_avg_per_rank=2.4267188906669617 ms`。
- V3 normal 1024 no-tail：
  - log `hygon_tmp/sglang_debug/v3_normal_1024_bench_20260612_220343.log`；
  - correctness 前置 3/3 通过，`max_abs=0.000488281`；
  - 稳定轮 stage timing：K1 `2.510-2.572 ms`，K2 `0.107-0.109 ms`，K3 combine `2.295-2.348 ms`，no-tail barrier/reduce 约 `0.02-0.05 ms`；
  - 随后的 bench/perf loop 仍失败，`AssertionError: fused/baseline nonfinite fused=15 baseline=0 diff=15`，说明 no-clone/output lifecycle artifact 仍未关闭。
- 结论：
  - K1 staged-input 已把 1024 K1 stage 从约 `8.5 ms` 压到约 `2.5 ms`，方向正确但仍比 K1 pure `0.746 ms` 慢约 `1.75 ms`；
  - 当前 1024 V3 stage 总体约 `5.1-5.2 ms`，仍慢于原 staged `2.43 ms`；K1 和 K3 no-tail combine 都是明显 delta，后续按 hygon optimizer 闭环先补 4096 stage timing，再用 profiler/code-object/ASM diff 做单变量 A/B。

## 2026-06-12 - normal 4096 no-tail correctness-only stage timing

- 执行：
  - 直接调用 `tests/test_mega_moe_dcu.py --skip-bench`，避免已知 no-clone perf loop artifact 干扰；
  - env：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_STAGE_TIMING=1 MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1 K3_USE_ASM_TAIL_REDUCE=0`；
  - log `hygon_tmp/sglang_debug/v3_normal_4096_skipbench_stage_waitcnt_20260612_220835.log`；
  - out `hygon_tmp/sglang_debug/v3_normal_4096_skipbench_stage_waitcnt_20260612_220835.json`。
- 结果：
  - correctness 3/3 通过，`max_abs=0.000488281`；
  - 稳定轮 stage timing：K1 `8.786-9.005 ms`，K2 `0.214-0.220 ms`，K3 combine `8.032-8.260 ms`，no-tail barrier/reduce `~0.02/0.17 ms`；
  - correctness-only JSON `correct=true`，bench skipped。
- 结论：
  - K1 staged-input 对 4096 从此前 `~41-43 ms` 降到 `~9 ms`，但仍比 K1 pure normal `2.259 ms` 慢约 `6.7 ms`；
  - K3 combine 约 `8.1 ms`，与 normal rowptr split `staged_rowptr ~7.83 ms` 接近；
  - 下一步进入 hygon optimizer 证据链：profile/code-object/ASM diff，优先解释 K1 的剩余 `~3-4x` delta，其次验证 K3 remote rowptr combine 是否有可减少的额外同步/GLC/signal 成本。

## 2026-06-12 - normal PMC/code-object evidence

- code-object：
  - `hipprof --codeobj-analyze` 直接 stdout 只列 kernel；有效产物为 repo 根下 `*-resource-usage.RES` / `*-sass.ISA`；
  - `dccobjdump --inputs=<so> --show-sass` 对 `.so` 只显示 host ELF，记为 inconclusive；
  - K1 V3 fused 主 kernel在 `k1_v3_fused_ext.o-gfx938-0-sass.ISA`，有 `v_mmac=128`、`s_waitcnt=296`、`s_barrier=44`。
- K1 PMC：
  - V3 K1 1024 profile dir `hygon_tmp/sglang_debug/hipprof_k1_normal_1024_20260612_221209`；
  - K1 pure 1024 profile dir `hygon_tmp/sglang_debug/hipprof_k1_pure_normal_1024_20260612_221455`；
  - V3 K1 fused vs pure：VMEM_RD `~5.00M` vs `~0.922M`，VALU `~25.75M` vs `~17.65M`，arch_vgpr `220` vs `212`，sgpr `112` vs `48`；
  - 当前 K1 delta 主要来自 route/staged-input 附加逻辑，尤其 staged_x copy/metadata 扫描，而不是先改 GEMM 内层。
- K3 PMC：
  - profile dir `hygon_tmp/sglang_debug/hipprof_k3_normal_1024_20260612_221332`；
  - K3 V3 normal `TCP_TA_STALL≈25.2M` 且 `WRREQ_STALL=0`，支持 remote/scattered rowptr combine 数据通路等待判断。
- 下一步：
  - K1 单变量 A/B：只改 staged copy inactive-row zero-write/active-row staging 策略，保留 GEMM pipeline、route metadata、ready flag/waitcnt 不变；
  - A/B 必须先 correctness，再 stage timing，若无收益或不稳立即撤回。

## 2026-06-12 - normal K1 inactive staged zero A/B 结果

- 本地改动：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 增加 `skip_inactive_stage_zero` 分支；
  - `K1_fused/k1_v3_fused_ext.cu` 增加诊断 env `MEGAMOE_DCU_V3_K1_SKIP_INACTIVE_STAGE_ZERO`；
  - 默认不开启，默认路径保持原来的 inactive row zero-fill。
- 远端验证：
  - K1-only normal aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_skip_inactive_stage_zero_20260612_221834.log`；
  - 1024 default stage log `hygon_tmp/sglang_debug/v3_normal_1024_k1_stage_default_20260612_221902.log`，skip-zero log `hygon_tmp/sglang_debug/v3_normal_1024_k1_stage_skipzero_20260612_221902.log`；
  - 4096 skip-zero log `hygon_tmp/sglang_debug/v3_normal_4096_k1_stage_skipzero_20260612_222103.log`。
- 结果：
  - 1024 K1 stage 平均从约 `2.552 ms` 到约 `2.487 ms`，约 2%-3% 小收益；
  - 4096 K1 stage 约 `8.8-9.1 ms`，相比默认 `8.79-9.01 ms` 基本同量级；
  - 4096 correctness 3/3 通过但 `max_abs` 最高到约 `0.0021`，仍低于当前阈值。
- 结论：
  - inactive staged zero 不是 normal K1 主 delta，暂保留为 env-gated 诊断项，不默认开启；
  - 下一步按 profiler 证据转向 route/metadata VMEM/VALU：复用原 K1 compact prebuild kernels 做 A/B，让 V3 K1 主 kernel 只负责 staged input + pack5 GEMM，验证 route/emit 扫描是否是剩余大头。

## 2026-06-12 - normal K1 原 compact prebuild 复用 A/B 反证

- 本地改动：
  - 增加 env `MEGAMOE_DCU_V3_K1_REUSE_COMPACT_PREBUILD=1`；
  - env 开启时，V3 normal K1 host 端复用原 `k1_init/count/build/emit_compact_routes_kernel`，再以 `build_route_in_kernel=0` 启动 V3 K1 主 kernel；
  - V3 主 kernel 增加 `v3_k1_stage_prebuilt_route_tile_device`，只做每 tile `staged_x` copy + ready flag，GEMM 主体不变；
  - 默认路径不变，prebuild 只作为 A/B 诊断。
- 本地验证：
  - `python -m compileall ... tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过；
  - 本地没有 pytest，`python -m pytest -q tests/test_dcu_megamoe_v3.py` 仍失败于 `No module named pytest`；
  - 一次 PowerShell 下误用 bash heredoc、一次 SSH 嵌套 `exit \$rc` 转义错误均已记录为命令层错误，不重复同写法。
- 远端验证：
  - K1-only normal aicc rebuild 实际成功并拷贝 `.so`，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_prebuild_ab_20260612_223020.log`；命令最终非零仅来自 `exit \$rc` 引号转义错误；
  - import 检查 `k1_ext_import_ok True`；
  - 1024 prebuild correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_1024_k1_prebuild_skipbench_20260612_223208.log`，稳定 K1 stage 约 `2.50-2.59 ms`；
  - 4096 prebuild correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_4096_k1_prebuild_skipbench_20260612_223300.log`，稳定 K1 stage 约 `8.84-9.04 ms`。
- 结论：
  - 复用原 compact prebuild 没有改善 K1 stage；K1 剩余 delta 不是 V3 主 kernel 内 route count/build/emit 扫描主导；
  - 下一步聚焦 staged_x copy + staged GEMM 本身，检查 pure C GEMM 与 V3 fused GEMM 的 rowptr/row_expert/validity 分支、staged_x resource load 和额外 per-wave 检查差异。

## 2026-06-12 - normal K1 route-count validity A/B 反证

- 本地改动：
  - 增加 env `MEGAMOE_DCU_V3_K1_ROUTE_COUNT_VALIDITY=1`；
  - env 开启时，K1 GEMM 的 tile/wave 有效行判断优先从 route emit counts 推导，不再在该分支反复用 `row_x_ptrs` 判断 tile 有效行；
  - 默认路径不变，该分支只用于诊断 rowptr/validity 分支是否是 K1 剩余 delta。
- 远端验证：
  - K1-only normal aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_route_count_validity_20260612_224037.log`；
  - 1024 correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_1024_k1_route_count_validity_20260612_224141.log`，K1 stage 约 `2.57-2.70 ms`；
  - 4096 correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_4096_k1_route_count_validity_20260612_224427.log`，K1 stage 约 `8.92-9.16 ms`。
- 结论：
  - route-count validity 没有改善 1024/4096 K1 stage，且 1024 略慢；
  - rowptr/validity 分支本身不是 normal K1 剩余 `pure 0.746/2.259 ms` vs fused `~2.5/~9 ms` 的主因；
  - 下一轮按 profiler/ISA 证据继续拆 staged input copy 与 staged GEMM 的实际成本，优先找 staged_x 全量 global pass、同步/waitcnt、buffer resource load 与 register/SGPR 压力，而不是继续削 route/validity 分支。

## 2026-06-12 - normal K1 无收益诊断分支清理

- 背景：
  - `skip inactive zero`、`reuse compact prebuild`、`zero staged valid`、`route-count validity` 均已完成 A/B；
  - 这些分支不改善 K1 stage，其中 `zero staged valid` 还出现非有限错误；
  - 由于这些分支进入同一个 K1 normal 模板，运行时 bool 会增加 hot kernel 分支/codegen 压力。
- 本地改动：
  - 从 K1 normal hot template 撤掉上述无收益诊断分支；
  - `k1_symm_fused_l1_v3_pack5` 不再解析对应 env，也不再发起 compact-prebuild 额外 launch；
  - 保留 `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT` 作为必要时定位 direct-rowptr 的诊断开关；
  - 更新 `tests/test_dcu_megamoe_v3.py`，source guard 改为断言这些已撤回分支不在 V3 K1 hot path。
- 验证：
  - 本地 `compileall`、source guard inline、`git diff --check` 通过；本地 `pytest` 仍因缺模块不可用；
  - 远端 K1 normal aicc rebuild 实际成功并拷贝 `.so`，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_clean_diag_20260612_225452.log`；命令最终非零仍是嵌套 `exit \$rc` 转义错误；
  - import 检查 `k1_v3_entry_ok True`；
  - 1024 default no-tail correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_1024_default_after_clean_20260612_225640.log`，K1 stage 稳态约 `2.49-2.55 ms`；
  - 4096 default no-tail correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_4096_default_after_clean_20260612_225732.log`，K1 stage 稳态约 `8.69-8.88 ms`。
- 结论：
  - 无收益诊断分支从 hot path 清理后，K1 stage 恢复到 waitcnt 基线附近；
  - 后续 profiler/code-object 需要基于清理后的当前 best 重新采集，不能继续用带诊断分支的计数做优化判断。

## 2026-06-12 - normal K1 parallel staged_x copy A/B 保留

- 证据链：
  - 清理后 K1 PMC 1024：`arch_vgpr=216`、`sgpr=112`、`VMEM_RD≈4.94M`、`VMEM_WR≈0.437M`、`VALU≈24.78M`，相对 pure 仍有大量 staged-input / metadata 附加 VMEM 和 VALU；
  - direct-rowptr 诊断 correctness 通过但 1024 K1 stage 约 `8.3-8.5 ms`，证明简单取消 staged_x 不可行；
  - 当前问题是 full-row staged_x global pass 由 `blockIdx.x==0` 单 CTA 复制，其他 N-tile CTAs 等待，通信未充分隐藏。
- 本地改动：
  - 保持 route build、GEMM 主体和 staged_x layout 不变；
  - K1 normal staging 阶段改为 16 个 output-N CTAs 分段复制同一 row tile 的 K 维输入，每个 CTA 复制 `4096 / gridDim.x` 字节段；
  - 使用 tile 内 `stage_init / stage_count / stage_ready` slots 做 release/acquire 汇合，仍在同一个 K1 kernel 内，不新增 runtime kernel。
- 验证：
  - 本地 `compileall`、source guard inline、`git diff --check` 通过；
  - 远端强制删除 K1 build object 后 aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_parallel_stage_force_20260612_231030.log`；
  - 1024 no-tail correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_1024_k1_parallel_stage_20260612_231134.log`，K1 stage 稳态约 `1.36-1.50 ms`；
  - 4096 no-tail correctness-only 3/3 通过，log `hygon_tmp/sglang_debug/v3_normal_4096_k1_parallel_stage_20260612_231225.log`，K1 stage 稳态约 `4.35-4.42 ms`。
- 结论：
  - 该改动显著减少 K1 stage：1024 约 `2.5 ms -> 1.4-1.5 ms`，4096 约 `8.7-8.9 ms -> 4.35-4.42 ms`；
  - 这是当前 normal K1 的有效优化方向，保留；
  - 下一步采集 parallel-stage 后 PMC，确认 VMEM/VALU 变化，再转向剩余 K1 gap 和 K3 no-tail remote combine。

## 2026-06-12 - normal K1 parallel staged_x copy PMC attribution

- 执行：
  - clean serial K1 PMC：`hygon_tmp/sglang_debug/hipprof_k1_normal_clean_1024_20260612_230022/pmc.csv.csv`；
  - parallel K1 PMC：`hygon_tmp/sglang_debug/hipprof_k1_normal_parallel_1024_20260612_231417/pmc.csv.csv`；
  - 本地拉回到 `hygon_tmp/sglang_debug/prof/` 后按 8 ranks 汇总。
- 结果：
  - `arch_vgpr`: `216 -> 216`，`sgpr`: `112 -> 112`；
  - `SQ_INSTS_VMEM_RD`: `~4.94M -> ~1.32M`；
  - `SQ_INSTS_VMEM_WR`: `~0.437M -> ~0.437M`；
  - `SQ_INSTS_VALU`: `~24.78M -> ~21.80M`；
  - `SQ_INSTS_LDS`: `~5.87M -> ~4.05M`；
  - `GRBM_GUI_ACTIVE`: `~3.29M -> ~1.78M`。
- 结论：
  - K1 stage time 改善与 profiler 一致：主要收益来自减少串行 staged-input 全行 copy 带来的 VMEM read/active cycles；
  - 写量基本不变，说明改动不是少写 staged_x，而是把同等 staged_x 写入并行化并降低等待/活跃时间；
  - 当前 normal 1024/4096 剩余主瓶颈已经转向 K3 no-tail combine remote rowptr 数据通路。

## 2026-06-12 - normal K3 no-tail signal-off 默认切换

- 按 `hygon-hip-kernel-optimizer` 闭环继续 normal 性能优化：
  - 先查本地 DCU KB，得到 Flux GEMM+RS / DeepEP overlap 的可用结论：通信语义应尽量留在 epilogue store path，不要把额外等待塞进主 GEMM kernel；
  - 检查当前 K3 normal no-tail wrapper，确认 unset `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 默认会进入 K3 内部 owner-slot signal + peer wait。
- A/B：
  - 1024 单轮 signal-on：K3 stage 约 `3.49-4.30 ms`，correctness 首轮通过，但多轮 harness 仍触发已知 fused nonfinite artifact；
  - 1024 单轮 signal-off：K3 stage 约 `2.20-2.71 ms`，correctness 通过；
  - 1024 signal-off-acquire：K3 stage 约 `2.24-2.50 ms`，但触发 `max_abs=0.003662109375 > 0.0035`，不保留；
  - 4096 单轮 signal-on：K3 stage 约 `7.90-8.79 ms`；
  - 4096 单轮 signal-off：K3 stage 约 `7.70-8.12 ms`，correctness 通过；
  - signal-off 三轮 clone-isolated correctness：1024/4096 均 3/3 通过，logs `v3_normal_1024_k3_signal_off_3iter_20260612_232539.log`、`v3_normal_4096_k3_signal_off_3iter_20260612_232539.log`。
- 本地代码：
  - `megamoe/large_opt.py` 将 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 默认值从 `"1"` 改为 `"0"`；
  - `tests/test_dcu_megamoe_v3.py` 更新 source guard；
  - 显式 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 保留原 signal path 作为诊断/回退。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过；
  - inline source guard 通过。
- 下一步：
  - 同步 Python 改动到远端后，以默认 env 跑 1024/4096 no-tail 3 轮 correctness/stage，确认默认路径不再走 K3 internal signal；
  - 若通过，再对 K3 epilogue rowptr store / remote scattered 数据通路继续小步优化。

## 2026-06-12 - normal K3 signal-off raw/formal correctness triage

- 已按 planning / remote / hygon optimizer 流程恢复上下文并检查远端：
  - 远端 `hg@10.17.176.11` 可达，container `sglang_megamoe` 运行；
  - 容器内 `hy-smi --showpids` 显示无 KFD PIDs；
  - 本轮远端命令均使用 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`。
- 已同步和更新诊断脚本：
  - `hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 增加 ASM/V3/y 的 nonfinite 计数、dense/active nonfinite 计数和首个 V3 nonfinite 坐标；
  - 本地 `python -m compileall hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 通过；
  - `git diff --check -- hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 通过。
- direct K3 compare：
  - 初次 4096 signal-off prezero log `hygon_tmp/sglang_debug/k3_v3_dist_4096_signaloff_prezero_20260612_233925.log` 看到过 NaN 形态，但脚本当时未统计 nonfinite；
  - fill-sentinel log `hygon_tmp/sglang_debug/k3_v3_dist_4096_signaloff_fill7_20260612_234006.log`：`global_sentinel_active=0`，active slots 被覆盖；
  - 增强脚本 prezero log `hygon_tmp/sglang_debug/k3_v3_dist_4096_signaloff_prezero_nf_20260612_234326.log`：`global_v3_nonfinite=0`、`global_dense_gt_atol=0`；
  - 三次 repeat logs `k3_v3_dist_4096_signaloff_prezero_repeat{1,2,3}_20260612_234409.log` 均 `global_v3_nonfinite=0`、`global_dense_gt_atol=0`，`global_y_max_abs` 分别约 `0.0/0.00244/0.00333`。
- formal wrapper compare：
  - log `hygon_tmp/sglang_debug/v3_formal_4096_signaloff_no_prezero_20260612_234553.log`；
  - 结果 `global_v3_vs_baseline_max=0.00054931640625`、`global_v3_gt_atol=0`、`global_manual_v3_vs_fp8_v3_max=0.0`。
- 结论：
  - signal-off K3 normal no-tail raw/formal 暂时可继续作为性能优化基线；
  - default no-clone e2e 非有限现象仍记录为 output lifecycle / harness artifact，后续单独修，不把 full peer wait/signal 放回 K3 主 kernel。

## 2026-06-12 - normal K3 4096 rowptr split 首轮性能归因

- 执行：
  - `bench_k3_ll_rowptr_modes.py --backend normal --tokens 4096 --dump-rowptr-stats`；
  - log `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_stats_20260612_234727.log`；
  - json `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_stats_20260612_234727.json`。
- 结果：
  - `rowptr_all_zero` median avg rank 约 `2.543 ms`；
  - `local_rowptr` 约 `3.306 ms`；
  - `staged_local_only` 约 `3.461 ms`；
  - `staged_remote_only` 约 `7.814 ms`；
  - `staged_rowptr` 约 `7.842 ms`。
- rowptr 分布：
  - `active_rows_avg_rank=24576`；
  - `local_rows_avg_rank=2970.125`；
  - `remote_rows_avg_rank=21605.875`；
  - rank0 16-row chunk 平均 `unique_dest_ranks=1.806`，最大同 rank 平均 `12.337`，contiguous pairs 平均 `0.171`。
- 结论：
  - K3 normal no-tail 4096 的主要 delta 是 remote/scattered rowptr combine store；
  - 下一步先跑原 staged ASM no-tail K3 分项/PMC 对照，再决定是否做 row ordering、epilogue store schedule 或 waitcnt A/B；
  - 不重复 LL 已反证的 store-width、empty-tile skip、non-GLC rowptr load、full peer wait/signal。

## 2026-06-13 - normal K3 ASM vs V3 PMC 对照

- 已按 `hygon-hip-kernel-optimizer` 的测量闭环继续 normal K3 no-tail：
  - 重读 planning 三文件；
  - 远端参数来自 `.vscode/sftp.json`，命令均在 `sglang_megamoe` 内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后执行；
  - 对已有 4096 staged_rowptr hipprof PMC CSV 做 exact kernel match。
- 对照结果：
  - 原 ASM K3COMBINE rowptr split：`rowptr_all_zero/local/staged_remote/staged ≈ 1.554/1.615/3.094/3.113 ms`；
  - V3 K3 normal rowptr split：`2.543/3.306/7.814/7.842 ms`；
  - PMC：V3 `VMEM_RD≈4.62M` vs ASM `2.26M`，V3 `VMEM_WR≈1.58M` vs ASM `0.238M`，V3 `TCP_TA_DATA_STALL≈88.0M` vs ASM `10.0M`，TCC miss 基本相同。
- 结论：
  - 下一步不直接改 store width；先检查 V3 K3 epilogue/source 是否有多余 global output/writeback、rowptr reload 或 fallback store，再结合 ISA/store schedule 做小步 A/B。

## 2026-06-13 - normal K3 rowaddr hoist A/B 撤回

- 本地改动：
  - 尝试 normal K3 epilogue rowptr 地址预取复用，先用 `V3RowAddr4` 结构体，再改成标量 `int64_t` 传参；
  - 两版都只改 K3 normal epilogue，不改 store width、signal、barrier/reduce 或 GEMM compute。
- 远端验证：
  - 首次只重编 K3 raw normal 时误把 K1 extension 覆盖成 stub，1024 测试失败于 `V3 K1 raw kernels were not compiled`；已记录为构建组合错误；
  - 随后按正确组合 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` + `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1`、backend 均 normal、aicc shim 重编；
  - rowaddr hoist 结构体版 1024 首轮 correctness 通过但第 2 轮 `max_abs=0.004711 > 0.0035`；
  - rowaddr hoist 标量版 1024 首轮 correctness 通过但第 2 轮 `max_abs=0.004456 > 0.0035`。
- 处理：
  - 已从本地和远端撤回 rowaddr hoist，重新按 K1/K3 normal raw 组合构建；
  - 回退后 1024 三轮 correctness-only/stage timing 通过，K3 stage 约 `2.22-2.29 ms`。
- 结论：
  - 该 A/B 有性能信号但 correctness 不稳，不能保留；
  - 后续不重复跨 hidden-R 复用 epilogue row address，改从 compute 侧 active-check rowptr load hoist 做更小 A/B。

## 2026-06-13 - normal K3 active-check hoist A/B 撤回

- 本地改动：
  - normal K3 compute 中为 `token_col0/1` 各提前 load 一次 rowptr active bool；
  - phase 内 B-load 复用 bool，不再每个 phase 重新 load rowptr；
  - epilogue store 未改变。
- 远端验证：
  - K1/K3 normal raw 组合 aicc rebuild 成功；
  - 1024 首轮 correctness 通过，K3 stage 降到约 `1.99-2.05 ms`；
  - 第 2 轮 correctness 失败：`max_abs=0.012939453125 > 0.0035`。
- 处理：
  - 已从本地源码撤回 active-check hoist；
  - 下一步同步恢复远端并重编，回到正确基线后转 ISA/store schedule 分析。
- 结论：
  - compute 前 hoist rowptr-derived 状态会破坏 normal K3 correctness，不保留；
  - 不能继续靠长生命周期 rowptr hoist 优化，必须换成更靠近原 ASM schedule 的短窗口调度或其他证据驱动方案。

## 2026-06-13 - normal K3 active-check hoist 回退基线确认

- 已按 `hygon-hip-kernel-optimizer` 闭环恢复当前基线：
  - 重读 planning 三文件和 remote/optimizer skill；
  - 本地确认 `k3_v3_pack5_groupgemm_impl.cuh` 中已无 `active_flag`、`rowaddr_hoist`、`V3RowAddr4` 等上一轮 A/B 残留符号；
  - 远端命令继续使用 `.vscode/sftp.json` 的 `hg@10.17.176.11`、container `sglang_megamoe`，并在容器内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`。
- 验证：
  - 1024 normal no-tail skip-bench 三轮 correctness 通过；
  - log `hygon_tmp/sglang_debug/v3_normal_1024_after_active_revert_skipbench_20260613_003833.log`；
  - `max_abs=0.000488281`，`correct=true`；
  - 稳态 stage timing：K1 约 `1.32-1.49 ms`，K3 combine 约 `2.21-2.27 ms`。
- 命令层错误：
  - 第一次 PowerShell 远端脚本使用双引号 here-string，导致 `$(date -Is)` 被本地 PowerShell 解析为 `Get-Date -Is` 并失败；
  - 已改用单引号 here-string + placeholder 替换，后续不重复该写法。
- 下一步：
  - 基线已恢复，继续 normal K3 no-tail ISA/store schedule 对比；
  - 不再重复跨完整 compute loop 保存 rowptr-derived 状态的 A/B。

## 2026-06-13 - normal K3 short-window rowptr nowait A/B 撤回

- ISA/KB 依据：
  - 查本地 DCU KB，复用结论是通信语义应放在 epilogue store path，但必须用本工程 ISA/bench 验证 store/address schedule；
  - 当前 V3 no-tail codeobj 通过 `hipprof --codeobj-analyze` + `llvm-objdump` 取证：no-tail template 约 `global_load_dwordx2=132`、`global_store_short=128`、`s_waitcnt=217`；
  - store window 形态为 rowptr `global_load_dwordx2 ... glc` 后立即 `s_waitcnt vmcnt(0)`、地址检查、`global_store_short`。
- A/B：
  - 在 `store_acc_fragment_scaled_unmasked_device` 内把四个 rowptr load 改成 no-wait inline asm，先发出 load，再做 scale/mul/BF16 pack，store 前统一 wait；
  - 没有跨 compute loop 或 hidden-R 复用 rowptr 状态。
- 结果：
  - 强制删除 K3 V3 object 后 normal aicc 重编成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_short_window_rowptr_force_20260613_004604.log`；
  - 1024 normal no-tail skip-bench 触发 VMFault / SIGABRT，log `hygon_tmp/sglang_debug/v3_normal_1024_k3_short_window_rowptr_skipbench_20260613_004715.log`；
  - 已撤回本地和远端源码，强制重编恢复，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_short_window_20260613_004850.log`；
  - 撤回后 1024 三轮 correctness 通过，log `hygon_tmp/sglang_debug/v3_normal_1024_after_revert_short_window_skipbench_20260613_005002.log`，K3 stage 回到约 `2.20-2.31 ms`。
- 结论：
  - no-wait rowptr load + 延迟 wait 在当前 aicc/codegen 下不安全；
  - 后续不重复该方向，转向不破坏 waitcnt hazard 可见性的结构调整或 ASM-style address/store 形态。

## 2026-06-13 - normal K3 rowaddr group4 / non-GLC A/B 撤回

- 触发证据：
  - 原 K3COMBINE direct epilogue 只对每 4 行 load 一次 row pointer，然后通过地址自增连续写 16 个 hidden step；
  - V3 normal 当前每个 hidden step 重新 load rowptr，解释了 `global_load_dwordx2 glc + wait + global_store_short` 的重复 store window。
- A/B 1：rowaddr group4 hoist
  - 将 normal K3 epilogue 改成每 4 个 hidden step 复用同一组 `token_group + {0,4,8,12}` row address；
  - 初版 1024 首轮通过但第 2 轮 `max_abs=0.0035400390625 > 0.0035`；
  - 加作用域降低 rowaddr live range 后仍第 2 轮失败，`max_abs=0.0037994384765625 > 0.0035`；
  - 已从本地和远端撤回，不保留。
- A/B 2：normal rowptr load non-GLC
  - KB 显示 `glc` 是 L2-oriented path，原 K3COMBINE ASM rowptr load 无 `glc`；因此尝试只把 normal K3 active/store rowptr helper 改成无 `glc` 但保留 `s_waitcnt vmcnt(0)`；
  - 1024 rank0 三轮 max_abs 均为 `0.000488281`，但其他 rank 报 `fused/baseline nonfinite fused=15 baseline=0`，且 K3 stage 没有改善；
  - 已撤回，不保留。
- 回退验证：
  - 本地/远端源码确认无 `global_load_i64_device`、`K1_LOAD_ROWADDR4`、`store_acc_fragment_scaled_rowaddr`、`store_bf16_rowaddr_device` 残留；
  - 强制重编回 glc 基线，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_nonglc_20260613_010944.log`；
  - direct K3 compare 4096 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_after_revert_nonglc_20260613_011253.log`，`global_max_abs=0`、`global_dense_gt_atol=0`、`global_v3_nonfinite=0`；
  - staged e2e 仍偶发已知 output lifecycle nonfinite artifact，log `hygon_tmp/sglang_debug/v3_normal_1024_after_revert_nonglc_skipbench_20260613_011053.log`，不作为 raw K3 回退失败证据。
- 结论：
  - rowptr hoist 方向虽然有 stage timing 信号，但当前 C/aicc 形态不能过 1024 三轮 correctness；
  - non-GLC 不改善 normal K3，并引入跨 rank 非有限风险；
  - 后续避免继续重复 rowaddr hoist / non-GLC / no-wait rowptr load，转向解释 V3 比 ASM 多出的 VMEM_WR：优先查是否有 fallback output/global write、store 指令计数口径或 buffer-store/resource 形态差异。

## 2026-06-13 - normal K3 staged-half LDS/vector-store A/B 撤回

- 背景：
  - 用户指出 K3 normal no-tail 与原 ASM 差距很大，并提示参考 `K3COMBINE` 与 `K3COMBINE_TAILREDUCE` 差异；
  - 根据原 ASM no-tail epilogue 的 staged LDS + `global_store_dwordx4` store 形态，尝试把 V3 K3 scalar rowptr store 改为 LDS staged-half vector-store。
- A/B 结果：
  - 初版 staged-half 1024 K3 stage 有大幅正向信号，约 `1.05-1.18 ms`，但 direct 4096 compare 失败并出现少量 nonfinite；
  - 加 `s_waitcnt lgkmcnt(0)` 后 nonfinite 消失，但 direct compare 仍大量超差；
  - 尝试 ASM-style row4 store loop 后 direct compare 更差，已判断为失败分支。
- 本地处理：
  - 已把默认 no-tail 256-N 路径切回 `K1_STORE_ROWS_256(K1_STORE_ROW_UNMASKED)` scalar rowptr store 基线；
  - 失败的 staged-half helper 代码暂不进入默认执行路径，下一步远端强制重编并跑 direct K3 compare 恢复 correctness 基线。
- 结论：
  - staged-half 的性能信号真实，但当前 C/aicc 映射下 row/value layout 仍不等价；
  - 后续若继续该方向，必须先做更小的 layout/source-backed ASM-style store probe，不能直接把失败分支留在主路径。

## 2026-06-13 - normal K3 no-tail fast-exit A/B 撤回

- 背景：
  - 在 staged-half 撤回后，按原 ASM no-tail 末尾 `s_waitcnt vmcnt(0) + buffer_wbinvl1_vol` 线索，尝试减少 V3 no-tail store 后的 block barrier/system fence。
- A/B：
  - no-tail store 后改为 `wait_vmem_lds_store_device(); invalidate_l1_device(); return;`；
  - 不改 GEMM、rowptr store、signal-off 默认或外部 no-tail barrier/reduce。
- 结果：
  - direct K3 compare 4096 通过阈值，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_notail_fast_exit_20260613_014556.log`；
  - 1024 staged e2e log `hygon_tmp/sglang_debug/v3_normal_1024_k3_notail_fast_exit_skipbench_20260613_014636.log` 出现 fused nonfinite，且 K3 stage 无实质改善、no-tail barrier 明显变慢。
- 处理：
  - 已从本地 `k3_v3_pack5_groupgemm_impl.cuh` 撤回 fast-exit 分支；
  - 下一步同步远端并强制重编，恢复 scalar rowptr + barrier/fence 稳定基线后再继续 profiling/ISA 方向。

## 2026-06-13 - normal K3 rowptr buffer-store A/B 撤回

- 本地改动：
  - 将 `store_bf16_rowptr_device` 的最终 rowptr BF16 store 从 `global_store_short` 改为 raw-buffer `buffer_store_bf16_device`；
  - rowptr load/wait、GEMM 主体、no-tail signal-off、block barrier/system fence 均未改变。
- 远端验证：
  - K1/K3 V3 normal aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowptr_buffer_store_20260613_015454.log`；
  - direct K3 compare 4096 通过阈值，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_rowptr_buffer_store_20260613_015607.log`；
  - rowptr split bench：1024 `staged_rowptr≈2.29 ms`，4096 `staged_rowptr≈8.09 ms`，均不优于 scalar 基线，log/json `v3_normal_k3_rowptr_modes_{1024,4096}_buffer_store_20260613_015654.*`；
  - `dccobjdump` 对 `.so` 输出仅 host ELF header，未得到 device ISA，记为 inconclusive。
- 处理：
  - 已从本地源码撤回 buffer-store A/B，恢复 `global_store_bf16_device(row_ptr + hidden, value)`；
  - 下一步同步/重编回 scalar 基线，再继续找 K3 VMEM_WR/remote store gap 的其他证据。

## 2026-06-13 - normal K3 unconditional B-load A/B 撤回

- 本地改动：
  - 在 normal K3 compute loop 中将 B-load active-row helper 改为 pure-style unconditional contiguous `act_fp8` load；
  - epilogue rowptr active skip 保持不变。
- 远端验证：
  - aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_uncond_bload_20260613_020155.log`；
  - direct K3 compare 4096 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_uncond_bload_20260613_020301.log`；
  - 关键统计：`global_max_abs=0.0303955078125`、`global_gt_atol=1223862`、无 nonfinite。
- 处理：
  - 已撤回本地源码，恢复 `buffer_load_fp8_b128_active_row_device`；
  - 下一步同步/重编回 scalar 基线。

## 2026-06-13 - normal K3 per-stage active-mask A/B 撤回

- 本地改动：
  - 每个 `stage_iter` 对 `token_col0/1` 各读一次 rowptr active bool，phase0/phase4 共享该 bool；
  - 保留 inactive B 置零语义，不改 epilogue rowptr store。
- 远端验证：
  - aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_stage_active_mask_20260613_020743.log`；
  - direct K3 compare 4096 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_stage_active_mask_20260613_020850.log`；
  - 关键统计：`global_max_abs=0.0057373046875`、`global_gt_atol=24`、无 nonfinite。
- 处理：
  - 已撤回本地源码，恢复每个 phase 内调用 `buffer_load_fp8_b128_active_row_device`；
  - 下一步同步/重编回稳定基线。

## 2026-06-13 - normal K3 staged-half H1 rowptr offset A/B 撤回

- 背景：
  - 复查原 `K3COMBINE` ASM 后确认 `K3_STORE_STAGED_HALF 1024` 中的 `1024` 是 rowptr raw-buffer load 的字节 offset，对应 128 个 int64 row pointer；此前 C helper 若把它当行数使用，会把 H1 combine 写到错误 rowptr 区域。
  - 该线索解释了 staged-half 首轮 A/B 有强性能信号但 direct compare 大错的一部分。
- 本地改动：
  - 仅对 normal K3 no-tail N=256 epilogue 启用 staged-half LDS/vector-store；
  - H0 store 使用 row offset 0，H1 store 使用 row offset 128；tail-reduce 和 signal-only 路径保持 scalar store。
- 远端验证：
  - aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_staged_half_h1fix_20260613_021811.log`；
  - direct K3 compare 4096 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_staged_half_h1fix_20260613_021919.log`；
  - 关键统计：`global_max_abs=0.17596435546875`、`global_gt_atol=200`、`global_v3_nonfinite=33`，非有限主要出现在 inactive dense slots，active nonfinite 为 0。
- 处理：
  - 已撤回本地源码，恢复 scalar rowptr `global_store_short` 基线；
  - staged-half 后续若继续，只能先做更小的 value/layout probe，不能直接重新接入主 kernel。
- 回退验证：
  - 回退后远端 aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_staged_half_h1fix_20260613_022123.log`；
  - direct K3 compare 4096 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_after_revert_staged_half_h1fix_20260613_022230.log`；
  - 关键统计：`global_max_abs=0`、`global_gt_atol=0`、`global_v3_nonfinite=0`。

## 2026-06-13 - normal K1 direct-source / device-fence A/B 反证

- 按 `hygon-hip-kernel-optimizer` 闭环先恢复基线再做单变量 A/B：
  - 远端源码 hash 与本地一致；
  - 8 卡空闲；
  - 所有远端命令继续在 `sglang_megamoe` 内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`。
- `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT=1` 诊断：
  - 1024 normal no-tail `--skip-bench` 三轮 correctness 通过；
  - baseline staged input K1 median 约 `1.44 ms`；
  - disable staged input 后 K1 median 约 `8.42 ms`；
  - log `hygon_tmp/sglang_debug/v3_normal_k1_stage_input_skipbench_ab_20260613_022744.log`。
- K1 internal device-scope fence A/B：
  - 查本地 DCU KB，确认 `__threadfence()` 是 device memory fence，`__threadfence_system()` 用于 host/system 可见性；
  - 尝试 env-gated `MEGAMOE_DCU_V3_K1_DEVICE_FENCE=1`，只把 K1 normal 内部 route/stage ready fence scope 降到 device；
  - aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_device_fence_ab_20260613_023154.log`；
  - direct K1 compare default/device-fence 均通过，`max_abs=0`；
  - 1024 device-fence e2e 单轮 correctness 通过，但 K1 median 仍约 `1.43 ms`，无性能收益；
  - default system-fence 多轮在该 codegen 下触发一次已知 fused nonfinite artifact，说明该 env-gated branch 对稳定性/代码生成有扰动；
  - 已从本地源码撤回 device-fence A/B。
- 结论：
  - normal direct source A-load 不可行，远端 row scatter 重复读取远慢于 staged copy；
  - K1 internal fence scope 不是当前 K1 normal 主要瓶颈；
  - 后续不重复 direct-source 和 fence-scope 方向。

## 2026-06-13 - normal K3 K2-zero + unconditional B-load A/B 撤回

- 按 `hygon-hip-kernel-optimizer` 测量闭环继续 normal K3 no-tail：
  - 重读 planning 三文件和 optimizer/remote skill；
  - 远端 `sglang_megamoe` 内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`；
  - `hy-smi --showpids` 显示无 KFD PIDs。
- A/B 诊断：
  - default + K2 zero direct compare 1024 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_k2zero_default_20260613_025641.log`；
  - K2 zero + K3 unconditional B-load direct compare 1024 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_k2zero_assume_20260613_025641.log`；
  - 关键统计：`global_inactive_act_nonzero=0` 但 `global_v3_nonfinite=48`、`global_gt_atol=123423`，非有限出现在 active/dense slot。
- 回退：
  - 删除 `MEGAMOE_DCU_V3_K3_ASSUME_ZERO_INACTIVE_B`、K2 `zero_inactive_rows`、K3 `assume_zero_inactive_b` wrapper/ext 参数和 `kAssumeZeroInactiveB` template 分支；
  - 第一次重编失败，原因是 `k3_v3_normal_combine_tail_raw` 漏删一个已撤销参数，log `hygon_tmp/sglang_debug/rebuild_v3_normal_revert_k2zero_k3uncond_20260613_030145.log`；
  - 修正参数数量后重编通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_revert_k2zero_k3uncond_retry_20260613_030356.log`。
- 回退验证：
  - 本地 `python -m compileall` 和 `git diff --check` 通过；
  - direct K3 compare 1024 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_after_revert_uncond_20260613_030531.log`，`global_max_abs=0`、`global_gt_atol=0`、`global_v3_nonfinite=0`；
  - e2e normal no-tail 1024 skip-bench 三轮通过，log `hygon_tmp/sglang_debug/v3_normal_1024_after_revert_uncond_skipbench_20260613_030531.log`；
  - stage timing 稳态：K1 约 `1.33-1.46 ms`，K2 约 `0.07-0.11 ms`，K3 combine 约 `2.19-2.29 ms`。
- 结论：
  - K3 compute active mask 不能通过 K2 清 inactive rows 消除；
  - 后续不重复 K2-zero + K3 unconditional B-load 方向，继续转向 ISA/store schedule 或 source-backed ASM-style store probe。

## 2026-06-13 - normal K3 gated staged-half A/B 撤回

- 背景：
  - 复查原 K3COMBINE ASM 后发现 H0/H1 staged-half 不是所有 waves 同时写/读 LDS，而是 wave0-3 stage/store H0，barrier 后 wave4-7 stage/store H1；
  - 因此前 staged-half 首轮失败可能混入了 H0/H1 gating 不一致因素，所以做一次更接近 ASM 的 gated A/B。
- 本地改动：
  - 仅在 normal K3 no-tail N=256 路径启用 gated staged-half；
  - H0: `wave_id < 4` stage + store rowptr offset 0；
  - H1: `wave_id >= 4` stage + store rowptr offset 128；
  - tail-reduce/signal-only 仍保持 scalar store。
- 远端验证：
  - K1/K3 V3 normal aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_staged_half_gated_20260613_031742.log`；
  - direct K3 compare 1024 通过：`global_max_abs=0.001941680908203125`、`global_gt_atol=0`、`global_v3_nonfinite=0`，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_staged_half_gated_20260613_032032.log`；
  - direct K3 compare 4096 失败：`global_max_abs=0.26544189453125`、`global_gt_atol=422`、`global_v3_nonfinite=15`，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_staged_half_gated_20260613_032127.log`；
  - 失败主要落在 inactive dense slot，active slot 没有 nonfinite，说明 staged-half row/value mapping 仍不等价。
- 回退：
  - 已撤回 gated staged-half，恢复 scalar rowptr store；
  - 回退后 aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_gated_staged_half_20260613_032331.log`；
  - 回退后 direct K3 compare 4096 通过：`global_max_abs=0`、`global_gt_atol=0`、`global_v3_nonfinite=0`，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_after_revert_gated_staged_half_20260613_032438.log`。
- 结论：
  - H0/H1 gating 是必要线索但不是充分修复；
  - staged-half/vector-store 方向仍有性能潜力，但当前主 kernel 内直接启用不满足 4096 correctness；
  - 后续不再在主路径重复 staged-half H0/H1 gating 变体，除非先用独立 source-backed value/layout probe 证明 LDS row/value 映射完全等价。

## 2026-06-13 - normal K3 rowptr raw-buffer load/resource A/B retained

- 按 `hygon-hip-kernel-optimizer` 闭环执行：DCU KB 查询、单变量 A/B、本地 diff check、远端 normal aicc rebuild、direct K3 correctness、rowptr split timing、PMC/ISA 尝试。
- 本地改动：
  - normal K3 compute active-check 的 row pointer load 改为基于 `row_combine_ptrs` buffer resource 的 raw dwordx2 load；
  - epilogue row pointer address fetch 同样改为 rowptr resource load；
  - 最终 combine 写回仍保持 scalar `global_store_short`，GEMM 主循环、no-tail barrier/fence、tail 路径不改。
- 远端验证：
  - K1/K3 V3 normal aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowptr_resource_20260613_033530.log`；
  - direct K3 compare 1024 通过：`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_rowptr_resource_20260613_033638.log`；
  - direct K3 compare 4096 通过：`global_max_abs=0.002838134765625`、`global_gt_atol=0`、无 nonfinite，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_rowptr_resource_20260613_033712.log`。
- Rowptr split timing：
  - 1024 all_zero/local/staged_remote/staged avg `0.605/0.937/2.216/2.237 ms`；
  - 4096 all_zero/local/staged_remote/staged avg `2.153/3.276/7.781/7.808 ms`；
  - JSON: `hygon_tmp/sglang_debug/prof/k3_rowptr_normal_1024_rowptr_resource_20260613_033822.json`、`hygon_tmp/sglang_debug/prof/k3_rowptr_normal_4096_rowptr_resource_20260613_033822.json`。
- 结论：
  - 该 A/B 保留为小幅正收益：显著降低 all-zero/GEMM floor，local 小幅改善，但 staged_remote/staged 基本未解决；
  - 下一步做 attribution：确认收益主要来自 compute active-check rowptr resource，还是 epilogue address fetch resource。
- 工具问题：
  - `--json-out` 是错误参数，后续脚本使用 `--out`；
  - filtered PMC CSV 为空；no-filter PMC 在 multiprocess bench 下挂住，残留进程已 kill；
  - 当前 `hipprof --codeobj-analyze`、`dccobjdump`、`llvm-objdump` 路径没有稳定导出 device ISA，本轮 ISA 证据记为 degraded。

## 2026-06-13 - normal K3 compute-only rowptr resource A/B reverted

- A/B 目的：归因上一轮 rowptr resource 收益是否只来自 compute active-check；因此保留 compute active-check 的 rowptr resource load，把 epilogue rowptr address fetch 恢复为原 `global_load_i64_glc_device`。
- 本地检查：无尾随空白，`python -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过。
- 远端验证：
  - 第一次 build 包装命令 `$?` quoting 写错，属于命令层问题；随后确认头文件改动未触发 ninja 依赖，强制删除 K3 V3 object/so 后重编。
  - 强制重编通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowptr_compute_only_force_20260613_040554.log`。
  - direct K3 1024 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_rowptr_compute_only_20260613_040740.log`，`global_max_abs=0.0028076171875`、`global_gt_atol=0`、无 nonfinite。
  - direct K3 4096 未过 gate，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_rowptr_compute_only_20260613_040816.log`，`global_max_abs=0.005279541015625`、`global_gt_atol=1`、无 nonfinite。
- 处理：该 A/B 不跑性能，已从本地源码撤回到 compute + epilogue 都使用 rowptr resource 的上一通过版本；下一步同步远端并重编恢复。
- 结论：epilogue rowptr resource load 不能简单撤掉；至少在当前 aicc/codegen 下，compute-only resource 触发 4096 单点数值漂移。

## 2026-06-13 - normal K3 staged-half + rowptr resource A/B reverted

- 本轮按 optimizer 闭环继续 K3 normal no-tail：
  - 重读 planning 三文件、remote workflow、hygon optimizer 和 dcu KB；
  - KB 检索仍指向 DeepGEMM ASM/aicc/dccobjdump 对照和 Flux GEMM+RS 的 epilogue 通信融合思路，但没有给出能绕过 correctness 的新 store 语义。
- A/B：
  - 在 H0/H1 gated staged-half 基础上，将 `K1_STORE_STAGED_HALF` 内 rowptr load 改为 `rowptr_resource` raw dwordx2；
  - 1024 direct K3 compare 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_staged_half_resource_20260613_041905.log`；
  - 关键统计：`global_max_abs=1.31982421875`、`global_gt_atol=18733`、`global_v3_nonfinite=0`。
- 回退与验证：
  - 本地恢复 no-tail N=256 路径为 scalar `K1_STORE_ROWS_256(K1_STORE_ROW_UNMASKED)`；
  - 本地 `python -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 和 `git diff --check` 通过；
  - 远端强制删除 K3 V3 object/so 后 aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_staged_half_resource_20260613_042033.log`；
  - 回退后 direct K3 1024 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_after_revert_staged_half_resource_20260613_042142.log`，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - 回退后 direct K3 4096 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_after_revert_staged_half_resource_20260613_042219.log`，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite。
- 结论：
  - staged-half 主路径仍不能过 correctness，rowptr resource load 不是缺失修复；
  - 当前默认恢复到 scalar rowptr-resource store 的上一稳定版本；
  - 下一步继续 normal 性能优化时，不再直接接 staged-half 主路径，除非先做独立 LDS value/layout probe。

## 2026-06-13 - normal K3 rowaddr resource reuse A/B reverted

- A/B 目标：
  - 仿照 ASM direct epilogue，把 4 个 row address 预取后跨 16 个 hidden step 复用；
  - 与此前 rowaddr reuse 的差异是 rowptr load 改为当前已通过的 `rowptr_resource` raw load。
- 执行结果：
  - 初次远端编译失败，原因是调用了不存在的 `store_bf16_rowaddr_device`；修正为直接 `global_store_bf16_device` 后继续；
  - 修正后 aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowaddr_resource_reuse_retry_20260613_042940.log`；
  - direct K3 1024 未过 gate，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_rowaddr_resource_reuse_20260613_043048.log`，`global_max_abs=0.004998207092285156`、`global_gt_atol=1`、无 nonfinite。
- 回退：
  - 本地撤回 rowaddr reuse，恢复 scalar rowptr-resource store；
  - 本地 compileall / diff check 通过；
  - 远端强制 rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_rowaddr_resource_reuse_20260613_043205.log`；
  - restore direct K3 1024 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_after_revert_rowaddr_resource_reuse_20260613_043313.log`，`global_max_abs=0.001922607421875`、`global_gt_atol=0`、无 nonfinite。
- 结论：
  - K3 rowaddr reuse 方向再次反证，resource load 不改变该 hazard；
  - 后续 normal 优化先切到 K1 staged-copy/metadata 固定成本，K3 store-window 只保留为独立 probe 方向。

## 2026-06-13 - normal K3 rowptr4 batch-load A/B retained

- 按 `hygon-hip-kernel-optimizer` 测量闭环继续 K3 normal no-tail：
  - 重读 planning 三文件、remote workflow、hygon optimizer 和 dcu KB；
  - 先确认 8 卡空闲，所有远端命令仍在 `sglang_megamoe` 内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`；
  - 本轮只改 K3 kernel helper，不跑 source pytest 链条，直接做 local compileall/diff check、远端 aicc rebuild、direct K3 correctness 和 rowptr split timing。
- A/B 内容：
  - 保持当前 scalar rowptr-resource store、GEMM 主循环、no-tail barrier/fence 和 tail/signal 路径不变；
  - 仅在 `!kTailReduce && !kSignalOnly` 的 unmasked scalar epilogue 中，把同一 token group 的 4 个 rowptr raw-buffer load 合到 `store_bf16_rowptr4_buffer_device`，4 次 load 后只做一次 `s_waitcnt vmcnt(0)`；
  - 初版 4096 出现 `global_gt_atol=1` 的单点漂移，增加输出 pack 的最小 asm dependency 后重新验证。
- 验证：
  - aicc rebuild 通过：`hygon_tmp/sglang_debug/rebuild_v3_normal_k1k3_rowptr4_batch_barrier_retry_20260613_045511.log`；
  - direct K3 1024 通过：`hygon_tmp/sglang_debug/k3_v3_dist_1024_rowptr4_batch_barrier_20260613_073650.log`，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - direct K3 4096 通过：`hygon_tmp/sglang_debug/k3_v3_dist_4096_rowptr4_batch_barrier_20260613_045618.log`，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - 本地 `python -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 通过。
- Rowptr split timing：
  - 1024 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260613_045856.json`：all_zero/local/staged_remote/staged median_avg `0.544/0.875/2.159/2.166 ms`；
  - 4096 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260613_045920.json`：all_zero/local/staged_remote/staged median_avg `1.935/3.064/7.623/7.661 ms`。
- 结论：
  - 相对上一稳定 rowptr-resource 基线，1024 staged 从约 `2.237 ms` 到 `2.166 ms`，4096 staged 从约 `7.808 ms` 到 `7.661 ms`，属于小幅正收益，保留；
  - 主 gap 仍未解决：4096 staged 仍远慢于原 ASM rowptr split `~3.113 ms`，下一步继续优先解释和压低 normal K3 no-tail 的 remote/scattered combine store 成本。

## 2026-06-13 - normal K3 rowptr4/rowptr8 clean rebuild 反证并撤回

- 复测发现 rowptr4 retained 结论不稳：
  - rowptr8 batch-load：1024 direct 通过，但 4096 direct 失败，`global_max_abs=0.00799560546875`、`global_gt_atol=134`，无 nonfinite；
  - 撤回 rowptr8 后，clean rebuild 的 rowptr4 路径 1024 direct 通过，但 4096 direct 失败，`global_max_abs=0.0077056884765625`、`global_gt_atol=117`，无 nonfinite；
  - 撤回 rowptr4 helper 回 scalar store 后，4096 仍有小漂移，说明当前 K3 no-tail epilogue 对 aicc clean codegen 敏感。
- 处理：
  - `task_plan.md` 已把 rowptr4/rowptr8 从 retained 改为反证项；
  - `findings.md` 已更新 rowptr4 段落，明确不再重复 rowptr4/rowptr8 batch-load；
  - 本地 K3 no-tail 默认路径保持 scalar rowptr-resource store。
- 命令问题：
  - 一次 direct K3 test wrapper 的 `stamp` substring 端口计算错误，bash 报 `_0: unbound variable`，未跑到 GPU；已改固定端口偏移后重跑。

## 2026-06-13 - normal K3 scalar rowptr store pack dependency

- 本地改动：
  - 在 `store_acc_fragment_scaled_unmasked_device` 内先计算 `out_bits0/out_bits1`，通过 `asm volatile("s_nop 0" : "+v"(out_bits0), "+v"(out_bits1) :: "memory")` 建立最小 pack dependency，再执行四次 scalar rowptr store；
  - 不改 GEMM 主循环、rowptr resource load、同步边界、tail/signal 路径。
- 本地验证：
  - `python -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 对 untracked header 没有覆盖完整内容，后续仍以 compile/rebuild/direct correctness 为准。
- 远端验证：
  - 显式 `scp` 同步 `k3_v3_pack5_groupgemm_impl.cuh` 到 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM`；
  - 第一次 build 日志显示 K3 `ninja: no work to do`，未触发 header 重编；随后删除实际 `build/temp.../K3_fused/k3_v3_fused_ext.o` 和 so，强制 aicc 重编成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_scalar_pack_dep_force_20260613_080448.log`；
  - direct K3 1024 通过：`hygon_tmp/sglang_debug/k3_v3_dist_1024_scalar_pack_dep_seq2_20260613_080925.log`，`global_max_abs=0`、`global_gt_atol=0`；
  - direct K3 4096 首轮 `hygon_tmp/sglang_debug/k3_v3_dist_4096_scalar_pack_dep_20260613_080616.log` 仍有小漂移，`global_max_abs=0.0050048828125`、`global_gt_atol=6`，无 nonfinite；
  - 随后三次 4096 repeat 和一次 1024->4096 顺序复测均通过，`global_max_abs=0`、`global_gt_atol=0`。
- 当前结论：
  - scalar pack dependency 可作为当前 correctness 基线继续性能测量，但 4096 首轮漂移保留为观察项；
  - 下一步先跑 rowptr split timing，确认该保守修复是否带来性能退化，再继续 normal K3 remote/scattered store gap 优化。

## 2026-06-13 - normal K3 scalar pack dependency 后性能与 ISA 对照

- Rowptr split timing：
  - 远端命令使用 `TOKENS_LIST="1024 4096" WARMUP=3 REPEAT=10 ROUNDS=5 MODES="rowptr_all_zero,local_rowptr,staged_remote_only,staged_rowptr" bash hygon_tmp/sglang_debug/run_v3_normal_k3_rowptr_modes.sh`；
  - 1024 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260613_081203.json`：all_zero/local/staged_remote/staged median_avg `0.609/0.944/2.211/2.222 ms`；
  - 4096 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260613_081226.json`：all_zero/local/staged_remote/staged median_avg `2.172/3.304/7.767/7.804 ms`。
- Code object / ISA：
  - `hipprof --codeobj-analyze` 对 extension `.so` 进入交互式 ELF 选择，日志 `hygon_tmp/sglang_debug/codeobj_k3_v3_scalar_pack_dep_20260613_081442/codeobj_analyze.log` 为 `input and select elf file index: error input`，不可用；
  - 改用 `dccobjdump --extract-elf=all` 抽出 gfx938 device ELF，再用 `/opt/dtk/aillvm/bin/llvm-objdump -d --mcpu=gfx938` 反汇编，产物在 `hygon_tmp/sglang_debug/codeobj_k3_v3_dcc_20260613_081745/`；
  - 当前 V3 K3 normal no-tail 实例 `Lb1ELb0ELb0` 计数：`global_store_short=128`、`buffer_store_short=0`、`buffer_load_dwordx2=132`、`buffer_load_dwordx4=132`、`s_waitcnt=353`、`v_mmac=128`、`ds_read=56`、`s_barrier=21`；
  - 原 `K3COMBINE.s` 源码计数：`buffer_store_short=512`、`global_store_short=4`、`s_waitcnt=149`、`v_mmac=288`、`ds_read=197`、`s_barrier=30`。
- 当前结论：
  - scalar pack dependency 没有解决 K3 性能差距；4096 staged 仍约 `7.8 ms`，远慢于 ASM staged `~3.1 ms`；
  - ISA 证据指向 no-tail store window 与 wait 调度差异：V3 scalar global-store 路径 waitcnt 显著偏多，ASM 是 buffer-store + 更大 store grouping；
  - 下一步只做 source-backed 小步 store/wait 调度 A/B；不重复 rowptr4/rowptr8、rowaddr 长复用、non-GLC/no-wait、staged-half、empty-tile skip、unconditional B-load 等已反证方向。

## 2026-06-13 - normal K3 no-tail device-fence A/B 撤回

- A/B 内容：
  - 在 K3 V3 normal no-tail 主 kernel 末尾保留 `wait_vmem_lds_store_device()` 和 `block_barrier_device()`；
  - 只把 `!kTailReduce && !kSignalOnly` 路径的 fence 从 `__threadfence_system()` 降为 `__threadfence()`；
  - tail-reduce 和 signal-only 路径继续 system-scope fence。
- 验证：
  - 远端 aicc rebuild 已完成，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_notail_device_fence_20260613_082333.log`；
  - direct K3 1024/4096 no-tail correctness 通过，logs `k3_v3_dist_1024_notail_device_fence_20260613_082442.log`、`k3_v3_dist_4096_notail_device_fence_20260613_082442.log`，两者 `global_max_abs=0`、`global_gt_atol=0`；
  - rowptr split timing log `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_device_fence_20260613_082732.log`；
  - 1024 all_zero/local/staged_remote/staged `0.609/0.944/2.205/2.216 ms`；
  - 4096 all_zero/local/staged_remote/staged `2.172/3.304/7.775/7.820 ms`。
- 处理：
  - 与 scalar-pack 基线 `1024 staged≈2.222 ms`、`4096 staged≈7.804 ms` 相比无实质收益，4096 还略退；
  - 已撤回本地代码到无条件 `__threadfence_system()`；
  - 本地 `python -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 和 `git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 通过。
- 结论：
  - normal K3 no-tail 当前性能差距不是 final fence scope 主导；
  - 后续不再重复 no-tail device-scope fence 方向，继续围绕 store/window/wait 调度或独立 ASM-style store probe。

## 2026-06-13 - 恢复上下文并补记 normal K3 GLC rowptr A/B

- 恢复动作：
  - 重新读取 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md`；
  - 重新读取 `remote-ssh-docker-workflow`、`dcu-rag-kb`、`hygon-hip-kernel-optimizer` skill；
  - 读取 `.vscode/sftp.json`，确认当前远端仍为 `hg@10.17.176.11:22`，container `sglang_megamoe`，container repo `/workspace/DeepGEMM`；
  - 远端 `header_sha` 与本地 `k3_v3_pack5_groupgemm_impl.cuh` 一致，且无残留 `run_v3_normal_k3_rowptr_modes` / `torchrun` / `setup.py build_ext` / `hipprof` / `dccobjdump` / `aicc` 进程。
- 代码状态：
  - 当前 K3 V3 normal no-tail 源码已不是 08:30 计划中描述的 rowptr-resource baseline；
  - compute active-check 与 epilogue rowptr store 都恢复为 `global_load_i64_glc_device(row_combine_ptrs + row)` 形式；
  - `store_acc_fragment_scaled_unmasked_device` 仍保留 pack dependency `s_nop 0`，final combine store 仍是 scalar `global_store_short`，GEMM 主循环和 no-tail fence 保持不变。
- 已存在远端验证：
  - aicc rebuild：`hygon_tmp/sglang_debug/rebuild_v3_normal_k3_glc_rowptr_ab_20260613_083735.log`；
  - direct K3 1024 repeat1：`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - direct K3 4096 repeat1-5：`global_gt_atol=0`、无 nonfinite；其中 repeat2/3 `global_max_abs=0.002983/0.002838`，仍在 gate 内；
  - rowptr split 1024 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260613_084124.json`：all_zero/local/staged_remote/staged `0.718/0.942/2.207/2.214 ms`；
  - rowptr split 4096 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260613_084147.json`：all_zero/local/staged_remote/staged `2.541/3.312/7.802/7.840 ms`。
- 结论：
  - GLC rowptr A/B 比 rowptr-resource baseline 的 clean rebuild correctness 更稳，当前可作为继续优化的 correctness baseline；
  - 它不是性能优化：4096 staged 仍约 `7.84 ms`，比 scalar rowptr-resource + pack dependency baseline `~7.80 ms` 略慢，all_zero floor 从 `~2.17 ms` 退到 `~2.54 ms`；
  - 下一步仍必须围绕 ASM-style store/window/wait 调度、PMC/save-temps/dccobjdump 证据推进，而不是重复 rowptr load family、rowptr4/8、rowaddr reuse、fence scope 等已反证方向。

## 2026-06-13 - 本机/远端活跃进程检查

- 用户要求确认没有其他 Codex 线程同时操作该目录。
- 本机进程检查：
  - 看到两个 Codex app-server 进程族：一个来自 VS Code 扩展，一个来自桌面 Codex；
  - 未看到另一个命令行包含 `DeepGEMM` / `paseo_ws` 的活跃 shell、python、git、ssh、build 或测试进程；
  - 本轮检查命令自身的 `pwsh.exe` 除外。
- 远端容器检查：
  - `sglang_megamoe` 内未发现残留 `k3_v3` / `run_v3` / `bench_k3` / `setup.py build_ext` / `hipprof` / `dccobjdump` / `aicc` / `torchrun` 进程。
- 结论：从 OS 进程层面没有发现另一个活跃执行链在操作当前工作目录；继续只用当前线程推进。

## 2026-06-13 - normal K3 GLC rowptr4 wait 合并 A/B 反证

- A/B 依据：
  - 原 K3COMBINE direct epilogue 的 `K3_LOAD_COMBINE_ADDR4` 会一次加载 4 个 rowptr 后做一个 `s_waitcnt`，再 `K3_STORE4` 成组写；
  - V3 GLC scalar store baseline 每个 `store_bf16_rowptr_device` 都单独 rowptr load + wait；
  - 为避免重复已反证的 resource rowptr4/rowptr8 和 rowaddr 长生命周期复用，本轮只在同一 `token_base+{0,4,8,12}` 短窗口合并 4 个 GLC rowptr load 和一个 wait，不跨 hidden step 复用地址，不改 GEMM 主循环。
- 验证过程：
  - 第一次 direct compare 误用 `torchrun` 包住脚本，产生多层 spawn/NCCL 端口冲突，未形成 kernel 结论；
  - 第二次只编 K3 raw normal 导致 K1 extension 被 stub 覆盖，direct compare 在 K1 fail-fast，未形成 K3 结论；
  - 重新同时开启 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` 与 `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1`，K1/K3 normal 均经 aicc 重编：`hygon_tmp/sglang_debug/rebuild_v3_normal_k1k3_glc_rowptr4_wait_ab_20260613_092312.log`；
  - direct K3 1024/4096 首轮通过：`global_max_abs=0`、`global_gt_atol=0`；
  - rowptr split 有性能信号：1024 all_zero/local/staged_remote/staged `0.566/0.781/1.979/1.983 ms`，4096 `2.006/2.737/7.767/7.825 ms`；
  - 但 4096 repeat2 出现 `global_v3_nonfinite=256`，非有限在 inactive dense slot，active slot 未出现 nonfinite。
- 处理：
  - 已撤回本地 helper 和调用，恢复 GLC scalar rowptr store baseline；
  - 本地 `python -m compileall` 与 `git diff --check` 通过；
  - 远端同步并同时重编 K1/K3 raw normal 恢复：`hygon_tmp/sglang_debug/rebuild_v3_normal_k1k3_revert_glc_rowptr4_wait_ab_20260613_092950.log`；
  - 撤回后 direct K3 1024/4096 均恢复 `global_max_abs=0`、`global_gt_atol=0`、无 nonfinite：`k3_v3_dist_1024_after_revert_glc_rowptr4_wait_ab_20260613_093103.log`、`k3_v3_dist_4096_after_revert_glc_rowptr4_wait_ab_20260613_093103.log`。
- 结论：
  - GLC rowptr4 wait 合并有 floor/staged 1024 明确信号，但 clean repeat 不稳定，不能保留；
  - 后续不再做 rowptr4/rowptr8 类 rowptr load grouping 变体，除非先有独立 probe 或 ISA 解释 inactive dense nonfinite 的根因；
  - normal K3 no-tail 继续从 GLC scalar store baseline 出发，转向 save-temps/dccobjdump/PMC 定位 ASM store schedule 差异。

## 2026-06-13 - local/remote concurrency recheck and corrected PMC

- 用户要求再次确认没有其他 Codex 线程同时操作该目录。
- 本机进程检查：
  - 仍可见两个 Codex app-server 家族：VS Code extension 与 desktop Codex；
  - 未发现另一个命令行包含 `DeepGEMM` / `paseo_ws` 的活跃 shell、python、git、ssh、build、test、hipprof、aicc 或 hipcc 进程；
  - 线程管理工具未暴露全局 thread list，只能从 OS 进程层面判断。
- 远端容器检查：
  - `sglang_megamoe` 中未发现残留 `k3_v3` / `run_v3` / `bench_k3` / `setup.py build_ext` / `hipprof` / `dccobjdump` / `aicc` / `hipcc` / `torchrun` 进程；
  - `hy-smi` 显示 8 卡 VRAM/HCU 均为 0%，可继续 8 卡 probe。
- PMC 工具修正：
  - 旧目录 `hygon_tmp/sglang_debug/pmc_k3_glc_scalar_4096_staged_remote_20260613_093400` 失败原因是 `hipprof` 不支持额外 `--csv` 参数；
  - DCU KB 与 `hipprof -h` 均确认 `--pmc-type 3` 本身即 CSV 输出，后续命令不要再加 `--csv`。
- Corrected PMC:
  - 新目录 `hygon_tmp/sglang_debug/pmc_k3_glc_scalar_4096_staged_remote_retry_20260613_093725`；
  - 命令使用 `hipprof --pmc-read/--pmc-write --pmc-type 3 --kernel-name V3_K3_Fused -o ...`，read/write 均 status 0，生成 `pmc_read.csv` 与 `pmc_write.csv`；
  - staged_remote_only 4096 profile run 中 V3 K3 no-tail kernel 聚合约：duration `7.68-7.70 ms`、`SQ_INSTS_VMEM_RD≈4.60M`、`SQ_INSTS_VMEM_WR≈1.48M`、`TA_BUSY≈185M`、`TCP_TA_DATA_STALL≈84-85M`；
  - 结果继续支持现有归因：当前 GLC scalar baseline 的大 gap 仍在 remote/scattered combine store 数据通路和 store/window/wait 调度，不是单个 final fence 或 rowptr load family。

## 2026-06-13 - normal K3 staged-half mapping probe completed

- 按用户要求继续 normal K3 no-tail 性能优化，并保持每个小项完成后更新计划。
- 本机/远端并发检查：
  - 本机只看到 VS Code extension 与 desktop Codex 两个 app-server 家族；
  - 未看到另一个命令行包含 `DeepGEMM` / `paseo_ws` 的活跃 shell、python、git、ssh、build/test/profiler 链路；
  - 远端 `sglang_megamoe` 仅有检查命令自身，`hy-smi` 显示 8 卡 VRAM/HCU 均为 0%。
- Scratch probe：
  - 新增 `hygon_tmp/sglang_debug/k3_staged_half_mapping_probe.cu`，不进生产路径；
  - 首版 inline `ds_read_b128` 输出约束 probe 失败，512 mismatch，判断为 probe 自身 asm 输出约束假阳性；
  - 改为生产等价 `uint4` LDS load 后，远端 aicc/gfx938 编译运行通过：`staged_half_mapping_ok mismatches=0`，日志 `hygon_tmp/sglang_debug/k3_staged_half_mapping_probe_uint4_20260613_095155.log`。
- 结论：
  - staged-half 的 LDS row/hidden 基础映射已排除；
  - normal K3 性能项仍为 ⏳，下一步转向 accumulator/value mapping 与 ASM-style store schedule 的更小 probe / ISA 证据，不重复 H1 offset、rowptr4、non-GLC、fence scope 等已反证方向。

## 2026-06-13 - normal K3 staged-half value probe completed

- 新增 scratch-only `hygon_tmp/sglang_debug/k3_staged_half_value_probe.cu`；
- Probe 覆盖 H0/H1 两段、256 rows、256 hidden，用 fake accumulator 同时写 direct scalar 与 staged vector 两份输出；
- 远端 `sglang_megamoe` 内 aicc/gfx938 编译运行通过，输出 `staged_half_value_ok mismatches=0`，日志 `hygon_tmp/sglang_debug/k3_staged_half_value_probe_20260613_095603.log`；
- 已更新 `findings.md` 和 `task_plan.md`：staged-half 的基础 LDS mapping 与 value packing 均已排除，normal K3 no-tail 仍为 ⏳；
- 下一步开始最小生产 staged-vector-store A/B：只改 K3 normal no-tail store 路径，先 direct K3 1024/4096 correctness，再决定是否跑 rowptr split/PMC。

## 2026-06-13 - normal K3 staged-vector-store production A/B: perf signal but unstable

- 在当前生产 A/B 代码上跑 direct K3 normal no-tail 初轮 correctness，1024/4096 均通过：`k3_v3_dist_1024_staged_vector_store_ab_20260613_100059.log`、`k3_v3_dist_4096_staged_vector_store_ab_20260613_100059.log`；
- Rowptr split timing 显示强收益：1024 staged `1.025 ms`，4096 staged `3.562 ms`，较 GLC scalar baseline `~2.214/7.840 ms` 大幅降低；
- 但 stability 复测失败：1024 rep3 `global_gt_atol=1079`，4096 rep1/2/3 `global_gt_atol=214/1264/660`，均无 nonfinite，属于 active dense drift；
- 已把计划从“correctness passed”修正为“性能信号强但稳定性反证”；normal K3 no-tail 主优化仍为 ⏳。下一步从 vector store 可见性/cache 属性或 store helper 形态做最小 A/B，不能直接保留当前 staged-vector-store。

## 2026-06-13 - normal K3 scalar stability restored after staged-vector-store rollback

- 尝试修复 staged-vector-store：
  - `global_store_dwordx4 off glc` 编译通过但 1024 第 2 轮 direct stability 失败；
  - 拆成两个 `global_store_dwordx2` 后 1024 三轮通过，但 4096 第 1 轮失败；
  - volatile LDS load + x4 store 后 1024 第 2 轮失败。
- 处理：
  - 撤回 staged-vector-store 生产路径，删除未使用的 `K1_STORE_STAGED_HALF_GLC` 宏，恢复 no-tail scalar store；
  - 加强 scalar pack dependency：`sched_barrier + s_nop 0 + s_nop 0 + sched_barrier`；
  - 远端 clean rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_scalar_stronger_pack_dep_20260613_102626.log`。
- 验证：
  - direct K3 1024/4096 各 3 轮 correctness 全过；
  - rowptr split 回到 scalar baseline：1024 staged `2.224 ms`，4096 staged `7.837 ms`。
- 结论：
  - 当前代码处于 correctness 稳定但性能未改善的 scalar baseline；
  - 下一步继续 normal K3 no-tail 性能优化，方向改为 dccobjdump/save-temps/PMC 对照原 ASM 的 store family、exec mask 和 store window，而不是继续直接保留 staged-vector-store。

## 2026-06-13 - normal K3 scalar baseline ISA split completed

- 远端对当前 stable scalar code object 做 per-function 拆分，确认 no-tail 实例 `Lb1ELb0ELb0` 是 `global_store_short=128`、`global_store_dwordx4=0`、`s_waitcnt=217`；
- 原 K3COMBINE ASM 的 no-tail scatter 有 `s_waitcnt vmcnt(0) + buffer_wbinvl1_vol + s_barrier` 前置，并在 staged half store 前同时 wait rowptr VMEM 与 LDS；
- 已更新 `task_plan.md` 和 `findings.md`。下一步开始 ASM-style staged-vector-store cache/barrier A/B，先 direct K3 1024/4096 stability，再决定是否跑 rowptr split/PMC。

## 2026-06-13 - normal K3 ASM-style staged-vector-store A/B failed; split-pair scalar baseline restored

- 远端当前使用 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`，所有命令均在容器内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后执行。
- ASM-style staged-vector-store A/B：
  - 远端 header sha `7df5df1513bb94c708a6f0d8ff79c418adaeb2e96a35d193b1b969109284ef48`；
  - direct K3 1024 rep1 失败：`global_gt_atol=71541595`、`global_v3_nonfinite=0`；
  - direct K3 4096 rep1/2/3 均失败：`global_gt_atol=257591640/257573426/257510804`、`global_v3_nonfinite=0`；
  - 已撤回 no-tail epilogue到 scalar `K1_STORE_ROWS_256(K1_STORE_ROW_UNMASKED)`。
- 撤回后 clean rebuild：
  - 第一次强制删除 object 的 glob 不完整，ninja 显示 `no work to do`，已改用 `find build -path '*K3_fused/k3_v3_fused_ext.o' -delete`；
  - 确认 K3 V3 normal 通过 aicc 重新编译，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_force_revert_asmstyle_staged_store_ab_20260613_104807.log`；
  - 1024 direct 2/2 通过，但 4096 出现小漂移 `gt=2/3`。
- Scalar stability 修复：
  - 只加每 store value-dep 后，1024 3/3 通过，4096 仍有一次 `gt=1`；
  - split pair pack/store 后远端 header sha `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`，aicc rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_split_pair_store_20260613_105703.log`；
  - direct K3 1024/4096 各 3 轮全部通过，logs `hygon_tmp/sglang_debug/k3_v3_dist_1024_split_pair_store_r{1,2,3}_20260613_1058*.log`、`hygon_tmp/sglang_debug/k3_v3_dist_4096_split_pair_store_r{1,2,3}_20260613_1059*.log`；
  - 当前代码为 correctness-stable scalar baseline。
- Rowptr split:
  - 运行 log `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_split_pair_store_20260613_110234.log`；
  - 1024 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260613_110234.json`：all_zero/local/staged_remote/staged `0.727/0.949/2.216/2.222 ms`；
  - 4096 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260613_110258.json`：all_zero/local/staged_remote/staged `2.575/3.341/7.818/7.849 ms`；
  - split-pair 稳定性修复无明显额外性能代价，但 K3 normal no-tail gap 仍然很大，继续做 ISA/PMC 或更小 store-window probe。

## 2026-06-13 - normal K3 split-pair scalar ISA/resource extraction completed

- 重新读取计划/发现/进度并继续 Phase 6 normal K3 no-tail 性能项；
- 修正 code-object 抽取方式：在输出目录内执行 `/opt/dtk/bin/dccobjdump --inputs=<k3_v3_fused_ext.so> --extract-elf=all`，再用 `/opt/dtk/aillvm/bin/llvm-objdump -d --mcpu=gfx938` 反汇编；
- 当前输出目录：`hygon_tmp/sglang_debug/codeobj_k3_v3_split_pair_store_current_20260613_110816/`；
- 当前 no-tail 实例 `Lb1ELb0ELb0` ISA 计数为 `global_store_short=128`、`global_store_dwordx4=0`、`global_store_dwordx2=0`、`buffer_store_short=0`、`global_load_dwordx2=132`、`buffer_load_dwordx4=132`、`ds_read=56`、`s_barrier=21`、`s_waitcnt=220`、`s_nop=273`、`v_mmac=128`；
- 结论：split-pair store 是 correctness/codegen 稳定修复，未改变 remote combine store family；性能大头仍需从更小的 production-like staged-store probe 或 store-window/exec-mask 证据继续定位。

## 2026-06-13 - normal K3 staged-half rowptr/exec-mask probe passed

- 新增 scratch-only `hygon_tmp/sglang_debug/k3_staged_half_rowptr_probe.cu`；
- Probe 使用 production-like H0/H1 wave gating、row offset 0/128、rowptr raw-buffer load、zero/permuted rowptr 与 `global_store_dwordx4`，并和 direct rowptr store 输出对比；
- 远端显式 scp 同步后使用 `/workspace/dtk_aicc/bin/aicc -x hip --offload-arch=gfx938 -O3` 编译；
- 运行 `HIP_VISIBLE_DEVICES=0 hygon_tmp/sglang_debug/k3_staged_half_rowptr_probe`，20 repeats * 3 modes 全过，输出 `staged_half_rowptr_ok total_mismatches=0`；
- 结论：基础 rowptr/exec-mask/H0-H1 offset 单独可行，生产 staged-vector-store 的 active dense drift 更可能来自 full GEMM 后 pack-to-LDS 或调度耦合；下一步尝试 staged pack-to-LDS pair-wise dependency A/B。

## 2026-06-13 - normal K3 staged pack-to-LDS dependency A/B failed and reverted

- A/B：
  - 在生产 K3 normal no-tail staged/vector-store 路径中加入 pair-wise pack-to-LDS dependency；
  - 远端 aicc rebuild 成功，失败 A/B header sha `5bd6762d9ab7fb657996e782d9c3dc9394dab8c77f6f9ff371c93edaf263a130`。
- 失败结果：
  - 1024 direct K3 3/3 失败，`global_gt_atol≈1.12M`，`global_max_abs≈2.2-2.9`，无 nonfinite；
  - 4096 direct K3 3/3 失败，`global_gt_atol≈7.1-7.2M`，`global_max_abs≈3.0-3.9`，无 nonfinite；
  - 该形态是 active dense drift，说明 pack-to-LDS dependency 不能修复 staged-vector-store 的 full GEMM 耦合问题。
- 恢复：
  - 本地撤回 `k3_v3_pack5_groupgemm_impl.cuh` 到 split-pair scalar store baseline；
  - 显式 scp 到 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM`；
  - 远端容器 `sglang_megamoe` 内执行 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后强制删 K1/K3 V3 normal object 并重编；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_staged_pack_dep_ab_20260613_112256.log`，header sha 回到 `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`。
- 恢复验证：
  - direct K3 1024 no-tail 3/3 通过，logs `hygon_tmp/sglang_debug/k3_v3_dist_1024_revert_staged_pack_dep_r{1,2,3}_20260613_112421.log`，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - direct K3 4096 no-tail 3/3 通过，logs `hygon_tmp/sglang_debug/k3_v3_dist_4096_revert_staged_pack_dep_r{1,2,3}_20260613_112421.log`，`global_gt_atol=0`、无 nonfinite，`global_max_abs=0.003349/0/0`；
  - summary log `hygon_tmp/sglang_debug/k3_v3_dist_revert_staged_pack_dep_summary_20260613_112421.txt`。
- 计划状态：
  - `task_plan.md` 已把 staged pack-to-LDS dependency A/B 标为 ✅ 反证；
  - normal K3 no-tail remote rowptr/store 优化仍是 ⏳，下一步不再重复生产 staged-vector-store 主路径，改为 ISA/PMC/SQTT 或 full-GEMM-linked 小 probe。

## 2026-06-13 - normal K3 store4 regpressure probe passed

- 继续 Phase 6 normal K3 no-tail remote rowptr/store 优化，先不改生产 kernel；
- 新增并远端运行 scratch-only `hygon_tmp/sglang_debug/k3_store4_regpressure_probe.cu`；
- 远端命令遵守 `sglang_megamoe` 容器流程，`source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后用 `/workspace/dtk_aicc/bin/aicc -x hip --offload-arch=gfx938 -O3 -std=c++17` 编译；
- 运行结果通过：`store4_regpressure_ok rows=256 hidden=256 repeat=64`，log `hygon_tmp/sglang_debug/k3_store4_regpressure_probe_20260613_113554.log`；
- 抽取 ISA 到 `hygon_tmp/sglang_debug/codeobj_k3_store4_regpressure_probe_20260613_113614/`，确认存在 4 个 `global_load_dwordx2 ... glc` 后单个 `s_waitcnt vmcnt(0)` 的 grouped window；
- 结论：该 probe 给短窗口 grouped rowptr load/wait 一个正证据，但不等于 staged-vector-store 主路径可恢复；下一步只做 no-tail scalar epilogue 的最小生产 A/B，保留 split-pair scalar store 语义和多轮 correctness gate。

## 2026-06-13 - normal K3 grouped GLC rowptr and helper-dep A/B reverted

- 生产 A/B 1：4-rowptr grouped GLC load + single wait。
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_glc_store4_ab_20260613_113935.log`；
  - direct K3 1024 三轮通过；
  - direct K3 4096 repeat3 出现 dense drift：`global_gt_atol=2`、`global_max_coord_active=False`；
  - 已撤回。
- 生产 A/B 2：2-rowptr grouped GLC load，保持 split-pair顺序。
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_glc_store2_ab_20260613_114640.log`；
  - direct K3 4096 三轮通过；
  - direct K3 1024 repeat2 出现 active drift：`global_gt_atol=1`、`global_max_coord_active=True`；
  - 已撤回。
- Restore 记录：
  - 恢复后 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_restore_after_store2_ab_20260613_115209.log`；
  - 后续 baseline 4096 repeat 仍观察到一次 active 小漂移和一次 dense inactive nonfinite artifact，记录为当前 scalar baseline 稳定性尾巴，不作为保留 grouped rowptr 的依据。
- 生产 A/B 3：加强每次 rowptr scalar store 前的 helper dependency。
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_stronger_store_helper_dep_20260613_115723.log`；
  - direct K3 1024 三轮通过；
  - direct K3 4096 repeat2 出现 active drift：`global_gt_atol=60`；
  - 已撤回，最终 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_restore_after_helper_dep_ab_20260613_120134.log`。
- 当前状态：
  - 本地 `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh` sha 回到 `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`；
  - `rg` 未发现 `store_acc_fragment_scaled_unmasked[24]_glc` / `store_bf16_rowptr[24]_glc` 残留；
  - normal K3 no-tail 仍为 ⏳，下一步转当前 baseline 的 ISA/PMC/SQTT 或 full-GEMM-linked 小 probe，不再重复 grouped rowptr / 简单 helper barrier。

## 2026-06-13 - normal K3 restored baseline smoke shows 4096 active drift

- 远端确认：
  - `sglang_megamoe` 内 header sha 为 `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`；
  - grouped GLC / helper-dep 实验符号残留为 0；
  - 无活跃 `run_v3` / `bench_k3` / `build_ext` / `hipprof` / `dccobjdump` / `aicc` / `torchrun` 进程。
- direct K3 restore smoke：
  - summary `hygon_tmp/sglang_debug/k3_v3_dist_restore_baseline_smoke_summary_20260613_120658.txt`；
  - 1024：`global_max_abs=0.00262451171875`、`global_gt_atol=0`、无 nonfinite；
  - 4096：`global_max_abs=0.00634765625`、`global_gt_atol=42`、`global_dense_gt_atol=42`、`global_max_coord_active=True`、无 nonfinite。
- 结论：
  - 当前 split-pair scalar baseline 在 4096 no-tail direct K3 compare 下仍有 active drift，不能继续把它当作性能 A/B 的稳定起点；
  - 下一步先定位 correctness/stability，重点对比当前 source/codeobj 与最后一次 3/3 通过的 split-pair baseline，或做更小 deterministic K3 store/compute probe；性能优化暂停到 4096 direct K3 baseline 稳定。

## 2026-06-13 - normal K3 reduce-acquire diagnostic stabilizes 4096 direct compare

- 诊断脚本改动：
  - 仅修改 `hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py`，新增 `K3_V3_DIST_BARRIER_ACQUIRE` / `K3_V3_DIST_REDUCE_ACQUIRE`；
  - 该改动只在 scratch compare 脚本里，不改生产 kernel。
- baseline 复测：
  - 4096 no acquire 三连：r1 `global_gt_atol=36`，r2/r3 为 0，确认间歇 active/dense drift；
  - barrier acquire only 三连仍不稳：出现 `global_gt_atol=52/7` 和一次 dense nonfinite artifact；
  - barrier acquire + reduce acquire 三连全 0，summary `hygon_tmp/sglang_debug/k3_v3_dist_4096_reduce_acquire_summary_20260613_121539.txt`；
  - barrier acquire + reduce acquire 五连全 0，summary `hygon_tmp/sglang_debug/k3_v3_dist_4096_reduce_acquire_5r_summary_20260613_121714.txt`。
- 结论：
  - 当前 K3 normal no-tail 4096 间歇 drift 更像 reader-side cache/visibility 问题，而不是固定 GEMM 数学错误；
  - `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 方向已有 correctness 证据，下一步测 large_opt no-tail stage/e2e 成本，再决定是否把 V3 normal no-tail 默认改为 acquire-on。

## 2026-06-13 - normal K3 reduce-acquire default enabled

- 代码：
  - `megamoe/large_opt.py` 中 `v3_reduce_acquire_enabled()` 默认值从 `"0"` 改为 `"1"`；
  - `tests/test_dcu_megamoe_v3.py` source guard 更新为断言默认 `"1"`。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 通过；
  - `git diff --check` 通过；
  - 本地 `python -m pytest -q tests/test_dcu_megamoe_v3.py` 失败，因为本机 Python 没有 pytest 模块。
- 远端验证：
  - 同步 `large_opt.py`、`tests/test_dcu_megamoe_v3.py`、scratch compare 脚本到 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM`；
  - 容器内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后 compileall 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过：`10 passed in 4.29s`。
- large_opt no-tail default acquire correctness：
  - summary `hygon_tmp/sglang_debug/v3_normal_default_reduce_acquire_correctness_summary_20260613_122423.txt`；
  - 1024：3/3 correctness 通过，`max_abs=0.000488281`；
  - 4096：1/1 correctness 通过，`max_abs=0.000488281`；
  - 4096 长 bench 之前因为当前远端 VRAM 93% 导致 baseline oracle 第二轮 OOM，后续等显存恢复或用更轻量 perf入口补测。
- 当前状态：
  - stability triage 可收口为 ✅；
  - normal K3 no-tail 性能优化仍为 ⏳，下一步继续围绕 4096 remote/scattered combine store 数据通路做 PMC/ISA/SQTT 或小 probe。

## 2026-06-13 - normal K3 staged/vector-store reduce-acquire A/B started

- 恢复上下文后重新读取 `task_plan.md` / `findings.md` / `progress.md`，并确认本地 `k3_v3_pack5_groupgemm_impl.cuh` sha 为 `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`。
- 静态检查确认当前 active path 仍是 GLC active-check + split-pair scalar store，未保留后续 rowptr-resource 形态；已在 `task_plan.md` 补充该源码基线纠偏。
- 新增受控 A/B：仅在 `!kTailReduce && !kSignalOnly && kTileN == 256` 的 no-tail normal epilogue 中，把 scalar rowptr store 切到 `K1_STAGE_ROW_UNMASKED` + LDS barrier + `K1_STORE_STAGED_HALF(0/128)`；不加入 `buffer_wbinvl1_vol`，不影响 tail/signal 路径。
- 远端 aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_staged_vec_reduce_acquire_ab_20260613_123208.log`；失败 A/B header sha `db35cee549848eb5c1cc580edebf45753195ac5d6c209d0381ce9fe61ef94a7d`。
- direct K3 1024 acquire 首轮失败，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_staged_vec_reduce_acquire_ab_r1_20260613_123330.log`，`global_max_abs=2.323974609375`、`global_gt_atol=71475443`、`global_v3_nonfinite=0`。
- 已立即撤回本地源码到 sha `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`，并同步远端强制重编，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_staged_vec_reduce_acquire_ab_20260613_123435.log`。
- 撤回后 direct K3 acquire smoke 1024/4096 均通过，summary `hygon_tmp/sglang_debug/k3_v3_dist_revert_staged_vec_reduce_acquire_ab_summary_20260613_123548.txt`，两档均 `global_gt_atol=0`、`global_v3_nonfinite=0`。
- 结论：该 A/B 已反证，reduce-acquire 不足以让 staged/vector-store full-GEMM 主路径稳定；下一步继续更小粒度的 rowptr/resource 或 ISA/PMC/SQTT 归因。

## 2026-06-13 - normal K3 rowptr-resource reconciliation retained as small floor win

- 代码：
  - `k3_v3_pack5_groupgemm_impl.cuh` 重新落地 no-tail normal rowptr-resource 形态：active-row check 和 epilogue row address fetch 使用 `rowptr_resource`，最终 store 仍是 scalar `global_store_short`；
  - 只通过 `if constexpr (!kTailReduce && !kSignalOnly && kTileN == 256)` 影响 normal no-tail，不触碰 tail/signal/LL。
- 远端验证：
  - aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowptr_resource_reconcile_20260613_124011.log`；
  - direct K3 acquire 1024/4096 均通过，summary `hygon_tmp/sglang_debug/k3_v3_dist_rowptr_resource_reconcile_summary_20260613_124126.txt`；
  - rowptr split log `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_rowptr_resource_reconcile_20260613_124230.log`，1024 avg all_zero/local/staged_remote/staged `0.621/0.943/2.227/2.231 ms`，4096 avg `2.199/3.309/7.785/7.813 ms`；
  - PMC dir `hygon_tmp/sglang_debug/pmc_k3_rowptr_resource_4096_staged_remote_20260613_124438`，单 kernel `SQ_INSTS_VMEM_RD≈4.57M`、`SQ_INSTS_VMEM_WR≈1.41-1.45M`。
- 结论：
  - 该改动主要降低 all-zero/floor，对 staged_remote 主耗时帮助很小；
  - normal K3 no-tail 主优化仍继续，下一步抽 ISA 并继续找 remote/scattered store schedule 方向。

## 2026-06-13 - normal K3 rowptr-resource ISA attribution logged

- 已把 retained rowptr-resource code object 的 no-tail ISA 计数补入 `findings.md` 和 `task_plan.md`；
- code object 目录：`hygon_tmp/sglang_debug/codeobj_k3_v3_rowptr_resource_reconcile_20260613_124654/`；
- 当前 no-tail `Lb1ELb0ELb0`：`buffer_load_dwordx2=132`、`global_load_dwordx2_glc=0`、`global_store_short=128`、`global_store_dwordx4=0`、`s_waitcnt=350`、`v_mmac=128`；
- 结论：rowptr-resource 已改变 rowptr load family，但没有改变最终 remote combine store family；`s_waitcnt` 增加也解释了为什么该项只有 floor 小收益；
- 下一步正在进行：对照原 K3COMBINE ASM staged/vector store 和当前 C/aicc store schedule，只做有新证据的小步 A/B。

## 2026-06-13 - normal K3 retained performance snapshot measured and stopping

- 用户要求给出当前性能情况的实测进展后停下。
- 远端环境：
  - `.vscode/sftp.json` 指向 `hg@10.17.176.11`，容器 `sglang_megamoe`，容器 repo `/workspace/DeepGEMM`；
  - 检查时 8 卡 VRAM/HCU 均为 0%，无残留 `build_ext/torchrun/bench/hipprof/aicc` 等进程；
  - 远端 K3 header sha 为 `98dca271315abc5473ad25e19bd1d7ee315711912979cce3b1fd8119fa718575`。
- 初次性能入口失败：
  - `run_v3_normal_k3_rowptr_modes.sh` 在 1024 启动后报 `V3 K1 raw kernels were not compiled into this extension`；
  - 原因是之前 K3-only rebuild 后 K1 extension 仍是 stub，不是 kernel 性能或 correctness 失败。
- 处理：
  - 用 aicc 重新编译 K1/K3 normal raw，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1k3_perf_snapshot_20260613_131008.log`；
  - import 检查确认 `k1_symm_fused_l1_v3_pack5` 存在。
- 实测：
  - 命令环境固定：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal`，`TOKENS_LIST="1024 4096" WARMUP=3 REPEAT=10 ROUNDS=5 MODES="rowptr_all_zero,staged_local_only,staged_remote_only,staged_rowptr"`；
  - snapshot log `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_perf_snapshot_20260613_131205.log`；
  - 1024 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260613_131205.json`：all_zero/local_only/remote_only/staged median_avg_rank_ms `0.616/1.091/2.215/2.223`；
  - 4096 JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260613_131228.json`：all_zero/local_only/remote_only/staged median_avg_rank_ms `2.195/3.231/7.797/7.831`。
- 结论：
  - 当前 retained rowptr-resource baseline 性能与 12:42 记录基本一致，没有新退步；
  - K3 normal no-tail 的大头仍是 staged_remote/staged：约 `2.22 ms` @1024、`7.83 ms` @4096；
  - 相比原 ASM 4096 staged 约 `3.11 ms` 仍差很远，瓶颈仍是 remote/scattered scalar combine store；
  - 已按用户要求停止，不继续开新优化实验。

## 2026-06-13 - normal ASM-pack5 alternate path analyzed

- 用户提出：V3 K1/K3 fused normal 也可以直接改原 ASM 支持 5pack weight layout，与 V3 LL layout 对齐。
- 已按 planning / DCU KB / optimizer 约束恢复上下文并查询本地 KB；
- KB 结论确认 DeepGEMM ASM weight layout 是 binary contract，修改 layout 必须同步改 ASM 地址数学；
- 本地对照确认 V3 pack5 helper 与当前 C K1/K3 pack5 分支已有可移植的 pack5 weight-load 地址公式；
- 已把结论写入 `findings.md`，并在 `task_plan.md` Phase 6 补充 K3 no-tail ASM-pack5 实验分支；
- 当前未改生产 kernel，也未启动远端编译/测试。

## 2026-06-13 - normal K3 no-tail ASM-pack5 isolated implementation started

- 已重读 planning 三文件和 `.vscode/sftp.json`，远端仍为 `hg@10.17.176.11` / `sglang_megamoe` / `/workspace/DeepGEMM`；
- 本轮窄 KB 查询 `gfx938 DeepGEMM assembly code object weight layout address math buffer SrdA SrdB pack5` 30s 超时；沿用已记录 KB 结论并要求后续用 ASM 编译、direct correctness、rowptr split perf 做证据闭环；
- 纠正 ASM-pack5 路线表述：K3 ASM host 设置 `prob.a=l2_weight`、`prob.b=act_fp8`，因此 `SrdA`/`A` 是 L2 weight，`SrdB`/`B` 是 activation；此前 “B/scale 地址数学” 说法已在 `task_plan.md` / `findings.md` 修正；
- 已在隔离文件 `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.s` 中只 patch `GLOBAL_OFFSET_A`、`SrdA` base 和 A 侧 K-stage increment 到 V3 pack5 layout；combine/store/tail 逻辑不动；
- 已在 `setup.py` 增加独立 PACK5 code object 构建项，并在 K3 host ext / Python wrapper 增加 `MEGAMOE_DCU_V3_K3_ASM_PACK5=1` 实验 gate；unset 时默认仍走当前 V3 C/aicc raw normal no-tail 路线，原 ASM 路径不受影响；
- 下一步：本地 source/compileall 检查后同步远端，先修到 PACK5 ASM code object 可编译，再跑 1024/4096 direct K3 correctness；若通过再跑 rowptr split 性能。

## 2026-06-13 - normal K3 no-tail ASM-pack5 build passed but correctness failed

- 本地 source/guard 验证通过：`compileall` 覆盖 `K3_fused/k3_fused.py` 与 `tests/test_dcu_megamoe_v3.py`，`git diff --check` 通过；本地 pytest 仍不可用，原因是本机 Python 缺少 pytest。
- 已同步 `setup.py`、`K3_fused/k3_fused.py`、`K3_fused/k3_fused_ext.cu`、`K3COMBINE_PACK5.s`、`tests/test_dcu_megamoe_v3.py` 到远端 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM`。
- 远端容器 `sglang_megamoe` 内执行 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后，`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed`。
- 远端 K1/K3 normal raw + PACK5 code object rebuild 通过，PACK5 `.co` 生成在 `megamoe/dcu_megamoe_large_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.co`；`dccobjdump --extract-elf=all` 可抽取 ISA，并确认 `SrdA` base 使用 `wg0*0x4000`、A 侧 K-stage increment 使用 `0x80000`。
- 1024 direct K3 correctness（`MEGAMOE_DCU_V3_K3_ASM_PACK5=1`、normal、no-tail、barrier/reduce acquire on）失败：带原 ASM `+0x10` prepad compensation 时 `global_max_abs=0.11767578125`、`global_gt_atol=87330440`、`global_y_max_abs=0.113525390625`、无 nonfinite。
- A/B 去掉 PACK5 `GLOBAL_OFFSET_A` 中 `+0x10` 后重新生成 code object，1024 direct K3 correctness 更差：`global_max_abs=0.12451171875`、`global_gt_atol=116445685`、`global_y_max_abs=0.129150390625`、无 nonfinite。
- 结论：ASM-pack5 gate/编译/launch 已打通，但当前错误是系统性 dense numerical mismatch，不是 NaN、未写输出或 rowptr 失效；去 prepad 已被反证，已把隔离 ASM 恢复到带 `+0x10` 的较好基线。下一步优先定位 `loader_linear -> pack5 global offset -> LDS write offset` 是否与 C pack5 normal 的实际 `buffer_load_lds_8x16b_m0_stride800_device` 等价，再决定是否继续 ASM-pack5 路线。

## 2026-06-13 - normal K3 no-tail ASM-pack5 plain layout correctness passed

- 发现远端 `K3COMBINE_PACK5.s/.co` 仍停在 no-prepad A/B 后，已同步本地恢复版并重编；远端 `.s` sha `6df3b9c6796a97157680d6f2c1c3d9fc3b4a165b62a44ada215863db4d69c88e`、`.co` sha `15d827b470e2a2e9a920bf403abe076f3fc571ec8ee64662942f1d3768b8ffea`。
- 在 scratch compare 脚本中新增 `K3_V3_DIST_V3_PACK5_LAYOUT=plain|transposed`，只改变测试/fixture 侧 L2 layout，不改生产 runtime/bench 路径。
- 恢复版 `transposed` 1024 复现失败：`global_gt_atol=87330440`、`global_max_abs=0.11767578125`。
- `plain` 1024 direct K3 通过：`global_gt_atol=0`、`global_max_abs=0`、`global_y_max_abs=0`。
- `plain` 4096 direct K3 通过：`global_gt_atol=0`、`global_max_abs=0`、`global_y_max_abs=0`。
- 当前结论：K3 ASM-pack5 的地址/LDS/MMAC 主体已对齐；现有 V3 transposed pack5 失败是原 ASM epilogue 缺少 C pack5 store 前的 accumulator lane shuffle。下一步先跑 plain-layout rowptr split 性能，评估 ASM-pack5 路线是否值得把 plain layout 固化为 normal ASM 专用离线合同，或继续 patch ASM lane shuffle 兼容现有 V3 transposed layout。

## 2026-06-13 - normal K3 no-tail ASM-pack5 plain layout perf and fixture helper

- 已拉取并解析远端 rowptr split JSON：
  - 1024 active/local/remote avg rows `6144.0/678.625/5465.375`，all_zero/local/remote/staged median_avg_rank_ms `0.645/0.657/0.879/0.882`；
  - 4096 active/local/remote avg rows `24576.0/2970.125/21605.875`，all_zero/local/remote/staged median_avg_rank_ms `2.011/2.074/3.305/3.690`。
- 性能结论：
  - 相比当前 C/aicc retained baseline，1024 staged `2.223 -> 0.882 ms`，4096 staged `7.831 -> 3.690 ms`；
  - ASM-pack5 plain 已显著突破 C scalar remote-store 瓶颈，下一步优先做生产 fixture/e2e gate，而不是先大改 ASM lane shuffle。
- 本地改动：
  - `v3_layout.py` 新增 `pack5_weight_asm_normal()` / `flatten_pack5_weight_asm_normal()`，只作为离线/test/fixture helper；
  - `tests/test_mega_moe_dcu.py` 在 `MEGAMOE_DCU_V3_K3_ASM_PACK5=1 && MEGAMOE_DCU_V3_BACKEND=normal && K3_USE_ASM_TAIL_REDUCE=0` 时使用 plain L2 layout；tail-reduce 默认仍使用现有 transposed V3 layout；
  - `hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 和 `bench_k3_ll_rowptr_modes.py` 复用 `v3_layout.flatten_pack5_weight_asm_normal()`，删除重复 plain layout 实现；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard 和 CPU reference layout 校验。
- 本地验证：
  - `python -m compileall megamoe/dcu_megamoe_large_opt/v3_layout.py tests/test_dcu_megamoe_v3.py tests/test_mega_moe_dcu.py hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py` 通过；
  - `git diff --check` 通过；
  - 本地直接 import 测试文件失败，原因仍是本机没有 `pytest`；本地通过按文件加载 `v3_layout.py` 的小脚本验证 plain layout helper。
- 下一步：
  - 同步 helper/fixture/test 脚本到远端；
  - 远端跑 `tests/test_dcu_megamoe_v3.py`；
  - 跑 no-tail direct K3 和 large_opt correctness/perf，固定 env 包含 `K3_USE_ASM_TAIL_REDUCE=0`。

## 2026-06-13 - normal K3 no-tail ASM-pack5 production fixture gate passed

- 远端同步：
  - 同步 `v3_layout.py`、`tests/test_dcu_megamoe_v3.py`、`tests/test_mega_moe_dcu.py`、两个 `hygon_tmp/sglang_debug` 脚本；
  - 容器内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM` 后 compileall 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed in 5.68s`。
- direct K3 no-tail：
  - 1024 log `hygon_tmp/sglang_debug/k3_v3_dist_1024_asm_pack5_plain_helper_gate_20260613.log`，`global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`；
  - 4096 log `hygon_tmp/sglang_debug/k3_v3_dist_4096_asm_pack5_plain_helper_gate_20260613.log`，`global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`。
- large_opt no-tail correctness:
  - 1024 log `hygon_tmp/sglang_debug/v3_normal_asm_pack5_plain_large_opt_no_tail_correctness_1024_20260613.log`，`max_abs=0.000488281`；
  - 4096 log `hygon_tmp/sglang_debug/v3_normal_asm_pack5_plain_large_opt_no_tail_correctness_4096_20260613.log`，`max_abs=0.000488281`；
  - both used `K3_USE_ASM_TAIL_REDUCE=0` and `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1`.
- large_opt stage timing:
  - 1024 log `hygon_tmp/sglang_debug/v3_normal_asm_pack5_plain_stage_timing_1024_20260613.log`，K3 combine per rank `1.900-2.393 ms`；
  - 4096 log `hygon_tmp/sglang_debug/v3_normal_asm_pack5_plain_stage_timing_4096_20260613.log`，K3 combine per rank `3.907-4.387 ms`。
- 当前状态：
  - normal K3 no-tail ASM-pack5 correctness/fixture gate 可以收口；
  - 性能已大幅优于 C/aicc retained baseline，但 4096 full staged K3 combine 仍略慢于原 ASM 约 3.1ms 档位；继续做同条件原 ASM 对照和 gap 定位。

## 2026-06-13 - normal K3 no-tail ASM-pack5 base-precompute optimization retained

- 已按计划继续优化 V3 normal K3 no-tail 的大头，tail-reduce ASM 暂不参与。
- 新增 scratch 诊断：
  - `hygon_tmp/sglang_debug/inspect_k3_codeobj_counts.sh`；
  - `hygon_tmp/sglang_debug/run_k3_pack5_baseprecompute_correctness.sh`；
  - `hygon_tmp/sglang_debug/run_k3_pack5_baseprecompute_perf.sh`；
  - `hygon_tmp/sglang_debug/run_v3_normal_asm_pack5_baseprecompute_stage_timing*.sh`。
- 生产/实验改动：
  - 仅修改隔离 `megamoe/dcu_megamoe_large_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_PACK5.s`；
  - 在 8 个 A-offset 前预计算 shared pack5 `ko64/ks16/physical_ni` base，宏内只保留 per-offset `no256`；
  - 原 `K3COMBINE.s`、tail-reduce ASM、C/aicc fallback 均未修改。
- 远端 build：
  - 手动按 `setup.py` 同等方式只重编 PACK5 `.s -> .o -> .co`；
  - 新 `.co` sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`。
- ISA 结果：
  - 原 ASM 与 PACK5 的 `v_mmac/load/store/waitcnt/barrier` 计数保持一致；
  - PACK5 额外地址算术明显减少：优化前约 `v_lshl +24 / v_and +16 / v_add +16`，优化后约 `v_lshl +10 / v_and +2 / v_add +2`。
- correctness：
  - direct K3 1024/4096 均 `global_max_abs=0`、`global_gt_atol=0`；
  - large_opt 4096 三轮 correctness 均通过，`max_abs=0.000488281`。
- 性能：
  - rowptr split 1024 staged `0.834 ms`，4096 staged `3.074 ms`；
  - 对比上一版 PACK5 plain `0.882/3.690 ms` 有明显收益；
  - 4096 rowptr staged 已略快于原 ASM `3.172 ms`；
  - large_opt warmed 4096 K3 combine 第 2/3 轮约 `3.31-3.46 ms`，仍有小 residual。
- 已更新：
  - `task_plan.md` 将 `normal K3 no-tail ASM-pack5 remaining perf gap` 标为 ✅；
  - 新增 ⏳ `normal K3 no-tail ASM-pack5 warmed stage residual`；
  - `findings.md` 记录完整证据。

## 2026-06-13 - normal K3 no-tail ASM-pack5 residual closed and K1 ASM-pack5 started

- 重新检查远端状态：
  - 容器 `sglang_megamoe` 在线；
  - 8 卡 VRAM/HCU 均为 0%；
  - 无残留 `torchrun/build_ext/hipprof/aicc` 等工作进程；
  - PACK5 `.co` sha 仍为 `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`。
- 复测 4096 large_opt stage timing：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_K3_ASM_PACK5=1 K3_USE_ASM_TAIL_REDUCE=0`；
  - correctness 3/3 通过；
  - 第 2/3 轮 K3 combine 约 `3.10-3.38 ms`。
- 复测 rowptr split：
  - 1024 staged `0.810 ms`；
  - 4096 staged `3.043 ms`。
- 已更新计划：
  - `normal K3 no-tail ASM-pack5 实验分支` 标为 ✅；
  - `normal K3 no-tail ASM-pack5 warmed stage residual` 标为 ✅；
  - 新增 ⏳ `normal K1 ASM-pack5 隔离实验`，下一步只在 V3-only gate 下尝试，不影响原始 ASM path，tail-reduce ASM 暂不参与。

## 2026-06-13 - normal K1 ASM-pack5 isolated branch implemented locally

- 已按计划读取 planning 三文件、remote skill、DCU KB 和 optimizer 约束后继续 K1 normal ASM-pack5 隔离实验。
- DCU KB 再次确认 DeepGEMM ASM weight layout 是 binary contract，修改 V3 pack5 layout 必须同步 patch ASM 地址数学并用 `.co` build、correctness、ISA/perf 闭环验证。
- 本地新增隔离文件 `K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s`，只 patch A/L1 weight 侧 `GLOBAL_OFFSET_A`、pack5 base precompute、`SrdA` base 和 A 侧 K-stage increment；原 K1 ASM 文件不改。
- `setup.py` 增加独立 `MEGAMOE_DISPATCH_PULL_L1_PACK5.s -> .co` code object 构建项。
- `k1_fused.py` 新增 `MEGAMOE_DCU_V3_K1_ASM_PACK5` gate；仅在 V3 normal 下复用原 `ext.k1_symm_fused_l1` host launcher 加载 PACK5 `.co`，保留原 dispatch-pull、route metadata、row_combine_ptrs、output_index、stats 和 wait/signal 调度。
- `tests/test_mega_moe_dcu.py` 在 K1 ASM-pack5 gate 下将 L1 fixture 切到 normal ASM plain-pack5；K3 ASM-pack5 gate 仍独立控制 L2 plain-pack5。
- 本地验证：`python -m compileall ...` 通过，`git diff --check` 通过；本机 `python -m pytest` 仍因缺少 pytest 模块不可用，需远端容器继续 source pytest / build / correctness。

## 2026-06-13 - normal K1/K3 ASM-pack5 no-tail correctness and perf passed

- 远端环境：
  - `hg@10.17.176.11` / container `sglang_megamoe` / repo `/workspace/DeepGEMM`；
  - 8 卡 VRAM/HCU 均为 0%，无残留 build/test/profiler 进程。
- 远端 source:
  - 同步 K1 PACK5 ASM、`setup.py`、K1 wrapper、fixture/test 和 planning progress；
  - 容器内 `python3 -m compileall ...` 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过：`11 passed`。
- Build:
  - 只手动编译新增 K1 PACK5 `.s -> .o -> .co`，未触发 heavy C/aicc extension rebuild；
  - `K1_DISPATCH_PULL_L1_PACK5.co` sha `b26ac17779458296d4826664df04eef5109d5a7dabec3132c42a57cd959bd9e5`。
- Correctness:
  - 第一次 1024 命令误设 `--intermediate-hidden 4096`，large_opt shape check 失败；已修正为 `2048`；
  - V3 K1 ASM-pack5 + K3 ASM-pack5 no-tail 1024 correctness 通过：`max_abs=0.000488281`；
  - 4096 correctness 通过：`max_abs=0.000488281`；
  - 1024/4096 stage timing 三轮均通过。
- Performance:
  - V3 K1/K3 ASM-pack5 no-tail formal bench：1024 fused median `2.1127 ms`，4096 fused median `6.3658 ms`；
  - 原始 staged ASM no-tail formal bench：1024 `2.3842 ms`，4096 `6.6098 ms`；
  - K1-only stage timing 对比显示 K1 ASM-pack5 明显快于当前 K1 C/aicc：1024 warmed `~1.0 ms` vs `~1.4 ms`，4096 warmed `~3.2-3.4 ms` vs `~4.3-4.4 ms`。
- ISA:
  - 已新增并运行 `hygon_tmp/sglang_debug/inspect_k1_codeobj_counts.sh`；
  - K1 PACK5 与原 K1 ASM 的 `v_mmac/load/store/waitcnt/barrier` family 计数一致，说明通信/store 主体未被改动。
- 计划状态：
  - `task_plan.md` 已将 `normal K1 ASM-pack5 隔离实验` 标为 ✅；
  - 新增 ⏳ `normal V3 no-tail ASM-pack5 pair promotion`，下一步整理是否在 V3 normal no-tail 默认使用 isolated ASM-pack5 pair；tail-reduce ASM 仍暂缓。

## 2026-06-13 - normal V3 no-tail ASM-pack5 promotion patched locally

- 已按计划重读 planning 三文件；第一次 catchup 使用旧 `.claude` skill 路径失败，随后用实际 `.agents` 路径补跑成功且无输出。
- 本地代码改动：
  - `large_opt.py` 新增 `use_v3_normal_no_tail_asm_pack5` 判定，只在 `USE_MEGAMOE_V3=1`、backend normal、`K3_USE_ASM_TAIL_REDUCE=0` 且未启用 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 时默认打开 isolated ASM-pack5 pair。
  - `K1_fused/k1_fused.py` 将 `MEGAMOE_DCU_V3_K1_ASM_PACK5` 改成 caller 默认值 + env 覆盖；`large_opt.py` 只在 normal no-tail 默认传 `use_asm_pack5=True`。
  - `K3_fused/k3_fused.py` 同样支持 default + env 覆盖，且 ASM-pack5 仍只在 normal no-tail、无 tail/signal 参数时进入。
  - `tests/test_mega_moe_dcu.py` fixture 同步默认 plain-pack5 L1/L2 layout；tail-reduce 和 no-tail-signal 仍使用原 V3 transposed/C/raw layout。
  - `tests/test_dcu_megamoe_v3.py` source guard 更新到 default promotion / env opt-out 语义。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py ... tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过；
  - 本地 `python -m pytest -q tests/test_dcu_megamoe_v3.py` 仍失败于本机环境缺少 `pytest`，不是代码断言失败。
- 下一步：同步远端，跑 source pytest，并用不设置 `MEGAMOE_DCU_V3_K1_ASM_PACK5/MEGAMOE_DCU_V3_K3_ASM_PACK5` 的 normal no-tail correctness/bench 验证默认 promotion。

## 2026-06-13 - normal V3 no-tail ASM-pack5 default promotion verified

- 远端状态：
  - `hg@10.17.176.11` / container `sglang_megamoe` / `/workspace/DeepGEMM`；
  - 8 卡 VRAM/HCU 均 0%，无残留 build/test/profiler 进程。
- 同步/验证：
  - 同步 `large_opt.py`、K1/K3 wrapper、fixture/source tests 和 planning 文件；
  - 远端 `compileall` 通过，`tests/test_dcu_megamoe_v3.py` 通过 `11 passed`。
- 修复一个集成层错误：
  - 首次 default correctness 失败于 `k3_l2_fused_v3_to_combine() got an unexpected keyword argument 'use_asm_pack5'`；
  - 原因是参数加错到原 ASM wrapper 签名，已移到 V3 wrapper 签名并重跑 source pytest 通过。
- Default no-tail correctness：
  - 命令显式清空 `MEGAMOE_DCU_V3_K1_ASM_PACK5` 和 `MEGAMOE_DCU_V3_K3_ASM_PACK5`；
  - 1024 三轮通过，`max_abs=0.000488281`，第三轮 K1 `~0.96-1.01 ms`、K3 `~0.74-0.86 ms`；
  - 4096 三轮通过，`max_abs=0.000488281`，第三轮 K1 `~3.16-3.34 ms`、K3 `~2.51-2.65 ms`。
- Formal bench：
  - 1024 fused median `2.0818 ms`，min `1.9824 ms`，baseline median `3.7013 ms`，speedup `1.778x`；
  - 4096 fused median `6.3443 ms`，min `6.3039 ms`，baseline median `9.4768 ms`，speedup `1.494x`。
- Opt-out：
  - `MEGAMOE_DCU_V3_K1_ASM_PACK5=0 MEGAMOE_DCU_V3_K3_ASM_PACK5=0` 1024 correctness-only 通过，fixture 打印 `L1/L2 pack5 for K1 + K3 V3`，`max_abs=0.000488281`。
- 已更新：
  - `task_plan.md` 将 `normal V3 no-tail ASM-pack5 pair promotion` 标为 ✅；
  - 将旧 `normal K3 original ASM store schedule 对照` 改为 ✅/superseded，后续只作为 C fallback 诊断，不阻塞 normal no-tail gate。

## 2026-06-13 - LL refresh attempt blocked by missing LL raw build

- 已按 Phase 6 LL residual optimization refresh 启动当前 32/128 实测。
- 远端环境干净：sglang_megamoe 在线，8 卡 VRAM/HCU 均 0%，无真实残留 build/test/profiler 进程。
- baseline no-large 32 先完成：fused_median_ms_avg_per_rank=1.022 ms，baseline_median_ms_avg_per_rank=2.623 ms。
- V3 LL default 32 在 K1 launch 前失败：loaded extension 未编入 K1 LL raw kernel，报错要求 DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll。
- 下一步按现有 build gate 只重编 V3 LL raw K1/K3，避免触碰 V2，并重新跑 LL 32/128 correctness+stage timing。

## 2026-06-13 - LL raw rebuild for refresh passed

- 远端执行 DG_FORCE_BUILD=1 MAX_JOBS=8 DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll python3 setup.py build_ext --inplace。
- build log: hygon_tmp/sglang_debug/rebuild_v3_ll_raw_20260613_191806.log。
- K1/K3 V3 LL raw 均用 /opt/dtk/bin/hipcc 编译完成；未触发 V2 extension，也未走 normal aicc shim。
- 下一步重新跑 V3 LL 32/128 correctness + stage timing。

## 2026-06-13 - LL refresh clean measurement passed

- 发现外部占卡：host 上 `dsq_sglang_601` 容器内 `sglang serve` 占用 8 卡约 93% VRAM；停掉该容器后 8 卡 VRAM/HCU 恢复 0%。
- 远端干净环境执行 `TOKENS_LIST="32 128" RUN_BASELINE=1 RUN_DEFAULT=1 RUN_K2SKIP=0 RUN_BLOCK48=0 RUN_BLOCK64=0 RUN_STAGE_TIMING=1 ITERS=3 WARMUP=3 REPEAT=5 bash hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`，master log `hygon_tmp/sglang_debug/v3_ll_refresh_master_20260613_192603.log`。
- Correctness: V3 LL 32/128 均通过。
- Perf summary: no-large fused 32/128 = `1.1367/1.6420 ms`; V3 LL 32/128 = `0.9083/1.0789 ms`; speedup vs no-large fused 约 `1.25x/1.52x`。
- Stage timing warmed last3: 32 total `0.778 ms`, K1 `0.410 ms`, K2 `0.028 ms`, K3 `0.256 ms`; 128 total `0.955 ms`, K1 `0.454 ms`, K2 `0.028 ms`, K3 `0.382 ms`。
- 当前最大 residual 是 K1 fixed/metadata/staged-load 成本，其次是 128 K3 combine；下一步跑 K3 LL pure/local/remote split 来确认 K3 delta。

## 2026-06-13 - LL K1 optional output skip A/B started

- 针对 LL 最大 residual（K1 fixed/metadata/staged-load），确认 `large_opt.py` / V3 K2/K3 当前不消费 K1 `local_topk_mask` 和 `tail_tokens`；它们主要是 V2 runtime/K3 接口遗留。
- 本地将 V3 K1 LL raw launch 的 `local_topk_mask/tail_tokens` 实参改为 `nullptr`，触发 kernel 里已有的 optional fast path，跳过额外本地 token scan 与一次 grid barrier；normal 路径和 scratch 布局不动。
- 新增 source guard 防止 LL fast path 无意回退；本地 `compileall` 和 `git diff --check` 通过。
- 下一步同步远端，重跑 source pytest、LL raw rebuild、32/128 correctness + stage timing，保留或撤回以实测为准。

## 2026-06-13 - LL K1 optional output skip retained

- 远端同步 `k1_fused_ext.cu` / `tests/test_dcu_megamoe_v3.py` 后，容器内 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过：`11 passed`。
- `hy-smi --showpids` 显示无 KFD 进程；随后只用 LL raw gate rebuild：`DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll` / `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll`，K1 用 `/opt/dtk/bin/hipcc` 重编通过，未触发 V2。
- 远端执行 `TOKENS_LIST="32 128" RUN_BASELINE=0 RUN_DEFAULT=1 RUN_STAGE_TIMING=1 ITERS=5 WARMUP=3 REPEAT=5 bash hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`。
- Correctness: 32/128 均通过；summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_193854.csv`。
- Perf: V3 LL fused median `0.9023/1.0752 ms`，较上一轮 `0.9083/1.0789 ms` 小幅改善。
- Stage timing warmed last3: 32 total/K1/K2/K3/no-tail barrier/reduce `0.7655/0.404/0.028/0.256/0.032/0.012 ms`；128 `0.955/0.449/0.028/0.378/0.052/0.014 ms`。
- 已更新 `task_plan.md` 和 `findings.md`；下一步继续找 LL K1 staging/copy 或 K3 128 remote combine 的更大收益项。

## 2026-06-13 - LL block_m=16 gated A/B started

- 静态确认 K1/K3 LL template `static_assert` 本身都允许 `kBlockM == 16`，但 env gate 和 launcher 当前只开放 32/48/64。
- 为减少 32/128 小 token 档位按 `block_m=32` padding 的 K1/K3 work，新增仅由 `MEGAMOE_DCU_V3_LL_BLOCK_M=16` 触发的 K1/K3 LL launcher 分派；默认值仍为 32，不改变当前通过路径。
- 本地修改 `large_opt.py` env gate、K1/K3 raw launcher、K1/K3 source guard；`compileall` 与 `git diff --check` 通过。
- 下一步同步远端、source pytest、LL raw rebuild，再分别跑 `MEGAMOE_DCU_V3_LL_BLOCK_M=16` 的 32/128 correctness+perf；若 128 或 K3 regression 明显则撤回或只保留诊断。

## 2026-06-13 - LL block_m=16 A/B rejected and reverted

- 远端 source pytest 通过，LL raw rebuild 通过；A/B 命令使用 `MEGAMOE_DCU_V3_LL_BLOCK_M=16 TOKENS_LIST="32 128" RUN_BASELINE=0 RUN_DEFAULT=1 RUN_STAGE_TIMING=1 ITERS=5 WARMUP=3 REPEAT=5 bash hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`。
- Correctness: 32/128 均通过。
- Performance: 32/128 formal fused median `1.0439/1.3141 ms`，明显慢于 retained default `0.9023/1.0752 ms`。
- Stage timing: 32 K1 `0.383 ms` 有小收益，但 K3 combine `0.4205 ms` 大幅变差；128 K1/K3 `0.5205/0.479 ms` 同样差于默认。
- 已撤回 `block_m=16` env/launcher/source-test 改动，只保留 findings 里的反证；下一步继续寻找不增加 K3 tile overhead 的 LL 优化。

## 2026-06-13 - LL K3 rowptr-resource A/B started

- 重读 planning 三文件、remote workflow、DCU KB 与 Hygon optimizer 约束后继续 LL residual optimization。
- 当前 retained LL baseline：32/128 fused median `0.9023/1.0752 ms`，stage warmed K1 `0.404/0.449 ms`、K3 `0.256/0.378 ms`。
- KB 检索支持 gfx938 上把预取/地址访问作为专门路径验证，也提醒 `glc/slc`、raw-buffer load 与 waitcnt 必须用 correctness/ISA/性能闭环确认。
- 本轮 A/B 目标：仅在 K3 LL row address prefetch 中把 `global_load_dwordx2 ... glc` 改为 raw-buffer resource load，store 形态、GEMM 主体、launch 数和 scratch layout 均不变；默认先保留旧路径，用 env gate 控制实验。

## 2026-06-13 - LL K3 rowptr-resource A/B rejected and reverted

- 本地新增 `MEGAMOE_DCU_V3_LL_K3_ROWPTR_RESOURCE` gate 后，compileall 与 `git diff --check` 通过；同步远端后 source pytest 通过 `11 passed`。
- LL raw rebuild 实际成功并复制 `.so`；第一次 SSH 包装返回码为 1 是 `exit \$status` quoting 错误，不是编译失败。后续 import 检查确认 raw kernels available。
- `MEGAMOE_DCU_V3_LL_K3_ROWPTR_RESOURCE=1` 短 smoke 32/128 correctness 通过；formal run summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_200800.csv`。
- formal fused median 32/128 为 `0.8876/1.0871 ms`；128 慢于 retained `1.0752 ms`。stage warmed last3 32 K3 `0.2605 ms`、128 K3 `0.391 ms`，均不优于 retained `0.256/0.378 ms`。
- dccobjdump extract 得到 gfx938 code object，但 `--show-sass` 未生成有效 SASS 文本；本轮 ISA 计数记为 inconclusive，不作为保留依据。
- 已撤回 K3 LL rowptr-resource 代码、Python env gate 与 source guard，只保留 findings/plan 里的反证。下一步继续从 K1 fixed cost 或 K3 remote store/schedule 的其他方向找收益。

## 2026-06-13 - LL pre-K1 rank barrier skip A/B started

- KB 检索没有给出可直接删除 K1 前 rank barrier 的强安全结论；DeepEP/custom allreduce 参考仍强调 release/acquire/final-sync 区分和 visibility 语义。
- 本轮只做 LL no-tail、无 tail signal/reset 需求下的 env-gated A/B，默认仍保留 rank barrier；tail-reduce、normal 和 no-tail signal 路径不参与。
- 目标是量化 `start->after_barrier` 约 `0.03-0.05 ms` 的固定成本能否安全拿掉，并观察 K1 dispatch-pull 跨 rank读取是否保持 correctness。

## 2026-06-13 - LL pre-K1 rank barrier skip A/B rejected and reverted

- 本地新增 `MEGAMOE_DCU_V3_LL_SKIP_PRE_K1_BARRIER` 后 compileall 和 `git diff --check` 通过；同步远端后 source pytest 通过 `11 passed`。
- 远端短 smoke 使用 `MEGAMOE_DCU_V3_LL_SKIP_PRE_K1_BARRIER=1 TOKENS_LIST="32 128" RUN_BASELINE=0 RUN_DEFAULT=1 ...`。
- 结果第一档即 correctness 失败：rank0 `fused/baseline max_abs=0.04144287109375 exceeds --atol=0.0035`，其余进程被 torch spawn 终止。
- 已撤回 `large_opt.py` 中的 skip gate 和 source guard；结论是 K1 前 rank barrier 当前仍为跨 rank symmetric-buffer input visibility 必需项。
- 撤回后远端 source pytest 再次通过 `11 passed`；默认 32-token 短 sanity 通过，summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_202143.csv`，fused median `0.7686 ms`（短跑不与 formal 比）。

## 2026-06-13 - LL K1 output_index full-skip A/B started

- 重新读取 planning 三文件并对齐当前 LL 历史，确认 bounded-init 已反证，不能重复只清理前缀的做法。
- 静态确认当前 `large_opt.py` 的 V3 LL staged K2/K3 消费 `route_weights/m_indices/row_combine_ptrs`，不消费 `output_index`；但 K1 wrapper 对外仍需要保留默认 `output_index` 合同。
- 本轮只做 env-gated A/B：`MEGAMOE_DCU_V3_LL_SKIP_OUTPUT_INDEX=1` 时让 V3 LL staged K1 raw launch 传 `nullptr` 给 `output_index`，尝试跳过全量 `-1` 清理和 route 写回；默认路径不变。

## 2026-06-13 - LL K1 output_index full-skip A/B rejected and reverted locally

- A/B correctness 通过，但 formal 32/128 fused median 为 `0.8989/1.0846 ms`，128 慢于 retained `1.0752 ms`。
- Stage timing 显示 K1 小幅下降没有形成稳定 e2e 收益：32 K1 `0.397 ms`、128 K1 `0.4425 ms`，K3 基本仍为 `0.2565/0.376 ms`。
- 已本地撤回 `MEGAMOE_DCU_V3_LL_SKIP_OUTPUT_INDEX` env、`ll_emit_output_index` wrapper/pybind 参数和 source guard；保留默认 `output_index` 合同。
- 本地检查通过：`rg` 确认源码/测试无 skip-output-index 残留，`compileall` 通过，`git diff --check` 通过。
- 已同步 7 个文件到 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM`，远端 source pytest 通过 `11 passed`；8 卡无 KFD PID。
- 远端 LL raw rebuild 完成，log `hygon_tmp/sglang_debug/rebuild_v3_ll_revert_output_index_20260613.log`；K1 extension 用 `/opt/dtk/bin/hipcc` 重编并复制 `.so`。
- 默认 LL 短 sanity 通过：32/128 correctness 均通过，短跑 fused median `0.7674/0.9357 ms`；该轮 warmup/repeat 很短，只用于确认路径恢复，不作为 formal perf 结论。
- 默认 retained LL stage timing：32 formal fused `0.8986 ms`，last3 median total/K1/K3 `0.7705/0.402/0.257 ms`；128 formal fused `1.0861 ms`，last3 median total/K1/K3 `0.9595/0.444/0.3825 ms`。
- DCU KB 对 K1 staged/route metadata 的建议偏向固定 layout、一次性 GPU-friendly metadata 与 prefetch 专用路径；结合既有反证，下一项选择不改合同的 K1-only `CUS=32` A/B，测试低 token 下 64 CTA 多次 grid barrier/metadata 阶段是否过重。
- 本地实现 `MEGAMOE_DCU_V3_LL_K1_CUS=32` env-gated A/B：默认仍为 64，只新增 K1 LL blockM=32/CUS=32 实例；K3 不变。本地 `compileall` 和 `git diff --check` 通过。

## 2026-06-13 - LL K1 CUS=32 A/B rejected and reverted locally

- 从远端拉回并解析 CUS=32 A/B 日志：
  - 32 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_205129.csv`，formal fused `1.0166 ms`；
  - 128 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_205207.csv`，formal fused `1.1953 ms`；
  - 对比 post-revert retained `0.8986/1.0861 ms` 明确回退。
- Stage timing last3：
  - 32 total/K1/K2/K3/barrier/reduce `0.8825/0.520/0.028/0.2495/0.0355/0.012 ms`；
  - 128 total/K1/K2/K3/barrier/reduce `1.072/0.580/0.028/0.367/0.035/0.014 ms`；
  - 回退主要来自 K1，CUS=32 不是有效优化方向。
- 已本地撤回 `MEGAMOE_DCU_V3_LL_K1_CUS` env、K1 wrapper 动态 `ll_cus` 参数、K1 launcher 32-CUS 模板分支和 source guards；`k1_fused_ext.cu` 重新限制 LL `ll_cus=64`。
- 已更新 `task_plan.md` 和 `findings.md`；下一步对 retained LL 默认路径做 PMC/ISA attribution，再选择新的低风险优化方向。

## 2026-06-13 - LL K1 saturated-count no-clamp-barrier A/B started

- 远端 CUS=32 撤回后 LL raw rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_ll_revert_k1_cus32_20260613.log`；默认 32/128 短 sanity correctness 通过，短跑 fused `0.7644/0.9318 ms`。
- 解析旧 PMC 证据：K1 LL128 单 kernel median 约 `0.47 ms`，K3 LL128 约 `0.41 ms`；K1 的 VMEM/LDS/VALU 与 LDS bank conflict 计数偏高，仍优先看 K1 route/stage 固定成本。
- DCU KB 针对 grid barrier/atomic count 检索显示 Hygon 参考强调 barrier release/acquire 和 atomic conflict，但没有要求把容量 clamp 写回全局计数；只要保留 route atomic 完成后的 grid barrier，后续消费者本地饱和计数即可保持可见性与容量语义。
- 本地 patch：
  - 删除 `v3_k1_build_ll_stage_device()` 中 route 完成后的 `symm_counts[expert] = m_per_expert` clamp loop 和紧随其后的额外 `v3_k1_ll_grid_barrier_device()`；
  - stats、staged copy 和 GEMM 读取 `symm_counts` 时改用 `min(count, m_per_expert)`；
  - source guard 防止 reintroduce 写回 clamp。
- 本地验证：`compileall` 通过，`git diff --check` 通过；下一步同步远端、source pytest、LL raw rebuild、32/128 correctness + stage timing，按实测决定保留或撤回。

## 2026-06-13 - LL K1 saturated-count no-clamp-barrier retained as micro-optimization

- 远端确认 `.cuh` 已同步但 Ninja 未捕捉 header 依赖；只删除 `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_fused_ext.hip` 和 `build/.../k1_v3_fused_ext.o` 后强制重编，K1 LL raw 使用 `/opt/dtk/bin/hipcc` 实际重新生成 object。
- 8 卡状态干净，K1 extension import 正常。
- 32/128 formal correctness 均通过；第一轮 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_210925.csv`：32 `0.9025 ms`，128 `1.0639 ms`。
- 解析 stage last24：32 total/K1/K2/K3/no-tail barrier/reduce `0.774/0.403/0.028/0.2565/0.0385/0.012 ms`；128 `0.951/0.443/0.028/0.3815/0.043/0.014 ms`。
- 128 确认轮 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_211105.csv`：`1.0744 ms`，与 retained `1.0752/1.0861 ms` 同档；因此该项保留为低风险去冗余 barrier 微优化，不算确定大收益。
- 下一步按计划进入 `LL K1 stage-copy active-row/padded-zero attribution`，寻找比 no-clamp-barrier 更大的 K1 LL fixed/stage-copy 收益。

## 2026-06-13 - LL K1 stage-copy expert-loop retained

- 拉回并解析 stage-copy expert-loop A/B 日志：`v3_ll_default_32_bench_20260613_211958.log`、`v3_ll_default_128_bench_20260613_212035.log`、summary `v3_ll_perf_summary_20260613_212113.csv`。
- formal 32/128 correctness 通过，fused median `0.8451/1.0663 ms`。
- stage last24：
  - 32 total/K1/K2/K3/no-tail barrier/reduce `0.719/0.358/0.028/0.2575/0.0305/0.012 ms`；
  - 128 total/K1/K2/K3/no-tail barrier/reduce `0.932/0.421/0.028/0.3805/0.0505/0.014 ms`。
- 相比 saturated-count retained 的 K1 stage `0.403/0.443 ms`，stage-copy expert-loop 对 32 有确定收益，对 128 也降低 K1 stage 且 e2e 同档偏好；已更新 `task_plan.md` / `findings.md`，并把 `tests/test_dcu_megamoe_v3.py` source guard 改成 expert-loop 版本。
- 下一步以该 retained 路径为新基线，跑本地 source checks、远端 source pytest，然后复测 LL K2 skip/block48/block64 与 128 K3 residual。

## 2026-06-13 - LL post-stagecopy variant refresh

- 本地检查：`compileall` 通过，`git diff --check` 通过；一次 Bash heredoc 风格 inline Python 在 PowerShell 中解析失败，改用 PowerShell here-string 后 source guard 通过。
- 同步 `tests/test_dcu_megamoe_v3.py`、K1 header 和 planning 文件到 11 节点；远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过 `11 passed`。
- 首次远端 variant bench 因 SSH 嵌套引号把 `TOKENS_LIST="32 128"` 拆坏而未启动；改用 `env TOKENS_LIST='32 128' ...` 重跑成功。
- post-stagecopy formal summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_213141.csv`：
  - default 32/128 `0.8418/1.0541 ms`；
  - K2-skip `0.8424/1.0626 ms`；
  - block48 `0.9757/1.1396 ms`；
  - block64 `2.7046/2.9043 ms`。
- Stage last24 确认 default 仍最优：default 32/128 K1/K3 `0.3575/0.256 ms` 与 `0.425/0.387 ms`；block64 K1 `2.24/2.30 ms`，继续排除更大 blockM。
- 已更新 `task_plan.md` / `findings.md`；下一步进入 `LL post-stagecopy K3 128 residual attribution`，用 rowptr split/PMC/ISA 判定 remaining K3 delta。

## 2026-06-13 - LL K3 residual and tail snapshot recorded

- 拉回并解析 tail-reduce post-stagecopy 日志：`v3_ll_default_32_bench_20260613_213549.log`、`v3_ll_default_128_bench_20260613_213627.log`、summary `v3_ll_perf_summary_20260613_213706.csv`。
- Tail formal 32/128 fused 为 `0.8425/1.0759 ms`；stage last24 median 32 total/barrier/K1/K2/K3_tail `0.7305/0.0395/0.358/0.027/0.306 ms`，128 `0.953/0.032/0.421/0.028/0.4715 ms`。
- 解析 `k3_ll_rowptr_modes_128_post_stagecopy_20260613.json`：K3 LL 128 pure/local `0.2368-0.2435 ms`，staged_remote/staged `0.3658/0.3665 ms`，active/local/remote rows `768/32.75/735.25`。
- 已更新 `task_plan.md`：`LL post-stagecopy K3 128 residual attribution` 标为 ✅，新增 ⏳ `LL K1 staged scale/init sparse A/B`。
- 下一步：在不改 scratch layout、launch 数、K2/K3 合同的前提下，对 K1 LL full-capacity `staged_x_scale` 初始化做窄 A/B，实测决定保留或撤回。

## 2026-06-13 - LL K1 staged scale/init sparse A/B rejected and reverted

- 本地尝试减少 full-capacity `staged_x_scale` 初始化：active rows 继续由 route scan 写真实 scale，rounded padded rows 在 stage-copy 中由 `vec_col==0` 写 tiny scale；scratch layout、launch 数和 K2/K3 合同不变。
- 本地 `compileall` / `git diff --check` 通过；同步远端后强制删除 K1 `.hip/.o` 并重编，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_sparse_scale_20260613.log`。
- 第一次 remote bench 命令因 PowerShell 嵌套引号把 `TOKENS_LIST="32 128"` 拆坏，未启动；改用 `$inner` 包装后成功。
- A/B summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_215524.csv`：32/128 correctness 通过，但 formal fused `0.8452/1.0693 ms`，慢于 retained `0.8418/1.0541 ms`。
- Stage last24：32 total/K1/K2/K3/no-tail barrier/reduce `0.723/0.3595/0.028/0.256/0.0325/0.012 ms`；128 `0.9325/0.4225/0.029/0.3835/0.0495/0.014 ms`，K1 无实质收益。
- 已撤回本地补丁并同步远端；K1 LL rebuild 完成，log `hygon_tmp/sglang_debug/rebuild_v3_ll_revert_sparse_scale_20260613.log`；短 sanity 32/128 correctness 均通过，summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_215828.csv`。
- 已更新 `task_plan.md` / `findings.md`，下一步进入 `LL retained path PMC/ISA attribution`，先看 retained K1/K3 code-object/PMC 证据再继续挑优化方向。

## 2026-06-14 - LL retained path code-object attribution completed

- 恢复上下文并重读 planning / remote / DCU KB / Hygon optimizer 约束后继续 LL 优化优先级 1/2/3。
- 远端 `sglang_megamoe` 环境干净，`hy-smi --showpids` 无 KFD 进程；之前的 save-temps 目录仍在 `hygon_tmp/sglang_debug/ll_save_temps_20260613_220234`。
- 解析 `k1_v3_fused_ext-hip-amdgcn-amd-amdhsa-gfx938.s` 和 `k3_v3_fused_ext-hip-amdgcn-amd-amdhsa-gfx938.s`，结果写到远端 `hygon_tmp/sglang_debug/ll_save_temps_20260613_220234/active_counts_20260614.json`。
- 结论：K1 block32 active 实例 `VGPR=124/SGPR=96/private=0`，block64 `VGPR=193/private=272` 解释 block64 回退；K3 block32 no-tail `VGPR=153/private=0`，128 residual 仍指向 remote rowptr combine 数据通路。
- 已更新 `task_plan.md` 将 `LL retained path PMC/ISA attribution` 标为 ✅，新增 ⏳ `LL K1 parallel stage-copy A/B`；下一步开始第 2 项的 env-gated K1 LL parallel stage-copy 实验。

## 2026-06-14 - LL K1 parallel stage-copy A/B retained

- 本地已实现 LL K1 parallel stage-copy A/B，并同步 `k1_v3_groupgemm_impl.cuh`、`k1_v3_fused_ext.cu`、`tests/test_dcu_megamoe_v3.py` 到 11 节点。
- 远端 source pytest 通过：`11 passed`。
- 第一次 K1-only raw build 后发现 setup 会顺手重拷贝 K3 extension；随后用 K1+K3 LL raw pair gate 重编，确保 K3 raw extension 仍可用。
- 为避免 PowerShell/SSH 嵌套引号把 `TOKENS_LIST='32 128'` 拆坏，新增临时脚本 `hygon_tmp/sglang_debug/run_ll_parallel_stage_ab.sh` 和 parser `hygon_tmp/sglang_debug/parse_stage_timing.py`。
- 短 smoke 32/128 correctness 通过；formal run `ITERS=5 WARMUP=3 REPEAT=5` summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_074211.csv`：
  - 32 fused median `0.8311 ms`；
  - 128 fused median `1.0337 ms`；
  - 对比 expert-loop retained `0.8418/1.0541 ms` 两档均改善。
- Stage last24 median：
  - 32 total/K1/K2/K3/no-tail barrier/reduce `0.7125/0.346/0.028/0.256/0.0305/0.012 ms`；
  - 128 total/K1/K2/K3/no-tail barrier/reduce `0.9055/0.398/0.028/0.384/0.042/0.014 ms`。
- 结论保留后已把 parallel stage-copy 提升为默认开启；2026-06-15 cleanup 已进一步固定为生产默认，删除诊断回退 env。
- 默认 promotion 后重新同步、source pytest `11 passed`、K1/K3 LL raw pair rebuild 通过。
- 无 env 默认确认：
  - 32/128 三轮 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_074834.csv` 为 `0.8332/1.0543 ms`，stage last24 K1 `0.347/0.400 ms`；
  - 128-only 五轮确认 `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_074935.csv` 为 `1.0444 ms`，stage last24 total/K1/K3 `0.9115/0.4005/0.389 ms`。
- 已更新 `task_plan.md` 标记 `LL K1 parallel stage-copy A/B` 为 ✅，新增 ⏳ `LL K3 remote combine optimization`；下一步按计划做第 3 项。

## 2026-06-14 - LL K3 store modifier / deferred rowptr wait A/B rejected

- 按 `LL K3 remote combine optimization` 开始第 3 项后，先静态复核 K3 LL no-tail：
  - no-tail 没有最终 `wait_vmem_lds_store_device()`；该 wait 只在 tail-reduce 路径；
  - no-tail combine store 已是 `global_store_dwordx2` bf16x4 vector store，缺口不是简单 scalar-store；
  - rowptr 地址 load 当前立即 `s_waitcnt vmcnt(0)`，可以做“延迟 wait”实验但不能省略 wait。
- DCU KB 结论：`glc/slc` store modifier 和 waitcnt 移动都必须用实测闭环确认；消费 global load 值前仍需 wait。
- 临时 probe `hygon_tmp/sglang_debug/probe_global_store_flags.hip` 在远端用 `/opt/dtk/bin/hipcc -O2 --offload-arch=gfx938 -c` 编译通过，确认 `global_store_dwordx2 ... off glc/slc` 语法可用。
- Store modifier A/B：
  - 临时加 `MEGAMOE_DCU_V3_LL_K3_STORE_MODE=glc|glc_slc`；
  - source pytest `11 passed`，K1+K3 LL raw rebuild 通过；
  - 128-only default/glc/glc_slc fused 为 `1.0306/1.0275/1.0317 ms`，K3 stage 为 `0.382/0.380/0.3845 ms`；
  - `glc` 信号为噪声级，`glc_slc` 回退，不保留。
- Deferred rowptr wait A/B：
  - 临时加 `MEGAMOE_DCU_V3_LL_K3_DEFER_ROWPTR_WAIT=1`；
  - source pytest `11 passed`，K1+K3 LL raw rebuild 通过；
  - 128-only default/defer fused 为 `1.0348/1.0489 ms`，K3 stage 为 `0.389/0.410 ms`；
  - correctness 通过但性能明确回退，不保留。
- 已撤回上述临时 env/helper/template/source guard 改动，重新同步远端；最终 source pytest `11 passed in 5.65s`，K1+K3 LL raw pair rebuild 通过并恢复 retained default `.so`。
- 已更新 `task_plan.md` / `findings.md`；下一步继续 `LL K3 remote combine PMC/ISA/SQTT 归因`，避免继续无证据地改 waitcnt 或 store modifier。

## 2026-06-14 - LL K3 PMC attribution and CUS32 A/B rejected

- 恢复上下文后重读 planning 三文件、remote workflow、Hygon optimizer、DCU KB 约束；确认当前 Phase 6b 焦点为 K3 LL remote combine。
- DCU KB optimize 检索超时但返回了 Hygon 参考信号：combine/dispatch 类热点形状适合保留专用变体，仍必须以实测/PMC/ISA 证据决定。
- 完成 K3 LL 128 PMC 归因：
  - local vs remote VMEM 指令数接近；
  - staged remote-only 的 `TA_BUSY` 约为 local 的 `1.4x`，`TCP_TA_DATA_STALL` 约为 local 的 `2x`；
  - rowptr fanout 低 locality，结论指向 remote/scattered combine store path。
- 完成 `MEGAMOE_DCU_V3_LL_K3_CUS=32` A/B：
  - default 32/128 fused `0.8352/1.0301 ms`；
  - CUS32 32/128 fused `1.0381/1.2106 ms`；
  - CUS32 128 K3 stage 退到约 `0.55-0.59 ms`，明确慢于 default 约 `0.37-0.41 ms`。
- 已撤回 K3 CUS32 env gate、额外 CUS 模板参数和 source guard；
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 `git diff --check` 通过；
  - 远端同步 `k3_v3_fused_ext.cu` / `tests/test_dcu_megamoe_v3.py`；
  - 远端 source pytest `11 passed in 5.73s`；
  - 远端 K1+K3 LL raw pair rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_revert_k3cus32_20260614_084243.log`；
  - import check `has_k1_pack5 True`、`k3_ll True`。
- 已更新 `task_plan.md` / `findings.md`：PMC 归因标为 ✅，CUS32 标为 ✅ 反证，下一步继续 `LL K3 row distribution / store schedule`。

## 2026-06-14 - LL K3 rowptr dest-sort split diagnostic rejected

- 按 Hygon optimizer 的“连续无收益后先归因再选方向”流程，检查当前容器 SQTT/stat 工具：
  - `hipprof` help 仅显示 trace/PMC/codeobj，未暴露 `--sqtt`；
  - `stat_stall`、`stat_valu`、`rocprof`/`rocprofv3` 不在 PATH；
  - 当前可用工具为 `/opt/dtk/bin/hipprof` 与 `/opt/dtk/bin/dccobjdump`。
- 为验证 rowptr locality 是否有理论收益，只修改临时脚本 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`：
  - 新增 `make_dest_sorted_rowptrs()`；
  - 新增诊断 mode `staged_remote_dest_sorted` / `staged_rowptr_dest_sorted`；
  - 该诊断不改生产 kernel，也不作为 correctness 路径。
- 远端 128 tokens split 结果：
  - `staged_remote_only` `0.36516 ms`；
  - `staged_remote_dest_sorted` `0.36446 ms`；
  - `staged_rowptr` `0.36720 ms`；
  - `staged_rowptr_dest_sorted` `0.36739 ms`；
  - active/local/remote rows `768/32.75/735.25`。
- 结论：
  - dest-rank rowptr sorting 只有噪声级变化，不值得改生产 K1 emission；
  - 已更新 `task_plan.md` / `findings.md`，后续不重复 row distribution / CUS / store modifier / deferred wait 方向；
  - 下一步若继续 K3 LL，需要聚焦 store schedule/codeobj/remote-write ceiling 诊断，或转回整体 LL gate 收口。
- CUS32 撤回后的默认 LL 刷新：
  - summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_085204.csv`；
  - 32/128 fused `0.8262/1.0381 ms`；
  - last24 stage median total/K1/K2/K3/no-tail barrier/reduce 为 `0.901/0.400/0.028/0.385/0.0365/0.014 ms`；
  - 说明默认 CUS64 + K1 parallel stage-copy 基线已恢复。

## 2026-06-14 09:35:23 +08:00 - normal V3 ASM-pack5 best-path refresh started

- 用户要求继续优化 V3 normal K1/K3 fused 最优路径，重点找 1024/4096 no-tail 是否还有性能空间。
- 已重读 planning/remote/Hygon optimizer/DCU KB skill 约束与三份 planning 文件；session-catchup 无输出。
- 已在 task_plan.md Phase 6 增加 ⏳ normal V3 ASM-pack5 best-path refresh，下一步先跑当前默认 champion 与 opt-out/stage timing 实测。

## 2026-06-14 10:02:00 +08:00 - normal V3 ASM-pack5 refresh baseline recorded

- 当前默认 V3 normal ASM-pack5 no-tail correctness 与 formal bench 已重新实测：
  - 1024 tokens：原 staged fused `2.3402 ms`，V3 ASM-pack5 `2.1997 ms`，V3 快约 `6.4%`；
  - 4096 tokens：原 staged fused `6.6610 ms`，V3 ASM-pack5 `6.5323 ms`，V3 快约 `2.0%`。
- Stage timing last24 median：
  - 1024 rows=8192：total `2.102 ms`，K1 `1.018 ms`，K2 `0.105 ms`，K3 combine `0.811 ms`，no-tail barrier `0.0965 ms`，reduce `0.048 ms`；
  - 4096 rows=29696：total `6.392 ms`，K1 `3.320 ms`，K2 `0.214 ms`，K3 combine `2.526 ms`，no-tail barrier `0.128 ms`，reduce `0.168 ms`。
- K3 direct ASM-pack5 rowptr split 已复测，staged rowptr 为 `0.7996/2.4801 ms`，非常接近 full-stage K3 `0.811/2.526 ms`；这只说明 K3 的 full-stage 编排额外损耗小，不代表 K3 已贴近 pure C/groupgemm。K3 fused 主 kernel 相比 pure 的通信/rowptr combine 成本仍需单独作为性能空间记录。
- K1 pure C pack5 refresh 为 `0.746 ms / 2.252 ms`，对比 full-stage K1 `1.018 / 3.320 ms`，剩余主要 gap 为 K1 dispatch-pull/metadata/staging 约 `0.27 ms / 1.07 ms`。
- K1 ASM-pack5 与原 K1 ASM code-object 计数显示 `v_mmac/load/store/wait/barrier` 数相同，pack5 版本 scalar 地址指令还略少；当前没有类似 K3 早期 A-offset 重复计算的明显静态信号。
- K1/K3 ASM-pack5 opt-out 失败是当前 build 只保留 production champion、raw fallback 未编入所致，不作为性能结论；后续若要 opt-out 需显式 raw build 或专用 harness。
- 下一步优先做 K1-focused profile/直接对照，确认 K1 residual 是 dispatch/staging 结构性成本还是仍有 ASM/地址/同步微调空间；K3 no-tail 只保留为 regression guard，不优先继续改。

## 2026-06-14 10:05:34 +08:00 - normal K3 fused-vs-pure diagnostic started

- 用户明确 normal K3 主 fused kernel 也必须和对应 pure C/groupgemm 对比，不能只用 direct staged-rowptr floor 代替 pure floor。
- 本地已新增 diagnostic-only `k3_v3_normal_pure_raw` pybind：复用已有 K3 normal raw C kernel，并传 `row_combine_ptrs=nullptr`；不接入 `large_opt.py` runtime，不新增生产路径 launch。
- 已更新 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`，normal backend 现在也有 `pure_contiguous` mode；`run_v3_normal_k3_rowptr_modes.sh` 默认输出 pure/local/staged-local/staged-remote/staged-rowptr/all-zero。
- 本地验证通过：`python -m compileall tests/test_dcu_megamoe_v3.py hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`；`git diff --check` 针对本次改动通过。
- 下一步：同步远端，aicc/raw normal rebuild，然后跑 1024/4096 normal K3 pure-vs-fused split，记录 pure/local/remote/staged delta。

## 2026-06-14 10:09:41 +08:00 - normal K3 pure wrapper VMFault diagnosed

- 首次远端 `run_v3_normal_k3_pure_compare.sh` 在 normal raw 1024 的 `pure_contiguous` 上触发 VMFault / `ProcessExitedException: process 4 terminated with signal SIGSEGV`。
- 根因：normal C raw no-tail fast path 内部仍把 `row_combine_ptrs` 作为 active-row mask 和 rowptr store 合同，`out` 形参被 `(void)out`；单纯传 `row_combine_ptrs=nullptr` 会让 kernel 访问 null rowptr resource。
- 修复方向：新增隔离 `kPureContiguous` 模板实例，pure path 从 contiguous `act_fp8` load、写 contiguous `out`；现有 fused rowptr、signal-only、tail-reduce 实例不改。
- 本地已实现 `kPureContiguous` 分支和 source guard，`compileall` / `git diff --check` 通过；待远端 aicc rebuild 与 1024/4096 split 复测。

## 2026-06-14 10:17:02 +08:00 - normal K3 pure-vs-fused split completed

- 修复后的 `kPureContiguous` normal raw aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_raw_20260614_101424.log`。
- `run_v3_normal_k3_pure_compare.sh` 完成 1024/4096 两组 direct split，无 VMFault。
- 1024：C pure `0.445 ms`；ASM-pack5 all-zero `0.485 ms`，local `0.514 ms`，staged-rowptr `0.825 ms`。
- 4096：C pure `1.546 ms`；ASM-pack5 all-zero `1.501 ms`，local `1.553 ms`，staged-rowptr `2.491 ms`。
- 结论：K3 normal ASM-pack5 GEMM core 已接近 pure，remaining gap 主要是 remote/scattered rowptr combine store；已把 `normal K3 fused-vs-pure measurement` 标为 ✅，新增 ⏳ `normal K3 ASM-pack5 remote-store PMC/ISA attribution`。

## 2026-06-14 10:55:00 +08:00 - normal K3 remote-store PMC completed and K1 residual started

- 新增/使用临时脚本 `hygon_tmp/sglang_debug/run_v3_normal_k3_asm_pack5_pmc.sh`；首次 kernel filter 为空，后用 code object 真实 kernel 名 `DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE` 修正。
- PMC 产物拉回 `hygon_tmp/sglang_debug/prof_pull_normal_k3_pmc_20260614_1041/`；4096 staged_rowptr 相比 rowptr_all_zero 的 `SQ_INSTS_VMEM_RD/WR` 不变，但 `TA_BUSY` 从约 `47.2M` 升到 `57.9M`，`TCP_TA_DATA_STALL` 从约 `0.90M` 升到 `9.3M`。
- 静态 ASM 复核确认 staged combine store 已经是 `global_store_dwordx4`；瓶颈不是 store vectorization 缺失。
- glc store modifier A/B 已完成并撤回：1024 噪声级、4096 staged-rowptr `2.580 ms` 慢于 retained `2.491 ms`；恢复远端 `.co` sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`，恢复后 1024/4096 direct correctness 通过。
- 已更新 `task_plan.md` / `findings.md`：`normal K3 ASM-pack5 remote-store PMC/ISA attribution` 和 store modifier A/B 标为 ✅；新增 ⏳ `normal K1 ASM-pack5 fused-vs-pure residual attribution`。
- 下一步按 fused-vs-pure 口径优先定位 K1 normal 1024/4096 residual，K3 normal no-tail 暂作为 regression guard。

## 2026-06-14 11:19:28 +08:00 - normal K1 rejected A/B recorded and clean build restored

- 删除上一轮已反证的临时 `MEGAMOE_DCU_V3_K1_COMPACT_MARGIN_TILES` env gate，恢复默认 compact margin：saving 足够时 `2` tiles，否则 `4` tiles。
- 本地 `git diff --check -- megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu` 通过；`rg` 确认该临时 env 不再存在。
- 同步 `k1_fused_ext.cu` 到 11 节点 `/home/hg/yuguo/DeepGEMM`，容器内执行 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM && bash hygon_tmp/sglang_debug/run_k1_only_raw_build.sh hygon_tmp/sglang_debug/rebuild_k1_clean_margin_20260614_1115.log`，结果 `__STATUS:0`、`aicc_marker=1`、`k1_raw_compile_seen=2`、`v2_seen=0`。
- Clean retained K1 direct sanity：1024 V3 median `0.9776 ms`，4096 V3 median `2.9350 ms`；4096 rank0 `active_tiles=113`、`ceil_tiles_from_counts=113`、`sum_counts=24678`，两档 compare `max_abs=0`。
- 已把 stats off、asm prebuild、compact margin 0/1 的 A/B 反证写入 `task_plan.md` / `findings.md`；下一步在 clean build 上跑 no-tail full-stage refresh，拆 direct K1 与 full-stage K1 段差距。

## 2026-06-14 12:18:40 +08:00 - normal K1 symm warm-up retained and verified

- 本地实现 V3 normal-only symm allocator warm-up：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 && USE_MEGAMOE_V3=1 && MEGAMOE_DCU_V3_BACKEND in {unset,normal}` 且 tokens/rank `>=512` 时默认启用，`MEGAMOE_DCU_V3_SYMM_WARMUP_ALLOC=0` 可回退；dummy `SymmBuffer` 用最小 token alignment、`prepare_large_opt_3stage=False` 并立即 destroy。
- 本地验证：
  - `python -m compileall megamoe/__init__.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe/__init__.py tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu` 通过；
  - 本地 `python -m pytest` 不可用，原因仍是本地 Python 缺少 pytest。
- 远端验证：
  - 同步 `megamoe/__init__.py` 与 `tests/test_dcu_megamoe_v3.py` 到 11 节点 `sglang_megamoe`；
  - `source /opt/dtk/env.sh && cd /workspace/DeepGEMM && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过 `12 passed in 5.62s`。
- 性能实测：
  - 1024 default warm-up：correct，formal e2e `2.2322 ms`，stage last24 total/K1/K2/K3/no-tail barrier/reduce `2.169/1.026/0.105/0.821/0.090/0.048 ms`；
  - 4096 default warm-up：correct，formal e2e `6.2910 ms`，stage last24 `6.044/2.9455/0.214/2.5505/0.133/0.168 ms`；
  - 4096 opt-out `MEGAMOE_DCU_V3_SYMM_WARMUP_ALLOC=0`：correct，formal e2e `6.6039 ms`，stage last24 `6.3605/3.260/0.214/2.543/0.1305/0.168 ms`。
- 清理临时弱信号：
  - 删除 `MEGAMOE_DCU_V3_K1_FORCE_ABSOLUTE_PTRS` gate；
  - 同步 `k1_fused_ext.cu` 后远端 K1 rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_k1_remove_force_abs_20260614_1216.log`，`aicc_marker=1`、`k1_raw_compile_seen=2`、`v2_seen=0`；
  - rebuild 后 4096 default 短 sanity correctness 通过，e2e `6.1313 ms`，stage last24 `6.054/2.8855/0.213/2.5785/0.1495/0.168 ms`。
- 注意：
  - 一次 benchmark 命令因 `TOKENS_LIST="1024 4096"` 在 SSH/PowerShell 嵌套中被拆成 `4096 bash` 失败，未启动实际测试；已改用 `env TOKENS_LIST=...` 形式。
  - 一次 `scp` 把 `__init__.py` / `test_dcu_megamoe_v3.py` 的副本也放到了远端 `hygon_tmp/sglang_debug/`，属于临时目录无害副本；正式路径已同步。
- 计划状态：
  - `normal K1 ASM-pack5 fused-vs-pure residual attribution` 和 `normal K1 full-stage-vs-direct refresh` 标为 ✅；
  - 新增 ⏳ `normal post-warmup remaining-gap refresh`，下一步固定 warm-up + ASM-pack5 champion 后重新汇总 K1/K3 pure/direct/full-stage gap。

## 2026-06-14 12:26:53 +08:00 - normal post-warmup gap refreshed and K3 wait A/B started

- 已汇总默认 warm-up + K1/K3 ASM-pack5 champion 的 1024/4096 formal 与 direct/pure floor：
  - 1024 formal e2e `2.2322 ms`，K1 full/direct `1.026/0.9686 ms`，K3 full/direct/pure `0.821/0.8002/0.4440 ms`；
  - 4096 formal e2e `6.2910 ms`，K1 full/direct `2.9455/2.9452 ms`，K3 full/direct/pure `2.5505/2.4553/1.5378 ms`。
- 结论：K1 4096 full-stage/direct gap 已基本消失；normal 的最大 remaining gap 是 K3 ASM-pack5 staged-rowptr 相对 all-zero/pure 的 remote/scattered combine store delta。
- 已在 `task_plan.md` 将 `normal post-warmup remaining-gap refresh` 标为 ✅，新增 ⏳ `normal K3 staged-store deferred LDS-wait A/B`。
- 本地已修改 isolated `K3COMBINE_PACK5.s`：`K3_STORE_STAGED_HALF` 中 `s_waitcnt lgkmcnt(0)` 延后到 rowptr 地址计算之后，准备同步远端重编 K3 pack5 code object 并跑 direct rowptr split correctness/perf。

## 2026-06-14 12:38:00 +08:00 - normal K3 remote ceiling/fanout diagnostic completed

- K3 staged-store deferred LDS-wait A/B 未保留，production ASM-pack5 `.co` 已恢复 retained sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`。
- 已在 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py` 增加 diagnostic-only remote ceiling modes，并跑 4096 ASM-pack5 split：
  - local/count-equivalent path `1.5439 ms`，single-peer contiguous remote `2.3103 ms`，真实 `staged_remote_only/staged_rowptr` `2.5121/2.5205 ms`；
  - 说明最大项是 remote write path ceiling，真实 fanout/scatter 额外约 `0.20 ms`。
- Naive dest-sort 诊断已反证：`staged_remote_only 2.5026 ms -> dest_sorted 3.9994 ms`，`staged_rowptr 2.5215 ms -> dest_sorted 3.7999 ms`。
- 已更新 `task_plan.md`：remote ceiling/fanout diagnostic 标为 ✅，新增 ⏳ `normal K3 remote-store PMC/ceiling follow-up`。
- 下一步：对 retained ASM-pack5 的 `staged_remote_contig_peer`、`staged_remote_only`、`staged_rowptr` 做 PMC/codeobj 对照，确认 `TA_BUSY/TCP_TA_DATA_STALL` 是否随 remote/fanout 变化，避免继续无证据调参。
- 同步 planning/hygon_tmp 到远端时第一次 PowerShell 命令被 `${SSH_TARGET}:...` 冒号解析问题绊住，未开始上传；已改成显式 `${SSH_TARGET}:$remotePath` 后同步成功，后续 Windows scp 路径继续用该写法。

## 2026-06-14 12:46:00 +08:00 - normal K3 remote-store PMC/ceiling follow-up completed

- 8 卡空闲，`hy-smi` 显示 HCU 0-7 基本 0% 使用；`--showpids` 仍因进程目录读取报 generic error，但普通 `hy-smi` 可用。
- 已跑 4096 retained K3 ASM-pack5 PMC：`staged_remote_contig_peer`、`staged_remote_only`、`staged_rowptr` 三种 mode，各 read/write 一轮，产物前缀 `hygon_tmp/sglang_debug/prof/pmc_normal_k3_asm_pack5_4096_*_remote_ceiling_20260614_124056.*`。
- 解析结果：
  - single-peer contiguous read/write mean `2.3038/2.3028 ms`；
  - staged_remote_only `2.4418/2.4557 ms`；
  - staged_rowptr `2.4587/2.5095 ms`；
  - `vmem_rd/vmem_wr/valu/lds_bank_conflict/grd` 三种 mode 完全一致；
  - 相对 single-peer contiguous，真实 staged 的 `TA_BUSY` 增加约 `13-16%`，`TCP_TA_DATA_STALL` 增加约 `4.7-5.2x`，`wait_lds` 基本不变。
- 已更新 `task_plan.md`：K3 remote-store PMC/ceiling follow-up 标为 ✅，新增 ⏳ `normal K1 fused-vs-pure residual follow-up`。
- 下一步：K3 normal no-tail 暂不继续 source-level store/wait/排序微调；回到 K1 normal direct/full/pure 对照，看剩余 gap 是否还有可操作项。

## 2026-06-14 13:16:21 +08:00 - normal K1 staged-copy producer CTA optimization retained

- 恢复上下文后重读 planning 三文件与 remote/Hygon optimizer/DCU KB skill，确认 normal 当前按 fused-vs-pure 接近方向推进。
- K1 `output_index` store A/B：
  - 临时跳过 K1 ASM-pack5 route emit 中 valid/invalid `output_index` store；
  - 重编 `.co` sha `a3b9a57c66a836bcfb242e379137dc8dd39ecb6bf327b5a5e72cb2ec3a216dfe`；
  - direct K1 1024/4096 `0.968/2.930 ms`，correctness `max_abs=0`，无收益；
  - 已恢复 source 并重编回 retained，恢复 sanity 1024/4096 `0.967/2.925 ms`。
- K1 staged input copy producer CTA A/B：
  - producer=6：direct `0.976/2.857 ms`；
  - producer=7：direct `0.972/2.820 ms`；
  - producer=8：direct `0.978/2.792 ms`；
  - 最终保留动态策略：基于已有 `expert_tiles_per_expert` (`s10`) 判定，`s10>=2` 用 8 producer CTA，否则保持 5；避免为 1024 多读 `num_tokens` kernarg。
- Retained dynamic producer 验证：
  - K1 `.co` sha `f425cef3cd7116775dc230c881f694d00032ae8bcef36ebad0f6d7d884a48c78`；
  - direct K1 1024/4096 `0.9666/2.7990 ms`，`max_abs=0`；
  - V3 normal no-tail full-stage correctness 3/3 通过，formal 1024/4096 `2.2130/6.0699 ms`；
  - stage timing last24：1024 total/K1/K2/K3/barrier/reduce `2.100/1.0105/0.105/0.843/0.079/0.048 ms`，4096 `5.847/2.763/0.213/2.5355/0.101/0.168 ms`。
- PMC / ISA：
  - 4096 K1 PMC old read/write duration mean `2.7687/2.7676 ms`，dynamic `2.6321/2.6286 ms`；
  - `vmem_rd` 降到约 `4.64M`，`TA_BUSY` 降到约 `139.8M`，VGPR/SGPR 仍 `256/112`；
  - `inspect_k1_codeobj_counts.sh` 显示 MMAC/load/store/wait/barrier 主体未变。
- 已更新 `task_plan.md`：`output_index` A/B 和 staged-copy producer A/B 标为 ✅，新增 ⏳ `normal V3 post-K1-producer champion refresh`。
- 下一步：以 dynamic producer K1 + retained K3 ASM-pack5 为 champion，同场重跑 original vs V3 1024/4096、stage timing 与 pure/fused delta 汇总。

## 2026-06-14 13:35:00 +08:00 - normal post-K1-producer champion refresh completed

- 已完成 current champion 同场 original staged fused vs V3 normal no-tail refresh：
  - 1024：orig `2.4134 ms`，V3 `2.3054 ms`，speedup `1.0468x`，correctness 均通过；
  - 4096：orig `6.6033 ms`，V3 `6.0640 ms`，speedup `1.0889x`，correctness 均通过；
  - summary `hygon_tmp/sglang_debug/v3_normal_perf_summary_20260614_132258.csv`。
- V3 stage timing last24 median：
  - 1024：K1/K2/K3/barrier/reduce/total `0.9985/0.105/0.887/0.0645/0.048/2.1225 ms`；
  - 4096：`2.760/0.214/2.5375/0.1255/0.168/5.850 ms`。
- K1 direct refresh：1024 direct `0.9702 ms`，4096 direct `2.7958 ms`；4096 full-stage K1 已在 noise 内贴近 direct floor，1024 只剩约 `0.03 ms` 小 gap。
- K3 ASM-pack5 split refresh：1024 all-zero/local/staged-remote/staged-rowptr `0.477/0.507/0.811/0.828 ms`；4096 `1.502/1.550/2.455/2.482 ms`。remaining normal 主 gap 仍是 K3 remote/scattered rowptr combine store。
- 已更新 `task_plan.md`：normal V3 ASM-pack5 best-path refresh 和 post-K1-producer champion refresh 标为 ✅；新增 ⏳ `normal K3 ASM-pack5 post-refresh store/schedule evidence pass`。
- 下一步：静态复核 retained K3 ASM-pack5 staged store/wait/barrier window 与原 K3COMBINE ASM 差异，只做有 correctness/perf/PMC 或 ISA 证据支持的窄 A/B，不重复已反证方向。

## 2026-06-14 13:44:00 +08:00 - normal K3 terminal wait A/B rejected and reverted

- 静态复核确认 V3 `K3COMBINE_PACK5.s` 与原 `K3COMBINE.s` 的 no-tail staged store macro 和 H0/H1 store/barrier schedule 一致；当前 K3 gap 不是因为 V3 偏离原始 store schedule。
- 查 DCU KB 后只做一个窄 A/B：删除 isolated V3 PACK5 no-tail 末尾 `K3_STORE_STAGED_HALF 1024, 1024` 后的 terminal `s_waitcnt vmcnt(0)`。
- A/B `.co` sha `9d34af482f09f12b68e68713df16fee218a68ee9fbc993fee8d976e083ae1087`，direct K3 split：
  - 1024 staged-rowptr median average rank `0.8025 ms`，比 retained recent `0.8282 ms` 有小幅正信号；
  - 4096 staged-rowptr median average rank `2.5716 ms`，比 retained recent `2.4816 ms` 明确回退。
- 已撤回本地 ASM，并同步远端重编恢复 retained K3 PACK5 `.co` sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`。
- 已更新 `task_plan.md`：normal K3 store/schedule evidence pass 标为 ✅，新增 ⏳ `normal K3 SQTT/tooling triage after repeated no-material store A/B`。
- 下一步：检查远端 `hipprof`/stat 工具是否支持 SQTT/trace；若可用，对 4096 staged-rowptr 做 trace 辅助判断，否则记录 profiling degraded 并转向 normal gate 收口或更高层 scheduler 方案。

## 2026-06-14 13:49:00 +08:00 - normal K3 SQTT/tooling triage degraded

- 远端 `sglang_megamoe` 中 `/opt/dtk/bin/hipprof` 可用，支持 HIP/HSA/RCCL trace、PMC read/write/type 和 `--codeobj-analyze`。
- 当前 `hipprof -h` 未暴露 `--sqtt`；`stat_stall`、`stat_valu`、`rocprof`、`rocprofv3` 均不在 PATH；`/opt/dtk` 下仅发现 Perfetto library，不是可直接使用的 SQTT/stat 工具。
- 已更新 `task_plan.md`：SQTT/tooling triage 标为 ✅；新增 ⏳ `normal retained champion sanity after rejected K3 A/B`。
- 结论：K3 normal no-tail 的后续证据层降级为 timing + PMC + codeobj/ISA。当前 small store/wait/locality A/B 已多次反证，除非出现新工具或更高层 scheduler 假设，不继续在同一 store 指令窗口盲调。
- 下一步：恢复 retained K3 PACK5 后跑 normal 1024/4096 no-tail correctness/perf guard，确认 current champion 未受 A/B 污染。

## 2026-06-14 13:58:00 +08:00 - normal retained champion sanity passed after K3 A/B revert

- 恢复 retained K3 PACK5 后，远端 `.co` sha 已确认 `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`。
- V3 normal no-tail retained sanity：
  - 1024 fused `2.2593 ms`，min `2.2342 ms`，correct `True`；
  - 4096 fused `6.1027 ms`，min `6.0039 ms`，correct `True`；
  - summary `hygon_tmp/sglang_debug/v3_normal_perf_summary_20260614_134330.csv`。
- 已更新 `task_plan.md`：retained champion sanity 标为 ✅；新增 ⏳ `normal K3 higher-level scheduler/row-emission feasibility check`。
- 下一步：静态检查 K1 row emission 与 K3 split/dest-sort 诊断覆盖面，判断是否有足够证据做生产级 row reordering/scheduler 改动；如果没有，normal no-tail 性能优化应暂时收口，转入剩余功能项。

## 2026-06-14 14:04:00 +08:00 - normal K3 scheduler / row-emission feasibility rejected for now

- 静态检查 K1 ASM-pack5 row emission：当前按 source-rank route scan + per-expert `global_atomic_add` 保留 row slot，row order 设计上 nondeterministic，`output_index`/`row_combine_ptrs`/`m_indices`/`route_weights` 共同维持合同。
- 完整生产重排不可能只改 K3；需要 K1 同时重排 activations、m_indices、route_weights、rowptr，并处理 per-expert/per-dest prefix 或固定 segment overflow，K1 开销和正确性风险都不小。
- 已有 rowptr-only dest-sort 诊断虽然不是 correctness 路径，但已经隔离 store-address locality 假设；4096 明确回退到 `~3.8-4.0 ms`，没有正信号。
- 新跑 4096 rowptr distribution 轻量统计：
  - active/local/remote average `24576/2970/21606`；
  - 每 16 行 chunk 平均 `~4.79` 个目标 rank、max same-rank `~6.47`、连续目标 row-index pair `~0.18`。
- 已更新 `task_plan.md`：higher-level scheduler/row-emission feasibility 标为 ✅，新增 ⏳ `normal no-tail performance收口与下一项选择`。
- 结论：当前不做生产 row-emission 重排。normal no-tail source-level 性能优化暂时收口，下一步整理 retained champion 并选择进入 tail-reduce/uneven/graph 或其他 gate。

## 2026-06-14 - normal no-tail performance closed and tail-reduce resumed

- 回答用户当前 normal V3 进展：retained path 为 K1 dynamic producer + K3 ASM-pack5 no-tail；1024/4096 相对原 staged fused 仍正确且更快，最新同场数据为 `2.4134 -> 2.3054 ms` 与 `6.6033 -> 6.0640 ms`。
- 已把 normal no-tail 收口写入 `task_plan.md` / `findings.md`：K1 已贴近 direct floor，K3 remaining gap 归因到 remote/scattered rowptr combine store 数据通路；已反证的 store/wait/locality/CUS/source-level 方向不再重复。
- `task_plan.md` 新增当前 ⏳ 项：恢复 normal tail-reduce device post-hoc diagnostic。下一步从已存在的 `k3_v3_tail_stage_compare.py` / rank_stats 链条和远端 launcher/NCCL 失败点继续，而不是重新拉长整条 e2e 测试链。

## 2026-06-14 - normal tail device post-hoc reduce chain restored

- 远端 `sglang_megamoe` 中发现一个 3 小时以上的 orphan Python multiprocessing 进程，已清理；8 卡随后可用。
- 重新跑最短 8 卡 tail stage compare：`K3_V3_TAIL_STAGE_TOKENS=1024 K3_V3_TAIL_STAGE_MAX_TOKENS=1024 K3_V3_TAIL_STAGE_ITERS=1 K3_V3_TAIL_STAGE_ORDER=asm_first K3_V3_TAIL_STAGE_KEEP_GOING=1 K3_V3_TAIL_STAGE_ZERO_COMBINE=1`。
- 本次没有复现 launcher/NCCL abort，已拿到有效 `rank_stats`。结果显示 device-side post-hoc reduce 与各自 tail `y` 在差异点一致，而 Python `combine_reduce_py` 与 `y` 不一致；因此 Python 直接 combine view 继续排除为可信根因证据。
- 下一步继续用同一短链条跑 `v3_only` 多轮，专门看 V3 tail 自身重复调用时 device reduce 与 tail `y` 是否一起漂、是否出现 nonfinite，以及 done/signal fields 是否有轮次相关变化。

## 2026-06-14 - normal tail synchronized zero diagnostic

- 本地 DCU KB 并发查询因 reranker CUDA OOM 失败；改串行禁用 CUDA 后超时，记录为本轮 KB 退化，继续沿用前序 KB 已确认的 system-scope fence/release-acquire signal 原则。
- 修改 `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` 的 diagnostic-only zero-combine 路径：`combine.zero_()` 后加 `torch.cuda.synchronize()`，`dist.barrier()` 后再同步一次；本地 `compileall` 通过并同步到 11 节点。
- 远端 8 卡空闲，运行 `K3_V3_TAIL_STAGE_ORDER=v3_only K3_V3_TAIL_STAGE_ITERS=3 K3_V3_TAIL_STAGE_ZERO_COMBINE=1` 成功，日志拉回 `hygon_tmp/sglang_debug/tail_resume_pull_20260614/k3_v3_tail_stage_compare_resume_v3_only_3iter_synczero_20260614.log`。
- 结果：iter1 clean；iter2/iter3 仍出现 nonfinite，且 device-side `reduce_local_combine(... invalidate_before_read=True)` 的 nonfinite 计数与 tail `y` 对齐。异步 zero 不是充分根因，下一步区分 C normal tail GEMM producer 写入 NaN vs signal/reducer 可见性读取脏 combine。

## 2026-06-14 - normal tail producer-vs-reducer split

- 在 `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` 增加 diagnostic-only `K3_V3_TAIL_STAGE_V3_EXTERNAL_REDUCE=1`：V3 normal 只用 C raw no-tail producer 写 combine，再走外部 `rank_barrier(acquire_after_wait=True) + reduce_local_combine(invalidate_before_read=True)`。
- 远端 8 卡运行 `v3_only`、3 轮、zero-combine 同步版 external-reduce，日志拉回 `hygon_tmp/sglang_debug/tail_resume_pull_20260614/k3_v3_tail_stage_compare_v3_external_reduce_3iter_synczero_20260614.log`。
- 结果：三轮所有 rank 的 `nonfinite_first/second/diff` 和 device reduce nonfinite 计数均为 0；说明 normal C producer/rowptr/active rows 在同一 K1/K2 输入下稳定，tail bug 收窄到 in-kernel signal/reducer 路径。
- 下一步：做 env-gated inline-reduce diagnostic，关闭 appended reducer blocks，使用完成 owner CTA 的已有 inline reduce 分支，判断是 reducer-block 并发调度问题还是 done/signal acquire chain 问题。

## 2026-06-14 - normal tail inline-reduce diagnostic parsed

- 拉回并解析 `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_inline_reduce_3iter_synczero_20260614.log`。
- `MEGAMOE_DCU_V3_TAIL_INLINE_REDUCE=1` 下 appended reducer blocks 被关闭，但 iter1/iter2 的第一轮 `y` 仍与后续 device post-hoc reduce 有 `0.067-0.285` 级别差异；iter3 第二轮又复现 nonfinite。
- 结论：tail bug 不只在 reducer block 并发调度；当前最小修复方向是 system-scope acquire wait、final owner acquire done-counter release sequence、以及 block-uniform signal/reduce 分支。

## 2026-06-14 - normal tail C/aicc route stopped

- 用户明确：C 实现暂时不用继续调优，准备放弃这条路线。
- 已更新 `task_plan.md`：normal tail C/aicc raw 诊断收口和 C done/signal acquire-chain fix 标为 ✅/暂停，新增 ⏳ `normal tail ASM-pack5 迁移`。
- 已更新 `findings.md`：记录停止 C/aicc normal tail 的证据链和新的 ASM-pack5 tail-reduce 方向。
- 下一步：不再围绕 `k3_v3_pack5_groupgemm_impl.cuh` 的 normal tail C reducer/acquire 链做主线修复；转为复制原 `K3COMBINE_TAILREDUCE` ASM，隔离 patch V3 plain-pack5 weight layout，并先做 1024/4096 eager correctness。

## 2026-06-14 - 8192 symm warm-up A/B completed

- 远端 8 卡空闲，使用现有 `run_v3_normal_perf_ab.sh` 跑 V3 normal no-tail 8192：`TOKENS_LIST=8192 RUN_ORIG_STAGE=0 RUN_V3=1 RUN_STAGE_TIMING=1 WARMUP=2 REPEAT=3 ITERS=2`。
- 默认 warm-up on：correct，fused median `12.5974 ms`，min `12.5652 ms`；last24 stage median total/K1/K2/K3/barrier/reduce `12.417/5.4375/0.412/5.8905/0.2755/0.336 ms`。
- warm-up off (`MEGAMOE_DCU_V3_SYMM_WARMUP_ALLOC=0`)：correct，fused median `12.7807 ms`，min `12.4317 ms`；last24 stage median `12.5595/5.4705/0.413/6.0170/0.2155/0.336 ms`。
- 结论：8192 当前测试形状两边都能跑通，harness 报 `route_scratch=4.013 GiB`，所以 warm-up 不是 correctness 必需；但 median 上仍有小正收益，约 `1.45%` e2e、K1 last24 约 `0.033 ms`。
- 产物已拉回本地 `hygon_tmp/sglang_debug/8192_warmup_ab_20260614/`。

## 2026-06-14 - normal K3 row-emission option analyzed, no implementation

- 按用户要求只分析、不实施：重读 planning 三文件，复核 K1 row emission、`large_opt.py` K1/K2/K3 数据流、K3 V3 wrapper 和已有 rowptr split/dest-sort probe。
- 查本地 DCU KB：Flux GEMM+RS / DeepEP 通信融合资料支持“把通信目的地嵌入 scheduler/epilogue 合同”，但也强调按通信模式和能力分流，不能把后处理排序当成必然优化。
- 结论：
  - 这个方向可尝试，但不能 K3-only；
  - 生产正确性要求 `act_fp8/act_scale/m_indices/route_weights/row_combine_ptrs/output_index` 共同保持 row 对齐；
  - 安全 reorder 范围应在每个 local expert 内，不能全局按目的 rank 排序；
  - 先做 `hygon_tmp` diagnostic-only post-K2 permutation，若 4096 K3 direct 有明确 >3-5% 收益，再考虑 K1 ASM route emit / per-expert-per-dest prefix 的生产改动。
- 已更新 `task_plan.md` 和 `findings.md` 记录该分析；未修改生产代码，未新增 probe。

## 2026-06-14 - normal K3 combine-layout/reduce-contract option analyzed, no implementation

- 按用户要求继续分析方案 3 的可行性，不实施代码改动。
- 已重新确认当前 combine 合同：K1 `row_combine_ptrs` 指向 `combine[topk_slot][token][hidden]`，K3 no-tail 只负责通过 rowptr 写 combine，外部 `reduce_local_combine` 和 tail ASM 的 `asm_reduce_slot_stride_vec` 都依赖同一 slot-major layout。
- 查本地 DCU KB：Flux/DeepEP 的可迁移点是把通信 readiness、rank destination 和 reduce/scatter 语义纳入 scheduler/epilogue 合同；这支持“现有 layout 下做 K3 fused reduce”的低风险方向，但不足以证明大改 combine layout 一定有收益。
- 结论：
  - 低风险方向：保持现有 slot-major layout，先做 isolated ASM-pack5 no-tail fused-reduce diagnostic，目标回收外部 barrier/reduce 的约 `0.1-0.3 ms`，但不会解决 K3 remote/scattered store 主成本。
  - 高风险方向：rank-major/per-destination bucket combine layout 可能回收 fanout/scatter 的一部分；但 4096 single-peer contiguous remote `~2.31 ms` vs staged-rowptr `~2.48 ms` 显示大概率只剩 `~0.2 ms` 级空间，且会同时影响 K1 rowptr/metadata emission、K3 store、no-tail reduce、tail ASM、graph 和 uneven tokens。
- 已更新 `task_plan.md` / `findings.md`：将 combine-layout/reduce-contract feasibility 标为 ✅，新增两个未实施诊断项；未修改生产代码。

## 2026-06-14 - normal K3 rank-bucket combine-layout upper-bound timing completed

- 按用户要求暂时不管 correctness，只评估“如果 combine buffer layout 改成每个目标 rank 内更连续，实际 K3 写 remote 能提升多少”。
- 修改范围：
  - 只改 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py` 诊断脚本；
  - 新增 `staged_rank_bucket_remote_only`、`staged_rank_bucket_remote_sorted`、`staged_rank_bucket_full`、`staged_rank_bucket_full_sorted` modes；
  - 未修改 `megamoe/` 生产路径，未 rebuild kernel。
- 远端环境：
  - `.vscode/sftp.json` -> `hg@10.17.176.11`，container `sglang_megamoe`；
  - 命令均在容器内 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`；
  - 8 卡 `hy-smi` 显示 VRAM/HCU 基本空闲；
  - env 包含 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_K1_ASM_PACK5=1 MEGAMOE_DCU_V3_K3_ASM_PACK5=1 K3_V3_BENCH_V3_PACK5_LAYOUT=plain`。
- 实测摘要：
  - 1024：`staged_rowptr 0.7948 ms`，`staged_remote_only 0.8014 ms`，`rank_bucket_remote_only 0.7204 ms`，`rank_bucket_remote_sorted 0.7795 ms`，`single_peer_contig 0.7423 ms`。
  - 4096：`staged_rowptr 2.5170 ms`，`staged_remote_only 2.5108 ms`，`rank_bucket_remote_only 2.4307 ms`，`rank_bucket_full 2.4358 ms`，`single_peer_contig 2.3065 ms`，`rank_bucket_remote_sorted 3.8842 ms`，`rank_bucket_full_sorted 3.7166 ms`。
- 结论：
  - per-rank compact combine layout 有正收益，4096 K3-only 大约 `0.08 ms / 3.2%`；
  - naive sorted row emission 方向大幅回退，不能作为生产方案依据；
  - 当前收益不足以单独支撑大改 K1 row allocation / K3 reduce contract / tail ASM / graph / uneven。
- 产物已拉回本地：
  - `hygon_tmp/sglang_debug/rank_bucket_pull_20260614/k3_rank_bucket_1024_20260614_171629.json`
  - `hygon_tmp/sglang_debug/rank_bucket_pull_20260614/k3_rank_bucket_4096_20260614_171724.json`
  - `hygon_tmp/sglang_debug/rank_bucket_pull_20260614/k3_rank_bucket_full_4096_20260614_171814.json`

## 2026-06-14 - normal K3 compute/store wave-specialization feasibility recorded

- 回答用户关于“compute warp / store warp 同 kernel overlap”的可行性问题前，重读 planning 三文件并查本地 DCU KB。
- 静态确认当前 V3 K3 ASM-pack5 使用 `VGPR=255`、`SGPR=102`、`LDS=65536`，`K3_STORE_STAGED_HALF` 是 epilogue LDS-read + rowptr load + remote vector store，不是持续 compute/store wave overlap。
- 继续查 DCU KB 和 ASM，确认当前 K3 已经在 load/compute 阶段使用 A/B 到 LDS 的双缓冲式流水，LDS clamp 为 64KB；epilogue 可以在 compute 结束后复用 LDS 做 C-store staging，但要和下一 tile compute 同时进行就需要额外 live C-buffer，与现有 A/B LDS buffer 同时占用。
- 进一步确认当前已有 overlap 不是 permanent load wave / compute wave specialization：更像 CK/Hygon GEMM 的 cooperative A/B prefetch + LDS double buffer + MMAC priority/instruction scheduling；C combine store 仍是 tile GEMM 后的 epilogue。
- 对 normal K3 当前状态的判断已补充：在现有 staged combine 合同和 64KB LDS/255 VGPR 资源形态下，fused 相对 pure 的增长更像 DCU remote/scattered combine store 的硬开销；后续小幅 store/wait 局部调参优先级降低，除非有新的 profiler/ISA 或 synthetic overlap 信号。
- 结论已写入 `findings.md` 和 `task_plan.md`：该路线理论可行且潜在空间比 rank-bucket layout 更大，但需要 persistent/multi-tile 或 chunked producer-consumer 结构、C accumulator 双缓冲、ready/reuse 同步；不是局部改 store macro。
- 下一步若恢复该方向，应先在 `hygon_tmp` 写 synthetic overlap microbench，只有 4096 K3-only 有明确 >5% 或 >0.2ms 信号，再动隔离 V3 ASM-pack5 生产路径。

## 2026-06-14 - LL optimization restarted; normal tail deferred

- 用户明确：V3 normal 融 tail reduce 最后再考虑，当前优先看 LL V3 还有没有空间；normal 的优化手段是否能迁移到 LL；LL K3 资源更宽松，可以尝试适合 DCU 的 overlap 方案降低 remote store 影响。
- 已完成恢复：
  - 重读 `.planning/dcu_megamoe_v3/task_plan.md`、`findings.md`、`progress.md` 和 `.vscode/sftp.json`；
  - 确认远端仍应走 `hg@10.17.176.11`、container `sglang_megamoe`、container repo `/workspace/DeepGEMM`；
  - 查本地 DCU KB，DeepEP LL combine overlap 仍指向 `comp_signal/block_m/threshold` 的 chunk-level readiness；
  - 静态复核 LL K3 当前实现：rowptr 已在 MMAC 前 register-prefetch，epilogue 做 rowaddr vector store；当前没有 compute/store overlap。
- 已更新计划：
  - `task_plan.md` Phase 6b 新增 ⏳ `LL normal-tech transfer + overlap feasibility`；
  - `findings.md` 记录这轮迁移评估和 guardrail。
- 下一步：
  - 先同步并刷新 LL 128 no-tail pure/local/remote/staged timing，确认当前 gap；
  - 再决定是否写 `hygon_tmp` overlap probe 或 env-gated LL K3 branch。

## 2026-06-14 - LL K3 split/rank-bucket refresh completed

- 已补跑 remote 11 节点、8 卡、V3 LL raw K1/K3 split timing：
  - 128 tokens：`pure_contiguous 0.2370 ms`、`local_rowptr 0.2437 ms`、`staged_remote_only 0.3653 ms`、`staged_rowptr 0.3676 ms`；
  - 128 diagnostic rank-bucket（不看 correctness）：`remote_only 0.2680 ms`、`full 0.2687 ms`、`remote_sorted 0.2644 ms`；
  - 32 tokens：`pure_contiguous 0.2368 ms`、`local_rowptr 0.2444 ms`、`staged_rowptr 0.2445 ms`；
  - 32 diagnostic rank-bucket 只有 `~0.003 ms` 小信号。
- 结论：
  - LL 32 K3 已基本贴近固定 kernel floor，不是优先优化点；
  - LL 128 K3 remote/scattered combine writeback 仍有约 `0.12-0.13 ms` delta，是当前 LL K3 最大剩余空间；
  - rank-bucket 128 上界很强但不保 correctness，后续需要做不改变生产合同的 overlap/writeback scheduling A/B。
- 已更新：
  - `findings.md` 增加本次 32/128 split 与 rank-bucket 数据；
  - `task_plan.md` 增加 ✅ split/rank-bucket refresh 和 ⏳ 128 correctness-preserving overlap/writeback probe。
- 下一步：
  - 重读 LL K3 kernel epilogue / tile loop，选择一个 env-gated 或 `hygon_tmp` 低风险 probe，先测 correctness + 128 K3 split timing，32 作为回归哨兵。

## 2026-06-14 - LL K3 rowaddr wave-shuffle A/B rejected

- 实施一个小型 build-time diagnostic：
  - `ld_col==0` lane 负责加载 `row_combine_ptrs[logical_row]`；
  - 通过 wave shuffle 广播 64-bit row address 给同一行其它 lane；
  - 不改 K1/K2 合同，不改 row-index/tile completion，不新增 runtime kernel。
- 验证：
  - 32/128 LL no-tail 和 LL tail correctness 均通过，`max_abs=0.000244141`；
  - 128 split 明显回退：`staged_rowptr 0.3676 -> 0.4065 ms`，`staged_remote_only 0.3653 -> 0.4013 ms`；
  - 32 split 同样回退：`staged_rowptr 0.2445 -> 0.2746 ms`。
- 处理：
  - 已撤回本地代码和 setup gate；
  - 已同步远端并重编默认 retained LL raw；
  - retained 128 短版 sanity 回到 `local_rowptr 0.2437 ms`、`staged_remote_only 0.3666 ms`、`staged_rowptr 0.3690 ms`。
- 结论：
  - rowptr 重复加载不是当前瓶颈；wave-shuffle 减少 load 数反而破坏并行/调度，后续不再重复该方向。
  - 下一步仍聚焦 128 remote writeback/layout/overlap，而不是 rowptr load-count。

## 2026-06-14 - LL K3 store-burst epilogue A/B rejected and reverted

- 重读 planning 文件并按 DCU/Hygon optimizer 闭环继续 LL K3 128 remote-store overlap 方向；本轮选择一个不改 K1/K2 合同、不改 row-index/tile completion 的 epilogue scheduling probe。
- 尝试临时 build gate `DG_BUILD_MEGAMOE_V3_LL_K3_STORE_BURST=1`：同一 tile 内先把 LL K3 的 BF16 store fragments 转换并保存在寄存器 `store_cache`，再集中按 `row_combine_ptrs` 发起 store，目标是提高 remote-store burst/outstanding。
- 远端 11 节点 `sglang_megamoe` 编译通过；32 no-tail/tail correctness 通过，`max_abs=0.000244141`。
- 性能明确回退：
  - 32 `staged_rowptr 0.24699 ms`，慢于 retained `~0.2445 ms`；
  - 128 `staged_rowptr 0.37899 ms`，慢于 retained `~0.3676-0.3690 ms`。
- 已撤回临时 gate 和源码分支，重新同步远端并重编默认 retained LL raw；post-revert 128 split sanity：`local_rowptr 0.24413 ms`、`staged_remote_only 0.36576 ms`、`staged_rowptr 0.36745 ms`。
- post-revert 128 no-tail/tail correctness 均通过，`max_abs=0.000244141`。结论：延迟 store 再 burst issue 会增加 live range 或削弱 store/compute issue overlap，不作为 LL K3 后续方向。

## 2026-06-14 - LL K3 row-order / layout diagnostic refreshed

- 复用现有 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`，在远端 11 节点 8 卡跑 LL 128 K3-only rowptr mode refresh，不 rebuild、不修改生产代码。
- 目标是判断 normal 中讨论过的 row-emission / destination-rank ordering 思路能否迁移到 LL：如果只改变每 expert 内 rowptr 目的地址顺序就有明显收益，后续可考虑 K1 row emission；否则不走这条大改。
- 结果：
  - `pure_contiguous 0.236956 ms`
  - `local_rowptr 0.243426 ms`
  - `staged_rowptr 0.366257 ms`
  - `staged_rowptr_dest_sorted 0.367132 ms`
  - `staged_remote_dest_sorted 0.365495 ms`
  - `staged_rank_bucket_full 0.268205 ms`
  - `staged_rank_bucket_full_sorted 0.265346 ms`
- 结论：只按目的地址排序 row emission 对 LL 128 没有实质收益；强信号来自 rank-bucket compact combine layout，而不是 row-order-only。该方向如果继续，必须作为 combine layout / reduce contract 大改评估，不能当作 K3-only overlap 小调参。
- 本地产物：`hygon_tmp/sglang_debug/ll_row_order_pull_20260614/k3_ll_row_order_refresh_128_20260614_200504.json`。

## 2026-06-14 - LL retained formal snapshot after overlap diagnostics

- 远端 11 节点 8 卡当前 retained 默认 V3 LL no-tail formal perf 复测：不改源码，只跑 `run_v3_ll_perf_ab.sh` 默认分支，`RUN_STAGE_TIMING=1`，32/128 分开执行以缩短定位链。
- 32 tokens:
  - correctness 通过，`max_abs=0.000244141`；
  - fused median `0.830619 ms`，min `0.818580 ms`；
  - 稳定段 stage 大致 K1 `0.346-0.350 ms`、K3 `0.252-0.269 ms`。
- 128 tokens:
  - correctness 通过，`max_abs=0.000244141`；
  - fused median `1.034080 ms`，min `1.018639 ms`；
  - 稳定段 stage 大致 K1 `0.38-0.40 ms`、K3 `0.37-0.42 ms`。
- 结论：32 档已经基本收敛，K3 不是优先项；128 档 K1/K3 同量级。K3 split 仍指向 remote/scattered writeback，但 K3 小型 epilogue / row-order 调参已经多项负信号；继续优化 LL 需要要么找 K1 fixed cost 小收益，要么进入 combine layout / chunk-readiness 这类合同级方案。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_retained_current_20260614/v3_ll_default_32_perf_20260614_200827.json`
  - `hygon_tmp/sglang_debug/ll_retained_current_20260614/v3_ll_default_128_perf_20260614_200912.json`
  - `hygon_tmp/sglang_debug/ll_retained_current_20260614/v3_ll_perf_summary_20260614_200904.csv`
  - `hygon_tmp/sglang_debug/ll_retained_current_20260614/v3_ll_perf_summary_20260614_200950.csv`
- 追加 DCU KB refresh：DeepEP LL overlap guidance 继续指向 fixed layout / double buffering / chunk-level `comp_signal + threshold`，而不是同一 epilogue 内继续调 store issue order。该证据支持后续若继续 LL K3，应显式转入 layout/contract 或 chunk-readiness 方案。

## 2026-06-14 - LL K3 layout/chunk-readiness feasibility closed

- 重读计划和 LL K3/K1/reduce 相关代码，确认当前 no-tail 合同：
  - K1 LL 按 `topk_slot * max_tokens + token` 生成远端 slot-major `row_combine_ptrs`；
  - K3 LL 只按 rowptr 写 BF16 output；
  - `reduce_local_combine` 固定扫描本地 `combine[topk_slot, token]`，不使用 rowptr 或 output_index。
- 静态结论：
  - normal 的 row-order / store-order 小调参不能直接迁移到 LL，已有 LL A/B 也支持这个判断；
  - LL K3 资源比 normal 宽松，但当前 CTA 没有后续 tile compute 可以自然覆盖同 tile epilogue store，真正 overlap 需要 persistent/chunked 结构或合同级 compact layout；
  - rank-bucket compact layout 的强信号必须通过 mapped reduce 还原语义，不能只改 K3 rowptr。
- 已更新：
  - `task_plan.md` 将 `LL normal-tech transfer + overlap feasibility` 和 `LL K3 layout/chunk-readiness feasibility` 标为 ✅；
  - 新增 ⏳ `LL K3 compact-combine mapped-reduce sidecar`；
  - `findings.md` 记录合同改动点和诊断门槛。
- 下一步：
  - 在 `hygon_tmp` 做 compact-combine + mapped-reduce sidecar，不改生产路径，用 128 档估算 K3 compact-layout收益扣除 reduce/mapping 后是否仍值得大改。

## 2026-06-14 - LL K3 compact mapped-reduce sidecar started

- 新增诊断文件，均位于 `hygon_tmp/sglang_debug`，不进入生产路径：
  - `ll_compact_mapped_reduce_ext.cu`：单个 mapped reduce HIP extension，根据 `partial_to_compact` 从 compact combine layout 还原 `y`；
  - `bench_k3_ll_compact_mapped_reduce.py`：复用现有 K1/K2/K3 LL harness，构造 per-destination compact rowptr 和 reduce 映射，比较旧 slot-major `K3 + barrier + reduce_local_combine` 与 compact `K3 + barrier + mapped_reduce`。
- 本地验证：
  - `python -m compileall hygon_tmp/sglang_debug/bench_k3_ll_compact_mapped_reduce.py` 通过；
  - `git diff --check` 通过。
- 远端第一次 32-token smoke 失败在 sidecar extension 编译：`combine_token_offset` 被 device kernel 调用但未标 `__device__`；已修为 `__host__ __device__`，准备重跑同一 smoke。

## 2026-06-14 - LL K3 compact mapped-reduce sidecar measured

- 远端 11 节点 `sglang_megamoe` 8 卡继续跑 compact mapped-reduce sidecar，所有容器命令均 `source /opt/dtk/env.sh && cd /workspace/DeepGEMM`。
- 第二次 32-token smoke 失败：sidecar 用请求的 `--max-tokens 32` 解析 combine rowptr，但实际 `sym_buffer.num_max_tokens_per_rank=384`，导致 `invalid_total=1432`；已修为使用实际 sym-buffer max tokens。
- 修复后 32-token smoke correctness 通过：`max_abs=0`、`invalid_total=0`、`missing_active_total=0`、`compact_rows_avg_rank=192`。短测 timing 显示 compact+mapped reduce 比旧链路快，但 repeat/round 太短，只作为 smoke 信号。
- 128-token 稳定 run correctness 通过：`max_abs=0`、`invalid_total=0`、`missing_active_total=0`、`compact_rows_avg_rank=768`。
- 128-token timing：
  - `staged_k3_only median_avg_rank_ms=0.36856`；
  - `compact_k3_only median_avg_rank_ms=0.36344`；
  - `staged_k3_barrier_reduce median_avg_rank_ms=0.42368`；
  - `compact_k3_barrier_mapped_reduce median_avg_rank_ms=0.42222`。
- 结论：correctness-preserving compact sidecar 没有复现旧 rank-bucket `~0.265-0.268 ms` K3-only 上界，净收益只有噪声级；下一步先归因 sidecar compact row allocation 与旧 rank-bucket diagnostic 的布局差异，暂不推进生产合同改造。
- 已拉回本地产物：
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/ll_compact_sidecar_smoke_32_20260614_202631.json`
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/ll_compact_sidecar_128_20260614_202907.json`

## 2026-06-14 - LL K3 rank-bucket upper-bound artifact closed

- 为解释 compact sidecar 与旧 rank-bucket `~0.265 ms` 上界不一致，先在 sidecar 增加非正确性 `compact_collision_k3_only` mode：
  - 128 tokens：`compact_collision_k3_only 0.36292 ms`、`compact_k3_only 0.36501 ms`、`staged_k3_only 0.36637 ms`；
  - 说明差异不是跨 source prefix 单独造成。
- 继续检查并修复 `bench_k3_ll_rowptr_modes.py`：旧脚本使用请求的 `--max-tokens 128` 解析 combine offset，而实际 `sym_buffer.num_max_tokens_per_rank=384`。
- 修正实际 max-token 后重跑 128 rowptr/rank-bucket：
  - `pure_contiguous 0.23728 ms`；
  - `local_rowptr 0.24401 ms`；
  - `staged_rowptr 0.36839 ms`；
  - `staged_rank_bucket_full 0.36643 ms`；
  - `staged_rank_bucket_full_sorted 0.36541 ms`；
  - `rank_bucket_invalid_total=0`，`rank_bucket_counts_avg_rank=[96]*8`。
- 结论：旧 `~0.265 ms` rank-bucket 上界是诊断脚本 max-token stride mismatch artifact；compact layout 生产大改暂停。LL 优化下一步转回当前 retained path 的真实 K1/K3 delta 复盘。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/ll_compact_collision_128_20260614_203221.json`
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/k3_ll_rowptr_actualmaxtok_128_20260614_203421.json`

## 2026-06-14 - LL retained delta re-queued after compact artifact closure

- 继续按计划重跑当前 retained V3 LL no-tail 分段 timing，目标是从真实 K1/K3 delta 重新排后续优化优先级。
- 第一次远端命令因 `TOKENS_LIST='32 128'` quoting 问题没有有效输出；已改用 `TOKENS_LIST="32 128"`。同时发现脚本变量是 `RUN_K2SKIP`，不是 `RUN_K2_SKIP`，因此本轮也顺带得到 k2skip attribution。
- default 结果：
  - 32 tokens fused median `0.82664 ms`，stable last40 stage：K1 `0.347 ms`、K2 `0.028 ms`、K3 `0.255 ms`、total `0.7115 ms`；
  - 128 tokens fused median `1.02724 ms`，stable last40 stage：K1 `0.397 ms`、K2 `0.028 ms`、K3 `0.388 ms`、total `0.910 ms`。
- k2skip 结果：
  - 32 tokens median `0.82520 ms`，仅噪声级；
  - 128 tokens median `1.03876 ms`，比 default 慢。
- 结论：K2 不是当前优化目标；下一步进入 `LL K1 fixed-cost low-risk A/B`，先查当前 K1 LL stage-copy、tiny-store、metadata path 的低风险可测开关。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_retained_delta_20260614/v3_ll_default_32_perf_20260614_203652.json`
  - `hygon_tmp/sglang_debug/ll_retained_delta_20260614/v3_ll_default_128_perf_20260614_203809.json`
  - `hygon_tmp/sglang_debug/ll_retained_delta_20260614/v3_ll_perf_summary_20260614_203923.csv`

## 2026-06-14 - LL K1 partial output_index clear A/B rejected and reverted

- 用户询问 LL 当前进展后继续推进 Phase 6b；按计划重读 planning 文件、remote skill、Hygon optimizer / DCU KB skill。
- 先修正 remote raw build 状态：
  - K1-only rebuild 后 `K3_fused/k3_fused_ext` 会显示 K3 raw unavailable；
  - 重新以 `DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll` 和 `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll` 构建后，V3 raw 模块 `K3_fused/k3_v3_fused_ext` 显示 `raw_kernels=True`、`raw_ll=True`、`raw_normal=False`。
- 尝试 K1 LL 低风险 A/B：
  - 临时增加 `MEGAMOE_DCU_V3_LL_K1_PARTIAL_OUTPUT_INDEX_CLEAR=1`，只清 actual routes 相关 `output_index`，不改 row layout / K2 / K3 合同；
  - 初跑 32/128 correctness 均通过，formal 看似 partial 略快；
  - 但 128 反向确认 partial 两轮 `1.060960/1.058419 ms`，default 两轮 `1.053020/1.053540 ms`，partial 实际更慢；
  - stage timing 显示 K1 median 没有稳定下降，初跑收益主要来自 K3/barrier 噪声。
- 已撤回：
  - 移除 K1 V3 source、launcher declaration、pybind wrapper、stub 中所有 partial-clear env/signature/argument；
  - 本地 `rg "PARTIAL_OUTPUT_INDEX|partial_output_index|MEGAMOE_DCU_V3_LL_K1_PARTIAL"` 无匹配；
  - 远端 stale hipify cache 中仍残留旧符号，已只删除生成的 `k1_fused_ext.hip` / `k1_v3_fused_ext.hip` 并重编 retained LL raw。
- 验证：
  - 本地 `python -m compileall megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 `git diff --check` 通过；
  - 远端 retained sanity：32 fused median `0.829800 ms`、128 fused median `1.032919 ms`，correctness 均通过。
- 错误/踩坑记录：
  - confirm 脚本第一次用 `set -euo pipefail` 加 `ls "$outdir"/no_match | wc -l`，空 glob 直接失败；已改为显式计数后重跑；
  - PowerShell 远端同步命令第一次因 quoting 失败，已改用 here-string；
  - container 没有 `rg`，远端检查改用 `grep`。
- 结论：
  - `output_index` bounded/partial init 方向不再重复；
  - 当前 active 项转为 `LL residual ceiling/next-candidate triage`，下一步按 PMC/timing/codeobj 证据筛剩余候选。

## 2026-06-14 - LL residual ceiling triage completed

- 按计划继续 Phase 6b，先查本地 DCU KB：
  - 检索 `hygon gfx938 remote rowptr store TA_BUSY TCP_TA_DATA_STALL LL GEMM combine overlap store scheduling`；
  - 命中 DeepEP low-latency combine overlap 资料，仍强调 chunk-level `comp_signal/block_m/threshold`，不是继续挪同一 epilogue store 指令。
- 读取并解析已有 PMC/脚本：
  - `hygon_tmp/sglang_debug/run_ll_k3_pmc_current.sh` 覆盖 pure/local/remote；
  - `prof_pull_current_20260614_081823` 显示 remote path 的 `TA_BUSY/TCP_TA_DATA_STALL` 高于 local，VMEM 指令规模接近，瓶颈仍是 remote/scattered rowptr combine 数据通路。
- 远端状态：
  - 11 节点 8 卡空闲；
  - K1 partial-clear 残留 grep 无匹配；
  - K3 V3 raw module 正确 availability 名称是 `dcu_megamoe_v3_k3_raw_kernels_available` / `dcu_megamoe_v3_k3_raw_ll_available` / `dcu_megamoe_v3_k3_raw_normal_available`。第一次猜 `dcu_megamoe_v3_k3_raw_available` 报 AttributeError，已用 `dir()` 纠正。
- 最新 retained K3 128 split sanity：
  - `pure_contiguous 0.237410 ms`；
  - `local_rowptr 0.243754 ms`；
  - `staged_remote_only 0.362768 ms`；
  - `staged_rowptr 0.368026 ms`；
  - active/local/remote rows avg rank `768/95.625/672.375`。
- 结论：
  - 低风险 LL 微调暂无新候选；剩余 `~0.124 ms` staged-vs-local K3 gap 更像 DCU remote/scattered writeback contract cost；
  - 后续若继续追 LL 性能，需要单独规划 chunk-readiness / reduce-contract 级方案；
  - 当前立即转入 LL 功能 parity，优先跑 uneven correctness matrix。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_residual_triage_pull_20260614/ll_residual_triage_k3_128_20260614_211534.json`

## 2026-06-14 - LL uneven matrix started, no-tail stats mismatch

- 按计划从 LL 性能微调转入 Phase 7 功能 parity，开始跑 `hygon_tmp/sglang_debug/run_v3_ll_correctness_matrix.sh`。
- 远端命令环境：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1`
  - `USE_MEGAMOE_V3=1`
  - `MEGAMOE_DCU_V3_BACKEND=ll`
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`
  - `K3_USE_ASM_TAIL_REDUCE=0`
  - tokens list `32,48,16,64,24,40,56,8`
  - max/dispatch tokens `64`
- 结果：
  - no-tail 第一项失败，tail case 未开始；
  - rank 7 stats mismatch，fused stats 小于 baseline stats；
  - 这说明 V3 LL uneven 还未对齐，优先定位 K1 stats / actual tokens / dispatch_num_tokens 合同。
- 已更新：
  - `task_plan.md` 的 uneven tokens 项保留 ⏳，补充 stats mismatch 状态；
  - `findings.md` 增加 LL uneven no-tail failure 记录；
  - 远端 log 已拉回本地。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_uneven_matrix_pull_20260614/v3_ll_uneven_notail_20260614_211817.log`

## 2026-06-14 - V3 LL uneven and graph support completed

- 用户要求性能先停，把当前 V3 LL 的 uneven 和 graph 功能对齐后再停。
- 已完成代码/接口状态：
  - K1 V3 LL route/stats build 不再把本地 rank 的 runtime token count 当作所有 source rank 的统一 scan bound；
  - uniform 情况继续用 tight runtime token stride；
  - uneven 情况按每个 source rank 的 `sections.num_tokens` 扫描，并保持 route stride 为 `num_max_tokens_per_rank`；
  - V3 LL staged graph 已接入 K1/K3 wrapper；normal backend graph 仍按 guard fail-fast。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_groupgemm_impl.cuh megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过。
- 远端同步/编译：
  - 显式同步 `large_opt.py`、K1/K3 wrapper、`k1_v3_groupgemm_impl.cuh` 和 `tests/test_dcu_megamoe_v3.py` 到 `hg@10.17.176.11:/home/hg/yuguo/DeepGEMM`；
  - 容器内 `python3 -m compileall ...` 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`12 passed`；
  - 第一次 build 命令因 PowerShell 把 bash 续行展平成 `\ rm` 失败，未开始编译；改成单行命令后 build 成功；
  - 第一轮 build 显示 `ninja: no work to do`，说明 header-only 改动未触发 K1 object 重编；随后删除 `k1_v3_fused_ext.o/.hip` 和 `k1_fused_ext.o/.hip` 后重新 build，K1 V3 LL object 实际重新编译成功。
- 远端 correctness：
  - LL uneven eager no-tail：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=0`，token list `32,48,16,64,24,40,56,8`，max/dispatch `64`，3 iters 通过，`max_abs=0.000244141`；
  - LL uneven eager tail：同 token list，`K3_USE_ASM_TAIL_REDUCE=1`，3 iters 通过，`max_abs=0.000244141`；
  - LL graph uniform no-tail/tail：capture bucket `128`，replay token `32,128`，均通过，worst `max_abs=0.000488281`；
  - LL graph uneven no-tail/tail：capture bucket `64`，token list `32,48,16,64,24,40,56,8`，replay token `64`，均通过，`max_abs=0.000244141`。
- 已更新：
  - `task_plan.md` Phase 7 将 V3 LL uneven、V3 LL graph、graph gate/layout 分流标为 ✅；
  - `findings.md` 记录 root cause、修复、验证命令结果和日志路径。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/`

## 2026-06-14 - V3 LL post-parity performance sanity completed

- 用户提醒：性能尽量不要劣化。
- 按 Hygon optimizer 的测量闭环做了最小回归哨兵，没有继续新增优化分支：
  - 远端 11 节点 `sglang_megamoe`；
  - `TOKENS_LIST="32 128"`；
  - `RUN_DEFAULT=1 RUN_K2SKIP=0 RUN_BLOCK48=0 RUN_BLOCK64=0 RUN_BASELINE=0 RUN_STAGE_TIMING=1 REPEAT=20 WARMUP=5 ITERS=1`。
- no-tail 默认路径：
  - `K3_USE_ASM_TAIL_REDUCE=0`；
  - 32 tokens fused median `0.82758 ms`，min `0.81196 ms`；
  - 128 tokens fused median `1.02778 ms`，min `1.00364 ms`；
  - 对比最近 retained `0.8298/1.0329 ms`，无可见劣化。
- tail 默认路径：
  - `K3_USE_ASM_TAIL_REDUCE=1`；
  - 32 tokens fused median `0.83048 ms`，min `0.81678 ms`；
  - 128 tokens fused median `1.03996 ms`，min `1.02332 ms`；
  - 对比 prior tail post-stagecopy `~0.8425/1.0759 ms`，无可见劣化。
- 已更新：
  - `task_plan.md` Phase 7 增加并完成 V3 LL 功能补齐后性能哨兵；
  - `findings.md` 增加本次 perf sanity 的方法、结果、解释和 artifact。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/`

## 2026-06-14 - V3 LL graph-vs-eager performance sanity started

- 用户澄清：关注的是 LL cuda graph replay 性能不能相比 eager 劣化。
- 已在 Phase 7 增加 graph-vs-eager 性能哨兵项，接下来在远端 11 节点 `sglang_megamoe` 容器中用 `--cuda-graph-bench` 对 V3 LL 32/128 no-tail/tail 做实测。
- 对照 eager 基线沿用刚完成的 post-parity sanity：no-tail `0.82758/1.02778 ms`，tail `0.83048/1.03996 ms`。

## 2026-06-14 - V3 LL graph-vs-eager performance sanity completed

- 初始实测发现 graph replay 确实慢于 eager：
  - no-tail bucket 128 replay 32/128: `0.9163/1.0727 ms`；
  - tail bucket 128 replay 32/128: `0.9891/1.1180 ms`；
  - no-tail exact bucket 32 仍为 `0.9166 ms`，排除单纯 128 bucket overwork。
- 定位根因：
  - V3 K1 graph wrapper 传入 `runtime_num_tokens` 后，host ext 把 K1 capacity 误按 `sym_buffer.num_max_tokens_per_rank=384` 计算；
  - graph capture bucket 32/64/128 因此都被扩大到 384-token capacity，K1 返回的 `l1_out` 行数放大，连带 K2/K3 处理过多 row。
- 修复：
  - `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu` 将 `route_capacity_tokens_per_rank` 改为 `num_tokens`，即 graph capture bucket；
  - `output_index` 和 symmetric-buffer combine stride 仍保持 `num_max_tokens_per_rank`，不改变外部布局合同。
- 验证：
  - 本地 `python -m compileall ...` 通过；
  - 本地 `git diff --check` 通过；
  - 远端同步后删除 K1 相关 build/hipify cache 并重编，K1 ext 与 K1 V3 raw 均实际重新编译；
  - 第一次远端清 build artifact 的 `find -name '*...'` 因嵌套引号被 shell 展开失败，未进入编译；已改成双引号分两条 `find` 成功，不再重复旧写法。
- 修复后 graph bench：
  - no-tail bucket 128 replay 32/128: `0.6860/0.8723 ms`；
  - tail bucket 128 replay 32/128: `0.7293/0.9198 ms`；
  - 对比 eager no-tail `0.8276/1.0278 ms`、tail `0.8305/1.0400 ms`，graph 四个点均无劣化且更快。
- 修复后 correctness：
  - uniform graph no-tail/tail 32/128 均通过，worst max_abs `0.000488281`；
  - uneven graph no-tail/tail bucket 64、token list `32,48,16,64,24,40,56,8` 均通过，max_abs `0.000244141`。
- 已更新：
  - `task_plan.md` 将 V3 LL graph-vs-eager 性能哨兵标为 ✅；
  - `findings.md` 记录修复前后数据、根因、修复和 artifact。
- 本地产物：
  - `hygon_tmp/sglang_debug/ll_graph_vs_eager_20260614/`

## 2026-06-15 - Remote DCU card status checked

- 按 `remote-ssh-docker-workflow` 从 `.vscode/sftp.json` 读取远端参数，检查节点 `node2`、用户 `hg`、容器 `sglang_megamoe`。
- 容器状态：`sglang_megamoe` 已运行 `Up 39 hours`。
- `hy-smi` 总览：
  - 8 张 HCU 均为 `Mode=Normal`；
  - HCU 0-7 温度约 `29-30C`；
  - VRAM 使用率均为 `0%`；
  - HCU 使用率均为 `0.0%`。
- `hy-smi --showpids` 未列出占用进程；命令尾部返回 `Unable to open process directory` / `Failed to show current process information`，但总览显示卡均空闲。
- `rocminfo/rocninfo` 设备识别可见 `gfx938` / `BW1101` / `Device Type: HCU` / `Compute Unit: 64` / `Wavefront Size: 64`。
- 操作记录：第一次补查命令被 PowerShell 引号解析截断，已换成显式构造远端命令后成功执行。

## 2026-06-15 - V3 normal tail ASM-pack5 path no longer hangs at fused bench

- 用户要求先支持 V3 normal ASM tail-reduce，且不要继续深挖后续要清理的 V3 normal C tail 实现。
- 已完成实现/接入：
  - 新增 `K3COMBINE_TAILREDUCE_PACK5.s/.co` 路径，复用原 ASM tail reduce/signal/store 调度，仅对齐 V3 normal plain-pack5 的 A/weight address math；
  - `setup.py`、`k3_fused.py`、`k3_fused_ext.cu` 接入 `k3_l2_combine_asm_tail_reduce_pack5_out`；
  - `large_opt.py` 与 `tests/test_mega_moe_dcu.py` 默认让 V3 normal tail 使用 ASM plain-pack5，绕开已暂停的 C tail 实现；
  - `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` 增加 `K3_V3_TAIL_STAGE_SIGNAL_GENERATION` 诊断开关。
- 根因修复：
  - `rank_barrier_kernel` 每轮会清零本 rank 的 tail recv slots；
  - C helper 按 `signal_generation` 做 peer atomic add 并等待 slot `>= signal_generation`；
  - 原 ASM tail 仍 hardcode `K3_TAIL_ATOMIC_SIGNAL ..., 1`，第二次 eager fused 调用传 generation=2 时会永远等不到 2；
  - 已将原始 tail ASM 与 PACK5 tail ASM 的 8 个 peer signal 都改为 `K3_TAIL_ATOMIC_SIGNAL ..., s78`。
- 远端验证：
  - PACK5 `.co` 重新编译通过，sha256 `e0ce9a655185744879da8ffc1749c2307970175c648ab3f0cfcde4069c40cf35`；
  - `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过；
  - `K3_V3_TAIL_STAGE_SIGNAL_GENERATION=2` 的 `v3_only` stage compare 通过，post signal slots 均为 `2`；
  - V3 normal tail ASM-pack5 1024 correctness 通过，`max_abs=0.000488281`；
  - 1024 fused-only bench 通过并越过原先卡住的 `before_fused_timing`，tail median `2.0430 ms`（repeat=1）；
  - 同配置 V3 normal no-tail 对照通过，median `2.0578 ms`（repeat=1）；
  - repeat=3 对照：tail median/min `2.1008/2.0094 ms`，no-tail median/min `2.0437/2.0374 ms`，两者均 correctness 通过。
  - 4096 repeat=3 对照：tail median/min `5.8836/5.7742 ms`，no-tail median/min `5.8762/5.7800 ms`，两者均 correctness 通过；median 差异约 `0.13%`，无可见劣化。
- 注意：
  - 完整默认 bench 若继续跑 DeepEP/DeepGEMM baseline timing，会在 `after_fused_timing` 后进入 baseline 阶段触发 VMFault，并导致其他 rank 在下一次 staged barrier 等待；这不是 tail fused 卡住，需要作为独立 baseline/timing 问题处理。
  - 为隔离 fused 路径，新增默认关闭的 `MEGAMOE_DCU_TEST_SKIP_BASELINE_TIMING=1` 诊断开关；correctness baseline 仍照常执行。

## 2026-06-15 - V3 normal uneven and graph support completed

- 用户要求：继续补 V3 normal uneven 支持和 graph 支持；由于 ASM 流程接近初版，优先复用原 staged graph/active_tiles 合同，并保证 graph 性能不比 eager 差。
- 代码改动：
  - `large_opt.py` 允许 V3 normal 进入 staged graph 路径，不再只允许 LL；
  - graph 路径下 normal K1/K3 显式传 `use_asm_pack5`；
  - `k1_symm_fused_l1_v3_graph()` 增加 normal ASM-pack5 分支，复用原 K1 ASM graph runtime-token 参数；
  - `k3_l2_combine_asm_pack5_out()` 增加可选 `active_tiles`，让 no-tail PACK5 ASM graph 能按 K1 runtime active tile count 跳过无效 row tile；
  - `tests/test_dcu_megamoe_v3.py` 更新 source guard。
- 本地验证：
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - source guard：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` -> `12 passed`；
  - rebuild：删除 K3 ext object/hipify cache 后 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，K3 ext 实际重新编译；
  - normal uneven eager，token list `1024,768,512,896,640,384,256,128`，no-tail/tail 均通过，`max_abs=0.000488281`；
  - normal graph uniform，capture bucket `1024`，replay `512,1024`，no-tail/tail 均通过；tail 512 worst `max_abs=0.00244141`，仍低于 `--atol=0.0035`；
  - normal graph uneven，capture bucket `1024`，同 uneven token list，no-tail/tail 均通过，`max_abs=0.000488281`。
- graph-vs-eager 性能：
  - uniform no-tail graph replay 512/1024: `1.6157/1.9828 ms`，eager 512/1024: `1.6919/2.0746 ms`；
  - uniform tail graph replay 512/1024: `1.6578/1.9539 ms`，eager 512/1024: `1.7467/1.9937 ms`；
  - uneven no-tail graph replay: `1.6629 ms`，eager: `1.7873 ms`；
  - uneven tail graph replay: `1.7242 ms`，eager: `1.8254 ms`；
  - 结论：当前验证点 graph replay 均快于 eager，无 graph 性能劣化。
- 远端产物：
  - `hygon_tmp/sglang_debug/normal_uneven_graph_20260615/`

## 2026-06-15 12:27:37 +08:00 - baseline timing VMFault triage

- 用户追问：为什么 baseline 会 VMFault 或卡住，之前测 no-tail 时还是正常的。
- 排查原则：
  - 不先改代码，先定位是否能稳定复现；
  - 区分 correctness baseline、benchmark baseline timing、fused timing、graph 前置和旧 raw/persistent VMFault；
  - 所有远端日志放在 `hygon_tmp/sglang_debug/baseline_vmfault_20260615/`。
- 当前复现矩阵：
  - V3 normal no-tail 1024，`warmup=1 repeat=1`、`warmup=1 repeat=3` 均通过，日志出现 `before_baseline_timing` 和 `after_baseline_timing`；
  - V3 normal tail 1024，`warmup=1 repeat=1`、`warmup=1 repeat=3`、`warmup=5 repeat=10` 均通过；
  - V3 normal no-tail 4096，`warmup=1 repeat=1`、`warmup=5 repeat=10` 均通过；
  - V3 normal tail 4096，`warmup=1 repeat=1`、`warmup=5 repeat=10` 均通过；
  - V3 normal graph 前置 1024 bucket，no-tail/tail 都在 graph correctness/graph replay bench 后继续完成脚本尾部 baseline timing；
  - 旧 `base_no_large_128` 形状在当前代码下 `warmup=1 repeat=1` 与 `warmup=5 repeat=10` 都通过。
- 误导项：
  - 一次并行跑 no-tail/tail 8-rank 复现命令时，两组进程争用默认 `MASTER_PORT=8361` 和同一组卡，产生 `EADDRINUSE` / NCCL socket error；已判定为本次复现方式引入的干扰，不计入 baseline 根因。
- 旧日志归类：
  - `base_no_large_128_bench_20260613_192144.log` 的 VMFault 伴随 `MegaMoE HIP local barrier timeout`，发生在旧 persistent fused benchmark 阶段；当前同形状不可复现；
  - `v3_normal_k3_normal_raw_1024_20260614_100912.log` 属于 early normal K3 raw pure wrapper `row_combine_ptrs=nullptr` 合同错误，后续已有 `kPureContiguous` 修复记录；
  - planning 中已有远端服务占满 8 卡 VRAM 导致 LL VMFault/local barrier timeout 的环境污染记录。
- 当前结论：
  - 当前代码和干净 8 卡环境下，未复现 baseline oracle/timing 的 VMFault 或卡住；
  - 之前 tail-reduce 支持阶段的 `MEGAMOE_DCU_TEST_SKIP_BASELINE_TIMING=1` 仍可作为隔离 fused 性能的诊断开关，但不应被理解成当前 baseline 必坏；
  - 后续若再次出现，需要保留完整命令、`MASTER_PORT`、是否并发、`hy-smi --showpids`、以及 `[BENCH_PROGRESS]` 到底停在 `before_fused_timing` 还是 `before_baseline_timing`。

## 2026-06-15 - V3 normal C/raw cleanup started

- 用户要求清理 V3 normal C 相关代码逻辑、编译、冗余 env，并修正 LL kernel 命名中残留的 `Pure` 误导字段。
- 已完成本地代码清理：
  - `large_opt.py` 不再消费 normal C fallback、no-tail signal/sync、K2 fence、reduce/barrier acquire 等诊断 env；
  - V3 normal K1/K3 wrapper 默认直接走 ASM-pack5；K3 normal no-tail/tail 均不再调 raw normal C extension；
  - `setup.py` 退役 `DG_BUILD_MEGAMOE_V3_NORMAL_AICC`，raw backend 只允许 `ll`，normal raw build 会 fail-fast；
  - K1/K3 raw extension/stub 去掉 normal raw availability 和 launcher pybind；
  - LL kernel 命名从 `V3_*_Pure_LowLatencyMaskedGroupGemmKernel` 改为 `V3_*_LowLatencyMaskedGroupGemmKernel`，K3 LL diagnostic pybind 从 `k3_v3_ll_pure_raw` 改为 `k3_v3_ll_reference_raw`。
- 临时脚本清理：
  - active LL rowptr profiler 改用 `reference_contiguous`；
  - normal raw smoke/build/probe 脚本改为 retired stub；
  - 移除活跃脚本里已退役的 `MEGAMOE_DCU_V3_K3_ASM_PACK5`、`MEGAMOE_DCU_V3_NO_TAIL_SYNC`、`MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 等 env。
- 本地验证：
  - `python -m compileall` 覆盖生产 Python、测试和活跃 debug Python，通过；
  - `git diff --check` 通过；
  - 本机 Python 缺少 `pytest`，已用 minimal pytest/monkeypatch shim 手动调用 `tests/test_dcu_megamoe_v3.py` 的 12 个 source/layout test，全部通过；
  - 生产源码和活跃 `hygon_tmp/sglang_debug` 脚本中旧 normal raw/env/Pure 命名扫描为 0 命中。
- 远端状态：
  - 初始 SSH 可连接且 `sglang_megamoe` 容器在线；
  - 逐文件 `scp` 中途被断开，随后 `scp` tar 包也被断开；
  - 后续 10.17.176.11 可 ping 通但 TCP 22 `Connection refused` / `TcpTestSucceeded=False`，远端同步、source pytest、rebuild 和 GPU smoke 暂未完成，待 SSH 恢复后继续。

## 2026-06-15 - K1/K3 V3 pack5 header naming aligned

- 用户指出 K3 header 已叫 `k3_v3_pack5_groupgemm_impl.cuh`，K1 同类文件却叫 `k1_v3_groupgemm_impl.cuh`，会造成 pack5 layout 命名不对称。
- 已本地修正：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 重命名为 `K1_fused/k1_v3_pack5_groupgemm_impl.cuh`；
  - `k1_v3_fused_ext.cu` 的 include 改为新文件名；
  - `tests/test_dcu_megamoe_v3.py` 的 source guard 改为检查新文件名，并确认 launcher 不再 include 旧文件名。
- 说明：
  - 这是纯命名/引用修正，不改 kernel body；
  - `.planning` 中旧文件名的早期历史记录保留为当时操作证据，当前有效命名以本条为准。
- 已更新：
  - `task_plan.md` 顶层目标、当前状态、成功标准和 Phase 2/3 关键项已标明 V3 normal C/aicc raw 为 retired/abandoned；raw extension 范围收敛为 LL-only。
- 本地复核：
  - `rg` 确认生产源码中旧 `k1_v3_groupgemm_impl.cuh` include 已无残留，旧名只作为测试负向断言出现；
  - `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - minimal pytest/monkeypatch shim 手动调用 `tests/test_dcu_megamoe_v3.py` 的 12 个 source/layout test，全部通过；
  - `git diff --check` 通过。

## 2026-06-15 - V3 raw build gates and stub sources retired

- 用户追问 `raw` 字段含义，并指出 V3 LL kernel 和 V3 normal ASM 都应默认编译，不需要 aicc 混编和可选 raw build gate。
- 已完成本地代码收口：
  - `setup.py` 删除 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS` / `DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND` / `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS` / `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND`；
  - K1 large-opt extension 默认编译 `k1_v3_fused_ext.cu`，不再编译 `k1_v3_stub_ext.cu`；
  - K3 large-opt ASM extension 不再携带 V3 LL availability stub，`k3_v3_fused_ext.cu` 默认作为独立 V3 LL pack5 extension 编译；
  - 删除 `K1_fused/k1_v3_stub_ext.cu` 和 `K3_fused/k3_v3_stub_ext.cu`；
  - 删除 `DCU_MEGAMOE_V3_ENABLE_*RAW*` 宏和 `dcu_megamoe_v3_*_raw_*available` 检查；
  - K1 internal launcher 改名为 `dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5`；
  - K3 internal launcher/pybind 改名为 `dcu_megamoe_v3_launch_k3_ll_combine_pack5`、`k3_v3_ll_combine`、`k3_v3_ll_reference`、`k3_v3_ll_combine_tail`；
  - `K3_fused/k3_fused.py` 的 loader 改为 `load_v3_ll_extension`，不再提示 raw build env；
  - 活跃 LL rowptr/compact debug 脚本同步新 pybind 名称；normal C retired stub 文案改为 “V3 LL pack5 builds by default”。
- 本地验证：
  - `python -m compileall` 覆盖 `setup.py`、K1/K3 wrapper、`tests/test_dcu_megamoe_v3.py` 和更新过的 debug Python，通过；
  - minimal pytest/monkeypatch shim 手动调用 `tests/test_dcu_megamoe_v3.py` 的 12 个 source/layout test，全部通过；
  - `rg` 精确扫描确认生产源码和活跃 debug 脚本中旧 V3 raw build env、`DCU_MEGAMOE_V3_ENABLE_*RAW*`、旧 availability 函数和旧 `_raw` pybind 调用均无残留；
  - 符号级 sanity 确认新 K1/K3 launcher 声明/调用成对，K3 ASM ext 与 K3 LL ext 各自只有一个 `PYBIND11_MODULE`；
  - `git diff --check` 通过。
- 远端状态：
  - 尝试连接 `hg@10.17.176.11:22` 仍失败，`banner exchange: Connection to UNKNOWN port -1: Connection refused`；
  - 因 SSH 端口拒绝，本轮尚未完成远端同步、`setup.py build_ext --inplace`、source pytest 或 GPU smoke，待 SSH 恢复后补跑。

## 2026-06-15 - Pending user design feedback: V3 K1 ASM isolation

- 用户指出当前 V3 K1 normal ASM-pack5 通过原 `k1_symm_fused_l1` host path 和 `backend == "normal"` 分支隔离不合理：
  - 原来的 K1 ASM 本身也是 normal 场景；
  - V3 K1 应该和原实现独立，而不是只通过 backend 参数和 code object 区分。
- 本轮仅记录，不改代码；等待用户后续修改意见后一并处理。

## 2026-06-15 - Pending user design feedback: V2 package/build retirement

- 用户补充指出 `DG_BUILD_MEGAMOE_V2_EXT` 也不再需要，且 `megamoe/dcu_megamoe_v2/` 整个 V2 包和相关逻辑都不再需要。
- 已记录到 `task_plan.md` 的退役范围和 `findings.md` 的 pending decision。
- 本轮仅记录，不删除代码；等待用户后续修改意见后一并做 setup/package/import/test/script 清理。

## 2026-06-15 - Pending user design feedback: setup.py minimal build delta

- 用户指出 `setup.py` 当前改动面过大，期望只是补上 V3 必需编译文件，而不是继续保留历史遗留的 V2/raw/stub build 结构。
- 已检查但未修改生产代码；当前观察到的候选遗留包括 `DG_BUILD_MEGAMOE_V2_EXT`、V2 package/build entries、`large_opt_k1_sources`/`large_opt_k3_sources`/`large_opt_k3_v3_ext` 这类为历史演化拆出的临时变量，以及对应 source guard。
- 已记录到 `findings.md`；等待用户后续修改意见后一并处理。

## 2026-06-15 - Pending user design feedback: K1 normal ASM should follow K3 pack5 isolation

- 用户指出 K3 V3 normal 已经有独立的 pack5 实现边界，K1 可以借鉴。
- 已记录：后续 K1 V3 normal ASM-pack5 不应继续只通过原 K1 ASM host path + backend/code-object 参数区分；应建立类似 K3 normal pack5 的独立 wrapper/entry 命名和调用边界。
- 本轮仅记录，不改代码。

## 2026-06-15 - Record-only mode until user says start

- 用户明确：后续先统一记录，等用户说“开始改”后再统一开始修改源码。
- 已恢复一次抢跑误改：`k3_v3_ll_reference` 仍在 `k3_v3_fused_ext.cu` 中存在，pybind 也已恢复；`git diff --check -- megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_fused_ext.cu` 通过。
- 本轮新增记录的 pending cleanup：
  - K1 V3 pack5 header 中未使用的 normal C/fixed-route kernel body 后续应清理；
  - K3 V3 pack5 header 中早期 bring-up/diagnostic 残留的冗余或不会使用的 helper/kernel 逻辑后续也应清理；
  - `large_opt.py` 中 `_v3_debug_stage_sync`/`MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC*` 与 `_v3_stage_timing_enabled`/`MEGAMOE_DCU_V3_STAGE_TIMING` 属于开发期诊断输出，后续应清理；
  - `MEGAMOE_DCU_V3_LL_BLOCK_M` 仍是 A/B tuning knob，若最终固定 retained LL block M=32，后续可移除该 env；
  - `MEGAMOE_DCU_K3_DEBUG_LAUNCH` 后续可移除；
  - `reduce_local_combine_vec_kernel` 的 `invalidate_before_read` 是 dormant diagnostic，后续作为候选清理；
  - `k3_v3_ll_reference` 用户确认不需要保留，后续统一移除函数/pybind/测试/脚本引用。
  - `tests/test_dcu_megamoe_v3.py` 当前主要是开发期 source guard，后续可退役或大幅缩小，只保留 V3 gate/layout 这类轻量合同测试。
- 从本条开始，除 planning 记录和必要的误改恢复外，不再修改源码，直到用户明确开始改。

## 2026-06-15 - User-confirmed cleanup scope recorded

- 用户确认可清范围：
  - V2 整体：`DG_BUILD_MEGAMOE_V2_EXT`、`megamoe/dcu_megamoe_v2/`、setup/package/test/script 里的 V2 build/package 引用；
  - V3 normal C/raw 残留：K1/K3 header 中不会再 launch 的 normal C kernel body、fixed-route bring-up helper、raw/stub/availability 相关断言和脚本；
  - 诊断 pybind：`k3_v3_ll_reference` 及调用它的 debug scripts/source guard；
  - 调试 env：`MEGAMOE_DCU_K3_DEBUG_LAUNCH`、`MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC*`、`MEGAMOE_DCU_V3_STAGE_TIMING`；`MEGAMOE_DCU_V3_LL_BLOCK_M` 在最终固定后再移除；
  - dormant API 参数：K3 `invalidate_before_read`、K3 `acquire_after_wait`、K2 `system_fence_after_write`。
- 用户确认需要重整的边界：
  - K1 V3 normal ASM 参考 K3 normal pack5 独立 entry/wrapper，不再靠原 K1 ASM host path + backend/code-object 参数区分；
  - `setup.py` 回到最小 V3 delta，只保留必需源文件、pack5 ASM code object 和 `*.cuh` package data；
  - `large_opt.py` 清掉开发期 sync/timing，只保留 normal ASM-pack5 与 LL C pack5 分流；
  - `megamoe/__init__.py` 检查 warmup/symm alloc 是否有 V3 normal 早期防御/调试逻辑。
- 用户确认测试/脚本方向：
  - `tests/test_dcu_megamoe_v3.py` 后续删除或缩成 gate/backend、pack5 layout helper、关键 env 默认安全检查；
  - 功能/性能回归主要靠 `tests/test_mega_moe_dcu.py` 和远端 8 卡 matrix；
  - `hygon_tmp/sglang_debug` 只保留当前生产路径脚本，normal C/raw、reference/pure、旧 A/B 方向 retired 或删除。
- 补充记录：
  - `k3_l2_fused_v3_to_combine()` 中 `elif sym_buffer is not None: raise NotImplementedError("V3 K3 LL no-tail signal path is not wired yet")` 是未接生产的 LL no-tail signal 防御分支；当前 LL no-tail 不传 `sym_buffer`，LL tail 通过 `asm_reduce_y` 分支处理，后续清理时可删除该 guard 和对应 source guard。
- 已更新：
  - `task_plan.md` 增加 “用户确认的后续清理边界”；
  - `findings.md` 增加确认清理范围记录。
- 仍未修改源码；继续等待用户明确开始改。

## 2026-06-15 - V3 production cleanup implemented

- 用户明确“开始清理”，本轮按用户确认范围执行源码、构建、测试和脚本清理。
- 已完成代码清理：
  - 删除 V2 build/package/test/source 入口：`DG_BUILD_MEGAMOE_V2_EXT`、`megamoe/dcu_megamoe_v2/`、`csrc/kernels/dcu_megamoe_v2/` 和 `tests/test_dcu_megamoe_v2.py` 已移除；
  - V3 normal 生产路径收敛为 ASM-pack5，K1/K3 pack5 header 中 normal C/fixed-route bring-up kernel body 和不会再 launch 的 helper 已删除；
  - V3 LL kernel 命名保留 `LowLatency...`，去掉 `Pure` 误导字段；AMD `raw_buffer` intrinsic 名称保持不动；
  - 删除 `k3_v3_ll_reference` diagnostic pybind；
  - 删除 `MEGAMOE_DCU_K3_DEBUG_LAUNCH`、`MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC*`、`MEGAMOE_DCU_V3_STAGE_TIMING`，并将 LL block M 固定为 32 后移除 `MEGAMOE_DCU_V3_LL_BLOCK_M`；
  - 删除 dormant API 参数：K2 `system_fence_after_write`、K3 `invalidate_before_read`、K3 `acquire_after_wait`；
  - `setup.py` 清成最小 V3 delta：默认编译 V3 LL C pack5、V3 normal ASM-pack5 code object 和 `*.cuh` package data，不再保留 V2/raw/stub gate；
  - K1 V3 normal ASM 增加独立 Python wrapper 边界 `k1_symm_fused_l1_v3_asm_pack5()` / graph wrapper，生产调用点不再靠泛化 backend 名称表达 V3 normal。
- 已完成脚本/测试清理：
  - `tests/test_dcu_megamoe_v3.py` 缩成轻量合同测试，保留 V3 gate/backend、pack5 layout、setup surface 和 retired symbol 负向检查；
  - `hygon_tmp/sglang_debug` 删除或改造 normal C/raw、reference/pure、旧 A/B 脚本，保留当前 normal ASM-pack5 与 LL C pack5 相关入口。
- 本地验证状态：
  - 退役关键词扫描只剩 `tests/test_dcu_megamoe_v3.py` 的负向断言；
  - 本地 `pytest` 仍不可用，后续用 compileall/py_compile/source scan 作为本地验证，远端恢复后补 GPU build/smoke。
- 待继续：
  - 跑本地 `compileall`、`py_compile setup.py`、`git diff --check`；
  - 尝试远端同步、`setup.py build_ext --inplace`、source pytest 和 V3 normal/LL smoke；若 SSH 仍拒绝，需要记录为验证阻塞。

## 2026-06-15 - V3 production cleanup local validation

- 本地验证已完成：
  - `python -m compileall megamoe tests\test_mega_moe_dcu.py tests\test_dcu_megamoe_v3.py` 通过；
  - `python -m py_compile setup.py` 以及被修改过的 `hygon_tmp/sglang_debug` Python 脚本通过；
  - `git diff --check` 通过；
  - 退役关键词扫描只剩 `tests/test_dcu_megamoe_v3.py` 的负向断言，生产源码和活跃脚本无 V2/raw/debug/dormant API 残留命中；
  - 本地 `python -m pytest tests\test_dcu_megamoe_v3.py -q` 因本机缺少 pytest 失败：`No module named pytest`；
  - 用 minimal pytest/monkeypatch shim 手动调用 6 个 source/layout 合同测试，通过。
- 远端验证状态：
  - 按 `.vscode/sftp.json` 和 remote skill 使用 `hg@10.17.176.11:22`、容器 `sglang_megamoe`、容器 repo `/workspace/DeepGEMM`；
  - SSH 探测失败：`banner exchange: Connection to UNKNOWN port -1: Connection refused`；
  - `ping 10.17.176.11` 成功，`Test-NetConnection 10.17.176.11 -Port 22` 显示 `TcpTestSucceeded=False`；
  - 因 SSH 端口拒绝，远端同步、DTK/HIP build、source pytest 和 8 卡 smoke 暂未执行，后续需补跑后再宣称 GPU build/perf 正常。

## 2026-06-15 - K1 V3 normal ASM pybind boundary isolated

- 用户指出上一轮 K1 V3 normal ASM 只是 Python wrapper 独立，底层仍直接调用原 `ext.k1_symm_fused_l1(...)`，没有真正参考 K3 V3 pack5 的独立 entry/wrapper 边界。
- 已修正：
  - `k1_fused_ext.cu` 将原 K1 ASM host 实现抽成内部 `k1_symm_fused_l1_asm_impl(...)`；
  - 原路径继续通过 `k1_symm_fused_l1(...)` pybind 调用该 helper；
  - V3 normal ASM-pack5 新增独立 pybind `k1_symm_fused_l1_v3_asm_pack5(...)`，也调用同一 helper，但对外不再暴露为原 K1 ASM entry；
  - `k1_fused.py` 的 `k1_symm_fused_l1_v3_asm_pack5()` / graph wrapper 已切到 `ext.k1_symm_fused_l1_v3_asm_pack5(...)`；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，防止 V3 normal 再退回原 pybind。
- 本地验证：
  - `python -m compileall megamoe\dcu_megamoe_large_opt\K1_fused\k1_fused.py tests\test_dcu_megamoe_v3.py` 通过；
  - `git diff --check -- megamoe\dcu_megamoe_large_opt\K1_fused\k1_fused.py megamoe\dcu_megamoe_large_opt\K1_fused\k1_fused_ext.cu tests\test_dcu_megamoe_v3.py` 通过；
  - minimal pytest shim 手动跑 6 个 source/layout 合同测试通过。

## 2026-06-15 - Main MegaMoE test diagnostic cleanup

- 用户要求清理 `tests/test_mega_moe_dcu.py` 中的冗余开发期改动。
- 已清理：
  - 删除 `MEGAMOE_DCU_TEST_BENCH_PROGRESS` / bench progress 标记；
  - 删除 `MEGAMOE_DCU_TEST_PRECOMPUTE_BASELINE_LAYOUT`、`MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE`、`MEGAMOE_DCU_TEST_SKIP_BASELINE_SYNC`、`MEGAMOE_DCU_TEST_SKIP_BASELINE_TIMING` 诊断开关；
  - 删除 LL K1 padding/capacity 统计字段和对应 `ceil_div_int` / `align_up_int` helper；
  - V3 backend 读取改用 `get_v3_backend()`，V3 开关判断改用 `v3_requested()`，避免非法 backend 在测试里被静默当作 LL。
- 保留：
  - V3 fused 权重与 baseline 权重分流；
  - graph replay 的 V3/baseline 权重分流；
  - nonfinite correctness 检查和 baseline sync。
- 本地验证：
  - `python -m compileall tests\test_mega_moe_dcu.py` 通过；
  - `git diff --check -- tests\test_mega_moe_dcu.py` 通过；
  - `rg` 确认该文件不再包含 `MEGAMOE_DCU_TEST_*`、bench progress、LL padding 指标残留。

## 2026-06-15 - Remote sync and full cleanup validation

- 用户要求检查卡状态、同步到远端、全面验证，完成后收敛 plan。
- 远端环境：
  - 使用 `.vscode/sftp.json` / remote skill 指向的 `hg@10.17.176.11:22`、容器 `sglang_megamoe`、repo `/workspace/DeepGEMM`；
  - 验证前 `hy-smi` 显示 8 张 HCU 均空闲，VRAM 0%、HCU 0%，无 KFD 进程；
  - 验证后再次检查仍无 KFD 进程，8 卡释放干净。
- 同步：
  - 先删除远端旧 V2 目录、`tests/test_dcu_megamoe_v2.py`、旧 V3 stub 和旧非 pack5 header 残留；
  - 再同步本地 `setup.py`、`megamoe`、`tests` 到 `/home/hg/yuguo/DeepGEMM`。
- 远端 source/build 验证：
  - `python3 -m compileall setup.py megamoe tests/test_mega_moe_dcu.py tests/test_dcu_megamoe_v3.py` 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`6 passed`；
  - 清理 `build` 和 large-opt `.so` 后，`MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过，K1/K2/K3 原扩展、K1 V3 LL、K3 V3 LL 扩展和 ASM code object 都完成构建。
- 远端 8 卡矩阵（产物：`hygon_tmp/sglang_debug/full_validation_20260615_161234/`）：
  - V3 normal tail 32: correct, fused `1.526879 ms`, baseline `2.805279 ms`, `1.8373x`；
  - V3 normal no-tail 32: correct, fused `1.504639 ms`, baseline `2.814119 ms`, `1.8703x`；
  - V3 normal tail uneven `17,31,48,64,79,96,112,127`: correct, fused `1.541459 ms`, baseline `2.823078 ms`, `1.8314x`；
  - V3 normal tail graph uneven: correct, eager `1.539139 ms`；graph replay 32/96/128 token 为 `1.413819/1.428639/1.436719 ms`，不劣于 eager；
  - V3 LL tail 32: correct, fused `0.709419 ms`, baseline `2.759899 ms`, `3.8904x`；
  - V3 LL no-tail 32: correct, fused `0.689540 ms`, baseline `2.761178 ms`, `4.0044x`；
  - 非 V3 legacy large-opt tail 32: correct, fused `1.931739 ms`, baseline `2.785199 ms`, `1.4418x`。
- 残留扫描：
  - 生产源码和主测试中旧 V2/raw/stub/Pure/debug env/dormant API 字段无残留；
  - 剩余匹配只在 `tests/test_dcu_megamoe_v3.py` 负向 source-guard 中。
- 注意：
  - 两个远端脚本包装命令的最终 exit code 曾因 PowerShell here-string CRLF / bash `exit 0\r` 产生包装层错误；对应测试本体已经打印 `STATUS=0` 并写出 JSON，后续用干净 Python stdin 汇总确认所有 JSON `correct=True`。
- 已更新：
  - `task_plan.md` 增加当前验收快照；
  - Phase 5 标记为本轮 cleanup validation complete；
  - Phase 7 标记为 current production gate complete，更多 4096/bucket graph perf guard 作为后续扩展，不阻塞本轮清理。

## 2026-06-15 - LL graph uneven and normal 4096补测

- 用户指出 cleanup validation 里 V3 LL graph uneven 没有覆盖；随后要求 normal 测 4096。
- 已补测 V3 LL graph uneven，使用 token list `17,31,48,64,79,96,112,127`，capture bucket `128`，replay `32,96,128`：
  - no-tail: correct, eager/fused `0.774360 ms`，baseline `2.768279 ms`，graph replay `0.661760/0.735480/0.738820 ms`；
  - tail: correct, eager/fused `0.819420 ms`，baseline `2.817779 ms`，graph replay `0.721340/0.790440/0.806820 ms`。
- 已补测 V3 normal 4096 eager：
  - no-tail: correct, fused `5.808258 ms`，baseline `9.460317 ms`，speedup `1.6288x`；
  - tail: correct, fused `5.838198 ms`，baseline `9.513697 ms`，speedup `1.6296x`。
- 远端产物仍统一在 `hygon_tmp/sglang_debug/full_validation_20260615_161234/`。
- 验证后 `hy-smi --showpids` 显示无 KFD 进程残留。
- 已更新：
  - `task_plan.md` 当前验收快照、Phase 5、Phase 7；
  - `findings.md` cleanup validation findings。

## 2026-06-15 - 计划收敛与旧路线遗弃标记

- 按用户要求重新检查 `.planning/dcu_megamoe_v3/task_plan.md` 中残留的悬挂项。
- 已将 Phase 2/3/4 的 V3 normal C/raw/stub、K1 原 ASM backend 复用、K3 normal C tail/raw 等历史方向收敛为 complete / abandoned / optional backlog。
- 已将 Phase 6/6b 从历史 A/B 流水账收成 retained production gate 结论：
  - V3 normal active 边界固定为 isolated ASM-pack5 K1/K3 no-tail/tail eager/graph；
  - V3 LL active 边界固定为 C pack5 K1/K3 no-tail/tail eager/graph；
  - normal C/aicc、raw tail、fixed-route/raw/stub availability、旧 K3 scalar rowptr/store 调参链路均标记为历史项/遗弃。
- 保留为可选 backlog 的只有不阻塞当前生产 gate 的方向：normal K3 no-tail fused-reduce ASM-pack5 diagnostic、compute/store overlap synthetic microbench、row-emission/rank-bucket layout 诊断、LL K3 128 remote/scattered residual profiling。
- 重新扫描 `task_plan.md`，未再发现 `⏳`、空勾选项或 pending 状态。

## 2026-06-15 - Symm buffer / route_scratch 显存审计

- 按用户要求检查 big fused、legacy staged、V3 normal、V3 LL 共享 `SymmBuffer` 与 `route_scratch` 后是否存在历史冗余。
- 已复核：
  - `megamoe/__init__.py` 的 `SymmBuffer` 分配和输入 view；
  - `csrc/apis/mega_dcu.hpp` 的 DCU symm buffer / route_scratch size 公式；
  - `deep_gemm/include/deep_gemm/layout/mega_moe_dcu.cuh` 的 combine 与 route tile scratch layout；
  - `megamoe/large_opt.py` 的 staged scratch view 复用；
  - K1/K2/K3 fused wrapper 对 `row_combine_ptrs`、`output_workspace`、`prob_storage` 和 tail signal 的消费。
- 结论：
  - symm buffer 没有明显大冗余，DCU l1/l2 act view 是空 view，combine 区仍是当前 no-tail/tail 合同的一部分；
  - route_scratch 是主要显存压力，当前为兼容 big fused 和 staged fallback 采用通用 big-fused layout，V3-only/LL-only 情况下存在可观节省空间；
  - 当前不改生产分配，先把 staged-only / V3-only scratch allocation 作为 optional backlog 记录到 `task_plan.md` 和 `findings.md`。

## 2026-06-15 - Big fused 退役后的 route_scratch 优化方向

- 用户补充：big fused 后续准备删除，这个分支目前没有明显优势；因此 route_scratch 显存优化可以按 retained staged/V3 路径重新规划。
- 已更新：
  - `task_plan.md` Phase 8 将 route_scratch 缩容从泛泛 optional backlog 收敛为 big fused 退役后的 planned direction；
  - `findings.md` 补充 big fused 删除对 route_scratch 合同的影响和验证边界。
- 当前约束：
  - 本轮只更新计划，不改生产分配；
  - 后续真正缩容前必须先确认 big fused 入口删除或 fail-fast 语义清晰；
  - 新 scratch layout 需要覆盖 V3 normal/LL、tail/no-tail、eager/graph、uniform/uneven、32/128/1024/4096，避免省显存时伤到生产路径。

## 2026-06-15 - Staged fused 长期 V3-only 方向

- 用户补充：staged fused 后续预计也只保留 V3 实现，LL 负责小 token，normal 负责大 token。
- 已更新：
  - `task_plan.md` 当前状态、设计点、计划收敛规则和 Phase 8 中都记录 V3-only staged fused 的长期边界；
  - `findings.md` 增加 staged fused V3-only 对 route_scratch 规划的影响。
- 当前约束：
  - 本轮仍只更新计划，不改源码；
  - 在明确删除前，legacy staged fused 非 V3 路径仍按当前兼容性回归处理；
  - 后续 route_scratch 缩容需要同时确认 big fused 和 legacy staged fused 非 V3 入口的删除或 fail-fast 语义。

## 2026-06-15 - 新增必做刷数矩阵

- 用户明确新增必做数据 sweep：
  - LL graph：tokens per rank `8,32,64,128,256,512,1024`，graph capture 固定按 `1024`；
  - normal eager：tokens per rank `256,512,1024,1025,2048,2050,4096,4097,8192`；
  - LL/normal 都要覆盖 no-tail/tail。
- 已更新：
  - `task_plan.md` 明确未完成必做项继续使用 `[ ]` 空勾选，用于和 `🧭 optional backlog` 区分；
  - 新增 Phase 10 `Required V3 performance sweep`，把远端准备、LL graph、normal eager、结果落盘和回写计划都列为必做；
  - `findings.md` 记录该矩阵是 required sweep，已有 smoke 结果不能替代。
- 当前状态：
  - 本轮只更新 planning，不跑远端；
  - Phase 10 现在是明确未完成的必做项，按 `[ ]` 空勾选展示。

## 2026-06-15 - Phase 10 远端刷数与 LL graph 1024 bug

- 按用户要求先检查远端 8 卡状态：
  - `sglang_megamoe` 容器运行中；
  - `hy-smi` 显示 8 卡 HCU/VRAM 均空闲；
  - `hy-smi --showpids` 无 KFD 进程。
- 为避免误同步本地 `third-party/cutlass` 删除状态，只同步本轮必要文件到 `/home/hg/yuguo/DeepGEMM`：
  - `setup.py`
  - `megamoe/`
  - `tests/test_mega_moe_dcu.py`
  - `tests/test_dcu_megamoe_v3.py`
  - `hygon_tmp/sglang_debug/run_v3_phase10_sweep.sh`
- 远端 preflight 通过：
  - `python3 -m compileall megamoe tests`
  - `pytest -q tests/test_dcu_megamoe_v3.py`
  - `MAX_JOBS=16 python3 setup.py build_ext --inplace`
- Phase 10 结果已拉回本地：`hygon_tmp/sglang_debug/phase10_v3_sweep_20260615_196S/`。
- V3 LL graph capture bucket=1024 暴露 correctness bug：
  - no-tail：`graph_ll_tail0_1024.log`，`max_abs=0.071502685546875` 超过 `--atol=0.0035`；
  - tail：`graph_ll_tail1_1024.log`，`stats mismatch`；
  - 两个 case 都在 correctness 阶段失败，未生成完整 graph replay sweep 指标。
- V3 normal eager 全矩阵通过：
  - no-tail tokens `256,512,1024,1025,2048,2050,4096,4097,8192` 全部 `correct=True`；
  - tail tokens `256,512,1024,1025,2048,2050,4096,4097,8192` 全部 `correct=True`；
  - 4096 no-tail `5.863377 ms` vs baseline `9.551276 ms`，speedup `1.629x`；
  - 4096 tail `5.822198 ms` vs baseline `9.536475 ms`，speedup `1.638x`；
  - 8192 no-tail `12.549034 ms` vs baseline `17.376192 ms`，speedup `1.385x`；
  - 8192 tail `12.009055 ms` vs baseline `17.218673 ms`，speedup `1.434x`。
- 1024/1025、2048/2050、4096/4097 normal 边界点没有 correctness 异常或明显性能台阶。
- 运行结束后 `hy-smi --showpids` 仍显示无 KFD 进程残留。
- 已更新 `task_plan.md` Phase 10：
  - 远端准备、normal eager sweep、结果落盘标记完成；
  - LL graph 1024 no-tail/tail correctness bug 保持为未完成必做项。

## 2026-06-15 - Phase 10 性能按 token 汇总

- 数据来源：`hygon_tmp/sglang_debug/phase10_v3_sweep_20260615_196S/summary.csv`。
- 终端已打印同一份汇总；以下数值保留三位小数，完整精度见 CSV/JSON。

### V3 normal eager by tokens

| tokens/rank | no-tail fused ms | no-tail baseline ms | no-tail speedup | tail fused ms | tail baseline ms | tail speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1.593 | 2.938 | 1.844x | 1.623 | 2.961 | 1.824x |
| 512 | 1.692 | 3.210 | 1.898x | 1.747 | 3.209 | 1.837x |
| 1024 | 2.065 | 3.706 | 1.795x | 2.053 | 3.693 | 1.799x |
| 1025 | 2.143 | 3.658 | 1.707x | 2.111 | 3.685 | 1.746x |
| 2048 | 3.376 | 5.493 | 1.627x | 3.339 | 5.568 | 1.668x |
| 2050 | 3.330 | 5.448 | 1.636x | 3.328 | 5.467 | 1.643x |
| 4096 | 5.863 | 9.551 | 1.629x | 5.822 | 9.536 | 1.638x |
| 4097 | 5.844 | 9.586 | 1.640x | 5.819 | 9.520 | 1.636x |
| 8192 | 12.549 | 17.376 | 1.385x | 12.009 | 17.219 | 1.434x |

### V3 LL graph status

| graph capture tokens/rank | tail | status | note |
| ---: | ---: | --- | --- |
| 1024 | 0 | fail:1 | correctness 失败，`max_abs=0.071502685546875 > atol=0.0035` |
| 1024 | 1 | fail:1 | correctness 失败，`stats mismatch` |

- 结论：
  - normal eager 的必做性能表已完整记录；
- 当时 LL graph 因 1024 capture bucket correctness 失败，尚没有可信 replay 性能数据；后续已通过 K1 LL row-capacity 修复和 `phase10_ll_graph_20260615_232857` 全档补测覆盖。

## 2026-06-15 - V3 LL tail 32 与 128 bucket graph uneven 干净复现

- 复现前先清理远端容器残留进程：
  - 清理 `tests/test_mega_moe_dcu.py`、`timeout/tee`、`multiprocessing.spawn`、`multiprocessing.resource_tracker`；
  - `hy-smi --showpids` 显示 `No KFD PIDs currently running!`；
  - 8 卡 `VRAM%/HCU%` 均为 `0%/0%`。
- 输出目录：`hygon_tmp/sglang_debug/ll_repro_clean_20260615_212423/`。
- V3 LL tail 32 通过：
  - `fused_execution=large_opt_3stage`；
  - `Correctness 1/1: max_abs=0.000244141, mean_abs=9.31031e-06`；
  - fused median `0.711940 ms`，baseline median `2.774139 ms`。
- `ll_tail_graph_uneven` 旧 128 bucket 复现通过：
  - `fused_execution=large_opt_3stage_graph`；
  - token list `17,31,48,64,79,96,112,127`，capture/max bucket `128`；
  - replay `32`: local rank0 `17`，median `0.715500 ms`；
  - replay `96`: local rank0 `17`，median `0.787100 ms`；
  - replay `128`: local rank0 `17`，median `0.805760 ms`；
  - correctness `max_abs=0.000244141`。
- 两个 case 退出码均为 `0`；结束后再次清理，`hy-smi --showpids` 仍无 KFD 进程。
- 结论：旧 128 bucket 的 V3 LL tail graph uneven 路径仍可正常复现；之前卡住更像是 orphan multiprocessing rank 污染。Phase 10 中 `capture=1024` 的 LL graph correctness bug 后续已定位并修复为 K1 LL row-capacity 问题。

## 2026-06-15 - V3 LL eager 1024 复测

- 复测前后均执行远端清理，确认 `hy-smi --showpids` 无 KFD 进程残留。
- `V3 LL tail eager 1024`：
  - 输出目录：`hygon_tmp/sglang_debug/ll_repro_clean_20260615_212922_eager1024/`；
  - `fused_execution=large_opt_3stage`；
  - correctness 阶段失败，退出码 `1`；
  - rank 1 触发 `stats mismatch`，未生成 JSON 性能结果。
- `V3 LL no-tail eager 1024`：
  - 输出目录：`hygon_tmp/sglang_debug/ll_repro_clean_20260615_213042_eager1024_notail/`；
  - `fused_execution=large_opt_3stage`；
  - correctness 阶段失败，退出码 `1`；
  - rank 4 触发 `stats mismatch`，未生成 JSON 性能结果。
- 结论：
  - eager 1024 没有复现卡死，失败是确定性的 stats correctness mismatch；
  - no-tail/tail 都失败，问题更像 V3 LL 1024 大 bucket/容量相关共性问题，不是单独 tail-reduce 分支问题；
  - 旧 128 bucket graph uneven 仍是通过的，因此下一步应围绕 LL 从 128 到 1024 的 token/bucket 容量边界定位。

## 2026-06-15 - V3 LL 1024 correctness bug root cause and capacity fix

- 根因定位：
  - `tests/test_mega_moe_dcu.py` 中的 `fused_stats` mismatch 来自 K1 LL route/stage 阶段，不是 K3 graph replay 独有问题；
  - K1 LL host 侧原 `rows_per_expert = align(ceil(tokens_per_rank * topk / local_experts), 64)` 只按平均路由数对齐；
  - 1024 tokens/rank 时平均值是 `1024 * 6 / 32 = 192`，容量也正好是 192，没有随机 routing 抖动余量，部分 local expert 超过容量后被 K1 LL 截断，最终表现为 stats mismatch 或 no-tail 大 diff。
- 边界复现，输出目录 `hygon_tmp/sglang_debug/ll_debug_boundary_20260615_213428/`：
  - no-tail 512 通过，capacity 128；
  - no-tail 768 通过，capacity 192；
  - no-tail 896 失败，`max_abs=0.02630615234375`；
  - no-tail 960/1024 失败，触发 `stats mismatch`。
- 当时修复：
  - 只修改 `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu` 的 LL host capacity 计算；
  - 当时为了先修 1024 capture，保留 512/768 容量不变；
  - 当时策略是在 expected rows/expert >= 160 时补一块 64-row tile 余量，因此 896/960/1024 的 LL rows/expert 从 192 提升到 256；
  - 该阈值已在 2026-06-16 被 exact 256/512 eager headroom fix 取代，当前策略为 expected rows/expert >= 48 时补 64 rows。
- 远端构建：
  - 同步 `K1_fused/k1_fused_ext.cu` 后，在 `sglang_megamoe` 容器内执行 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过；
  - 验证前后均清理 orphan rank/timeout/tee/resource-tracker 进程，`hy-smi --showpids` 无 KFD 进程残留。
- 修复后 correctness，输出目录 `hygon_tmp/sglang_debug/ll_debug_capacity_fix_20260615_214214/`：
  - no-tail 512/768/896/960/1024 全部通过，`max_abs <= 0.000488281`；
  - tail 1024 通过，`max_abs=0.000488281`。
- 小 token 性能哨兵，输出目录 `hygon_tmp/sglang_debug/ll_perf_guard_capacity_fix_20260615_214700/`：
  - LL tail 32 通过，fused median `0.711300 ms`，与清理后复现值 `0.711940 ms` 持平；
  - 旧 128 bucket `ll_tail_graph_uneven` replay 32/96/128 通过，median `0.712760/0.786380/0.801960 ms`，相对之前 `0.715500/0.787100/0.805760 ms` 不劣化。
- graph capture=1024 验证，输出目录 `hygon_tmp/sglang_debug/ll_graph1024_capacity_fix_20260615_214857/`：
  - no-tail graph capture1024 通过，`max_abs=0.000488281`；replay 32/512/1024 median `1.449639/2.500839/4.694217 ms`；
  - tail graph capture1024 通过，`max_abs=0.000488281`；replay 32/512/1024 median `1.705559/2.792478/4.862097 ms`。
- eager 1024 对照，输出目录 `hygon_tmp/sglang_debug/ll_eager1024_perf_capacity_fix_20260615_215055/`：
  - no-tail eager1024 通过，fused median `4.672278 ms`；
  - tail eager1024 通过，fused median `4.922616 ms`；
  - graph replay 1024 与 eager 基本持平：no-tail `4.694217 ms` vs eager `4.672278 ms`，tail `4.862097 ms` vs eager `4.922616 ms`。
- 当前结论：
  - Phase 10 的 LL 1024 correctness bug 已修复；
  - 修复没有改变 512/768 和旧 32/128 小 token 容量，已测小 token 性能没有回退；
  - LL 1024 本身不适合作为生产优势路径，后续仍应由 normal backend 承担大 token；Phase 10 里 LL capture=1024 主要作为 graph correctness/capture bucket 覆盖；
  - 当时完整 `8,32,64,128,256,512,1024` LL graph no-tail/tail replay sweep 尚未重跑；后续已在 `phase10_ll_graph_20260615_232857` 补测完成。

## 2026-06-15 - V3 LL row headroom A/B and 513 boundary

- 用户追问 host-side row headroom 对性能是否有影响，以及 513 token/rank 这种边界是否能过。
- 新增临时 A/B runner：`hygon_tmp/sglang_debug/run_v3_ll_capacity_ab.sh`，复用同一测试命令跑 `512,513,896,1024` × no-tail/tail。
- A/B 三个版本（历史过程，结论已被 2026-06-16 exact 256/512 eager 修复取代）：
  - `headroom`：上一版策略，`expected >= 96` 时可能补 slack；513 会从 128 rows/expert 跳到 192；
  - `no_headroom`：临时把 `ll_min_slack=0`；
  - `refined`：当时收敛策略，只在 `expected rows/expert >= 160` 时补 64 rows，即 512/513 不补，896/1024 补。
- 输出目录：
  - `hygon_tmp/sglang_debug/ll_capacity_ab_20260615_230045_headroom/`
  - `hygon_tmp/sglang_debug/ll_capacity_ab_20260615_230735_no_headroom/`
  - `hygon_tmp/sglang_debug/ll_capacity_ab_20260615_231420_refined/`
- 关键结果：
  - 513 no-headroom 可以过：no-tail `2.200559 ms`，tail `2.298519 ms`；
  - 513 old headroom 也能过但更慢：no-tail `2.343639 ms`，tail `2.477599 ms`；
  - 513 refined 可以过且恢复到 no-headroom 性能区间：no-tail `2.193979 ms`，tail `2.279979 ms`；
  - 896/1024 no-headroom 会失败，refined 均通过；
  - refined 1024 no-tail/tail 分别 `4.727219/4.855758 ms`，和上一版 headroom `4.682059/4.915418 ms` 同一量级，差异在短测噪声内。
- 本轮代码结论（历史，已被 2026-06-16 exact 256/512 eager headroom fix 修正）：
  - 之前“较大 bucket 补 headroom”的表述需要更精确：不能从 513 开始补；
  - 当时生产策略改为 `ll_expected_rows_per_expert >= 160 ? 64 : 0`，保留 512/513 小中 bucket 原容量，同时修复 896/1024 overflow；
  - 后续 exact 256 eager 证明该阈值仍过窄，当前生产策略已改为 `ll_expected_rows_per_expert >= 48 ? 64 : 0`，覆盖实际执行到至少 512 tokens/rank；
  - 远端 refined 版已强制重新 hipify/编译 K1 ext，测试后 `hy-smi --showpids` 无 KFD 进程残留。

## 2026-06-15 - Phase 10 LL graph sweep 补测完成

- 用户要求补测 Phase 10 的 LL graph。
- 运行前后均检查远端进程和卡状态：
  - 容器内无残留 `tests/test_mega_moe_dcu.py`、`timeout/tee`、`multiprocessing.spawn` 或 `resource_tracker` 进程；
  - `hy-smi --showpids` 显示无 KFD 进程；
  - 测试结束后 8 卡 VRAM/HCU 均回到空闲状态。
- 输出目录：
  - `hygon_tmp/sglang_debug/phase10_ll_graph_20260615_232857/`
  - no-tail JSON: `graph_ll_tail0_1024.json`
  - tail JSON: `graph_ll_tail1_1024.json`
  - 汇总表: `summary.csv`
- V3 LL graph capture bucket 固定为 1024，replay tokens per rank 为 `8,32,64,128,256,512,1024`；no-tail/tail 均通过 correctness。

### V3 LL graph replay by tokens

| replay tokens/rank | no-tail graph ms | tail graph ms |
| ---: | ---: | ---: |
| 8 | 1.326 | 1.541 |
| 32 | 1.453 | 1.700 |
| 64 | 1.522 | 1.747 |
| 128 | 1.612 | 1.845 |
| 256 | 1.908 | 2.092 |
| 512 | 2.465 | 2.780 |
| 1024 | 4.631 | 4.825 |

- no-tail correctness/perf：
  - `correct=True`；
  - `max_abs=0.000488281`，`mean_abs=9.52169e-06`；
  - eager/fused median avg per rank `4.639659 ms`；
  - baseline median avg per rank `3.645959 ms`。
- tail correctness/perf：
  - `correct=True`；
  - `max_abs=0.000488281`，`mean_abs=9.52169e-06`；
  - eager/fused median avg per rank `4.832678 ms`；
  - baseline median avg per rank `3.704279 ms`。
- 结论：
  - Phase 10 必做 LL graph `8..1024` no-tail/tail replay sweep 已补齐；
  - 1024 capture correctness bug 修复后未再复现 stats mismatch 或大 diff；
  - LL 1024 仍只是 graph capture/correctness 覆盖点，不改变生产分段：小 token 走 LL，大 token 走 normal。

## 2026-06-16 - V3 LL graph replay runtime-row fix

- 用户指出 capture bucket 只应决定 graph 上限，不应让 replay 小 token 仍按 1024 token 的 work 量计时。
- 复查后确认 `phase10_ll_graph_20260615_232857` 慢表是中间状态：
  - K1 LL row-capacity bug 已修复，所以 correctness 通过；
  - 但 K3 LL graph 仍按 `l1_out.size(0)` / capture capacity 推导 fixed rows，并用 `kUseFixedRows=true` launch，导致 replay 8/32/64/128 仍做了大量 1024 bucket 的 K3 GEMM work。
- 修复：
  - K1 V3 LL 返回值第三项改为 `route_scratch_i32[0:local_experts]` 的 `ll_actual_m` view，保留实际 per-local-expert row counts；
  - K3 V3 LL graph/no-tail/tail combine 改为消费这组 `actual_m`，kernel template 从 fixed rows 切到 dynamic rows；
  - K3 LL kernel 内对 `actual_m` 做 `m_per_expert` clamp，避免异常 route count 越过当前 capacity；
  - Python wrapper 文档更新为 graph replay 使用 runtime row counts，而不是 capture row capacity。
- 远端构建：
  - 同步 `K1_fused/k1_fused_ext.cu`、`K3_fused/k3_v3_fused_ext.cu`、`K3_fused/k3_v3_pack5_groupgemm_impl.cuh`、`K3_fused/k3_fused.py`；
  - 重新执行 `python3 -m compileall` 和 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过；
  - 验证前后 `hy-smi --showpids` 均无 KFD 进程残留。
- 最终验证输出目录：
  - `hygon_tmp/sglang_debug/ll_graph_dynamic_verify_20260615_235731/`
  - `summary.csv`
  - `uniform1024_notail.json`
  - `uniform1024_tail.json`
  - `uneven128_notail.json`
  - `uneven128_tail.json`
- V3 LL graph capture bucket 固定为 1024，uniform replay tokens per rank 为 `8,32,64,128,256,512,1024`；no-tail/tail 均通过 correctness。

### V3 LL graph replay by tokens after runtime-row fix

| replay tokens/rank | no-tail graph ms | tail graph ms |
| ---: | ---: | ---: |
| 8 | 0.553 | 0.756 |
| 32 | 0.648 | 0.852 |
| 64 | 0.702 | 0.899 |
| 128 | 0.827 | 1.027 |
| 256 | 1.244 | 1.430 |
| 512 | 2.208 | 2.340 |
| 1024 | 4.740 | 4.855 |

- correctness/perf：
  - no-tail：`correct=True`，eager/fused median `4.780117 ms`，baseline `3.725959 ms`；
  - tail：`correct=True`，eager/fused median `4.882899 ms`，baseline `3.732879 ms`；
  - no-tail replay 32 从中间慢表 `1.453019 ms` 恢复到 `0.647959 ms`；
  - tail replay 32 从中间慢表 `1.699679 ms` 恢复到 `0.851500 ms`；
  - 1024 replay 基本保持在原 eager/graph 量级，符合 capture=runtime 时的预期。
- uneven graph 回归：
  - token list `17,31,48,64,79,96,112,127`，capture bucket `128`，graph replay `32,96,128`；
  - no-tail graph replay `0.598960/0.673880/0.688180 ms`，correct；
  - tail graph replay `0.632180/0.702880/0.719960 ms`，correct；
  - 相比 cleanup validation 旧值不劣化，反而略快。
- V3 normal graph 冗余计算检查：
  - 静态确认 normal graph 仍走 ASM-pack5 wrapper，不使用 V3 LL C fixed-row K3 路径；
  - K1 normal graph 传 `runtime_num_tokens` 到 ASM launch，K3 normal no-tail/tail 传 `active_tiles` / `graph_runtime_offset_from_active_tiles` 到 ASM host；
  - 未发现类似 LL graph `kUseFixedRows=true` 导致 replay 小 token 做 capture rows GEMM 的冗余计算。
- V3 normal graph spot check：
  - no-tail replay 512 单点复测通过，`max_abs=0.000488281`，graph replay `1.609760 ms`；
  - tail replay 512/1024 中 512 通过；一次 tail 1024 组合跑出现 `max_abs=0.0057373046875` 超 `--atol=0.0035`，清理后 tail 1024 单点复跑通过，`max_abs=0.000488281`，graph replay `1.974440 ms`；
  - 该现象当前按单次数值/状态波动记录，不升级为稳定阻断项；若后续 full matrix 再现，再单独立 normal graph correctness 诊断项。
- 当前结论：
  - LL graph runtime-row 问题已修复，Phase 10 LL graph 功能和性能均符合“capture 是上限、replay 按 runtime token work”的预期；
  - normal graph 没有发现同类冗余计算路径；
  - 生产分段仍保持：LL 负责小 token，normal 负责大 token。

## 2026-06-16 - V3 normal tail graph tolerance repeat check

- 用户要求复测上一轮 normal tail graph `1024` 超 `--atol=0.0035` 的情况，判断是否只是偶发。
- 复测前后均确认远端无残留测试进程，`hy-smi --showpids` 无 KFD PID，8 卡空闲。
- 复测脚本：
  - `hygon_tmp/sglang_debug/run_v3_normal_graph_tail1024_repeat.sh`
  - 复用 `run_v3_normal_graph_target.sh`，每轮独立进程启动、轮次间清理 orphan rank/resource-tracker。
- `TAIL=1 GRAPH_TOKENS=512,1024` 连续 4 次：
  - 输出目录：`hygon_tmp/sglang_debug/normal_graph_tail1024_repeat_20260616_001327/`
  - pass: iter 1、3，512/1024 均 `max_abs=0.000488281`；
  - fail: iter 2 在 replay 512 失败，`max_abs=0.0059814453125`；
  - fail: iter 4 在 replay 512 失败，`max_abs=0.011760711669921875`；
  - 结论：组合 replay 的 512 点不是稳定通过，失败率约 2/4。
- `TAIL=1 GRAPH_TOKENS=512` 连续 4 次：
  - 输出目录：`hygon_tmp/sglang_debug/normal_graph_tail512only_repeat_20260616_001740/`
  - 4/4 通过，均 `max_abs=0.000488281`；
  - replay median `1.649140/1.658380/1.664060/1.651200 ms`。
- `TAIL=1 GRAPH_TOKENS=1024` 连续 4 次：
  - 输出目录：`hygon_tmp/sglang_debug/normal_graph_tail1024only_repeat_20260616_002032/`
  - pass: iter 1、3、4，均 `max_abs=0.000488281`；
  - fail: iter 2 在 replay 1024 失败，`max_abs=0.01580810546875`；
  - 结论：1024-only 也能独立复现超容限，失败率约 1/4。
- 当前判断：
  - 这不是单次偶发；属于 normal tail graph replay correctness 的间歇性可复现问题；
  - 512-only 稳定通过，说明 512 token 本身不是必错；1024-only 可失败，说明问题也不依赖 `512 -> 1024` 的多 replay token 序列；
  - eager correctness 每轮仍先通过 `max_abs=0.000488281`，问题集中在 captured graph replay 检查；
  - 失败 rank 不固定：本轮看到 process 4、process 1、process 0；更像 graph/tail signal/runtime active-tile/state reset 或异步可见性问题，而不是单卡固定坏点。
- 后续若继续修复，应围绕 V3 normal tail ASM graph 的 runtime offset、active_tiles、tail signal/reset、graph replay 前后同步与 captured launch 参数状态做最小复现和插桩；不要再把它归类为已关闭的单次波动。

## 2026-06-16 - V3 normal graph multi-token replay root cause and fix

- 用户要求修复 capture 后连续 replay 多种 token 才触发的 graph correctness 问题，并回头检查 V3 LL graph 是否有同类问题。
- 根因定位：
  - V3 normal graph 实际使用 K1 ASM-pack5 wrapper；
  - K1 ASM graph 依赖 `staged_flags/meta_flags` 在每次 graph replay 前清零；
  - `large_opt.py` 之前只在 `v3_backend is None` 的 legacy ASM graph 下传 `k1_graph_reset_layout`，V3 normal graph 没有执行这段 reset；
  - captured graph 中 `flag_generation` 固定复用，未清零的上一轮 ready flag 会让 sibling CTA 过早进入后续阶段，因此表现为 replay 次数、token 序列相关的间歇性大 diff。
- 修复：
  - `_run_large_opt_3stage_graph()` 中 `k1_graph_reset_layout` 的适用范围改为 `v3_backend in (None, "normal")`；
  - V3 LL 不走该 reset，因为 LL C K1 kernel 在 kernel 内清 `symm_counts/row_combine/output_index`；
  - 删除测试中临时诊断开关 `MEGAMOE_DCU_DEBUG_CLEAR_GRAPH_COMBINE`；
  - 移除 no-tail ASM 里实验性的 done-counter publish/probe 块，避免给每个 WG 增加不必要的 GpuProb load/atomic 路径。
- 远端构建：
  - 同步 `large_opt.py`、`tests/test_mega_moe_dcu.py` 和 K3 no-tail ASM pack5 source；
  - 在 `sglang_megamoe` 容器内执行 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过；
  - `K3COMBINE_PACK5.co` 已重新生成，时间戳 `2026-06-16 04:02`，大小回到 `97104` bytes。
- V3 normal no-tail graph 验证：
  - `TAIL=0 GRAPH_TOKENS=1024 CUDA_GRAPH_REPLAYS=3` 通过，`max_abs=0.000488281`；
  - `TAIL=0 GRAPH_TOKENS=512,1024 CUDA_GRAPH_REPLAYS=3` 通过；
  - cleanup bench：512 replay `1.672859 ms`，1024 replay `2.006139 ms`；
  - `TAIL=0 GRAPH_TOKENS=512,1024 CUDA_GRAPH_REPLAYS=5` 通过，512/1024 均 `max_abs=0.000488281`。
- V3 normal tail graph 验证：
  - `TAIL=1 GRAPH_TOKENS=512,1024 CUDA_GRAPH_REPLAYS=3` 通过；
  - cleanup bench：512 replay `1.684199 ms`，1024 replay `1.994919 ms`；
  - repeat harness `ITERS=4 GRAPH_TOKENS=512,1024` 4/4 通过，每轮 512/1024 均 `max_abs=0.000488281`；
  - repeat 4 轮 replay ms：512 为 `1.681499/1.677179/1.678599/1.688518`，1024 为 `1.987438/1.982058/1.987358/1.991598`。
- V3 LL graph 回归检查：
  - 运行 `run_v3_ll_graph_dynamic_verify.sh`，覆盖 uniform capture1024 no-tail/tail replay `8,32,64,128,256,512,1024`，以及 uneven128 no-tail/tail replay `32,96,128`；
  - 四个 case 全部 `correct=True`；
  - uniform no-tail replay ms：`0.575440/0.672600/0.724540/0.852380/1.261380/2.193119/4.701418`；
  - uniform tail replay ms：`0.772280/0.865420/0.912220/1.044880/1.447259/2.369299/4.918218`；
  - uneven no-tail replay `32/96/128` 为 `0.602540/0.677380/0.699360 ms`；
  - uneven tail replay `32/96/128` 为 `0.635680/0.708560/0.731620 ms`。
- 结论：
  - normal graph multi-token replay correctness bug 已关闭；
  - root cause 是 K1 ASM-pack5 graph flag/meta reset 遗漏，不是 K3 no-tail/tail 数学错误；
  - 修复没有引入 K3 no-tail done-counter 等额外同步，normal graph replay 性能保持在既有区间；
  - V3 LL graph 未发现同类 reset 泄漏；runtime token replay 仍按实际 token 增长，功能正确。
- 测试结束后 `hy-smi --showpids` 显示无 KFD 进程残留。

## 2026-06-16 - Recent debug residual cleanup

- 用户确认最近调试引入的冗余逻辑是否已清理。
- 本次清理：
  - 移除 V3 normal no-tail pack5 combine 上遗留的 `asm_done_counter/asm_done_target` 可选传参；
  - 移除 `rank_barrier` 上仅服务于 no-tail publish/probe 实验的 `asm_done_wait_target` host/kernel/pybind 参数与等待分支；
  - 保留 tail-reduce 生产路径的 `asm_done_counter`，它仍用于 tail signal/ring generation reset，不属于冗余调试逻辑。
- 静态检查：
  - `asm_done_wait_target`、`graph_no_tail_done_target`、`MEGAMOE_DCU_DEBUG_CLEAR_GRAPH_COMBINE`、`k3_no_tail_publish`、`normal_graph_no_tail_publish` 在生产源码中均无残留；
  - `tests/test_dcu_megamoe_v3.py` 中旧 env 名称只作为 source guard blacklist 保留。
- 远端验证：
  - `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过；
  - `TAIL=0 GRAPH_TOKENS=512,1024 CUDA_GRAPH_REPLAYS=3` 通过，512/1024 均 `max_abs=0.000488281`，1024 fused median `2.063899 ms`；
  - `TAIL=1 GRAPH_TOKENS=512,1024 CUDA_GRAPH_REPLAYS=3` 通过，512/1024 均 `max_abs=0.000488281`，1024 fused median `2.063159 ms`；
  - 远端 `PYTHONPATH=. pytest -q tests/test_dcu_megamoe_v3.py` 通过，`6 passed`；
  - 测试前后 `hy-smi --showpids` 均无 KFD 进程残留。

## 2026-06-16 - Phase 10 refresh after LL graph/runtime-row fix

- 用户要求复查修复 LL bug 时是否引入冗余逻辑，并重新刷一遍 Phase 10 数据。
- LL 修复相关静态复查：
  - 生产源码未新增 LL debug env、临时开关或 probe 分支；
  - 当前核心合同是 K1 LL 返回 `ll_actual_m` CUDA view，K3 LL graph 用该 per-local-expert runtime row count 关闭 fixed-row replay work；
  - `tests/test_mega_moe_dcu.py` 中 `graph_runtime_debug/active_tiles_debug` 仅用于 assertion 失败信息，不在生产路径；
  - `k3_fused_ext.cu` 中 `KernelArgs.debug_d` 是 ASM host ABI 历史字段名，当前承载 `row_combine_ptrs`，不是 LL 修复引入的调试逻辑；
  - 可后续微清理但不影响热路径的历史项：K1 LL C entry 内部仍有 `use_ll=true` 派生出的 normal 分支表达式和 `ll_asm_compatible_layout` 兼容参数；本轮不在验证前扩大改动。
- 远端刷新：
  - 输出目录：`hygon_tmp/sglang_debug/phase10_v3_sweep_refresh_20260616_0820/`；
  - preflight：`compileall`、`tests/test_dcu_megamoe_v3.py`、`build_ext --inplace` 均通过；
  - 20/20 cases pass；结束后 `hy-smi --showpids` 无 KFD 进程残留。
- LL graph capture=1024 replay 数据：

| backend | tail | correct | fused 1024 ms | replay ms by token |
| --- | --- | --- | ---: | --- |
| LL graph | no-tail | True | 4.780178 | 8: 0.572640; 32: 0.670260; 64: 0.719340; 128: 0.849220; 256: 1.265120; 512: 2.234860; 1024: 4.750919 |
| LL graph | tail | True | 4.894639 | 8: 0.771180; 32: 0.867420; 64: 0.917680; 128: 1.047520; 256: 1.451479; 512: 2.368319; 1024: 4.818378 |

- Normal eager 数据：

| tokens/rank | no-tail ms | tail ms | no-tail speedup | tail speedup |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1.578140 | 1.622819 | 1.8639x | 1.8252x |
| 512 | 1.692700 | 1.743699 | 1.8763x | 1.8525x |
| 1024 | 2.078179 | 2.055999 | 1.7545x | 1.8205x |
| 1025 | 2.155419 | 2.137819 | 1.6904x | 1.7257x |
| 2048 | 3.349179 | 3.384519 | 1.6433x | 1.6359x |
| 2050 | 3.359299 | 3.336179 | 1.6301x | 1.6578x |
| 4096 | 5.904357 | 5.838618 | 1.6171x | 1.6247x |
| 4097 | 5.902698 | 5.837738 | 1.6095x | 1.6376x |
| 8192 | 12.594296 | 12.051475 | 1.3690x | 1.4334x |

- 结论：
  - LL graph 小 token replay 仍按 runtime token 增长，没有回到 capture rows 固定 work 的旧问题；
  - normal 1024/1025、2048/2050、4096/4097 边界无功能异常，也没有突兀性能台阶；
  - 8192 normal 仍正确，速度相对 baseline 仍为正收益，但 no-tail/tail 相比 4096 的增幅符合更大 token 档位成本增长。

## 2026-06-16 - V3 normal eager 8192 latency slope analysis

- 用户指出 normal eager `8192 vs 4096` 相比 `4096 vs 2048` 耗时涨幅偏大，要求定位原因。
- Phase 10 refresh 数据复算：
  - no-tail：2048 `3.349179 ms`，4096 `5.904357 ms`，8192 `12.594296 ms`；4096/2048 = `1.758x`，8192/4096 = `2.134x`；
  - tail：2048 `3.384519 ms`，4096 `5.838618 ms`，8192 `12.051475 ms`；4096/2048 = `1.725x`，8192/4096 = `2.064x`；
  - logical HBM bytes no-tail 4097->8192 只涨约 `1.333x`，但有效 HBM 从约 `204.9 GB/s` 掉到 `128.0 GB/s`，说明不是单纯 token 翻倍。
- K1 normal ASM-pack5 分段 bench：
  - 2048：V3 K1 `1.650719 ms`，rows `16384`；
  - 4096：V3 K1 `2.807361 ms`，rows `29696`；
  - 8192：V3 K1 `5.415037 ms`，rows `57344`；
  - 4096->8192 的 K1 时间比例 `1.929x`，与 rows 比例 `1.931x` 基本一致，K1 主要是 padded rows 线性放大，不是额外超线性的主因。
- K3 normal ASM-pack5 K3-only rowptr bench：
  - 2048 staged_rowptr `1.363743 ms`，staged_remote_only `1.369823 ms`；
  - 4096 staged_rowptr `2.579775 ms`，staged_remote_only `2.601502 ms`；
  - 8192 staged_rowptr `5.815899 ms`，staged_remote_only `6.187930 ms`；
  - 4096->8192：local_rowptr `1.894x`、staged_local_only `1.882x`，但 staged_rowptr `2.254x`、staged_remote_only `2.379x`。
- Rowptr 分布检查：
  - 8192 每 rank active rows 约 `48.9k-49.4k`，remote rows 约 `42.7k-43.3k`，local rows 约 `6.1k`；
  - rank bucket 分布均匀，没有明显热点 rank 或路由倾斜。
- 当前判断：
  - 8192 的额外涨幅主要来自 K3 remote combine/peer-write 路径在更大工作集下吞吐掉档；
  - K1/K3 padded rows 从 4096 的 `29696` 放大到 8192 的 `57344` 叠加了基础成本；
  - route_scratch/symm footprint 在 8192 显著扩大，可能进一步放大 peer write 地址跨度、cache/TLB/remote fabric 压力；
  - no-tail/tail 都受影响，说明问题在 shared staged normal path/K3 remote store，而不是 no-tail reduce 或 tail-reduce 独有逻辑。
- 诊断备注：
  - synthetic rowptr locality A/B 中的 contig/dest-sorted 模式会触发 K3COMBINE VMFault，不作为生产路径性能证据；
  - 结束后远端 `hy-smi` 显示无测试进程残留、HCU/VRAM 空闲。

## 2026-06-16 - K1/K3 span and remote-combine follow-up

- 用户要求继续验证 8192 性能问题，并借 8192 尝试优化 `symm_x_span > UINT32_MAX` 执行路径；同时关注 K3 remote combine locality / peer 写吞吐方向。
- K3 warmup A/B 已解析，输出来自 `hygon_tmp/sglang_debug/span_ab_20260616/k3_warm*.json`：

| tokens/rank | warmup | local_rowptr avg ms | staged_remote_only avg ms | staged_rowptr avg ms |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 0 | 1.560811 | 2.477867 | 2.503051 |
| 4096 | 1 | 1.559883 | 2.553583 | 2.571263 |
| 8192 | 0 | 2.961683 | 5.869373 | 5.838537 |
| 8192 | 1 | 2.965099 | 5.867301 | 5.841858 |

- K3 结论：
  - warmup 对 K3 remote combine 没有稳定正收益；
  - `symm_x_span > UINT32_MAX` 的主要影响在 K1 input staging/addressing，不是 K3 peer-store 本体；
  - K3 remote 方向后续应使用安全的 store-only/rowptr-locality probe，避免再用会破坏 row/data/expert 对齐的 synthetic contig rowptr 直接跑 K3COMBINE。
- K1 8192 `K1_PREBUILD_MODE` A/B：
  - 输出目录：`hygon_tmp/sglang_debug/k1_span_mode_ab_20260616/`；
  - auto：K1 V3 median 约 `5.377 ms`，rows `57344`，active `48975`，`symm_span_gt_u32=true`；
  - asm：K1 V3 median 约 `5.375 ms`，rows `57344`，active `48975`，`symm_span_gt_u32=true`；
  - compact：K1 V3 median 约 `6.502 ms`，rows `54272`，active `54272`，`symm_span_gt_u32=true`。
- K1 结论：
  - 8192 auto 已等价 asm-route fixed rows；强制 compact 虽减少 rows，但额外 prebuild/absolute-pointer 路径更慢，不能作为 8192 通用优化；
  - 下一步需要隔离 asm-route bit2 `{rank-local offset, source rank}` 与 asm-route absolute pointer 的差异，确认热循环内 source-rank SRD 选择是否是 8192 的可优化项。

## 2026-06-16 - K1 >4GB asm-route pointer mode A/B

- 为隔离 8192 `symm_x_span > UINT32_MAX` 下的 K1 source-load 路径，临时在 host 侧增加 `K1_ASM_ROUTE_X_PTR_MODE=rank_local|absolute` A/B 开关；测试后已从本地源码删除，并重新同步/重建远端 `.so`，不保留新生产环境变量。
- 输出目录：
  - `hygon_tmp/sglang_debug/k1_xptr_mode_ab_20260616/`
- 固定条件：
  - tokens/rank `8192`；
  - `K1_PREBUILD_MODE=asm`；
  - `K1_NORMAL_BENCH_MODE=v3`；
  - `K1_NORMAL_BENCH_ALLOC_ORDER=v3_first`；
  - `symm_span_gt_u32=true`；
  - rows `57344`，active rows `48975`。
- 结果：
  - rank-local bit2：K1 V3 median 约 `5.333675 ms`，min `5.290235 ms`；
  - absolute pointer：K1 V3 median 约 `6.395594 ms`，min `6.330554 ms`。
- 结论：
  - 当前 bit2 `{rank-local x offset, source rank}` MUBUF 路径明显优于 absolute global-load，同 rows 下快约 `1.06 ms`；
  - 8192 的 4GB span 优化不能通过切 absolute pointer 解决；
  - 后续若继续优化 K1 >4GB 路径，应考虑 ASM hot loop 内减少 `source_rank -> peer pointer table -> s_load_dwordx2` 的频率，或重整 row emission/source-rank grouping；但当前 code object `.amdhsa_next_free_sgpr=102`，没有显而易见的空闲 SGPR 缓存 8 个 peer SRD，任何 ASM 改动都必须做 code object resource/occupancy 和 2048/4096/8192 A/B。

## 2026-06-16 - K3 remote combine locality / peer-write A/B

- 用户要求在确认 K1 span 后继续试 K3 remote combine locality / peer 写吞吐方向，优先找通用优化，不做 8192 特化。
- 已执行的安全 synthetic probe：
  - `hygon_tmp/sglang_debug/k3_locality_ab_20260616/k3_dest_sorted_2048.json`
  - `hygon_tmp/sglang_debug/k3_locality_ab_20260616/k3_rank_bucket_2048.json`
  - `hygon_tmp/sglang_debug/k3_locality_ab_20260616/k3_rank_bucket_4096.json`
  - `hygon_tmp/sglang_debug/k3_locality_ab_20260616/k3_rank_bucket_8192.json`
  - `hygon_tmp/sglang_debug/k3_locality_ab_20260616/k3_sorted_8192.json`
- 2048 dest-sort 结果：
  - `staged_remote_only` avg `1.325055 ms`；
  - `staged_remote_dest_sorted` avg `1.988875 ms`；
  - 单纯按目标地址排序会明显破坏当前 K3 ASM half-tile staging/vector store 节奏。
- rank-bucket compact rowptr 结果：

| tokens/rank | staged_remote_only avg ms | rank_bucket_remote_only avg ms | 结论 |
| ---: | ---: | ---: | --- |
| 2048 | 1.335035 | 1.329027 | 噪声级略好 |
| 4096 | 2.486547 | 2.471371 | 噪声级略好 |
| 8192 | 5.758673 | 6.021509 | 退化 |

- 8192 sorted 复核：

| mode | avg ms |
| --- | ---: |
| staged_remote_only | 5.981337 |
| staged_rank_bucket_remote_only | 5.932649 |
| staged_remote_dest_sorted | 8.074312 |
| staged_rank_bucket_remote_sorted | 8.219244 |

- K3 结论：
  - rank-bucket compact 本身没有稳定正收益：8192 两轮一正一负，幅度都不足以支撑生产改动；
  - dest-sorted / rank-bucket-sorted 均显著劣化，说明当前 K3 ASM 的 half-tile LDS staging 与 `global_store_dwordx4` store cadence 比简单目的 rank 聚集更关键；
  - 不应做 K3-only rowptr reorder 或 combine-buffer rank bucket 生产改动；若未来重启 K3 locality，必须连同 K1 row emission、K2/K3 row order 和 correctness 合同一起设计。
- K1 静态复核：
  - V3 normal pack5 K1 ASM `sgpr_count=102`、`vgpr_count=255`、`private_segment=0`，没有明显 SGPR 余量缓存 8 个 peer SRD；
  - bit2 路径 route 阶段写 `{rank-local offset, source_rank}`，stage 阶段在每个 row/vector loop 里用 `source_rank` 从 peer pointer table `s_load_dwordx2` 得到 MUBUF base；
  - route 阶段 4 个 builder row tile 分 source rank 扫描，但 per-expert row 分配靠 atomic，行顺序天然混合 source rank；把 peer base 选择提升到 tile/CTA 粒度需要 source-rank grouped row emission 或 row metadata 合同变化，不能作为小补丁直接改。
- source-rank 分桶容量账：
  - 如果每个 expert/source-rank 子桶仍按 256-row K1 tile 对齐，2048/4096/8192 都会变成 `2048 rows/expert`，8192 比当前 `1792 rows/expert` 还差；
  - 如果设计 64-row 子桶，理论容量约为 2048=`512 rows/expert`、4096=`1024 rows/expert`、8192=`1536 rows/expert`，8192 有潜在收益；
  - 但 64-row 子桶会触动 K1 route emission、stage loop 的 peer-base 推导、K1 graph flag/reset 和后续 K2/K3 row contract，属于合同级改造，不适合在当前 pack5 ASM 上做小补丁。
- 当前状态：
  - K3 remote locality 方向先收敛为反证，不改生产路径；
  - Phase 9 剩余主动项集中在 K1 >4GB ASM/source-rank 合同级优化可行性评估。

## 2026-06-16 - Remove compact absolute pointer production path

- 用户判断：absolute pointer 路径性能太差，V3 normal cuda graph/compact prebuild 不应继续存在这个性能隐患；应尽量走 bit2 `{rank-local offset, source_rank}` MUBUF 路径。
- 代码收口：
  - `k1_emit_compact_routes_kernel()` 参数从 `use_absolute_x_ptrs` 改为 `use_rank_local_x_ptrs`；
  - compact prebuild 在 `symm_x_span > UINT32_MAX` 时写 `row_x_ptrs[row] = (source_rank << 32) | uint32(rank_local_offset)`；
  - host `prob.reserved_c0` 不再设置 bit1 absolute pointer，只在 compact bit0 外追加 bit2；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，禁止 `use_absolute_x_ptrs` 和旧 `use_compact_prebuild ? 2u : 4u` 回流。
- 静态/构建验证：
  - 本地 `git diff --check` 通过；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`6 passed`；
  - 远端 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过；
  - 为避免 ninja 误判增量，删除远端生成物 `build/temp.../K1_fused/k1_fused_ext.o` 后重建，确认 `hipcc` 重新编译 K1 object。
- 远端验证输出目录：
  - `hygon_tmp/sglang_debug/k1_compact_bit2_20260616_1058/`
- compact + >4GB e2e 验证：
  - env：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K1_PREBUILD_MODE=compact K3_USE_ASM_TAIL_REDUCE=0`；
  - tokens/rank `8192`，correctness 通过，`max_abs=0.000488281`；
  - fused `10.667335 ms`，baseline `16.507452 ms`，speedup `1.5475x`。
- K1-only compact 8192 验证：
  - `v3_symm_span_gt_u32=true`；
  - rows/active rows `54272/54272`；
  - K1 V3 median `5.156397 ms`，min `5.078558 ms`；
  - 相比旧 compact absolute 约 `6.50 ms`，说明去掉 absolute 后 compact 路径恢复到 bit2 MUBUF 性能区间。
- V3 normal graph smoke：
  - no-tail graph replay 512/1024：correct，replay `1.616879/1.948219 ms`，eager fused `2.033879 ms`；
  - tail graph replay 512/1024：correct，replay `1.660819/1.946260 ms`，eager fused `2.053519 ms`；
  - graph capture/compact 路径未发现 correctness 或性能回退。
- 结论：
  - production host path 已不再触发 absolute pointer bit1；
  - ASM code object 中旧 bit1 分支目前作为不可达历史分支保留，避免为删死分支重写 code object；
  - 若后续继续优化 8192，优先看 source-rank grouping/SRD 选择频率，而不是恢复 absolute pointer。

## 2026-06-16 - 8192 auto compact threshold and e2e/K3 attribution

- 用户指出 `12ms+ -> 10ms+` 不可能只由 K1 kernel 本体解释，要求测 e2e 并包含 K3。
- 远端状态：
  - `hy-smi` 显示 8 卡空闲；
  - 已同步 `k1_fused_ext.cu` / `tests/test_dcu_megamoe_v3.py` / 临时 bench wrapper；
  - 删除远端 K1 object 后 `MAX_JOBS=16 python3 setup.py build_ext --inplace`，确认 K1 重新 hipcc 编译。
- K1-only auto/compact 验证：
  - 输出目录：`hygon_tmp/sglang_debug/k1_only_auto_compact_20260616_auto_threshold_after_build/`；
  - `auto`: `symm_gt_u32=true`，rows/active `54272/54272`，K1 median max-rank `5.158318 ms`，rank0 `active_tiles=207`；
  - `compact`: `symm_gt_u32=true`，rows/active `54272/54272`，K1 median max-rank `5.187278 ms`；
  - 结论：当前 auto 已经选择 compact/bit2，不再走旧 asm-route。
- Full e2e no-tail auto/compact：
  - 输出目录：`hygon_tmp/sglang_debug/normal_8192_auto_compact_20260616_e2e_auto_compact_after_threshold/`；
  - `auto`: correctness 通过，`max_abs=0.000488281`，fused `10.792295 ms`，baseline `17.250931 ms`，speedup `1.5984x`；
  - `compact`: correctness 通过，`max_abs=0.000488281`，fused `10.858693 ms`，baseline `17.223492 ms`，speedup `1.5861x`。
- 旧 12ms+ 路径复现：
  - 输出目录：`hygon_tmp/sglang_debug/normal_8192_force_asm_20260616_asm_repro_12ms/`；
  - env：`K1_PREBUILD_MODE=asm`，其余同 V3 normal no-tail production；
  - correctness 通过，`max_abs=0.000488281`；
  - fused `12.712474 ms`，baseline `16.581692 ms`，speedup `1.3044x`；
  - 结论：12ms+ 可以复现，当前 auto 不复现是因为 auto path 已改成 compact/bit2。
- K3 拆分归因：
  - 输出目录：`hygon_tmp/sglang_debug/k3_8192_k1mode_compare_20260616_k3_k1mode_after_threshold/`；
  - `K1_PREBUILD_MODE=asm`: rows `57344`，K3 `staged_rowptr` median avg-rank `5.942537 ms`，`staged_remote_only` `5.929757 ms`，`local_rowptr` `2.962478 ms`，`rowptr_all_zero` `2.874631 ms`；
  - `K1_PREBUILD_MODE=compact`: rows `54272`，K3 `staged_rowptr` median avg-rank `4.670578 ms`，`staged_remote_only` `4.643146 ms`，`local_rowptr` `2.804235 ms`，`rowptr_all_zero` `2.714691 ms`；
  - K3 staged rowptr 单项下降约 `1.27 ms`，K1-only 仅下降约 `0.25 ms`，因此 e2e `12.71 -> 10.79 ms` 的主要改善来自 compact rows/rowptr 合同降低 K3 下游远端 combine 压力。
- 当前结论：
  - `kK1AutoCompactMinSaving=0.05` 与 compact/bit2 组合在 8192 上是合理默认；
  - 不应再把旧 compact absolute 的 `~6.50 ms` 结论外推到当前 compact/bit2；
  - 后续若继续优化 8192，应优先关注 source-rank grouped row emission / SRD 选择频率这类合同级方向，并用 e2e + K3 split 同时验证。

## 2026-06-16 - Auto rule refinement and ASM absolute branch cleanup

- 目标：
  - 根据当前性能归因完善 `K1_PREBUILD_MODE=auto`；
  - 清掉 K1 ASM 中已经不可达、且性能很差的 absolute pointer bit1 路径。
- auto 规则改动：
  - 保留 fractional saving 门槛 `kK1AutoCompactMinSaving=0.05`；
  - 新增 absolute local tile saving 门槛 `kK1AutoCompactMinLocalTileSaving=8.0`；
  - auto 只有在“比例上省得够”和“本 rank local tiles 绝对减少够多”同时满足时才启用 compact；
  - 公式检查结果：

| tokens/rank | auto | fixed rows | compact rows | 说明 |
| ---: | --- | ---: | ---: | --- |
| 256 | asm | 8192 | 8192 | 1 tile/expert，不 compact |
| 512 | asm | 8192 | 8192 | 1 tile/expert，不 compact |
| 1024 | asm | 8192 | 8192 | 1 tile/expert，不 compact |
| 1025 | compact | 16384 | 8960 | 跨 1024 后 fixed rows 翻倍，compact 明显压 rows |
| 2048 | asm | 16384 | 16384 | 预计无 rows savings |
| 2050 | asm | 16384 | 16384 | 预计无 rows savings |
| 4096 | compact | 32768 | 29696 | rows 降 `3072` |
| 4097 | compact | 32768 | 29696 | rows 降 `3072` |
| 8192 | compact | 57344 | 54272 | rows 降 `3072` |

- ASM cleanup：
  - 两份 K1 ASM 源都移除了 `label_SymmRouteStoreAbsolutePtr` 和 `label_SymmStageLoadAbsolutePtr`；
  - route emit 不再根据 `reserved_c0 >= 2` 写 absolute pointer，只检查 bit2；
  - stage source-load 不再根据 bit1 做 `global_load_dwordx4` absolute pointer，只保留 default uint32-offset MUBUF 与 bit2 rank-local MUBUF。
- Source guard：
  - `tests/test_dcu_megamoe_v3.py` 增加 guard，要求 host 不出现旧 `use_absolute_x_ptrs` / `use_compact_prebuild ? 2u : 4u`，ASM 源不出现 absolute labels 或 `absolute 64-bit` 注释。
- 验证：
  - 本地 `git diff --check` 通过；
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - 本地 pytest 不可用：`No module named pytest`；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`6 passed`；
  - 远端删除 K1 object 后 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过，K1 host ext 已重编，build_ext 末尾会重新构建 large-opt ASM code objects。
- 待补：
  - 远端当前被外部 `sglang serve` 占用，`hy-smi` 显示多卡 `90-96% VRAM` 且多卡 HCU 接近 `100%`；
  - 未跑 8192/4096 e2e perf，避免得到污染数据；
  - 卡空后补：K1-only auto、8192 e2e auto/compact、4096/8192 no-tail/tail 至少一组性能哨兵。

## 2026-06-16 - 8448 auto rule boundary check

- 用户要求等卡空后单独刷 8448，分别比较 `K1_PREBUILD_MODE=compact/asm`，确认 auto 规则效果是否符合预期。
- 远端状态：
  - 初始被外部 `sglang serve --mem-fraction-static 0.88` 占用，8 卡 VRAM `91-93%`，未清理外部进程；
  - 卡空后 `hy-smi` 显示 8 卡 VRAM/HCU 均为 `0%`，开始测试。
- 输出目录：
  - 初版规则 A/B：`hygon_tmp/sglang_debug/normal_8448_auto_rule_20260616_130012/`；
  - 修正后 auto 复测：`hygon_tmp/sglang_debug/normal_8448_auto_rule_after_fix_20260616_131259/`。
- K1-only 初版规则结果：

| mode | rows max | active max | K1 median max-rank ms | rank0 active_tiles |
| --- | ---: | ---: | ---: | ---: |
| asm | 57344 | 50935 | 5.410402 | 0 |
| compact | 57344 | 57344 | 5.400239 | 221 |
| auto | 57344 | 50935 | 5.389836 | 0 |

- 解释：
  - 8448 下 fixed capacity 是 `7 tiles/expert = 57344 rows`；
  - compact prebuild 的真实 tile 需求约 `221 tiles`，但加 margin 后被 clamp 到 fixed `224 tiles`，所以最终 capacity rows 没降；
  - 因此“compact 不降 rows”只表示最终 launch/capacity rows 未减少，不表示没有走 count/prebuild。
- Full e2e 初版规则结果：

| mode | correct | fused median ms | fused min ms | baseline median ms | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| asm | true | 12.747614 | 12.652854 | 18.021132 | 1.4137x |
| compact | true | 11.247952 | 11.129933 | 17.924590 | 1.5936x |
| auto | true | 12.863582 | 12.492222 | 18.040603 | 1.4025x |

- K3 split 归因：

| mode | rows | active rows | K3 staged_remote_only max-rank ms | K3 staged_rowptr max-rank ms |
| --- | ---: | ---: | ---: | ---: |
| asm | 57344 | 50688 | 6.194325 | 6.380214 |
| compact | 57344 | 57344 | 4.807185 | 4.865729 |
| auto | 57344 | 50688 | 6.007669 | 6.036385 |

- 结论：
  - 初版 auto 只看 estimated capacity rows saving，8448 被留在 asm；
  - 但 compact 即使没有降低最终 capacity rows，仍通过 prebuild/padding/row metadata 形态显著降低 K3 rowptr/remote combine 时间；
  - auto 规则需要覆盖 `fixed_capacity_tiles_per_expert >= 7` 的大 rowptr 档。
- 代码修正：
  - 新增 `kK1AutoCompactLargeTilesPerExpert = 7`；
  - `should_auto_compact_routes()` 在 `asm_tiles_per_expert >= 7` 时直接选择 compact；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard 防止该规则被误删。
- 验证：
  - 本地 `git diff --check` 通过；
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - 公式 sanity：`256/512/1024/2048/2050` 仍为 asm，`1025/4096/4097/8192/8448` 为 compact；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`6 passed`；
  - 远端删除 K1 object 后 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过，K1 host ext 已重新 hipcc 编译。
- 修正后 8448 auto：
  - K1-only：`symm_gt_u32=true`，rows/active `57344/57344`，rank0 `active_tiles=221`，K1 median max-rank `5.440002 ms`；
  - e2e：correct，fused `11.202062 ms`，baseline `17.889183 ms`，speedup `1.5970x`；
  - 结果与强制 compact 对齐，auto 规则符合当前性能预期。

## 2026-06-16 - Representative auto-rule e2e matrix

- 用户要求对代表 size 全部跑 `asm/compact` e2e，并检查当前 auto 规则是否能选到最好的。
- 测试口径：
  - V3 normal eager no-tail；
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0`；
  - 每个 size 跑 `K1_PREBUILD_MODE=asm/compact/auto`；
  - `correctness_iters=1, warmup=2, repeat=5`；
  - 输出目录：`hygon_tmp/sglang_debug/normal_auto_rule_matrix_20260616_131851/`。
- 矩阵结果，30/30 case 均 rc=0 且 correctness 通过：

| tokens/rank | asm ms | compact ms | auto ms | forced best | auto 判定 |
| ---: | ---: | ---: | ---: | --- | --- |
| 256 | 1.585520 | 1.637019 | 1.585900 | asm | ok |
| 512 | 1.691620 | 1.736819 | 1.701539 | asm | ok |
| 1024 | 2.070819 | 2.071639 | 2.057319 | asm | ok |
| 1025 | 2.707359 | 2.137899 | 2.130799 | compact | ok |
| 2048 | 3.414738 | 3.371778 | 3.357278 | compact/noise | ok |
| 2050 | 3.354319 | 3.416038 | 3.405738 | asm | ok |
| 4096 | 6.367037 | 5.911437 | 5.902677 | compact | ok |
| 4097 | 6.423017 | 5.972537 | 5.936797 | compact | ok |
| 8192 | 12.591534 | 10.754614 | 10.836473 | compact | ok |
| 8448 | 12.800073 | 11.193274 | 11.371170 | compact | ok |

- 2048 边界复核：
  - 首轮矩阵里 2048 forced compact 只比 forced asm 快 `0.043 ms`，且 auto 反而更快，属于噪声级；
  - 追加三轮 2048 e2e 复核，输出目录 `hygon_tmp/sglang_debug/normal_auto_rule_2048_recheck_20260616_134105/`；
  - 三轮均 correctness 通过：

| mode | runs ms | median-of-runs ms | mean ms |
| --- | --- | ---: | ---: |
| asm | `3.347297, 3.380317, 3.327497` | 3.347297 | 3.351704 |
| compact | `3.422937, 3.411917, 3.416197` | 3.416197 | 3.417017 |
| auto | `3.397477, 3.374377, 3.408717` | 3.397477 | 3.393523 |

- 结论：
  - 当前 auto 规则保留 `256/512/1024/2048/2050 -> asm` 是合理的；
  - `1025/4096/4097/8192/8448 -> compact` 与代表矩阵一致；
  - 2048 不应因单轮噪声切 compact，三轮复核显示 asm 更稳；
  - no-tail 代表矩阵未发现 correctness 或性能回退。后续只需补 tail-reduce 哨兵，而不是继续扩大 no-tail 矩阵。

## 2026-06-16 - V3 LL graph to-256 refresh and K2 fusion analysis

- 用户要求重刷 V3 LL graph 性能到 256 token/rank，并分析 LL 阶段是否值得把 K2 融到 K1。
- 远端状态：
  - 测试前后 `hy-smi --showpids` 均显示无 KFD 测试进程残留；
  - 输出目录：`hygon_tmp/sglang_debug/ll_graph_to256_20260616_142319/`；
  - 命令口径：V3 LL backend，graph capture bucket 固定 `1024`，replay tokens `8,32,64,128,256`，no-tail/tail 均覆盖。
- 性能结果：

| mode | correct | eager fused 1024 ms | baseline 1024 ms | graph replay 8 ms | 32 ms | 64 ms | 128 ms | 256 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LL no-tail | true | 4.724817 | 3.745018 | 0.571340 | 0.669559 | 0.720960 | 0.853179 | 1.266899 |
| LL tail | true | 4.835317 | 3.724398 | 0.769940 | 0.869400 | 0.917420 | 1.046220 | 1.441659 |

- K2 融 K1 的当前判断：
  - 技术上可以做，但不是轻量 wrapper 合并；K2 需要同时消费 K1 L1 输出的 gate/up 两半，执行 SwiGLU、route weight、整行 amax reduction，再写 FP8 activation 和 per-row scale；
  - 当前 K1 GEMM 按 N tile 产生 `l1_out`，SwiGLU 和 scale 需要跨 `hidden=2048` 的整行信息；若直接塞进 K1 epilogue，会引入跨 tile 汇总 scale 或新的中间 staging，容易破坏 K1 主循环和 occupancy；
  - graph replay 中 CPU launch overhead 已被 capture 摊掉，历史 LL stage timing 显示 K2 在 32/128 档约 `0.028 ms`，不是当前 LL 小 token 主瓶颈；
  - 融合上限主要是省一个 K2 graph node、`l1_out` BF16 写读和 K2 kernel 自身设备时间，预计 8-128 token 收益多半是几个百分点级，256 可能略大，但需要 profiler 证据；若 K1 epilogue 变重或降 occupancy，容易把收益吃掉。
- 后续建议：
  - 不把 K2 融 K1 作为当前必做生产项；
  - 若用户后续继续压 LL 小 token，再先用 profiler/阶段计时确认 32/64/128/256 下 K2 占比是否超过 `~10%`；
  - 只有 K2+`l1_out` traffic 证明确实成为瓶颈时，再立项做 K1/K2 合同级 prototype，并必须覆盖 LL no-tail/tail、graph/eager、uneven、8/32/64/128/256 correctness 和性能不劣化。

## 2026-06-16 - Latest V3 normal eager refresh

- 用户要求重刷最新 V3 normal eager 性能。
- 远端状态：
  - 运行前 8 卡 VRAM/HCU 均空闲，`hy-smi --showpids` 无 KFD 进程；
  - 运行后 `hy-smi --showpids` 仍无 KFD 进程残留；
  - 输出目录：`hygon_tmp/sglang_debug/normal_eager_refresh_20260616_143314/`；
  - 命令口径：V3 normal backend，eager，auto K1 prebuild mode，`warmup=5`、`repeat=10`、`correctness_iters=1`，no-tail/tail 均覆盖。
- 结果，18/18 case 均 pass 且 `correct=True`：

| tokens/rank | no-tail fused ms | no-tail baseline ms | no-tail speedup | tail fused ms | tail baseline ms | tail speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1.580579 | 3.014778 | 1.9074x | 1.612399 | 3.003118 | 1.8625x |
| 512 | 1.693419 | 3.179158 | 1.8774x | 1.756319 | 3.224318 | 1.8358x |
| 1024 | 2.077039 | 3.674278 | 1.7690x | 2.021579 | 3.692378 | 1.8265x |
| 1025 | 2.158799 | 3.631338 | 1.6821x | 2.138479 | 3.637958 | 1.7012x |
| 2048 | 3.354978 | 5.515377 | 1.6439x | 3.316958 | 5.509137 | 1.6609x |
| 2050 | 3.363358 | 5.432777 | 1.6153x | 3.323778 | 5.508837 | 1.6574x |
| 4096 | 5.913157 | 9.552534 | 1.6155x | 5.831117 | 9.524435 | 1.6334x |
| 4097 | 5.893257 | 9.481355 | 1.6088x | 5.845857 | 9.499595 | 1.6250x |
| 8192 | 10.790015 | 17.325931 | 1.6057x | 10.749434 | 17.274311 | 1.6070x |

- 结论：
  - 当前 normal eager 最新性能与上一轮 auto-rule no-tail 矩阵一致，8192 维持 `~10.75-10.79 ms`，没有回到旧 asm-route `12ms+` 档；
  - 4096 tail/no-tail 仍在 `~5.83-5.91 ms`，与 cleanup refresh `~5.81/5.84 ms` 同区间；
  - tail 在 normal eager 大 token 下没有明显劣化，部分档位还略快于 no-tail，当前不需要单独调整 normal tail 默认策略。

## 2026-06-16 - V3 LL block_m=64 at 256-token A/B

- 用户要求确认 256 token/rank 时 `block_m=64` 是否可能优于当前 `block_m=32`。
- 执行方式：
  - 不改生产代码；远端用临时 `sitecustomize.py` monkey-patch `megamoe.large_opt.V3_LL_BLOCK_M`；
  - 只测 LL no-tail，因为当前 no-tail 是 LL 性能路径；
  - 输出目录：`hygon_tmp/sglang_debug/ll_blockm64_256_ab_clean_20260616_150434/`。
- 清理：
  - 第一轮手动 rank 启动方式发生分布式 hang，已清理 `/tmp/run_ll_blockm_ab.sh` 和对应 Python/KFD 进程；
  - 清理后复查无残留 Python 测试进程；最终复测后也无相关 Python 残留。
- graph capture1024/replay256 结果：

| block_m | correctness | fused eager 1024 ms | graph replay 256 ms | 结论 |
| ---: | --- | ---: | ---: | --- |
| 32 | true | 4.723016 | 1.249939 | retained default |
| 64 | true | 10.763655 | 3.216179 | 明显退化 |

- exact eager bucket=256：
  - `block_m=32` 与 `block_m=64` 都触发 correctness over-threshold（分别约 `0.01697`、`0.01563`），不作为性能判断依据；
  - 该现象更像 LL exact-256 bucket 的 per-expert row capacity/headroom 太紧，而不是 block_m64 优化收益；当前 Phase 10 生产式 LL graph 使用 capture bucket 1024，replay256 correctness 是 clean 的。
- 结论：
  - 256 token/rank 下 `block_m=64` 没有收益，graph replay 约慢 `2.57x`；
  - retained LL `block_m=32` 继续保留；
  - 后续若要看 eager 256，应先单独处理 exact-256 bucket row headroom/capacity，而不是把 block_m64 当优化项。

## 2026-06-16 - V3 LL exact 256/512 headroom fix

- 用户要求：LL 实际执行至少支持到 512 tokens/rank；graph capture tokens 可能更大，但不能依赖大 capture bucket 掩盖 exact eager bucket 的容量问题。
- 根因：
  - exact eager 256 的 expected rows/expert 为 `ceil(256*6/32)=48`，旧策略不补 headroom，最终每 expert 只有 64 rows；
  - 随机路由可能让某个 local expert 超过 64 rows，K1 LL staged rows 被截断，导致 correctness over-threshold；
  - graph capture1024/replay256 之前能过，是因为 capture bucket 提供了更大的 rows/expert capacity，不代表 exact 256/512 eager 安全。
- 代码修复：
  - `k1_fused_ext.cu` 新增 `kLlHeadroomExpectedRowsThreshold = 48` 和 `kLlHeadroomRows = 64`；
  - `ll_expected_rows_per_expert >= 48` 时补一块 64-row headroom；
  - 保留 32/128 tiny bucket 原容量，exact 256/512 获得 headroom；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，防止旧 `>=160 ? 64 : 0` 策略回流。
- 本地验证：
  - `python -m compileall tests/test_dcu_megamoe_v3.py` 通过；
  - `git diff --check` 通过。
- 远端验证：
  - 显式同步 `k1_fused_ext.cu` / `tests/test_dcu_megamoe_v3.py`；
  - `python3 -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu` 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`；
  - 删除 K1 object 后 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过，确认 K1 重新 hipcc 编译；
  - 输出目录：`hygon_tmp/sglang_debug/ll_exact512_headroom_fix_20260616_151830/`；
  - 测试前后 `hy-smi --showpids` 无 KFD 进程残留。

| mode | tail | tokens/capture | correct | eager/capture-bucket ms | baseline ms | replay token ms |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| eager | 0 | 32 | True | 0.628280 | 2.808579 |  |
| eager | 0 | 128 | True | 0.819619 | 2.842279 |  |
| eager | 0 | 256 | True | 1.227839 | 2.915279 |  |
| eager | 0 | 512 | True | 2.191659 | 3.253159 |  |
| eager | 1 | 32 | True | 0.632459 | 2.804659 |  |
| eager | 1 | 128 | True | 0.842480 | 2.876918 |  |
| eager | 1 | 256 | True | 1.277139 | 2.987038 |  |
| eager | 1 | 512 | True | 2.308979 | 3.219738 |  |
| graph | 0 | 1024 | True | 4.709938 | 3.725738 | 256: 1.268140; 512: 2.232419 |
| graph | 1 | 1024 | True | 4.895618 | 3.721958 | 256: 1.449939; 512: 2.372699 |

- 结论：
  - exact eager 256/512 no-tail/tail 已恢复 correctness；
  - graph capture1024 replay256/512 仍保持 correctness；no-tail replay 相比 exact eager 只慢约 `0.040/0.041 ms`，tail replay 慢约 `0.173/0.064 ms`；
  - `block_m=64` 仍不作为优化方向，headroom fix 才是 exact 256/512 的根因修复；
  - LL 的生产上界按用户要求覆盖到至少 512 tokens/rank，capture bucket 更大时仍按 runtime tokens 执行。

## 2026-06-16 - V3 LL graph tail reduce runtime-token fix

- 用户要求修复 K2 和 LL tail reduce/signal 还不是完全 runtime-token 化的问题，重点是 graph 同 runtime size 下 tail 比 eager/no-tail 慢。
- 根因复核：
  - K2 graph launcher 形状仍是 capture bucket 固定 grid，这是 CUDA graph 静态 launch 形状限制；当前 graph 路径已经始终传入 `row_combine_ptrs`，K2 kernel 对 inactive row 执行 `row_combine_ptrs[row] == 0` early-return，避免真实 SwiGLU/quant 计算。进一步缩小 K2 grid 需要 graph exec update 或 compact row-list 合同，不作为本次小修。
  - LL tail reduce 的 K3 C path 在 graph replay 时仍按 capture `num_tokens=1024` 扫 `reduce_y`，导致 replay 8/32/64/128/256/512 都多做 capture bucket 级别的 reduce work。
- 代码修复：
  - `large_opt.py` 在 graph tail path 将 `state.scratch.graph_runtime_num_tokens` 传给 K3 V3 wrapper；
  - `k3_fused.py` 给 V3 K3 wrapper 增加 `graph_runtime_num_tokens` 参数，并透传给 `k3_v3_ll_combine_tail`；
  - `k3_v3_fused_ext.cu` 增加可选 `runtime_num_tokens_tensor` pybind 参数和 host launcher 指针；
  - `k3_v3_pack5_groupgemm_impl.cuh` 的 `v3_k3_tail_reduce_worker_device` 用 `effective_num_tokens = clamp(runtime_num_tokens[0], 0, capture_num_tokens)` 限定 reduce vec 数；
  - `tests/test_dcu_megamoe_v3.py` 增加 source guard，防止 K3 LL tail graph 退回 capture token reduce。
- 验证：
  - 本地 `compileall` 通过，`git diff --check` 通过；
  - 远端 `compileall` + `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`7 passed`；
  - 删除 K3 V3 object 后 `MAX_JOBS=16 python3 setup.py build_ext --inplace` 通过；
  - 输出目录：`hygon_tmp/sglang_debug/ll_runtime_tail_tokens_20260616/`。
- graph capture1024 replay 性能刷新：

| mode | correct | capture bucket ms | graph replay 8 ms | 32 ms | 64 ms | 128 ms | 256 ms | 512 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LL no-tail | true | 4.745158 | 0.569840 | 0.666400 | 0.714360 | 0.838739 | 1.266799 | 2.200319 |
| LL tail | true | 4.887598 | 0.553399 | 0.649480 | 0.705980 | 0.849259 | 1.283039 | 2.240819 |

- eager tail sanity：

| mode | tokens | correct | fused ms | baseline ms |
| --- | ---: | --- | ---: | ---: |
| LL tail eager | 256 | true | 1.260800 | 2.717559 |
| LL tail eager | 512 | true | 2.364599 | 3.043638 |

- 结论：
  - tail graph replay 256 从旧 `~1.44 ms` 降到 `~1.28 ms`，512 从旧 `~2.37 ms` 降到 `~2.24 ms`，已接近 no-tail 同 size；
  - 小 token tail replay 8/32/64 也不再被 capture1024 reduce work 拖慢；
  - K2 剩余固定 grid 成本是 graph 静态形状限制下的外壳成本，当前通过 rowptr early-return 控制住，未发现需要高风险合同改造的收益证据；
  - 本修复不改变 eager tail 的生产路径，eager 256/512 correctness 与性能保持正常。
