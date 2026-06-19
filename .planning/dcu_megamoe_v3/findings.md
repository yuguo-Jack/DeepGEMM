# DCU MegaMoE V3 Findings

## 初始代码地图

- `megamoe/large_opt.py` 是现有 DCU MegaMoE staged K1/K2/K3 编排入口。
  - K1 当前调用 `k1_symm_fused_l1_asm(...)`。
  - K2 当前调用 `swiglu_quant_channelwise_out(...)`。
  - K3 当前调用 `k3_l2_fused_asm_to_combine(...)`。
  - `K3_USE_ASM_TAIL_REDUCE=1` 时 K3 走 tail-reduce fused code object；否则 K3 只写 combine buffer，再执行 `rank_barrier + reduce_local_combine`。

## `large_opt.py` staged flow 细节

- shared state/scratch：
  - `_route_scratch_views()` 在 `sym_buffer.route_scratch` 上切出 `k1_active_tiles`、`l1_out`、`act_fp8`、`act_scale`、`k3_prob_storage`、`graph_runtime_num_tokens`、`asm_tail_done_counter`、`asm_tail_signal_addrs`；
  - `_state()` 缓存这些 view，并在 tail-reduce 开启时通过 `build_asm_tail_signal_addrs()` 准备 signal address；
  - `prepare_large_opt_3stage()` 只在 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE` force/auto 时预热 state。
- eager staged path `fp8_mega_moe_large_opt_3stage()`：
  - shape 固定 EP8、experts=256、topk=6、hidden=4096、intermediate=2048；
  - K1 前先调用 `rank_barrier()`，用于让 symmetric-buffer input copies 对 ASM dispatch-pull 可见，并可重置 tail signal；
  - K1 调用 `k1_symm_fused_l1_asm()`，返回 `l1_out, route_weights, m_indices, output_index, row_combine_ptrs`；
  - K2 复用 `swiglu_quant_channelwise_out()`，输入 K1 的 `l1_out/route_weights`，输出 `act_fp8/act_scale`；
  - K3 tail-reduce on：`k3_l2_fused_asm_to_combine()` 传入 `asm_done_counter/signal_addrs/asm_reduce_y/sym_buffer/output_workspace/prob_storage`，K3 内完成 combine + local reduce；
  - K3 tail-reduce off：`k3_l2_fused_asm_to_combine()` 只写 combine buffer，随后复用 `rank_barrier()` 和 `reduce_local_combine()`。
- graph staged path `_run_large_opt_3stage_graph()`：
  - 使用 graph bucket 的 `graph_max_tokens` 做 shape/scratch sizing，真实 runtime token 由 `sym_buffer.cuda_graph_num_tokens` 提供；
  - K1 前 `rank_barrier()` 还负责 reset K1 graph flags/meta flags，并把 runtime token 写到 `state.scratch.graph_runtime_num_tokens`；
  - K1 调用 `k1_symm_fused_l1_asm_graph()`，K2 仍复用同一 `swiglu_quant_channelwise_out()`；
  - K3 tail-reduce on 额外传 `active_tiles` 和 `graph_runtime_offset_from_active_tiles`；
  - K3 tail-reduce off 后调用 `reduce_local_combine_graph()`，用 runtime token 控制实际输出。
- V3 集成含义：
  - 第一版 eager 可以只替换 K1/K3 wrapper 调用，K2、scratch/state/cache 和外部 reduce/barrier 尽量不动；
  - graph V3 需要额外对齐 `active_tiles`、runtime token offset、K1 flags reset 和 capture-safe 参数，适合在 eager 稳定后做；
  - K1 pre-rank-barrier 第一版先保留，后续再独立 A/B 移除。

## 原 K1 ASM 路径

- Python wrapper: `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py`
- C++/HIP host/ext: `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu`
- ASM code object name:
  - `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1.co`
- K1 wrapper 负责：
  - 校验 EP8 / experts=256 / topk=6 / hidden=4096 / alignment=256；
  - 根据 sym buffer 指针计算 graph-safe input address range；
  - 在 route_scratch 中切出 `row_combine_ptrs`、`route_weights`、`row_x_ptrs`、`row_x_scales`、`m_indices`、`output_index`、`staged_x`、flags；
  - 可选 compact prebuild；
  - launch ASM K1 并返回 `l1_out, route_weights, m_indices, output_index, row_combine_ptrs`。

## 原 K3 ASM 路径

- Python wrapper: `megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py`
- C++/HIP host/ext: `megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused_ext.cu`
- ASM code object:
  - no-tail: `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.co`
  - tail: `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE.co`
- K3 wrapper 负责：
  - no-tail: K3 L2 GEMM 结果写入 `row_combine_ptrs` 指向的 combine buffer；
  - tail: 同 kernel 还通过 `asm_done_counter` / `asm_signal_addrs` 做 tail signal，然后把本地 combine reduce 到 `y`；
  - graph 额外支持 `active_tiles` 和 runtime token offset。

## V2 可复用资产

- V2 已有 Python 包和扩展：
  - `megamoe/dcu_megamoe_v2/K1_fused/*`
  - `megamoe/dcu_megamoe_v2/K3_fused/*`
  - `megamoe/dcu_megamoe_v2/runtime.py`
- V2 kernel 主体仍来自 `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`。
- 可复用能力：
  - K1 LL/normal dispatch-pull + L1 groupgemm；
  - K1 metadata 输出：`route_weights`、`row_expert/m_indices`、`output_index`、`row_combine_ptrs`、`local_topk_mask`、`tail_tokens`；
  - K3 LL rowptr + tail reduce；
  - K3 normal copy-stage + tail reduce；
  - pack5 L1/L2 weight layout。

## 用户确认的 V3 设计决议

- V3 门控：
  - `USE_MEGAMOE_V3` 只在 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1` 时生效；
  - 未 forced 3-stage large_opt 时，V3 env 不改变原始 DCU MegaMoE 路径；
  - 默认路径、原 staged fused ASM 路径和命令行参数必须保持兼容。
- 后端选择：
  - 使用 `MEGAMOE_DCU_V3_BACKEND=ll|normal`；
  - V3 开启但未设置 backend 时默认 `normal`；
  - 显式设置 `MEGAMOE_DCU_V3_BACKEND=ll` 时才走 LL；
  - LL 骨架适用于 tokens per rank `<512`；
  - normal 骨架适用于 tokens per rank `>=512`；
  - 不复用 `MEGAMOE_DCU_V2_BACKEND` 作为 V3 对外接口。
- 代码位置：
  - V3 直接放进 `megamoe/dcu_megamoe_large_opt`；
  - K1/K3 kernel 剥离到已有 `K1_fused` / `K3_fused` 文件夹；
  - 不继续改动 `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp` 作为生产实现。
- layout：
  - V3 权重 layout 与原始 DCU MegaMoE 不同；
  - 权重差异在单测/离线准备阶段提前转好；
  - DCU MegaMoE 执行和 benchmark 流程只接收已转换好的 V3 pack5 权重；
  - 不在 runtime/bench 中引入权重处理 kernel、权重重排 kernel 或额外 launch；
  - 除替换 K1/K3 计算 kernel 本身外，不在 `dcu_megamoe_large_opt` 集成中引入额外 kernel；
  - layout/dispatch 必须做增量分流，避免默认路径受 V3 pack5 假设影响。
- 同步与功能：
  - K1 前 rank barrier kernel 第一版 V3 先保留；
  - correctness 和功能跑通后，再独立 A/B 尝试消掉 rank barrier，并用跨 rank 可见性和性能数据决定是否保留；
  - uneven tokens 与 cuda graph 最终需要对齐现有 DCU MegaMoE 功能；
  - 单测少改动，外部测试入口和参数尽量复用；V3 权重只在 setup/fixture 阶段提前准备。
- 性能门槛：
  - tokens per rank `<512`：V3 LL 要快于 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0`；
  - tokens per rank `>=512`：V3 normal 要快于 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1` 且 `USE_MEGAMOE_V3=0`；
  - Phase 6 固定 bench 档位：32/128 跑 LL，1024/4096 跑 normal。
- baseline 角色：
  - baseline 只作为 correctness oracle；
  - 性能相对 baseline 更好不具有决定意义；
  - baseline timing 可以记录作参考，但不参与 Phase 6 是否通过的判断。

## pure normal/LL 到 fused V3 的扩展原则

- pure 5pack C normal/LL kernel 本身不承担跨 rank 通信语义：
  - K1 pure 更接近本地/已分组输入上的 L1 groupgemm；
  - K3 pure 更接近本地 rowptr/combine 已准备好的 L2 groupgemm + reduce；
  - 通信相关的 symmetric-buffer peer 读取、dispatch-pull、route metadata、combine pointer、tail signal 等需要从现有 DCU MegaMoE staged fused 代码中抽取合同。
- V3 的核心不是“直接替换成 pure”，而是：
  - 保留 pure 5pack C groupgemm inner core；
  - 在 K1 外层补齐 dispatch-pull、row metadata、`row_combine_ptrs`、`route_weights`、`m_indices`、`output_index`、stats；
  - 在 K3 外层补齐 combine pointer 写回、no-tail external reduce 兼容、tail-reduce fused 语义、跨 rank 完成/可见性；
  - 不新增单独通信 kernel，通信语义必须进入 K1/K3 替换 kernel 或复用原已有 barrier/reduce。
- pipeline 融合策略：
  - 由于 C 实现相对 ASM 更容易调整，首选把通信操作藏进 GEMM pipeline，而不是在 groupgemm 前后挂独立步骤；
  - 通信融合不能破坏原 5pack C kernel 内已有 GEMM 流水；
  - load/compute/store 的核心调度、tile 组织和数据复用路径应尽量保持不变；
  - K1 的 dispatch-pull、route metadata、row pointer 生成应尽量融入 global load、route/tile scheduling 和 staged input 准备；
  - K3 的 combine 写回、tail reduce、完成信号应尽量融入 epilogue、store 和本地 reduce 阶段；
  - 如果通信退化为独立前处理/后处理，会引入额外 launch 或内存 pass，直接稀释 pure 5pack C core 的性能优势；
  - 如果为了通信融合重写或打乱原 GEMM pipeline，风险比通信逻辑本身更高；除非有明确性能证据，否则应回到保护原流水的方案。
- 性能控制思路：
  - 把 pure-vs-fused 劣化拆成 K1 dispatch/metadata、K1 groupgemm、K3 groupgemm、combine write、tail reduce 几块；
  - K1/K3 fused 的实际 staged 通信链路必须持续对照对应原始 C 5pack groupgemm/pure kernel；LL 和 normal 都要记录分项 delta，并把明显差距作为优先优化对象；
  - 优先避免额外 global memory pass、额外 launch、runtime 权重处理和大规模中间重排；
  - normal/LL 都要保留 pure 中已经证明有效的 5pack C 访问模式和 tile 组织；
  - 如果中途 K1/K3 fused 相对 pure 劣化明显，需要提前定位并优化，不等 Phase 6 才处理；
  - 最终是否达标仍以 Phase 6 的 tokens 分段性能门槛为准。

## DCU KB 初始检索结论

- 检索主题：Hygon gfx938 MegaMoE staged fused K1 dispatch pull / K3 combine tail reduce replacement。
- 返回的 Hygon 知识主要强调：
  - gfx936/gfx938 上为热点形状保留专门路径是合理的；
  - Hygon GEMM/MoE kernel 需要保留架构特化、builtin/ASM 证据和 shape 专用分派；
  - 替换 ASM 路径时应以代码生成 ISA/正确性/同步语义为证据，而不是只看源码意图。
- 对 V3 计划的含义：
  - V3 不做泛化 kernel，继续锁定 DeepSeek-V4-Flash EP8 形状；
  - 先功能对齐，再性能对照；
  - tail reduce / rank signal 是高风险同步点，必须单独验证。

## 已收敛的 bench 档位

- LL：tokens per rank 32、128。
- normal：tokens per rank 1024、4096。
- 512 作为边界语义，不作为固定性能 gate 档位；需要时可用于额外诊断。

## 2026-06-10 Phase 0 静态接口复核

### K1 staged ASM wrapper/ext 真实合同

- Python 入口：
  - `k1_symm_fused_l1_asm()` 用 eager `num_tokens`；
  - `k1_symm_fused_l1_asm_graph()` 用 `graph_max_tokens`，并传入 capture-safe `runtime_num_tokens`；
  - 两者都要求 EP8、experts=256、topk=6、hidden=4096、alignment=256，且 token 数不能超过 `sym_buffer.num_max_tokens_per_rank`。
- K1 ext 在 `sym_buffer.route_scratch` 内自行切 metadata/workspace，不由 `large_opt.py` 预切：
  - route header：`counts[32]`、`tile_bases[33]`、`tile_experts[capacity_tiles]`、`expert_tile_to_compact[capacity_tiles]`、`capacity_tiles * 16` stage flags；
  - `row_combine_ptrs`: int64 `[capacity_rows + 512]`，512 padding 用于 K3 vectorized store 安全；
  - `route_weights`: fp32 `[capacity_rows]`；
  - `row_x_ptrs`: int64 `[capacity_rows]`；
  - `row_x_scales`: fp32 `[capacity_rows]`；
  - `m_indices`: int32 `[capacity_rows]`；
  - `output_index`: int32 `[num_ranks * num_max_tokens_per_rank, topk]`；
  - `staged_x`: fp8 `[capacity_rows, hidden]`，放在 route task workspace 和 metadata 之后；
  - `staged_flags` / `meta_flags` 放在 `staged_x` 后，并由 graph 前 rank barrier reset。
- K1 launch 的 `GpuProb` 额外合同：
  - `staged_x` at `+0x80`，`staged_flags` at `+0x88`，`symm_base` at `+0x90`，`local_sym_buffer` at `+0x98`；
  - `reserved_c0` 是模式 bitfield：bit0 compact prebuild，bit1 absolute row pointers，bit2 asm-route stores `{rank-local x offset, source rank}`；
  - `route_weights/output_index/row_combine_ptrs/meta_flags/local_expert_stats` 分别传给 ASM；
  - `KernelArgs.m_indics` 指向 `m_indices`，`KernelArgs.row_x_offsets` 复用 `row_x_ptrs` 存储。
- K1 graph 合同：
  - `k1_graph_flag_reset_layout()` 复算 compact-capacity 布局，返回 `flags_offset/flags_numel/meta_flags_offset/meta_flags_numel/total_rows/fixed_capacity_tiles`；
  - graph rank barrier 在 K1 前清零 flags/meta flags，并把 `sym_buffer.cuda_graph_num_tokens` clamp 后写入 `state.scratch.graph_runtime_num_tokens` 和 sym-buffer runtime-token slot；
  - K1 graph launch 固定 `force_compact_prebuild=True`，用 runtime token pointer 控制真实 token 数。
- 迁移含义：
  - V3 K1 不能只替换一个 ASM code object；必须复用或复制 K1 ext 的 scratch allocator、graph reset layout、`row_combine_ptrs` padding、`output_index` capacity 和 runtime-token 合同；
  - 第一版保留 K1 前 `rank_barrier()` 是必要的，因为它既做 cross-rank input visibility，又在 graph/tail path 清理 flags 和 signal。

### K3 staged ASM wrapper/ext 真实合同

- Python 入口 `k3_l2_fused_asm_to_combine()` 强制要求 `output_workspace` 和 `prob_storage`：
  - no-tail：调用 `k3_l2_combine_asm_out()`，K3 GEMM 输出通过 `row_combine_ptrs` 写入 combine buffer；
  - tail-reduce：调用 `k3_l2_combine_asm_tail_reduce_out()`，额外传 `asm_done_counter`、`asm_signal_addrs`、`asm_reduce_y`、`sym_buffer`、token/shape metadata；
  - graph path 可传 `active_tiles` 和 `graph_runtime_offset_from_active_tiles`。
- K3 ext `GpuProb` 合同：
  - `scaleA/scaleB` at `+0x70/+0x78`；
  - tail signal: `asm_done_counter` at `+0x80`、`asm_signal_addrs` at `+0x88`、`asm_done_target` at `+0x90`；
  - tail reduce: `asm_reduce_y` at `+0xa0`、`asm_reduce_combine` at `+0xa8`、`asm_reduce_total_vecs` at `+0xb0`、`asm_reduce_blocks` at `+0xc0`；
  - graph: `active_tiles` at `+0xc8`，`graph_reserved_c4` 存从 `active_tiles` 到 runtime token 的 byte offset。
- no-tail 语义：
  - `prob.d` 指向 `row_combine_ptrs`，ASM epilogue 用 per-row pointer 写 remote/local combine buffer；
  - launch 后仍需 `rank_barrier()`，再调用 `reduce_local_combine()` 把本地 combine buffer reduce 到 `y`。
- tail-reduce 语义：
  - `prob.d` 仍指向 `row_combine_ptrs`；
  - `prob.asm_reduce_combine` 由 `combine_token_offset(...)` 计算到 sym-buffer combine 区；
  - `asm_done_target = ceil(rows/256) * ceil(hidden/256)`，global work items 是 `gemm_workgroups + reduce_workgroups`；
  - kernel 内完成 combine store 后通过 system-scope signal 协调，再 reduce 到 `asm_reduce_y`；
  - graph tail path 用 `active_tiles` 加 runtime-token offset 控制真实 token 范围。
- `build_asm_tail_signal_addrs()` 使用 sym-buffer signal slot `[8, 15]`：
  - `addrs[peer_rank] = peer_signal_ptr + (8 + rank_idx) * 4`；
  - `addrs[8 + peer_rank] = local_signal_ptr + (8 + peer_rank) * 4`；
  - regular rank barrier 使用 `[0, num_ranks)`，local-block barrier 使用 16 和 17，staged rank barrier 使用 18 和 19。
- 迁移含义：
  - V3 K3 no-tail 必须保留 row pointer combine 写回合同，不能退化成直接写 `output_workspace` 后新增后处理；
  - V3 K3 tail 必须把 signal 和 local reduce 合入同一个替换 kernel，否则不能等价替换 `K3COMBINE_TAILREDUCE`。

### V2 C pack5 LL/normal 参考能力与缺口

| 项目 | V2 LL/normal 已有能力 | 与 staged fused V3 的差距 |
| --- | --- | --- |
| backend 选择 | `MEGAMOE_DCU_V2_BACKEND` 默认 `ll`；LL/normal 都有 K1/K3 launcher | V3 需要 `MEGAMOE_DCU_V3_BACKEND`，默认 `normal`，且只在 forced 3-stage large_opt 下生效 |
| weight layout | `flatten_pack5_weight()` 输出 `[expert, n*k]` pack5；执行路径只消费 pack5 | V3 可复用 pack5 layout helper，但需要 V3-owned API/fixture 命名，不能让默认 Marlin layout 或 V2 env 泄漏进 staged path |
| K1 LL | `V2_K1_LowLatencyMaskedGroupGemmKernel<..., true, false>`，支持 dispatch-pull、route_weights、row_expert、output_index、row_combine_ptrs、local_topk_mask、tail_tokens、stats | V2 runtime workspace 不等于 large_opt K1 ASM scratch；需要 stage-owned allocator 或适配到 K1 ext 当前 scratch 合同；还要支持 graph runtime token/reset 合同 |
| K1 normal | `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<..., K=4096,N=4096,...>`，支持 normal row-stage metadata | 需要迁入 K1_fused 并去掉 V2 独立 runtime 依赖；保持 K1 前 rank barrier 和 large_opt K2/K3 输出合同 |
| K3 LL | 低延迟 rowptr tail-reduce launcher，直接用 `row_output_ptrs` 写/reduce 到 `y` | 目前是 tail-reduce 风格；V3 还要补 no-tail combine-only 合同，支持 staged external `reduce_local_combine` |
| K3 normal | copy-stage + tail-reduce，使用 `l2_workspace`、`row_expert`、`local_topk_mask`、`tail_tokens`、`grid_barrier` | 需要补 staged K3 no-tail `row_combine_ptrs` combine-only 路径，并对齐 ASM tail signal/graph active_tiles 语义 |
| workspace | V2 workspace 有 `staged_x/staged_x_scale/problem_size/grid_barrier/local_topk_mask/tail_tokens` 等 pipeline metadata | large_opt 当前只预切 K1/K2/K3 通用区域，K1 ASM ext 另行切 metadata；V3 应在 K1_fused/K3_fused 内 stage-owned 管理，不把 V2 runtime 作为生产路径 |
| graph/uneven | V2 支持 `dispatch_num_tokens=-1` 从 sym-buffer runtime token 读 uneven/graph-like token | large_opt graph 需要 `active_tiles`、runtime token offset、K1 flags reset 和 capture-safe launch arg storage，不能直接复用 V2 eager contract |

### DCU KB 同步语义检索

- 初次并行检索 `dcu-rag-kb-query` / `dcu-rag-kb-optimize` 30s 超时；改为更窄 query 后成功。
- 命中 Hygon `hygon-extend` / flux reduce-scatter 与 custom allreduce 参考：
  - system-scope signal 前使用 `__threadfence_system()` 或 `fence.acq_rel.sys`；
  - signal 通常用 `atomicAdd_system` 或 system-scope store/atomic；
  - wait 侧轮询 system-visible flag，部分参考在 signal 前后用 `__syncthreads()` 保证 block 内一致性。
- 对 V3 的含义：
  - K3 tail-reduce 的 combine store 完成信号必须保守保持 system-scope fence + system-scope signal；
  - 如果后续尝试移除 K1 前 rank barrier或放宽 tail signal，必须用 correctness、cross-rank 可见性和 profiler/ISA 证据支撑；
  - 通信隐藏应优先在 epilogue/store/reduce 内插入 fence/signal，不应把 signal 提前到 combine store 可见性之前。

## 2026-06-10 Phase 1 gate/backend 实现记录

- 新增 `megamoe/dcu_megamoe_large_opt/v3_config.py`：
  - `USE_MEGAMOE_V3` 只解析 `1/true/yes/on` 为 true；
  - `v3_enabled()` 同时要求 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE` 是 forced value：`1/true/yes/on/large_opt/3stage`；
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=auto/threshold/adaptive` 即使 `USE_MEGAMOE_V3=1` 也不会启用 V3；
  - `MEGAMOE_DCU_V3_BACKEND` 支持 `ll|normal`，未设置或空值默认 `normal`，非法值只在 V3 branch 查询 backend 时抛错。
- `large_opt.py` 新增 `_selected_v3_backend()`：
  - 返回 `None` 表示旧 ASM path；
  - 返回 `ll` 或 `normal` 时，K1/K3 call site 选择 V3 wrapper；
  - eager 与 graph 两条 staged flow 都接入同一 gate。
- K1/K3 wrapper 新增 V3 dispatch point：
  - `k1_symm_fused_l1_v3()` / `k1_symm_fused_l1_v3_graph()`；
  - `k3_l2_fused_v3_to_combine()`；
  - 当前 kernel 尚未迁入时显式 fallback 到原 ASM wrapper，保证打开 V3 gate 也不会运行半成品 kernel。
- 新增 `tests/test_dcu_megamoe_v3.py`：
  - 不依赖 GPU 或扩展模块加载 `v3_config.py`；
  - 覆盖 forced-only gate、backend 默认 normal、`ll/normal` 校验、非法 backend 报错；
  - 用 source-level guard 检查 `large_opt.py` 仍通过 `v3_backend is not None` 才选择 V3 wrapper，否则保持 ASM。
- 验证：
  - `python -m compileall ...` 通过；
  - 本地 `pytest` 命令和 `python -m pytest` 均不可用，原因是本地 Python 环境缺少 pytest；
  - 已用 inline Python 执行等价 gate/backend/source 断言，输出 `v3 gate/source assertions passed`。

## 2026-06-10 Phase 1 V3 pack5 weight contract

- 新增 `megamoe/dcu_megamoe_large_opt/v3_layout.py`：
  - 明确 docstring：只用于 offline fixtures / tests；
  - 提供 `pack5_weight()`、`flatten_pack5_weight()`、`unpack_pack5_weight()`、`pack5_shape()`、`pack5_flat_offset()`；
  - 提供 `transform_fp8_weights_for_mega_moe_v3_pack5()`，作为测试/离线 setup 的 BF16 -> FP8 pack5 helper；
  - `large_opt.py` 和执行路径没有导入该模块，避免 runtime/bench 内出现权重转换或 repacking。
- 新增/扩展 `tests/test_dcu_megamoe_v3.py`：
  - CPU 上验证 V3 pack5 helper 与既有 pack5 reference layout 对齐；
  - 只验证 layout helper，不把 V3 layout 隐式接到默认 MegaMoE weight transform。
- 验证：
  - `python -m compileall megamoe/dcu_megamoe_large_opt/v3_layout.py tests/test_dcu_megamoe_v3.py` 通过；
  - inline Python layout 对照输出 `v3 layout assertions passed`；
  - 最终 inline Python 综合断言输出 `v3 local assertions passed`，同时确认 `large_opt.py` 不导入 `v3_layout`；
  - `git diff --check` 通过。

## 2026-06-10 Phase 2 K1 build/static entry

- `setup.py` 当前 build 结构：
  - large-opt K1 extension 名称为 `megamoe.dcu_megamoe_large_opt.K1_fused.k1_fused_ext`；
  - 现状只编译 `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu`；
  - V2 K1 extension 则编译 `K1_fused/k1_fused_pybind.cpp` + `K1_fused/k1_fused_ext.cu`，后者通过 `#include "../../../csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp"` 复用 V2 kernel 主体。
- 对 V3 K1 迁移的含义：
  - 最小扰动方案是在 existing large-opt K1 extension module 内追加 V3 launcher/raw kernel source，而不是新建一个 Python import path；
  - `PYBIND11_MODULE` 应继续只存在于 large-opt `k1_fused_ext.cu`，新增 stage-owned `.cu` 只提供 raw launcher symbol；
  - `setup.py` 需要把新增 stage-owned K1 V3 source 加入同一个 large-opt K1 extension 的 `sources`；
  - 不能让 V3 生产实现继续 include 或依赖 `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`，否则违反“不要继续把生产实现堆到 K1_groupgemm_v2.cpp”的边界。
- V2 K1 pybind/raw launcher 对照：
  - LL raw launcher 调用 `V2_K1_LowLatencyMaskedGroupGemmKernel<32,4096,4096,..., kUseSymmStage=true, kUseRowPtrs=false>`；
  - normal raw launcher调用 `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16<256,256,true,true,4,4096,4096,...>`；
  - 两者都输出 `route_weights`、`row_expert_out/m_indices`、`output_index`、`row_combine_ptrs`、`local_topk_mask`、`tail_tokens` 和可选 local expert stats。
- staged large-opt K1 allocator 对照：
  - 原 ASM wrapper 已在 `k1_fused_ext.cu` 内切出 `row_combine_ptrs`、`route_weights`、`row_x_ptrs`、`row_x_scales`、`m_indices`、`output_index`、`staged_x` 和 graph flags；
  - V3 K1 应复用这套 scratch layout 和返回 tuple，避免外层 `large_opt.py`、K2、K3 再适配一次；
  - V3 pack5 weight shape 将不同于 ASM Marlin shape，C++ wrapper 需要单独校验 `[local_experts, 4096 * 4096]` flat pack5 或等价 contiguous view。
- 额外观察：
  - 静态搜索时传入 `pyproject.toml`，仓库不存在该文件导致 `rg` exit code 1；搜索结果仍确认 `setup.py` 是当前扩展 build 入口。

## 2026-06-10 Phase 2 K1 stage-owned source migration

- 已新增 stage-owned K1 V3 源：
  - `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_groupgemm_impl.cuh`：从 V2 C pack5 K1 source 机械复制而来，作为 V3 后续改动的本地实现体；
  - `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_fused_ext.cu`：定义 `dcu_megamoe_v3_launch_k1_ll_symm_stage_raw()` 和 `dcu_megamoe_v3_launch_k1_normal_symm_stage_raw()`；
  - `setup.py` 将 `k1_v3_fused_ext.cu` 加入同一个 large-opt K1 extension，并将 `*.cuh` 加入 K1 package data。
- 当前迁移状态：
  - V3 raw launcher 仍调用复制来的 `V2_K1_LowLatencyMaskedGroupGemmKernel` 和 `V2_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16` 模板名；
  - 这是 stage-owned copy 的第一步，不再 include `csrc/kernels/dcu_megamoe_v2/k1_groupgemm_v2.cpp`；
  - Python `k1_symm_fused_l1_v3()` 仍显式 fallback 到 ASM，尚未调用 raw launcher。
- scratch/layout 关键差异：
  - V2 LL/normal 都要求自己的 `route_scratch_i32`：`symm_counts[32] + symm_src_ranks[32 * rows_aligned_per_expert] + symm_src_tokens[32 * rows_aligned_per_expert]`；
  - V2 normal 还要求 `grid_barrier` 至少按 `_v2_grid_barrier_ints`: `16 * ceil(launch_rows/256) + 2`，而不是旧 pybind check 中的较小值；
  - 因此 V3 K1 wrapper 不能直接把 `route_scratch.data_ptr()` 传给 raw launcher，必须在现有 `route_scratch` 内切出 V3 metadata/grid-barrier/local-mask/tail-token 视图，再返回原 ASM-compatible tuple。
- 同步语义检索：
  - 使用 `dcu-rag-kb-query` 检索 Hygon/DCU in-kernel grid barrier reuse；命中 DeepEP `internode_ll.cu` 风格 grid barrier，核心假设仍是 caller 提供已初始化 counter；
  - 结论：不要用额外 memset/zero kernel 初始化 V3 grid barrier；后续应把 barrier state 初始化或 epoch 化并入 K1 V3 kernel/metadata flow，或者证明 route_scratch 初始状态安全。
- 验证：
  - `git diff --check` 通过；
  - `python -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/v3_config.py megamoe/dcu_megamoe_large_opt/v3_layout.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `tests/test_dcu_megamoe_v3.py` 增加 source-level guard，检查 `setup.py` 包含 `k1_v3_fused_ext.cu` / `*.cuh`，且 V3 launcher/impl 不 include V2 `csrc` source；
  - 远端容器内 `python3 -m compileall ...` 通过；
  - 远端 `setup.py build_ext --inplace` 尚未完成：第一次失败于 remote checkout 缺少既有 V2 pybind source；同步 V2 source 后第二次 build 10 分钟超时，后续 SSH status 检查又超时。

## 2026-06-10 Phase 2 K1 V3 low-level wrapper

- 新增 large-opt K1 extension 低层入口 `k1_symm_fused_l1_v3_pack5()`：
  - 暴露在 `megamoe.dcu_megamoe_large_opt.K1_fused.k1_fused_ext`；
  - Python public `k1_symm_fused_l1_v3()` 仍保持 ASM fallback，暂不调用该入口；
  - 入口用于下一步 K1 metadata/unit correctness，不进入 production eager path。
- wrapper 合同：
  - 输入权重要求 V3 pack5 FP8，覆盖 `[local_experts, 4096 * 4096]`，scale 为 `[local_experts, 4096]`；
  - shape 继续锁定 EP8、experts=256、local_experts=32、topk=6、hidden=4096、alignment=256；
  - 返回 tuple 对齐 ASM：`l1_out, route_weights, m_indices, output_index, row_combine_ptrs`；
  - `output_index` 仍按 `[num_ranks * num_max_tokens_per_rank, topk]` 分配，保证 K3/外层合同不缩容。
- scratch 规划：
  - 在 `route_scratch` 的 task-workspace 前段切出 V3-only `route_scratch_i32`、`grid_barrier`、`local_topk_mask`、`tail_tokens`、`row_combine_ptrs`、`route_weights`、`m_indices`、`output_index`、`staged_x_scale`；
  - `staged_x` 放在 `route_workspace_bytes` 后的 large-opt reserved FP8 x region，不覆盖 `large_opt.py` 已暴露的 `l1_out`、K2 `act_fp8/act_scale` 或 K3 prob storage；
  - `row_combine_ptrs` 继续带 `kK1RowPointerPadding=512` padding。
- 当前同步/初始化取舍：
  - 低层入口暂用 `hipMemsetAsync(grid_barrier, 0, ...)` 保证 V2-derived grid barrier unit bring-up 可复现；
  - 该入口未接入 production Python path；后续真正启用 V3 K1 前，需要将 barrier 初始化替换为 epoch/in-kernel init，避免执行路径引入额外 memory op；
  - DCU KB 已提示 grid barrier 类实现通常假设 counter 初值已初始化，所以这部分必须在 correctness 前单独验证。
- 验证状态：
  - 本地 `python -m compileall tests/test_dcu_megamoe_v3.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py` 通过；
  - 本地 source guard 输出 `k1 v3 low-level entry guard assertions passed`；
  - `git diff --check` 通过；
- 远端 `10.17.176.13:22` 当前连续 SSH timeout，尚未完成 DTK 编译确认。

## 2026-06-10 remote workflow correction

- 本轮用户纠正远端工作流参数后，重新核对：
  - `.vscode/sftp.json` 当前为 `hg@10.17.176.11:22`；
  - host repo 为 `/home/hg/yuguo/DeepGEMM`；
  - Docker container 为 `sglang_megamoe`；
  - host `/home/hg/yuguo` bind 到 container `/workspace`，container repo 为 `/workspace/DeepGEMM`。
- `.codex/skills/remote_work/SKILL.md` 中残留的 `megamoe` container 与 `DeepDEMM` 路径已修正为 `sglang_megamoe` / `DeepGEMM`，避免后续恢复上下文时误用旧节点或旧容器。
- 远端 `/workspace/DeepGEMM` 当前 git branch 显示为 `main`，更像编译工作区；本轮用显式 sync 覆盖本地相关文件，不依赖远端分支状态。
- 远端 checkout 起初缺少 V2 package/reference files，导致 `tests/test_dcu_megamoe_v3.py` 的 V2 pack5 layout reference 读取失败；已将本地 `megamoe/dcu_megamoe_v2` 与 `csrc/kernels/dcu_megamoe_v2` 必要文件同步到远端工作区。
- 在 `sglang_megamoe` 中执行：
  - `python3 -m compileall megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/v3_config.py megamoe/dcu_megamoe_large_opt/v3_layout.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 在补齐 V2 reference 后通过，结果 `5 passed`。
- `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 已在 `sglang_megamoe` 内启动；当前 `dcc -cc1` 正在编译 `k1_v3_fused_ext.hip`，尚未得到最终 build 结果。

## 2026-06-10 Phase 2 K1 default build recovery

- 长时间 build 异常定位：
  - `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 卡在 `k1_v3_fused_ext.hip` 的 `dcc -cc1` codegen；
  - 该 TU 只有少量 launcher 代码，但默认 include 了 4800 行级别的 `k1_v3_groupgemm_impl.cuh`，并实例化多个 LL/normal 重模板；
  - 观察到 `dcc -cc1` 长时间 99% CPU、目标 `.o` 仍为 0 bytes，属于模板 codegen 爆炸，不是 Python build 逻辑卡住。
- 修复取舍：
  - V3 权重 layout 仍强制 pack5，`k1_symm_fused_l1_v3_pack5()` 的输入和 shape check 不变；
  - `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` 只表示显式编译当前还在 low-level bring-up 的 K1 raw template kernels；
  - 默认 build 只编译轻量 raw launcher 符号和可解释报错，避免未接 production 的重模板 TU 阻塞常规扩展构建；
  - 避免使用 `PACK5` 作为 build flag 名称，以免误导为 V3 layout 可选。
- 验证：
  - 本地 `python -m compileall ...` 通过；
  - 本地 source assertions 确认旧 `DG_BUILD_MEGAMOE_V3_PACK5` / `DCU_MEGAMOE_V3_ENABLE_PACK5_KERNELS` 已移除，V3 pack5 入口仍存在；
  - 远端 `sglang_megamoe` 中 `python3 -m compileall ... && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `5 passed`；
  - 远端默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，返回码 `0`；
  - 该 build log/status 已归档到 `hygon_tmp/sglang_debug/deepgemm_v3_build_default_20260610_144010.log` 和 `.status`。

## 2026-06-10 Phase 2 K1 core source correction

- 用户纠偏：
  - 不要使用 `dcu_megamoe_v2` 下的 K1 实现作为 V3 core，因为该融合实现性能很差；
  - K1 V3 应使用 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 中原始 pure K1 5pack C 实现，再按计划扩展 dispatch-pull 通信语义。
- 对当前代码的影响：
  - 先前 `k1_v3_groupgemm_impl.cuh` 从 V2 real-flow source 机械复制，方向错误，必须从 V3 K1 core 路径移除；
  - `k1_v3_fused_ext.cu` 不应 include V2-derived impl，也不应出现 `V2_DeepGemm...` / `V2_K1...` kernel 名；
  - `tests/test_dcu_megamoe_v3.py` 需要增加 source guard，明确 V3 K1 core 来源是 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 迁入的 stage-owned pure kernel，而不是 V2 source。
- 对计划的影响：
  - Phase 2 增加“移除 V2-derived K1 core 依赖”作为当前进行项；
  - Phase 2 K1 stage-owned 迁入项改为从 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 迁入原始 pure normal/LL kernel；
  - 后续通信扩展必须保护该 pure kernel 的 LDS/load/compute/store pipeline，不以 V2 real-flow 包装为性能基线。

## 2026-06-10 原始 ASM 的参考边界

- 用户补充：
  - `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s` 是原始 DeepGEMM 汇编文件；
  - 原始 DCU MegaMoE 的 K1 fused / K3 fused ASM 都是在该原始 GEMM ASM 基础上扩通信得到；
  - V3 K1/K3 fused 扩通信时可以参考这些 ASM 的通信语义、插入点、wait/signal 处理；
  - 但 V3 仍必须尽量不破坏 pure C 5pack GEMM 骨干流水，并尽量把通信隐藏到计算 pipeline 内。
- 本地 DCU KB 检索：
  - 查询 `hygon gfx938 MegaMoE fused GEMM communication hiding pipeline ASM reference waitcnt`；
  - 命中 `Report_deepgemm_asm_grouped_fp8_bf16_gemm_gfx936_gfx938.md`，确认 `C_groupgemm_fp8` 的主形状是 `MT256x256x128`、`n/k=4096`、`experts=32`、`topk=6`，与当前 K1 pure source 对齐；
  - 同时命中 Hygon communication-compute fusion 汇总，提示通信融合应按通信模式和可见性能力显式分支，不能默认所有通信写回都无成本或总是有利。
- 对 V3 的含义：
  - ASM 是 fused 语义和同步时序参考，不是 V3 C kernel 主体；
  - K1 先从 ASM 抽取 dispatch-pull/route metadata/可见性要求，再嵌入 pure C 的 load/route/tile scheduling；
  - K3 后续从 ASM 抽取 combine/tail-reduce/signal 语义，再嵌入 pure C 的 epilogue/store/reduce；
  - 后续任何 waitcnt、inline asm 或 signal 调整都需要本地 KB/ISA/正确性证据支撑。

## 2026-06-10 K1 pure source correction validation

- 本地 V3 K1 source guard：
  - `megamoe/dcu_megamoe_large_opt/K1_fused/k1_v3_groupgemm_impl.cuh` 已不存在；
  - `setup.py` 不再引用旧 `k1_v3_groupgemm_impl.cuh`，raw pure TU 为 `k1_v3_pure_ext.cu`；
  - `k1_v3_fused_ext.cu` 只保留 raw availability/stub；
  - `k1_v3_pure_groupgemm_impl.cuh` / `k1_v3_pure_ext.cu` / V3 low-level ext source 中不再出现 `V2_`、`dcu_megamoe_v2` 或 `V2-derived`。
- pure source 对齐：
  - normal kernel body 与 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 中 `DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16` 段逐字一致，仅 kernel 名替换为 `V3_K1_Pure_...`；
  - LL kernel body 与 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 中 `K1_LowLatencyMaskedGroupGemmKernel` 段逐字一致，仅 kernel 名替换为 `V3_K1_Pure_...`；
  - harness-only helper、host-side fixture、ASM launch harness 没有迁入 V3 production source。
- 本地验证：
  - `python -m compileall setup.py tests/test_dcu_megamoe_v3.py megamoe/large_opt.py megamoe/dcu_megamoe_large_opt/v3_config.py megamoe/dcu_megamoe_large_opt/v3_layout.py megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused.py megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py` 通过；
  - source guard 输出 `v3 k1 pure source guard passed`；
  - `git diff --check` 通过；
  - 新文件检查为 LF、无 UTF-8 BOM。
- 远端状态：
  - 已在本地 `hygon_tmp/sglang_debug/` 准备单 tar 同步包，避免逐文件 scp 触发 SSH 连接重置；
  - 11 节点 SSH 当前返回 `Connection refused`，远端 pytest/default build/raw TU 编译 probe 尚未补跑。

## 2026-06-10 K1 raw pure compile-time issue

- 现象：
  - 默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 已在 11 节点 `sglang_megamoe` 中通过，返回码 0；
  - 但显式 `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` 的 raw bring-up probe 在 `k1_v3_pure_ext.hip` device codegen 上运行 10 分钟以上；
  - `ps` 显示 `/opt/dtk/dcc/bin/dcc -cc1 ... k1_v3_pure_ext.hip` 占用 99% CPU，说明卡在 DTK device codegen，不是 Python/ninja 卡住。
- 原因判断：
  - `k1_v3_pure_ext.cu` 单 TU 同时实例化 pure normal 256x256 和 LL 多个 `BLOCK_M` 变体；
  - 这些不是小 kernel，而是包含大量 inline device helper、raw buffer load、MMAC、unrolled GEMM pipeline 的重模板；
  - 对 DTK/DCC 来说，一次编译所有 raw 变体不适合作为常规验证路径。
- 已采取：
  - 手动终止该 raw probe；日志位于 `hygon_tmp/sglang_debug/deepgemm_v3_pure_single_tu_build_raw_20260610_150922.log`，status 因中断未写出；
  - 新增 `DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal|ll|all`；
  - raw flag 开启时默认只编 `normal`，显式 `ll` 才编 LL，只有需要全量 stress 时才用 `all`；
  - `k1_v3_pure_ext.cu` 用 `DCU_MEGAMOE_V3_ENABLE_K1_RAW_LL` / `DCU_MEGAMOE_V3_ENABLE_K1_RAW_NORMAL` 条件化实例化；
  - low-level wrapper 增加 per-backend availability 检查，避免调用未编译 backend。
- 结论：
  - 默认路径保持轻量，不再被 raw pure GEMM 模板拖慢；
  - raw bring-up 后续必须按 backend/变体分开编译、分开记录时间和结果。

## 2026-06-10 K1 raw normal backend probe

- 远端同步：
  - 使用 `.vscode/sftp.json` 的 `hg@10.17.176.11:22` 和容器 `sglang_megamoe`；
  - 本地改动通过 `hygon_tmp/sglang_debug/deepgemm_v3_sync_20260610_153518.tar` 同步到 `/home/hg/yuguo/DeepGEMM`；
  - 容器内验证 `DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND` 已存在，且旧 `k1_v3_groupgemm_impl.cuh` 不存在。
- 远端验证：
  - `python3 -m compileall ... && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `6 passed`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_backend_scoped_pytest_20260610_153558.log`；
  - 默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，日志 `hygon_tmp/sglang_debug/deepgemm_v3_backend_scoped_build_default_20260610_153621.log`。
- raw normal probe：
  - 命令使用 `timeout 240s env DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal DG_FORCE_BUILD=1 MAX_JOBS=2 python3 setup.py build_ext --inplace`；
  - 结果 status 为 `124`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_raw_normal_build_20260610_153737.log`；
  - 日志显示 K1 extension build 停在 `[1/3] k1_v3_fused_ext.cu` 和 `[2/3] k1_fused_ext.hip`，尚未完成 full extension link；没有残留 `setup.py` / `hipcc` / `dcc` 进程。
- 结论：
  - backend-scoped raw flag 已避免默认 build 变慢，但 raw bring-up 仍不应走全量 `setup.py build_ext`；
  - 下一步应把 raw pure normal 编译/验证拆成更小的 targeted TU 或单独 build harness，先只编 `k1_v3_pure_ext.cu` 及必要 launcher，避免被全量扩展 rebuild 成本掩盖。

## 2026-06-10 V2 编译超时历史复核

- 用户指定历史 session `019e6ecc-aaed-74e1-aa6e-78b8ee3133f3` 后，已在本地 codex session 记录中复核 V2 当时的处理方式。
- 关键结论：
  - V2 的 K1/K3 extension compile flags 使用了 `-mllvm -enable-num-vgprs-768=true`；
  - V2 还建立过独立 `csrc/kernels/dcu_megamoe_v2/Makefile` 与 `scripts/build_dcu_megamoe_v2.sh`，用于 raw/pure kernel bring-up，避免每次都走完整 Python extension rebuild；
  - 历史记录中 V2 raw/source 变更后的 compile+run 约 60 秒量级；binary up-to-date 后运行更快。
- 与当前 V3 的差异：
  - V3 large-opt K1 raw gate 已经 backend-scoped，但 `large_opt_k1_hipcc_flags` 原先只带 `-DNDEBUG`，没有 V2 extension 使用的 VGPR codegen 限制；
  - 这会让 raw pure normal `MT256x256x128` 重模板更容易在 DTK/DCC codegen 阶段长时间卡住。
- 本轮修正：
  - `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1` 时，V3 large-opt K1 extension flags 现在追加 `-mllvm -enable-num-vgprs-768=true`；
  - 默认 build 不开启 raw flag，所以常规 edit/build loop 仍保持轻量；
  - `tests/test_dcu_megamoe_v3.py` 增加 source-level guard，防止该 V2 编译保护再次丢失。
- 验证结果：
  - 远端 `python3 -m compileall ... && PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，结果 `6 passed`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_vgpr_pytest_20260610_161410.log`；
  - 远端默认 `DG_FORCE_BUILD=1 MAX_JOBS=8 python3 setup.py build_ext --inplace` 通过，status `0`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_vgpr_build_default_20260610_161431.log`；
  - 远端 raw normal `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal DG_FORCE_BUILD=1 MAX_JOBS=2 python3 setup.py build_ext --inplace` 在 240s hard timeout 内通过，status `0`，日志 `hygon_tmp/sglang_debug/deepgemm_v3_vgpr_raw_normal_20260610_161554.log`；
  - import smoke 确认 `megamoe.dcu_megamoe_large_opt.K1_fused.k1_fused_ext` 暴露 `k1_symm_fused_l1_v3_pack5`，且无残留 `setup.py`/`hipcc`/`dcc` 编译进程。

## 2026-06-10 V2 编译边界收紧

- 用户明确：V2 不要管、不要编，当前 V3 工作中把 V2 当作不存在。
- 本轮调整：
  - `setup.py` 新增/保留 `DG_BUILD_MEGAMOE_V2_EXT`，但默认值为 `0`；
  - 因此默认 `python3 setup.py build_ext --inplace` 只编主扩展与 `dcu_megamoe_large_opt` K1/K2/K3 extension，不再编译 `megamoe.dcu_megamoe_v2.*` extension；
  - 如果将来为了历史对照确实需要 V2，必须显式 `DG_BUILD_MEGAMOE_V2_EXT=1`，不作为 V3 常规验证路径。
- aicc 相关观察：
  - `/opt/dtk/bin/aicc` 对历史 Makefile 的 `-mllvm -enable-num-vgprs-768=true` 报 unknown option；
  - `/workspace/dtk_aicc/bin/aicc` 搭配 `ROCM_PATH=/workspace/dtk_aicc HIP_PATH=/opt/dtk/hip HIP_CLANG_PATH=/workspace/dtk_aicc/aillvm/bin HIP_ROCCLR_HOME=/opt/dtk DEVICE_LIB_PATH=/opt/dtk/dcc/dccgcn/bitcode` 可接受该 flag；
  - 用 `/workspace/dtk_aicc` 编译 V3 K1 pure normal object 已通过，产物 `hygon_tmp/sglang_debug/k1_v3_pure_ext_aicc.o`。

## 2026-06-10 V3 normal backend aicc policy

- 用户澄清后的编译规则：
  - V3 K1 fused normal backend 必须用 aicc；
  - V3 K3 fused normal backend 必须用 aicc；
  - 其他路径没有强制要求，可继续使用原 hipcc/现有 setup 逻辑。
- 本地 KB 与 pure K1 README 一致：
  - small-token LL 推荐 hipcc；
  - large-token normal C pack5 路径推荐 aicc；
  - 关键 flags 仍为 `--offload-arch=gfx938`、`-O3`、`-DNDEBUG`、`-mllvm -enable-num-vgprs-768=true`。
- `setup.py` 现有实现：
  - `DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1` 默认开启；
  - `_mark_v3_normal_aicc(ext)` 给 V3 normal extension 加 marker；
  - `CustomBuildExt.build_extension()` 只在该 marker 存在时临时把 `torch.utils.cpp_extension.ROCM_HOME` 切到 repo 内 shim；
  - shim 路径为 `hygon_tmp/sglang_debug/v3_aicc_rocm/bin/hipcc`，实际 exec `/workspace/dtk_aicc/bin/aicc`；
  - 构建后恢复原 `ROCM_HOME`，所以主扩展、K2、默认 K1/K3 fallback 不被污染。
- 当前验证结果：
  - 默认 build：K1/K2/K3 都走 `/opt/dtk/bin/hipcc`，`unexpected_aicc_marker=0`、`unexpected_raw_pure=0`、`v2_seen=0`；
  - V3 normal raw build：K1 extension 与 K3 extension 都打印 `Building ... with V3 normal aicc shim`，并使用 `v3_aicc_rocm/bin/hipcc -> /workspace/dtk_aicc/bin/aicc`；
  - V2 extension 没有进入构建日志，`v2_seen=0`；
  - 远端日志：`hygon_tmp/sglang_debug/deepgemm_v3_k1k3_normal_aicc_build_20260610_170658.log`。
- 注意：
  - K3 目前还没有 V3 normal C/fused kernel TU；当前只是把 K3 V3 normal extension 编译策略固定下来；
  - Phase 3 接 K3 normal TU 时必须挂到 `large_opt_k3_ext` 及同一 aicc marker 下，不能另开 hipcc-only path。

## 2026-06-10 K1 V3 normal low-level smoke

- low-level smoke 使用 aicc-built K1 normal raw path，在 fake symmetric buffer 上验证了两件事：
  - route metadata 生成有效：`active_routes=3072`、`route_weight_sum=3072.0`、`unique_experts_first_expected=12`；
  - all-ones 输入/权重时 active sample 输出均值为 `4096.0`，说明 normal pure GEMM core 在 low-level V3 入口中实际计算。
- 该结果只证明 normal raw core、pack5 权重输入、metadata 切片和 wrapper 参数基本可跑；还不是 production fused 形态。
- 当前 low-level wrapper 仍存在两个临时 bring-up 成本：
  - `hipMemsetAsync(grid_barrier, 0, ...)` 用于 debug 初始化；
  - `k1_v3_stage_rows_from_ptrs_kernel` 把 `row_x_ptrs/row_x_scales` 复制到 contiguous `staged_x/staged_x_scale`。
- 后续 K1 normal 融合重点：
  - 保持原 pure C 5pack normal kernel 的 tile/pipeline 结构；
  - 把 logical row 到 source row pointer 的映射放进 input B load 阶段；
  - 把 x-scale 读取改成 logical-row indexed scale；
  - normal production path 不再依赖额外 stage rows kernel。

## 2026-06-10 K1 V3 normal fused isolation correction

- 用户指出不要为了复用 pure body 强行保留 pure/fused 两套并行 kernel；如果需要改，可以重写必要部分并做好 V3 fused 隔离。
- 本轮收敛后的结构：
  - `k1_v3_fused_ext.cu` 是显式 raw/fused build gate 下编译的 heavy V3 K1 TU；
  - `k1_v3_stub_ext.cu` 只在默认 build 中提供 availability/stub symbols，避免默认路径编 heavy template；
  - `k1_v3_groupgemm_impl.cuh` 保留原始 K1 pure C 5pack pipeline 派生的实现；
  - normal path 只有一个 `V3_K1_Fused_DeepGemm...` kernel，不再同时保留 `V3_K1_Pure_DeepGemm...` normal kernel。
- normal fused kernel 当前相对原 pure normal 的有意差异：
  - kernel 输入从 contiguous `x/x_scale` 改为 `row_x_ptrs/row_x_scales`；
  - B/input load 从 `x_resource + token * K + phase_k` 改成 per-logical-row `row_x_ptrs[token] + phase_k`；
  - x-scale loader 从 contiguous `x_scale` 改成 logical-row indexed `row_x_scales`；
  - tile shape、stage order、weight load、MMAC loop、scale/store 结构保持来自原 pure normal pipeline。
- 已删除的临时 bring-up 成本：
  - `k1_v3_stage_rows_from_ptrs_kernel` 定义和 launch 已删除；
  - normal launcher 不再接收 `staged_x/staged_x_scale`。
- 仍需处理的非 production 成本：
  - low-level wrapper 仍通过 route init/count/build/emit helper kernel 生成 metadata；
  - 这些仅限 low-level bring-up，不能作为最终 V3 production 执行路径。
- 远端验证：
  - source-level pytest：`6 passed`；
  - 默认 build：不编 V3 heavy fused TU，不走 aicc，V2 未出现；
  - explicit K1 normal aicc build：`k1_v3_fused_ext` 通过 aicc shim 编译；
  - zero/ones smoke：直接 row-ptr load 后 ones active sample 仍为 `4096.0`。

## 2026-06-10 K1 V3 normal padded rows observation

- normal fused kernel 只写每个 expert 的有效 rows；padded rows 仍可能保留 `torch::empty` 的未初始化值。
- 因此 debug smoke 不能用 full `out.abs().max()` 判断 correctness，否则 padded rows 可能出现 NaN 噪声。
- 正确的 low-level smoke 指标应限定在 `output_index` 指向的 active rows：
  - zero case active rows max 为 `0.0`；
  - ones case active sample mean 为 `4096.0`。
- `hipMemsetAsync(grid_barrier)` 已从 normal low-level bring-up 中删除；后续 normal fused path 当前不依赖 grid barrier 初始化。

## 2026-06-10 K1 compact prebuild reuse finding

- 原始 large-opt K1 ASM wrapper 已有 compact prebuild route kernel：
  - `k1_init_compact_routes_kernel`;
  - `k1_count_compact_routes_kernel`;
  - `k1_build_compact_tiles_kernel`;
  - `k1_emit_compact_routes_kernel`。
- 这些 kernel 在 `use_compact_prebuild` 时已经负责生成 `route_weights`、`row_x_ptrs`、`row_x_scales`、`m_indices`、`output_index` 和 `row_combine_ptrs`，因此 V3 normal low-level 不需要另起一套 route build kernel。
- 关键差异：
  - compact prebuild 产出的是 compact tile list，不保证每个 expert 占用固定 `rows_aligned_per_expert` stride；
  - 因此 V3 normal GEMM 不能再用 `tile_token / rows_aligned_per_expert` 推导 expert；
  - 正确消费方式是从 `m_indices[row]` 读取 tile expert，并用 `row_x_ptrs[row] > 0` 判断 active/padded rows。
- 本轮实现后：
  - V3 normal low-level 的 route scratch header/capacity/route-grid 公式对齐原 ASM prebuild；
  - V3 normal GEMM 的 weight expert 来自 `row_expert[tile_token]`；
  - store mask 来自 `row_x_ptrs`，避免 compact padding row 被当成有效 row；
  - 4096 token expert-ramp smoke 验证了 32 个 local experts 的 compact tile expert 映射，`expert_ramp_abs_err=0.0`。
- 设计结论：
  - 复用原 compact prebuild 不违反“不要新增 kernel”的要求，因为它属于原始 DCU MegaMoE K1 ASM 路径已有的 production 机制；
  - 后续 public V3 normal 接入时应复用原 ASM 的 `K1_PREBUILD_MODE` / auto 策略，而不是引入 V3 专用 prebuild 开关；
  - 若后续尝试把 route build 完全内联进 single GEMM kernel，需要另走 correctness/perf/ISA 证据闭环，不能覆盖当前可复用的 compact prebuild baseline。

## 2026-06-10 K3 V3 build/source boundary finding

- K3 V3 normal/LL 需要和 K1 一样做成 stage-owned source boundary：
  - 默认 large-opt K3 extension 只编原 `k3_fused_ext.cu` 和轻量 `k3_v3_stub_ext.cu`；
  - 显式 `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1` 才编 `k3_v3_fused_ext.cu`；
  - `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=normal` 触发 K3 extension 的 V3 normal aicc marker；
  - K1 raw gate 不开时，K3 normal raw build 不应编 K1 heavy TU。
- 远端验证结论：
  - 默认 build：`v2_seen=0`、`aicc_marker=0`、`k3_heavy_seen=0`；
  - K3 normal raw build：`v2_seen=0`、`aicc_marker=1`、`k3_heavy_seen=3`、`k1_heavy_seen=0`；
  - availability：`k3_raw=True`、`k3_normal=True`、`k3_ll=False`。
- 设计含义：
  - 后续 K3 C pack5 core 可以安全放在 `K3_fused` 下的新 V3-owned header/source 中；
  - 在 core 未完成前，public `k3_l2_fused_v3_to_combine()` 保持 ASM fallback，避免 V3 gate 运行半成品；
  - 生产 V3 K3 source 不能 include `dcu_megamoe_v2` 或暴露 `V2_` kernel 名；如需参考 V2，只能抽取合同和可验证的 pure fragment。

## 2026-06-10 V3 主 kernel 派生路径收紧

- 用户再次明确：
  - V3 K1/K3 fused normal/LL 主 kernel 相对 pure C normal/LL 的改动，应主要参考原始 DCU MegaMoE K1/K3 fused ASM 相对原始 groupgemm ASM 的差异；
  - 原始 groupgemm ASM 文件为 `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s`；
  - 差异点预期主要是 weight layout 不同以及通信语义注入；
  - V3 要尽可能复用 DCU MegaMoE 现有周边实现，只有无法复用的边界代码才重写；
  - V2 fused/real-flow 实现不要作为 V3 主 kernel 结构来源，避免被 V2 的低性能融合路径干扰。
- 对当前代码的影响：
  - K1 当前 stage-owned normal core 已按原 pure C 5pack 路线推进，后续补通信时要继续用 original ASM vs K1 fused ASM 的差异图校准 dispatch-pull、metadata、wait/signal 插入点；
  - K3 不能从 V2 C fused/real-flow source 机械提取主 kernel 进入生产路径；
  - 任何已经生成的 V2-derived K3 V3 groupgemm header / launcher 调用都应撤回到边界 stub，等 ASM 差异图和 pure C 主体映射清楚后再实现。
- 后续工作顺序：
  1. 保持 K3 V3 raw TU 只作为显式 build/aicc 边界壳，不执行半成品 kernel；
  2. 对比原始 groupgemm ASM 与 K1 dispatch-pull fused ASM，形成 K1 通信插入点表；
  3. 对比原始 groupgemm ASM 与 K3 combine / tail-reduce fused ASM，形成 K3 epilogue/store/reduce/signal 插入点表；
  4. 再把这些差异映射到 pure C normal/LL 5pack 主体，避免复制 V2 的 copy-stage 或调度结构。

## 2026-06-10 original ASM diff 初版

- 本轮对比文件：
  - original groupgemm ASM: `hygon_tmp/K1_groupgemm_fp8/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16.s`;
  - K1 dispatch-pull fused ASM: `megamoe/dcu_megamoe_large_opt/K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1.s`;
  - K3 combine fused ASM: `megamoe/dcu_megamoe_large_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE.s`;
  - K3 tail-reduce fused ASM: `megamoe/dcu_megamoe_large_opt/K3_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE_TAILREDUCE.s`。
- DCU KB 结论：
  - `Report_deepgemm_asm_grouped_fp8_bf16_gemm_gfx936_gfx938.md` 确认当前 original ASM / pure C 参考形状为 `MT256x256x128`、`n=k=4096`、`experts=32`、`topk=6`、gfx938；
  - 通信融合建议保持控制面稳定，把通信放进 tile scheduler / epilogue / writeback 等自然阶段；wait、signal、system-visible store 需要 correctness、ISA 或 profiler 证据支撑，不能凭直觉放宽。

### K1 original ASM vs dispatch-pull ASM

- K1 fused ASM 在 GEMM 主体之前新增两段主要逻辑：
  - `SYMMROUTE stage`：按 rank/source 分片扫描 symmetric-buffer route 信息，生成 `route_weights`、`m_indices`、`output_index`、`row_x_ptrs`、`row_x_scales`、`row_combine_ptrs` 和 compact tile metadata；
  - `MegaMoE dispatch-pull stage`：16 个 N-tile CTA 协作把 256x4096 输入 tile pull/stage 到 `staged_x`，用 `staged_flags` 的 done counter 和 generation flag 同步，消费者等待后再进入原 GEMM。
- 关键参数/语义：
  - fused K1 额外使用 `staged_x`、`staged_flags`、`symm_base`、`local_sym_buffer`、route scratch/counts、rank、`num_tokens`、`max_tokens_per_rank`、meta flags 和 stats 等参数；
  - dispatch-pull 支持 local/base pointer、absolute wide pointer、rank-local pointer table 三类来源，invalid rows zero-fill；
  - staged pull 侧有 cache invalidation / wait 语义，第一版 V3 继续保留 K1 前 rank barrier，后续只有在跨 rank 可见性证据充分时再 A/B。
- 映射到 V3 pure C 5pack：
  - V3 K1 normal 已经复用原 K1 compact prebuild 生成 metadata，production 第一版不应再造独立 route build；
  - 主 GEMM 当前从 `row_x_ptrs/row_x_scales/m_indices` 直接消费 compact metadata，这是“复用周边 prebuild + row-ptr load”路线；
  - 若要更贴近 ASM dispatch-pull，需要把 staged input 或等价 peer row load 融入 pure C global load/tile scheduling，而不是新增 copy kernel；
  - 后续 K1 实现重点是证明 direct row-ptr load 与 staged_x pull 在 correctness、跨 rank可见性和性能上的取舍，或者在同一主 kernel 内加入 stage/pull，不破坏 pure C 5pack MMAC pipeline。

### K3 original ASM vs combine ASM

- K3 no-tail fused ASM 的主要新增点集中在 epilogue/store：
  - graph path 有 `active_tiles` gate，`wg1 >= active_tiles` 时直接退出；
  - epilogue 保留 `row_combine_ptrs` 指针表，按 row pointer scatter 到 combine buffer，而不是只写 contiguous output；
  - 通过 half-tile LDS staging、`K3_LOAD_COMBINE_ADDR4`、`K3_STORE4`、`K3_STORE_STAGED_HALF` 等宏，把 combine 写回放在 C tile store 阶段；
  - padded/invalid rows 依靠 row pointer valid mask 避免越界 store。
- 映射到 V3 pure C 5pack：
  - V3 K3 no-tail 不能先写 `output_workspace` 再新增 postprocess kernel；
  - 应在 pure C epilogue/store 阶段把原 contiguous store 改成 `row_combine_ptrs` 指向的 combine scatter，保留 tile/compute 主干；
  - graph active tiles、row pointer padding 和 external `rank_barrier + reduce_local_combine` 合同需要继续复用。

### K3 original ASM vs tail-reduce ASM

- K3 tail-reduce fused ASM 在 no-tail combine 基础上额外新增：
  - done counter、peer signal address table、done target、reduce output、reduce combine base、reduce vector/block 数和 graph runtime state 参数；
  - `K3_TAIL_ATOMIC_SIGNAL`、`K3_TAIL_WAIT_SIGNAL`、`K3_TAIL_APPLY_GRAPH_RUNTIME_STATE`、`K3_TAIL_LOAD_ACCUM_*`、`K3_TAIL_PACK_REDUCE_OUT` 等尾部 reduce 宏；
  - 额外 reducer workgroups，用于等待所有 rank signal 后处理主 GEMM 覆盖不到的 reduce block；
  - 主 GEMM workgroup combine store 完成后，lane0 递增 done counter；只有最后完成的本地 WG 发布 peer signal 并执行本地 reduce，避免所有 WG 做全局等待造成 resident-WG deadlock 风险。
- 映射到 V3 pure C 5pack：
  - V3 tail path 必须把 combine store 完成、done counter、peer signal、local reduce 放在同一个替换 kernel 内，不能退化为外部额外 launch；
  - signal/wait/fence 语义需要保持保守：store 完成后再 signal，wait 侧使用 system-visible load/atomic 和 cache invalidation；具体 fence/inline asm 写法需在实现前再次查 DCU KB 并用 ISA/正确性验证；
  - extra reducer WG、graph runtime active tile 和 reduce block 上界属于功能合同，不可从 V2 tail-reduce 路线机械照搬后忽略。

## 2026-06-10 K3 pure source 初筛

- 搜索范围：
  - `hygon_tmp`；
  - `megamoe/dcu_megamoe_large_opt`；
  - 排除 `megamoe/dcu_megamoe_v2/**` 和 `csrc/kernels/dcu_megamoe_v2/**`。
- 初筛结果：
  - 未找到一个非 V2、独立命名的 K3 pure C pack5 production source；
  - 当前非 V2 可信核心来源仍是 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 与 original groupgemm ASM；
  - large-opt K3 目录只有原 ASM wrapper、K3 combine/tail ASM 和当前 V3 stub/heavy 边界壳。
- 对 Phase 3 的含义：
  - K3 normal pure 主体应从 original pure groupgemm body 派生为 `N=4096, K=2048` 的 L2 groupgemm，再按 K3 ASM diff 把 store epilogue 改成 `row_combine_ptrs` scatter；
  - 不能把“缺少现成 K3 pure 文件”当作理由回退到 V2 source；
  - 下一步先检查 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 的模板参数、store epilogue 和 N/K 约束，确认可否用同一 body 派生 K3 normal/LL。

## 2026-06-10 K3 ext 与 pure body 参数差异

- large-opt K3 ASM ext 合同：
  - `act_fp8` shape 为 `[total_rows, intermediate]`；
  - K3 固定 `hidden=4096, intermediate=2048`；
  - `m_indices` 长度覆盖 `total_rows`，每行给出 expert；
  - 原 ASM L2 weight shape 是 Marlin `[local_experts, hidden / 16, intermediate * 16]`，scale shape `[local_experts, hidden]`；
  - no-tail `prob.d` 指向 `row_combine_ptrs`，tail path 额外使用 done counter / signal addrs / reduce y / sym buffer / graph active tiles。
- 当前 K1 V3 normal C 派生 header 仍是 K1 形状：
  - kernel 内硬编码 `kProblemN=4096`、`kProblemK=4096`；
  - weight expert stride 使用 `0x01000000`，对应 `4096 * 4096` FP8 bytes；
  - scale loader 用 `tile_expert * kProblemN`；
  - output store 写 contiguous `out[row, hidden]`。
- K3 normal 派生要求：
  - compute 主体需要变成 `K=2048, N=4096`，num-k-stages 从 32 降为 16，expert weight stride 从 K1 的 16 MiB 降为 8 MiB；
  - pack5 weight offset 必须按 V3 pack5 `[expert, n, k]` 的 `n=4096,k=2048` 重新计算，不能沿用 K1 stride 常量；
  - store helper 要从 contiguous `output_workspace` store 改为 `row_combine_ptrs[row] + hidden` scatter；
  - no-tail 第一版只实现 combine scatter，继续复用外部 `rank_barrier + reduce_local_combine`；tail signal/reduce 后续单独按 ASM diff 和 DCU KB 证据实现。

## 2026-06-10 K3 V3 normal no-tail raw kernel 初版

- 新增 K3-owned header：
  - `megamoe/dcu_megamoe_large_opt/K3_fused/k3_v3_pack5_groupgemm_impl.cuh`；
  - 来源是 K1 pure/V3 normal 5pack groupgemm 主体的机械派生，再按 K3 ext contract 和 K3 ASM diff 做参数/epilogue 替换；
  - 不是 `k3_v3_groupgemm_impl.cuh`，避免恢复此前 V2-derived header 名和实现路径。
- 当前 normal no-tail 差异：
  - kernel 固定 `N=4096, K=2048`；
  - stage order 不使用 K1 的 `stage_iter ^ 16`，避免 16-stage K3 上越界；
  - `act_fp8/act_scale` 使用 contiguous row load；
  - weight offset 用 `kProblemK` 参数化，K3 expert stride 为 `4096 * 2048`；
  - epilogue 使用 `row_combine_ptrs` scatter，在 GEMM store 阶段写 combine buffer；
  - raw launcher 只在 `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1` 且 normal backend 时 launch，tail path 暂时显式 no-op。
- 当前限制：
  - store 先用 scalar BF16 row-pointer write 做 correctness bring-up，尚未实现 K3 ASM 的 vectorized/staged-half combine store；
  - public `k3_l2_fused_v3_to_combine()` 仍 fallback 到 ASM，不会运行半成品；
  - tail-reduce signal/reduce 仍未实现，需后续按 DCU KB + ISA/正确性闭环处理。

## 2026-06-10 K3 V3 normal raw aicc 编译边界问题

- 远端默认 build 与 source pytest 已通过，但显式 K3 normal raw aicc build：
  - `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1`
  - `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=normal`
  - `DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1`
  - `DG_FORCE_BUILD=1 MAX_JOBS=2 python3 setup.py build_ext --inplace`
  在 600s hard timeout 后失败。
- 日志显示当前 per-extension aicc marker 作用在整个 `megamoe.dcu_megamoe_large_opt.K3_fused.k3_fused_ext` extension：
  - aicc 先编原 `k3_fused_ext.hip`；
  - 再编 `k3_v3_stub_ext.cu`；
  - timeout 前尚未稳定到达新的 `k3_v3_fused_ext.hip` heavy TU。
- 结论：
  - 这是 build 边界过粗，不是 K3 pack5 no-tail kernel 已被证明无法编译；
  - 下一步不能重复同一条 600s raw build；
  - 应把 K3 V3 raw normal heavy TU 拆到独立 raw extension 或等价的单 TU aicc 编译边界，保持原 K3 ASM wrapper/stub 继续走 hipcc。

## 2026-06-10 K3 V3 raw normal split extension 编译验证

- `setup.py` 已将 K3 raw normal heavy TU 拆成独立 extension：
  - 主 extension：`megamoe.dcu_megamoe_large_opt.K3_fused.k3_fused_ext`，只包含原 ASM wrapper 与 `k3_v3_stub_ext.cu`，继续走 hipcc；
  - raw extension：`megamoe.dcu_megamoe_large_opt.K3_fused.k3_v3_fused_ext`，只包含 `k3_v3_fused_ext.cu`，normal raw backend 下打 aicc marker。
- 远端验证：
  - source pytest `7 passed`；
  - 默认 build status `0`，`v2_seen=0`、`aicc_marker=0`、`k3_raw_tu_seen=0`；
  - raw normal build status `0`，`v2_seen=0`、`aicc_marker=1`，日志显示 `k3_fused_ext` 仍用 `/opt/dtk/bin/hipcc`，`k3_v3_fused_ext` 使用 `/workspace/dtk_aicc/bin/aicc` shim。
- 结论：
  - K3 raw normal aicc timeout 的直接原因已解除；
  - 后续 K3 no-tail correctness smoke 可以直接 import `K3_fused.k3_v3_fused_ext`，public staged wrapper 仍保留 ASM fallback，避免半成品进入生产路径。

## 2026-06-10 K3 V3 normal no-tail raw smoke

- smoke 脚本：`hygon_tmp/sglang_debug/k3_v3_normal_smoke.py`。
- all-ones 形状：
  - rows=256，hidden=4096，intermediate/K=2048，local_experts=1；
  - `act_fp8`、`weight_pack5`、`act_scale`、`weight_scale` 全 1；
  - `m_indices` 全 0；
  - `row_combine_ptrs` 直接指向本地 BF16 combine buffer 的每一行。
- all-ones 结果：
  - `dcu_megamoe_v3_k3_raw_normal_available() == True`；
  - sample mean = `2048.000000`；
  - sample max_abs_err = `0.000000`。
- pack5/expert-pattern 结果：
  - 首次加入 hidden pattern 后失败，暴露出 `k3_v3_fused_ext.cu` launch 实例化为 `<256, 256, false>`，即 baseline/Marlin weight-load 分支；
  - V3 normal 权重合同是 pack5，必须与 K1 normal 一样实例化 `<256, 256, true>`；
  - 改成 pack5 分支并重编后，row0 hidden pattern 完全对齐；
  - 初始 rows=256 且一个 256-row tile 内混 expert 时失败：row 128 仍使用 tile 首行 expert0；
  - 根因不是 pack5 load，而是当前 normal K3 core 与原 pure body 一致，按 256-row tile 读取 `row_expert[tile_token]`，要求每个 tile expert homogeneous；
  - 修正 smoke 为 rows=512、两个 256-row tile 分别 expert0/expert1 后通过；
  - pattern sample mean = `7680.000000`，sample max_abs_err = `0.000000`。
- 结论：
  - K3 V3 normal raw no-tail kernel 已能通过独立 aicc-built extension import/launch；
  - no-tail epilogue 的 scalar `row_combine_ptrs` scatter 在单 expert/all-ones 与 tile-homogeneous multi-expert pack5 pattern 场景正确；
  - staged 接入时必须保证 K3 normal 消费的是 compact tile list / tile-homogeneous row ordering，不能把任意逐行 expert 混排直接交给当前 normal tile kernel；
  - 这仍只是 low-level smoke，不覆盖 real K1/K2 staged input、tail-reduce、graph active tiles 或跨 rank可见性。

## 2026-06-10 K3 tail-reduce 同步语义复核

- 本轮按 `dcu-rag-kb` 要求重新查询 Hygon/gfx938 通信融合与同步语义：
  - Hygon/flux GEMM+ReduceScatter guidance 明确把通信/reduce-scatter 语义放在 epilogue store path，而不是后置 rewrite kernel；
  - Hygon allreduce / reduce-scatter 参考在 signal 前使用 system-scope 可见性保证，常见形式为 `__threadfence_system()` 或 `fence.acq_rel.sys` 后再 `atomicAdd_system` / system-scope store；
  - wait 侧使用 system-visible load，并在低层 ASM 中配合 cache invalidation。
- 对照原 K3 tail-reduce ASM：
  - `K3_SCATTER_C_TILE_TO_COMBINE` 在 combine scatter store 后有 `s_waitcnt vmcnt(0)`、`buffer_wbinvl1_vol`、`s_barrier`；
  - 每个 GEMM WG 在 combine store 完成后由 lane0 增加 `asm_done_counter`；
  - 只有最后完成的本地 GEMM WG 发布 peer signal，然后执行本地 reduce；
  - 额外 reducer WGs 只负责等待 peer signal 后分担 reduce block，避免所有 GEMM WG 全局等待造成 resident-WG deadlock；
  - `K3_TAIL_WAIT_SIGNAL` 使用 `global_load_dword ... glc slc`、`s_waitcnt vmcnt(0)` 和 `buffer_wbinvl1_vol` 轮询。
- 对 V3 C pack5 tail raw 的直接约束：
  - tail path 仍必须是一发 K3 fused kernel：GEMM combine scatter、done counter、peer signal wait 和 local reduce 在同一 kernel 内完成；
  - signal 必须在本 rank combine stores 完成并完成 system-scope fence 后发布；
  - 等待 peer signal 的工作只能由最后本地 GEMM WG 和额外 reducer WG 承担，不能让所有 GEMM WG 进入 peer wait；
  - no-tail raw scalar rowptr scatter 已通过，只能在 epilogue/store/reduce 自然边界补 tail 语义，不能改写 GEMM load/compute 主干。

## 2026-06-10 K3 tail raw aicc local-memory resource finding

- 远端 `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=normal DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1` 构建确认：
  - `v2_seen=0`；
  - `k1_heavy_seen=0`；
  - aicc marker 只作用在独立 `K3_fused.k3_v3_fused_ext` raw extension；
  - 因此构建边界正确，失败不是 V2 或 K1 混入。
- 失败原因：
  - aicc 报 `local memory (65552) exceeds limit (65536)`；
  - 超限发生在 `V3_K3_Fused_...<256,256,true,true>` tail 模板；
  - 这属于 tail-reduce 附加路径的编译资源压力，不是 no-tail raw 或 GEMM 主干已知错误。
- KB/optimizer 参考：
  - Hygon/ROCm 优化知识中低 occupancy / spill / LDS 或 local-memory 资源压力需要先缩减寄存器 live range、shared/local usage，再重新编译验证；
  - CK/Hygon 参考中也有因 VGPR spill 禁用过大 instance 的模式，说明这种失败应作为资源边界处理，而不是继续盲目加模板。
- 本轮最小修复策略：
  - 不改 K3 GEMM load/compute/store 主体；
  - 去掉 tail wait helper 的 device `printf`，保留 timeout `abort()`；
  - tail reduce worker 从 8 个同时 live 的 float 累加改为逐 BF16-pair 累加/pack，降低 tail 路径寄存器和本地内存压力。
- 第二轮定位：
  - 资源报错仍停在 `65552 > 65536`，说明主要不是 debug `printf` 或 reduce worker live range；
  - K3 normal GEMM 主体的 `lds_stage` 为 `2 * 256 * 128 = 65536B`，已经正好用满每 block LDS；
  - tail path 额外 `__shared__ int tail_is_last_gemm_block` 使 tail 模板 LDS 超限；
  - 尝试用 `__syncthreads_or(tail_is_last_thread != 0)` 做 block-wide 广播后，aicc 报告 local memory 变为 `65792 > 65536`，说明该 primitive 在当前 aicc/gfx938 lowering 下也会带来额外 LDS/本地资源。
- 第三轮修复：
  - 使用 `done_counter[1]` 作为 V3 tail owner slot；
  - last GEMM WG 的 thread0 在 `atomicAdd_system(done_counter, 1)` 命中 `done_target` 后写入 `blockIdx.x + 1`；
  - 同 block 线程只在 `done_counter[1] == blockIdx.x + 1` 时进入 signal/wait/reduce；
  - raw tail wrapper 要求 `done_counter.numel() >= 2`，第 0 个 int32 是 done count，第 1 个 int32 是 owner slot；
  - 该方案不增加 LDS，也避免非 last WG 因 `done_counter[0] == done_target` 误判自己是 last。

## 2026-06-10 K3 tail raw single-rank smoke

- 编译注意：
  - header-only 修改没有总是触发远端 `k3_v3_fused_ext.o` 重编；
  - 后续验证 K3 raw header 改动时，需要显式删除 `build/temp.../K3_fused/k3_v3_fused_ext.o` 和已复制的 raw `.so`，确认日志出现 `k3_v3_fused_ext.hip -o`。
- 第一轮 tail smoke:
  - topk=1 all-ones 下 signal/owner 正常：`done_counter=[16, 10]`、`signal_slot8=1`；
  - 数值失败为 row1 全 0，sample mean `1920`，max error `2048`。
- 根因：
  - K3 GEMM 主体在 store 前让 loader waves `wave_id >= kComputeWaves` return；
  - last GEMM block 只有前 512 个 compute-thread 会进入 tail reduce；
  - 原 reduce worker 用 `blockDim.x=768` 分片，导致 token1 vec tasks 主要落到不会进入 last-block reduce 的 thread 512..767；
  - extra reducer block 虽然有 768 线程，但分片宽度不一致会造成任务归属错误。
- 修复：
  - tail reduce worker 固定使用 `kTailReduceThreads=512`，`threadIdx.x >= 512` 直接 return；
  - task 初始值和 stride 都按 512 active threads 计算；
  - clean rebuild 后 topk=1 smoke：mean `2048`，max error `0`，done `[16, 13]`，signal slot8 `1`；
  - topk=6 smoke：mean `12288`，max error `0`，done `[96, 82]`，signal slot8 `1`。
- 结论：
  - K3 V3 normal tail raw single-rank 最小闭环已经通过；
  - 仍未证明多 rank、真实 staged K1/K2 输入、graph active tiles 或 uneven tokens。

## 2026-06-10 K1 raw rowptr e2e 纠偏

- 用户指出 `dcu_megamoe_v3_launch_k1_normal_symm_stage_raw` 内没有真正融合通信语义。
- 复核当前代码确认：
  - `k1_symm_fused_l1_v3_pack5()` 在 host 侧先 launch `k1_init_compact_routes_kernel`、`k1_count_compact_routes_kernel`、`k1_build_compact_tiles_kernel`、`k1_emit_compact_routes_kernel`；
  - `dcu_megamoe_v3_launch_k1_normal_symm_stage_raw()` 只消费这些 kernel 预生成的 `row_x_ptrs/row_x_scales/m_indices`，再运行 pack5 rowptr GEMM；
  - 该 raw normal kernel 不从 symmetric buffer 内部完成 dispatch-pull，不在 GEMM load/tile scheduling 内生成 route metadata，也不写 `row_combine_ptrs/output_index/stats`。
- 结论：
  - 当前 raw rowptr K1 只能作为 low-level GEMM smoke，不能作为 staged V3 K1 fused 功能对齐，也不能用于 e2e correctness；
  - K1-only e2e 的前置条件必须是 K1 主 kernel 本身融合 dispatch-pull、route metadata、row_combine_ptrs、output_index 和 stats 语义；
  - public staged wrapper 已撤回 raw rowptr K1 接入，`USE_MEGAMOE_V3=1` 在 K1 真正 fused 前 fail-fast，避免产生无效 correctness 结论。
- 后续方向：
  - 回到 original groupgemm ASM vs K1 dispatch-pull fused ASM 差异图，把 `SYMMROUTE stage` / dispatch-pull 的必要语义映射进 pure C 5pack K1 主体；
  - 不再用 raw rowptr + 独立 route prebuild kernels 跑 e2e。

## 2026-06-10 K1 normal single-kernel fixed-route fused 骨架

- 实现方向：
  - 在 `V3_K1_Fused_DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_C_TN_MT256XNX128_BF16` 开头增加可选 `build_fixed_route_in_kernel`；
  - normal raw launcher 传入 `sym_buffer`、`grid_barrier`、rank/shape、`route_weights`、`output_index`、`row_combine_ptrs`、stats；
  - `blockIdx.x == 0` 的 CTA 为本 tile 扫描 symmetric-buffer route，生成该 tile 的 `row_x_ptrs/row_x_scales/m_indices/route_weights/output_index/row_combine_ptrs`，然后写 per-tile ready flag；
  - 同一 tile 的其他 GEMM CTA 等 ready flag 后进入原 5pack GEMM load/compute/store 主体。
- 与 raw rowptr 旧路线的关键差异：
  - `k1_symm_fused_l1_v3_pack5()` normal path 不再 launch `k1_init/count/build/emit_compact_routes_kernel`；
  - metadata 生成与 GEMM 在同一 kernel 内完成，低层验证不再依赖独立 route prebuild kernels；
  - 当前先用 fixed per-expert tile layout，尚未恢复 compact tile scheduling，因此还不是最终性能路径。
- 编译资源问题：
  - 第一版在 K1 GEMM kernel 内添加 `__shared__ int tile_match_count` 后 aicc 报 `local memory (65552) exceeds limit (65536)`；
  - 根因是 pure K1 normal GEMM 的 LDS 已接近/达到 64KB，不能再添加任何 shared/local 状态；
  - 修复为复用已有 `grid_barrier` scratch 的 per-tile slot：每 tile 16 个 int slot，slot0=ready flag，slot1=match counter；
  - 修复后 K1 raw normal aicc clean build 通过。
- 远端 low-level smoke：
  - env：`DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=normal DG_BUILD_MEGAMOE_V3_NORMAL_AICC=1`；
  - ones：`active_routes=3072`，`route_weight_sum=3072.0`，`active_sample_abs_mean=4096.0`；
  - expert_ramp：sample expert outputs 4096/8192/... 对齐，`expert_ramp_abs_err=0`。
  - all-ranks skew 128 tokens：8 个 source rank 都路由到 rank0 local experts 时，`active_routes=6144`、`route_weight_sum=6144.0`、`expert_ramp_abs_err=0`，证明 fixed-route skeleton 能读 peer sym-buffer pointer table；
  - all-ranks skew 512 tokens：期望 24576 条 local routes，但 fixed capacity 只有 8192 rows，因此只保留 8192 条；这是 capacity/overflow 限制，不是 GEMM 数值错误。
- 当前限制：
  - public staged/e2e wrapper 仍默认 fail-fast；
  - 只有显式设置 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 时才允许 staged wrapper 调用 fixed-route skeleton；该开关只用于 launch/metadata bring-up，不得用于宣称 e2e correctness；
  - fixed-route 每 tile 重扫 routes，正确性方向优先，性能会差；
  - fixed capacity 按随机均匀 topk 估算，极端 skew 会 overflow；后续需要对齐原 K1 capacity/overflow 行为或建立明确 fallback/diagnostic；
  - 还未验证 8-rank真实 e2e、uneven、graph、LL 或 compact scheduling。

## 2026-06-10 K1 V3 metadata/unit correctness

- 新增临时脚本 `hygon_tmp/sglang_debug/k1_v3_metadata_compare.py`：
  - 手工构造 fake symmetric buffer；
  - 同一份 route/input 分别跑原 K1 ASM wrapper 和 V3 fixed-route K1 wrapper；
  - 原 K1 ASM 侧强制 `force_compact_prebuild=True`，作为可复用周边 metadata oracle；
  - 校验 row-side metadata：`row_combine_ptrs` 反查 task 后，逐 active row 检查 `m_indices`、`route_weights`、combine pointer 和 stats。
- 重要发现：
  - ASM launch 后的 `output_index` 不应直接作为 V3 row-id oracle；在某些路径/shape 下它可表现为不同 row 分配或包含非本 rank active 标记；
  - K2/K3 staged correctness 的关键合同是 row-side metadata：`row_combine_ptrs + m_indices + route_weights` 与 stats；
  - V3 fixed-route 可以与 ASM compact oracle 产生不同 row id/order，只要 row-side metadata 自洽且 K3 tile-homogeneous 约束满足，后续输出仍可对齐。
- 远端验证：
  - 128 tokens、all source ranks 都路由到 rank0 local experts，容量内：ASM 与 V3 row-side metadata 均通过，`active_rows=6144`、`stats_mismatch=0`；
  - 1024 tokens、rank0-only local experts：ASM 与 V3 row-side metadata 均通过，`active_rows=6144`、`stats_mismatch=0`；
  - 1024 tokens、all source ranks、`global_round_robin` 均匀到 256 experts：ASM 与 V3 row-side metadata 均通过，`active_rows=6144`、`stats_mismatch=0`。
- 当前限制：
  - 该验证仍是 metadata/unit，不是 staged e2e correctness；
  - fixed-route 容量内语义通过，但 compact scheduling/capacity/overflow 行为还未对齐原 K1；
  - LL、uneven tokens、graph 仍未覆盖。

## 2026-06-10 fixed-route staged bring-up 不能作为 correctness

- 用户再次指出：功能必须先对齐，不能把当前 fixed-route staged run 称为 e2e correctness。
- 重新界定：
  - `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 下的 1024 tokens tail on/off run 只用于定位 V3 K1 fixed-route launch、metadata、K2、原 K3 ASM 链路是否能跑完；
  - 这类 run 不覆盖原 K1 compact scheduling、capacity/overflow、LL、uneven tokens、graph，也没有证明 V3 K1 与原 K1 fused 行为等价；
  - 因此 public V3 staged wrapper 继续默认 fail-fast，K1-only correctness 要等 K1 主 kernel 对齐原 K1 route/capacity/output/stats contract 后再恢复。
- 额外根因修正：
  - staging run 中曾出现的 `hipExtModuleLaunchKernel invalid configuration argument` 已定位为测试 fixture 把 V3 pack5 L1 权重误传给 baseline DeepGEMM oracle；
  - fixture 已拆成 `baseline_l1/l2_weights` 与 `fused_l1/l2_weights`，K1-only bring-up 只让 fused L1 使用 V3 pack5，baseline 和原 K3 ASM 仍使用原 layout；
  - 这只是修复测试 layout 污染，不代表 V3 K1 功能已经对齐。

## 2026-06-10 K1 V3 fixed-route eager capacity 对齐

- 原 K1 eager ASM wrapper 的 fixed-capacity 估算使用真实 `num_tokens`，只有 graph/forced compact 这类 runtime-token 路径才按 `num_max_tokens_per_rank` 做保守容量。
- V3 fixed-route bring-up 曾统一使用 `num_max_tokens_per_rank` 估 capacity：
  - 1024 tokens、`num_max_tokens_per_rank=1152` 时，normal capacity 从 1 tile/expert 膨胀到 2 tiles/expert，rows 从 8192 变成 16384；
  - 这会扭曲后续 K1-only 链路定位和性能判断，也偏离原 K1 eager shape contract。
- 已改为：
  - eager/no runtime-token：`route_capacity_tokens_per_rank = num_tokens`；
  - graph/runtime-token：`route_capacity_tokens_per_rank = num_max_tokens_per_rank`；
  - `output_index` 仍按 `num_ranks * num_max_tokens_per_rank * topk` 分配，保持外部 scratch/graph contract 不缩容。
- 该改动只是 capacity contract 对齐，不等价于 compact scheduling 已完成；fixed-route 默认 fail-fast 仍保留。

## 2026-06-10 K1 V3 rowptr A-load 数值修复

- 用户纠正后重新收窄验证口径：当前 fixed-route staged run 不能叫 e2e correctness；只能用低层 compare 定位 K1 metadata/output。
- 新增临时脚本 `hygon_tmp/sglang_debug/k1_v3_output_compare.py`，在同一 fake symmetric buffer 和同一份权重上分别运行原 K1 ASM 与 V3 K1 fixed-route，并按 `row_combine_ptrs` 映射 active rows 比较 K1 `l1_out`。
- 现象：
  - `k1_v3_metadata_compare.py` 在 rank7 random 1024/1152 场景下 ASM/V3 row-side metadata 完全对齐；
  - 但 K1 output compare 曾出现大误差，ASM 接近直接 dequant reference，V3 偏离，说明问题在 V3 GEMM load/store 而非 route metadata。
- 根因：
  - V3 normal rowptr A-load helper 使用 `make_buffer_resource_device(row_ptr)` 构造 raw-buffer resource；
  - `row_ptr` 是按 lane/row 变化的 divergent pointer，raw-buffer resource descriptor 更适合 uniform/SGPR-like 基址；
  - 在 gfx938/aicc 路径上该用法会导致部分 lane/列读错值。
- 修复：
  - `buffer_load_fp8_b128_rowptr_device()` 改为从 `row_ptr + row_byte_offset` 做普通 global vector load；
  - 该改动只修正 rowptr A-load，暂不声称是性能最优形式；后续若要恢复 raw-buffer/lds load，需要 hipprof + ISA + output compare 证明。
- 远端验证：
  - source pytest：`tests/test_dcu_megamoe_v3.py` 结果 `8 passed`；
  - K1 normal aicc clean build status `0`，`aicc_marker=1`，`k1_raw_compile_seen=2`，`k3_raw_seen=0`，`v2_seen=0`；
  - K1 output compare rank7 random 1024/1152：`asm_active=6091`、`v3_active=6091`、`common=6091`、`missing_in_v3=0`、`extra_in_v3=0`、`max_abs=0.0`、`mean_abs=0.0`。
- 边界：
  - 这是 K1 fixed-route 低层数值对齐，不是 staged e2e correctness；
  - 原 K1 compact scheduling/capacity/overflow、LL、uneven tokens、graph 仍未对齐。

## 2026-06-10 K1 V3 normal ASM-style route scanner

- 目的：
  - 用户指出 fixed-route staged run 不能称为 e2e correctness，功能必须向原 K1 fused contract 对齐；
  - 原 K1 fused ASM 的非 prebuilt route 路线由前 4 个 row tiles 做 `SYMMROUTE` scanner，按 source rank 分片扫描，再发布每个 tile 的 `meta_flags`；
  - 旧 V3 fixed-route skeleton 是每个 expert tile 自己重扫所有 routes，功能上能定位但控制面偏离原 ASM。
- 代码变化：
  - `v3_k1_build_fixed_route_tile_device()` 仍保留函数名和固定容量语义，但内部改为 ASM-style scanner：
    - `tile_id < min(num_ranks, 4)` 的 CTAs 执行 route scan；
    - `source_rank = tile_id; source_rank < num_ranks; source_rank += scanner_tiles`；
    - `blockIdx.x` 的 16 个 N-tile CTAs 分片扫描 route offsets；
    - per-expert row counter 改用 `route_scratch_i32[local_expert]`；
    - 最后一个 scanner CTA 发布所有 row tiles 的 ready flag。
  - 主 K1 V3 normal kernel 增加 `route_scratch_i32` 参数链，launcher 不再忽略该 scratch header。
  - 初始化 owner CTA 在发布 init flag 前增加 block 内同步，避免 reset 尚未完成时 scanner 先读 metadata。
- 远端验证：
  - source pytest：`tests/test_dcu_megamoe_v3.py` 结果 `8 passed`；
  - K1 normal aicc clean build status `0`，`aicc_marker=1`，`k1_raw_compile_seen=2`，`k3_raw_seen=0`，`v2_seen=0`；
  - K1 output compare rank7 random 1024/1152：`asm_active=6091`、`v3_active=6091`、`max_abs=0.0`、`mean_abs=0.0`；
  - K1 fixed-route staged debug 1024/1152 tail-reduce on/off 均完成，rows=8192，reported diff `max_abs=0.000488281`。
- 4096-token 观察：
  - K1 output compare rank7 random 4096/4096：`asm_active=24446`、`v3_active=24446`、`max_abs=0.0`；
  - 但原 ASM auto path 输出 rows 为 `29696`，V3 fixed-capacity scanner rows 为 `32768`；
  - 说明当前 V3 数值和 row-side metadata 已能对齐，但 compact capacity/scheduling 仍未对齐原 K1 auto/compact 行为。
- 边界：
  - 该 scanner 是 fixed-capacity asm-style route fusion，不是最终 compact scheduling；
  - public staged wrapper 仍默认 fail-fast，`MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 仍只能用于 debug。

## 2026-06-10 K1 V3 normal compact-capacity route list

- 用户纠正后再次收紧验证口径：
  - 当前 staged run 只能叫 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1` 下的 debug/bring-up；
  - 真正 e2e correctness 必须等 K1 主 kernel 对齐原 K1 fused dispatch-pull、compact scheduling/capacity、row-side metadata、output_index 和 stats 合同后再恢复；
  - 本轮只推进 K1 V3 normal 的 compact-capacity / row-tile list 对齐，不解除 public wrapper fail-fast。
- 代码变化：
  - V3 normal host wrapper 复用原 K1 ext 的 `compact_capacity_tiles()` 估算，normal 后端在 compact capacity 小于 fixed capacity 时传 `compact_capacity_in_kernel=1`；
  - `v3_k1_build_fixed_route_tile_device()` 在同一 K1 kernel 内完成 compact count -> tile_bases/tile_experts build -> emit row metadata；
  - count 阶段按原 `k1_count_compact_routes_kernel` 合同只按 local expert 计数，不提前跳过 `weight == 0`；emit 阶段仍跳过 zero weight；
  - scanner CTA 在发布 count/emit completion flag 前加 `__threadfence()`，参考 DCU KB producer-consumer flag / grid_barrier 模式，避免前序 global writes 与 flag 可见性乱序；
  - GEMM load/compute/store 主体未重排，仍按每 tile 首行 `row_expert[tile_token]` 选择 pack5 expert 权重。
- 远端验证：
  - source pytest：`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 结果 `8 passed`；
  - K1 normal aicc clean build：`__STATUS:0`，`aicc_marker=1`，`k1_raw_compile_seen=2`，`k3_raw_seen=0`，`v2_seen=0`；
  - K1 output compare 4096/4096 rank7 random：`asm_rows=29696`、`v3_rows=29696`、`asm_active=24446`、`v3_active=24446`、`missing_in_v3=0`、`extra_in_v3=0`、`max_abs=0.0`、`mean_abs=0.0`；
  - K1 output compare 1024/1152 rank7 random：`asm_rows=8192`、`v3_rows=8192`、`asm_active=6091`、`v3_active=6091`、`max_abs=0.0`；
  - staged debug 1024/1152 tail on/off：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP=1`，K3 ASM 保持原 layout，结果均 status 0，reported `max_abs=0.000488281`；
  - staged debug 4096 requested tokens tail on/off：同上 env，结果均 status 0，reported `max_abs=0.000488281`，route_scratch 约 2.012 GiB。
- 失败与修复：
  - 首次远端 source pytest 因 source guard 仍要求 `: fixed_capacity_tiles;` 失败；
  - 修复为检查 normal host 支持 compact/fixed 二选一、V3 main kernel 内 compact count/build/emit 和 flag fence，重跑通过。
- 边界：
  - 这证明 K1 V3 normal 的 compact rows、active rows、row-side metadata 和 K1 output 在当前 random 1024/4096 对比脚本中与原 K1 ASM 对齐；
  - 还未覆盖 LL、uneven tokens、cuda graph、zero-weight 专门用例、overflow fallback 和 public K1-only correctness gate；
  - staged debug 仍只是 V3 K1 + 原 K2 + 原 K3 ASM 的定位验证，不作为最终 e2e correctness。

## 2026-06-10 K1 V3 normal public K1-only gate

- 代码变化：
  - `k1_symm_fused_l1_v3()` normal backend 不再要求 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP`；
  - LL backend 仍明确 fail-fast，提示当前 K1-only correctness gate 只支持 `MEGAMOE_DCU_V3_BACKEND=normal`；
  - graph wrapper 仍 fail-fast，cuda graph 对齐留在 Phase 7；
  - `large_opt.py` 的 K3 launcher 继续强制原 ASM，因此本阶段只替换 K1，避免 K1/K3 同时替换导致定位困难。
- 远端验证 env：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal`；
  - 不再设置 `MEGAMOE_DCU_V3_ALLOW_FIXED_ROUTE_BRINGUP`；
  - K3 分别跑 `K3_USE_ASM_TAIL_REDUCE=1` 和 `0`；
  - L1 使用 V3 pack5 layout，L2 使用原 ASM layout。
- 远端验证结果：
  - 1024/1152 tail on：status 0，baseline correctness `max_abs=0.000488281`、`mean_abs=9.50943e-06`；
  - 1024/1152 tail off：status 0，baseline correctness `max_abs=0.000488281`、`mean_abs=9.50943e-06`；
  - 4096 requested tokens tail on：status 0，baseline correctness `max_abs=0.000488281`、`mean_abs=9.45465e-06`；
  - 4096 requested tokens tail off：status 0，baseline correctness `max_abs=0.000488281`、`mean_abs=9.45465e-06`。
- 结论：
  - normal K1-only staged correctness gate 已恢复，且不依赖 debug env；
  - 这不是完整 V3 K1/K3 e2e，因为 K3 仍是原 ASM，LL/uneven/graph 也未覆盖；
  - 下一阶段可以在保持 K1 normal 结果可回归的前提下，继续推进 LL K1-only 或 V3 K3 staged 接入。

## 2026-06-10 K1 LL staged probe 边界修正

- 术语修正：
  - LL 当前验证只能称为 K1-only staged probe/correctness gate 候选，不能称为完整 e2e correctness；
  - 完整 e2e correctness 必须在 V3 K1/K2/K3 功能链路、tail-reduce/no-tail、eager/graph、uneven tokens 都覆盖后再使用该称呼。
- 已完成的 LL 低层能力：
  - K1 LL main kernel 内加入 symmetric-buffer stage helper、route metadata、`output_index`、`row_combine_ptrs`、`local_topk_mask`、`tail_tokens` 和 staged FP8/scales；
  - fake symmetric buffer raw smoke 通过：
    - 32 tokens、all ranks：`active_routes=1536`、`route_weight_sum=1536.0`、`row_ptrs_nonzero_total=1536`、`expert_ramp_abs_err=0.0`；
    - 128 tokens、rank-local：`active_routes=768`、`route_weight_sum=768.0`、`row_ptrs_nonzero_total=768`、`expert_ramp_abs_err=0.0`。
- 失败的 staged probe：
  - env：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=1`；
  - shape：8 卡，32 tokens per rank，L1 V3 pack5，K2 原逻辑，K3 原 ASM tail-reduce；
  - 结果：`fused/baseline max_abs=0.0831298828125`，超过 `--atol=0.0035`；
  - 日志：`hygon_tmp/sglang_debug/v3_ll_k1only_tokens32_tail1_20260610.log`。
- 当前判断：
  - fake raw smoke 只能证明 LL K1 单 kernel 的受控 metadata/stage/GEMM 可工作；
  - staged 失败更可能来自 LL row layout / `row_combine_ptrs` / padded row 合同与 K2/K3 原 ASM 消费方式不完全一致，或者真实随机 route 下的 metadata 差异；
  - 下一步必须拆分 K1 LL output 与原 ASM/直接 reference、K2 输入输出、K3 ASM 消费合同，不能用调大容差或继续称为 e2e 通过来掩盖。

## 2026-06-10 K1 LL barrier 与 K1 output compare

- 远端诊断首次运行 `K1_V3_COMPARE_BACKEND=ll` 的 K1 output compare 时 180s 超时，容器内残留 `python3 hygon_tmp/sglang_debug/k1_v3_output_compare.py`；
- 根因判断：
  - LL helper 使用 `v3_k1_ll_grid_barrier_device()`，但 host 侧没有 `hipMemsetAsync`，kernel 内也没有 normal backend 的 epoch/init；
  - `grid_barrier[0/1]` 来自复用的 `route_scratch`，未初始化时第一轮 barrier 可能死等；
  - 这违反了 KB 中 grid barrier/producer flag 需要先建立可见 counter/flag 状态的模式。
- 修复：
  - 增加 `v3_k1_ll_grid_barrier_init_device()`；
  - host 使用 `next_fused_l1_flag_generation()` 为 LL 传入 `barrier_epoch`；
  - LL main kernel 内先由 block0/thread0 写 `barrier[0]=0`、fence、`barrier[1]=epoch`，所有 block 等 phase 可见后再进入后续 grid barrier；
  - 不新增 runtime kernel launch。
- 远端验证：
  - source pytest：`8 passed`；
  - LL raw clean build：`__STATUS:0`、`k1_raw_compile_seen=3`、`k3_raw_seen=0`、`v2_seen=0`、`aicc_marker=0`；
  - K1 output compare：`K1_V3_COMPARE_BACKEND=ll K1_V3_COMPARE_RANK=7 K1_V3_COMPARE_TOKENS=32 K1_V3_COMPARE_MAX_TOKENS=128 K1_V3_COMPARE_EXPERT_MODE=random`；
  - 结果：ASM rows 8192、V3 rows 2048、ASM active 203、V3 active 203、common 203、missing/extra 0、`max_abs=0.0`、`mean_abs=0.0`。
- 当前结论：
  - 至少对随机 32-token fake symmetric buffer，V3 LL K1 的 dispatch/stage/GEMM 输出可按 `row_combine_ptrs` 与原 K1 ASM 对齐；
  - 下一步需要重跑真实 8 卡 K1-only staged probe；若仍失败，优先定位 K2/K3 对 LL row layout 的消费合同。

## 2026-06-10 K1 LL staged failure root cause narrowed to K3 ASM layout

- 真实 8 卡 K1-only staged probe 在 barrier 修复后仍失败：
  - tail-reduce on：`max_abs=0.09228515625 > 0.0035`；
  - 强制 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 后仍失败：`max_abs=0.072265625 > 0.0035`。
- 新增单 GPU fake sym-buffer 诊断 `hygon_tmp/sglang_debug/k1_v3_ll_k2k3_compare.py`：
  - 同一份 random route / weight 先跑原 K1 ASM，再跑 V3 LL K1；
  - 按 `row_combine_ptrs` 对齐 active rows，逐级比较 K1、K2 act、K2 scale、K3 no-tail combine。
- 远端诊断结果：
  - K1：ASM/V3 active=203、common=203、missing/extra=0、`max_abs=0.0`；
  - K2 act：active/common=203、`max_abs=0.0`；
  - K2 scale：active/common=203、`max_abs=0.0`；
  - K3 ASM combine：common=203、`k3_combine_max_abs=97.0`、`mean_abs=13.263659477233887`。
- 结论：
  - 当前 LL K1 的 64-row/expert layout 可以保证 K1/K2 row-wise 数值对齐；
  - 原 K3 ASM 仍隐含 256-row tile-homogeneous expert layout，不能直接消费 4 个 expert 混在同一个 256-row tile 内的 LL output；
  - 如果要按计划先做“K1 V3 + 原 K3 ASM”定位验证，LL K1 staged path 需要临时使用 K3-ASM-compatible 256-row/expert stride；
  - 最终 V3 K3 LL 接入后，才能恢复/保留真正 64-row LL layout 以服务性能目标。

## 2026-06-10 K1 LL K1-only staged gate passed with ASM-compatible layout

- 代码隔离：
  - `k1_symm_fused_l1_v3()` 增加 `ll_asm_compatible_layout=False`；
  - low-level LL smoke/compare 默认仍使用 64-row true LL layout；
  - `large_opt.py` 当前 K3 仍为原 ASM，因此 `MEGAMOE_DCU_V3_BACKEND=ll` 的 K1-only staged gate 传 `ll_asm_compatible_layout=True`，使用 256-row/expert stride 满足原 K3 ASM tile-homogeneous 假设；
  - 该兼容布局是 K1-only 定位阶段的桥，不是最终 LL 性能路径。
- 单卡逐级验证：
  - `K1_V3_LL_ASM_COMPAT=1` 后，fake random 32/384 rank7：
  - K1 `max_abs=0.0`；
  - K2 act `max_abs=0.0`；
  - K2 scale `max_abs=0.0`；
  - K3 ASM combine `max_abs=0.0`。
- 8 卡 K1-only staged correctness：
  - env 基本组合：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll`；
  - 32 tokens, tail-reduce on：status 0，`max_abs=0.000244141`、`mean_abs=9.31031e-06`；
  - 32 tokens, tail-reduce off：status 0，`max_abs=0.000244141`、`mean_abs=9.31031e-06`；
  - 128 tokens, tail-reduce on：status 0，`max_abs=0.000244141`、`mean_abs=8.97296e-06`；
  - 128 tokens, tail-reduce off：status 0，`max_abs=0.000244141`、`mean_abs=8.97296e-06`。
- 当前结论：
  - V3 K1-only staged gate 现在 normal 与 LL 都已覆盖；
  - 这仍不是完整 V3 e2e correctness，因为 K3 V3 staged、uneven tokens 和 cuda graph 仍未对齐；
  - 下一步进入 K3 V3 staged no-tail，再做 tail-reduce。

## 2026-06-10 K3 V3 no-tail staged wiring boundary

- 用户再次纠正术语：K1-only staged gate 不能称为 e2e correctness，完整功能对齐必须等 V3 K1/K2/K3 链路、tail/no-tail、eager/graph 和 uneven tokens 都覆盖。
- 本轮接入策略：
  - 仅在 eager staged、`MEGAMOE_DCU_V3_BACKEND=normal` 且 `K3_USE_ASM_TAIL_REDUCE=0` 时，让 K3 调用 V3 normal no-tail raw combine；
  - tail-reduce 继续走原 K3 ASM，直到 V3 tail signal / local reduce 合同接入并验证；
  - LL 继续走 K1-only staged gate 的原 K3 ASM 兼容布局，直到 V3 K3 LL 真正接入；
  - graph path 仍保持原 ASM，因为 V3 K1 graph 当前未对齐。
- layout 合同：
  - K1-only staged gate：L1 使用 V3 pack5，L2 保持原 ASM layout；
  - K3 V3 normal no-tail staged：L1/L2 都在 fixture/offline 阶段提前转为 V3 pack5，执行路径不新增 repack kernel。
- `k3_l2_fused_v3_to_combine()` 现在 fail-fast 区分未完成能力：
  - backend 非 normal、tail-reduce、graph/active_tiles、tail-signal metadata 都显式拒绝；
  - V3 raw extension 未按 `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS=1 DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=normal` 编译时显式报错；
  - normal no-tail 只调用 `k3_v3_normal_combine_raw()`，仍复用外部 `rank_barrier + reduce_local_combine`。

## 2026-06-10 K3 V3 normal partial-tile root cause narrowing

- 单卡 fake symmetric-buffer 逐级诊断确认：
  - V3 K1 normal 输出无 nonfinite；
  - K2 `act_fp8/act_scale` 无 nonfinite；
  - 原 K3 ASM 在同一 K1/K2 输出上与直接 FP8 reference 对齐；
  - V3 K3 raw normal no-tail 的错误集中在自身 K3 compute/combine。
- pack5 B-load 大布局修正后，K3 V3 no-tail 从全局大错收敛到稀疏 partial-tile 错误；`local_round_robin` 这类 full-tile 场景可与原 K3 ASM `max_abs=0`。
- 进一步定位：
  - random route 下每个 local expert 只有一个 256-row tile，active rows 是 dense prefix，tile 末尾为 inactive padding；
  - 原 K3 V3 C core 继承 pure contiguous GEMM 假设，A 侧 `act_fp8` 对 inactive rows 仍直接加载；
  - 在 K3 主 kernel 内按 `row_combine_ptrs[row] <= 0` 将 inactive A-load 置零后，random partial tile 误差从 `max_abs≈20+`、`mean≈2e-2` 收敛到 `max_abs≈4.4`、`mean≈8e-5`，full-tile 仍为 0 error；
  - 剩余错误出现在 dense-prefix boundary wave，例如 wave row0 valid、row16 invalid，说明 MT256 lowlat store/MMA fragment 对半 wave partial 的物理 row 映射仍需专门处理。
- 当前判断：
  - 不应回退到 V2 K3 real-flow；
  - 不应再改 pack5 大布局方向，full-tile/full-route 已验证；
  - 下一步应验证 64-row/partial-wave 诊断或为 MT256 boundary wave 增加 kernel-internal masked handling，仍不得新增 runtime kernel。

## 2026-06-11 K3 V3 normal partial-tile 反证记录

- tile64 诊断反证：
  - no-tail normal 从 `<256,256,true>` 临时切到 `<64,256,true>` 后，source guard 因期望 256 而失败，full-tile smoke 的 pattern 也失败，`pattern_sample_max_abs_err=16384`；
  - random partial compare 退化到 `k3_combine_max_abs=107.5`、`mean_abs=13.8988`；
  - 结论：残差不是简单 MT256 切成 64-row tile 可以解决，且 lowlat weight/store layout 隐含 256-row 合同。
- partial no-shuffle 反证：
  - 恢复 `<256,256,true>`，只在 partial masked store 关闭 lowlat accumulator shuffle；
  - full-tile smoke 仍通过，但 random partial compare 退化到 `max_abs=108.5`、`mean_abs=13.4450`，列模 16 呈规律性错位；
  - 结论：lowlat shuffle 对 accumulator/列映射是必须的，不能在 partial path 关闭。
- target-row-only / union mask 反证：
  - 将 A-load active mask 改为只看 shuffle 后 target row，会产生整行归零，random partial `max_abs=78.0`、`mean_abs=0.1321`；
  - 改为 logical row 与 target row 的 union 后出现 nonfinite，`k3_combine_rhs_nonfinite=528`，最大误差为 NaN；
  - 结论：当前最稳基线仍是 logical-row active mask：full-tile/local_round_robin 对齐，random partial 无 nonfinite，残差约 `max_abs=4.375`。后续应继续围绕 MT256 boundary wave 的 accumulator/store mapping 做更窄诊断，而不是更改 tile shape 或关闭 shuffle。
- no-mask + prezero 反证：
  - 为排除 K2 inactive rows 未初始化，诊断脚本增加 `K3_V3_COMPARE_PREZERO_ACT=1`，在 K2 前把 `act_fp8/act_scale` 预清零；
  - K3 V3 临时去掉 logical-row active mask 后，random partial 仍失败，`k3_combine_max_abs=28.875`、`mean_abs=0.0259957`；
  - 结论：问题不是简单的 inactive K2 输出脏数据；完全不 mask 会让 MMAC 消费 padding 行并扩大错误，因此当前代码已恢复 logical-row active mask。
- 当前稳定基线：
  - `<256,256,true>` + logical-row active mask + lowlat shuffle；
  - source pytest 通过、K3 full-tile smoke 通过；
  - random partial 仍有稀疏残差，例如 1024 tokens / max 1152 / rank7 random 下 `k3_combine_max_abs=3.625`、`mean_abs=8.94e-06`、`k3_all_diff_gt_1=92`；
  - 最大误差 wave 的行 928-937 有效、938-959 无效，`wave_row0_valid=True`、`wave_row16_valid=False`，说明残差集中在少于 16 个有效 row 的 MT256 尾 wave。

## 2026-06-11 K3 V3 normal partial-tile 收敛

- 追加 wave 统计后修正判断：
  - 残差不是单纯 `<16` 有效行尾 wave；
  - 大误差会出现在 partial tile 中的完整 `16+16` wave，且通常只影响前 16 行，后 16 行正常；
  - 这说明问题在 partial tile 下的 wave/epilogue/codegen 行为，而不是 pack5 大 layout 或 row_combine pointer。
- 反证与修复路径：
  - wave-level unmasked store 可修掉完整 wave，但会把错误转移到 `16+7` 一类半满 wave；
  - 在 masked macro 中对 `mask==0xf` 分流到 unmasked helper 反而引起 aicc/codegen 敏感回退；
  - 最终可行方案是更接近 pure C 主体：所有 compute waves 都执行同一 GEMM loop，invalid row 的 B-load 由 `row_combine_ptrs` logical-row mask 置零，epilogue 统一走 `store_acc_fragment_scaled_unmasked_device()`，实际写出仍由 `store_bf16_rowptr_device()` 的 rowptr guard 阻止无效行写 combine。
- 远端验证：
  - K3 V3 normal aicc raw rebuild status `0`，log `hygon_tmp/sglang_debug/k3_v3_all_waves_compute_rebuild_20260611_015214.log`；
  - source pytest `tests/test_dcu_megamoe_v3.py`：`8 passed`；
  - K3 raw smoke：ones / pattern 均 `max_abs_err=0`；
  - random partial compare：1024 tokens / max 1152 / rank7 random / seed 1234 下，ASM 与 V3 K3 combine `max_abs=0.0`、`mean_abs=0.0`，wave buckets 全部 `gt_1=0`；
  - 验证 log：`hygon_tmp/sglang_debug/k3_v3_all_waves_compute_verify_20260611_015338.log`。
- 边界：
  - 这是 K3 normal no-tail raw/single-card staged-compare 通过，不是 8 卡 eager staged correctness；
  - 下一步必须跑 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0` 的 8 卡 eager staged no-tail，再决定 Phase 3/4 对应项是否打勾。

## 2026-06-11 K3 V3 normal no-tail 分布式等价与 staged 残差

- 8 卡 staged no-tail 对 baseline 仍未通过：
  - thread0-only fence 版本：`v3_normal_k3_notail_staged_4096_fence_20260611_021648.log`，rank 6 `max_abs=0.00594329833984375`；
  - all-thread `__threadfence_system()` 版本：`v3_normal_k3_notail_staged_4096_allthread_fence_20260611_022244.log`，rank 4 `max_abs=0.007080078125`；
  - all-thread fence rerun：`v3_normal_k3_notail_staged_4096_allthread_fence_rerun_20260611_022845.log`，rank 2 `max_abs=0.0035400390625`，略高于 `0.0035`。
- 原 staged ASM oracle 自身通过：
  - `orig_3stage_notail_4096_oracle_check_20260611_023100.log` status 0，`max_abs=0.000488281`；
  - 因此 4096 残差不是 baseline oracle 自身不稳定。
- 通过临时诊断 `hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 在同一份 V3 K1/K2 输出上分别跑原 K3 ASM 和 V3 K3 no-tail：
  - 修正 combine offset 使用实际 `sym_buffer.num_max_tokens_per_rank=4224`，避免 requested 4096 与 scratch capacity 4224 混淆；
  - 补齐 `swiglu_quant_channelwise_out(..., row_combine_ptrs=row_combine_ptrs)`，对齐真实 staged 中 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=1536` 的 rowptr skip 行为；
  - 8 卡 4096 requested tokens 诊断 log：`k3_v3_dist_y_compare_4096_rowptrk2_20260611_023616.log`，status 0；
  - 结果：`global_max_abs=0.0`，`global_gt_atol=0`，`global_y_max_abs=0.0`，每 rank rows 均为 29696，combine/y numel 均为 100663296。
- 当前结论：
  - 在真实 distributed capacity 与 K2 rowptr skip 条件下，V3 K3 normal no-tail 对原 K3 ASM 的同输入行为已等价；
  - 完整 staged 对 baseline 的小残差更可能来自 V3 K1/K2 与原 ASM baseline 的输入/fixture/layout/wrapper 差异，或测试 oracle 中 L1/L2 layout 分流细节，而不是 K3 no-tail combine 主 kernel 本体；
  - 下一步应构造“原 K1 ASM + 原/V3 K3”与“V3 K1 + 原/V3 K3”的同一测试链路，逐段比较 K1/K2 后的 `act_fp8/act_scale/row_combine_ptrs` 和最终 `y`。

## 2026-06-11 K3 no-tail official staged residual after signal-only split

- signal-only 完成同步的编译问题已修复：
  - `kTailReduce` 与 `kSignalOnly` 尾部同步拆成独立 `if constexpr` 分支，no-tail signal-only 改成 thread0-only helper；
  - 远端 aicc normal raw rebuild 通过，log `hygon_tmp/sglang_debug/k3_v3_signalonly_split_rebuild_20260611_032948.log`；
  - source pytest `tests/test_dcu_megamoe_v3.py` 通过，`8 passed`。
- signal-only 不是正确性修复：
  - official staged normal/no-tail 1024 signal-only 通过，`max_abs=0.000488281`；
  - official staged normal/no-tail 4096 signal-only 仍失败，典型 `max_abs=0.0060577392578125`；
  - 加 debug sync 或关闭 K2 inactive-row skip 仍不能稳定通过；
  - dual compare 显示 formal top-level signal path `global_v3_vs_asm_max=0.0050048828125`，但 manual no-signal V3 K3 对 ASM 只有 `0.00054931640625`。
- 因此 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 改为默认关闭，仅作为显式诊断开关：
  - unset 时 official no-tail 不传 signal tensors；
  - 设置 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 时才启用 no-tail signal-only 路径。
- default no-signal 仍未解决 official 4096：
  - official staged 4096 default no-signal 仍失败，典型 `max_abs=0.00628662109375`；
  - `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=1` 可把误差压低到接近阈值但仍有失败样本，说明强同步有帮助但不是完整解释；
  - prezero formal compare 可消除一部分 NaN/脏数据现象，但仍有 `0.004+` 残差。
- chain compare 的关键反证：
  - `hygon_tmp/sglang_debug/v3_chain_compare_4096_default_nosignal_20260611_034341.log` status 0；
  - `global_y_v3k1_asmk3_vs_asm_max=0.00054931640625`；
  - `global_y_v3k1_v3k3_vs_asm_max=0.00054931640625`；
  - `global_y_v3k3_vs_asmk3_max=0.0`；
  - chain 脚本在 K3 前显式 `zero_local_combine(...); dist.barrier()`，这与 official top-level 路径不同，是下一步定位重点。
- 当前判断：
  - K3 V3 pack5 compute/layout 和同输入 combine 写回已基本排除；
  - K1/K2 链路在 chain compare 中也能与 ASM 输出对齐到 `~5e-4`；
  - official staged 4096 残差更像 top-level wrapper/scratch 初始化、combine 清零时机、stream/order 或 pre-K3 可见性差异；
  - 不能新增 runtime kernel 来清零或同步，若最终需要修复，必须嵌入现有 K1/K3 main kernel 或复用原已有 barrier/reduce 合同。

## 2026-06-11 K3 no-tail gate fix and sync evidence

- 修复了 no-tail default gate 的 Python wrapper bug：
  - `large_opt.py` 中 `use_v3_k3_no_tail_signal` 原本存在，但 no-tail V3 分支仍无条件把 `sym_buffer`、`asm_done_counter`、`asm_signal_addrs` 等 signal 参数传给 `k3_l2_fused_v3_to_combine()`；
  - 修复后只有 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 时才传 signal kwargs，unset 时是真正的 no-signal combine-only；
  - 远端 source pytest `tests/test_dcu_megamoe_v3.py` 通过，`8 passed`。
- gate fix 后的 official staged normal/no-tail 4096：
  - default no-signal 仍有 rank 失败，典型 `max_abs=0.00506591796875 > 0.0035`；
  - formal dual compare 通过阈值，`global_v3_vs_baseline_max=0.00311279296875`、`global_v3_vs_asm_max=0.00311279296875`；
  - 说明 gate bug 已显著收敛 public V3 差异，但 official correctness 仍未稳定过关。
- K3 active-row store 覆盖已排除为主因：
  - sentinel 诊断 `K3_V3_DIST_V3_PREZERO=0 K3_V3_DIST_V3_FILL=0.125` 下，`global_sentinel_active=0`、`global_sentinel_inactive=25165824`、`global_y_max_abs=0.0`；
  - V3 K3 写全 active rows，sentinel 只留在 inactive/padding，reduce 不消费这些 sentinel。
- debug sync 证据：
  - full `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=1` 后 official staged 4096 通过，`max_abs=0.000488281`；
  - 只 sync `after_k3_combine` 仍失败，典型 `max_abs=0.00384521484375`；
  - 只 sync `after_k2` 仍失败，典型 `max_abs=0.00595855712890625`；
  - sync `after_k2 after_k3_combine` 通过，`max_abs=0.000488281`；
  - sync `after_k1 after_k2 after_k3_combine` 也通过，说明 K1 后同步不是必要条件。
- DCU KB 对当前判断的支持：
  - Hygon/ROCm custom collective 资料强调 producer 写 combine 后需要 system-scope fence/release，consumer 等待后需要 acquire/可见性保证；
  - “final sync 不需要 prior-write visibility” 只适用于后续没有消费者读取 prior writes 的场景；
  - 当前 K3 no-tail 后还有 `reduce_local_combine` 读取 combine buffer，因此 post-K3 barrier 不是 final-only barrier，必须确认其 release/acquire 和 L1/system visibility 语义。
- 下一步假设：
  - 不再把 K3 pack5 compute/layout、active-row store、signal-only 当作主因；
  - 优先检查已有 `rank_barrier`、signal load/store、`reduce_local_combine` 读 combine 前是否缺少 acquire/invalidate/system visibility；
  - 修复方向必须复用已有 barrier/reduce/kernel 路径，不新增 runtime launch。

## 2026-06-11 K3 no-tail visibility boundary evidence

- consumer-side patch 验证结果：
  - `rank_barrier(acquire_after_wait=True)` + `reduce_local_combine(invalidate_before_read=True)` 只在 V3 normal/no-tail 打开后，official 4096 default no-signal 仍失败；
  - 典型失败从小残差到 rank 局部 nonfinite 均出现，说明只修 reduce consumer 入口不等价于 debug sync。
- selective sync 重新验证：
  - `after_k2` only 仍失败，典型 `max_abs=0.00543212890625`；
  - `after_k3_combine` only 仍失败，出现 `fused/baseline nonfinite fused=4 baseline=0 diff=4`；
  - `after_k2,after_k3_combine` 组合通过，`max_abs=0.000488281`。
- 对当前 root cause 的收敛：
  - `_v3_debug_stage_sync()` 只做 `torch.cuda.synchronize()`，没有 `dist.barrier()`，因此它提供的是 host-side local GPU work completion / cache visibility 强度，而不是额外跨 rank 排队；
  - K2 和 V3 K3 raw host wrapper 都使用 `at::cuda::getCurrentCUDAStream().stream()`，因此不是显式跨 stream launch 错配；
  - 原 K3 combine ASM 在 epilogue 后有 `s_waitcnt vmcnt(0)` 和 `buffer_wbinvl1_vol`；V3 K3 C 版当前只在 store 后做 `wait_vmem_lds_store_device(); block_barrier_device(); __threadfence_system()`；
  - 下一步应在 V3 K3 main kernel 的两个边界补最小可见性语义：入口读 K2 `act_fp8/act_scale` 前 invalidate L1，出口 combine store 后 wait/invalidate/fence，对齐 ASM diff 方向而不改 GEMM 主循环。
- 入口/出口 invalidate 反证：
  - 入口 + 出口同时加 `buffer_wbinvl1_vol` 后 official 4096 变为大量 nonfinite；
  - entry-only 同样大量 nonfinite，说明 K3 主 kernel 开头全线程 invalidate L1 不是安全的 K2->K3 可见性修复；
  - exit-only 不产生大量 nonfinite，但仍无法通过 official 4096，说明 store 后 invalidate 不是充分修复；
  - 当前应撤回入口 invalidate 假设，不把 `buffer_wbinvl1_vol` 作为通用边界补丁继续叠加。后续重点转向 official path 与 chain compare 的真实差异：combine 清零、row pointer 覆盖、workspace 复用和测试迭代时序。

## 2026-06-11 K2->K3 visibility diagnostic setup

- 重新查 DCU KB：
  - Hygon microbenchmark 资料指出 `global_load_* ... off glc` 是 L2-oriented / bypass-L1 风格的显式 load path，并要求消费前 `s_waitcnt vmcnt(0)`；
  - 这与原 K3 combine ASM 中 row pointer 部分的 `global_load_dwordx2 ... off glc` 证据吻合，但该资料没有直接证明它能修复当前 staged residual，因此只能作为诊断方向。
- 原 K3 ASM 与 V3 K3 C path 的差异：
  - 原 K3 ASM 的 row-combine pointer 读取有 `global_load_dwordx2 ... off glc` 路径；
  - 原 K3 ASM 的 A 侧 K2 输出读取仍主要是 `buffer_load_*` 到 LDS，不是 glc；
  - V3 K3 当前读 K2 `act_fp8/act_scale` 走 raw/buffer load，K3 same-input compare 已通过，因此不能把 K3 compute/layout 重新当主因。
- 新增默认关闭诊断开关：
  - `MEGAMOE_DCU_V3_K2_SYSTEM_FENCE=1` 只在 V3 normal/no-tail 路径把 `system_fence_after_write=True` 传入现有 K2 kernel；
  - K2 每个实际写 `act_fp8/act_scale` 的线程在写出后执行 `__threadfence_system()`；
  - 该实验不新增 runtime kernel，不改变默认路径；目标是验证它能否替代或部分替代 `after_k2` host sync。
- 判读计划：
  - 若 K2 fence + `after_k3_combine` debug sync 通过，而 K2 fence 单独仍失败，则说明 K2->K3 与 K3->reduce 是两个独立弱边界；
  - 若 K2 fence 单独通过，则优先把修复收敛到 K2 producer visibility 与 K3/reduce 默认边界；
  - 若 K2 fence 不改善，则继续回到 official vs chain compare 的 combine workspace/iteration sequencing，而不是扩大 K3 GEMM 主体改动。

## 2026-06-11 K3 V3 rowptr store 形式诊断

- 8 卡 official normal/no-tail 4096 的现有证据：
  - K3 V3 raw/same-input 与原 K3 ASM 已通过；
  - K2 fence + `after_k3_combine` host sync 可通过，K2 fence only / after_k3 only / reduce acquire(glc read) / signal-only 均不能单独通过；
  - 因此 K2->K3 和 K3->reduce 是两个独立弱边界，且 K3->reduce 不能只靠 reduce consumer 侧 acquire/glc 解释。
- DCU KB 与本地 ASM 证据：
  - Hygon/gfx938 KB 命中 `global_store_dwordx4 v[addr], v[data], off`、`global_store_short v[addr], vdata, off` 等 inline/ISA 形式；
  - 原 K3 combine ASM 的 `K3_STORE4` 使用 `global_store_short v[136:137], data, off` 写 row-combine pointer 指向的 BF16 结果；
  - 原 K3 staged/vector path 使用 `global_store_dwordx4 addr, v[232:235], off`，而不是编译器普通 C 指针 store 降出的 flat store；
  - V3 C rowptr store 此前是 `row_ptr[hidden] = value`，生成 ISA 曾观察到 `flat_store_short` / `flat_store_short_d16_hi`，与原 K3 ASM store family 不一致。
- 本轮诊断方向：
  - 在 V3 K3 pack5 主 kernel 内把 rowptr combine BF16 store 改为显式 inline asm `global_store_short %0, %1, off`；
  - 不新增 runtime kernel，不改变 V2，不改 GEMM 主循环，只替换 epilogue rowptr store 形式；
  - 远端必须用 aicc 编译、source pytest、raw smoke/compare 和 official 4096 no-tail 验证，必要时用保存的 ISA/编译产物确认真正出现 `global_store_short`。

## 2026-06-11 K3 no-tail print/sync 反证与 release/acquire 方向

- `global_store_short` 已经让 V3 K3 generated ISA 的 rowptr store family 对齐原 K3 combine ASM，但 official 8 卡 4096 no-tail default 仍失败；`after_k3_combine` host sync/debug print 可让误差降到通过阈值，说明它暴露的是时序/可见性窗口，不是功能修复。
- 已撤回默认 host sync bridge：
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC` 默认从 `1` 改为 `0`；
  - 显式打开时只做诊断同步，不再打印，避免用 stdout timing 掩盖问题；
  - source guard 改为保护“默认关闭诊断”，不再保护 print/sync 路径。
- 本轮 DCU KB 命中 Hygon allreduce 参考 `custom_all_reduce.cuh`：
  - `barrier_at_start` 不需要 prior-write visibility；
  - `barrier_at_end<final_sync=false>` 在同步后仍有消费者时使用 release store + acquire load；
  - `barrier_at_end<final_sync=true>` 才能使用 relaxed/volatile 风格。
- 对当前问题的含义：
  - K3 no-tail 后还有 `reduce_local_combine` 读取 combine buffer，因此 post-K3 rank barrier 不是 final sync；
  - 不能接受 host sync/print 作为修复，应继续补足设备侧 release/acquire 或等价可见性语义；
  - 原 K3 ASM no-tail epilogue 明确是 `global_store_short` 写 combine 后 `s_waitcnt vmcnt(0)` + `buffer_wbinvl1_vol` 再 `s_endpgm`，V3 需要用 generated ISA 和最小 A/B 继续确认缺失点。

## 2026-06-11 K3 no-tail post-print 反证收敛

- V3 K3 no-tail generated ISA 已确认不再是普通 flat rowptr store：
  - aicc save-temps 目录：`hygon_tmp/sglang_debug/k3_v3_savetemps_nosync_20260611_120040/`；
  - `global_store_short=384`、`flat_store_short=0`；
  - tail epilogue 有 `global_store_short ...`、`s_waitcnt vmcnt(0)`、`s_barrier`、`buffer_wbinvl1_vol`、`s_endpgm`。
- 撤回 host sync / print 诊断桥后，official 4096 normal/no-tail 默认路径重新暴露真实失败：
  - default `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`：`max_abs=0.005859375`；
  - `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1`：`max_abs=0.004638671875`，略有改善但不通过；
  - `MEGAMOE_DCU_V3_K2_SYSTEM_FENCE=1 MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1`：`max_abs=0.00818634033203125`，变差。
- K2 producer fence 不是当前充分修复：
  - K2 save-temps 目录：`hygon_tmp/sglang_debug/k2_savetemps_fence_20260611_121241/`；
  - 生成 ISA 在函数出口附近出现 `s_waitcnt vmcnt(0) lgkmcnt(0)` + `buffer_wbinvl1_vol`；
  - 因此失败不能再归因于 K2 fence 源码位置过早或没有等待 global store。
- legacy peer-to-peer barrier 也不是充分修复：
  - 将 `rank_barrier(acquire_after_wait=True)` 临时切到现有 `mega_moe_rank_barrier(signal_buffers, rank_idx, num_ranks)` + invalidate 后，official 4096 仍失败；
  - 这说明“把 staged ticket barrier 换成已有 peer barrier”不能等价替代 debug sync。
- 当前 root-cause 方向：
  - K3 same-input、chain compare 和 generated ISA 已基本排除 K3 pack5 compute/layout、active-row store family、K2 fence codegen 作为单点主因；
  - chain compare 默认在 K3 前 `zero_local_combine(...); dist.barrier()`，而 official top-level staged path 不做该准备；
  - 下一步优先用 chain 的 `PREZERO/PREBARRIER` toggles 复现实验，确认 official 残差是否来自 combine buffer 初值、workspace 复用、或 K3 前跨 rank排序/可见性窗口。

## 2026-06-11 K3 no-tail rowptr glc root cause

- 失败重新复现：
  - 恢复 `fence.acq_rel.sys` 失败实验后的远端 `.so`，source pytest `10 passed`；
  - V3 normal raw/aicc rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_restore_after_fence_fail_20260611_125726.log`；
  - official 8 卡 4096 normal/no-tail default 仍失败，log `hygon_tmp/sglang_debug/v3_official_restore_default_notail_4096_20260611_125904.log`，典型 `max_abs=0.00457763671875`。
- aicc fence/builtin 探针：
  - probe 源码在 `hygon_tmp/sglang_debug/fence_probe.cu`，远端 run dir `hygon_tmp/sglang_debug/fence_probe_20260611_130134/`；
  - `__threadfence_system()` 生成 `buffer_wbinvl1_vol`，但不会自己插 `s_waitcnt vmcnt(0)`；
  - `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` 基本不生成额外 flush；
  - `__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent")` 生成 `buffer_wbinvl1_vol`；
  - `"system"` scope builtin 编译失败：`Unsupported atomic synchronization scope`；
  - 因此上轮 `fence.acq_rel.sys` 失败后，不应继续沿用 AMD system-fence 记忆猜修复；V3 K3 里显式 `wait_vmem_lds_store_device()` 仍是必要的。
- rowptr load 反证与修复：
  - 原 K3 ASM 对 `row_combine_ptrs` 读取使用 `global_load_dwordx2 ... off glc`；V3 C 版此前只把 rowptr store 改为 `global_store_short`，rowptr load 仍是普通 C load；
  - 只把 epilogue/store 侧 rowptr 读取改成 glc 后，official 4096 仍失败且变差，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_store_notail_4096_20260611_130804.log`，典型 `max_abs=0.007598876953125`；
  - 将 active-row mask 与 epilogue/store 两处 `row_combine_ptrs` 读取都改成 `global_load_dwordx2 ... off glc` 并在 load 后 `s_waitcnt vmcnt(0)` 后，official 8 卡 normal/no-tail 4096 通过，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_all_notail_4096_20260611_131155.log`，`max_abs=0.000488281`；
  - 4096 三轮通过，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_all_notail_4096_iters3_20260611_131320.log`，三轮 `max_abs=0.00138855 / 0.000488281 / 0.000488281`；
  - 1024 通过，log `hygon_tmp/sglang_debug/v3_official_rowptr_glc_all_notail_1024_20260611_131421.log`，`max_abs=0.000488281`；
  - 所有通过命令均为 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`，因此不依赖打印或 host sync。
- ISA 证据：
  - save-temps run dir `hygon_tmp/sglang_debug/k3_v3_rowptr_glc_savetemps_20260611_131748/`；
  - `k3_v3_fused_ext-hip-amdgcn-amd-amdhsa-gfx938.s` 统计：`global_load_dwordx2.*glc=396`、`global_store_short=384`、`flat_store_short=0`；
  - 关键片段显示 active-row mask 与 epilogue store 均生成 `global_load_dwordx2 ... off glc` 后接 `s_waitcnt vmcnt(0)`，再执行 `global_store_short`。
- 当前结论：
  - 真实 root cause 是 V3 K3 对 `row_combine_ptrs` 可见性语义没有完整对齐原 K3 ASM：只对齐 store family 不够，必须对齐 rowptr load 的 glc 语义；
  - selective host sync 中 `after_k2 + after_k3_combine` 才通过的现象可解释为：host sync 同时掩盖了 K3 入口 rowptr/K2 输出可见性与 K3 出口 combine 消费窗口；rowptr glc 修复后现有 post-K3 barrier/reduce 已足够通过 no-tail normal official。

## 2026-06-11 K3 no-tail clean 多轮复测失败

- 用户提醒不能接受“加打印/host sync 才过”的路径后，重新运行 clean official 8 卡 4096 normal/no-tail：
  - 环境显式设置 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`、`MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=0`；
  - source guard `10 passed`；
  - 4096 三轮日志 `hygon_tmp/sglang_debug/v3_clean_notail_4096_after_user_bug_20260611_132639.log`；
  - 第 1/3 轮通过，后续 rank 1 出现 `max_abs=0.00389862060546875 > 0.0035`。
- 新证据修正前序结论：
  - rowptr glc 修复不是 no-tail official 多轮稳定性的完整 root cause；
  - 当前失败模式更像同一进程/同一 symmetric buffer 内多轮复用时的状态生命周期问题，而不是单次 K3 same-input compute/layout；
  - 下一轮诊断应优先区分：`y_fused` 未清理/异步覆盖、combine buffer 复用、route_scratch rowptr/active mask 复用、baseline layout/cache 交互、以及 K1/K2/K3 之间的迭代间 device ordering。

## 2026-06-11 K1 metadata release fence 与 no-tail oracle A/B

- clean 多轮失败后的进一步定位：
  - K1 scanner CTAs 会分别写 `expert_counts`、`row_x_ptrs`、`row_x_scales`、`m_indices`、`route_weights`、`output_index`、`row_combine_ptrs` 等 metadata；
  - 原实现里非最后 CTA 在 `atomicAdd(route_flags + kCountDoneSlot/kEmitDoneSlot, 1)` 前只有普通 device-scope fence；
  - 最后 CTA 再 system fence 并发布 tile-ready 不能替其他 CTA release 它们各自写出的 metadata；
  - DCU/Hygon release 参考要求跨 CTA/跨 kernel 可见的发布点在 signal/flag 前使用 system-scope release，因此 K1 完成计数前改为 `__threadfence_system()`。
- 强制重编后的结果：
  - 远端 build 已强制删除 K1 对象并用 aicc 重编，build log `hygon_tmp/sglang_debug/rebuild_v3_k1_metadata_release_forced_20260611_140647.log`；
  - 1024 normal/no-tail clean 三轮通过，log `hygon_tmp/sglang_debug/v3_clean_notail_1024_k1_release_20260611_142052.log`；
  - 4096 normal/no-tail clean 三轮通过，log `hygon_tmp/sglang_debug/v3_clean_notail_4096_k1_release_20260611_142341.log`；
  - 两个命令均显式设置 `MEGAMOE_DCU_V3_NO_TAIL_SYNC=0`、`MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC=0`，因此不是靠 V3 debug host sync 或打印通过。
- baseline oracle A/B：
  - 新增默认关闭的测试诊断开关 `MEGAMOE_DCU_TEST_SKIP_BASELINE_SYNC=1`；
  - V3 1024 skip baseline sync、不 clone fused 会失败，`max_abs=0.00390625`；
  - V3 1024 skip baseline sync + clone fused 通过；
  - 原 ASM staged 1024 skip baseline sync 通过。
- 当前解释：
  - normal/no-tail production V3 的必要修复是 K3 rowptr glc + K1 metadata system release；
  - skip baseline sync 的剩余失败不是生产 runtime 里的 host sync 依赖，而是 correctness harness 中 DeepEP baseline oracle 内部异步/缓冲时序与 V3 更快路径相互作用；
  - 正式 correctness 继续使用默认 oracle 隔离同步来保证比较的是完成后的 fused/baseline 结果；生产 V3 路径没有新增同步 kernel、打印或 host-side bridge。

## 2026-06-11 K3 tail-reduce ASM diff 线索

- V3 normal/tail staged 首次 e2e 失败为 fused nonfinite，1024 tokens log `hygon_tmp/sglang_debug/v3_tail_1024_initial_20260611_145520.log`。
- 失败 A/B：
  - 在 tail wait 后额外做全线程 `buffer_wbinvl1_vol` 仍失败，log `hygon_tmp/sglang_debug/v3_tail_1024_invalidate_20260611_151415.log`，已撤回；
  - tail reduce worker 将 combine `uint4` load 改成 `global_load_dwordx4 ... glc` 仍失败，log `hygon_tmp/sglang_debug/v3_tail_1024_glc_20260611_154554.log`，已撤回；
  - 因此当前不能把问题归因于 reduce worker 普通 `uint4` load 缺 glc，也不能继续叠同步。
- 对比 `K3COMBINE.s` 与 `K3COMBINE_TAILREDUCE.s` 后的关键差异：
  - tail 版在 GEMM combine store 后新增 tail signal/reduce 段，先 `s_waitcnt vmcnt(0)`、`buffer_wbinvl1_vol`、`s_barrier`，再处理 done/signal/reduce；
  - tail 版有 extra reducer WG 分支，`blockIdx` 超过 GEMM workgroups 后进入 reducer，不参与 GEMM；
  - host 侧原 ASM 的 `asm_reduce_blocks` 为 64/128，tail ASM 的 extra reducer 使用 `worker_idx * 0x300 + lane` 起步、以 `reduce_blocks * 0x300` 为 stride，其中 `0x300` 对应 768 lanes/block；
  - 最后一个 GEMM WG 负责 `global_atomic_add ... glc` 到 done counter、向 peer signal slots 发 generation，并在 `reduce_blocks > 0` 时退出；只有 debug/safety fallback 或无 extra reducer 时才用最后 GEMM WG 的 lanes 做 reduce。
- 对 V3 C 的含义：
  - 不能照搬 ASM 主干，但 tail reduce 工作划分语义应对齐：extra reducer blocks 负责 reduce，worker count 是 `reduce_blocks`，每 block 768 threads；最后 GEMM block只做 publish/signal，避免和 extra reducers 重叠写 `y`；
  - 该改动只作用于 epilogue/tail reduce 调度边界，不改变 pure 5pack K3 GEMM load/compute/store 主体。
- reducer 768-lane 调度单独验证失败：
  - log `hygon_tmp/sglang_debug/v3_tail_1024_reducer768_20260611_161622.log`，仍为 fused nonfinite；
  - 因此工作划分不一致不是充分根因。tail ASM 还有一处和 V3 C 不同：extra reducer wait 完成后恢复 full EXEC，执行 `s_barrier` 后再 `buffer_wbinvl1_vol`，而 V3 C wait helper 目前只让 thread0 在 wait loop 里/结束时 invalidate，barrier 后没有 full-wave invalidate。

## 2026-06-11 K3 tail signal wait 语义候选差异

- 为避免继续猜同步语义，重新查本地 DCU KB：
  - Hygon `gfx938` all-gather/GEMM 参考中的 flag wait 常见形态是 `global_load_dword ... glc`、`s_waitcnt vmcnt(0)`、循环中 `buffer_wbinvl1_vol`；
  - Hygon memory microbenchmark 资料说明 `global_load_* ... glc` 偏 L2-oriented/bypass-L1，`global_load_* ... glc slc` 更偏 DRAM-oriented 或更少缓存复用路径，需要实测确认；
  - `system_barrier.hpp` 参考中 DCU_ASM 分支的强 acquire load 可用 `__hip_atomic_load(..., __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)`，但它不等价证明会生成原 ASM tail wait 的 `glc slc` 形态。
- 本地代码差异：
  - 公共 `deep_gemm::mega::load_signal_system()` 当前是 `__hip_atomic_load(..., __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)`；
  - 原 `K3COMBINE_TAILREDUCE.s` 的 `K3_TAIL_WAIT_SIGNAL` 先用 `global_load_dwordx2 ... glc` 读 signal address，再用 `global_load_dword ... glc slc` 轮询 signal value，每次后接 `s_waitcnt vmcnt(0)` 和 `buffer_wbinvl1_vol`；
  - V3 tail 当前 wait helper 用 relaxed system atomic load 轮询 signal value，虽然已有 invalidate，但 load 指令族与原 tail ASM 不一致。
- 判读：
  - 当前 wait-invalidate 补丁已经编译但因远端 `sglang serve` 占满显存还没有有效 e2e 结果；
  - 若该补丁在有效 e2e 中仍失败，下一单变量候选应是把 V3 tail wait 的 signal-value load 改成局部 helper 的 `global_load_dword ... off glc slc` + `s_waitcnt vmcnt(0)`，只作用于 tail signal wait，不改变 GEMM 主循环和 reduce worker 普通 combine load；
  - 在当前补丁未验证前不叠加该改动，避免两个同步变量混在一次结果里。

## 2026-06-11 K3 tail y-split codegen/调度结论

- 单卡 raw 反证链条：
  - `kTailReduce=false` no-tail raw combine 正确，signal-only no-tail raw 也正确；
  - `kTailReduce=true` tail raw 在 y-split 前 combine buffer 本身就是错误值，reduce_y 只是忠实 reduce 了错误 combine；
  - 因此当前问题不是 tail reduce worker 读 combine 的 glc/acquire 单点问题，也不是 self signal 发布本身破坏 combine，而是 tail 专门化下 GEMM/combine CTA 调度或 codegen 形态改变导致的写入错误。
- ISA/调度观察：
  - 线性 tail 版本把 GEMM 和 extra reducer 混在同一个一维 `blockIdx.x` 空间，tail 专门化生成的寄存器/分支压力明显高于 no-tail；
  - noinline wait/reduce helper 没有修复 raw combine，说明简单拆调用边界不是根因。
- 有效修正：
  - 将 extra reducer 改为 y-split：`grid.x=wg_m`、`grid.y=wg_n + reduce_rows`，只有 `tile_row_idx >= wg_n` 的额外 y 行执行 reducer；
  - GEMM CTA 的 `(blockIdx.x, blockIdx.y)`、`tile_token`、`tile_hidden` 与 no-tail 完全同形态，completion owner 也改为 `blockIdx.y * wg_m + blockIdx.x + 1`；
  - 该改动只调整 tail reducer 调度边界，不改变 pure 5pack K3 GEMM load/compute/store 主循环。
- 结果：
  - y-split 后单卡 tail raw topk=1/topk=6 均通过，no-tail raw 无回归；
  - 该结果可以证明低层 tail combine/reduce 数据路径已恢复，但不能替代 8 卡 staged e2e；后续仍需用 tail-reduce 1024/4096 e2e 验证跨 rank signal 和 baseline oracle correctness。

## 2026-06-11 K3 tail stage compare 缩短链路结论

- 完整 e2e 里 V3 normal/tail 1024 第三轮出现 nonfinite 或阈值外残差；为了排除 DeepEP baseline oracle、权重准备和 full wrapper 长链路，新增 stage compare：
  - 同一份 V3 K1 fused + K2 输出；
  - 原 ASM K3 tail 使用原始 L2 layout；
  - V3 K3 tail 使用 pack5 L2 layout；
  - 每轮分别 reset tail done/signal 后跑 ASM 与 V3，直接比较两份 `y`。
- 结果：
  - 8 卡 1024 三轮中，第 1/2 轮 ASM 与 V3 完全一致；
  - 第 3 轮没有 nonfinite，但多个 rank 出现小漂，最大 `0.00390625`；
  - 这说明当前 remaining issue 已从“tail raw combine/reduce 错”收窄到“重复调用时 V3 tail 与 ASM tail 的状态/可见性语义仍有细差”。
- 判读：
  - full e2e baseline oracle 不是唯一根因，因为 stage compare 不调用 baseline 也能复现小漂；
  - 权重 layout 不是粗错，因为前两轮完全一致，且 no-tail same-input 之前已验证；
  - 下一步应围绕 `asm_done_counter[0/1]` reset/owner、tail signal slots generation、combine buffer 覆盖和 V3 reduce worker 读 combine 的 acquire 语义做单变量诊断。

## 2026-06-11 K3 tail worker combine load 指令族更新

- 重新对照原 `K3COMBINE_TAILREDUCE.s` 后，tail reduce 读取 combine buffer 的宏形态是 `global_load_dwordx4 ... off` 后接 `s_waitcnt vmcnt(0)`，不是 `glc`。
- y-split 前的 `glc` A/B 发生在 `kTailReduce=true` 下 combine 本身错误的阶段，不能继续作为 y-split 后的有效反证。
- V3 tail worker 已改为显式 `global_load_dwordx4 ... off + s_waitcnt vmcnt(0)`：
  - 单卡 raw tail smoke topk=1/topk=6 通过，说明该指令族没有破坏低层 combine/reduce 数据路径；
  - full K1+K3 normal rebuild 已确认 K1/K3 V3 normal aicc shim 均生效；
  - 8 卡 stage compare 在 full rebuild 状态下第 2 轮 rank7 漂到 `0.006805419921875`，因此该指令族不能作为 tail staged 修复。
- 当前判断：
  - combine worker load family 不是 sufficient root cause；
  - remaining issue 更像 repeated-call state / signal reset / completion owner / output overwrite / reducer ordering 的细差；
  - 下一轮应先增强 stage compare 证据采集，不继续在 K3 GEMM 主体或 reduce worker load 上叠加猜测性修复。

## 2026-06-11 K3 tail stage compare order/zero-combine 结论

- 临时 stage compare 进一步把 full e2e 链条缩短到 “同一份 V3 K1/K2 输出 + K3 tail ASM/V3 直接对比”：
  - `asm_only` 连续三轮稳定，说明原 ASM tail、tail signal reset 和诊断 harness 本身能承受重复调用；
  - `v3_only` 连续调用会出现小漂，且在 `v3_first` / `asm_first` 顺序下可触发 V3 非有限值；
  - `K3_V3_TAIL_STAGE_ZERO_COMBINE=1` 不能彻底消除 V3 first/V3 only 的不稳定，排除“只是本地 combine buffer 旧值”作为充分根因。
- V3 tail 当前 done/signal 发布与原 ASM tail 的关键差异：
  - 原 ASM tail 由执行 `global_atomic_add` 后判定为最后一个 GEMM WG 的 lane0 直接发布 peer signal；
  - V3 tail 先把 completion owner 写到 `done_counter[1]`，随后所有 GEMM CTA 通过读取 owner slot 判断自己是否 signal；
  - stage compare 里 V3 的 `done_counter[1]` owner 值随轮次/rank 变化，而 ASM 诊断中的第二槽保持 0；这不是功能合同本身，但说明 V3 引入了原 ASM 没有的跨 CTA owner-slot 可见性路径。
- DCU KB 本轮检索到的 Hygon gfx938 参考继续支持保守信号路径：
  - wait loop 常见形态是 `global_load_dword ... glc`、`s_waitcnt vmcnt(0)`、`buffer_wbinvl1_vol`，然后 block 内 `s_barrier`；
  - signal 发布应由确定的发布者在 system-visible store/atomic 之后完成，而不是额外依赖普通 owner slot 让其它 CTA 二次决策。
- 下一步单变量假设：
  - 不改 K3 pure 5pack GEMM 主循环、不改 reduce worker load；
  - 只把 V3 tail signal 改成由 `atomicAdd_system(done_counter, 1)` 判定出的最后 GEMM CTA thread0 直接调用 peer signal；
  - `done_counter[1]` 可继续作为诊断/`reduce_blocks<=0` fallback owner 信息，但不再作为 `reduce_blocks>0` 正常 tail path 的 signal 决策依据。

## 2026-06-11 K3 tail direct-signal A/B 反证

- 将 V3 tail signal 改成“最后 GEMM CTA thread0 直接 signal peers”后，8 卡 `v3_only` stage compare 第 1 轮就出现多 rank nonfinite：
  - log `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_direct_signal_v3_only_20260611_192159.log`；
  - 与旧 owner-slot 版本相比，直接 signal 更早暴露非有限值，说明 owner-slot 不是可直接删除的唯一根因，它可能提供了当前代码偶然依赖的延迟/可见性窗口。
- 该 A/B 已撤回，不作为修复方向。
- 新判断：
  - remaining issue 更可能是 tail signal generation、peer signal slot reset/累计、remote combine 写入可见性、或 reducer wait 后读 combine 的时机组合问题；
  - 下一步应先扩展 stage compare 读取 local signal slots `[8, 15]` 与 done counter，而不是继续修改 K3 GEMM 主循环或 reduce worker load。

## 2026-06-11 K3 tail signal-slot 诊断与 combine 视图修正

- direct-signal 回滚到旧 owner-slot 路径后，增强 stage compare 读取 local tail signal slots `[8, 15]`：
  - `v3_only` 与 `asm_first` 中，V3 launch 前本地 recv slots 均为 0，V3 launch 后 send/recv slots 均到达 generation 1；
  - done counter 到达目标，且 signal slots 没有显示未 reset、generation 丢失或 peer-visible slot 未到达；
  - 因此当前证据不支持“tail signal slot reset/generation 到达失败”作为根因。
- `asm_first` 组合下继续能让 V3 second 触发 nonfinite 或小漂，但 signal slots 仍正常：
  - 这把问题从 signal slot 生命周期进一步压到 V3 tail combine 写入可见性、active-row 覆盖或 reducer 读取语义。
- 新增的 Python `combine_reduce` 诊断目前不能直接作为根因证据：
  - 在 `asm_first` 中，即使原 ASM tail 输出 `y` 是 finite，Python 视图对 combine buffer 的直接求和也可能读到 nonfinite；
  - 这说明诊断视图可能没有完全对齐原 tail reducer 的 active rows、`row_combine_ptrs`、`output_index` 或 `combine_token_offset` 合同；
  - 下一步必须先按 `deep_gemm::comm::SymmetricBuffer::get_sections()`、`combine_token_offset()` 和 K1 metadata 生成方式修正诊断，再判断是否存在 reducer 读早或读错 slot。

## 2026-06-11 Normal tail 暂缓与 LL 切换状态

- 按用户要求，K3 normal tail-reduce 暂缓，不标完成：
  - 已反证方向包括 tail worker load 指令族、direct-signal、signal slot reset/generation；
  - 新增 device post-hoc `reduce_local_combine(... invalidate_before_read=True)` 诊断用于区分 Python combine view 与真实 device-side reduce，但两次远端 torchrun 都在 NCCL/launcher 层失败，没有打印 `rank_stats`，因此没有新的 kernel 结论。
- 后续恢复 normal tail 时，优先继续该 device post-hoc reduce 诊断：
  - 如果 device reduce 与 tail `y` 一致而 Python view 不一致，说明 Python 诊断视图问题；
  - 如果 device reduce 与 tail `y` 不一致，再定位 combine 覆盖、row mapping 或 reducer 读可见性。
- 当前工作焦点切到 fused LL：
  - K1 LL 已通过 K1-only staged gate，但当时为兼容原 K3 ASM 使用 256-row/expert stride；
  - 接 K3 V3 LL 后需要复核是否恢复/保留 true LL 64-row layout，避免把 ASM-compatible 临时 layout 当作最终 LL 性能路径；
  - LL 路径固定先验证 32/128 tokens，先 no-tail，再 tail-reduce。

## 2026-06-11 K3 LL no-tail 编译面收窄

- 第一版曾尝试用已有 V3 fused rowptr kernel 的 `<64, 256, true>` lowlat specialization 作为 K3 LL no-tail；远端 hipcc/dcc 在 `k3_v3_fused_ext.hip` 上运行超过 15 分钟无日志推进，已停止该 build。
- 为避免重复 normal/fused 大模板编译坑，K3 LL no-tail 改为复用 `V3_K3_Pure_LowLatencyMaskedGroupGemmKernel` 主体：
  - 增加固定 `m_per_expert` 模式，不依赖额外 per-expert count metadata；
  - 增加可选 `row_combine_ptrs` store，rowptr 为 0 的 padded/inactive row 自动跳过写回；
  - launcher 只实例化 pure LL blockM 32/48/64 specialization，不再实例化 768-thread fused `<64, 256, true>`。
- 该路径更符合 LL 先保护 pure C GEMM 骨干的约束；性能仍需后续用 32/128 tokens 做 correctness 后再测。

## 2026-06-11 K3 LL no-tail/tail correctness 结论

- K3 V3 LL no-tail 使用 pure LL 5pack 主体的 fixed-row rowptr-store 模式，不实例化此前超时的 768-thread fused specialization；32/128 tokens 三轮 8 卡 correctness 均通过。
- K3 V3 LL tail-reduce 复用同一 pure LL 5pack 主体，在 kernel 内追加跨 rank done/signal 和本地 reduce：
  - reducer block 使用 256 threads，即当前 LL kernel blockDim，而不是 normal tail 的 768 threads；
  - tail signal/done 路径保留 owner-slot 语义，未采用此前 normal tail 已反证的 direct-signal A/B；
  - L2 权重在 V3 LL no-tail/tail 都使用 pack5 layout，K1 LL 不再为了原 ASM K3 fallback 强制 256-row/expert stride。
- 8 卡 correctness：
  - `K3_USE_ASM_TAIL_REDUCE=0`：32/128 tokens 三轮通过；
  - `K3_USE_ASM_TAIL_REDUCE=1`：32/128 tokens 三轮通过；
  - 四个通过命令均为 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll`。
- 仍未覆盖：
  - LL performance gate；
  - LL uneven tokens；
  - LL cuda graph；
  - normal tail-reduce remaining issue。

## 2026-06-11 LL performance gate 初测

- 固定命令设置：8 卡、`hidden=4096`、`intermediate=2048`、`experts=256`、`topk=6`、`correctness_iters=1`、`warmup=2`、`repeat=5`。
- V3 LL no-tail：
  - 32 tokens：`1.1678 ms`，log `hygon_tmp/sglang_debug/v3_ll_notail_32_bench_20260611_205604.log`；
  - 128 tokens：`1.6858 ms`，log `hygon_tmp/sglang_debug/v3_ll_notail_128_bench_20260611_205900.log`。
- V3 LL tail:
  - 32 tokens：`1.1704 ms`，log `hygon_tmp/sglang_debug/v3_ll_tail_32_bench_20260611_205801.log`；
  - 128 tokens：`1.7118 ms`，log `hygon_tmp/sglang_debug/v3_ll_tail_128_bench_20260611_210045.log`。
- `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0` persistent_fused 对照：
  - 32 tokens：`1.0018 ms`，log `hygon_tmp/sglang_debug/base_no_large_32_bench_20260611_205701.log`；
  - 128 tokens：`1.5558 ms`，log `hygon_tmp/sglang_debug/base_no_large_128_bench_20260611_205952.log`。
- 初步结论：
  - V3 LL correctness 已通过，但 performance gate 尚未通过；
  - tail/no-tail V3 timing 接近，说明当前低 token 缺口不主要来自 no-tail 外部 reduce；
  - 需要进入分段/ profiler 定位，优先判断固定 launch/staged metadata 开销、K1 route build、K2、K3 pure LL block 配置和 tail reducer extra blocks 哪一项主导。
- profiler 状态：
  - `hipprof --stats --hip-trace --follow-fork` 在当前 8 卡 VRAM 约 95% 的环境下导致 fixture 的 pack5 权重分配 OOM；
  - 该 profiler 结果不作为 kernel 根因证据，后续等卡空再跑或先用 event 分段。
- 静态候选：
  - V3 LL 32/128 当前 rows/expert 对齐到 64，K1/K3 总行数为 32 local experts * 64 = 2048；
  - K2 默认只有 `num_tokens >= 1536` 才接收 `row_combine_ptrs` 并跳过 inactive rows；
  - 低 token 下真实有效 row 远小于 2048，而 K2 默认仍对 2048 rows 做 SwiGLU/quant；
  - 下一步优先 A/B `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0`，如果有效，再考虑把 V3 LL 默认改为传 `row_combine_ptrs`。
- 次级候选：
  - K1/K3 LL 默认 `ll_block_m=32`，对 64-row/expert layout 每个 expert 需要 2 个 m tile；
  - 已增加默认不变的 `MEGAMOE_DCU_V3_LL_BLOCK_M=32|48|64` A/B 开关，待 GPU 可用时验证 `block_m=64` 是否减少低 token tile 开销。

## 2026-06-11 LL blockM A/B plumbing 修正

- 静态检查发现 `large_opt.py` 已在 V3 LL K3 tail/no-tail 两个分支设置 `k3_kwargs["ll_block_m"] = _v3_ll_block_m()`，但 `K3_fused/k3_fused.py` 中 V3 wrapper `k3_l2_fused_v3_to_combine()` 的签名缺少 `ll_block_m`。
- 影响：
  - 默认 `MEGAMOE_DCU_V3_LL_BLOCK_M=32` 的路径如果重新从最新本地文件运行，会在 Python 参数绑定处失败；
  - `MEGAMOE_DCU_V3_LL_BLOCK_M=64` A/B 更无法启动，无法判断 K1/K3 LL tile 数是否是低 token 性能缺口来源。
- 修复：
  - V3 K3 wrapper 补 `ll_block_m: int = 32`，并继续传入 `k3_v3_ll_combine_raw` / `k3_v3_ll_combine_tail_raw`；
  - source guard 增加 stage timing、LL blockM 默认值/合法值、K1/K3 wrapper 参数传递检查。
- 验证：
  - 本地 compileall 与 diff check 通过；
  - 远端容器 source guard `10 passed in 4.31s`。
- blockM A/B 解释风险：
  - K1 LL launch 目前只在 `ll_block_m=32` 且 `valid_rows_per_expert <= 16` 时实例化 `kMaskTinyStore=true`；
  - `ll_block_m=48/64` 当前固定 `kMaskTinyStore=false`，因此 32-token 低负载场景会多写 padded L1 rows；
  - 后续 `MEGAMOE_DCU_V3_LL_BLOCK_M=64` A/B 必须先看 correctness，再结合 K2 skip 和 stage timing 解释性能；如果 block64 更慢，不能直接判定“减少 m tile 无效”，还要考虑 padded store 开销。

## 2026-06-11 V3 graph fail-fast 边界

- 当前 V3 eager LL 已有 K1/K3 no-tail/tail correctness，但 graph contract 尚未对齐：
  - K1 V3 graph wrapper 仍是显式 `NotImplementedError`；
  - K3 V3 wrapper 不接受 `active_tiles` / `graph_runtime_offset_from_active_tiles`；
  - 原 `large_opt.py` graph 分支在 K1 之后仍写着 K3 ASM fallback。
- 风险：
  - V3 测试/fixture 在 V3 K3 启用时会把 L2 权重切成 pack5；
  - 如果后续只先放开 K1 V3 graph，而 K3 graph 仍落到原 ASM fallback，就会形成 pack5 L2 layout 与原 ASM layout 的错配。
- 当前处理：
  - 在 `_run_large_opt_3stage_graph()` 入口对 `v3_backend is not None` 直接 fail-fast；
  - 这不是 graph 功能完成，只是把未完成边界从 K1 wrapper 抬到 graph staged 入口，避免半接状态被误用。

## 2026-06-12 LL performance tuning 启动

- 按用户要求，优化顺序切为：先 LL performance，达标后 normal performance，最后补 uneven/graph/normal tail 等未完成项。
- 本轮 DCU KB 检索主题为 `Hygon gfx938 MegaMoE LL fused K1 K2 K3 low token padding inactive rows blockM stage timing`：
  - DeepEP LL overlap 资料强调 combine/dispatch overlap 应由细粒度 compute signal/chunk threshold 驱动，说明后续如果固定 launch/阶段时间仍高，应优先找 K1/K3 内部 tile-ready/compute-ready 粒度，而不是粗粒度全 expert 等待；
  - DeepGEMM grouped FP8 资料确认当前形状和 pack5 C groupgemm 来源仍是正确参考面；
  - MoE block size/padding 资料支持对 low-token padding 和 blockM 做 A/B，而不是先大重写 kernel。
- 当前静态首要候选保持不变：
  - V3 LL 32/128 使用 64 rows/expert，32 local experts 形成 2048 row capacity；
  - 默认 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=1536` 使低 token 不传 `row_combine_ptrs` 给 K2，K2 会处理大量 padded/inactive row；
  - 因此第一轮 A/B 仍是 `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0`，第二轮才看 `MEGAMOE_DCU_V3_LL_BLOCK_M=64`。
- 远端状态：
  - `hy-smi` 显示 HCU 0%，但 8 卡 VRAM 仍 95-96%；
  - host 上 `sglang serve` PID 166043 及 scheduler 子进程仍在占显存；
  - 按工作流约束不擅自 kill 服务，因此暂不运行 8 卡 fixture/bench。
- 已将 `run_v3_ll_perf_ab.sh` 改成带 VRAM guard 的 A/B runner：
  - 默认 `MAX_VRAM_PCT=90`，当前 96% 会在 fixture allocation 前退出，避免重复 OOM；
  - 支持 same-run default / K2 skip / blockM64 / 可选 baseline / 可选 stage timing；
  - 每个 case 额外写 `*_perf_*.json`，由 `summarize_v3_ll_perf.py` 汇总。
- blockM A/B 的静态干扰项：
  - K3 LL pure-rowptr launcher 对所有 blockM 都实例化 `kMaskTinyStore=true`；
  - K1 LL launcher 此前只有 blockM=32 在 `valid_rows_per_expert <= 16` 时启用 tiny-store mask，block64 会多写 padded L1 rows；
  - 已只给计划要测的 K1 block64 增加 `kMaskTinyStore=true` 分支，尽量减少 block64 A/B 中的 padded-store 噪声，同时不扩 block48 编译面。

## 2026-06-12 LL no-tail A/B 与 stage timing 结论

- 用户授权后已杀掉占卡 `sglang serve`，8 卡 VRAM 从 95-96% 降到 0%，随后运行 `hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`。
- no-tail 32/128 A/B 结果：
  - default 32: `1.1771 ms`；`K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 32: `1.1835 ms`，略慢；
  - default 128: `1.7210 ms`；`K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0` 128: `1.6959 ms`，只小幅改善；
  - `K2_SKIP_INACTIVE_ROWS_MIN_TOKENS=0 + MEGAMOE_DCU_V3_LL_BLOCK_M=64` 明显退化：32 `2.8951 ms`，128 `3.5302 ms`。
- 结论：
  - 低 token LL 当前缺口不主要来自 K2 inactive row padding；
  - block64 分支不是当前优化方向，至少在当前 K1/K3 LL 实例化和 codegen 下不能减少端到端耗时；
  - 下一步不要继续盲调 blockM，应转向 K3/K1 分段和 pure-vs-fused 差异。
- stage timing 解析（p50，no-tail）：
  - 32 default：K1 `0.417 ms`，K2 `0.028 ms`，K3 combine `0.669 ms`，no-tail barrier `0.044 ms`，reduce `0.012 ms`；
  - 32 k2skip：K1 `0.415 ms`，K2 `0.027 ms`，K3 combine `0.667 ms`，no-tail barrier `0.041 ms`，reduce `0.012 ms`；
  - 128 default：K1 `0.457 ms`，K2 `0.028 ms`，K3 combine `1.136 ms`，no-tail barrier `0.049 ms`，reduce `0.014 ms`；
  - 128 k2skip：K1 `0.456 ms`，K2 `0.028 ms`，K3 combine `1.141 ms`，no-tail barrier `0.047 ms`，reduce `0.014 ms`。
- 当前主耗时排序：
  - K3 LL combine 是第一优化对象；
  - K1 LL 是第二对象；
  - K2 skip inactive 和 external reduce 暂时不是主瓶颈。

## 2026-06-12 K3 LL rowptr vector-store 优化结论

- 改动：
  - `K3_fused/k3_v3_pack5_groupgemm_impl.cuh` 在 LL epilogue 增加 `store_bf16x4_rowptr_device()`；
  - 每个 `bf16x4_t` 从原先 4 次 `global_store_short` 改成 1 次 `global_store_dwordx2`；
  - 仍然通过 `row_combine_ptrs` 做 combine 写回，不新增 runtime kernel，不改变 pure 5pack GEMM 主循环，只改 rowptr epilogue store 粒度。
- 验证：
  - 本地 `py_compile`/source guard/diff check 通过；
  - 远端容器 source guard `tests/test_dcu_megamoe_v3.py` 通过；
  - 远端 K1/K3 LL raw rebuild 后 import sanity 正常；
  - `dccobjdump` 对 Python extension `.so` grep `global_store_dwordx2` 未输出可用行，记为 inconclusive，不作为 ISA 失败。
- 8 卡 no-tail stage timing，命令含 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=0 MEGAMOE_DCU_V3_STAGE_TIMING=1`：
  - 32 tokens last3 median：total `0.9265 ms`，K1 `0.4125 ms`，K2 `0.028 ms`，K3 combine `0.399 ms`，no-tail barrier `0.039 ms`，reduce `0.012 ms`；
  - 128 tokens last3 median：total `1.0660 ms`，K1 `0.452 ms`，K2 `0.028 ms`，K3 combine `0.5005 ms`，no-tail barrier `0.0315 ms`，reduce `0.014 ms`；
  - 对比优化前 stage timing：K3 combine 从 32 tokens 约 `0.669 ms` 降到 `0.399 ms`，128 tokens 从约 `1.136 ms` 降到 `0.5005 ms`。
- 8 卡 e2e timing：
  - no-tail default 32：`1.048-1.054 ms` 量级；no-tail default 128：`1.190-1.191 ms` 量级；
  - tail 32/128 在同轮短测中为 `0.913/1.068 ms`，说明 tail 与 no-tail 的小差异需要单独按同一 repeat/warmup 配置复核，不能把不同轮次混成一个结论。
- 当前判断：
  - K3 LL rowptr scalar store 是有效瓶颈，vector-store 已收回大部分 K3 combine 额外成本；
  - 优化后 K1 与 K3 已接近，下一步不能继续单盯 K3，需要建立 K1/K3 fused 实际链路 vs pure C groupgemm 的分项 delta，再决定压 K1 dispatch/metadata 还是继续压 K3 epilogue。

## 2026-06-12 LL pure-vs-fused 基线与 K3 rowptr-load hoist

- 新增诊断入口：
  - `k3_v3_ll_pure_raw(...)` 只暴露已有 K3 LL pure/groupgemm launcher 的 contiguous output 形态，用于诊断 pure baseline；
  - `hygon_tmp/sglang_debug/bench_k3_ll_pure_raw.py` 对同一 LL K3 kernel 比较 contiguous pure 与本地 rowptr combine，不进入生产 runtime/bench 路径。
- K1 pure LL 基线：
  - 使用 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 原始 C LL harness；
  - 32 tokens 约 `0.300 ms`，128 tokens 约 `0.308 ms`；
  - 对照 rowptr-load hoist 后 staged K1 约 `0.416/0.453 ms`，delta 约 `+0.116/+0.145 ms`。
- K3 LL raw/local 基线：
  - vector-store 后、rowptr-load hoist 前，contiguous pure 32/128 约 `0.2457/0.2457 ms`，local rowptr combine 约 `0.2652/0.2943 ms`；
  - rowptr-load hoist 后，contiguous pure 32/128 约 `0.2453/0.2454 ms`，local rowptr combine 约 `0.2334/0.2385 ms`；
  - local rowptr 比 contiguous pure 更快是因为 inactive rows 直接跳过 BF16 convert/store，这个结果只能作为 rowptr epilogue 成本参考，不能解读成 GEMM 本体更快。
- K3 LL rowptr-load hoist 改动：
  - 每个 logical row 只执行一次 `global_load_i64_glc_device(row_combine_ptrs + logical_row)`；
  - `row_addr <= 0` 时跳过后续 BF16 convert/store；
  - rep loop 内复用 `row_addr` 调 `store_bf16x4_rowaddr_device(...)`，没有改 pure 5pack GEMM 主循环。
- 8 卡 no-tail staged 短测：
  - 32 tokens e2e `0.9831 ms`，stage last3 median total `0.865 ms`，K1 `0.416 ms`，K3 `0.336 ms`；
  - 128 tokens e2e `1.1237 ms`，stage last3 median total `0.9835 ms`，K1 `0.4525 ms`，K3 `0.4205 ms`；
  - 相比 vector-store 后，K3 从 `0.399/0.5005 ms` 继续降到 `0.336/0.4205 ms`。
- 8 卡 tail staged 短测：
  - `K3_USE_ASM_TAIL_REDUCE=1` 下 32 tokens e2e `0.8285 ms`，128 tokens e2e `0.9955 ms`；
  - tail 当前明显快于 no-tail，主要因为省掉外部 no-tail barrier/reduce 链路；后续仍需按同一 repeat/warmup 配置复核。
- 当前 delta 排序：
  - 128 tokens：K3 staged vs local rowptr raw 仍约 `+0.182 ms`，K1 staged vs pure 约 `+0.145 ms`；
  - 32 tokens：K1 staged vs pure 约 `+0.116 ms`，K3 staged vs local rowptr raw 约 `+0.103 ms`；
  - 按“优先优化差异大的”原则，下一轮先继续 K3 128 通信链路/remote rowptr store 差异定位，再回到 K1 128 dispatch/metadata。

## 2026-06-12 LL K1 source-rank row grouping A/B 反证

- 假设来源：
  - DCU KB/flux reduce-scatter 参考提示 rank-aware tile/swizzle 与 epilogue scatter 可以降低跨 rank scatter 抖动；
  - V3 K1 LL 当前按 `atomicAdd(symm_counts + expert)` 生成 expert 内行顺序，source rank 会交错，K3 通过 `row_combine_ptrs` 写回时目标 rank 可能频繁切换。
- A/B 改动：
  - 增加临时 `MEGAMOE_DCU_V3_LL_RANK_GROUP_ROWS=1`，在每个 expert 内按 `source_rank` 做二级 count/prefix/emit；
  - 额外 scratch 为 `2 * local_experts * num_ranks`，只影响临时 A/B 分支。
- 验证结果：
  - 初次 rebuild 暴露 kernel signature 少传 `rank_group_rows`，修正后 LL raw rebuild 通过；
  - 8 卡 no-tail 32/128 correctness 均通过；
  - rank-group on 的 e2e：32 `1.0067 ms`，128 `1.1255 ms`，对比 rowptr-load hoist 后默认路径 32 `0.9831 ms`、128 `1.1237 ms` 没有收益；
  - stage timing：32 K1/K3 约 `0.4215/0.337 ms`，128 K1/K3 约 `0.464/0.425 ms`，均不优于默认路径。
- 结论：
  - source-rank row grouping 在当前 K1 LL 生成 metadata 的额外开销抵消了潜在 K3 store locality 收益；
  - 该临时分支已撤回，后续不沿这个方向继续调；
  - 撤回后重编并直接跑 correctness/perf，32/128 均 correct，e2e `0.9883/1.1061 ms`，stage last3 K1/K3 为 32 `0.411/0.335 ms`、128 `0.452/0.4185 ms`，确认恢复到 hoist 后基线量级。

## 2026-06-12 LL K3 rowptr `global_store_dwordx4` lane-pair A/B 反证

- 假设：
  - K3 LL staged 与 local rowptr raw 的最大差异集中在实际 remote/staged rowptr combine store；
  - 尝试把同一 row 上相邻 `ld_col` lane 的两个 `bf16x4` 合并为一个 `global_store_dwordx4`，减少 remote store 指令数。
- 结果：
  - 该 inline asm 形态在 gfx938/hipcc 下可编译，build log 为 `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_storex4_20260612_105627.log`；
  - 直接 8 卡 e2e correctness 在 32 tokens 第一轮失败，`fused/baseline max_abs=0.056884765625 > 0.0035`；
  - 初始判断为 lane-pair/shuffle 或 `global_store_dwordx4` 数据打包方式不能等价替换每 lane 的 `global_store_dwordx2`。
- 进一步探针：
  - 新增 `hygon_tmp/sglang_debug/probe_global_store_dwordx4.cu` 作为诊断-only 探针；
  - 单线程固定 pattern 写入证明 `global_store_dwordx4` helper 的 dword 顺序正确；
  - lane-pair 探针复现初版问题：`__shfl` 放在 only-even-lane 分支内时，source odd lane 没有参与同一条 shuffle，peer 值读为 0；
  - 将 `__shfl` 移到所有 lane 共同执行后，lane-pair 探针通过。
- 修正后 A/B：
  - K3 中改成所有 lane 先执行 `__shfl`、仅 even `ld_col` lane 执行 `global_store_dwordx4`；
  - 8 卡 direct correctness 32/128 通过，但性能退化：e2e `1.0054/1.1601 ms`，stage K3 last3 为 32 `0.346 ms`、128 `0.4265 ms`；
  - 对比 dwordx2 基线 e2e `0.9873/1.1170 ms`、stage K3 `0.337/0.4185 ms`，说明减少 store 指令不抵消额外 shuffle/pack/codegen 成本。
- 处置：
  - 最终撤回 `global_store_u32x4_device` / `store_bf16x8_rowaddr_device` 和 lane-pair store 逻辑；
  - 最终恢复重编 log 为 `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_storex4_final_revert_20260612_111143.log`；
  - 恢复后 direct 32/128 correctness 通过，e2e `0.9873/1.1170 ms`，stage last3 K1/K3 为 32 `0.4145/0.337 ms`、128 `0.456/0.4185 ms`。

## 2026-06-12 LL K3 staged/local/remote rowptr split 诊断

- 目的：
  - 验证 K3 128 staged 比 local rowptr raw 多出的约 `0.18 ms` 是 kernel epilogue 可优化开销，还是跨 rank remote combine store 本身。
- 诊断方法：
  - 只改 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py` 诊断脚本，不改生产 kernel；
  - 同一份 V3 K1/K2 输出下，按本 rank combine 地址范围把 `row_combine_ptrs` 拆成 `staged_local_only`、`staged_remote_only` 和 `rowptr_all_zero`；
  - K3 仍调用同一个 `k3_v3_ll_combine_raw` kernel，只改变诊断 rowptr 输入。
- 128 tokens 结果：
  - 平均 active rows/rank `768.0`，local rows/rank `32.75`，remote rows/rank `735.25`；
  - `pure_contiguous` median avg rank `0.2449 ms`；
  - `local_rowptr` median avg rank `0.2461 ms`；
  - `rowptr_all_zero` median avg rank `0.2311 ms`；
  - `staged_local_only` median avg rank `0.2339 ms`；
  - `staged_remote_only` median avg rank `0.4071 ms`；
  - `staged_rowptr` median avg rank `0.4084 ms`。
- 结论：
  - K3 128 的 staged delta 几乎完全由 remote combine store rows 贡献；
  - local rowptr 路径已经接近 pure/raw，K3 GEMM 主体和本地 rowptr epilogue 不是当前主要问题；
  - `global_store_dwordx4` 和 source-rank row grouping 已被反证，后续不再重复这些 K3 store-width/grouping 方向，除非有新的 profiler/ISA 证据；
  - LL 性能优化焦点转向 K1 128 的 dispatch/metadata/staged-load 额外成本。

## 2026-06-12 LL K1 unused local mask/tail token A/B

- 依据：
  - V3 large_opt/K2/K3 后续链路只消费 K1 返回的 `l1_out`、`route_weights`、`m_indices`、`output_index`、`row_combine_ptrs`；
  - `local_topk_mask` / `tail_tokens` 是 V2-style metadata，当前 V3 staged path 不消费；
  - DCU KB 没有给出可无证据删除跨 rank barrier 的条目，因此本次不动 K1 前 rank barrier，也不动 staged rows/route metadata 同步，只跳过 unused optional metadata 和其后 barrier。
- 改动：
  - LL launcher 对 `dcu_megamoe_v3_launch_k1_ll_symm_stage_raw` 传 `nullptr` 给 `local_topk_mask` / `tail_tokens`；
  - `v3_k1_build_ll_stage_device` 只在 optional metadata 指针非空时执行该段 grid barrier。
- 8 卡验证：
  - rebuild log：`hygon_tmp/sglang_debug/rebuild_v3_ll_k1_skip_unused_mask_20260612_112821.log`；
  - no-tail 32/128 direct e2e correctness 通过；
  - e2e：32 `0.9795 ms`，128 `1.1230 ms`；
  - stage last3：32 total/K1/K3 `0.8675/0.408/0.3355 ms`，128 total/K1/K3 `0.987/0.448/0.4215 ms`。
- 结论：
  - K1 stage 小幅改善，约 `6-8 us`；
  - e2e 128 仍在噪声范围，主差距不在 optional mask/tail metadata；
  - 当前 K1 主要 delta 仍应继续定位 route scan、source row staging copy、metadata publish/wait 与 GEMM 主体耦合成本。

## 2026-06-12 LL K1 source pointer / scale route-stage hoist

- 假设：
  - K1 LL staged copy loop 原先每个 16B vector 都通过 `source_rank/source_token` 重新 `get_sections(peer_sym_buffers[source_rank], ...)`；
  - 单独的 scale staging loop 也按 row 再次 `get_sections`；
  - route emit 阶段已经持有 `sections.x` / `sections.x_sf` 和 `token_idx`，适合把 source row pointer 与 scale 直接写到 staged metadata，避免后续重复计算。
- 改动：
  - `route_scratch_i32 + kExperts` 后的 `2 * row_capacity` int32 scratch 由 `{source_rank, source_token}` 改为一个 int64 source `x` row pointer；
  - route emit 时写 `symm_src_x_ptrs[row]` 和 `staged_x_scale[row]`；
  - init 阶段给 `staged_x_scale` 写默认 scale，padded rows 不再需要独立 scale loop；
  - staged copy loop 直接从 source pointer 加 `vec_col` 读 FP8 row vector。
- 验证注意：
  - 第一次只改 `.cuh` 后，远端 ninja 没有感知 header 依赖，`k1_v3_fused_ext.o` 未重编；已显式删除 object 并 touch `.hip` 后强制重编，真实 rebuild log 为 `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_stage_ptr_hoist_forced_20260612_113411.log`。
- 8 卡 no-tail：
  - 32 e2e `0.9729 ms`，stage last3 total/K1/K3 `0.858/0.402/0.338 ms`；
  - 128 e2e `1.1115 ms`，stage last3 total/K1/K3 `0.9875/0.452/0.4195 ms`；
  - correctness 32/128 均通过。
- 8 卡 tail:
  - 32 e2e `0.9502 ms`，stage last3 total/K1/K3tail `0.8285/0.4055/0.362 ms`；
  - 128 e2e `1.1181 ms`，stage last3 total/K1/K3tail `0.9915/0.448/0.4765 ms`；
  - correctness 32/128 均通过。
- 结论：
  - 该 hoist 对 32-token K1 stage 有可见收益，对 128-token K1 stage 收益较小并有噪声；
  - e2e no-tail 32/128 均优于上一基线，tail 32 也明显优于此前短测，当前保留；
  - K1 128 剩余差距主要不再是重复 `get_sections`，更可能来自全 route scan、atomic row allocation、staged_x 写入量或 GEMM launch/tile 固定成本。

## 2026-06-12 LL K1 skip padded staged_x zero-write A/B 反证

- 假设：
  - K1 LL staged copy 会给 `ceil(symm_counts[expert] / blockM) * blockM` 范围内的 padded rows 写零；
  - K2 对 `row_combine_ptrs[row] == 0` 或 `route_weight == 0.0f` 有提前返回/写零保护；
  - 因此可以尝试只 staging 有效 route rows，减少低 token 下 padded staged_x 写量。
- A/B 改动：
  - `K1_fused/k1_v3_groupgemm_impl.cuh` 的 staged copy 条件临时改为 `row_in_expert < symm_counts[expert]`，跳过 rounded padded rows 的 zero-write。
- 验证：
  - forced rebuild log：`hygon_tmp/sglang_debug/rebuild_v3_ll_k1_skip_padded_stage_zero_20260612_114307.log`；
  - no-tail 32/128 correctness 通过；
  - e2e：32 `0.9712 ms`，128 `1.1199 ms`；
  - stage last3：32 total/K1/K3 `0.8545/0.405/0.336 ms`，128 `0.987/0.454/0.422 ms`。
- 结论：
  - 与 pointer/scale hoist 基线相比，K1 stage 没有改善并略退，128 e2e 也从 `1.1115 ms` 退到约 `1.12 ms`；
  - 该方向已撤回，不继续跑 tail；
  - 保留 padded staged_x zero-write，后续 K1 delta 优先看 route scan/atomic allocation/metadata publish-wait 或 GEMM fixed tile cost。

## 2026-06-12 LL K1 compact route-scan stride A/B

- 假设：
  - LL 32/128 fixed-size eager 下 `runtime_num_tokens` 已知，但 route emit 原先按 `num_ranks * num_max_tokens_per_rank * topk` 全量扫描；
  - 在 32/128 tokens、`num_max_tokens_per_rank=384` 时分别会扫描约 12x/3x 空 route slot；
  - 可以把 scan stride 收窄到真实 `runtime_num_tokens`，同时保持 `output_index` 初始化/写回仍按 `num_max_tokens_per_rank` layout，避免破坏后续合同。
- 改动：
  - `v3_k1_build_ll_stage_device` 中引入 `route_token_stride = clamp(runtime_num_tokens, 0, num_max_tokens_per_rank)`；
  - route emit 循环只扫 `num_ranks * route_token_stride * topk`；
  - `output_index` 写回使用 `source_rank * num_max_tokens_per_rank * topk + token_idx * topk + topk_slot`，layout 不变；
  - `runtime_num_tokens < 0` 时保留 max-token fallback，避免 graph/未来 runtime-token path 被本次 eager A/B 改坏。
- 8 卡 no-tail correctness/perf：
  - 32 correct，e2e `0.9843 ms`，stage last3 total/K1/K3 `0.8515/0.3985/0.337 ms`；
  - 128 correct，e2e `1.1037 ms`，stage last3 total/K1/K3 `0.986/0.444/0.421 ms`。
- 8 卡 tail correctness/perf：
  - 32 correct，e2e `0.9504 ms`，stage last3 total/K1/K3tail `0.8275/0.4015/0.3635 ms`；
  - 128 correct，e2e `1.1044 ms`，stage last3 total/K1/K3tail `0.9945/0.4485/0.4795 ms`。
- 结论：
  - 该 A/B 对 K1 128 有小幅正收益，对 32 处在噪声/轻微退化区间；
  - 因 correctness 通过且 K1 stage 没有变坏，先保留；
  - 主 gap 没有解决，下一步仍需按 pure-vs-fused delta 优先看 K3 128 remote combine store 和 K1 fixed tile/metadata 成本。

## 2026-06-12 Flux / DeepEP 通信隐藏检索结论

- 本次按用户要求查询本地 DCU KB：
  - `Report_flux_reducescatter_gemm_comm_compute_fusion_gfx936.md`；
  - `Summary_DeepEP_Flux_Hygon_Communication_Compute_Fusion.md`；
  - `Report_deepep_low_latency_dispatch_combine_overlap_gfx936.md`；
  - 以及 Flux `reduce_scatter` 源码中的 `epilogue_reduce_scatter.hpp`、`epilogue_vectorized_reduce_scatter.hpp`、`gemm_v2_reduce_scatter.hpp`、`tile_scheduler/*`。
- 可借鉴点：
  - Flux GEMM+RS 把分布式输出布局表达在 epilogue store 中，而不是 GEMM 后再做额外 rewrite；这与当前 K3 rowptr epilogue 方向一致。
  - Flux 的 rank-aware tile swizzle 把 tile ownership 与 rank/通信拓扑绑定；但我们此前 K1 source-rank row grouping 已被反证，因此后续若做 swizzle，应在 K3 tile ownership/epilogue 侧做短 A/B，而不是重复 K1 row grouping。
  - DeepEP LL combine overlap 使用 `comp_signal`、`block_m`、`threshold` 做 chunk-level readiness，而不是等整个 expert 或整轮 GEMM 完成；映射到 V3 的方向是让 K3 combine/tail 的 remote writeback 与 tile/epilogue 粒度更紧，而不是加独立 kernel。
  - Flux 还区分 intranode/across-node/no-NVLink/fused-reduction capability；映射到本仓库即所有通信调度变化都要按 32/128、tail/no-tail、local/remote rowptr split 分开测。
- 当前落地优先级：
  - 不再重复 `global_store_dwordx4`、source-rank grouping 这类已反证方向；
  - 下一步优先检查 K3 LL tile 到 `row_combine_ptrs` 目标 rank 的映射，尝试最小可回滚的 tile-order/epilogue-readiness A/B；
  - 若 K3 仍被 remote store 本身锁死，再回到 K1 的 route/metadata/GEMM fixed-cost。

## 2026-06-12 LL K3 tile-level rowptr readiness A/B 反证

- A/B 内容：
  - 在 K3 LL 主 kernel 的每个 tile 开始处预读本 tile 的 `row_combine_ptrs` 到 shared memory；
  - 若该 tile 所有 rowptr 都是 inactive，则在 GEMM 前 `continue`，试图把 DeepEP/Flux 的 chunk-level readiness 映射为 empty tile skip；
  - epilogue 复用 shared row address，避免重复 rowptr load。
- 结果：
  - 远端重编成功，log 为 `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_tile_rowptr_ready_20260612_125416.log`；
  - no-tail 32 tokens 前几轮出现 K3 stage 下降信号，但同一轮随后触发 staged rank barrier timeout，多个 rank abort；
  - 典型报错为 `MegaMoE HIP staged rank barrier timeout: rank=2 ticket=184 generation=24 arrival=191 release=23`，torch spawn 随后终止进程。
- 结论：
  - 该实现不能保留：即使早期 K3 timing 有收益，也破坏了当前 no-tail 通信/可见性链路的稳定性，或引入了会让后续 rank barrier 无法收敛的硬件/同步错误；
  - 已撤回到已验证的 rowptr-load hoist 基线；
  - 后续继续借鉴 Flux/DeepEP 时，不再做“全空 rowptr 直接跳过 GEMM tile”的语义变化；优先尝试不改变 tile 完成语义的 epilogue 预取、store 排布或诊断-only tile/remote 分布统计。

## 2026-06-12 LL K3 rowptr register-prefetch A/B

- A/B 内容：
  - 不再做 empty-tile skip，不改变 K3 tile 完成语义、tail signal 或 no-tail external reduce 合同；
  - 在每个 K3 LL tile 开始处，为当前 lane 负责的 `kMRepeats` 行提前读取 `row_combine_ptrs` 到 `row_addr_prefetch[]`；
  - epilogue 复用寄存器中的 row address，仍保留 `row_addr <= 0` inactive row skip；
  - GEMM 主循环、remote store helper 和 no-tail/tail completion 语义都不改。
- 验证：
  - 远端 K3 LL raw rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_rowptr_reg_prefetch_20260612_130658.log`；
  - no-tail 32/128 correctness 通过，e2e `0.9137193784/1.0727392435 ms`；
  - no-tail stage last3 total/K1/K3 `0.769/0.401/0.2575 ms`、`0.956/0.4415/0.379 ms`；
  - tail 32/128 correctness 通过，e2e `0.8936993778/1.0852793753 ms`；
  - tail stage last3 total/K1/K3tail `0.787/0.4015/0.318 ms`、`1.000/0.442/0.4605 ms`。
- 结论：
  - 该 A/B 有稳定正收益并保留；
  - 对 no-tail 32/128，K3 stage 相比撤回基线约 `0.336/0.419 ms` 降到 `0.2575/0.379 ms`；
  - 说明当前 K3 LL 低 token 的一部分 delta 确实来自 rowptr metadata load 与 epilogue store 节奏未充分重叠，而不是只有 remote store 带宽本身；
  - 128 no-tail 仍有约 `0.14 ms` 级 staged-vs-local gap，后续可继续看 K3 remote store 指令调度/occupancy，或转向 K1 128 `+0.14 ms` delta。

## 2026-06-12 LL K1 output_index bounded-init A/B 反证

- 假设：
  - K1 LL route scan 已收窄到真实 token stride；
  - `output_index` 清表仍按 `num_ranks * num_max_tokens_per_rank * topk` 全量写 `-1`；
  - fixed-size eager 后续 K2/K3 表面只依赖真实 token 范围，因此可以只清每 rank 前缀，保留写回 layout 的 `num_max_tokens_per_rank` stride。
- 结果：
  - 正确 raw rebuild 后，no-tail 32/128 短链条在 32 tokens 首轮触发 HIP VMFault，并导致 staged rank barrier timeout；
  - 典型 log：`hygon_tmp/sglang_debug/v3_ll_k1_output_index_bounded_notail_20260612_133739.log`。
- 结论：
  - 当前 `output_index` 全量清表属于保守合同，不能只根据 Python 层直接消费关系收窄；
  - 该方向已撤回，后续 K1 metadata 优化应优先看 route scan/atomic/staged copy/GEMM fixed tile，而不是缩小 `output_index` 清零范围。

## 2026-06-12 LL K1 direct source A-load A/B 反证

- 假设：
  - K1 LL staged/pure delta 中，`staged_x` copy pass 是一个主要候选成本；
  - route 阶段已经记录 source row pointer 和 scale，GEMM A-load 可以直接从 source pointer 读取，从而把 dispatch-pull 更直接地藏进 GEMM load；
  - 这比额外 staging pass 更贴近 Flux/DeepEP 的通信/compute overlap 思路。
- A/B 实现：
  - 保留 route metadata、`row_combine_ptrs`、`output_index` 全量清表和 GEMM compute 主体；
  - `kDirectSymmLoad` 分支跳过 `staged_x` copy，A-load 通过 `symm_src_x_ptrs[logical_row]` 读 source row；
  - direct 分支编译通过。
- 结果：
  - no-tail 32/128 correctness 通过；
  - e2e 退到 `1.3673/1.9367 ms`；
  - 128 K1 stage 退到 `1.16-1.34 ms` 区间，显著慢于 register-prefetch 基线约 `0.44 ms`。
- 结论：
  - source row pointer 直接散读会严重破坏 A-load coalescing/latency hiding，远慢于当前 staging pass；
  - K1 通信融合不能简单把 staging copy 去掉，必须保留或重构成更 coalesced 的 staging/load 形态；
  - 该 A/B 已撤回。
- 工具状态：
  - `dccobjdump` 对当前 K1 `.so` / `.o` 未展开 device ISA，只输出 host ELF header；ISA evidence 暂记 inconclusive。

## 2026-06-12 Flux / CUDA MegaMoE 复查与下一步映射

- 复查本地 DCU KB：
  - `Flux GEMM reduce scatter communication compute fusion epilogue tile scheduler Hygon gfx936 gemm rs` 命中 `Summary_DeepEP_Flux_Hygon_Communication_Compute_Fusion.md` 和 `Report_flux_reducescatter_gemm_comm_compute_fusion_gfx936.md`；
  - 初次 `CUDA MegaMoE grouped GEMM communication overlap dispatch combine fused kernel` 查询被本地 reranker CUDA OOM 打断；禁用 `CUDA_VISIBLE_DEVICES` 后重查成功；
  - CUDA/MegaMoE 方向的强匹配较弱，主要返回 grouped GEMM expert dispatch / problem descriptor 与 Hygon DeepEP/Flux summary，而不是新的通信隐藏实现。
- 可确认的复用原则：
  - Flux 的最强映射仍是把 scatter/reduce 语义放在 epilogue store path，并让 scheduler 知道 rank/world/scatter pointer，而不是做额外 rewrite；
  - DeepEP LL combine overlap 的最强映射是按 `block_m`/threshold/compute signal 做 chunk readiness，不等全 expert；
  - CUDA grouped GEMM 结果支持“专家 token counts / problem descriptors 是热路径的一部分”，映射到当前 K1 是 route metadata、counts、rowptr 和 problem rows 必须便宜且直接可消费。
- 对当前 LL 优化的结论：
  - K1 direct source A-load 已反证，说明不能为了“通信进 GEMM load”牺牲 staged input 的 coalescing；
  - K3 empty-tile skip 已反证，说明 chunk/tile readiness 不能改变当前 no-tail/tail 完成语义；
  - 下一步只做不改变 tile 完成语义的小步：先在 K3 LL epilogue rowptr path 消除 contiguous-output-only 地址计算，再根据 correctness/perf 决定保留或撤回。

## 2026-06-12 本工程 CUDA MegaMoE 复查

- 用户纠正 CUDA MegaMoE 参考就在本工程内，后续不再把外部/KB 的弱匹配当主要依据。
- 直接复查的本地文件：
  - `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`；
  - `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`；
  - `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`。
- 可转译的核心点：
  - CUDA MegaMoE 的 L2 BF16 epilogue 直接写远端 combine buffer，`TokenSrcMetadata {rank_idx, token_idx, topk_idx}` 决定目标 rank/token/topk；这确认 V3 K3 的通信语义应继续留在 epilogue/store 路径，而不是新增后处理。
  - CUDA 路径在 dispatch 阶段把 source token/topk 和 combine metadata 写入 workspace，同时 scheduler 在寄存器中缓存 per-expert token count；这对应 DCU V3 的 K1 route metadata、rowptr、expert counts 和 tile schedule 必须保持低成本且直接可被 K3 消费。
  - CUDA combine reduce 在 NVLink/grid barrier 后按 token/topk chunk 做 load/reduce/store；这与当前 DCU tail reduce 的 signal + local reduce 方向一致，但 CUDA 的 TMA/shared-memory 机制不能直接搬到 DCU。
  - CUDA L2 remote store 先把 TMEM 结果规整到 shared memory，再用 one-warp-per-row 的 `float4` remote store；DCU V3 已反证简单 lane-pair `dwordx4` 合并，因此若继续借鉴，只能做更小粒度的 epilogue/store 调度或 metadata 降本 A/B。
- 对当前 LL 性能调优的影响：
  - K3 128 staged/local/remote split 已证明 remote rowptr store 本身是主差距，后续 K3 方向不重复 store width 和 empty tile skip；只做不改变完成语义的 epilogue 小改或 profiler/ISA 证据驱动的 store 调度。
  - K1 direct source A-load 已反证，CUDA 的 remote pull 使用 TMA + shared staging，不能简化为 DCU GEMM 直接散读 source row；DCU K1 应保留 coalesced staging，再优化 metadata/count/scan/wait 成本。

## 2026-06-12 LL K3 epilogue cleanup A/B 保留

- A/B 内容：
  - 在 `V3_K3_Pure_LowLatencyMaskedGroupGemmKernel` 的 rowptr combine 路径中，不再为每个 active row 计算 contiguous/pure output-only 的 `out_warp` 地址；
  - `out_warp` 只在 `row_combine_ptrs == nullptr` 的 pure/contiguous store 分支计算；
  - 不改变 MMAC 主循环、rowptr register-prefetch、remote `global_store_dwordx2`、tile 完成语义或 tail/no-tail signal/reduce。
- 验证：
  - remote rebuild 成功，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_epilogue_cleanup_20260612_141242.log`；
  - no-tail 32/128 correctness 通过，summary `0.9280993640/1.0627396256 ms`；
  - tail 32/128 correctness 通过，summary `0.8761793897/1.3148792535 ms`，其中 128 median 被 `start->after_barrier` outlier 拉高，min 仍约 `1.0691395104 ms`；
  - stage parse：no-tail last3 128 K1/K3 `0.445/0.3775 ms`，tail last3 128 K1/K3tail `0.4395/0.454 ms`。
- 结论：
  - 该改动对 K3 stage 基本中性，tail 32/128 没有看到 K3tail 退化；
  - 因 correctness 通过且源码更贴近 rowptr combine 专用路径，保留；
  - 后续继续优先优化 K3 remote combine store 和 K1 metadata/staging 的较大 delta。

## 2026-06-12 LL K1 source-rank CTA route-scan A/B 反证

- 假设：
  - 本工程 CUDA MegaMoE 在 dispatch/scheduler 中显式缓存 source/rank/expert metadata；
  - DCU V3 K1 LL route scan 当前在每个 route task 内重复 `get_sections()`，可以尝试按 source rank 分配 CTA，让每个 CTA 只解析一次 peer sections，再扫描本 rank routes。
- A/B 内容：
  - 只改 `v3_k1_build_ll_stage_device` 的 route scan 分配；
  - `output_index` 全量清表、`row_combine_ptrs` 合同、staged_x copy、K1 GEMM 主体和 K3 都不改；
  - 每个 CTA 按 `blockIdx.x % num_ranks` 绑定 source rank，避免每 route 重新计算 source sections。
- 验证：
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_rank_cta_scan_20260612_142843.log`；
  - no-tail 32 correctness 通过，e2e `0.9042394981 ms`，stage last3 K1/K3 `0.4055/0.256 ms`；
  - no-tail 128 correctness 通过，e2e `1.0739393830 ms`，stage last3 K1/K3 `0.444/0.379 ms`；
  - 对比 register-prefetch/cleanup 基线，K1 stage 没有稳定改善，128 基本相同，32 略噪声。
- 结论：
  - 重复 `get_sections()` 不是当前 K1 LL 主要瓶颈，或者该收益被 route emit 顺序/CTA 分布变化抵消；
  - 该 A/B 还会改变 route emit 的并行顺序，对后续 uneven 更有风险；
  - 已撤回并重编恢复，恢复确认 log `hygon_tmp/sglang_debug/v3_ll_k1_rank_cta_scan_revert_notail128_20260612_143253.log`，128 no-tail correctness 通过，e2e `1.0798391104 ms`。

## 2026-06-12 LL K3 rowptr prefetch non-GLC A/B 反证

- 背景：
  - rowptr register-prefetch 已保留，仍需判断预取时是否必须使用 `global_load ... glc`；
  - 本地 DCU KB 对 `glc` 的证据显示它常用于绕过/控制缓存和跨设备可见性语义，普通 load 只能作为 A/B，不能凭源码直觉替换。
- A/B 内容：
  - 在 `V3_K3_Pure_LowLatencyMaskedGroupGemmKernel` 的 rowptr prefetch 中，将 `global_load_i64_glc_device(row_combine_ptrs + logical_row)` 临时改成普通 `row_combine_ptrs[logical_row]`；
  - 不改变 MMAC 主循环、rowptr register-prefetch、remote store helper、tail/no-tail 完成语义。
- 结果：
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_ll_k3_rowptr_nonglc_20260612_143628.log`；
  - no-tail 32 correctness 通过，stage last3 K1/K3 `0.4035/0.249 ms`，32 档相对 GLC 基线略快；
  - no-tail 128 correctness 通过，但 stage last3 K1/K3 `0.449/0.4105 ms`，K3 明显慢于 GLC/cleanup 基线约 `0.3775 ms`。
- 结论：
  - non-GLC 对 128 tokens 主要性能门槛不利，且 rowptr 是跨 rank combine 地址合同的一部分，不能为了 32 档微小收益牺牲 128；
  - 该 A/B 已撤回，恢复 `global_load_i64_glc_device`；
  - 恢复验证 128 no-tail correctness 为 true，stage last3 K1/K3 `0.4485/0.3805 ms`，回到 GLC/cleanup 基线附近；
  - 后续 K3 rowptr 优化继续从 epilogue/store 调度或 profiler/ISA 证据驱动，不重复普通 load 替换方向。

## 2026-06-12 LL K1 staged-copy compact iteration A/B 反证

- 假设：
  - 本工程 CUDA MegaMoE scheduler 把 expert token count 缓存在寄存器并用 counts 驱动 block iteration；
  - DCU V3 K1 staged copy 已有 `symm_counts[expert]`，可以尝试避免扫描完整 `row_capacity * kStageVecsPerRow`。
- A/B 内容：
  - staged copy 从全局线性 `row_capacity` 扫描改成 expert 外循环；
  - 每个 expert 只遍历 `ceil(symm_counts[expert] / kBlockM) * kBlockM` 行；
  - padded rows 仍写 zero，不改变 K2/GEMM 对 rounded rows 的读取合同。
- 结果：
  - no-tail 32/128 correctness 均通过；
  - 32 parse last3 K1/K3 `0.420/0.254 ms`；
  - 128 parse last3 K1/K3 `0.4545/0.379 ms`，K1 慢于当前线性 copy-loop 基线约 `0.44-0.448 ms`。
- 结论：
  - 该改动没有把 CUDA scheduler count-cache 思路有效转译到 DCU K1 staged copy；
  - 外层 expert 循环和更差的调度/并行分布抵消了减少空 row 扫描的收益；
  - 已撤回，K1 下一步不要再做简单 expert 外循环 copy 压缩；若继续动 staged copy，需要 profiler/ISA 或更接近原线性 coalesced 调度的方案。

## 2026-06-12 LL code object 资源分析

- 工具路径：
  - `hipprof --codeobj-analyze` 在当前非交互 SSH 管道中只稳定完成 ELF/kernel 列表输出，未输出资源明细；
  - 改用 `dccobjdump --extract-elf=all` 抽出 gfx938 code object，再读取自动生成的 `*-resource-usage.RES`；
  - `.so` 直接传给 `dccobjdump --show-resource-usage` 只显示 host ELF，属于 inconclusive，不能作为资源证据。
- K1 V3 LL:
  - 当前使用的 block32 实例资源约为 `sgpr_count=100`、`sgpr_spill_count=2`、`vgpr_count=124`；
  - block48 增至 `vgpr_count=151`；
  - block64 增至 `vgpr_count=193` 且 `private_segment_fixed_size=272`，与此前 block64 timing 明显退化一致；
  - 后续 K1 不宜再通过更大 blockM 追 tile 数，除非先解决寄存器/private segment 压力。
- K3 V3 LL:
  - no-tail block32 为 `sgpr_count=100`、`sgpr_spill_count=1`、`vgpr_count=153`；
  - tail block32 为 `sgpr_count=100`、`sgpr_spill_count=16`、`vgpr_count=130`；
  - tail block64 同样 `sgpr_spill_count=16` 且 `vgpr_count=189`；
  - no-tail 当前更像 remote store/epilogue 调度成本，tail 则有明确 SGPR spill 信号。
- 下一步含义：
  - 优先做 K3 LL tail-only 常量化/参数缩减 A/B，固定 EP8/topk=6/done_target=64/reduce_blocks=64 等 shape 合同，观察 SGPR spill 和 tail stage timing 是否改善；
  - 不改变 no-tail 完成语义，不新增 runtime kernel，不重复 empty-tile skip、non-GLC rowptr、storex4 lane-pair 等已反证方向。

## 2026-06-12 LL K3 tail 参数常量化 A/B 反证

- 假设：
  - K3 LL tail block32 code object 显示 `sgpr_spill_count=16`、`kernarg_segment_size=384`；
  - LL staged 合同固定 EP8、experts=256、topk=6、done_target=64、reduce_blocks=64、signal_generation=1；
  - 将 tail signal/reduce helper 改成 fixed shape 版本可能降低 SGPR spill，并改善 tail 128 阶段耗时。
- A/B 结果：
  - fixed shape + topk unroll 后，资源表改善为 tail block32 `kernarg_segment_size=128`、`sgpr_spill_count=6`；
  - 32/128 tail correctness 通过，128 e2e 一轮约 `1.100 ms`，rerun 约 `1.085 ms`；
  - stage timing 128 last3 K3tail 约 `0.456-0.463 ms`，没有稳定优于旧基线 `0.454-0.461 ms`；
  - 去掉 topk unroll 后资源不变，128 e2e 约 `1.097 ms`，K3tail last3 约 `0.463 ms`，仍无收益。
- 结论：
  - 这说明 tail 的 SGPR spill 虽然明显，但不是当前 128 tail 阶段的主瓶颈，或者 spill 降低被额外控制流/代码形态抵消；
  - 该 A/B 已撤回并重编恢复，恢复版 128 tail correctness 通过，e2e 约 `1.088 ms`；
  - 后续不要只为降低 code-object spill 牺牲代码简洁性，除非 PMC/SQTT/ISA 或稳定 timing 同时支持。

## 2026-06-12 LL K3 PMC read/write 对照

- 目的：
  - 用户要求优先优化差异大的 LL 路径；K3 128 staged remote combine 与 local rowptr/pure 仍有明显 delta；
  - 在继续改 kernel 前，用 `hipprof --pmc-read/--pmc-write --pmc-type 3` 对 `local_rowptr` 和 `staged_remote_only` 做同口径短 profile。
- 工具和脚本：
  - 拉取并解析 `hygon_tmp/sglang_debug/prof/pmc_v3_ll_k3_notail128_20260612_153148.csv`；
  - 临时脚本 `hygon_tmp/sglang_debug/parse_pmc_k3.py` 输出 duration、VMEM/VALU/LDS、TA/TCC/TCP 等摘要；
  - `bench_k3_ll_rowptr_modes.py` 增加 `--modes`，便于单 mode profile。
- 128 tokens 结果：
  - `local_rowptr` read/write profile 的 kernel median 约 `0.279/0.277 ms`；
  - `staged_remote_only` read/write profile 的 kernel median 约 `0.402/0.405 ms`；
  - 两者 `VMEM_RD=819200`、`VMEM_WR≈16128`、`VALU≈3251456`、`LDS≈1048576` 基本相同；
  - remote-only 的 `TA_BUSY` 从约 `18.0M` 增至 `25.1-25.6M`，`TCP_TCP_TA_DATA_STALL_CYCLES` 从约 `0.80M` 增至 `1.54-1.61M`；
  - `write_req_stall` 在 read/write 拆分 profile 中近 0。
- 结论：
  - K3 128 remote-only 慢不是因为 store 指令数量更多，也没有看到明显 TCC 写请求 stall；
  - 更像 rowptr 指向跨 rank/scattered combine 地址后，数据通路和地址/缓存路径等待变高；
  - 后续不再重复已反证的 store-width、lane-pair dwordx4、写请求 stall 方向；
  - 如果继续动 K3，应优先分析 rowptr 地址分布、tile/order 与 remote rank/stride 的关系，且只能做不改变 tile 完成语义的小步 A/B。

## 2026-06-12 LL K1 clamp barrier fold A/B 反证

- 假设：
  - K1 LL build stage 在 route scan 后单独做 `symm_counts` clamp，然后立刻做一次 grid barrier；
  - 可把 clamp/stats 合入 stage-copy 前，用 `min(symm_counts[expert], m_per_expert)` 保护 copy，再依赖 stage-copy 末尾已有 grid barrier 保证 GEMM 前 counts 可见；
  - 这样不改 K1 前 rank barrier、不新增 kernel、不改通信语义，只少一次 K1 内部 grid barrier。
- 结果：
  - 远端 K1 LL raw rebuild 成功；
  - no-tail 32/128 correctness 通过，128 no-tail last3 K1 约 `0.449 ms`；
  - tail 32/128 correctness 通过，128 tail last3 K1 约 `0.4435 ms`；
  - 对比恢复后 128 no-tail K1 约 `0.4415 ms` 和此前基线约 `0.44-0.45 ms`，没有稳定收益。
- 结论：
  - 单独减少这一次 K1 内部 grid barrier 不是当前 K1 128 delta 主因；
  - 该 A/B 已撤回并远端重编恢复，恢复版 128 no-tail correctness 通过；
  - K1 后续应继续从 staged copy/metadata/GEMM 固定成本入手，避免把单个 barrier 当成主瓶颈。

## 2026-06-12 LL rowptr 地址分布诊断 caveat

- 临时给 `bench_k3_ll_rowptr_modes.py` 增加 `--dump-rowptr-stats`，尝试按 peer combine buffer range 反推出 `row_combine_ptrs` 的目标 rank 和 16-row chunk 分布。
- 结果中只识别到平均约 256 个有效地址，但同一脚本的 active row 数约 768，说明当前 Python 侧“peer buffer base + combine offset”的区间匹配没有覆盖 K1 生成的全部 rowptr。
- 已核对 K1 生成公式使用 device `dcu_peer_sym_buffer_ptrs(local_sym_buffer)` 和 `get_sections(...).combine`，Python 诊断不能把未识别地址当作非法 rowptr 或优化证据。
- 该诊断仅保留为 caveat：后续如果要基于 rowptr 目标 rank/order 做 A/B，必须先做 device-side 或 wrapper-side 的可靠归属统计，不能用这版 invalid 计数下结论。

## 2026-06-12 LL Phase 6 性能门槛收口

- 当前 LL 代码恢复到保留项集合：K3 rowptr register-prefetch、K3 epilogue cleanup、K1 source ptr/scale hoist、K1 compact route stride；已撤回 direct source A-load、empty-tile skip、storex4 lane-pair、source-rank route scan、compact staged-copy、clamp barrier fold 等无收益或有风险方向。
- 8 卡 no-tail gate：
  - baseline `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0` 32/128 median 约 `1.135/1.631 ms`；
  - V3 LL `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=0` 32/128 median 约 `0.760/0.938 ms`；
  - correctness 均通过。
- 8 卡 tail gate：
  - baseline `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=0` 32/128 median 约 `1.142/1.642 ms`；
  - V3 LL `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=ll K3_USE_ASM_TAIL_REDUCE=1` 32/128 median 约 `0.776/0.973 ms`；
  - correctness 均通过。
- 结论：
  - Phase 6 的 `<512` 档位目标已达成；
  - LL 剩余 delta 主要作为后续优化余量，不阻塞 normal；
  - 下一阶段优先建立 normal 1024/4096 对照，比较 V3 normal 与原 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 && USE_MEGAMOE_V3=0` staged fused 路径。

## 2026-06-12 normal K3 rowptr split 准备

- 静态复核 normal K3 raw path：
  - `k3_v3_normal_combine_raw` 的 normal kernel 当前没有独立 contiguous pure-store 分支，`out` 形参不作为普通 contiguous output 合同，实际输出语义是通过 `row_combine_ptrs` 写 combine buffer；
  - 因此 normal K3 的首轮 pure-vs-fused 拆分不能照搬 LL 的 `pure_contiguous` 模式，应先对比 `local_rowptr`、`staged_rowptr`、`staged_local_only`、`staged_remote_only` 和 `rowptr_all_zero`。
- 已扩展诊断脚本：
  - `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py` 增加 `--backend normal`，复用现有 K1 route/K2/staged rowptr 构造和 rowptr split 逻辑；
  - 新增 `hygon_tmp/sglang_debug/run_v3_normal_k3_rowptr_modes.sh`，默认跑 1024/4096 normal K3 rowptr split；
  - 本地 `python -m compileall` 和 `git diff --check` 通过；本地无 `bash`，shell 语法检查等待远端恢复后执行。
- 用途：
  - normal gate 如果显示 V3 normal 慢于原 staged fused，先用 stage timing 判断 K1/K3；若 K3 差异大，立即用该 rowptr split 区分本地 rowptr 开销、跨 rank rowptr/remote combine 数据通路开销和空 row/tile 固定成本。
- 本地 DCU KB 复查：
  - `hygon-extend` 的 Flux GEMM+ReduceScatter 资料继续支持把通信语义放在 epilogue store path，并按 rank-aware tile/swizzle 与 intranode/remote/no-link 模式拆测；
  - 对当前 normal K3 的含义是先测 `local_rowptr` / `staged_remote_only`，再考虑 tile order、rowptr 目标 rank 分布或 epilogue 调度；不要在没有证据时重复 LL 已反证的 store-width/lane-pair/empty-tile 方向。

## 2026-06-12 normal K1 unused mask/tail 候选

- V3 normal K1 launch 仍传入 `local_topk_mask` / `tail_tokens`，而 stage builder 只要任一非空就会扫描本地 tokens、写 mask/tail，并追加一次 grid barrier。
- V3 staged normal 的后续 K2/K3 合同不消费这两个输出；当前返回值也只有 `l1_out/route_weights/m_indices/output_index/row_combine_ptrs`。
- 因此把 normal launch 改成传 `nullptr, nullptr` 是低风险 A/B 候选，和 LL 已验证保留的 unused mask/tail 优化一致。
- 该改动仍需远端 aicc rebuild 后用 1024/4096 correctness + stage timing 证明；不能仅凭静态分析宣称收益。

## 2026-06-12 normal K1 unused mask/tail A/B 反证

- A/B 现象：
  - V3 normal K1 launch 传 `nullptr, nullptr` 后，1024 no-tail correctness loop 一度通过，但后续 perf/repeat 出现 `AssertionError: fused/baseline nonfinite fused=1 baseline=0 diff=1`；
  - 这说明该改动不是稳定优化，不能作为 normal gate 的基线。
- 代码根因：
  - `v3_k1_build_ll_stage_device` 中 `if (local_topk_mask != nullptr || tail_tokens != nullptr)` 分支除了写未消费的 mask/tail，还会执行一次 `v3_k1_ll_grid_barrier_device(...)`；
  - normal 分支改成 `nullptr,nullptr` 实际删除了一个 K1 stage 内部 readiness barrier，而不是单纯删除未使用输出。
- DCU KB 对应原则：
  - `hygon-extend` 的 DeepEP/Flux 通信融合资料强调 routing/readiness metadata 是 hot path，barrier/readiness 状态应作为 GEMM scheduling 合同的一部分；
  - 因此当前证据下不能为了省少量本地 token 扫描而跳过 readiness barrier。
- 处置：
  - 已恢复 normal launch 传 `local_topk_mask.data_ptr<uint8_t>()` 和 `tail_tokens.data_ptr<int32_t>()`；
  - 后续 normal 优化不重复该省 barrier 方向，如要移除必须拆成单独 readiness-safe A/B，并保留等价 barrier 或证明无需该 barrier。

## 2026-06-12 normal 1024 no-tail 初测

- 构建：
  - 远端 11 节点 `sglang_megamoe` 容器恢复，V3 K1/K3 normal raw extension 通过 aicc shim 编译；
  - build log 包括 `Building ... K1_fused.k1_fused_ext with V3 normal aicc shim` 和 `Building ... K3_fused.k3_v3_fused_ext with V3 normal aicc shim`。
- 对照：
  - 原 staged normal 1024 no-tail：`fused_median_ms_avg_per_rank ≈ 2.4109 ms`，correct true；
  - V3 normal 1024 no-tail 在恢复 mask/tail 后：correct true，`fused_median_ms_avg_per_rank ≈ 11.1902 ms`。
- stage timing：
  - K1 阶段约 `8.3-8.8 ms`；
  - K2 阶段约 `0.07-0.15 ms`；
  - K3 combine 阶段约 `2.16-2.25 ms`；
  - reduce 约 `0.047 ms`。
- 结论：
  - 当前 normal no-tail 主要差距在 K1，K3 次之；
  - 继续补 4096 数据后，按 stage delta 优先级优化，不能默认只看 K3。

## 2026-06-12 normal 4096 no-tail 可见性诊断

- 原 staged 4096 no-tail：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=0 K3_USE_ASM_TAIL_REDUCE=0` 通过；
  - 结果文件 `hygon_tmp/sglang_debug/orig_stage_normal_4096_perf_20260612_171557.json`，`fused_median_ms_avg_per_rank ≈ 6.6148 ms`。
- V3 normal 4096 no-tail：
  - `MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0` 在 stage timing/perf 链路失败；
  - log `hygon_tmp/sglang_debug/v3_normal_4096_bench_20260612_171635.log`，错误为 `fused/baseline nonfinite fused=2 baseline=0 diff=2`；
  - 失败前阶段耗时约为 K1 `41-42 ms`、K3 combine `7.7-8.0 ms`、reduce `0.165 ms`。
- 最小诊断：
  - 不开 `MEGAMOE_DCU_V3_STAGE_TIMING` 的 correctness-only 四组 base/sync/acquire/both 均通过 3/3；
  - 开 `MEGAMOE_DCU_V3_STAGE_TIMING=1` 后 base 复现 fused 非有限值；
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 可让 stage-timing correctness 通过 3/3；
  - acquire/reduce acquire flags 不能稳定解决，且出现 `max_abs` 超阈值。
- 当前判断：
  - 这不是 K1 mask/tail layout 问题，也不是可以靠 host sync 解决的最终方案；
  - 证据指向 K3 no-tail combine store 完成/跨 rank 可见性，在外部 `rank_barrier + reduce_local_combine` 前仍有时序缺口；
  - 下一步验证 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 的 in-kernel signal/wait 路径，判断是否能替代 host sync，同时继续保持“不新增 runtime kernel”的约束。
- no-tail signal 结果：
  - `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1 MEGAMOE_DCU_V3_STAGE_TIMING=1` 下，4096 correctness-only 3/3 通过；
  - log `hygon_tmp/sglang_debug/v3_normal_4096_diag_stage_signal_20260612_173053.log`，`max_abs=0.000488281`；
  - K3 combine 阶段约 `7.9-8.6 ms`，no-tail external barrier/reduce 仍约 `0.02-0.04/0.165 ms`；
  - 该结果支持把 no-tail signal 作为候选 correctness 修复，但还需要 1024/4096 perf 评估和源码收口，不能只凭 correctness 合入最终 gate。
- 1024 signal 对比隔离：
  - `signal` 原样 1024 correctness-only 失败，rank6 报 `fused/baseline nonfinite fused=118 baseline=0 diff=118`；
  - `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 后 1024 stage-timing correctness-only 5/5 通过；
  - `signal + MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 仍失败，转为 `max_abs=0.003704071044921875 > 0.0035`；
  - 随后用 `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 进入完整 perf runner 仍在 1024 correctness 阶段失败，rank5 报 `fused/baseline nonfinite fused=135 baseline=0 diff=135`；
  - 因此 clone 只能说明短诊断存在测试侧/时序敏感性，不能作为稳定 correctness/perf 隔离变量，也不能替代 kernel 侧可见性修复。
- 当前 A/B：
  - no-tail signal 原先使用 `kSignalOnly=true` 的独立轻量完成分支，缺少 tail-reduce 分支中更完整的 completion owner、block barriers 与 peer signal/wait 结构；
  - 本地临时改为在 no-tail signal path 复用 `kTailReduce=true` 模板，同时传 `reduce_blocks=0`、`reduce_y=nullptr`，即复用已有完成 signal/wait 语义但不执行 tail reduce，也不新增 runtime kernel；
  - 该 A/B 需要远端 aicc rebuild 后先跑 1024/4096 stage-timing correctness，再跑 perf；若失败则撤回。

## 2026-06-12 normal no-tail signal generation / ACQ_REL 诊断

- tail completion 模板复用 A/B 已反证：
  - no-tail signal path 临时改成复用 `<..., kTailReduce=true>` completion/wait 结构、`reduce_blocks=0`；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_signal_tailtmpl_20260612_174303.log`；
  - 1024 stage-timing correctness 第 2 轮失败，log `hygon_tmp/sglang_debug/v3_normal_1024_diag_signal_tailtmpl_20260612_174419.log`，rank0 报 `fused/baseline nonfinite fused=12 baseline=0 diff=12`；
  - 该 A/B 已撤回到 signal-only 模板，不再重复。
- generation 诊断：
  - DCU KB 对 repeated sync point 的建议是使用可区分 generation/epoch，并配合 release/acquire 可见性，避免旧 launch 的 signal 污染新 launch；
  - Python 侧 `_LargeOptState` 已增加 `asm_signal_generation`，tail 和 no-tail signal 每次 eager launch 递增；
  - 初始实现后仍失败，静态复核发现 `k3_v3_normal_combine_signal_raw(... signal_generation)` 接收了 generation，但内部 `dcu_megamoe_v3_launch_k3_normal_combine_raw` 曾固定给 kernel 传 `1`；
  - 已修复 internal launch signature/call sites，normal tail/no-tail signal 现在透传 `signal_generation`。
- generation 修复后的现象：
  - 1024 signal/no-clone/no-sync correctness 不再立即失败，但第 4 轮仍出现 rank7 `fused/baseline nonfinite fused=45 baseline=0 diff=45`；
  - 1024 signal + `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 通过 5/5；
  - 1024 perf runner 在 signal+clone+stage timing 下正确，`fused_median_ms_avg_per_rank` 约 `11.2729 ms`，K1 stage 约 `8.4-8.7 ms`，K3 stage 约 `2.25-2.45 ms`；
  - 4096 signal+clone+stage timing 仍会在多轮后出现 `max_abs` 超阈值，典型 rank0/rank4 `0.0044-0.0049 > 0.0035`；
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 仍可让短诊断通过，说明 host sync 覆盖的是设备侧完成/可见性时序，而不是最终可接受修复。
- 当前最小假设：
  - generation 修复排除了“固定 generation=1”作为唯一根因；
  - 剩余 4096 漂移更像 done counter completion 与 peer signal/wait 之间缺少 system-scope acquire/release 序；
  - 已按 KB 中 Hygon/Flux release/acquire 模式准备最小 A/B：normal no-tail/tail completion 的 `done_counter` 从 `atomicAdd_system`/relaxed 形态改为 `__hip_atomic_fetch_add(..., __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM)`，peer signal 使用 `__ATOMIC_RELEASE`；
  - 该 A/B 尚未远端 build/test，不能视为结论。
- ACQ_REL A/B correctness 结果：
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_acqrel_20260612_180534.log`，V3 K1/K3 fused normal 均走 aicc shim；
  - 1024 no-tail signal + stage timing + no clone + no host sync：5/5 通过，log `hygon_tmp/sglang_debug/v3_normal_1024_diag_signal_acqrel_20260612_180714.log`，`max_abs=0.000488281`；
  - 4096 no-tail signal + stage timing + no clone + no host sync：5/5 通过，log `hygon_tmp/sglang_debug/v3_normal_4096_diag_signal_acqrel_20260612_180819.log`，`max_abs` 最高 `0.000885010`，低于 `0.0035`；
  - 这支持 ACQ_REL done counter + release peer signal 是当前 no-tail signal 可见性修复方向，但还需 perf/stage timing 量化和后续 normal 性能优化。
- 完整 bench 链路补充：
  - 仅 ACQ_REL 在完整 no-skip 链路仍不稳定：runner 1024 log `v3_normal_1024_bench_20260612_180953.log` 报 rank4 非有限；手工 no-skip 1024 log `v3_normal_1024_manual_noskip_acqrel_20260612_181206.log` 报 rank6 非有限；clone 仍失败，log `v3_normal_1024_manual_noskip_acqrel_clone_20260612_181358.log`；
  - KB 和本地源码确认 `deep_gemm::mega::load_signal_system` 是 system-scope relaxed load，Hygon/Flux 非 final sync 参考使用 release + acquire；
  - 将 no-tail signal-only peer wait 改为 system-scope acquire load 后，1024 完整 no-skip bench 通过，log `v3_normal_1024_manual_noskip_wait_acquire_20260612_181744.log`，`fused_median_ms_avg_per_rank ≈ 11.2548 ms`；
  - 4096 在 wait-acquire 后仍失败，说明 reduce 读侧还需要可见性处理；
  - 加 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 后 4096 完整 no-skip bench 通过，log `v3_normal_4096_manual_wait_acquire_reduce_acquire_20260612_182006.log`，`fused_median_ms_avg_per_rank ≈ 50.9254 ms`；
  - 当前 no-tail 稳定组合是：K3 completion ACQ_REL + peer signal release + peer wait acquire + post-K3 rank barrier acquire + reduce invalidate/glc read。
- normal 性能含义：
  - 1024 当前 stage timing 约 K1 `8.3-8.7 ms`、K3 `2.25-2.50 ms`；
  - 4096 当前 stage timing 约 K1 `42 ms`、K3 `8 ms`；
  - normal 下一轮优化应优先 K1 stage，而不是继续在 K3 signal/reduce 上堆同步。

## 2026-06-12 normal no-tail owner-slot / correctness harness 诊断

- 默认 no-tail signal 收敛到 Python 默认后，直接 default env 仍能复现 1024 多轮 correctness 非有限：
  - runner log `hygon_tmp/sglang_debug/v3_normal_1024_bench_20260612_182633.log`，第 3 轮左右 rank6 `fused=30`；
  - direct log `hygon_tmp/sglang_debug/v3_normal_1024_manual_default_20260612_182817.log`，第 3 轮左右 rank7 `fused=24`；
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC=1` 也会第 4 轮左右失败，log `v3_normal_1024_manual_default_sync_20260612_182940.log`，rank7 `fused=18`。
- KB / 参考：
  - Hygon/Flux/SGLang allreduce 资料继续指向 release/acquire + system fence；
  - SGLang HIP allreduce 注释明确非 final sync 需要 fence，避免 signal 可见早于数据可见。
- A/B：
  - normal `kSignalOnly` 从最后 CTA 直接 signal 改为 `done_counter[1]` owner-slot：最后 CTA release-store owner id，owner CTA acquire-load 后 signal/wait peers；
  - 这不是此前失败的 tail-template A/B：没有启用 tail worker/reduce，也不改变 GEMM 主循环。
- 结果：
  - owner-slot 1024 default no-clone 前 4 轮通过，第 5 轮仍报 rank6 `fused=3`，log `v3_normal_1024_manual_default_owner_slot_20260612_183435.log`；
  - owner-slot + `MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1` 1024 通过 5/5 并完成 bench，`fused_median_ms_avg_per_rank ≈ 11.2150 ms`；
  - owner-slot + clone 4096 通过 3/3 并完成 bench，`fused_median_ms_avg_per_rank ≈ 50.9933 ms`。
- 结论：
  - owner-slot 对多轮非有限有改善但不是充分修复；
  - clone 隔离后 correctness 通过，说明 `run_fused` 产物本身可与 baseline 对齐，但 no-clone correctness loop 仍有 baseline oracle / output 生命周期 / 多 rank 迭代时序 artifact；
  - 性能 bench 不包含 clone，因此可以先用 clone 隔离 correctness 前置继续 normal K1/K3 性能优化；
  - no-clone artifact 必须后续单独修，不能把 clone 当生产 kernel correctness 的最终证据。

## 2026-06-12 normal pure-vs-fused 基线

- K1 pure normal pack5 通过 `hygon_tmp/sglang_debug/run_v3_normal_pure_refs.sh` 执行，aicc 路径需使用 `/workspace/dtk_aicc/bin/aicc` 及对应 ROCM/HIP/device-lib 环境；`/opt/dtk/bin/aicc` 不接受 `-mllvm -enable-num-vgprs-768=true`。
- K1 pure log: `hygon_tmp/sglang_debug/v3_normal_k1_pure_pack5_20260612_184430.log`。
  - 1024 tokens: median `0.745737 ms`，min `0.745318 ms`。
  - 4096 tokens: median `2.25922 ms`，min `2.25905 ms`。
- K3 rowptr split JSON:
  - 1024: `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260612_184527.json`；
  - 4096: `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260612_184551.json`。
- K3 `mode_summary.median_avg_rank_ms`:
  - 1024: `rowptr_all_zero 0.7176`、`local_rowptr 0.9389`、`staged_local_only 1.1004`、`staged_remote_only 2.2078`、`staged_rowptr 2.2162`。
  - 4096: `rowptr_all_zero 2.5415`、`local_rowptr 3.3051`、`staged_local_only 3.4587`、`staged_remote_only 7.7993`、`staged_rowptr 7.8297`。
- 对当前 V3 normal no-tail 的含义：
  - K3 staged rowptr 数据基本解释当前 K3 stage `~2.2 / ~8.0 ms`，主要成本来自 remote rowptr 数据通路；这不是当前最大 delta。
  - K1 fused stage 约 `8.4-8.7 ms / 41-43 ms`，相对 K1 pure `0.746 / 2.259 ms` 是最大差距，下一轮必须优先定位 normal K1 route/stage-copy/GEMM 内部成本。
  - 不应继续把 normal 第一优化点放在 K3 signal/reduce 或已反证的 store-width/empty-tile 方向。

## 2026-06-12 normal K1 staged-input A/B 首轮

- 改动意图：
  - normal K1 原 fused GEMM 直接通过 `row_x_ptrs` 做分散 A-load，stage timing 约 `8.4-8.7 ms / 41-43 ms`，明显偏离 pure normal K1 `0.746 / 2.259 ms`；
  - A/B 让同一 K1 main kernel 的 builder CTA 在 tile ready 后把 source `x` 按 row stage 到 contiguous `staged_x`，随后 GEMM A-load 改成 contiguous buffer resource load；
  - 没有新增 runtime kernel，没有改 K2/K3 wrapper 合同，仍使用 V3 pack5 L1 layout。
- 首轮结果：
  - 远端 K1-only aicc rebuild 实际更新 K1 `.so`，但当时 8 卡被 `sglang serve` 占用约 91-93% VRAM；
  - 1024 no-tail `--skip-bench` 在 occupied 环境下显示 K1 stage 下降到约 `2.3-3.1 ms`，说明 staged contiguous A-load 是正向性能方向；
  - correctness 仍失败，rank 侧最大 `max_abs=0.12060546875 > 0.0035`，不能保留为完成优化。
- 当前最小根因假设：
  - K1 stage ready flag 当前是 `flags[slot] = epoch` + system relaxed load 等待；
  - 本地 DCU KB 的 Flux/DeepEP/Hygon 参考对同类跨 CTA/跨通信数据可见性使用 release signal + acquire wait；
  - 因此下一步只把 K1 ready flag 收敛到 system-scope release/acquire，再重编复测，不同时叠加其他优化。

## 2026-06-12 normal K1 staged-input ready flag release/acquire A/B

- 代码改动：
  - `v3_k1_store_ready_flag_device()` 改成 `__hip_atomic_store(..., __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM)`；
  - `v3_k1_wait_ready_flag_device()` 改成 system-scope acquire load；
  - 只改 ready flag 语义，不改 staged copy loop、GEMM 主循环、K2/K3 wrapper 或 runtime launch 数。
- 构建：
  - 远端 K1-only rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1_stage_acquire_20260612_190739.log`；
  - build 确认 `K1_fused.k1_fused_ext` 使用 V3 normal aicc shim；
  - K3 raw 未重新编译，只有 existing K3 extension relink/no-work。
- 低层验证：
  - `k1_v3_normal_smoke.py` local-rank `expert_ramp` 通过，log `hygon_tmp/sglang_debug/k1_v3_normal_smoke_stage_acquire_20260612_190859.log`，`expert_ramp_abs_err=0`；
  - `K1_V3_SMOKE_ALL_RANKS=1` simulated all-ranks `expert_ramp` 通过，log `hygon_tmp/sglang_debug/k1_v3_normal_smoke_allranks_stage_acquire_20260612_190928.log`，`expert_ramp_abs_err=0`；
  - 这只证明 single-GPU route/stage/GEMM 基础数值正确，不替代 8 卡 cross-rank correctness。
- ISA / code-object caveat：
  - `dccobjdump --show-sass` 直接作用于 K1 `.so` 和 build `.o` 都只识别到 host `elf64-x86-64`，未导出 device ISA；
  - 本轮不能把 acquire/release 的最终 ISA 作为已验证证据，后续若需要 ISA 证明需定位 aicc/hipcc 的 device code object 或使用其他 DTK dump 路径。
- 当前阻塞：
  - 远端 8 卡仍被 `sglang serve --model-path /data2/MiMo-V2-Flash-Channel-FP8` 占用约 135-138GB/卡；
  - 1024/4096 8 卡 correctness/perf 需要卡释放或用户授权处理该服务。

## 2026-06-12 normal K1 staged-input 单卡 rank5 诊断补充

- 最短复现：
  - `hygon_tmp/sglang_debug/k1_v3_output_compare.py` 使用 fake symmetric buffer，x/x_scale/topk_weight 全为 1，随机 topk，K1 ASM vs K1 V3 按 `row_combine_ptrs` 对齐比较；
  - rank5/random/1024 能在单 GPU 上复现间歇性错值，不需要 K2/K3/e2e；
  - metadata 对齐：active/common/missing/extra 都一致，`m_indices` expert 也一致；因此当前问题不是 route metadata、row_combine_ptrs 或 output_index contract。
- 错误形态：
  - 失败时 ASM 输出接近 ref，V3 同 expert 但单行/单列有限偏离，`max_abs` 约 `0.6~1.1`；
  - x 全为 1 时 source_rank/token/topk_slot 不应影响 GEMM 数值；错误落在不同 source/topk 的事实更支持 staged input 数据偶发不正确，而不是 weight layout；
  - 当前没有稳定 NaN/Inf，之前 nonfinite 更可能是同一数据可见性问题的更坏表现。
- 已反证：
  - K1 source row load 加 GLC 并不能修复，且会让链路更差，已撤回；
  - 只把 ready flag 改为 release/acquire 不充分；
  - 只让发布 flag 的 tid0 做 `__threadfence_system()` 不充分，因为 staged_x/metadata 写入来自整 CTA；
  - all-thread system fence 后仍不充分，说明可能还需要 VMEM wait、raw/global store-load cache path 处理，或回到 direct rowptr A-load 作诊断分界。
- 当前候选：
  - `s_waitcnt vmcnt(0)` + all-thread system fence before ready flag 是正在验证的最小 A/B；
  - 若该 A/B 仍失败，应优先做两个诊断而不是继续堆同步：1) direct rowptr A-load 恢复诊断，确认 GEMM core/weight 路径本身是否稳定；2) staged fill-ones 诊断，确认 staging store-load 是否是唯一变量。
- 本地已准备 direct-rowptr 诊断开关：
  - `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT=1` 会让 normal K1 GEMM B-load 直接从 `row_x_ptrs` 读取，跳过 `staged_x` global store/load；
  - 该开关只用于定位，不作为性能优化候选；direct rowptr A-load 之前已显示性能很差，若 correctness 通过，只说明 staged-input 通路仍需修，不代表应回退到 direct 路径。

## 2026-06-12 normal K1 staged-input waitcnt 修复结果

- 远端容器重启后 inplace K1 `.so` 丢失，已重新 K1-only normal rebuild，K1 extension 继续走 V3 normal aicc shim，未编 V2。
- `s_waitcnt vmcnt(0)` + all-thread system fence + ready flag release/acquire 后，rank5/random/1024 单卡短链路 10/10 通过：
  - log `hygon_tmp/sglang_debug/k1_v3_output_compare_rank5_repeat_waitcnt_20260612_215816.log`；
  - 每轮 `asm_rows == v3_rows == 8192`、`asm_active == v3_active == common == 6145`、`missing_in_v3=0`、`extra_in_v3=0`；
  - `asm_nonfinite=0`、`v3_nonfinite=0`、`max_abs=0`。
- 结论：
  - 之前的有限错值符合同 kernel global store/load 可见性缺口；只用 release/acquire flag 或只用 fence 不充分，必须在发布 ready flag 前等待 VMEM store 完成；
  - 该修复只证明 K1 staged-input 短链路正确性，不等价于 8 卡 e2e correctness 或性能达标；
  - 由于 normal K1/K3 性能仍远落后原 staged fused ASM，后续优化不继续堆同步，转向“原始 groupgemm ASM vs K1/K3 fused ASM”的差异映射和 stage timing/PMC 证据。

## 2026-06-12 normal ASM-diff 优化方向补充

- DCU KB `Report_deepgemm_asm_grouped_fp8_bf16_gemm_gfx936_gfx938.md` 命中当前核心形状：`MT256x256x128`、gfx938、FP8 grouped GEMM、`v_mmac_f32_16x16x32_fp8_fp8`、768 threads、64KB two-stage LDS。
- 对 V3 normal 的直接含义：
  - K1/K3 fused normal 大 size 性能不应显著慢于原 staged ASM；若慢，优先怀疑通信/metadata 外壳与 GEMM pipeline 脱节，而不是 pure C groupgemm 骨干本身；
  - K1 重点检查原 fused ASM 是否把 dispatch-pull/staged input/metadata 只作为 tile scheduling 的轻量附加语义，而当前 C path 是否额外扫描、清表、stage 了过多行或引入过重 grid barrier；
  - K3 重点检查原 K3COMBINE 是否在 epilogue 直接按 rowptr 写 remote combine，避免 C path 的过早 rowptr load、额外 GLC/同步和不必要 completion signal 成本；
  - 后续改动必须用 1024/4096 stage timing、必要时 hipprof/PMC/code-object resource 分项证明，不用“应该更快”的直觉替代。

## 2026-06-12 normal 1024 no-tail stage timing after K1 waitcnt

- 原 staged 1024 no-tail：correctness 3/3 通过，`fused_median_ms_avg_per_rank=2.4267188906669617 ms`。
- V3 normal 1024 no-tail：
  - correctness 前置 3/3 通过，`max_abs=0.000488281`；
  - 稳定轮 stage timing：K1 `2.510-2.572 ms`，K2 `0.107-0.109 ms`，K3 combine `2.295-2.348 ms`，barrier/reduce 约 `0.02-0.05 ms`；
  - perf loop 仍触发 no-clone artifact：`fused/baseline nonfinite fused=15 baseline=0 diff=15`。
- 归因：
  - K1 staged-input waitcnt 修复使 K1 从 `~8.5 ms` 降到 `~2.5 ms`，但相对 K1 pure normal `0.746 ms` 仍有约 `1.75 ms` delta；
  - K3 no-tail combine 当前 `~2.3 ms`，与此前 normal staged rowptr split 的 `staged_remote_only/staged_rowptr ~2.21 ms` 同量级，主要是 remote rowptr combine 数据通路；
  - 1024 V3 stage 总体约 `5.1-5.2 ms`，仍慢于原 staged `2.43 ms`，下一步必须按 hygon optimizer flow 获取 4096 分段、code-object/hipprof/ASM diff 证据，再做单变量 A/B，不能直接凭直觉调参。

## 2026-06-12 normal 4096 no-tail stage timing after K1 waitcnt

- V3 normal 4096 no-tail correctness-only/stage timing：
  - env：`MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_STAGE_TIMING=1 MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1 K3_USE_ASM_TAIL_REDUCE=0`；
  - log `hygon_tmp/sglang_debug/v3_normal_4096_skipbench_stage_waitcnt_20260612_220835.log`；
  - correctness 3/3 通过，`max_abs=0.000488281`；
  - 稳定轮 stage timing：K1 `8.786-9.005 ms`，K2 `0.214-0.220 ms`，K3 combine `8.032-8.260 ms`，barrier/reduce 约 `0.02/0.17 ms`。
- 归因：
  - K1 staged-input 对 4096 从此前 `~41-43 ms` 降到 `~9 ms`，但相对 K1 pure `2.259 ms` 仍慢约 `6.7 ms`；
  - K3 combine 与 normal rowptr split 的 `staged_rowptr ~7.83 ms` 基本同量级，说明 K3 主要受 remote rowptr combine store 数据通路限制；
  - normal 4096 目前不是功能 blocker，而是明确性能 blocker：K1 剩余 delta 最大，K3 次之，后续优化必须先拿 profiler/code-object/ASM diff 证据。

## 2026-06-12 normal PMC / code-object evidence

- code-object / ISA：
  - `dccobjdump --inputs=<so> --show-sass` 直接看 `.so` 仍只识别 host `elf64-x86-64`，不能作为 device ISA 证据；
  - `hipprof --codeobj-analyze` 在当前 DTK 会导出 `*-gfx938-0-resource-usage.RES` / `*-sass.ISA` 等文件，但 stdout 只列 kernel 名，资源证据以 `.RES` 和 `.ISA` 文件为准；
  - K1 V3 fused 主 kernel 在 `k1_v3_fused_ext.o-gfx938-0-sass.ISA`，统计约 `v_mmac=128`、`buffer_load=38`、`buffer_store=256`、`global_load=37`、`global_store=45`、`s_waitcnt=296`、`s_barrier=44`；
  - K1 `.RES` 中 pure/LL template 资源显示 normal-ish variants `sgpr_count≈100-102`、`vgpr_count=124/151/193`，但运行态 PMC 的 V3 fused 主 kernel报告 `arch_vgpr=220`、`sgpr=112`、`lds=65536`，更贴近当前实际路径；
  - K3 no-tail `.RES` 显示 no-tail variants `sgpr_count=100`、`sgpr_spill_count=1`、`vgpr_count=188/158/153`；运行态 K3 fused PMC 报 `arch_vgpr=216`、`sgpr=48`、`lds=65536`。
- K1 V3 vs K1 pure normal PMC（1024，PMC 本身放大 timing，timing 不作为速度证据）：
  - V3 K1 fused 主 kernel：`arch_vgpr=220`、`sgpr=112`、`VMEM_RD≈5.00M`、`VMEM_WR≈0.437M`、`VALU≈25.75M`、`LDS≈5.90M`、`TCC_MISS≈6.65M`、`TCP_TA_STALL≈3.60M`；
  - K1 pure normal pack5：`arch_vgpr=212`、`sgpr=48`、`VMEM_RD≈0.922M`、`VMEM_WR≈0.393M`、`VALU≈17.65M`、`LDS≈3.61M`、`TCC_MISS≈5.77M`、`TCP_TA_STALL≈3.07M`；
  - 差异主要是 V3 K1 的 route/staged-input 逻辑带来额外 `~4.1M` VMEM read 和 `~8.1M` VALU，而不是 GEMM 本体完全失速。
- K3 V3 normal 1024 PMC：
  - `arch_vgpr=216`、`sgpr=48`、`VMEM_RD≈1.28M`、`VMEM_WR≈0.397M`、`VALU≈15.54M`、`TCC_MISS≈2.31M`；
  - `TCP_TA_STALL≈25.2M` 且 `WRREQ_STALL=0`，继续支持“remote/scattered rowptr combine 数据通路等待”而非单纯写请求拥塞。
- 下一步 A/B 选择：
  - K1 优先尝试减少 staged-input 的全量 row copy/zero-write 或把 staging 更靠近 active rows；不能先动 MMAC/GEMM pipeline；
  - K3 不重复 store-width、empty-tile skip、non-GLC 等 LL 已反证方向，先保守评估 no-tail signal/GLC/sync 是否有可移除成本。

## 2026-06-12 normal K1 inactive staged zero A/B

- A/B 内容：
  - 默认路径保持 inactive row staged_x zero-fill；
  - 设置 `MEGAMOE_DCU_V3_K1_SKIP_INACTIVE_STAGE_ZERO=1` 时，K1 staged-input copy 对 `row_x_ptrs[row] <= 0` 的 inactive rows 不再写 zero。
- 结果：
  - 1024 K1 stage 平均约 `2.552 ms -> 2.487 ms`，只有小幅收益；
  - 4096 K1 stage 仍约 `8.8-9.1 ms`，与默认基本同量级；
  - 4096 correctness 通过但 `max_abs` 最高约 `0.0021`，比 waitcnt 默认结果更接近阈值。
- 结论：
  - inactive-row zero-fill 不是 normal K1 剩余 `pure 0.746/2.259 ms` vs fused `~2.5/~9 ms` 的主因；
  - 该分支只保留作诊断，不默认开启；
  - 下一轮应针对 PMC 指出的 route/staged-input 附加 VMEM/VALU，优先复用原 DCU MegaMoE compact prebuild kernels，避免 V3 K1 主 kernel 继续承担 count/build/emit 全量 route 扫描。

## 2026-06-12 normal K1 original compact-prebuild reuse A/B

- A/B 内容：
  - `MEGAMOE_DCU_V3_K1_REUSE_COMPACT_PREBUILD=1` 时，V3 normal K1 host 端先复用原 K1 compact prebuild kernels；
  - V3 K1 主 kernel 跳过 route count/build/emit，只做 tile-local staged input copy 和 pack5 GEMM；
  - 默认不开启，不影响当前 V3 normal 默认路径。
- 结果：
  - 1024 correctness-only 3/3 通过，稳定 K1 stage 仍约 `2.50-2.59 ms`；
  - 4096 correctness-only 3/3 通过，稳定 K1 stage 仍约 `8.84-9.04 ms`；
  - 与 waitcnt 默认路径 `~2.51-2.57 ms / ~8.79-9.01 ms` 基本同量级。
- 结论：
  - K1 剩余 delta 不由 V3 主 kernel 的 route count/build/emit 扫描主导；
  - stage timing 把 prebuild 也计入 K1 段，因此该 A/B 不是计时口径作弊；
  - 后续不继续沿“复用 prebuild 降主耗时”方向重复尝试，除非 profiler 明确显示 route counters 再次成为瓶颈；
  - 下一步应拆 staged input copy 与 staged GEMM：重点看 rowptr/row_expert/validity 分支、staged_x resource load、每 wave valid-row scan、以及 pure C normal 的 `tile_token / rows_aligned_per_expert` 固定布局与 V3 compact row layout 差异。

## 2026-06-12 normal K1 route-count validity A/B

- A/B 内容：
  - `MEGAMOE_DCU_V3_K1_ROUTE_COUNT_VALIDITY=1` 时，K1 fused normal 用 route emit counts 推导 tile 内有效行数，减少 rowptr validity 检查路径；
  - 不改变 staged input copy、ready flag/waitcnt、pack5 GEMM 主循环或默认路径。
- 结果：
  - 1024 correctness-only 3/3 通过，K1 stage 约 `2.57-2.70 ms`，不优于默认 `2.51-2.57 ms`；
  - 4096 correctness-only 3/3 通过，K1 stage 约 `8.92-9.16 ms`，不优于默认 `8.79-9.01 ms`；
  - log: `hygon_tmp/sglang_debug/v3_normal_1024_k1_route_count_validity_20260612_224141.log`、`hygon_tmp/sglang_debug/v3_normal_4096_k1_route_count_validity_20260612_224427.log`。
- 结论：
  - rowptr/validity 分支不是 normal K1 stage 主要 delta；
  - 该 A/B 只保留为 env-gated 诊断，不进入默认路径；
  - 下一步需要用 profiler/code-object evidence 拆 staged_x copy pass 和 staged GEMM 本体成本，重点看额外 VMEM/VALU、register/SGPR 压力、`s_waitcnt`/barrier 数和 staged global store/load 口径。

## 2026-06-12 normal K1 hot-path cleanup after ineffective A/B

- 已撤回项：
  - `MEGAMOE_DCU_V3_K1_SKIP_INACTIVE_STAGE_ZERO`：1024 仅小幅收益、4096 中性；
  - `MEGAMOE_DCU_V3_K1_REUSE_COMPACT_PREBUILD`：correctness 通过但 K1 stage 不改善；
  - `MEGAMOE_DCU_V3_K1_ZERO_STAGED_VALID`：出现 fused nonfinite，不安全；
  - `MEGAMOE_DCU_V3_K1_ROUTE_COUNT_VALIDITY`：1024/4096 correctness 通过但 K1 stage 不改善。
- 清理后验证：
  - 1024 default no-tail 3/3 通过，K1 stage 稳态约 `2.49-2.55 ms`；
  - 4096 default no-tail 3/3 通过，K1 stage 稳态约 `8.69-8.88 ms`；
  - 对应日志：`v3_normal_1024_default_after_clean_20260612_225640.log`、`v3_normal_4096_default_after_clean_20260612_225732.log`。
- 结论：
  - 无收益 runtime branch 会对 K1 normal codegen/调度产生可见压力，后续 A/B 失败后应及时撤出 hot kernel；
  - 当前 best 回到 waitcnt staged-input 路径，下一轮 profiler/code-object 要以清理后版本重新采集。

## 2026-06-12 normal K1 parallel staged_x copy A/B

- 触发证据：
  - 清理后 K1 PMC 1024 仍有 `VMEM_RD≈4.94M`、`VALU≈24.78M`，高于 pure `VMEM_RD≈0.922M`、`VALU≈17.65M`；
  - `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT=1` direct-rowptr 诊断 1024 correctness 通过但 K1 stage 约 `8.3-8.5 ms`，说明不能直接取消 staged_x；
  - 关键低效点是 staged_x 全行 copy 由单个 `blockIdx.x==0` CTA 串行完成，其他 output-N CTAs 等待。
- A/B 内容：
  - 保持 staged_x 全局 layout 和 GEMM 主体不变；
  - 将每个 row tile 的 4096B 输入 staging 按 `gridDim.x=16` 分段，由所有 output-N CTAs 并行复制；
  - 用同 tile 的 `stage_init/stage_count/stage_ready` flag 做 release/acquire 汇合，确保 full row staged 后所有 CTAs 再进入 GEMM。
- 结果：
  - 1024 no-tail 3/3 通过，K1 stage 稳态约 `1.36-1.50 ms`；
  - 4096 no-tail 3/3 通过，K1 stage 稳态约 `4.35-4.42 ms`；
  - 对应日志：`v3_normal_1024_k1_parallel_stage_20260612_231134.log`、`v3_normal_4096_k1_parallel_stage_20260612_231225.log`。
- 结论：
  - parallel staged_x copy 是当前 normal K1 第一轮有效性能优化；
  - 它符合“不新增 runtime kernel”和“把 dispatch/staging 隐藏进 GEMM tile scheduling”的约束；
  - 后续优化应在该版本上继续 profile，剩余 K1 delta 主要来自 staged_x global pass 本身、stage-ready 汇合、以及 route metadata/validity 固定成本。

### PMC attribution

- clean serial staged copy vs parallel staged copy, 1024 tokens:
  - `arch_vgpr/sgpr` 保持 `216/112`，说明收益不是来自寄存器压力变化；
  - `SQ_INSTS_VMEM_RD` 从约 `4.94M` 降到 `1.32M`；
  - `SQ_INSTS_VMEM_WR` 约 `0.437M` 基本不变；
  - `SQ_INSTS_VALU` 从约 `24.78M` 降到 `21.80M`；
  - `GRBM_GUI_ACTIVE` 从约 `3.29M` 降到 `1.78M`。
- 解释：
  - 原单 CTA staging 让 route/stage/GEMM 在 K1 内形成串行瓶颈，且 profiler 计入大量 staged-input 附加读和活跃周期；
  - 并行 staging 不减少总 staged_x 写入，但把 staging 工作分摊到 16 个 output-N CTAs，并减少其他 CTAs 的等待时间；
  - 当前 K1 仍慢于 pure，但 K3 no-tail combine 现在成为 normal e2e/stage 的更大瓶颈。

## 2026-06-12 normal K3 no-tail signal-off A/B

- 触发证据：
  - K1 parallel staged_x copy 后，normal 1024/4096 的 K3 no-tail combine 成为主要剩余瓶颈；
  - DCU KB / Flux GEMM+RS 参考建议通信语义应放在 epilogue store path，避免把额外同步/等待塞进主 GEMM kernel；
  - 当前 production wrapper unset `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 默认走 K3 内部 owner-slot signal + peer wait。
- A/B 内容：
  - `signal_on`: K3 internal owner-slot signal path；
  - `signal_off`: `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=0`，K3 只做 rowptr combine store，后续复用既有 rank barrier + `reduce_local_combine`；
  - `signal_off_acquire`: signal off，同时打开 `MEGAMOE_DCU_V3_BARRIER_ACQUIRE=1 MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1`。
- 结果：
  - 1024 单轮：signal-on K3 stage 约 `3.49-4.30 ms`；signal-off 约 `2.20-2.71 ms` 且 correctness 通过；signal-off-acquire K3 约 `2.24-2.50 ms`，但随后触发边界 `max_abs=0.003662109375 > 0.0035`，不作为候选。
  - 4096 单轮：signal-on K3 stage 约 `7.90-8.79 ms`；signal-off 约 `7.70-8.12 ms` 且 correctness 通过。
  - signal-off 三轮 clone-isolated correctness：
    - 1024 log `hygon_tmp/sglang_debug/v3_normal_1024_k3_signal_off_3iter_20260612_232539.log`，3/3 通过，K3 stage 稳态约 `2.19-2.29 ms`；
    - 4096 log `hygon_tmp/sglang_debug/v3_normal_4096_k3_signal_off_3iter_20260612_232539.log`，3/3 通过，K3 stage 稳态约 `7.79-8.10 ms`。
- 结论：
  - no-tail K3 内部 owner-slot signal 在 1024 上是明显性能负担，4096 上也没有收益；
  - 既有 K3 后 rank barrier + local reduce 已足以支撑 clone-isolated correctness，因此 normal no-tail 默认改为 signal-off；
  - 显式 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL=1` 保留为可见性诊断/回退开关；
  - 继续优化 K3 时不要再把 full peer wait 放回 K3 主 kernel，优先围绕 epilogue rowptr store 和 remote/scattered 数据通路做小步 A/B。

## 2026-06-12 normal K3 no-tail signal-off correctness / rowptr attribution follow-up

- DCU KB / Flux-DeepEP overlap 复核：
  - 若后续步骤要消费 peer write，需要有可见性保证；但 no-tail K3 后本来就有 rank barrier + `reduce_local_combine`，因此 K3 主 kernel 内 full peer owner-slot signal 不应作为默认性能路径；
  - 通信语义仍应留在 GEMM epilogue/store path，避免在主 GEMM path 插入额外 peer wait。
- direct K3 compare：
  - `k3_v3_distributed_combine_compare.py` 已加非有限计数和首个非有限坐标定位；
  - 4096 signal-off prezero/fill-sentinel/三次 repeat 均未复现 active/dense 非有限：`global_v3_nonfinite=0`、`global_dense_v3_nonfinite=0`、`global_active_v3_nonfinite=0`；
  - fill-sentinel 结果 `global_sentinel_active=0`，说明 active slots 被 V3 K3 覆盖，不是 active row 未写；
  - 早先一次 prezero 日志显示过 NaN 形态，但后续增强脚本 repeat 未复现，暂记录为待观察，不能作为 raw K3 根因。
- formal wrapper compare：
  - 4096 signal-off no-prezero formal run：`global_v3_vs_baseline_max=0.00054931640625`、`global_v3_gt_atol=0`、`global_manual_v3_vs_fp8_v3_max=0`；
  - 结论是 full wrapper + manual V3 path 在 clone/formal 隔离下正确；default no-clone e2e 非有限更像 output lifecycle / harness artifact，单独跟踪。
- K3 rowptr split 4096：
  - `rowptr_all_zero` 约 `2.543 ms`，接近 K3 GEMM floor；
  - `local_rowptr` 约 `3.306 ms`，`staged_local_only` 约 `3.461 ms`；
  - `staged_remote_only` 约 `7.814 ms`，`staged_rowptr` 约 `7.842 ms`；
  - row distribution: 平均每 rank active rows 约 `24576`，remote rows 约 `21606`，16-row chunk 中平均 unique dest ranks `1.806`，最大同 rank 平均 `12.337`。
- 结论：
  - normal K3 no-tail 4096 的主要 gap 是 remote/scattered rowptr combine store 数据通路，不是 GEMM floor；
  - 原 K3COMBINE ASM 也有大量 scalar `buffer_store_short`，所以不能仅凭“store width”判断；
  - 下一步按 optimizer flow 对比原 K3 ASM no-tail 与 V3 K3 no-tail 的 stage timing、PMC read/write、resource/ISA store schedule，再选择是否尝试 row ordering、epilogue store schedule 或 waitcnt 调整。

## 2026-06-13 normal K3 no-tail ASM vs V3 PMC 对照

- 原 ASM no-tail stage timing（`USE_MEGAMOE_V3=0`，`K3_USE_ASM_TAIL_REDUCE=0`，`MEGAMOE_DCU_V3_STAGE_TIMING=1`）：
  - 1024 last3 median：K1 `1.242 ms`，K2 `0.087 ms`，K3 combine `0.901 ms`，no-tail barrier `0.154 ms`，reduce `0.047 ms`；
  - 4096 last3 median：K1 `3.562 ms`，K2 `0.215 ms`，K3 combine `2.642 ms`，no-tail barrier `0.217 ms`，reduce `0.166 ms`。
- V3 normal signal-off stage timing：
  - 1024 last3 median：K1 `1.408 ms`，K2 `0.107 ms`，K3 combine `2.244 ms`，barrier `0.105 ms`，reduce `0.047 ms`；
  - 4096 last3 median：K1 `4.402 ms`，K2 `0.217 ms`，K3 combine `7.919 ms`，barrier `0.183 ms`，reduce `0.167 ms`；
  - 结论：normal 主要 gap 已从 K1 转到 K3，4096 K3 约 `+5.28 ms`。
- ASM rowptr split 4096（同一 bench 脚本，ASM L2 使用 Marlin layout，K1 metadata 用 V3 normal）：
  - ASM: `rowptr_all_zero/local_rowptr/staged_remote_only/staged_rowptr ≈ 1.554/1.615/3.094/3.113 ms`；
  - V3: `2.543/3.306/7.814/7.842 ms`；
  - 结论：V3 不只是 remote store 慢，GEMM floor 也比 ASM 高约 `0.99 ms`，但主 gap 仍在 remote/scattered epilogue path。
- PMC exact kernel match（4096 staged_rowptr，24 launches/ranks 汇总）：
  - ASM K3COMBINE：`VMEM_RD≈2.264M`，`VMEM_WR≈0.238M`，`VALU≈58.9M`，`LDS≈10.2M`，`arch_vgpr=256`，`sgpr=112`，`TA_BUSY≈64.5M`，`TCP_TA_DATA_STALL≈10.0M`，`TCC_MISS≈5.29M`，`TCC_WRREQ_STALL=0`；
  - V3 K3 normal：`VMEM_RD≈4.625M`，`VMEM_WR≈1.576M`，`VALU≈55.6M`，`LDS≈9.86M`，`arch_vgpr=216`，`sgpr=32`，`TA_BUSY≈193.1M`，`TCP_TA_DATA_STALL≈88.0M`，`TCC_MISS≈5.29M`，`TCC_WRREQ_STALL=0`；
  - ratio：V3/ASM `VMEM_RD≈2.04x`，`VMEM_WR≈6.63x`，`TA_BUSY≈3.0x`，`TCP_TA_DATA_STALL≈8.8x`，TCC miss 基本持平。
- 当前优化假设：
  - K3 normal no-tail 主要问题不是 MMAC/compute 指令数，也不是 TCC miss 数量暴增；
  - 优先定位 V3 epilogue 每 row/lane 的 rowptr load、fallback output store、debug/output workspace store 或多余 global write，解释 `VMEM_WR` 为 ASM 的 `6.6x`；
  - 若源码确认没有明显多余写，再进入 code-object/ISA 对比 store schedule、waitcnt 与 resource 使用。

### rowaddr hoist A/B 反证

- A/B 内容：
  - normal K3 epilogue 将 `token_group0/1 + {0,4,8,12}` 的 rowptr 地址提前 load 一次，并让所有 hidden-R store 复用；
  - 第一版用 `V3RowAddr4` 结构体传参，第二版改成 8 个 `int64_t` 标量传参；
  - 不改变 store width、不改变 no-tail signal/barrier/reduce、不改变 GEMM compute。
- 结果：
  - 两版 1024 首轮 correctness 均通过，K3 stage 有正向信号（约 `2.0-2.18 ms`，低于基线 `2.22-2.29 ms`）；
  - 但三轮 correctness 第 2 轮失败：结构体版 `max_abs=0.004711 > 0.0035`，标量版 `max_abs=0.004456 > 0.0035`；
  - 撤回后 1024 三轮 correctness 通过，K3 stage 回到约 `2.22-2.29 ms`。
- 结论：
  - 跨 hidden-R 复用 epilogue row address 在当前 normal K3 no-tail 下不安全，不能作为默认优化；
  - 该方向后续不重复，除非先用 direct K3 compare/ISA 证明具体 hazard 并给出新的同步或 scheduling 证据；
  - 下一步转向 compute 侧 rowptr active-check hoist：只 hoist active 布尔，不 hoist epilogue output address。

### active-check hoist A/B 反证

- A/B 内容：
  - normal K3 compute 中 `token_col0/1` 的 rowptr active 判断从每个 K phase 内的 `global_load_i64_glc_device(row_combine_ptrs + token_col)` 改为 compute 前各 load 一次 bool；
  - phase 内 B-load 只根据 bool 决定是否返回 zero pack，epilogue store 完全不变。
- 结果：
  - 1024 首轮 correctness 通过，K3 stage 有正向信号（约 `1.99-2.05 ms`，低于基线约 `2.22-2.29 ms`）；
  - 三轮 correctness 第 2 轮失败：`max_abs=0.012939 > 0.0035`；
  - 已撤回该 A/B。
- 结论：
  - 在 normal K3 当前 codegen 下，跨 compute loop 长生命周期保存 rowptr-derived active 状态不安全；
  - 后续不重复“rowptr/active 跨 compute 长生命周期 hoist”方向；
  - 下一步应优先用 code-object/ISA 对比原 ASM 与 V3 store/load schedule，或尝试只在极近邻指令窗口内调度 rowptr load/store，不跨越完整 GEMM compute。

### short-window rowptr nowait A/B 反证

- 触发证据：
  - 当前 V3 normal no-tail codeobj HTML / LLVM objdump 显示 no-tail 函数每个 store group 中有 `global_load_dwordx2 ... glc` 后立即 `s_waitcnt vmcnt(0)`、`v_cmp_lt_i64`、exec mask、`global_store_short`；
  - no-tail template 函数计数约 `global_load_dwordx2=132`、`global_store_short=128`、`s_waitcnt=217`、`v_mmac=128`；
  - 原 ASM K3COMBINE 源在 epilogue store window 中是成批 `buffer_store_short`，store window 本身不包含逐 store rowptr `global_load_dwordx2 + s_waitcnt`。
- A/B 内容：
  - 在 `store_acc_fragment_scaled_unmasked_device` 内先发出 `token_base + {0,4,8,12}` 的 rowptr `global_load_dwordx2 ... glc`，不立即 wait；
  - 利用等待窗口执行 scale/mul/BF16 pack；
  - store 前统一 `s_waitcnt vmcnt(0)`，再按四个 row address 检查并 `global_store_short`；
  - 不跨 compute loop、不跨 hidden-R 复用 rowptr 状态，不改变 store width、signal、barrier/reduce 或 GEMM 主循环。
- 结果：
  - K3 V3 normal aicc 强制重编成功；
  - 1024 normal no-tail skip-bench correctness 启动后触发 VMFault / SIGABRT，log `hygon_tmp/sglang_debug/v3_normal_1024_k3_short_window_rowptr_skipbench_20260613_004715.log`；
  - 撤回后强制重编，1024 三轮 correctness 重新通过，log `hygon_tmp/sglang_debug/v3_normal_1024_after_revert_short_window_skipbench_20260613_005002.log`。
- 结论：
  - 对 rowptr `global_load_dwordx2` 使用 no-wait inline asm 并延后消费在当前 aicc/codegen 下不安全；
  - 后续不重复“手写 no-wait rowptr load + 延迟 wait”的方向；
  - 若继续优化 wait/store schedule，应优先考虑不破坏 compiler/hazard 可见性的结构调整，或回到 ASM-style address preparation / buffer-store 形态，而不是直接移除每次 load helper 内的 wait。

### rowaddr group4 / non-GLC A/B 反证

- ASM 差异线索：
  - 原 K3COMBINE direct epilogue 通过 `K3_LOAD_COMBINE_ADDR4` 对 4 行 rowptr load 一次，然后 `K3_INC_ADDR4` 每次地址加 0x20，连续覆盖 16 个 hidden step；
  - V3 normal C epilogue 当前每个 hidden step 重新 load rowptr，再做 `global_store_short`，这是 V3 `VMEM_RD/TCP_TA_DATA_STALL` 放大的主要代码形态之一。
- rowaddr group4 A/B：
  - 将 rowptr 复用缩短为 4 个 hidden step 一组，而不是此前失败的 16-step 长 live range；
  - 初版与 scoped live-range 版都有 stage timing 信号，但 1024 第 2 轮分别失败在 `max_abs=0.0035400390625` 和 `0.0037994384765625`；
  - 说明当前 C/aicc 下“跨 hidden step 复用 rowptr address”仍不满足 correctness gate，不再继续缩小为 group2 重复试。
- non-GLC rowptr A/B：
  - KB 说明 `glc` 是 L2-oriented load path；原 K3COMBINE rowptr load 无 `glc`，因此做 normal-only no-GLC rowptr load 试验；
  - 1024 rank0 三轮数值通过，但其他 rank 出现 fused nonfinite，且 K3 stage 未改善；
  - 该方向也撤回，不作为 normal K3 默认路径。
- 回退后 raw K3 证据：
  - direct K3 compare 4096 在回退版本上 `global_max_abs=0`、`global_dense_gt_atol=0`、`global_v3_nonfinite=0`；
  - e2e nonfinite 仍按 output lifecycle / harness artifact 跟踪，不能用来重新引入 K3 内 full peer wait/signal。

## 2026-06-13 normal K3 staged-half LDS/vector-store A/B 撤回

- ASM 线索：
  - 原 K3COMBINE no-tail 在 epilogue 中先把半个 tile 的 BF16 结果写入 LDS，再由半数 waves 通过 `ds_read_b128` + `global_store_dwordx4` 写回 rowptr；
  - 对应源码宏为 `K3_STAGE_TILE_H0/H1`、`K3_STORE_STAGED_HALF`，中间有 `s_waitcnt lgkmcnt(0); s_barrier`；
  - 该路径解释了原 ASM 的低 `VMEM_WR` 指令数和 batched store 形态，是正常值得尝试的方向。
- A/B 内容：
  - 在 V3 K3 normal no-tail 256-N 路径中仿照 ASM，把 scalar rowptr BF16 store 改为先 `ds_write_b16` 到 LDS，再 `ds_read_b128` 后 `global_store_dwordx4`；
  - 第一版没有显式 LDS drain，第二版在 stage 写后加入 `s_waitcnt lgkmcnt(0)`，第三版尝试 ASM-style row4 store loop。
- 结果：
  - 第一版 1024 e2e K3 stage 降到约 `1.05-1.18 ms`，但 correctness 失败；direct 4096 compare 出现 `global_max_abs=0.98828125`、`global_gt_atol=62758`、`global_v3_nonfinite=6`；
  - 加 `lgkmcnt(0)` 后 nonfinite 消失，但 direct 4096 仍错误：`global_max_abs=1.1761474609375`、`global_gt_atol=59757`；
  - row4 store loop 更差：direct 4096 `global_max_abs=2.730224609375`、`global_gt_atol=3939045`；
  - save-temps/ISA 证明该版本确实生成 `ds_write_b16`、`ds_read_b128`、`global_store_dwordx4`，不是 address-space lowering 到 flat/global 的问题。
- 结论：
  - staged-half 有明确性能潜力，但当前 C/aicc 映射下数值不等价；`lgkmcnt` 修掉 LDS 可见性非有限，但没有修掉 row/value mapping；
  - row4 store loop 已反证，不能保留；
  - 默认 no-tail 路径已回退到已验证正确的 scalar rowptr store 基线；后续若恢复该方向，必须先做更小的 source-backed ASM-style store probe 或定位 LDS staged tile 的 exact layout，再接回主 kernel。

## 2026-06-13 normal K3 no-tail fast-exit A/B 撤回

- ASM 线索：
  - 原 K3COMBINE no-tail epilogue 在 combine store 后以 `s_waitcnt vmcnt(0); buffer_wbinvl1_vol` 结束；
  - V3 C no-tail 默认在 store 后继续执行 `wait_vmem_lds_store_device(); block_barrier_device(); __threadfence_system()`，理论上可能有多余 block barrier/fence 成本。
- A/B 内容：
  - 仅对 `!kTailReduce && !kSignalOnly` no-tail 路径尝试 fast-exit：store 后 `wait_vmem_lds_store_device(); invalidate_l1_device(); return;`；
  - 不改变 rowptr store、GEMM 主循环、外部 no-tail rank barrier/reduce。
- 结果：
  - direct K3 compare 4096 通过阈值：`global_max_abs=0.0032958984375`、`global_gt_atol=0`、无 nonfinite；
  - 1024 staged e2e 首轮 `max_abs=0.000488281` 后，后续 rank 触发 `fused/baseline nonfinite fused=39 baseline=0 diff=39`；
  - stage timing 没有改善，K3 combine 仍约 `2.19-2.25 ms`，且 no-tail barrier 在多 rank 上升到约 `0.8-0.94 ms`。
- 结论：
  - no-tail fast-exit 不是有效性能优化，并且破坏当前 staged e2e 生命周期/可见性稳定性；
  - 已从本地源码撤回，默认路径恢复 `wait_vmem_lds_store_device(); block_barrier_device(); __threadfence_system()`；
  - 后续不再把 no-tail 的 block barrier/fence 简化作为主优化方向，除非先有新的 ISA/PMC 证据证明该边界是主要瓶颈且 correctness 可稳定通过。

## 2026-06-13 normal K3 rowptr buffer-store A/B 撤回

- 触发证据：
  - 原 K3COMBINE ASM epilogue 使用 `buffer_store_short` 写 combine；
  - V3 scalar 基线使用 `global_store_short`，且 V3 PMC 中 `VMEM_WR` 明显高于 ASM；
  - V3 源码已有 raw-buffer store helper，可做单变量 store-family A/B。
- A/B 内容：
  - `store_bf16_rowptr_device` 保持 rowptr `global_load_dwordx2 glc + s_waitcnt`、地址检查、GEMM 主体和同步语义不变；
  - 只把最终 BF16 store 从 `global_store_bf16_device(row_ptr + hidden, value)` 改为基于 row address resource 的 `buffer_store_bf16_device(...)`。
- 结果：
  - direct K3 compare 4096 通过阈值：`global_max_abs=0.00286865234375`、`global_gt_atol=0`、无 nonfinite；
  - rowptr split bench 显示无性能收益且 4096 退化：1024 `staged_rowptr≈2.29 ms`，4096 `staged_rowptr≈8.09 ms`，慢于 scalar 基线约 `2.24/7.84 ms`；
  - `dccobjdump` 对 Python extension `.so` 只识别 host ELF，没有拆出 device ISA，本次 ISA 证据记为 inconclusive，不用于保留该分支。
- 结论：
  - 单纯把 rowptr scalar store 改成 per-store raw-buffer resource 不解决 V3 的 remote/scatter store gap，还会增加地址/resource 构造成本；
  - 已撤回到 `global_store_short` scalar 基线；
  - 后续如果继续 ASM-style store family，必须先找到 device code object/ISA dump 或做独立 source-backed store probe，不能在主 kernel 内重复 per-store resource 化。

## 2026-06-13 normal K3 unconditional B-load A/B 撤回

- 假设：
  - V3 normal compute path 在每个 K phase 用 `row_combine_ptrs` 判断 B 行 active，导致大量重复 rowptr load；
  - no-tail epilogue 也会按 rowptr 跳过 inactive rows，因此尝试让 compute path 像 pure C groupgemm 一样无条件读 `act_fp8`，只在 store 阶段过滤。
- A/B 内容：
  - 将 `K1_DEEPGEMM_PHASE_LOAD_B` 和 phase0 B-load 从 `buffer_load_fp8_b128_active_row_device(...)` 改为 `buffer_load_fp8_b128_pack_device(input_resource, token_col * kProblemK + phase_k)`；
  - 不改变 epilogue rowptr store、GEMM tile organization 或同步语义。
- 结果：
  - 4096 direct K3 compare 失败：`global_max_abs=0.0303955078125`、`global_gt_atol=1223862`，无 nonfinite；
  - 失败说明 compute 阶段 active mask 不是单纯冗余，inactive column 的 B 值会污染可比较输出，或当前 MMAC lane mapping 需要在 compute 前置零。
- 结论：
  - 已撤回 unconditional B-load；
  - 后续不能直接移除 compute active mask；若要减少 rowptr load，只能在不跨长生命周期且能证明 lane mapping 正确的方式下局部缓存/生成 mask，或者改变 input padding/fixture 让 inactive B 确认为 zero。

## 2026-06-13 normal K3 per-stage active-mask A/B 撤回

- 假设：
  - full unconditional B-load 失败说明 inactive B 必须置零；
  - 但当前每个 K phase 都重新读取 rowptr active，phase0 和 phase4 在同一 stage 内可以共享一个 active bool，缩短 live range 后可能比此前 full-compute active hoist 更安全。
- A/B 内容：
  - 在每个 `stage_iter` 内为 `token_col0/1` 各读取一次 `row_combine_ptrs > 0`；
  - phase0 和 phase4 的 B-load 使用该 bool 决定加载 `act_fp8` 或 zero pack；
  - 不跨 stage 复用 active mask，不改 epilogue store。
- 结果：
  - direct K3 compare 4096 未通过：`global_max_abs=0.0057373046875`、`global_gt_atol=24`，无 nonfinite；
  - 比 unconditional B-load 的大面积错误轻很多，但仍超过 correctness gate。
- 结论：
  - 当前 normal K3 对 B active mask 的 rowptr check 必须保持在每个 phase 的 helper 内，至少在现有 aicc/codegen 下不能复用 active bool；
  - 已撤回 per-stage active-mask A/B；
  - 后续不再重复 active-mask hoist，除非先有更小 ISA/source probe 证明 hazard 和修复方式。

## 2026-06-13 normal K3 staged-half H1 rowptr offset A/B 撤回

- 新线索：
  - 原 `K3COMBINE` ASM 的 `K3_STORE_STAGED_HALF 1024` 参数是 rowptr raw-buffer load 的 byte offset，不是 row count；
  - 对 int64 rowptr 而言，1024 byte 等价于 128 行；因此 C helper 若把 H1 offset 写成 1024 行，会直接错写第二半 tile 的 combine pointer。
- A/B 内容：
  - 只在 normal K3 no-tail N=256 epilogue 启用 staged-half LDS/vector-store；
  - H0 使用 row offset 0，H1 使用 row offset 128，tail-reduce/signal-only 保持 scalar rowptr store。
- 结果：
  - 远端 aicc rebuild 成功；
  - 4096 direct K3 compare 仍失败：`global_max_abs=0.17596435546875`、`global_gt_atol=200`、`global_v3_nonfinite=33`；
  - 相比首轮 staged-half 的 `global_max_abs≈0.99/1.17` 和大量超差，H1 offset 修正确实收窄错误，但 value/layout 仍未完全等价。
- 结论：
  - H1 offset 是 staged-half 的必要修正，但不是充分修复；
  - 默认路径已恢复 scalar rowptr store；
  - staged-half 后续要先做独立 source-backed value/layout probe，定位 LDS row/value mapping，再考虑回到主 K3 kernel。

## 2026-06-13 normal K1 direct-source / fence-scope 反证

- `MEGAMOE_DCU_V3_K1_DISABLE_STAGE_INPUT=1` 证明 normal K1 不能直接从 `row_x_ptrs` 在 GEMM 主循环中远端散读：
  - 1024 no-tail correctness 通过；
  - K1 stage 从 staged-input baseline 约 `1.44 ms` 退到约 `8.42 ms`；
  - 结论是 staged copy 对 normal 后端仍是必要通信隐藏层，不能为了更接近 pure C 形态直接去掉。
- K1 internal fence-scope A/B：
  - DCU KB/HIP 同步资料确认 `__threadfence()` 对同设备线程可见，`__threadfence_system()` 面向 host/system；
  - 但把 K1 internal route/stage ready fence 降为 device-scope 后，1024 K1 stage 没有收益，direct K1 compare 虽通过但 env-gated branch 扰动 default 多轮稳定性；
  - 结论是 K1 normal 当前 delta 不是 fence scope 主导，后续不重复该方向。

## 2026-06-13 normal K3 K2-zero + unconditional B-load A/B 撤回

- 假设：
  - 之前 pure-style unconditional B-load 失败，可能是因为 K2 对 inactive rows 直接 return，导致 `act_fp8/act_scale` 中有 stale 值；
  - 若 K2 在同一个 K2 kernel 内把 inactive rows 清零，K3 compute 可能可以移除每 phase 的 rowptr active check，恢复 contiguous B-load 形态。
- A/B 内容：
  - 临时增加 `MEGAMOE_DCU_V3_K3_ASSUME_ZERO_INACTIVE_B=1`，让 K2 对 `row_combine_ptrs[row] == 0` 的 inactive rows 写零；
  - K3 normal no-tail 在该 env 下走 pure-style unconditional B-load，epilogue rowptr store/filter 保持不变；
  - 不新增 runtime kernel。
- 结果：
  - 1024 e2e 有性能信号：K3 stage 多数 rank 从约 `2.23 ms` 降到约 `1.97-2.04 ms`，但 fused 输出出现 nonfinite；
  - direct compare 1024 证明 K2 zeroing 生效：`global_inactive_act_nonzero=0`、`global_inactive_act_nonfinite=0`；
  - 同一 direct compare 下 K3 unconditional B-load 仍失败：`global_max_abs=0.020660400390625`、`global_gt_atol=123423`、`global_v3_nonfinite=48`，且 nonfinite 出现在 active/dense slot。
- 处理：
  - 已撤回 `zero_inactive_rows` / `assume_zero_inactive_b` API、env gate 和 K3 专用 template instantiation；
  - 回退后重编通过：`hygon_tmp/sglang_debug/rebuild_v3_normal_revert_k2zero_k3uncond_retry_20260613_030356.log`；
  - 回退后 direct K3 1024 通过：`global_max_abs=0`、`global_gt_atol=0`、`global_v3_nonfinite=0`；
  - 回退后 e2e 1024 no-tail 三轮 correctness 通过，K1 stage 约 `1.33-1.46 ms`、K3 combine 约 `2.19-2.29 ms`。
- 结论：
  - K3 compute active mask 不是只用于挡 stale inactive B；它是当前 normal K3 MMAC/lane 语义的一部分；
  - 后续不要重复 “K2 清 inactive + K3 unconditional B-load” 方向。

## 2026-06-13 normal K3 gated staged-half A/B 反证

- 新假设：
  - 原 K3COMBINE ASM 的 no-tail staged-half 是 H0/H1 分阶段执行：wave0-3 stage/store H0，wave4-7 stage/store H1；
  - 此前 staged-half 失败可能不只是 H1 rowptr offset，也可能是 H0/H1 wave gating 与 LDS 复用时序不等价。
- A/B 内容：
  - V3 K3 normal no-tail N=256 路径按 ASM 时序做 gated staged-half；
  - H0 使用 rowptr offset 0，H1 使用 rowptr offset 128；
  - 每个 half stage 后 `lgkmcnt(0)` + block barrier，store 后再 barrier；
  - tail-reduce/signal-only 保持 scalar rowptr store。
- 结果：
  - 1024 direct K3 compare 通过：`global_max_abs=0.001941680908203125`、`global_gt_atol=0`、无 nonfinite；
  - 4096 direct K3 compare 失败：`global_max_abs=0.26544189453125`、`global_gt_atol=422`、`global_v3_nonfinite=15`；
  - nonfinite 主要在 inactive dense slot，active slot 无 nonfinite，说明不是单纯 store 可见性，而是 row/value mapping 或 inactive slot 写入行为仍和 ASM 不等价。
- 结论：
  - H0/H1 gating 修正后仍不能过 4096，因此 staged-half 不能作为当前 normal K3 主路径优化；
  - 后续若继续该方向，必须先做独立 probe：用可追踪 pattern 验证 `stage_acc_fragment_scaled_unmasked_device` 到 `K1_STORE_STAGED_HALF` 的 row/hidden mapping 与 ASM 完全一致；
  - 主优化方向回到 scalar rowptr store 基线上的可证明结构调整、PMC/ISA/SQTT 证据，而不是继续直接改 staged-half 主路径。

## 2026-06-13 normal K3 rowptr raw-buffer load/resource A/B

- 假设：
  - V3 normal K3 在 compute active-check 和 epilogue address fetch 中重复执行 rowptr global dwordx2 load，是 VMEM read/TCP stall floor 的一部分；
  - 将 rowptr load 改成 `row_combine_ptrs` buffer resource raw dwordx2 load，可能更接近 ASM rowptr load 形态，同时避免此前 no-wait/non-GLC 的 hazard。
- A/B 内容：
  - compute B-load active-check 使用 rowptr resource load；
  - epilogue row pointer address fetch 使用 rowptr resource load；
  - final combine store 仍为 scalar `global_store_short`，不同于已撤回的 per-store raw-buffer store。
- 结果：
  - direct K3 1024/4096 correctness 通过，4096 `global_max_abs=0.002838134765625`、`global_gt_atol=0`、无 nonfinite；
  - rowptr split 4096 all_zero 从约 `2.543 ms` 降到 `2.153 ms`，local 从约 `3.306 ms` 到 `3.276 ms`，staged_remote/staged 仍约 `7.781/7.808 ms`。
- 结论：
  - rowptr load form 是 K3 floor 的一部分，但主要 remote/scattered combine-store gap 仍未解决；
  - 保留该改动，但需要进一步 attribution，避免 epilogue resource load 对 remote store path 产生隐性负收益；
  - profiler 当前 degraded：filtered PMC 空、no-filter PMC 挂住、ISA dump 未稳定拿到 device code object，后续继续优先用 direct correctness + rowptr split timing 做小步归因。

## 2026-06-13 normal K3 compute-only rowptr resource A/B 反证

- A/B 内容：compute active-check 继续使用 rowptr resource load，epilogue rowptr address fetch 恢复到原 `global_load_i64_glc_device`。
- 结果：direct K3 1024 通过，但 4096 出现 `global_max_abs=0.005279541015625`、`global_gt_atol=1`，无 nonfinite。
- 结论：
  - epilogue rowptr resource load 不能视为无关优化项直接移除；
  - 该失败更像 aicc/codegen/数值调度敏感点，而不是明显的可见性问题；
  - 后续 rowptr resource 归因不再用“epilogue 回到 glc”的主路径 A/B，除非先有 ISA 或更小 probe 解释这个单点漂移。

## 2026-06-13 normal K3 staged-half + rowptr resource A/B 反证

- 新假设：
  - 此前 staged-half H0/H1 gating 失败时，staged store 仍使用 global/glc rowptr load；
  - rowptr raw-buffer load/resource A/B 已证明 scalar epilogue 下 rowptr resource load 可正确，因此尝试把 staged-half store 的 rowptr load 也改成 resource load。
- A/B 内容：
  - normal K3 no-tail N=256 路径按 H0/H1 gated staged-half 执行；
  - `K1_STORE_STAGED_HALF` 内 rowptr load 改为 `buffer_load_i64_device(rowptr_resource, row * 8)`；
  - H0 row offset 0，H1 row offset 128；tail-reduce/signal-only 仍为 scalar store。
- 结果：
  - aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_staged_half_resource_20260613_041515.log`；
  - direct K3 1024 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_staged_half_resource_20260613_041905.log`；
  - 关键统计：`global_max_abs=1.31982421875`、`global_gt_atol=18733`、`global_v3_nonfinite=0`。
- 回退：
  - 撤回 staged-half 主路径，恢复 scalar `K1_STORE_ROWS_256(K1_STORE_ROW_UNMASKED)`，rowptr resource scalar store 保留；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_staged_half_resource_20260613_042033.log`；
  - direct K3 1024/4096 均通过，logs `k3_v3_dist_1024_after_revert_staged_half_resource_20260613_042142.log`、`k3_v3_dist_4096_after_revert_staged_half_resource_20260613_042219.log`，两者 `global_max_abs=0`、`global_gt_atol=0`、无 nonfinite。
- 结论：
  - staged-half 的根因仍是 LDS row/value mapping 或 inactive row 写入行为不等价，不是 rowptr load family 本身；
  - 后续不再把 staged-half 主路径变体直接接入 K3 normal no-tail；若要恢复，只先做独立 value/layout probe，对比 `stage_acc_fragment_scaled_unmasked_device` 到 `K1_STORE_STAGED_HALF` 的 row/hidden 映射。

## 2026-06-13 normal K3 rowaddr resource reuse A/B 反证

- 假设：
  - ASM direct epilogue 通过 `K3_LOAD_COMBINE_ADDR4` 先取 4 个 row address，然后用 `K3_INC_ADDR4` 连续写 16 个 hidden step；
  - 之前 rowaddr group/reuse 使用 global/glc rowptr load 时出现少量 correctness drift；当前 rowptr resource scalar path 已通过，因此尝试 resource rowptr + rowaddr reuse。
- A/B 内容：
  - no-tail N=256 epilogue 中，分别为 `token_group0` 和 `token_group1` 预取 4 个 row address；
  - 后续 16 个 R step 复用这些 row address，只做 hidden offset 加法和 scalar BF16 store；
  - tail-reduce/signal-only 继续走原 scalar rowptr store。
- 结果：
  - 首次远端编译因调用不存在的 `store_bf16_rowaddr_device` 失败，属于实现错误，已修正为直接 `global_store_bf16_device`；
  - 修正后 aicc rebuild 通过，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowaddr_resource_reuse_retry_20260613_042940.log`；
  - direct K3 1024 未过 gate，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_rowaddr_resource_reuse_20260613_043048.log`，`global_max_abs=0.004998207092285156`、`global_gt_atol=1`、无 nonfinite。
- 回退：
  - 撤回 rowaddr reuse，恢复 scalar rowptr-resource store；
  - restore rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_rowaddr_resource_reuse_20260613_043205.log`；
  - restore direct 1024 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_after_revert_rowaddr_resource_reuse_20260613_043313.log`，`global_max_abs=0.001922607421875`、`global_gt_atol=0`、无 nonfinite。
- 结论：
  - rowaddr long-live reuse 是当前 C/aicc 下的 correctness 敏感点，resource rowptr load 不能修复；
  - normal K3 no-tail 后续不再重复 rowaddr group/reuse 方向，除非能先拿到 ISA 证明 live-range/scheduling hazard 并做更小的 source probe。

## 2026-06-13 normal K3 rowptr4 batch-load A/B clean rebuild 后撤回

- 背景：
  - 当前稳定 rowptr-resource 基线已证明 raw-buffer rowptr load 能降低 K3 floor，但 scalar store 路径仍每 4 行内重复执行 rowptr load + wait；
  - 原 ASM direct epilogue 有批量 rowptr load 的结构线索，但跨 hidden-step rowaddr reuse 已多次反证，因此本次只做同一 R step/token group 内的短生命周期 batch-load，不跨 R 复用地址。
- A/B 内容：
  - 新增 `store_bf16_rowptr4_buffer_device`，一次取同一 token group 的 4 个 row pointer，单次 `s_waitcnt vmcnt(0)` 后分别做 scalar `global_store_short`；
  - 新增 `store_acc_fragment_scaled_unmasked4_device`，只在 normal K3 no-tail `!kTailReduce && !kSignalOnly` 路径调用；
  - tail-reduce/signal-only 继续走旧 scalar helper，避免影响暂缓的 normal tail。
- 结果：
  - direct K3 1024/4096 均通过，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - 1024 rowptr split all_zero/local/staged_remote/staged 从上一基线 `0.605/0.937/2.216/2.237 ms` 改为 `0.544/0.875/2.159/2.166 ms`；
  - 4096 rowptr split all_zero/local/staged_remote/staged 从上一基线 `2.153/3.276/7.781/7.808 ms` 改为 `1.935/3.064/7.623/7.661 ms`。
- 结论：
  - 该改动首轮是正确且可测的小幅正收益，说明 rowptr wait/load grouping 是 K3 floor 的一部分；
  - 但 clean rebuild 后同一 rowptr4 路径在 4096 direct correctness 出现 `global_gt_atol>0`，后续 rowptr8 batch-load 在 4096 也失败；
  - 因此 rowptr4/rowptr8 不能作为 retained 优化，已撤回到 scalar rowptr-resource store；后续不要重复 rowptr4/rowptr8 batch-load，除非先有更小 source-backed probe 或 ISA 证据解释 aicc/codegen 数值漂移；
  - 主 gap 仍未解决：4096 staged 仍比原 ASM no-tail `~3.113 ms` 慢约 2.5 倍，主要差距仍在 remote/scattered combine store 数据通路和 ASM store schedule 形态。

## 2026-06-13 normal K3 scalar rowptr store pack dependency

- 背景：
  - rowptr4/rowptr8 撤回后，默认 no-tail 路径回到 scalar rowptr-resource store；
  - clean rebuild 下 scalar 路径首轮 4096 direct compare 仍出现小漂移，形态为 active dense slot 少量 `global_gt_atol`，无 nonfinite；
  - DCU KB 命中 Hygon 参考中用 `s_nop` / dummy dependency 规避 hazard 的案例，也提示 bf16 convert/pack 与 inline asm 组合存在 codegen/spill 敏感性。
- 改动：
  - 仅在 `store_acc_fragment_scaled_unmasked_device` 内把两个 BF16 pack 结果都先计算出来；
  - 增加 `asm volatile("s_nop 0" : "+v"(out_bits0), "+v"(out_bits1) :: "memory")`，再执行四次 scalar rowptr store；
  - 不改变 GEMM 主循环、rowptr resource load、no-tail block barrier/system fence 或 tail/signal 路径。
- 验证：
  - 本地 `python -m compileall megamoe/dcu_megamoe_large_opt/K3_fused/k3_fused.py tests/test_dcu_megamoe_v3.py` 通过；
  - 远端强制删除 K3 V3 object/so 后 aicc rebuild 通过：`hygon_tmp/sglang_debug/rebuild_v3_normal_k3_scalar_pack_dep_force_20260613_080448.log`；
  - direct K3 1024 通过：`hygon_tmp/sglang_debug/k3_v3_dist_1024_scalar_pack_dep_seq2_20260613_080925.log`，`global_max_abs=0`、`global_gt_atol=0`；
  - direct K3 4096 在首轮 `k3_v3_dist_4096_scalar_pack_dep_20260613_080616.log` 出现 `global_max_abs=0.0050048828125`、`global_gt_atol=6`，随后三次 4096 repeat 和一次 1024->4096 顺序复测均 `global_max_abs=0`、`global_gt_atol=0`。
- 结论：
  - scalar pack dependency 不是性能优化，只是降低 aicc 调度敏感性的保守修复；
  - 当前可继续跑 rowptr split 性能，但首次 4096 漂移要作为观察项保留；进入 e2e/bench 前仍需至少一次 1024/4096 direct correctness 复测。

## 2026-06-13 normal K3 no-tail scalar-pack 后 ISA 归因

- rowptr split 复测：
  - 1024 all_zero/local/staged_remote/staged `0.609/0.944/2.211/2.222 ms`；
  - 4096 all_zero/local/staged_remote/staged `2.172/3.304/7.767/7.804 ms`；
  - 与 scalar rowptr-resource 基线基本一致，说明 pack dependency 主要是 correctness/codegen 稳定性修复，不是性能修复。
- 工具路径：
  - `hipprof --codeobj-analyze` 对 Python extension `.so` 会要求交互式选择 device ELF，当前自动命令不可用；
  - 可行路径是 `dccobjdump --inputs=<so> --extract-elf=all` 抽出 `*gfx938.out`，再用 `/opt/dtk/aillvm/bin/llvm-objdump -d --mcpu=gfx938` 反汇编。
- 当前 V3 K3 normal no-tail ISA 计数：
  - `global_store_short=128`、`buffer_store_short=0`；
  - `buffer_load_dwordx2=132`、`buffer_load_dwordx4=132`；
  - `s_waitcnt=353`、`v_mmac=128`、`ds_read=56`、`s_barrier=21`。
- 原 `K3COMBINE.s` 源码计数：
  - `buffer_store_short=512`、`global_store_short=4`；
  - `s_waitcnt=149`、`v_mmac=288`、`ds_read=197`、`s_barrier=30`。
- 解释：
  - 不能只按 store 指令数量判断；ASM 使用更多 `buffer_store_short`，但 waitcnt 更少且 store window 组织不同；
  - V3 当前主要风险是每 rowptr/store 附近的 wait 调度和 scalar global-store data path 导致 remote/scattered combine 路径长尾；
  - 后续小步 A/B 应围绕 store/wait scheduling 或独立 ASM-style store probe，不能重复已反证的 rowptr4/rowptr8、rowaddr 长生命周期复用、non-GLC/no-wait、staged-half、empty-tile skip、unconditional B-load。

## 2026-06-13 normal K3 no-tail device-fence A/B 反证

- A/B：no-tail 路径 final fence 从 `__threadfence_system()` 降为 `__threadfence()`，tail/signal 路径保持 system-scope。
- Correctness：direct K3 1024/4096 no-tail 均通过，说明该单点没有立刻破坏当前 direct compare。
- Performance：
  - 1024 all_zero/local/staged_remote/staged `0.609/0.944/2.205/2.216 ms`；
  - 4096 all_zero/local/staged_remote/staged `2.172/3.304/7.775/7.820 ms`；
  - 相对 scalar-pack 基线没有实质收益，4096 staged 略退。
- 结论：
  - final fence scope 不是 normal K3 no-tail remote/scattered combine gap 主因；
  - 已撤回到 `__threadfence_system()`，后续不重复该方向；
  - 继续聚焦 store/window/wait 调度、rowptr/store data path 或独立 source-backed ASM-style store probe。

## 2026-06-13 normal K3 GLC rowptr A/B 结论

- 背景：
  - scalar rowptr-resource + pack dependency baseline 在 clean rebuild 后仍观察到 4096 首轮/偶发 `global_gt_atol>0`；
  - 为降低 rowptr load family 和 aicc scheduling 的 correctness 敏感性，将 compute active-check 与 epilogue row pointer fetch 都恢复为 `global_load_i64_glc_device(row_combine_ptrs + row)`。
- 验证：
  - aicc rebuild log：`hygon_tmp/sglang_debug/rebuild_v3_normal_k3_glc_rowptr_ab_20260613_083735.log`；
  - direct K3 1024 repeat1 `global_max_abs=0`、`global_gt_atol=0`；
  - direct K3 4096 repeat1-5 全部 `global_gt_atol=0`、无 nonfinite，其中 repeat2/3 `global_max_abs=0.002983/0.002838`；
  - 1024 rowptr split all_zero/local/staged_remote/staged `0.718/0.942/2.207/2.214 ms`；
  - 4096 rowptr split all_zero/local/staged_remote/staged `2.541/3.312/7.802/7.840 ms`。
- 结论：
  - GLC rowptr load 版本是当前更稳的 correctness baseline，但不是性能优化；
  - 相比 rowptr-resource + pack dependency baseline，4096 staged 基本持平略慢，all_zero/GEMM floor 明显退化；
  - K3 normal no-tail 主 gap 仍是 remote/scattered combine store 数据通路与 store/window/wait 调度，下一步需要用 save-temps / dccobjdump / PMC 支撑 ASM-style store schedule 或更小 probe；
  - 后续不要把 GLC rowptr A/B 记录成性能 retained 项，它只是当前继续优化的安全起点。

## 2026-06-13 normal K3 GLC rowptr4 wait 合并 A/B 反证

- 线索：
  - 原 K3COMBINE ASM 的 direct epilogue 用 `K3_LOAD_COMBINE_ADDR4` 一次取 4 个 rowptr，一个 `s_waitcnt vmcnt(0)` 后 `K3_STORE4` 成组写；
  - V3 GLC scalar store baseline 每个 scalar store helper 各自执行 rowptr load + wait，wait/window 明显更碎。
- A/B：
  - 不跨 hidden step 复用 row address；
  - 只在同一 `token_base+{0,4,8,12}` 短窗口合并 4 个 GLC rowptr load 和一个 wait；
  - store 保持 scalar BF16 global store，不改 GEMM/MMAC 主循环、fence 或 tail/signal 路径。
- 结果：
  - 1024/4096 首轮 direct compare 通过；
  - rowptr split 有性能信号：1024 staged 从约 `2.214 ms` 到 `1.983 ms`，4096 all_zero/local 从约 `2.541/3.312 ms` 到 `2.006/2.737 ms`，4096 staged 仍约 `7.825 ms`；
  - 4096 repeat2 出现 `global_v3_nonfinite=256`，非有限在 inactive dense slot，active slot 未出现 nonfinite。
- 结论：
  - rowptr4/wait grouping 的性能信号真实，但当前 C/aicc codegen 下仍不稳定，和此前 rowptr4/rowptr8 resource 方向属于同类风险；
  - 已撤回到 GLC scalar store baseline；
  - 后续不重复 rowptr load grouping，除非先有独立 source-backed probe 或 ISA 证据解释 inactive dense nonfinite；优化应转向 ASM store schedule 的更小、更可验证形态或 PMC/SQTT 先定位。

## 2026-06-13 normal K3 GLC scalar baseline corrected PMC

- 工具结论：
  - `hipprof --pmc-type 3` 已经导出 CSV；当前 DTK `hipprof` 不支持额外 `--csv` 参数。
  - 可用命令形态：`hipprof --pmc-read --pmc-type 3 --kernel-name V3_K3_Fused -o <out> <app>`，write focus 同理。
- corrected PMC 目录：
  - `hygon_tmp/sglang_debug/pmc_k3_glc_scalar_4096_staged_remote_retry_20260613_093725`
  - read/write 均生成 CSV，且都捕获到 16 条 V3 K3 normal no-tail kernel 记录。
- 聚合值：
  - `pmc_read.csv`: median duration `7.696 ms`，`SQ_INSTS_VMEM_RD≈4.595M`，`SQ_INSTS_VMEM_WR≈1.482M`，`TA_BUSY≈185.3M`，`TCP_TA_DATA_STALL≈84.1M`。
  - `pmc_write.csv`: median duration `7.731 ms`，`SQ_INSTS_VMEM_RD≈4.598M`，`SQ_INSTS_VMEM_WR≈1.482M`，`TA_BUSY≈185.5M`，`TCP_TA_DATA_STALL≈84.6M`。
- 解释：
  - 与早期 ASM 对照中的 V3 `VMEM_WR` 偏高、`TCP_TA_DATA_STALL` 远高于 ASM 的结论一致；
  - read/write focus 都显示同一个数据通路长尾，后续优化不应再重复 rowptr load family / final fence / rowptr grouping，而应继续对照原 ASM 的 store family、store window、wait placement 和生成 ISA。

## 2026-06-13 normal K3 staged-half LDS mapping probe

- 背景：
  - 早前 H0/H1 gated staged-half 主路径在 1024 direct compare 通过、4096 direct compare 失败，表现为少量 active/dense drift 与 inactive dense nonfinite；
  - 需要先确认 `stage_acc_fragment_scaled_unmasked_device` / `K1_STORE_STAGED_HALF` 的 LDS row/hidden 基础映射是否等价于原 ASM `K3_STAGE_TILE_H0/H1` + `K3_STORE_STAGED_HALF`。
- Probe：
  - 新增 scratch-only `hygon_tmp/sglang_debug/k3_staged_half_mapping_probe.cu`，不进入生产 build；
  - 按当前 C staged 写法把可追踪 pattern 写入 LDS，再按 staged-half store 的 `vec_idx/row_half/hidden_vec/+0/+256/+512/+768` 规则读出并 `global_store_dwordx4` 到全局内存；
  - 首版 probe 使用 inline `ds_read_b128` 输出约束，出现 512 个 mismatch，坏点集中在每个 16B vector 的后半部分，判断为 probe asm output constraint 假阳性；
  - 改成生产等价的 `const uint4 st = lds_vec[...]` 后，远端 aicc/gfx938 运行 `staged_half_mapping_ok mismatches=0`，日志 `hygon_tmp/sglang_debug/k3_staged_half_mapping_probe_uint4_20260613_095155.log`。
- 结论：
  - 当前 C staged-half 的 LDS row/hidden 基础映射是正确的；
  - 早前 staged-half 主路径失败不应再归因于简单 LDS 行列错位，更可能来自 accumulator/value mapping、inactive row 写入语义、exec mask、rowptr/store window 或 aicc 调度敏感性；
  - 后续若继续 staged/vectorized store，必须先做 accumulator/value mapping 或 ASM-style store helper probe，而不是重复 H1 offset、rowptr load family 或 rowptr4 grouping。

## 2026-06-13 normal K3 staged-half value mapping probe

- Probe：
  - 新增 scratch-only `hygon_tmp/sglang_debug/k3_staged_half_value_probe.cu`；
  - 同一个 kernel 内用 fake accumulator 同时走 direct scalar store 与 staged-half `uint4` vector store，覆盖 H0/H1 两段、256 rows、256 hidden；
  - 远端 aicc/gfx938 编译运行通过：`staged_half_value_ok mismatches=0`，日志 `hygon_tmp/sglang_debug/k3_staged_half_value_probe_20260613_095603.log`。
- 结论：
  - `stage_acc_fragment_scaled_unmasked_device` 的 BF16 pack/value path 与 direct scalar store bitwise 对齐；
  - 旧 staged-half 主路径 4096 失败不应继续归因于 value packing 或 LDS vector-store基础映射；
  - 下一步可以做一次最小生产 staged-vector-store A/B；若仍失败，重点检查 rowptr load/exec mask/inactive row 合同、combine buffer lifecycle 和 aicc store schedule。

## 2026-06-13 normal K3 staged-vector-store 生产 A/B 初轮与稳定性反证

- A/B 内容：
  - 在 `!kTailReduce && !kSignalOnly` 的 normal K3 N=256 epilogue 中，用 H0/H1 两段 staged vector store 替代 scalar rowptr store；
  - rowptr 继续使用当前 correctness 更稳的 `global_load_i64_glc_device`，不复用已反证的 rowptr resource / rowptr4 grouping；
  - tail-reduce/signal-only 路径保持原 scalar store，避免干扰暂缓的 normal tail。
- 初轮验证：
  - 远端此前已用 aicc 重编 V3 normal K1/K3，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1k3_k3_staged_glc_20260613_095820.log`；
  - direct K3 1024 no-tail 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_staged_vector_store_ab_20260613_100059.log`，status 0 且无 `global_gt_atol/global_v3_nonfinite` gate 失败；
  - direct K3 4096 no-tail 通过，log `hygon_tmp/sglang_debug/k3_v3_dist_4096_staged_vector_store_ab_20260613_100059.log`，status 0 且无 `global_gt_atol/global_v3_nonfinite` gate 失败。
- Rowptr split：
  - 1024 `rowptr_all_zero/local/staged_remote/staged = 0.599/0.769/1.018/1.025 ms`；
  - 4096 `rowptr_all_zero/local/staged_remote/staged = 2.124/2.725/3.509/3.562 ms`；
  - 相对 GLC scalar baseline 的 1024 staged `~2.214 ms`、4096 staged `~7.840 ms`，vector store 性能信号很强，接近原 ASM no-tail 4096 staged `~3.113 ms`。
- 稳定性反证：
  - 1024 stability rep1/rep2 通过，但 rep3 失败，log `hygon_tmp/sglang_debug/k3_v3_dist_1024_staged_vector_store_stability_r3_20260613_100428.log`，`global_max_abs=0.322265625`、`global_gt_atol=1079`、无 nonfinite；
  - 4096 stability 多轮失败，logs `k3_v3_dist_4096_staged_vector_store_stability_r1/r2/r3_20260613_100428.log`，`global_gt_atol` 分别为 214/1264/660，无 nonfinite；
  - 漂移发生在 active dense slot，不是旧 staged-half 的 inactive nonfinite 形态。
- 结论：
  - staged-vector-store 已证明是正确的性能方向，但当前生产形态不是 stable correctness；
  - 后续不应直接保留该 A/B，也不再重复无证据的 H1 offset/value/layout 方向；
  - 下一步只做最小修复：vector global store cache/visibility 属性、exec mask、aicc store schedule 或 store helper 形态；若修不稳，必须撤回到 GLC scalar baseline。

## 2026-06-13 normal K3 staged-vector-store 修复尝试与 scalar 稳定回退

- `global_store_dwordx4 off glc` A/B：
  - aicc 支持编译，但 1024 stability 第 2 轮失败，`global_max_abs=0.073974609375`、`global_gt_atol=202`；
  - 结论：给 x4 store 加 `glc` 不能修复 active dense drift。
- `global_store_dwordx2` 拆分 A/B：
  - 1024 三轮 direct correctness 全过；
  - 4096 第 1 轮失败，`global_max_abs=0.52099609375`、`global_gt_atol=1917`；
  - 结论：减小 vector store 粒度不足以稳定大 size remote combine。
- volatile LDS load A/B：
  - 1024 第 2 轮失败，`global_max_abs=0.210205078125`、`global_gt_atol=311`；
  - 结论：主因不是普通 `uint4` LDS load 被 aicc 重排这一单点。
- 回退与稳定修复：
  - 撤回 staged-vector-store 生产路径，恢复 GLC rowptr + scalar `global_store_short`；
  - clean rebuild 后 scalar baseline 曾出现偶发小漂移，因此把 pack dependency 从单个 `s_nop 0` 加强为 `sched_barrier + s_nop 0 + s_nop 0 + sched_barrier`；
  - direct K3 1024/4096 各 3 轮通过，logs `k3_v3_dist_1024_scalar_stronger_pack_dep_r{1,2,3}_20260613_102626.log`、`k3_v3_dist_4096_scalar_stronger_pack_dep_r{1,2,3}_20260613_102626.log`；
  - rowptr split 1024 `0.718/0.941/2.211/2.224 ms`，4096 `2.541/3.314/7.796/7.837 ms`，和 GLC scalar baseline 基本一致。
- 结论：
  - 当前生产路径回到稳定 scalar baseline；
  - staged vector store 证明性能方向很强，但 x4/glc/x2/volatile LDS 都不足以稳定，后续需要更接近原 ASM 的 store family/exec-mask/window 证据或独立 store probe，而不是继续在生产主路径里试无证据变体。

## 2026-06-13 normal K3 scalar baseline per-function ISA split

- 远端 clean code object：
  - `hygon_tmp/sglang_debug/codeobj_k3_v3_scalar_strong_dep_clean_20260613_103244/k3_v3_fused_ext.cpython-310-x86_64-linux-gnu.so-hipv4-amdgcn-amd-amdhsa--gfx938.out.s`；
  - `.so` mtime 对齐 scalar stronger pack-dependency rebuild，避免使用 stale root `*gfx938*.out`。
- V3 K3 fused 三个模板实例计数：
  - `Lb1ELb0ELb0` no-tail：`global_store_short=128`、`global_store_dwordx4=0`、`buffer_store_short=0`、`global_load_dwordx2=132`、`buffer_load_dwordx4=132`、`ds_read=56`、`s_barrier=21`、`s_waitcnt=217`、`v_mmac=128`；
  - `Lb1ELb0ELb1` signal-only：`global_store_short=128`、`s_waitcnt=229`；
  - `Lb1ELb1ELb0` tail-reduce：`global_store_short=128`、`global_store_dwordx4=2`、`s_waitcnt=241`。
- 原 K3COMBINE ASM 源码线索：
  - direct scalar macro `K3_STORE4` 仍是 `global_store_short`，但 no-tail scatter 主体 `K3_SCATTER_C_TILE_TO_COMBINE` 先执行 `s_waitcnt vmcnt(0)`、`buffer_wbinvl1_vol`、`s_barrier`；
  - staged/vector scatter 循环用 `global_load_dwordx2` 取 rowptr、`ds_read_b128` 取 staged half，然后同时 `s_waitcnt vmcnt(0)` 和 `s_waitcnt lgkmcnt(0)` 后按 rowptr 非零 exec mask 发 `global_store_dwordx4`。
- 解释：
  - 之前 staged-vector-store A/B 的强性能信号和 active dense drift 更像缺少 ASM-style scatter 前置 cache/barrier 或 wait/window 语义，而不是 LDS row/value mapping；
  - 下一步只做最小 A/B：在 no-tail N=256 staged-vector-store 前补 `wait_vmem_lds_store_device()` + `invalidate_l1_device()` + `block_barrier_device()`，并保持 direct K3 1024/4096 多轮 stability gate；如果仍失败，回到 scalar baseline并转向更小的 store/window probe。

## 2026-06-13 normal K3 ASM-style staged-vector-store A/B 反证与 scalar split-pair 稳定修复

- ASM-style staged-vector-store A/B：
  - 在 no-tail N=256 epilogue 先将 256 rows stage 到 LDS；
  - 补 `wait_vmem_lds_store_device()`、`buffer_wbinvl1_vol`、`block_barrier_device()` 后再执行 H0/H1 staged vector store；
  - direct K3 1024 首轮即系统性错误，`global_max_abs=2.1171875`、`global_gt_atol=71541595`；
  - direct K3 4096 三轮均系统性错误，`global_gt_atol≈257M`，无 nonfinite。
- 结论：
  - 单纯把原 ASM scatter 前置 cache/barrier 语义搬到 C staged-vector-store 生产路径并不能修复 active dense drift，反而暴露出 row/exec/store schedule 语义错位；
  - 已撤回 no-tail epilogue 到 scalar `K1_STORE_ROWS_256(K1_STORE_ROW_UNMASKED)`，后续不再直接推进生产 staged-vector-store 主路径，除非先有更小 store/window probe 或 ISA 证据。
- Scalar clean rebuild 稳定性：
  - 撤回 staged-vector-store 后，clean rebuild 的 scalar path 在 1024 direct 通过，但 4096 两轮出现小漂移，`global_gt_atol=2/3`；
  - 仅增加每次 store 前 value dependency 仍不够，4096 三轮里仍有一次 `global_gt_atol=1`；
  - 将 scalar store 改成 pair0 pack+store、pair1 pack+store，并在每个 scalar rowptr store 前保留 value dependency 后，direct K3 1024/4096 各 3 轮全部 `global_gt_atol=0`、`global_v3_nonfinite=0`、`global_max_abs=0`。
- 当前状态：
  - split-pair scalar store 是当前 correctness-stable baseline；
  - 这是稳定性修复，不是性能优化；
  - rowptr split 已确认它没有明显额外性能代价：1024 all_zero/local/staged_remote/staged `0.727/0.949/2.216/2.222 ms`，4096 `2.575/3.341/7.818/7.849 ms`；
  - 主 gap 仍是 4096 remote/scattered combine store 数据通路，下一步继续做 store/window/ISA 证据驱动优化。

## 2026-06-13 normal K3 split-pair scalar ISA/resource 取证

- 修正 code-object 抽取流程：
  - `dccobjdump --extract-elf=all` 需要在目标输出目录内运行，让工具把 `*gfx938.out` 写到当前目录；
  - 旧的 `--output=<dir>` 形态只打印 host ELF 信息，未产出 gfx938 ELF。
- 当前 split-pair scalar build：
  - header sha `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`；
  - code object 目录 `hygon_tmp/sglang_debug/codeobj_k3_v3_split_pair_store_current_20260613_110816/`；
  - no-tail 实例 `Lb1ELb0ELb0` 计数：`global_store_short=128`、`global_store_dwordx4=0`、`global_store_dwordx2=0`、`buffer_store_short=0`、`global_load_dwordx2=132`、`buffer_load_dwordx4=132`、`ds_read=56`、`s_barrier=21`、`s_waitcnt=220`、`s_nop=273`、`v_mmac=128`。
- 结论：
  - split-pair 稳定性修复主要增加调度保护和 value dependency，没有改变 remote combine 的 scalar store family；
  - 当前性能瓶颈仍然不是 GEMM 主循环，而是 no-tail epilogue 的 rowptr/scattered remote store 数据通路；
  - 原 K3COMBINE 主路径的 staged/vector scatter 仍是性能方向，但此前 H1 offset、H0/H1 gating、value/layout probe 与生产 staged-vector-store stability 已说明不能直接重放主路径；下一步应先构造更小的 production-like staged-store probe，隔离 wave gating、exec mask、rowptr offset 和同-kernel 调度干扰。

## 2026-06-13 normal K3 staged-half rowptr/exec-mask scratch probe

- 新增 scratch-only probe：`hygon_tmp/sglang_debug/k3_staged_half_rowptr_probe.cu`。
- Probe 覆盖：
  - H0 使用 wave0-3 stage/store、row offset 0；
  - H1 使用 wave4-7 stage/store、row offset 128；
  - staged store 使用 rowptr raw-buffer dwordx2 load、一个 wait 后 `global_store_dwordx4`；
  - rowptr 模式覆盖 all-active、部分 zero、permuted physical row；
  - 20 repeats * 3 modes。
- 远端验证：
  - aicc/gfx938 编译运行通过；
  - log `hygon_tmp/sglang_debug/k3_staged_half_rowptr_probe_20260613_*.log`；
  - 输出 `staged_half_rowptr_ok total_mismatches=0`。
- 结论：
  - staged/vector store 的基础 rowptr offset、zero-row exec mask 和 H0/H1 wave gating 在独立环境里可稳定 bitwise 对齐 direct rowptr store；
- 生产 K3 staged-vector-store 的 active dense drift 更可能来自 full GEMM 后的 BF16 pack-to-LDS codegen/调度耦合、寄存器压力或同-kernel store schedule，而不是单独 rowptr/exec mask；
- 下一步优先尝试 staged path 的 pair-wise pack dependency，复用 scalar split-pair 稳定修复思路，而不是重复 rowptr4/H1 offset/gating 变体。

## 2026-06-13 normal K3 staged pack-to-LDS dependency A/B 反证

- A/B 内容：
  - 在生产 K3 normal no-tail staged/vector-store 路径中，把 `stage_acc_fragment_scaled_unmasked_device` 改成 pair0 pack-to-LDS + dependency，再 pair1 pack-to-LDS + dependency；
  - 保持 H0/H1 wave gating、row offset 0/128、rowptr raw-buffer load 和 `global_store_dwordx4`，意图复用 split-pair scalar store 的 codegen 稳定思路。
- 结果：
  - 远端 aicc rebuild 成功，失败 A/B header sha 为 `5bd6762d9ab7fb657996e782d9c3dc9394dab8c77f6f9ff371c93edaf263a130`；
  - direct K3 1024 三轮均失败，`global_max_abs≈2.217/2.899/2.369`，`global_gt_atol≈1121006/1129243/1123331`，无 nonfinite；
  - direct K3 4096 三轮均失败，`global_max_abs≈3.872/2.969/3.168`，`global_gt_atol≈7177439/7204522/7127298`，无 nonfinite；
  - 漂移是 active dense 大面积错误，不是旧 inactive nonfinite 形态。
- 恢复：
  - 已撤回到 split-pair scalar store baseline，远端 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_revert_staged_pack_dep_ab_20260613_112256.log`；
  - 恢复后 direct K3 1024 三轮全过，`global_max_abs=0`、`global_gt_atol=0`、无 nonfinite；
  - 恢复后 direct K3 4096 三轮全过，`global_gt_atol=0`、无 nonfinite，`global_max_abs=0.003349/0/0`，仍在 correctness gate 内。
- 结论：
  - 独立 rowptr/exec probe 与 value/mapping probe 只能证明基础语义成立；full GEMM 下 staged-vector-store 仍存在 register schedule、LDS/store window 或 aicc codegen 耦合问题；
  - 不再直接重复生产 staged-vector-store 主路径或 pack-to-LDS dependency 变体；
  - 下一步应先做更深 ISA/PMC/SQTT 或 full-GEMM-linked 小 probe，把 full GEMM 寄存器压力和 store schedule 纳入实验条件后再决定能否重新打开 vector store 路线。

## 2026-06-13 normal K3 store4 regpressure scratch probe

- Probe：
  - 新增 scratch-only `hygon_tmp/sglang_debug/k3_store4_regpressure_probe.cu`；
  - 同一 kernel 内对比当前 split-pair scalar store 与 ASM-style 4 个 `global_load_dwordx2 ... glc` + 单个 `s_waitcnt vmcnt(0)` + scalar `global_store_short`；
  - probe 人为加入多组 float 计算和 64 次 repeat，目的是把 rowptr/store 短窗口放到比简单 unit probe 更接近 full-GEMM 的寄存器压力下，但仍不进入生产路径。
- 远端验证：
  - aicc/gfx938 编译运行通过，log `hygon_tmp/sglang_debug/k3_store4_regpressure_probe_20260613_113554.log`；
  - 输出 `store4_regpressure_ok rows=256 hidden=256 repeat=64`；
  - code object 目录 `hygon_tmp/sglang_debug/codeobj_k3_store4_regpressure_probe_20260613_113614/`；
  - ISA 计数 `global_load_dwordx2=32`、`global_store_short=32`、`s_waitcnt=22`，并捕获到四个 GLC rowptr load 后单个 `s_waitcnt vmcnt(0)` 的 grouped window。
- 结论：
  - 短窗口 4-rowptr grouped GLC load + scalar store 本身在 scratch/regpressure 条件下能 bitwise 对齐 split-pair scalar baseline；
  - 该结果不能推翻此前生产 rowptr4/rowptr8/GLC rowptr4 wait grouping 的失败记录，也不支持直接恢复 staged-vector-store 主路径；
  - 下一步可做一个最小生产 A/B：只在 no-tail scalar epilogue 的同一 `token_base+{0,4,8,12}` 短窗口合并 rowptr load/wait，保持 split-pair value dependency、scalar `global_store_short`、不跨 hidden step 复用 row address，并用 direct K3 1024/4096 多轮 correctness 决定是否继续。

## 2026-06-13 normal K3 grouped GLC rowptr production A/B 反证

- A/B 1：4-rowptr grouped GLC load + single `s_waitcnt vmcnt(0)` + scalar store。
  - 本地 header sha `b141de58e739c10e4a9dc05ddf1964047dc79776c0b6a886cca678eb9b83568d`；
  - 远端 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_glc_store4_ab_20260613_113935.log`；
  - direct K3 1024 三轮均 `global_gt_atol=0`；
  - direct K3 4096 repeat3 出现 `global_gt_atol=2`、`global_dense_gt_atol=2`，max coord 不在 active slot，但属于 dense drift，不能保留。
- A/B 2：2-rowptr grouped GLC load，保持 split-pair pack/store 顺序。
  - 本地 header sha `5b70ac6b78dbbf791dfbb09b4011b9f22190e54f5b5717b7d4a7d0da8a68d7e4`；
  - 远端 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_glc_store2_ab_20260613_114640.log`；
  - direct K3 4096 三轮均 `global_gt_atol=0`；
  - direct K3 1024 repeat2 出现 `global_gt_atol=1` 且 `global_max_coord_active=True`，不能保留。
- 恢复：
  - 已撤回到 split-pair scalar baseline，源码 sha 回到 `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`；
  - rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_restore_after_store2_ab_20260613_115209.log`；
  - 4096 restore smoke 仍出现一次 active 小漂移，后续 repeat 发现 dense inactive nonfinite artifact，说明 baseline 自身仍有 4096 稳定性尾巴，不应再在无新证据下重复 rowptr grouping。
- 结论：
  - scratch store4 regpressure probe 只能证明短窗口本身可编译可运行，不能外推到 full GEMM K3 epilogue；
  - grouped GLC rowptr 生产形态已经 4-rowptr 与 2-rowptr 双反证，后续不再重复该方向；
  - 下一步需要当前 baseline 的 ISA/PMC/SQTT 或更贴 full-GEMM 的小 probe 来解释 store schedule / register pressure / output lifecycle，而不是继续调 rowptr load grouping。

## 2026-06-13 normal K3 stronger store helper dependency A/B 反证

- A/B：
  - 仅在 `store_bf16_rowptr_value_dep_device` 里增加 `sched_barrier + s_nop/s_nop + sched_barrier`，不改 GEMM 主循环、rowptr load family 或 scalar store family；
  - 本地 header sha `f99183d6dd9fdf7ca6af27b6a3af15075678665a2f83b6a2813dfddfe75a1a1a`；
  - 远端 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_stronger_store_helper_dep_20260613_115723.log`。
- 结果：
  - direct K3 1024 三轮均 `global_gt_atol=0`；
  - direct K3 4096 repeat2 出现 `global_gt_atol=60` 且 `global_max_coord_active=True`，不能保留。
- 恢复：
  - 已撤回到 split-pair scalar baseline，源码 sha 回到 `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`；
  - 远端 rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_restore_after_helper_dep_ab_20260613_120134.log`。
- 结论：
  - 简单加强 store helper dependency / barrier 不能解决 4096 active drift，也不会触及当前 PMC 指向的 remote/scattered store 数据通路；
  - 后续不要继续叠 barrier 或 `s_nop`，除非 ISA/SQTT 明确显示具体 hazard。

## 2026-06-13 normal K3 restored split-pair baseline 4096 active drift

- 远端状态：
  - header sha `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`；
  - 生产文件中无 grouped GLC rowptr helper / stronger-helper A/B 残留；
  - 容器内无残留 build/test/profiler 进程。
- Restore smoke：
  - direct K3 1024 no-tail：`global_gt_atol=0`、`global_v3_nonfinite=0`；
  - direct K3 4096 no-tail：`global_gt_atol=42`、`global_dense_gt_atol=42`、`global_max_coord_active=True`、`global_v3_nonfinite=0`。
- 解释：
  - 这不是 inactive dense nonfinite artifact，而是 active dense drift；
  - 因为同一 sha 曾在 `revert_staged_pack_dep` 后 1024/4096 各 3 轮通过，当前需要先排除 stale object、aicc codegen variance、test seed/launch ordering、K1/K2 producer差异或 hidden source diff；
  - 在该问题未稳定前，normal K3 no-tail 性能数据只能作为诊断，不可作为 retained 优化依据。

## 2026-06-13 normal K3 reader-side acquire 诊断

- KB 线索：
  - DeepEP/Hygon barrier 参考在 signal 前使用 `memory_fence()`，其实现是 `__threadfence_system()`，并在 barrier 中配合 block/rank 同步；
  - Flux reduce-scatter 参考有 `SystemBarrier::wait_eq` 与 `fence.acq_rel.sys` / `__threadfence_system()` 的 reader/writer 可见性组合；
  - 这些证据支持把 4096 间歇 drift 先当作 remote combine reader-side cache/visibility 问题诊断，而不是继续改 GEMM 主循环。
- 诊断结果：
  - 无 acquire：4096 direct K3 三连中一轮 active/dense drift；
  - 只开 barrier acquire：4096 direct K3 仍出现 drift，且有一次 dense nonfinite artifact；
  - barrier acquire + reduce acquire：4096 direct K3 三连全 0，随后五连全 0。
- 解释：
  - 由于 direct compare 脚本先 clone combine 再 reduce，`reduce_acquire` 修复 combine diff 的现象仍可能受间歇性影响，需要继续用 large_opt stage/e2e 和更多复测确认；
  - 但该方向明显优于继续 rowptr grouping / helper barrier；
  - 若 large_opt no-tail 开启 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 后 correctness 稳定且性能代价可接受，应考虑把 V3 normal no-tail 的默认可见性策略改为 acquire-on 或收紧测试默认 env。

## 2026-06-13 normal K3 reduce-acquire default-on

- 代码策略：
  - `MEGAMOE_DCU_V3_REDUCE_ACQUIRE` 默认值从 `0` 改为 `1`；
  - 作用范围仍被 large_opt gate 限制在 `v3_backend == "normal" and not use_tail_reduce`；
  - 显式设置 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=0` 仍可关闭；
  - 不改 K3 kernel，不新增 runtime launch，只让 no-tail rank barrier wait 走 acquire，并让 `reduce_local_combine` 在读 combine 前做 invalidate/acquire。
- 验证：
  - 本地 `python -m compileall megamoe/large_opt.py tests/test_dcu_megamoe_v3.py hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 通过；
  - 本地 `python -m pytest` 不可用：`No module named pytest`；
  - 远端 `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed`；
  - large_opt no-tail 默认 acquire correctness：1024 三轮全过，4096 一轮通过，summary `hygon_tmp/sglang_debug/v3_normal_default_reduce_acquire_correctness_summary_20260613_122423.txt`；
  - 4096 长 bench 当前受远端 93% VRAM 和 baseline oracle OOM 限制，未完成长轮次性能测量。
- 性能线索：
  - 1024 stage timing 中 `after_no_tail_barrier` 约 `0.02-0.25 ms`、`after_reduce` 约 `0.045 ms`；
  - 4096 首轮 stage timing 中 K3 combine 仍约 `7.8-8.0 ms`，`after_reduce` 约 `0.156 ms`；
  - acquire 默认主要是 correctness/visibility 修复，尚不是 K3 remote-store 性能优化。

## 2026-06-13 normal K3 staged-vector-store reduce-acquire retest

- A/B：
  - 基于当前 sha `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`，只在 no-tail normal `kTileN==256` epilogue 中启用 `K1_STAGE_ROW_UNMASKED` + LDS barrier + `K1_STORE_STAGED_HALF(0/128)`；
  - 不加入 `buffer_wbinvl1_vol`，并保持 `MEGAMOE_DCU_V3_REDUCE_ACQUIRE=1` 的 direct compare 环境。
- 结果：
  - 远端 aicc rebuild 成功，失败 A/B header sha `db35cee549848eb5c1cc580edebf45753195ac5d6c209d0381ce9fe61ef94a7d`；
  - direct K3 1024 首轮失败，`global_max_abs=2.323974609375`、`global_gt_atol=71475443`、`global_dense_gt_atol=71475443`、`global_v3_nonfinite=0`；
  - 已撤回到 sha `0e3d2188ee53b6114e5c3b50b94ad648f995c5b535e4d036859eaed0a3b54200`，重编后 1024/4096 direct K3 acquire smoke 均 `global_gt_atol=0`。
- 结论：
  - reduce-acquire 能稳定 reader-side visibility，但不能修复 staged/vector-store 在 full GEMM 下的 value/store schedule 耦合；
  - 后续不再重复 staged/vector-store 生产主路径，除非先用 ISA/SQTT 或 full-GEMM-linked probe 定位具体 hazard。

## 2026-06-13 normal K3 rowptr-resource reconciliation

- 代码：
  - 当前源码未保留计划中记录的 rowptr-resource 形态，因此重新落地最小版本；
  - no-tail normal `kTileN==256` 的 active-row check 改用 `buffer_load_fp8_b128_active_row_buffer_device`；
  - no-tail normal epilogue row address fetch 改用 `store_acc_fragment_scaled_unmasked_buffer_device`，最终 store 仍是 scalar `global_store_short`，不启用 vector store。
- 正确性：
  - 远端 aicc rebuild log `hygon_tmp/sglang_debug/rebuild_v3_normal_k3_rowptr_resource_reconcile_20260613_124011.log`；
  - direct K3 acquire 1024/4096 均 `global_gt_atol=0`、`global_v3_nonfinite=0`，summary `hygon_tmp/sglang_debug/k3_v3_dist_rowptr_resource_reconcile_summary_20260613_124126.txt`。
- 性能：
  - rowptr split 1024 avg all_zero/local/staged_remote/staged `0.621/0.943/2.227/2.231 ms`；
  - rowptr split 4096 avg all_zero/local/staged_remote/staged `2.199/3.309/7.785/7.813 ms`；
  - PMC 4096 staged_remote `hygon_tmp/sglang_debug/pmc_k3_rowptr_resource_4096_staged_remote_20260613_124438` 显示单 kernel `SQ_INSTS_VMEM_RD≈4.57M`、`SQ_INSTS_VMEM_WR≈1.41-1.45M`，相比此前 GLC scalar `4.60M/1.48M` 仅小幅下降，TA/TCP stall 仍同量级。
- 结论：
  - rowptr-resource 可作为 floor 小收益保留；
  - 主瓶颈仍是 remote/scattered scalar combine store，不是 rowptr load family 本身。

## 2026-06-13 normal K3 rowptr-resource ISA attribution

- Code object：
  - `hygon_tmp/sglang_debug/codeobj_k3_v3_rowptr_resource_reconcile_20260613_124654/`。
- 当前 retained rowptr-resource no-tail 实例 `Lb1ELb0ELb0` ISA 计数：
  - `global_load_dwordx2=0`、`global_load_dwordx2_glc=0`；
  - `buffer_load_dwordx2=132`；
  - `global_store_short=128`、`global_store_dwordx4=0`、`buffer_store_short=0`；
  - `s_waitcnt=350`、`v_mmac=128`。
- 解释：
  - rowptr active-check 和 epilogue row address fetch 的 load family 已经从 global/glc 切成 buffer/resource；
  - 但最终 combine 写回仍是 scalar `global_store_short`，没有变成原 K3COMBINE ASM 的 staged/vector-store family；
  - `s_waitcnt` 比此前 split-pair GLC scalar baseline 的约 `220` 更高，因此 rowptr-resource 只能解释 all-zero/floor 小幅下降，不能解决 4096 staged_remote 约 `7.8 ms` 的主耗时。
- 下一步：
  - 继续逐段比较原 `K3COMBINE.s` 的 `K3_STORE_C4` / `K3_STORE_STAGED_HALF` / scatter loop 与当前 C/aicc 生成 ISA，找出原 ASM staged/vector store 稳定而 C full-GEMM staged/vector-store 漂移的差异点；
  - 不再把 rowptr load family 作为主要优化方向，除非后续 PMC/SQTT 显示 rowptr load 再次成为瓶颈。

## 2026-06-13 normal K3 inline staged-store-window A/B and retained perf snapshot

- Inline staged-store-window A/B：
  - 基于原 `K3_STORE_STAGED_HALF` 顺序，尝试把 no-tail N=256 staged store 窗口固定为 rowptr buffer load、LDS vector read、显式 `s_waitcnt vmcnt/lgkmcnt`、`global_store_dwordx4`；
  - 失败 A/B header sha 为 `929d968a5bc83e09390491ad6d94e1dc8a1382d5ff7f76aafb72b6ab88b27ae9`；
  - direct K3 acquire 1024 失败：`global_max_abs=0.669921875`、`global_gt_atol=23379`、无 nonfinite；
  - direct K3 acquire 4096 失败：`global_max_abs=2.580810546875`、`global_gt_atol=417576`、无 nonfinite；
  - 该错误是 active dense drift，不能保留；已撤回到 rowptr-resource + scalar `global_store_short` baseline，远端 header sha 回到 `98dca271315abc5473ad25e19bd1d7ee315711912979cce3b1fd8119fa718575`。
- 13:12 retained baseline performance snapshot：
  - 初次直接跑 rowptr modes 失败，原因是远端 K1 extension 被 K3-only rebuild 留成 stub，报 `V3 K1 raw kernels were not compiled into this extension`；
  - 已用 aicc 重新编译 K1/K3 normal raw，log `hygon_tmp/sglang_debug/rebuild_v3_normal_k1k3_perf_snapshot_20260613_131008.log`，并确认 `k1_symm_fused_l1_v3_pack5` 符号存在；
  - 实测 log `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_perf_snapshot_20260613_131205.log`；
  - JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_1024_20260613_131205.json`：rows=8192，active_avg=6144，local_avg=678.625，remote_avg=5465.375；all_zero/local_only/remote_only/staged median_avg_rank_ms 为 `0.616/1.091/2.215/2.223`；
  - JSON `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_4096_20260613_131228.json`：rows=29696，active_avg=24576，local_avg=2970.125，remote_avg=21605.875；all_zero/local_only/remote_only/staged median_avg_rank_ms 为 `2.195/3.231/7.797/7.831`。
- 结论：
  - 当前 retained rowptr-resource baseline 没有新的性能退步，但也没有突破；
  - 1024/4096 staged 仍约 `2.22/7.83 ms`，大幅慢于原 ASM rowptr split 4096 staged 约 `3.11 ms`，更远于 pure C groupgemm目标；
  - 差距仍集中在 remote/scattered scalar combine store，rowptr load family 的 floor 小优化已经基本吃完。

## 2026-06-13 normal ASM-pack5 路线可行性分析

- 用户提出替代思路：让原 K1/K3 fused ASM normal 路径直接支持 V3 5pack weight layout，使 normal 与 V3 LL layout 一致。
- DCU KB 检索结论：
  - DeepGEMM ASM 的 Marlin weight layout 是 binary contract；
  - 原 host packing 顺序是 `expert -> n_outer -> k_outer -> n_inner -> k_inner`，默认 `n_tile=16/k_tile=16`；
  - 修改 weight layout 不能只换 host 侧 tensor，必须同步修改 ASM/inline 地址数学，否则会 silent correctness break。
- 本地代码证据：
  - `v3_layout.py` 与 `hygon_tmp/K1_groupgemm_fp8/k1_gemm.cpp` 的 pack5 都是 `expert -> ko(k64) -> no(n256) -> ni16 -> ks(k16) -> physical_ni -> ki`，其中 `physical_ni=(logical_ni & 3) * 4 + (logical_ni >> 2)`；
  - K3 ASM host 合同中 `prob.a=l2_weight`、`prob.b=act_fp8`，因此 ASM 内 `SrdA`/`A` 是 L2 weight，`SrdB`/`B` 是 K2 activation；此前写成 “B/scale 地址数学” 不准确，ASM-pack5 第一版应 patch weight/SrdA 地址数学；
  - 当前 V3 C K1/K3 normal 已有 pack5 weight-load 地址公式，可作为 ASM pack5 patch 的直接参考；
  - 原 K3COMBINE ASM 已稳定实现 rowptr combine epilogue、`K3_STORE_C4`/`K3_STORE_STAGED_HALF`、exec mask 和 waitcnt/store schedule，而当前 C/aicc 路线的主要未解问题正是 full-GEMM 下 vector/staged store 稳定性。
- 初步判断：
  - K3 no-tail 是最值得优先尝试的 ASM-pack5 目标：只改 weight/SrdA 访存，尽量保留原 ASM combine/store 调度，有机会直接绕开 C/aicc scalar remote-store 瓶颈；
  - K1 也可行但次优先级，因为 K1 还耦合 dispatch-pull 与 metadata，且当前 C K1 已比 K3 更接近可接受区间；
  - 风险点是 pack5 是否只需改变 global B-load offset。如果原 ASM 的 LDS/register fragment 顺序对旧 layout 做了隐含 swizzle，可能还要改 LDS 写入或 vgprB 读出顺序，难度会从“地址 patch”升级到“ASM GEMM 主体重排”。
- 建议验证顺序：
  1. 新建隔离 K3 no-tail ASM-pack5 code object，不替换默认路径；
  2. 只 patch weight/SrdA 地址公式到 pack5，保留原 K3COMBINE epilogue；
  3. 先跑 direct K3 1024/4096 correctness，再跑 rowptr split；
  4. 如果 no-tail 成功，再复制同一 weight-load patch 到 K3 tail-reduce ASM；
  5. 最后再评估 K1 ASM-pack5。

## 2026-06-13 normal K3 no-tail ASM-pack5 build/correctness result

- 隔离实现已验证到 code-object 层：
  - `K3COMBINE_PACK5.s/.co` 独立存在，Python 侧由 `MEGAMOE_DCU_V3_K3_ASM_PACK5=1` 选择；
  - unset 时仍走当前 V3 C/aicc raw normal no-tail，原始 DCU MegaMoE ASM `.co` 与 wrapper gate 不受影响；
  - 远端 build 可生成 PACK5 code object，`dccobjdump` 确认 pack5 patch 进入 ISA。
- 当前 PACK5 patch 的主要内容：
  - `SrdA` base = V3 pack5 expert stride + hidden tile stride，其中 hidden tile stride 用 `wg0 * 0x4000`；
  - `GLOBAL_OFFSET_A` 按 pack5 `ko64/no256/ks16/physical_ni` 公式计算；
  - A 侧 K-stage increment 改为 `0x80000`，对应 `128 * (4096 * 4)`；
  - combine/store/waitcnt/rowptr epilogue 保持原 `K3COMBINE` ASM。
- Direct correctness 结果：
  - 1024 normal no-tail、acquire on、PACK5 gate on：带原 ASM `+0x10` prepad compensation 时大规模 dense mismatch，`global_gt_atol=87330440`，无 nonfinite；
  - 去掉 `+0x10` 后 mismatch 增加到 `global_gt_atol=116445685`，因此去 prepad 不是正确方向；
  - 已恢复 `+0x10`，后续不重复该 A/B。
- 解释：
  - 当前输出不是全零且无 nonfinite，rowptr/combine epilogue可以运行；问题更像 weight data 进入 MMA 的顺序不对；
  - 如果 pack5 global address 与 C helper 已一致，下一层需要检查原 ASM 的 `m0`/LDS 写入步进、`vgprLocalWriteAddrA`、A LDS read swizzle 是否隐含旧 Marlin layout；
  - 若需要改 LDS write/read 主体，K3 ASM-pack5 复杂度会从“地址 patch”升高到“weight fragment layout patch”，届时要和继续优化 C/aicc rowptr-resource baseline 的收益风险对比。

## 2026-06-13 normal K3 no-tail ASM-pack5 layout diagnostic

- 远端曾因 `.s/.co` 停在 no-prepad A/B 版本导致一次 `plain` fixture 诊断 VMFault；同步本地恢复版后重新 build，确认远端 `.s` sha 为 `6df3b9c6796a97157680d6f2c1c3d9fc3b4a165b62a44ada215863db4d69c88e`，`.co` sha 为 `15d827b470e2a2e9a920bf403abe076f3fc571ec8ee64662942f1d3768b8ffea`。
- 新增 scratch-only `K3_V3_DIST_V3_PACK5_LAYOUT=plain|transposed` 诊断开关，仅影响 `hygon_tmp/sglang_debug/k3_v3_distributed_combine_compare.py` 的 L2 test fixture，不改 runtime/bench 执行路径，也不新增权重处理 kernel。
- 恢复版 `transposed` fixture 复现原失败：
  - 1024 direct K3：`global_max_abs=0.11767578125`、`global_gt_atol=87330440`、`global_y_max_abs=0.113525390625`、无 nonfinite。
- `plain` fixture 结果：
  - 1024 direct K3：`global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`；
  - 4096 direct K3：`global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`。
- 解释：
  - `SrdA/GLOBAL_OFFSET_A/incrA`、producer `loader_linear`、`m0`/LDS write offset、local read 和原 ASM combine/store 调度本身可以服务 pack5；
  - 当前 transposed V3 pack5 失败来自 `physical_ni=(logical_ni&3)*4+(logical_ni>>2)` 与原 ASM store/accumulator lane 顺序不匹配；
  - C pack5 K3 在 store 前有 `shuffle_acc_lane_device(c*, lowlat_acc_source_lane)`，而原 ASM epilogue没有等价 lane shuffle；
  - 后续二选一：要么给 normal ASM-pack5 定义 test/offline 侧 plain 5pack layout contract，复用原 ASM store 调度快速验证性能；要么继续修改 ASM epilogue/accumulator lane shuffle，使其兼容现有 V3 transposed pack5 layout。

## 2026-06-13 normal K3 no-tail ASM-pack5 plain layout perf

- 远端 JSON：
  - `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_asm_pack5_plain_1024_20260613.json`
  - `hygon_tmp/sglang_debug/v3_normal_k3_rowptr_modes_asm_pack5_plain_4096_20260613.json`
- 运行条件：
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_K3_ASM_PACK5=1`
  - `backend=asm_pack5`、`v3_pack5_layout=plain`、8 卡、warmup=3、repeat=10、rounds=5。
- 1024 tokens per rank：
  - active/local/remote avg rows: `6144.0 / 678.625 / 5465.375`;
  - median_avg_rank_ms all_zero/local_only/remote_only/staged: `0.645 / 0.657 / 0.879 / 0.882`;
  - median_max_rank_ms all_zero/local_only/remote_only/staged: `0.821 / 0.827 / 0.982 / 1.010`.
- 4096 tokens per rank：
  - active/local/remote avg rows: `24576.0 / 2970.125 / 21605.875`;
  - median_avg_rank_ms all_zero/local_only/remote_only/staged: `2.011 / 2.074 / 3.305 / 3.690`;
  - median_max_rank_ms all_zero/local_only/remote_only/staged: `2.550 / 2.611 / 3.682 / 3.819`.
- 对照当前 C/aicc retained baseline：
  - 1024 staged `2.223 ms` -> ASM-pack5 plain `0.882 ms`;
  - 4096 staged `7.831 ms` -> ASM-pack5 plain `3.690 ms`;
  - 4096 staged 已接近此前原 ASM 约 `3.11 ms` 的量级，瓶颈已经从 C scalar remote-store 主路径转为 layout 合同与生产接入。
- 决策：
  - 先保留 isolated `K3COMBINE_PACK5.s/.co` gate，不影响原 K3 ASM；
  - 新增 normal-ASM 专用离线 plain-pack5 helper，用于 fixture/bench 准备；
  - 暂不优先改 ASM epilogue lane shuffle 兼容现有 transposed V3 layout，因为 plain layout 已 bitwise 且性能收益显著，改 epilogue风险更高。

## 2026-06-13 normal K3 no-tail ASM-pack5 production fixture gate

- 代码边界：
  - `v3_layout.py` 新增 `pack5_weight_asm_normal()` / `flatten_pack5_weight_asm_normal()`，只作为离线/test/fixture helper；
  - `tests/test_mega_moe_dcu.py` 只有在 `MEGAMOE_DCU_V3_K3_ASM_PACK5=1 && MEGAMOE_DCU_V3_BACKEND=normal && K3_USE_ASM_TAIL_REDUCE=0` 时使用 plain L2 layout；
  - tail-reduce 默认仍使用现有 V3 transposed pack5，避免把 no-tail ASM 实验 layout 泄漏到 C/raw tail。
- 远端 source 验证：
  - `python3 -m compileall ...` 通过；
  - `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过，`10 passed in 5.68s`。
- direct K3 no-tail helper gate：
  - env 固定 `MEGAMOE_DCU_V3_K3_ASM_PACK5=1 K3_USE_ASM_TAIL_REDUCE=0 K3_V3_DIST_V3_PACK5_LAYOUT=plain K3_V3_DIST_BARRIER_ACQUIRE=1 K3_V3_DIST_REDUCE_ACQUIRE=1`;
  - 1024：`global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`、`global_v3_nonfinite=0`;
  - 4096：`global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`、`global_v3_nonfinite=0`.
- large_opt no-tail correctness：
  - env 固定 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_K3_ASM_PACK5=1 K3_USE_ASM_TAIL_REDUCE=0 MEGAMOE_DCU_TEST_CLONE_FUSED_BEFORE_BASELINE=1`;
  - 1024 correctness-only：`max_abs=0.000488281`、`mean_abs=9.50943e-06`;
  - 4096 correctness-only：`max_abs=0.000488281`、`mean_abs=9.45465e-06`.
- large_opt stage timing：
  - 1024 K3 combine per-rank `1.900-2.393 ms`;
  - 4096 K3 combine per-rank `3.907-4.387 ms`;
  - rowptr split microbench仍更低，说明 full staged path 里还有 launch/cache/producer-state 或 row distribution 差异需要继续用同一条件对照原 ASM。

## 2026-06-13 normal K3 no-tail ASM-pack5 base-precompute optimization

- 目标：
  - 不改原 `K3COMBINE.s` 路径；
  - 只在隔离 `K3COMBINE_PACK5.s` 内降低 V3 pack5 weight A-offset 的重复地址算术；
  - tail-reduce ASM 不参与本轮。
- 静态证据：
  - 新增 scratch 诊断 `hygon_tmp/sglang_debug/inspect_k3_codeobj_counts.sh`，修正 `dccobjdump` 用法为 `--inputs=<co> --extract-elf=all` 并计数 `.ISA`；
  - 优化前 PACK5 与原 ASM 的 `v_mmac=640`、`buffer_load_dwordx4=28`、`buffer_load_ubyte=128`、`buffer_load_dwordx2=8`、`global_store_dwordx4=8`、`global_store_short=128`、`s_waitcnt=149`、`s_barrier=31` 完全一致；
  - 优化前 PACK5 相比原 ASM 的额外向量地址算术为 `v_lshl +24`、`v_lshr +8`、`v_and +16`、`v_add +16`；
  - 将所有 8 个 `GLOBAL_OFFSET_A(..., vgprOffsetL=8, ...)` 共享的 `ko64/ks16/physical_ni` 预计算到 `vgprPack5OffsetBaseA` 后，PACK5 额外向量地址算术收敛到约 `v_lshl +10`、`v_lshr +1`、`v_and +2`、`v_add +2`；load/store/waitcnt/barrier 仍与原 ASM 一致。
- 正确性：
  - 新 PACK5 `.co` sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`；
  - direct K3 no-tail 1024/4096，env 固定 `MEGAMOE_DCU_V3_K3_ASM_PACK5=1 K3_USE_ASM_TAIL_REDUCE=0 K3_V3_DIST_V3_PACK5_LAYOUT=plain K3_V3_DIST_BARRIER_ACQUIRE=1 K3_V3_DIST_REDUCE_ACQUIRE=1`；
  - 两档均 `global_max_abs=0`、`global_gt_atol=0`、`global_y_max_abs=0`、`global_v3_nonfinite=0`。
- rowptr split 性能：
  - 1024 active/local/remote avg rows `6144.0/678.625/5465.375`，all_zero/local/remote/staged median_avg_rank_ms `0.477/0.483/0.818/0.834`；
  - 4096 active/local/remote avg rows `24576.0/2970.125/21605.875`，all_zero/local/remote/staged median_avg_rank_ms `1.506/1.541/3.088/3.074`；
  - 相比上一版 PACK5 plain，staged 从 `0.882/3.690 ms` 降到 `0.834/3.074 ms`；
  - 4096 rowptr staged 已略快于原 ASM 同条件 `3.172 ms`，说明主要 PACK5 floor/gemm 地址开销已消掉。
- large_opt stage timing：
  - 1024 一轮 correctness 通过，K3 combine per-rank `1.988-2.182 ms`；
  - 4096 三轮 correctness 均通过，`max_abs=0.000488281`、`mean_abs=9.45465e-06`；
  - 4096 首轮 K3 combine `4.28-4.56 ms` 偏冷，第 2/3 轮回落到约 `3.31-3.46 ms`；
  - 若继续优化最后 `~0.1-0.3 ms`，需要进一步确认是 launch/cache/producer-state、row distribution、还是 stage timing 入口自身的 warmup 差异，不能再用 C/aicc scalar-store 方向解释。

## 2026-06-13 normal K3 no-tail ASM-pack5 warmed residual复测

- 远端环境：
  - `hg@10.17.176.11` / container `sglang_megamoe` / `/workspace/DeepGEMM`；
  - 8 卡 `hy-smi` 均为 VRAM/HCU `0%`，无残留 `torchrun/build_ext/hipprof/aicc` 等进程；
  - `K3COMBINE_PACK5.co` sha 仍为 `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`。
- 4096 large_opt stage timing 复测：
  - env 固定 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal MEGAMOE_DCU_V3_K3_ASM_PACK5=1 K3_USE_ASM_TAIL_REDUCE=0`；
  - correctness 3/3 通过，`max_abs=0.000488281`；
  - 第 1 轮 K3 combine 仍偏冷，约 `3.82-4.44 ms`；
  - 第 2 轮 K3 combine 约 `3.10-3.27 ms`，第 3 轮约 `3.17-3.39 ms`。
- rowptr split 同环境复测：
  - 1024 staged median_avg_rank_ms `0.810`，staged_remote `0.791`；
  - 4096 staged median_avg_rank_ms `3.043`，staged_remote `3.070`；
  - 4096 all_zero/local/staged_remote/staged 为 `1.501/1.540/3.070/3.043 ms`。
- 结论：
  - K3 no-tail ASM-pack5 本体已基本逼近/略优于原 ASM rowptr split 档位；
  - full staged 剩余 `~0.06-0.34 ms` 主要表现为 warmup 和 rank spread，优先记录为编排/producer-state residual；
  - 后续 normal 性能大头转向 K1 normal，K3 no-tail 不再回到 C/aicc scalar-store 主线；tail-reduce ASM 暂缓。

## 2026-06-13 normal K1 ASM-pack5 isolated result

- 实现边界：
  - 新增隔离 `K1_fused/DeepGemm_W8A8_F8_MARLIN_PERCHANNEL_ASM_TN_MT256X256X128_BF16_MEGAMOE_DISPATCH_PULL_L1_PACK5.s/.co`；
  - 原 K1 dispatch-pull ASM 文件不改；
  - Python gate 为 `MEGAMOE_DCU_V3_K1_ASM_PACK5=1`，仅 V3 normal 生效；
  - Host 侧复用原 `ext.k1_symm_fused_l1`，通过 zero-copy view 让 flat plain-pack5 L1 tensor 通过原 Marlin 三维 shape check，data pointer 不变；
  - 只 patch A/L1 weight 侧 `GLOBAL_OFFSET_A`、pack5 base precompute、`SrdA` base 和 A 侧 K-stage increment，保留 B/staged activation、dispatch-pull、route metadata、row_combine_ptrs、output_index、stats 和 wait/signal 调度。
- build/source:
  - 远端 source pytest `tests/test_dcu_megamoe_v3.py` 通过：`11 passed`；
  - PACK5 `.co` sha `b26ac17779458296d4826664df04eef5109d5a7dabec3132c42a57cd959bd9e5`；
  - 首次 correctness 命令误设 `--intermediate-hidden 4096`，被 large_opt shape check 拦住；修正为 `2048` 后继续。
- correctness:
  - V3 K1 ASM-pack5 + K3 ASM-pack5 no-tail 1024：`max_abs=0.000488281`、`mean_abs=9.48142e-06`；
  - V3 K1 ASM-pack5 + K3 ASM-pack5 no-tail 4096：`max_abs=0.000488281`、`mean_abs=9.39676e-06`；
  - stage timing 1024/4096 三轮 correctness 均 3/3 通过。
- stage timing:
  - 1024 warmed 第三轮 K1 ASM-pack5 `~0.956-1.013 ms`，K3 ASM-pack5 `~0.873-1.043 ms`，total `~2.21-2.34 ms`；
  - 同条件 K1 C/aicc + K3 ASM-pack5 第三轮 K1 `~1.36-1.43 ms`，K3 `~0.81-0.88 ms`，total `~2.48-2.75 ms`；
  - 4096 warmed 第三轮 K1 ASM-pack5 `~3.175-3.393 ms`，K3 ASM-pack5 `~2.447-2.710 ms`，total `~6.48-6.66 ms`；
  - 同条件 K1 C/aicc + K3 ASM-pack5 第三轮 K1 `~4.298-4.444 ms`，K3 `~3.016-3.252 ms`，total `~8.06-8.27 ms`；
  - 原始 staged ASM no-tail 第三轮 1024 total `~2.42-2.89 ms`，4096 total `~6.51-6.68 ms`。
- formal bench:
  - V3 K1/K3 ASM-pack5 no-tail 1024 fused median `2.1127 ms`，原 staged ASM no-tail 1024 `2.3842 ms`；
  - V3 K1/K3 ASM-pack5 no-tail 4096 fused median `6.3658 ms`，原 staged ASM no-tail 4096 `6.6098 ms`；
  - 对 DeepEP/DeepGEMM baseline speedup 分别为 `1.7565x` 和 `1.4990x`；原 staged ASM 为 `1.5437x` 和 `1.4381x`。
- ISA/resource attribution:
  - 原 K1 `.co` sha `3c48f568840a29efbdb02aaaf3528e9ec28e151897a422e7839b88c242db82f0`；
  - K1 PACK5 `.co` 与原 K1 ASM 均为 `v_mmac=640`、`buffer_load_dwordx4=32`、`buffer_load_ubyte=128`、`buffer_load_dwordx2=2`、`buffer_store_short=640`、`s_waitcnt=185`、`s_barrier=33`；
  - PACK5 仅改变地址算术计数：`s_lshl 46->36`、`s_add 130->124`、`s_mul 114->107`、`v_lshl 509->519`、`v_and 93->95`、`v_add 2428->2430`、`v_mul 552->544`。
- 结论：
  - K1 ASM-pack5 路线已正确且性能优于当前 K1 C/aicc，同时 K1/K3 ASM-pack5 pair 的 1024/4096 no-tail bench 已超过原始 staged ASM；
  - 下一步整理 V3 normal no-tail 的 gate/default 策略；tail-reduce ASM 仍暂缓，不用该结果宣称 tail-reduce 完成。

## 2026-06-13 normal V3 no-tail ASM-pack5 default promotion

- 代码边界：
  - `large_opt.py` 只在 `USE_MEGAMOE_V3=1`、`MEGAMOE_DCU_V3_BACKEND=normal`、`K3_USE_ASM_TAIL_REDUCE=0` 且未启用 `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL` 时默认传入 `use_asm_pack5=True`；
  - K1/K3 wrapper 都支持 caller default + env 覆盖，`MEGAMOE_DCU_V3_K1_ASM_PACK5=0` / `MEGAMOE_DCU_V3_K3_ASM_PACK5=0` 可回退 C/aicc fallback；
  - K3 ASM-pack5 仍只在 normal no-tail 且无 `sym_buffer`/tail signal 参数时进入，tail-reduce 不使用 plain-pack5 默认 layout；
  - 原始 non-V3 K1/K3 ASM code object 和 wrapper gate 不变。
- 本地/远端 source 验证：
  - 本地 `compileall` 与 `git diff --check` 通过；本机 pytest 缺失仍不可用；
  - 远端容器 `sglang_megamoe` 内 `compileall` 通过，`PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` 通过：`11 passed`。
- 集成 bug 与修复：
  - 首次远端 default correctness 失败于 Python 层 `TypeError: k3_l2_fused_v3_to_combine() got an unexpected keyword argument 'use_asm_pack5'`；
  - 根因是 `use_asm_pack5` 错加到原 ASM wrapper 签名，未加到 V3 wrapper；已移回 V3 wrapper，source pytest 重跑通过。
- Default no-tail correctness/stage timing：
  - 命令显式 `env -u MEGAMOE_DCU_V3_K1_ASM_PACK5 -u MEGAMOE_DCU_V3_K3_ASM_PACK5`，固定 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1 USE_MEGAMOE_V3=1 MEGAMOE_DCU_V3_BACKEND=normal K3_USE_ASM_TAIL_REDUCE=0`；
  - 1024 三轮 correctness 均通过，`max_abs=0.000488281`、`mean_abs=9.48142e-06`，第三轮 warmed K1 `0.961-1.007 ms`、K3 combine `0.743-0.857 ms`；
  - 4096 三轮 correctness 均通过，`max_abs=0.000488281`、`mean_abs=9.39676e-06`，第三轮 warmed K1 `3.159-3.335 ms`、K3 combine `2.510-2.652 ms`。
- Formal bench：
  - 1024 default V3 ASM-pack5 pair fused median `2.081798881 ms`、min `1.982397884 ms`、baseline median `3.701338530 ms`、speedup `1.77795x`；
  - 4096 default V3 ASM-pack5 pair fused median `6.344276965 ms`、min `6.303937256 ms`、baseline median `9.476775885 ms`、speedup `1.49375x`；
  - 相比前一次显式 env bench `2.1127/6.3658 ms` 基本一致，说明 default promotion 没引入额外开销。
- Opt-out smoke：
  - `MEGAMOE_DCU_V3_K1_ASM_PACK5=0 MEGAMOE_DCU_V3_K3_ASM_PACK5=0`、1024 correctness-only 通过，fixture 打印回到 `L1/L2 pack5 for K1 + K3 V3`，`max_abs=0.000488281`；
  - 说明 default promotion 可关闭，C/aicc fallback 的 layout 没被 plain-pack5 默认污染。

## 2026-06-13 LL refresh clean measurement

- 环境教训：远端 `dsq_sglang_601` 中的 `sglang serve` 会占满 8 卡 VRAM，且容器外进程不在 `sglang_megamoe` 内，必须检查 host `docker ps` / cgroup 后清理；否则 LL 测试会被高 VRAM/HCU 状态污染，甚至触发 local barrier timeout/VMFault dump。
- V3 LL raw rebuild：`DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll` / `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll` 用 `/opt/dtk/bin/hipcc` 编译通过；normal aicc 与 V2 extension 未参与。
- 干净 8 卡实测：V3 LL 32/128 correctness 通过，fused median `0.9083/1.0789 ms`；同轮 no-large fused `1.1367/1.6420 ms`，V3 LL 已过 `<512` 性能门槛。
- Warmed stage timing last3：32 的 K1/K2/K3/no-tail barrier/reduce 约 `0.410/0.028/0.256/0.037/0.012 ms`；128 约 `0.454/0.028/0.382/0.042/0.014 ms`。当前 LL residual 优先级：K1 固定成本 > K3 128 combine。

## 2026-06-13 LL K1 optional output skip

- 代码事实：`large_opt.py` 的 V3 staged LL 链路只消费 K1 返回的 `l1_out/route_weights/m_indices/output_index/row_combine_ptrs`，当前 V3 K2/K3 不消费 K1 内部的 `local_topk_mask/tail_tokens`；这些是 V2 runtime/K3 风格接口遗留。
- A/B 改动：仅让 V3 K1 LL raw launch 对 `local_topk_mask/tail_tokens` 传 `nullptr`，触发 `v3_k1_build_ll_stage_device()` 已有 fast path，跳过额外本地 token scan 和一次 grid barrier；scratch 布局、normal 路径、K1/K3 launch 数都不变。
- 验证：远端 source pytest `11 passed`；LL raw rebuild 使用 `/opt/dtk/bin/hipcc`，未触发 V2 extension；32/128 V3 LL correctness 均通过。
- 性能：formal fused median `0.9023/1.0752 ms`，较 clean baseline `0.9083/1.0789 ms` 小幅下降；warmed last3 stage timing 32 为 total/K1/K2/K3/no-tail barrier/reduce `0.7655/0.404/0.028/0.256/0.032/0.012 ms`，128 为 `0.955/0.449/0.028/0.378/0.052/0.014 ms`。
- 结论：该项收益小但稳定、风险低，可保留；LL 下一步仍优先 K1 固定 staging/copy 开销，其次 K3 128 remote combine。

## 2026-06-13 LL block_m=16 A/B rejected

- 动机：K1/K3 LL template `static_assert` 允许 `kBlockM=16`，而当前 env/launcher 只开放 32/48/64；理论上 32 tokens 档位每 expert 平均约 6 route，`block_m=16` 可以减少 K1 padding work。
- A/B 改动：临时开放 `MEGAMOE_DCU_V3_LL_BLOCK_M=16`，新增 K1/K3 LL launcher 分派，默认仍 32；远端 source pytest 通过，LL raw rebuild 通过。
- 结果：`MEGAMOE_DCU_V3_LL_BLOCK_M=16` correctness 32/128 均通过，但 formal fused median 恶化到 `1.0439/1.3141 ms`，明显慢于 retained 32 默认 `0.9023/1.0752 ms`。
- Stage 证据：32 warmed last3 K1 从约 `0.404` 降到 `0.383 ms`，但 K3 combine 从 `0.256` 涨到 `0.4205 ms`；128 K1/K3 约 `0.5205/0.479 ms`，均差于默认。
- 结论：小 block_m 降低 K1 padding 但增加 K3 tile/epilogue overhead，端到端不划算；已撤回 16 gate，不保留该方向。

## 2026-06-13 LL K3 rowptr-resource A/B rejected

- 动机：LL 128 的 K3 rowptr staged split 仍明显慢于 pure/local，且 K3 LL row address prefetch 使用 `global_load_dwordx2 ... glc`；normal K3 曾用 raw-buffer rowptr load 降低 floor，因此尝试把同一思路局部转到 LL。
- A/B 改动：仅在 `V3_K3_Pure_LowLatencyMaskedGroupGemmKernel` 的 row address prefetch 中把 `global_load_i64_glc_device(row_combine_ptrs + logical_row)` 改为 raw-buffer resource load；GEMM 主体、`global_store_dwordx2` store 形态、scratch layout 和 launch 数均不变。实验 gate 只覆盖 `ll_block_m=32`，避免额外 block48/64 模板膨胀。
- 验证：远端 source pytest 通过；LL raw rebuild 通过；`MEGAMOE_DCU_V3_LL_K3_ROWPTR_RESOURCE=1` 下 32/128 correctness 均通过。dccobjdump extract 可生成 gfx938 code object，但本轮 `--show-sass` 未产出有效 SASS 文本，ISA 计数记为 inconclusive，不作为保留证据。
- 性能：formal fused median 32/128 为 `0.8876/1.0871 ms`，对比 retained `0.9023/1.0752 ms` 是 32 小幅波动改善、128 明确回退；stage warmed last3 中 32 K3 `0.2605 ms` 对比 retained `0.256 ms`，128 K3 `0.391 ms` 对比 retained `0.378 ms`。
- 结论：raw-buffer rowptr load 没有降低 LL K3 remote combine gap，反而让 128 K3 变慢；已撤回代码和 env gate，后续不重复该方向，除非先有新的 ISA/PMC 证据。

## 2026-06-13 LL pre-K1 rank barrier skip A/B rejected

- 动机：LL retained stage timing 中 `start->after_barrier` 仍有约 `0.03-0.05 ms` 固定成本，计划中允许在 correctness 和跨 rank 可见性证据充分时尝试 K1 前 rank barrier A/B。
- KB 结论：检索没有给出可直接删除该 barrier 的强安全证据；DeepEP/custom allreduce 参考仍强调 release/acquire 和 final-sync visibility 语义需要区分。
- A/B 改动：临时新增 `MEGAMOE_DCU_V3_LL_SKIP_PRE_K1_BARRIER=1`，只允许 LL no-tail、无 tail signal/reset 路径跳过 K1 前 rank barrier；normal、tail-reduce 和 no-tail signal 路径不参与，默认仍保留 barrier。
- 结果：短 smoke 32/128 在第一档直接 correctness 失败，rank0 报 `fused/baseline max_abs=0.04144287109375 exceeds --atol=0.0035`，多进程随后被 torch spawn 终止。
- 结论：K1 前 rank barrier 对 dispatch-pull 跨 rank input visibility 仍必需；已撤回代码和 env gate，后续不要重复 skip-pre-K1-barrier 方向，除非先有新的输入 copy 可见性机制或 system-scope signal 证据。

## 2026-06-13 LL K1 output_index full-skip A/B rejected

- 动机：当前 staged V3 LL K2/K3 不消费 `output_index`，尝试在 LL large_opt K1 raw launch 中传 `nullptr`，跳过全量 `-1` 清理和 route 写回，默认合同仍保留。
- 验证：远端 source pytest 通过 `11 passed`；LL raw rebuild 通过；`MEGAMOE_DCU_V3_LL_SKIP_OUTPUT_INDEX=1` 下 32/128 correctness 均通过。
- Formal perf：32/128 fused median `0.8989/1.0846 ms`，对比 retained `0.9023/1.0752 ms` 是 32 小幅噪声改善、128 明确回退。
- Stage 证据：32 warmed last3 total/K1/K3 约 `0.778/0.397/0.2565 ms`；128 warmed last3 total/K1/K3 约 `0.956/0.4425/0.376 ms`，K1 有很小下降但没有转化成稳定 e2e 收益。
- 结论：`output_index` 清理/写回不是当前 LL 主要瓶颈；跳过 metadata 返回还会增加 wrapper contract 风险。已撤回 env、pybind 参数、Python wrapper 参数和 source guard，后续不重复该方向。

## 2026-06-13 LL K1 CUS=32 A/B rejected

- 动机：retained LL 的 K1 stage 固定成本仍高于 pure LL，尝试把 K1 LL blockM=32 的 CUS 从 64 降到 32，验证低 token 下 CTA 数和多次 grid barrier/metadata 阶段是否过重；K3、block_m、scratch layout 与默认路径不变。
- A/B 改动：临时新增 `MEGAMOE_DCU_V3_LL_K1_CUS=32`，只影响 K1 LL raw launcher 的 `DCU_MEGAMOE_V3_LAUNCH_K1_LL(32, 32, *)` 分支；默认仍为 CUS=64。
- 验证：远端 source pytest `11 passed`，LL raw rebuild 通过；32/128 correctness 均通过。
- Formal perf：CUS=32 的 32/128 fused median 为 `1.0166/1.1953 ms`，明显慢于 post-revert retained `0.8986/1.0861 ms`。
- Stage 证据：CUS=32 warmed last3 32 total/K1/K3 为 `0.8825/0.520/0.2495 ms`，128 为 `1.072/0.580/0.367 ms`；K3 有小幅波动但 K1 显著回退，对比 retained K1 `0.402/0.444 ms` 劣化明确。
- 结论：降低 K1 CUS 没有减少有效 fixed cost，反而损害 K1 GEMM/staging 并拖慢 e2e；grid barrier/CTA 数不是当前可通过 CUS=32 解决的主瓶颈。已撤回 env、wrapper 参数、launcher 32-CUS 模板和 source guard，后续不重复该方向，除非先有 PMC/ISA 证据显示 CUS/occupancy 是瓶颈。

## 2026-06-13 LL K1 saturated-count no-clamp-barrier A/B retained

- 动机：K1 LL route 扫描后已有一次 grid barrier 保证 `symm_counts` atomic 完成，原实现随后把 `symm_counts[expert]` 写回 clamp 到 `m_per_expert` 并再做一次 grid barrier；KB 检索没有发现必须全局写回 clamp 的同步要求，只要 stats/staged-copy/GEMM 消费处本地饱和即可保持容量语义。
- A/B 改动：删除 route 完成后的全局 clamp loop 和额外 grid barrier；stats、stage-copy 和 GEMM 读取 `symm_counts` 时使用 `min(count, m_per_expert)`。该改动不改变 scratch layout、K2/K3 合同、launch 数、CUS 或 block_m。
- 构建注意：第一次远端 build 因 Ninja 未捕捉 `.cuh` 依赖而 `no work to do`；确认 `.cuh` 新于 K1 object 后只删除 `K1_fused/k1_v3_fused_ext.hip` 和对应 `build/.../k1_v3_fused_ext.o`，强制重编 K1 LL raw，日志 `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_no_clamp_barrier_forced_20260613.log`。
- Correctness/perf：32/128 formal correctness 均通过；第一轮 formal fused `0.9025/1.0639 ms`，stage last24 total/K1/K3 为 `0.774/0.403/0.2565 ms` 与 `0.951/0.443/0.3815 ms`。128 确认轮 fused `1.0744 ms`，说明收益不稳定，主要与 retained `1.0752/1.0861 ms` 同档。
- 结论：作为低风险微优化保留，因为它去掉冗余写回和 barrier 且未引入 32/128 correctness 回退；但不把它记为主要性能突破。LL 下一步仍看 K1 stage-copy/route 固定成本和 K3 128 residual。

## 2026-06-13 LL K1 stage-copy expert-loop retained

- 动机与诊断：
  - stage-row 诊断显示 32 tokens 下 active rows 约 `192`、rounded padded rows `1024`、capacity rows `2048`，128 tokens 下 active rows约 `768`、rounded padded rows `1052`、capacity rows `2048`；
  - retained K1 stage-copy 仍按 capacity rows 扫描，32 档约一半以上是无效 work；
  - KB/optimizer 检索建议 grouped GEMM 以 expert count / padded valid-work 驱动调度，但要保持 GEMM hot path 不混入 routing 分支。
- A/B 改动：
  - 在 `v3_k1_build_ll_stage_device()` 中把 stage-copy 从线性 `row_capacity * kStageVecsPerRow` 扫描改成 per-expert loop；
  - 每个 expert 只遍历 `ceil(min(symm_counts[expert], m_per_expert) / kBlockM) * kBlockM` 行；
  - active row 从已记录的 `symm_src_x_ptrs` 复制，rounded padded row 仍写 zero，不改变 K2/K3/GEMM 合同，不新增 launch，不改变 blockM/CUS。
- 构建与正确性：
  - 强制删除 K1 `.hip/.o` 后重编，log `hygon_tmp/sglang_debug/rebuild_v3_ll_k1_stagecopy_expert_loop_20260613.log`；
  - formal 32/128 correctness 均通过，bench log `hygon_tmp/sglang_debug/v3_ll_k1_stagecopy_expert_loop_bench_20260613.log`。
- 性能：
  - formal fused 32/128 为 `0.8451/1.0663 ms`，对比 saturated-count retained `0.9025/1.0744 ms`；
  - stage last24 total/K1/K2/K3/no-tail barrier/reduce：
    - 32: `0.719/0.358/0.028/0.2575/0.0305/0.012 ms`；
    - 128: `0.932/0.421/0.028/0.3805/0.0505/0.014 ms`；
  - K1 stage 相对上一轮 `0.403/0.443 ms` 明确下降，32 e2e 收益最大，128 在噪声内偏正。
- 结论：
  - 该实现作为当前 LL retained 路径保留，并加 source guard 防止回退到 full-capacity copy；
  - 这次结果是在 2026-06-13 saturated-count no-clamp 保留后重新评估，且实际 K1 stage 有明确下降，因此覆盖 2026-06-12 的简单 staged-copy compact iteration 反证；
  - 下一步新基线下复测 K2 skip/block48/block64 是否仍无收益，并继续用 PMC/ISA 或 rowptr split 解释 128 K3 remote-combine residual。

## 2026-06-13 LL post-stagecopy K2/blockM refresh

- 命令：
  - `TOKENS_LIST="32 128" RUN_BASELINE=0 RUN_DEFAULT=1 RUN_K2SKIP=1 RUN_BLOCK48=1 RUN_BLOCK64=1 RUN_STAGE_TIMING=1 ITERS=3 WARMUP=3 REPEAT=5 bash hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`
  - 首次运行因 SSH 嵌套引号把 `TOKENS_LIST` 拆坏，仅报 `bash: 128 RUN_BASELINE=0: command not found`，未启动 bench；随后用 `env TOKENS_LIST='32 128' ...` 重跑成功。
- Formal summary：
  - default 32/128 `0.8418/1.0541 ms`；
  - K2-skip 32/128 `0.8424/1.0626 ms`，无收益；
  - block48 32/128 `0.9757/1.1396 ms`，回退；
  - block64 32/128 `2.7046/2.9043 ms`，严重回退。
- Stage last24：
  - default 32 total/K1/K2/K3 `0.718/0.3575/0.028/0.256 ms`；
  - default 128 `0.930/0.425/0.029/0.387 ms`；
  - K2-skip 32/128 `0.7305/0.9425 ms` total，K2 时间仍约 `0.027/0.029 ms`；
  - block48 32/128 K1/K3 `0.4055/0.3265 ms` 与 `0.440/0.454 ms`，两档均慢；
  - block64 32/128 K1 `2.2445/2.298 ms`，与 code-object 资源中 block64 VGPR/private-segment 压力一致。
- 结论：
  - stage-copy retained 后，K2-skip/block48/block64 旧反证仍成立；
  - 继续把 LL default block32 作为性能基线，下一步只看 K3 128 remote-combine residual 或更深 PMC/ISA。

## 2026-06-13 LL post-stagecopy K3 128 residual attribution

- K3 LL 128 rowptr split 使用当前 stage-copy expert-loop retained 路径，命令输出为 `hygon_tmp/sglang_debug/k3_ll_rowptr_modes_128_post_stagecopy_20260613.json`。
- split 结果：
  - `pure_contiguous` `0.2368 ms`，`local_rowptr` `0.2435 ms`，`rowptr_all_zero` `0.2424 ms`，`staged_local_only` `0.2394 ms`；
  - `staged_remote_only` `0.3658 ms`，`staged_rowptr` `0.3665 ms`；
  - active/local/remote rows avg per rank 为 `768.0/32.75/735.25`，rows capacity `2048`。
- 结论：
  - post-stagecopy 后 K3 LL 128 的 remaining delta 仍主要来自 remote rowptr combine 数据通路，而不是 pure GEMM 本体或 local rowptr；
  - 已反证过的 rowptr-resource、blockM、K2-skip 方向不再重复；后续 LL 优先回到 K1 fixed/stage init 或需要新 ISA/PMC 证据的 remote-store 方向。

## 2026-06-13 LL tail-reduce post-stagecopy snapshot

- 命令：
  - `env TOKENS_LIST='32 128' RUN_BASELINE=0 RUN_DEFAULT=1 RUN_K2SKIP=0 RUN_BLOCK48=0 RUN_BLOCK64=0 RUN_STAGE_TIMING=1 K3_USE_ASM_TAIL_REDUCE=1 ITERS=3 WARMUP=3 REPEAT=5 bash hygon_tmp/sglang_debug/run_v3_ll_perf_ab.sh`
- Summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260613_213706.csv`：
  - tail 32 formal fused `0.8425 ms`；
  - tail 128 formal fused `1.0759 ms`；
  - correctness 32/128 均通过。
- Stage last24 median：
  - 32 total/barrier/K1/K2/K3_tail `0.7305/0.0395/0.358/0.027/0.306 ms`；
  - 128 total/barrier/K1/K2/K3_tail `0.953/0.032/0.421/0.028/0.4715 ms`。
- 结论：
  - tail-reduce 在 stage-copy retained 后仍 correctness 稳定；
  - 32 档 tail 与 no-tail formal 同档，128 tail 比同场 no-tail default `1.0541 ms` 慢，主要体现在 K3_tail `~0.4715 ms`；normal tail 暂缓的状态不受该 LL 结果影响。

## 2026-06-13 LL K1 staged scale/init sparse A/B rejected

- 动机：
  - stage-copy expert-loop 已把 staged input copy 从 full capacity 收敛到 rounded active rows；
  - K1 init 仍对 `row_capacity=2048` 写 `staged_x_scale[idx] = 1.0e-4f / 448.0f`；
  - 尝试只在 rounded padded rows 写安全 tiny scale，active rows 继续由 route scan 写真实 `sections.x_sf[token_idx]`。
- KB 参考：
  - `dcu-rag-kb-optimize` 返回 Hygon DeepGEMM grouped GEMM 与 padding-strategy 资料，支持减少 padding work 作为方向，但没有给出必须 full-capacity scale init 或可以无条件删掉的证据；
  - 因此该项只按 correctness + timing 闭环决定。
- A/B 改动：
  - 移除 full-capacity init loop 中的 `staged_x_scale[idx]` 写入；
  - 在 expert-loop stage-copy 中，`row_in_expert >= expert_count && vec_col == 0` 时写 padded row tiny scale；
  - 不改变 scratch layout、launch 数、K2/K3 合同、blockM/CUS 或 staged_x zero 写入。
- 验证与性能：
  - 首次远端 bench 命令因 PowerShell 嵌套引号把 `TOKENS_LIST` 拆坏，未启动 bench；改用 `$inner` 包装后成功；
  - remote K1 LL rebuild 通过，32/128 correctness 均通过；
  - formal 32/128 fused `0.8452/1.0693 ms`，慢于 retained `0.8418/1.0541 ms`；
  - stage last24 32 total/K1/K2/K3/no-tail barrier/reduce `0.723/0.3595/0.028/0.256/0.0325/0.012 ms`；
  - stage last24 128 `0.9325/0.4225/0.029/0.3835/0.0495/0.014 ms`。
- 结论：
  - sparse scale init 没有降低 K1 stage，128 formal 明确回退；
  - 已撤回到 full-capacity tiny scale init，远端 K1 LL rebuild 后 32/128 短 sanity correctness 通过；
  - 后续不重复 scale-init sparse 方向，除非先有 PMC/ISA 证明 full-capacity scale store 是主要瓶颈。

## 2026-06-14 LL retained path code-object attribution

- 直接对 `.so` 用 `dccobjdump --show-sass` 未得到有效函数级 SASS，按 Hygon optimizer fallback 使用 `hipcc -save-temps=obj` 生成 gfx938 `.s` 后解析 active 模板实例。
- K1 LL block32/CUS64 active 实例：
  - mask true/false 两个版本资源相同：`VGPR=124`、`SGPR=96`、`private_segment=0`；
  - 指令族约 `v_mmac=1024`、`global_load_dwordx4=388`、`global_load_dwordx2=7`、`global_store_dwordx2=12`、`s_waitcnt=525`、`s_barrier=10`；
  - block48 资源升到 `VGPR=151`，block64 升到 `VGPR=193` 且 `private_segment=272`，与 block64 K1 stage `~2.24/2.30 ms` 的严重回退一致。
- K3 LL block32/CUS64 active 实例：
  - no-tail 资源 `VGPR=153`、`SGPR=96`、`private_segment=0`，约 `v_mmac=512`、`global_load_dwordx4=196`、`global_load_dwordx2=2`、`global_store_dwordx2=16`、`s_waitcnt=287`；
  - tail 资源 `VGPR=130`、`private_segment=0`，但额外 tail 逻辑带来更多 load/store/barrier，性能上 128 tail 仍慢于 no-tail。
- 结论：
  - K1 默认 block32 没有 scratch spill，剩余 fixed/stage 成本更可能来自 route/metadata/staged-copy/wait，而不是寄存器灾难；
  - block64 已由资源证据解释，继续排除；
  - K3 128 remaining delta 不是 obvious register/scratch 问题，仍与 rowptr remote combine 数据通路和 store/wait schedule 相关；
  - 下一项低风险优化选 LL 专用 parallel stage-copy，先不触碰 K3 store 形态。

## 2026-06-14 LL K1 parallel stage-copy A/B retained

- 动机：
  - expert-loop stage-copy retained 后仍由单 CTA group 逐 expert 复制 rounded active rows；
  - 32/128 的 rounded rows 约 `1024/1052`，copy work 已少于 capacity rows，但 K1 stage 仍是 LL 最大 fixed cost；
  - 该 A/B 只增加同一 K1 main kernel 内的 stage-copy 并行度，不新增 runtime kernel，不改变 route/GEMM/K2/K3 合同。
- A/B 改动：
  - `v3_k1_build_ll_stage_device()` 新增 `kParallelStageCopy` 模板参数；
  - A/B 时通过临时 env 且 blockM=32/CUS=64 启用，使用 `gridDim.x / kExperts` 个 CTA 分摊每个 expert 的 stage-copy；
  - active rows 仍从 `symm_src_x_ptrs` 复制，rounded padded rows 仍写 zero；保留后已提升为默认开启，2026-06-15 cleanup 已固定为生产默认并删除诊断回退 env。
- 构建与验证：
  - 远端 source pytest `11 passed`；
  - K1/K3 LL raw pair rebuild 通过，K1/K3 raw 都进 `.so`；
  - 32/128 formal correctness 均通过，summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_074211.csv`。
- 性能：
  - parallel stage-copy formal fused 32/128 为 `0.8311/1.0337 ms`；
  - 对比 expert-loop retained `0.8418/1.0541 ms`，两档均改善；
  - stage last24 median：
    - 32 total/K1/K2/K3/no-tail barrier/reduce `0.7125/0.346/0.028/0.256/0.0305/0.012 ms`；
    - 128 total/K1/K2/K3/no-tail barrier/reduce `0.9055/0.398/0.028/0.384/0.042/0.014 ms`；
  - K1 stage 相对上一 retained `0.3575/0.425 ms` 下降，K3 基本同档。
- Default promotion 确认：
  - 改为默认开启、临时 opt-out 后，远端重新 K1/K3 LL raw rebuild；
  - 无 env 默认 32/128 三轮 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_074834.csv` 为 `0.8332/1.0543 ms`，其中 128 有波动但 stage last24 K1 `0.400 ms` 证明默认已走 parallel；
  - 追加 128-only 五轮确认 summary `hygon_tmp/sglang_debug/v3_ll_perf_summary_20260614_074935.csv` 为 `1.0444 ms`，stage last24 total/K1/K3 `0.9115/0.4005/0.389 ms`。
- 结论：
  - parallel stage-copy 作为当前 LL retained 路径保留；
  - 下一优先级转向 K3 LL 128 remote rowptr combine delta，避免继续在已反证的 blockM/CUS/rowptr-resource/scale-init 方向重复。

## 2026-06-14 LL K3 store modifier A/B rejected

- 背景：K3 LL no-tail 当前 epilogue 已用 `store_bf16x4_rowaddr_device()` 打包成 `global_store_dwordx2`，不是简单 scalar short store；normal ASM 的 scalar-store 结论不能直接套到 LL。
- KB/ISA 结论：Hygon 资料确认 `glc/slc` 是真实 cache/coherency 语义，是否有收益必须实测；probe `hygon_tmp/sglang_debug/probe_global_store_flags.hip` 验证 `global_store_dwordx2 ... off`、`off glc`、`off glc slc` 都可编译。
- A/B 改动：临时增加 `MEGAMOE_DCU_V3_LL_K3_STORE_MODE=glc|glc_slc`，只改 K3 LL no-tail rowptr combine store modifier，GEMM 主循环、rowptr load、K1/K2 合同和 launch 数不变。
- 验证：远端 source pytest `11 passed`，K1+K3 LL raw pair rebuild 通过，32/128 correctness 在该轮通过。
- 128-only 结果：
  - default fused `1.0305997431 ms`，K3 last24 `0.382 ms`，stage total `0.9085 ms`；
  - `glc` fused `1.0275392383 ms`，K3 `0.380 ms`，stage total `0.8895 ms`；
  - `glc_slc` fused `1.0316999853 ms`，K3 `0.3845 ms`，stage total `0.9005 ms`。
- 结论：`glc` 只有约 `0.002 ms` K3、`0.3%` e2e 的噪声级信号，`glc_slc` 回退；已撤回 env、helper 与 source guard，不保留该方向。后续若再碰 store modifier，必须先有 PMC/SQTT 证明 cache/coherency 是主瓶颈。

## 2026-06-14 LL K3 deferred rowptr wait A/B rejected

- 背景：K3 LL no-tail rowptr 地址当前用 `global_load_i64_glc_device()` 预取，该 helper 内部立即 `s_waitcnt vmcnt(0)`；理论上可以在 GEMM 主循环中先发 rowptr load，在 epilogue consume 前再 wait，尝试隐藏 rowptr load latency。
- KB 结论：`s_waitcnt vmcnt(0)` 在消费 global load 值前仍是必须语义，只能移动位置，不能省略。
- A/B 改动：临时增加 `MEGAMOE_DCU_V3_LL_K3_DEFER_ROWPTR_WAIT=1`，发出 `global_load_dwordx2 ... glc` 后不立即 wait，在 epilogue row address 消费前执行 `s_waitcnt vmcnt(0)`；store 形态、GEMM 主循环和合同不变。
- 验证：远端 source pytest `11 passed`，K1+K3 LL raw rebuild 通过；default/defer correctness 均通过。
- 128-only 结果：
  - default fused `1.0348001122 ms`，K3 last24 `0.389 ms`，stage total `0.9085 ms`；
  - defer fused `1.0488800108 ms`，K3 `0.410 ms`，stage total `0.928 ms`。
- 结论：延迟 wait 没有隐藏 latency，反而拉长 K3 和 e2e；已撤回 env、no-wait helper、wait alias、模板参数与 source guard。下一步不再盲目移动 waitcnt，转用 profiler/PMC 或更小 split harness 看 remote rowptr combine 的真实 stall。

## 2026-06-14 LL K3 remote combine PMC attribution

- 工具链：
  - 通过 `hipprof --pmc --pmc-read/--pmc-write --pmc-type 3 --follow-fork --kernel-name V3_K3_Pure_LowLatencyMaskedGroupGemmKernel` 对 K3 LL 128 split modes 做 PMC；
  - 产物在远端 `hygon_tmp/sglang_debug/pmc_ll_k3_*_current_20260614_081823.summary.json`，已拉回本地 `hygon_tmp/sglang_debug/prof_pull_current_20260614_081823`。
- 关键对比：
  - `local_rowptr` median 约 `0.48-0.55 ms`，`staged_remote_only` median 约 `0.54-0.66 ms`；
  - VMEM 指令数接近：`SQ_INSTS_VMEM_RD` 均约 `819200`，`SQ_INSTS_VMEM_WR` 均约 `16128`；
  - 但 remote path 的 `TA_BUSY` 约 `24.9-25.5M`，local path 约 `18.0M`，约 `1.38-1.41x`；
  - remote path 的 `TCP_TA_DATA_STALL` 约 `1.57-1.61M`，local path 约 `0.79M`，约 `2x`；
  - remote path 的 `SQ_WAIT_ANY` / LDS wait 也更高。
- Rowptr locality：
  - staged remote-only active/local/remote rows 约 `768/32.75/735.25`；
  - 每 16-row chunk 常常 fan out 到约 `4-5` 个 destination rank，contiguity 较低。
- 结论：
  - K3 LL 128 residual 不是额外 GEMM 指令或显著更多 VMEM 指令导致；
  - 主瓶颈更像 scattered remote rowptr combine 引发的 TA/TCP stall 和跨 rank store path 压力；
  - 后续应优先研究 row distribution/locality、rank fanout、epilogue/store scheduling 或 split harness，而不是继续调 CUS、store modifier 或 waitcnt 位置。

## 2026-06-14 LL K3 CUS=32 A/B rejected

- 动机：PMC 指向 remote store path/TA stall，尝试只降低 K3 no-tail CUS 到 32，看是否能降低跨 rank写入压力；该实验只对 `ll_block_m=32 && !tail_reduce` 生效，K1/K2/GEMM 合同不变。
- A/B 实现：
  - 临时增加 `MEGAMOE_DCU_V3_LL_K3_CUS=32`，把 K3 LL no-tail launch grid 从 CUS64 改为 CUS32；
  - source pytest 通过，K1+K3 LL raw pair rebuild 通过，32/128 correctness 均通过。
- 性能：
  - default 32/128 fused `0.8352/1.0301 ms`；
  - CUS32 32/128 fused `1.0381/1.2106 ms`，两档明确回退；
  - stage timing 中 128 K3 combine 从 default 约 `0.37-0.41 ms` 退到 CUS32 约 `0.55-0.59 ms`。
- 结论：
  - 降低 K3 CUS 没有缓解 remote combine，反而减少并行度导致 K3 stage 明显拉长；
  - 已撤回 env gate、CUS 模板参数和 source guard，远端重新 build 后 `has_k1_pack5=True`、`k3_ll=True`；
  - 后续不再重复 K3 CUS32 方向，除非有新的 PMC/SQTT 证明当前结论失效。

## 2026-06-14 LL K3 rowptr dest-sort split diagnostic rejected

- 动机：PMC 与 rowptr stats 显示 remote rowptr chunk fanout 较散，尝试在不改生产 kernel 的 split harness 中把 rowptr 按目标 rank 排序，验证 locality 本身是否能降低 K3 remote combine。
- 实验方式：
  - 只修改 `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`；
  - 新增 `staged_remote_dest_sorted` 与 `staged_rowptr_dest_sorted` 诊断 mode；
  - 对每个 expert 内的 rowptr 按 destination rank / combine row index 排序；act、scale、m_indices 和生产 K1/K3 不变，因此该实验只看 store locality 上限，不作为 correctness 路径。
- 128 tokens 结果：
  - `staged_remote_only` `0.36516 ms`；
  - `staged_remote_dest_sorted` `0.36446 ms`；
  - `staged_rowptr` `0.36720 ms`；
  - `staged_rowptr_dest_sorted` `0.36739 ms`；
  - active/local/remote rows 仍为 `768/32.75/735.25`。
- 结论：
  - 简单按目标 rank 提高 rowptr locality 没有可操作收益，效果在噪声内；
  - 不应为了该方向改生产 K1 row emission 或重做 source-rank grouping；
  - K3 LL remaining delta 更可能是 remote write path ceiling / TA stall 本身，而不是 row order 可轻易修复。

## 2026-06-14 normal V3 ASM-pack5 refresh baseline

- 当前 production champion 是 V3 normal no-tail 默认 K1/K3 ASM-pack5 pair，1024/4096 correctness 通过，formal e2e 为 `2.1997/6.5323 ms`，对原 staged fused `2.3402/6.6610 ms` 分别快约 `6.4%/2.0%`。
- Stage timing last24 显示：
  - 1024 rows=8192：K1 `1.018 ms`，K3 `0.811 ms`；
  - 4096 rows=29696：K1 `3.320 ms`，K3 `2.526 ms`。
- K3 ASM-pack5 direct rowptr split 为：
  - 1024：all-zero `0.4776 ms`，local `0.5096 ms`，staged-remote `0.8153 ms`，staged-rowptr `0.7996 ms`；
  - 4096：all-zero `1.5001 ms`，local `1.5495 ms`，staged-remote `2.4683 ms`，staged-rowptr `2.4801 ms`；
  - full staged K3 `0.811/2.526 ms` 已贴近 direct staged rowptr floor；这只说明真实 large_opt 编排对 K3 的额外损耗很小，不表示 K3 已贴近 pure C/groupgemm 下界。
- K1 pure C pack5 refresh 为 `0.746/2.252 ms`，对应 full staged K1 为 `1.018/3.320 ms`，normal 剩余差距集中在 K1 dispatch-pull/metadata/staged input/同步结构，而不是 K3 no-tail combine。
- K3 的性能空间仍必须按 pure-vs-fused 衡量：pure C/groupgemm 是 GEMM 骨干下界，direct K3 staged-rowptr 是“带真实 rowptr combine 通信语义的 K3 主 kernel”下界，full-stage K3 是 e2e 编排中的 K3 段。当前只能说明 full-stage 与 direct K3 接近；若继续优化 K3，应聚焦主 kernel 内 rowptr combine/store/remote write 成本，而不是 full-chain 编排。
- K1 original ASM vs K1 ASM-pack5 code-object 计数：
  - `v_mmac=640`、`buffer_load_dwordx4=32`、`buffer_load_ubyte=128`、`buffer_store_short=640`、`s_waitcnt=185`、`s_barrier=33` 均相同；
  - pack5 版本 scalar `s_lshl/s_add/s_mul` 更少，但 vector 地址类指令略多；
  - 当前没有发现 K1 ASM-pack5 地址数学有类似 K3 早期重复 A-offset 的强信号。
- 结论：normal 下一轮优化应优先围绕 K1 residual 做 profile/direct harness/同步或 staging A/B；K3 no-tail ASM-pack5 只作为 correctness/perf regression guard。若继续 K3，应先有 PMC/SQTT/ISA 新证据，避免重复 C/aicc scalar-store、vector-store-window、rowptr-resource 等已反证路线。

## 2026-06-14 normal K3 pure-vs-fused measurement correction

- 口径修正：K3 direct staged-rowptr 只能作为“带 rowptr combine 通信语义的主 kernel 下界”，不能替代 pure C/groupgemm 下界。
- 当前 full-stage K3 接近 direct staged-rowptr，只能说明 `large_opt` 编排和 K3 wrapper 额外成本小；K3 主 kernel 内 rowptr combine、remote/scattered store、communication semantics 相对 pure 的 delta 仍需要单独实测。
- 已新增 diagnostic-only `k3_v3_normal_pure_raw` 入口，复用已有 K3 normal raw C kernel 且传 `row_combine_ptrs=nullptr`，不接 runtime；后续 1024/4096 split 应同时记录 `pure_contiguous/local_rowptr/staged_remote/staged_rowptr`。

## 2026-06-14 normal K3 pure-vs-fused split results

- 实测文件拉回到 `hygon_tmp/sglang_debug/normal_k3_pull_20260614_1017/`；远端原始 JSON 在 `hygon_tmp/sglang_debug/v3_normal_k3_*_20260614_101*.json`。
- 1024 tokens / rows=8192：
  - C normal pure-contiguous `0.445 ms`；C normal staged-rowptr `2.261 ms`；
  - ASM-pack5 rowptr-all-zero `0.485 ms`，local-rowptr `0.514 ms`，staged-remote `0.827 ms`，staged-rowptr `0.825 ms`。
- 4096 tokens / rows=29696：
  - C normal pure-contiguous `1.546 ms`；C normal staged-rowptr `7.946 ms`；
  - ASM-pack5 rowptr-all-zero `1.501 ms`，local-rowptr `1.553 ms`，staged-remote `2.491 ms`，staged-rowptr `2.491 ms`.
- 结论：
  - ASM-pack5 的 all-zero/GEMM floor 已经与 C pure-contiguous 同档，4096 甚至略快；当前 K3 normal 大头不是 GEMM core 或 pack5 weight 地址数学。
  - ASM-pack5 staged-rowptr 相对 C pure 的 gap 约 `0.38 ms / 0.95 ms`，几乎都来自 remote/scattered rowptr combine store：all-zero 到 staged-rowptr 增量约 `0.34 ms / 0.99 ms`，local-rowptr 增量只有约 `0.03 ms / 0.05 ms`。
  - 后续 K3 normal 优化应只围绕 ASM-pack5 no-tail epilogue/remote-store/wait schedule/TA-TCP stall 做证据驱动微调；不再回到 C/aicc scalar-store fallback 当主线。

## 2026-06-14 normal K3 ASM-pack5 remote-store PMC and glc A/B

- DCU KB / Flux / DeepEP 参考方向：通信语义适合贴在 GEMM epilogue/store 路径并用 mode gate 控制；这支持继续围绕 K3 combine epilogue 做窄 A/B，而不是重写 GEMM core。
- PMC 产物已拉回到 `hygon_tmp/sglang_debug/prof_pull_normal_k3_pmc_20260614_1041/`；kernel filter 修正为 code object 内真实名字 `DeepGemm_W8A8_F8_PERCHANNEL_ASM_TN_MT256X256X128_BF16_K3COMBINE`。
- 4096 tokens PMC 对比：
  - `rowptr_all_zero` read/write median 约 `1.587/1.590 ms`，`TA_BUSY≈47.17M`，`TCP_TA_DATA_STALL≈0.90M`；
  - `local_rowptr` read/write median 约 `1.645/1.640 ms`，`TA_BUSY≈49.2M`，`TCP_TA_DATA_STALL≈2.36-2.45M`；
  - `staged_remote_only` read/write median 约 `2.432/2.438 ms`，`TA_BUSY≈56.5M`，`TCP_TA_DATA_STALL≈8.49-8.51M`；
  - `staged_rowptr` read/write median 约 `2.406/2.392 ms`，`TA_BUSY≈57.9M`，`TCP_TA_DATA_STALL≈9.29-9.31M`。
- `SQ_INSTS_VMEM_RD/WR` 在各 mode 中一致，约 `2,264,320/237,568`；K3 staged store path 已经是 `K3_STORE_STAGED_HALF` 里的 `global_store_dwordx4`，不是 scalar store。
- glc A/B：把四条 staged `global_store_dwordx4 ... off` 临时改成 `off glc` 后，direct correctness 1024/4096 通过；perf 中 1024 staged-rowptr `0.818 ms` vs retained `0.825 ms` 只有噪声级改善，4096 `2.580 ms` vs retained `2.491 ms` 明确回退。
- 已撤回 glc 并恢复远端 PACK5 `.co` sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`，恢复后 direct correctness 1024/4096 仍通过。
- 结论：K3 normal no-tail ASM-pack5 的 GEMM core 已贴近 pure，full-stage K3 又贴近 direct staged-rowptr；remaining gap 是 remote/scattered rowptr combine store 数据通路上限。下一步 normal 优先转向 K1 residual，K3 只保留 regression guard 或需要新 PMC/SQTT 证据时再做更窄 epilogue A/B。

## 2026-06-14 normal K1 direct residual and rejected narrow A/B

- Clean retained K1 direct sanity after removing temporary margin gate:
  - 1024 V3 K1 direct median `0.978 ms`，rows `8192`，active rows per rank约 `6082-6214`，`max_abs=0`；
  - 4096 V3 K1 direct median `2.935 ms`，rows `29696`，rank0 `active_tiles=113`、`ceil_tiles_from_counts=113`、`sum_counts=24678`，`max_abs=0`。
- Pure C pack5 reference remains `0.746/2.252 ms` for 1024/4096 uniform harness, but 4096 pure uses 96 tiles while real fused route uses about 113 active tiles and capacity rows `29696`; direct K1 residual must therefore be interpreted with tile count correction, not as a raw `2.935 - 2.252 ms` gap.
- V3-only K1 4096 PMC showed no scratch spill: `VGPR=256`、`SGPR=112`，`SQ_INSTS_VALU≈79.8M`，`SQ_INSTS_VMEM_RD≈4.99M`，`SQ_INSTS_VMEM_WR≈1.94M`，`SQ_WAIT_INST_LDS≈25.0M`，`TA_BUSY≈144M`，`TCP_TA_DATA_STALL≈18.7M`。当前信号更像 dispatch-pull/metadata/staged input 和 LDS/global memory path 的结构成本，而不是寄存器 spill。
- Rejected A/B:
  - `K1_NORMAL_BENCH_STATS=0` 对 1024/4096 只有噪声级变化，stats 不是主要瓶颈；
  - `K1_PREBUILD_MODE=asm` 在 4096 走 rows `32768`，约 `3.155 ms`，慢于 compact/default rows `29696` 的约 `2.92-2.94 ms`；
  - `MEGAMOE_DCU_V3_K1_COMPACT_MARGIN_TILES=0` 会让 4096 rank0 active rows 从 `24678` 掉到 `24666`，route overflow/loss，判定 incorrect；
  - `MEGAMOE_DCU_V3_K1_COMPACT_MARGIN_TILES=1` 保持 active rows 但 4096 约 `2.942 ms`，无收益。
- 已删除临时 `MEGAMOE_DCU_V3_K1_COMPACT_MARGIN_TILES` gate，远端 K1-only raw build 通过，build log `hygon_tmp/sglang_debug/rebuild_k1_clean_margin_20260614_1115.log`，`aicc_marker=1`、`v2_seen=0`。
- 下一步：在 clean retained build 上重跑 no-tail full-stage timing，将 full-stage K1 段与 direct K1 对齐；若 full-stage 仍明显高于 direct，再定位 stage wrapper、barrier、route scratch/reset 或 timing harness 差异。

## 2026-06-14 normal K1 symm allocation warm-up retained

- K1 4096 full-stage/direct gap 的关键根因已收敛到 symmetric buffer allocation layout，而不是 K1 ASM-pack5 GEMM 主体：
  - direct harness `asm_first` / dummy allocation 后，V3 symm span 落在 `<=4GB`，K1 direct 约 `2.94-2.95 ms`；
  - production first symm buffer 容易让 `symm_x_span > UINT32_MAX`，K1 走 ASM bit2 `{rank-local offset, source rank}` MUBUF 路径，full-stage K1 约 `3.26-3.32 ms`。
- 已在 `megamoe/__init__.py` 加 V3 normal-only one-time symm allocator warm-up：
  - 仅在 HIP + `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1` + `USE_MEGAMOE_V3=1` + backend normal/unset + tokens/rank `>=512` + 目标 shape `(256,6,4096,2048)` 时默认启用；
  - `MEGAMOE_DCU_V3_SYMM_WARMUP_ALLOC=0` 可关闭；
  - dummy `SymmBuffer` 使用最小 token alignment，且 `prepare_large_opt_3stage=False`，立即 `destroy()`；不在 K1/K3 执行/bench 热路径新增 kernel。
- 远端 source pytest：`12 passed in 5.62s`。
- 4096 formal A/B：
  - warm-up default：correct，formal e2e `6.2910 ms`，stage last24 total/K1/K2/K3/barrier/reduce `6.044/2.9455/0.214/2.5505/0.133/0.168 ms`；
  - `MEGAMOE_DCU_V3_SYMM_WARMUP_ALLOC=0`：correct，formal e2e `6.6039 ms`，stage last24 total/K1/K2/K3/barrier/reduce `6.3605/3.260/0.214/2.543/0.1305/0.168 ms`。
- 1024 warm-up default correctness 通过，formal e2e `2.2322 ms`，stage last24 total/K1/K2/K3/barrier/reduce `2.169/1.026/0.105/0.821/0.090/0.048 ms`。
- 已删除临时弱信号 gate `MEGAMOE_DCU_V3_K1_FORCE_ABSOLUTE_PTRS`，因为 absolute-pointer path 只给 4096 带来噪声级改善，不能解决主瓶颈；远端 K1 rebuild `hygon_tmp/sglang_debug/rebuild_k1_remove_force_abs_20260614_1216.log` 通过，`aicc_marker=1`、`k1_raw_compile_seen=2`、`v2_seen=0`。
- K1 rebuild 后 4096 default sanity correctness 通过，短跑 e2e `6.1313 ms`，stage last24 total/K1/K2/K3/barrier/reduce `6.054/2.8855/0.213/2.5785/0.1495/0.168 ms`；该短跑只作 rebuild sanity，不替代 formal A/B。
- 结论：
  - K1 normal 4096 full-stage 已基本回到 direct fast path；剩余 K1 delta 相对 pure 需按 active tile count 与 dispatch-pull semantics 理解，不再优先追 host/stage 编排。
  - normal 下一优先级转为 post-warmup remaining-gap refresh：固定 warm-up + ASM-pack5 champion 后，重新汇总 1024/4096 formal、K1 direct floor、K3 pure/direct floor；若继续优化，K3 remote/scattered rowptr combine store 仍是最大主-kernel gap。

## 2026-06-14 normal post-warmup remaining-gap refresh

- 固定默认 V3 normal symm warm-up + K1/K3 ASM-pack5 champion 后，formal correctness/perf 仍通过：
  - 1024 tokens：formal e2e `2.2322 ms`，stage total/K1/K2/K3/barrier/reduce `2.169/1.026/0.105/0.821/0.090/0.048 ms`；
  - 4096 tokens：formal e2e `6.2910 ms`，stage total/K1/K2/K3/barrier/reduce `6.044/2.9455/0.214/2.5505/0.133/0.168 ms`。
- Direct / pure floor 对照：
  - K1 direct floor 为 `0.9686/2.9452 ms`，full-stage K1 minus direct 为 `+0.0574/+0.0003 ms`；4096 的 K1 stage gap 已基本由 warm-up 消掉，1024 剩余约 `0.06 ms` 属小项。
  - K3 C pure-contiguous 为 `0.4440/1.5378 ms`，ASM-pack5 all-zero 为 `0.4757/1.4990 ms`，ASM-pack5 staged-rowptr 为 `0.8002/2.4553 ms`。
  - Full-stage K3 minus direct staged-rowptr 为 `+0.0208/+0.0952 ms`，说明 full-stage 编排额外损耗小；主 kernel 内 staged-rowptr 相对 all-zero 的 remote/scattered combine 增量仍是 `+0.3245/+0.9563 ms`，相对 C pure 为 `+0.3562/+0.9175 ms`。
- DCU KB/Flux 检索给出的可迁移结论仍是“通信语义放在 epilogue store backend 内，并保留可替换后端做 A/B”；当前 V3 K3 ASM-pack5 已符合该方向，下一步只做 epilogue wait/store scheduling 的窄 A/B，不回到 C/aicc scalar-store 或额外 runtime kernel。
- 新 A/B：在 `K3_STORE_STAGED_HALF` 中保留 `s_waitcnt vmcnt(0)` 于 rowptr 消费前，把 `s_waitcnt lgkmcnt(0)` 延后到 rowptr 地址计算之后、第一条 `global_store_dwordx4` 之前，测试地址计算能否覆盖部分 LDS read latency；不改 MMAC 主循环、rowptr 语义或 store vector width。

## 2026-06-14 normal K3 remote-write ceiling and fanout diagnostics

- `K3_STORE_STAGED_HALF` deferred LDS-wait A/B 未保留：生产 K3 ASM-pack5 已恢复到 retained `.co` sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`，后续不继续在没有新 PMC/SQTT 证据时移动该 wait。
- 在 diagnostic-only `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py` 中增加 normal K3 remote ceiling modes，不进入生产路径：
  - `staged_remote_contig_local`：远端行数相同但指向本地 contiguous combine buffer；
  - `staged_remote_contig_peer`：远端行数相同且全部指向单个 peer rank 的 contiguous combine buffer；
  - 继续保留真实 `staged_remote_only` / `staged_rowptr`。
- 4096 tokens / ASM-pack5 结果 `v3_normal_k3_rowptr_modes_asm_pack5_refresh_4096_20260614_123423.json`：
  - `rowptr_all_zero` `1.5006 ms`；
  - `local_rowptr` `1.5486 ms`；
  - `staged_remote_contig_local` `1.5439 ms`；
  - `staged_remote_contig_peer` `2.3103 ms`；
  - `staged_remote_only` `2.5121 ms`；
  - `staged_rowptr` `2.5205 ms`；
  - remote rows average `21605.875`。
- 解释：
  - local contiguous remote-count path 与 local/all-zero 同档，说明额外 row 数本身不是瓶颈；
  - single-peer contiguous 仍比 local 慢约 `0.77 ms`，这是 remote write path 的硬成本信号；
  - real staged_remote_only 比 single-peer contiguous 再慢约 `0.20 ms`，说明还有少量 fanout/scatter 成本，但不是 naive rowptr 重排能直接吃掉的大头。
- Dest-sort split 诊断 `v3_normal_k3_rowptr_modes_asm_pack5_refresh_4096_20260614_123520.json`：
  - `staged_remote_only` `2.5026 ms`；
  - `staged_remote_dest_sorted` `3.9994 ms`；
  - `staged_rowptr` `2.5215 ms`；
  - `staged_rowptr_dest_sorted` `3.7999 ms`。
- 结论：
  - normal K3 的 naive destination-rank rowptr sorting 明确回退，不应进入生产 K1 row emission 或 K3 epilogue；
  - 当前 K3 normal no-tail 的最大 remaining gap 仍是 remote write 数据通路 / TA-TCP stall ceiling，次要项才是 fanout/scatter；
  - 下一步只对 retained ASM-pack5 的 `staged_remote_contig_peer`、`staged_remote_only`、`staged_rowptr` 做 PMC/codeobj 对照，若 counters 显示 remote path 已到 ceiling，则 K3 no-tail 不继续做无证据源级微调。

## 2026-06-14 normal K3 remote-store ceiling PMC follow-up

- PMC 命令：`run_v3_normal_k3_asm_pack5_pmc.sh`，4096 tokens，modes 为 `staged_remote_contig_peer staged_remote_only staged_rowptr`，`WARMUP=1 REPEAT=4 ROUNDS=3`，产物前缀 `hygon_tmp/sglang_debug/prof/pmc_normal_k3_asm_pack5_4096_*_remote_ceiling_20260614_124056.*`。
- Timing / counters 摘要：
  - `staged_remote_contig_peer` read/write mean `2.3038/2.3028 ms`，`TA_BUSY≈51.05/51.19M`，`TCP_TA_DATA_STALL≈1.84/1.82M`；
  - `staged_remote_only` read/write mean `2.4418/2.4557 ms`，`TA_BUSY≈57.67/57.68M`，`TCP_TA_DATA_STALL≈8.57/8.62M`；
  - `staged_rowptr` read/write mean `2.4587/2.5095 ms`，`TA_BUSY≈59.24/60.06M`，`TCP_TA_DATA_STALL≈9.54/9.57M`。
- Stable counters:
  - `vmem_rd=2264320`、`vmem_wr=237568`、`valu=59080192`、`lds_bank_conflict=3801088`、`grd=1425408` 在三种 mode 中一致；
  - `wait_lds` 约 `16.3-16.5M`，没有随 staged/fanout 增加而上升。
- 比例：
  - 相对 single-peer contiguous，`staged_remote_only` time `1.060x`、`TA_BUSY 1.130x`、`TCP_TA_DATA_STALL 4.66x`；
  - `staged_rowptr` time `1.067x`、`TA_BUSY 1.160x`、`TCP_TA_DATA_STALL 5.19x`。
- 结论：
  - K3 normal no-tail remaining gap 不是额外 GEMM/VMEM 指令、LDS wait 或 VGPR/SGPR 资源引起；
  - single-peer contiguous 已经体现 remote write hard ceiling，真实 staged 的 fanout/scatter 只再增加约 `0.14-0.21 ms`，且 counters 指向 TA/TCP 数据通路；
  - 在没有 SQTT 或新的 row locality 证据前，K3 ASM-pack5 不再做 source-level store/wait/排序微调；normal 下一步转回 K1 fused-vs-pure residual。

## 2026-06-14 normal K1 staged-copy producer CTA retained

- 背景：K3 normal no-tail 已基本归因到 remote/scattered rowptr store ceiling，normal 继续回到 K1 fused-vs-pure residual。K1 direct 1024/4096 retained 约 `0.967/2.925 ms`，pure C pack5 约 `0.749/2.254 ms`；4096 按 capacity/active tile 修正后仍有约 `0.2 ms` 可追。
- `output_index` store A/B：
  - 临时跳过 route emit 中 valid/invalid `output_index` 的两条 `buffer_store_dword`，保留 tail clear、row_combine、m_indices、route_weights 与 GEMM 主体；
  - correctness 保持 `max_abs=0`，但 direct K1 1024/4096 为 `0.968/2.930 ms`，与 retained `0.967/2.925 ms` 同档；
  - 结论：`output_index` store 不是 K1 normal residual 主因，已恢复，不继续该方向。
- Staged input copy producer A/B：
  - 原 ASM-pack5 在 `label_SymmRouteMetaReady` 后只用 `wg0 < 5` 的 CTA 把 row_x 指向的输入 copy 到 contiguous `staged_x`，其余 N-tile CTA 等 per-row-tile counter；
  - producer=6：direct K1 1024/4096 `0.976/2.857 ms`；
  - producer=7：direct K1 1024/4096 `0.972/2.820 ms`；
  - producer=8：direct K1 1024/4096 `0.978/2.792 ms`，4096 最好但 1024 小退；
  - retained 版本改成基于已有 `expert_tiles_per_expert` (`s10`) 的动态策略：`s10 >= 2` 用 8 producer CTA，否则保持 5，不额外读取 `num_tokens` kernarg。
- Retained dynamic producer 结果：
  - K1 code object sha `f425cef3cd7116775dc230c881f694d00032ae8bcef36ebad0f6d7d884a48c78`；
  - direct K1 1024/4096：`0.9666/2.7990 ms`，correctness `max_abs=0`；
  - full-stage V3 normal no-tail：1024 `2.2130 ms`，4096 `6.0699 ms`，correctness 3/3 通过；
  - stage timing last24：1024 total/K1/K2/K3/barrier/reduce `2.100/1.0105/0.105/0.843/0.079/0.048 ms`；4096 `5.847/2.763/0.213/2.5355/0.101/0.168 ms`。
- PMC / ISA 证据：
  - 4096 K1 PMC old read/write duration mean `2.7687/2.7676 ms`，dynamic `2.6321/2.6286 ms`；
  - `vmem_rd` 从约 `4.98M` 降到约 `4.64M`，`TA_BUSY` 从约 `144.6M` 降到约 `139.8M`，VGPR/SGPR 保持 `256/112`，LDS 指令数不变；
  - code-object 计数仍为 `v_mmac=640`、`buffer_load_dwordx4=32`、`buffer_load_ubyte=128`、`buffer_store_short=640`、`s_waitcnt=185`、`s_barrier=33`，说明 GEMM 主体未被破坏。
- 结论：
  - K1 normal 的可操作 residual 主要在 staged input copy 并行度，而不是 output_index/stats/prebuild/compact margin；
  - dynamic producer 策略可保留，下一步以该 K1 + retained K3 ASM-pack5 为 normal champion，重跑 original vs V3 同场 refresh 与 pure/fused delta 汇总。

## 2026-06-14 normal post-K1-producer champion refresh

- Same-session original staged fused vs V3 normal no-tail 已重跑，环境为 `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1`、`USE_MEGAMOE_V3=1`、`MEGAMOE_DCU_V3_BACKEND=normal`、no-tail，`WARMUP=3 REPEAT=5 ITERS=3`：
  - 1024 tokens：原 staged fused `2.4134 ms`，V3 normal `2.3054 ms`，V3 speedup `1.0468x`，两者 correctness 均为 `True`；
  - 4096 tokens：原 staged fused `6.6033 ms`，V3 normal `6.0640 ms`，V3 speedup `1.0889x`，两者 correctness 均为 `True`；
  - summary file: `hygon_tmp/sglang_debug/v3_normal_perf_summary_20260614_132258.csv`。
- V3 stage timing last24 median：
  - 1024：K1 `0.9985 ms`，K2 `0.105 ms`，K3 `0.887 ms`，no-tail barrier `0.0645 ms`，reduce `0.048 ms`，total `2.1225 ms`；
  - 4096：K1 `2.760 ms`，K2 `0.214 ms`，K3 `2.5375 ms`，no-tail barrier `0.1255 ms`，reduce `0.168 ms`，total `5.850 ms`。
- K1 direct floor refresh：
  - 1024 direct K1 `0.9702 ms`，full-stage K1 `0.9985 ms`，remaining full/direct gap about `0.03 ms`；
  - 4096 direct K1 `2.7958 ms`，full-stage K1 `2.760 ms`，within measurement noise；K1 4096 full-stage orchestration gap is no longer a meaningful blocker.
- K3 ASM-pack5 split refresh：
  - 1024：rowptr-all-zero `0.47696 ms`，local-rowptr `0.50682 ms`，staged-remote-only `0.81128 ms`，staged-rowptr `0.82816 ms`；
  - 4096：rowptr-all-zero `1.50164 ms`，local-rowptr `1.55039 ms`，staged-remote-only `2.45529 ms`，staged-rowptr `2.48156 ms`；
  - V3 ASM-pack5 all-zero/GEMM floor is near C pure/groupgemm floor, but staged-rowptr adds about `0.35 ms / 0.98 ms` over all-zero at 1024/4096.
- A raw C rowptr split command was accidentally run with backend `normal`/raw C path and produced slow staged-rowptr numbers around `7.7-7.9 ms` for 4096; this is marked as wrong-path/no-conclusion and must not be used as a champion K3 result.
- DCU KB / Flux GEMM+RS retrieval reconfirmed the high-level pattern: keep communication semantics in the GEMM epilogue writeback backend and use mode gates for A/B. Current V3 K3 ASM-pack5 already follows that epilogue-store pattern, so the next useful work is a narrow store/schedule evidence pass, not another C fallback rewrite.
- Current normal conclusion:
  - V3 normal no-tail is now clearly faster than original staged fused at both gate sizes.
  - K1 4096 has been pulled close to direct floor by dynamic producer CTA; 1024 K1 has only a small residual.
  - K3 normal no-tail remains the largest fused-vs-pure gap because of remote/scattered rowptr combine store data-path cost. Previous PMC showed the issue as `TA_BUSY` / `TCP_TA_DATA_STALL`, not extra GEMM work or missing vector-store width.

## 2026-06-14 normal K3 terminal store wait A/B rejected

- Static compare of V3 `K3COMBINE_PACK5.s` and original `K3COMBINE.s` showed the staged no-tail store macro and no-tail H0/H1 store schedule are structurally identical:
  - same `K3_STORE_STAGED_HALF` rowptr load + LDS read + `s_waitcnt vmcnt(0)` + `s_waitcnt lgkmcnt(0)` + four `global_store_dwordx4` stores;
  - same H0 stage/barrier/store/barrier then H1 stage/barrier/store/final `s_waitcnt vmcnt(0)` schedule.
- DCU KB query for waitcnt rules reconfirmed that `vmcnt`/`lgkmcnt` are required before consuming global/LDS load results; it did not provide a source-backed reason to remove barriers or load-consumer waits. A narrow terminal-wait A/B was still tested because the final no-tail `s_waitcnt vmcnt(0)` is after the last global stores and before branch-to-end.
- A/B: comment out only the final no-tail terminal `s_waitcnt vmcnt(0)` after `K3_STORE_STAGED_HALF 1024, 1024` in isolated V3 PACK5 code object.
  - A/B code object sha `9d34af482f09f12b68e68713df16fee218a68ee9fbc993fee8d976e083ae1087`;
  - 1024 direct split: staged-rowptr median average rank `0.8025 ms` vs retained recent `0.8282 ms`, a small positive signal;
  - 4096 direct split: staged-rowptr median average rank `2.5716 ms` vs retained recent `2.4816 ms`, clear regression on the main normal gate size.
- Decision: reject and revert. Production K3 PACK5 `.co` restored to retained sha `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`.
- Conclusion: terminal store wait removal is not a viable normal K3 optimization. With store modifier, deferred LDS wait, dest-sort, CUS, rowptr-resource and terminal wait all rejected, next evidence step is SQTT/tooling triage if available; otherwise normal K3 no-tail source-level store/wait tuning should be considered near ceiling for the current contract.

## 2026-06-14 normal K3 SQTT/tooling triage

- Remote container `sglang_megamoe` tool check:
  - `hipprof` is available at `/opt/dtk/bin/hipprof`;
  - `hipprof -h` exposes HIP/HSA/RCCL traces, PMC read/write/type and `--codeobj-analyze`;
  - no `--sqtt` option is exposed by this DTK `hipprof`;
  - `stat_stall`, `stat_valu`, `rocprof`, and `rocprofv3` are not in PATH;
  - DTK search only found Perfetto library support, not an exposed SQTT/stat command.
- Conclusion:
  - K3 normal remote-store profiling is degraded to timing + PMC + code object/ISA on this environment.
  - Because PMC already showed stable instruction counts with `TA_BUSY/TCP_TA_DATA_STALL` increases, and because repeated source-level store/wait/locality A/Bs did not retain, current K3 no-tail ASM-pack5 should be treated as near the practical source-level ceiling under the existing rowptr combine contract.
  - Further K3 normal performance work should require either new tooling evidence, a higher-level scheduler/row emission contract change with a clear hypothesis, or a tail-reduce/no-tail semantics change. Do not continue small store modifier/waitcnt/locality guesses.

## 2026-06-14 normal retained champion sanity after rejected K3 A/B

- After reverting terminal-wait A/B, remote K3 PACK5 `.co` sha is back to retained `c2c8888560bd455fe9615fbe27893b42b62d9ba056dee26f2360b2914e6a2b9d`.
- V3 normal no-tail retained sanity command used `RUN_ORIG_STAGE=0 RUN_V3=1 RUN_STAGE_TIMING=1 WARMUP=3 REPEAT=5 ITERS=2`:
  - 1024 tokens: `2.2593 ms`, correctness `True`, min `2.2342 ms`;
  - 4096 tokens: `6.1027 ms`, correctness `True`, min `6.0039 ms`;
  - summary file: `hygon_tmp/sglang_debug/v3_normal_perf_summary_20260614_134330.csv`.
- Interpretation:
  - The rejected K3 terminal-wait A/B did not contaminate retained champion after rebuild.
  - Timing is in the current normal champion band; minor 1024/4096 variation is consistent with repeated-rank startup/barrier jitter and K3 remote-store variability seen in prior runs.
  - Current normal no-tail source-level store/window tuning has no retained improvement beyond dynamic K1 producer. Next normal-only performance work must either be higher-level scheduling/row emission with a clear contract, or move to remaining feature work once normal gate is accepted.

## 2026-06-14 normal K3 scheduler / row-emission feasibility check

- Production K1 ASM-pack5 row emission is per-local-expert atomic row allocation:
  - route scanners split source-rank routes across wg0 CTAs;
  - valid target routes reserve row slots with `global_atomic_add`;
  - K1 writes `output_index`, `row_x_ptrs`, `route_weights`, `row_combine_ptrs`, and `m_indices` for the reserved row;
  - source comment explicitly says row order is intentionally nondeterministic and `output_index` carries the mapping.
- A production row reordering by destination rank is therefore not a local K3 tweak. It would require changing K1 allocation strategy so `act_fp8/act_scale/m_indices/route_weights/row_combine_ptrs` stay aligned, likely needing either:
  - per-expert/per-destination counts and prefix offsets before row emission; or
  - fixed per-destination row segments, which risks overflow/skew or wastes K3 rows; or
  - an extra compaction/sort pass, which violates the no-extra-kernel/no-extra-pass preference.
- Existing split diagnostic coverage:
  - `make_dest_sorted_rowptrs()` sorts rowptrs inside each expert group only, without reordering activations/m_indices, so it is not a correctness path;
  - however it isolates the store-address locality hypothesis, and that hypothesis already regressed badly at 4096 (`staged_remote_only 2.50 ms -> dest_sorted 4.00 ms`, `staged_rowptr 2.52 ms -> dest_sorted 3.80 ms`).
- New 4096 rowptr distribution probe:
  - rows `29696`, active rows average `24576`, local rows average `2970`, remote rows average `21606`;
  - per 16-row chunk, average unique destination ranks `~4.79`, max same-rank rows `~6.47`, contiguous destination row-index pairs only `~0.18`;
  - the rowptr stream is highly fanned out, but naive locality sorting has already shown negative timing, so fanout alone is not enough evidence for a K1 row-emission rewrite.
- Decision:
  - Do not implement production row-emission reorder now.
  - Any future scheduler rewrite must first prove a positive signal in a more faithful diagnostic that reorders `act_fp8/act_scale/m_indices/rowptr` together, and must budget K1 row allocation overhead; current evidence does not justify the risk.

## 2026-06-14 normal no-tail performance closure

- Current retained normal no-tail champion is K1 ASM-pack5 dynamic staged-copy producer plus K3 ASM-pack5 no-tail, with V3 normal gate defaulting to that isolated path.
- Latest same-session correctness/perf refresh:
  - 1024 tokens: original staged fused `2.4134 ms`, V3 normal `2.3054 ms`, speedup `1.0468x`, correctness passed.
  - 4096 tokens: original staged fused `6.6033 ms`, V3 normal `6.0640 ms`, speedup `1.0889x`, correctness passed.
- K1 normal is no longer the main blocker:
  - 1024 direct K1 `0.9702 ms`, full-stage K1 `0.9985 ms`, remaining gap about `0.03 ms`.
  - 4096 direct K1 `2.7958 ms`, full-stage K1 `2.760 ms`, within noise.
- K3 normal no-tail remains the dominant pure-vs-fused delta:
  - 1024 K3 ASM-pack5 all-zero/local/staged-remote/staged-rowptr `0.477/0.507/0.811/0.828 ms`.
  - 4096 K3 ASM-pack5 all-zero/local/staged-remote/staged-rowptr `1.502/1.550/2.455/2.482 ms`.
  - PMC/diagnostics point to remote/scattered rowptr combine store data path (`TA_BUSY` / `TCP_TA_DATA_STALL`), not extra GEMM work, missing vector store, or K1/full-stage orchestration.
- Rejected or closed normal no-tail directions include C/aicc scalar-store fallback, store modifier, deferred wait, terminal wait removal, dest-sort/row-emission reorder, CUS changes, rowptr-resource toggles, no-tail extra signal, and staged-vector-store production A/B without stability.
- Remote tooling currently exposes `hipprof` timing/PMC/codeobj analysis but no usable SQTT/stat interface, so further no-tail store-window tuning would be low evidence. The next normal work item should resume tail-reduce correctness/precision from the existing device post-hoc reduce / stage compare diagnostic break point.

## 2026-06-14 normal tail device post-hoc reduce resumed

- Cleared one stale orphan Python multiprocessing process in the remote `sglang_megamoe` container before rerunning tail diagnostics; 8 DCUs were otherwise idle.
- The previous launcher/NCCL abort did not reproduce after cleanup. A minimal 8-rank `k3_v3_tail_stage_compare.py` run with `K3_V3_TAIL_STAGE_ITERS=1`, `K3_V3_TAIL_STAGE_ORDER=asm_first`, `K3_V3_TAIL_STAGE_ZERO_COMBINE=1`, and `K3_V3_TAIL_STAGE_KEEP_GOING=1` printed valid `rank_stats`.
- In this 1-iter `asm_first` resume run:
  - no rank reported nonfinite values;
  - only small V3-vs-ASM differences appeared on two ranks, `max_abs≈0.00223-0.00226`, still below the `0.0035` threshold;
  - `first_y_minus_device_combine_at_diff` and `second_y_minus_device_combine_at_diff` were `0.0` for all ranks, so device-side `reduce_local_combine(... invalidate_before_read=True)` agreed with each tail `y` at the compared coordinate;
  - Python `combine_reduce_py` values differed from `y` at the same coordinates, confirming the earlier suspicion that the direct Python combine view is not a reliable root-cause oracle for tail.
- Conclusion: the tail diagnostic chain is usable again, and the next useful experiment is repeated `v3_only` / order-swapped stage compare using device-side reduce fields, not Python combine-buffer summation.

## 2026-06-14 normal tail synchronized zero diagnostic

- Patched diagnostic-only `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare.py` so `K3_V3_TAIL_STAGE_ZERO_COMBINE=1` synchronizes after `combine.zero_()` and again after the cross-rank `dist.barrier`. This avoids treating an asynchronous local zero kernel as a producer-side correctness signal.
- DCU KB retrieval for the current signal/visibility question degraded: parallel local queries hit CUDA OOM in the reranker, and a single CPU/disabled-CUDA query timed out. The retained prior KB finding still applies: use system-scope fence plus release/acquire signal for cross-rank GPU memory visibility, and do not relax wait/fence ordering without measured evidence.
- Re-ran 8-rank `v3_only` 3-iter synchronized-zero stage compare at 1024 tokens. Result: nonfinite values still reproduce after iter1, so async zero is not a sufficient root cause.
- Parsed key fields from `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_resume_v3_only_3iter_synczero_20260614.log`:
  - iter1: no nonfinite values in either V3 call.
  - iter2: first call nonfinite counts by rank `[0,0,0,0,0,160,96,0]`; second call `[2,32,5,9,0,0,0,0]`.
  - iter3: first call clean; second call rank4 has 5 nonfinite values.
  - For every nonfinite case, `nonfinite_*_combine_reduce_device` matches the tail `y` nonfinite count, and `*_y_minus_device_combine_at_diff` is zero or NaN at the same location.
- Conclusion: the failure is not only Python combine view, and not only unsynchronized diagnostic zero. The next split must distinguish whether the normal C tail GEMM writes NaNs into combine, or whether peer-store/signal/reduce ordering lets the reducer read stale/invalid combine data.

## 2026-06-14 normal tail producer-vs-reducer split

- Added diagnostic-only `K3_V3_TAIL_STAGE_V3_EXTERNAL_REDUCE=1` to `k3_v3_tail_stage_compare.py`. In this mode V3 normal uses the C raw no-tail combine producer with `use_asm_pack5=False`, then runs the existing external `rank_barrier(acquire_after_wait=True) + reduce_local_combine(invalidate_before_read=True)`.
- 8-rank 1024-token `v3_only` / `zero_combine=1` / synchronized-zero / external-reduce run completed 3 iterations with no nonfinite values:
  - iter1/iter2/iter3 all had `nonfinite_first = nonfinite_second = nonfinite_diff = 0` on all ranks;
  - device-side reduce also reported zero nonfinite values on every rank;
  - done counters stayed zero as expected because this mode bypasses tail done/signal.
- Conclusion: the normal C producer can write stable combine data under the same K1/K2/rowptr setup. The intermittent normal tail failure is now localized to the in-kernel tail signal/reducer path, especially the appended reducer-block schedule or the done/signal acquire chain, rather than the GEMM math or rowptr coverage itself.

## 2026-06-14 normal tail inline-reduce diagnostic

- Pulled and parsed `hygon_tmp/sglang_debug/k3_v3_tail_stage_compare_inline_reduce_3iter_synczero_20260614.log`.
- This run set `MEGAMOE_DCU_V3_TAIL_INLINE_REDUCE=1`, so the normal tail launch used `reduce_blocks=0` and only the completion owner CTA ran the existing inline reduce branch after peer signals.
- Results:
  - iter1/iter2 had no nonfinite values, but all ranks showed large first-call `y - device_posthoc_reduce` deltas: about `0.247-0.285` in iter1 and `0.067-0.081` in iter2. The second call in each iter matched post-hoc device reduce.
  - iter3 produced second-call nonfinite values on ranks 0/4/5, and device-side post-hoc reduce saw matching or larger nonfinite counts on the same ranks.
  - `done_counter[0]` reached the expected `512` in every call; `done_counter[1]` recorded a completion owner id, so this is not simply an incomplete local grid count.
- Conclusion:
  - Appended reducer blocks are not the sole cause. Even owner-CTA inline reduce can read a different combine view than a later device post-hoc reduce.
  - The next fix should target the done/signal visibility chain: peer waits should be acquire loads, the final completion owner should acquire the done-counter release sequence before signaling, and the signal/reduce branch should be entered block-uniformly instead of by per-thread relaxed global loads.

## 2026-06-14 normal tail C/aicc route stopped

- User direction: stop spending optimization/debug time on the C/aicc normal tail implementation for now and prepare to abandon that route.
- Current evidence supports that pivot:
  - V3 normal C no-tail producer plus external `rank_barrier + reduce_local_combine` is clean, so K1/K2/rowptr/producer math are not the core blocker.
  - V3 normal C tail with appended reducer blocks still produces intermittent nonfinite/mismatch.
  - Disabling appended reducer blocks and using the completion-owner inline reduce still produces mismatch/nonfinite, so the issue is not isolated to reducer-block scheduling.
  - Peer-wait acquire-only is insufficient; stronger completion-owner acquire / block-uniform branch attempts hit aicc local memory limits around the tail specialization.
- Decision:
  - Do not continue tuning or repairing the C/aicc raw normal tail path as the production route.
  - Keep C raw tail only as a diagnostic/fallback reference.
  - Move normal tail production work to an isolated ASM-pack5 tail-reduce route: copy original `K3COMBINE_TAILREDUCE` ASM, patch V3 plain-pack5 weight/SrdA address math, and preserve the original ASM tail signal/wait/reduce/store schedule.

## 2026-06-14 normal 8192 symm warm-up A/B

- User asked whether the V3 normal symm warm-up helps at 8192 tokens and whether the no-warm-up path still works.
- Remote 8-card V3 normal no-tail A/B used `TOKENS_LIST=8192 RUN_ORIG_STAGE=0 RUN_V3=1 RUN_STAGE_TIMING=1 WARMUP=2 REPEAT=3 ITERS=2`.
- Both paths passed correctness at `num_tokens_per_rank=8192`, with `sym_buffer=0.420 GiB` and `route_scratch=4.013 GiB` reported by the harness.
- Results:
  - warm-up on: fused median `12.5974 ms`, min `12.5652 ms`, baseline median `17.3312 ms`, speedup vs baseline `1.3758x`; last24 stage medians total/K1/K2/K3/no-tail-barrier/reduce `12.417/5.4375/0.412/5.8905/0.2755/0.336 ms`.
  - warm-up off (`MEGAMOE_DCU_V3_SYMM_WARMUP_ALLOC=0`): fused median `12.7807 ms`, min `12.4317 ms`, baseline median `17.3551 ms`, speedup vs baseline `1.3579x`; last24 stage medians `12.5595/5.4705/0.413/6.0170/0.2155/0.336 ms`.
- Interpretation:
  - At 8192, warm-up is not required for correctness; the no-warm-up path also runs successfully even with `route_scratch` slightly above 4 GiB.
  - Warm-up still has a small positive median signal, about `1.45%` end-to-end and about `0.033 ms` on K1 last24 median in this short run, but the effect is much smaller than the earlier 4096 bad-allocation case.
  - This does not prove a true 4 GiB addressing limit is solved by warm-up; it only shows the current 8192 allocation/test shape does not fail either way.

## 2026-06-14 normal K3 faithful row-emission diagnostic feasibility

- User asked to reconsider the higher-level row-emission / combine-contract option but not implement it yet.
- Static contract check:
  - `large_opt.py` passes K1 outputs `l1_out/route_weights/m_indices/output_index/row_combine_ptrs` into K2/K3; K2 produces row-aligned `act_fp8/act_scale`.
  - `K3_fused/k3_fused.py` consumes `act_fp8/act_scale/m_indices/row_combine_ptrs`; K3 has no independent knowledge of `output_index`.
  - K1 compact route emission currently reserves rows with per-local-expert atomics and writes `output_index`, `row_x_ptrs`, `row_x_scales`, `route_weights`, `m_indices`, and `row_combine_ptrs` for the same row.
- Therefore a production reorder is not a K3-only patch. Any row order change must keep these row-indexed arrays aligned: `l1_out`, later `act_fp8/act_scale`, `m_indices`, `route_weights`, `row_combine_ptrs`, `output_index`, and local stats/count metadata.
- K3 grouping constraint:
  - K3 groupgemm expects rows to remain grouped by local expert. A global sort by destination rank would interleave experts and break or complicate tile/expert mapping.
  - The only plausible safe reorder scope is inside each local expert segment, for example bucket by destination rank and combine row index while preserving the expert-major layout.
- Existing rowptr-only dest-sort does not prove production behavior because it reorders only store addresses, not the activations and metadata. It was still useful as a locality probe and showed no positive signal or a regression, so a faithful diagnostic needs a higher bar.
- DCU KB / Flux guidance supports the general concept only at scheduler/epilogue-contract level: communication destination should be known to the GEMM scheduler/epilogue, rank-aware swizzles can help, and fused communication behavior must be capability/mode gated. It does not justify blind post-sort or K3-only changes.
- Recommended next step if attempted:
  - build a diagnostic-only sidecar under `hygon_tmp`, after K1+K2, that creates an expert-local permutation by `(dest_rank, combine row)` and applies it consistently to `act_fp8`, `act_scale`, `m_indices`, `route_weights`, and `row_combine_ptrs`;
  - run K3 direct ASM-pack5 no-tail against the permuted tensors and compare correctness plus K3-only timing at 1024/4096;
  - do not count the extra permutation kernel/copy as production, only use it as an upper-bound signal.
- Difficulty/risk:
  - sidecar diagnostic: medium, good isolation, no production risk;
  - production C/prebuild row allocation: medium-high and may slow K1;
  - production K1 ASM route emit bucketing with per-expert/per-dest prefix: high, because it touches row reservation and metadata emission in assembly;
  - graph/uneven/tail follow-up after row-order changes: high.
- Expected upside is limited by remote-store ceiling: 4096 single-peer contiguous remote was still around `2.31 ms`, while staged-rowptr was around `2.48 ms`; the reachable gain from fanout/scatter cleanup is likely a fraction of the full pure-vs-fused gap, not the whole `~1 ms`.

## 2026-06-14 normal K3 combine-layout / reduce-contract feasibility

- User asked to analyze option 3 more carefully before implementation. This is the broader contract/layout path, not the expert-local row-emission reorder alone.
- Current DCU staged contract:
  - K1 emits `row_combine_ptrs[row] = combine + (topk_slot * num_max_tokens_per_rank + token_idx) * hidden`, so the symmetric combine buffer is slot-major: `[topk_slot][token][hidden]`.
  - K3 no-tail writes BF16 GEMM output through `row_combine_ptrs`.
  - no-tail then uses external `rank_barrier + reduce_local_combine`, whose reduce loop reads exactly the same slot-major `partial_row = topk_slot * max_tokens + token_idx`.
  - tail ASM setup passes `asm_reduce_combine` plus `asm_reduce_slot_stride_vec = max_tokens * hidden_vecs`, so tail-reduce also assumes slot-major combine layout.
  - K2 mainly depends on row alignment and zero/nonzero rowptrs; exact combine layout is hidden behind K1 rowptrs until reduce.
- DCU KB / Flux / DeepEP guidance:
  - communication metadata should be part of the hot-path scheduler/epilogue contract, not an unstructured post-pass;
  - reduce-scatter style semantics are most naturally placed in epilogue/writeback when the final layout is distributed;
  - this supports a K3 epilogue/reduce-contract change, but does not justify a blind rowptr sort or layout rewrite without an upper-bound probe.
- Option 3 splits into two different implementation classes:
  1. Keep current slot-major layout and fuse no-tail reduce into isolated ASM-pack5 K3. This is lower risk because K1 rowptrs, combine layout, and tail stride remain compatible. Expected upside is mainly removing the external barrier/reduce launch and memory pass, roughly `0.1-0.3 ms` on 1024/4096 based on current stage timing. It does not remove the main remote/scattered store cost.
  2. Change combine layout to rank-major/per-destination buckets. This could improve rowptr store locality and recover part of fanout/scatter overhead, but it requires K1 metadata/rowptr emission, K3 store, no-tail reduce, tail ASM stride, graph, and uneven tokens to change together. Existing remote ceiling data bounds likely gain: 4096 single-peer contiguous remote was about `2.31 ms` while staged-rowptr was about `2.48 ms`, so layout may recover only around `0.2 ms` unless it also changes remote-write behavior.
- Not attractive for now:
  - direct remote accumulation into `y` with atomics/adds: high precision/order risk and likely poor remote atomic performance;
  - remote contiguous staging plus later transpose/reduce as production: may lower K3 store time but adds another memory pass or a much larger fused reducer, conflicting with no-extra-kernel/no-extra-pass goals.
- Recommended order:
  1. Finish/validate ASM-pack5 tail-reduce migration first; it proves V3 pack5 can reuse the original ASM signal/reduce schedule.
  2. Try a no-tail fused-reduce ASM-pack5 diagnostic under the existing slot-major layout if e2e overhead from external reduce remains important.
  3. Only after that, and only with a diagnostic upper-bound, consider rank-major/bucket combine layout. Production should require a >3-5% K3 direct gain at 4096 before touching K1 row allocation or graph/uneven contracts.

## 2026-06-14 normal K3 rank-bucket combine-layout upper-bound timing

- User asked to ignore correctness temporarily and measure whether making remote combine writes more contiguous in actual K3 would be worth a larger contract rewrite.
- Added diagnostic-only rowptr modes to `hygon_tmp/sglang_debug/bench_k3_ll_rowptr_modes.py`; production `megamoe/` code was not changed.
  - `staged_rank_bucket_remote_only`: keep actual K3 rows/experts/tiles and original row order, but remap every remote rowptr to compact rows inside its original destination rank's combine buffer.
  - `staged_rank_bucket_full`: same idea for all active rows including local rank.
  - `*_sorted`: additionally sort rowptrs within each expert group by destination rank / compact row; correctness intentionally ignored.
- 8-card remote timing, backend `asm_pack5`, plain V3 L2 layout, no correctness checks:
  - 1024 tokens:
    - `staged_rowptr`: `0.7948 ms`
    - `staged_remote_only`: `0.8014 ms`
    - `staged_rank_bucket_remote_only`: `0.7204 ms`
    - `staged_rank_bucket_remote_sorted`: `0.7795 ms`
    - `staged_remote_contig_peer`: `0.7423 ms`
  - 4096 tokens:
    - `staged_rowptr`: `2.5170 ms`
    - `staged_remote_only`: `2.5108 ms`
    - `staged_rank_bucket_remote_only`: `2.4307 ms`
    - `staged_rank_bucket_full`: `2.4358 ms`
    - `staged_remote_contig_peer`: `2.3065 ms`
    - `staged_rank_bucket_remote_sorted`: `3.8842 ms`
    - `staged_rank_bucket_full_sorted`: `3.7166 ms`
- Interpretation:
  - Rank-major compact combine layout has a real positive signal when row order is kept: about `0.074 ms / 9.4%` at 1024 and about `0.081 ms / 3.2%` at 4096 K3-only.
  - The 4096 rank-bucket gain recovers only about `0.08 ms` of the `staged_remote_only -> single-peer contiguous` gap of about `0.20 ms`; the remaining cost is still remote-write path / topology behavior, not only combine row contiguity.
  - Sorting rowptr addresses inside expert groups is strongly negative. That suggests a future production rewrite should not assume “rank-sorted row emission” helps unless it also preserves GEMM/store scheduling properties; naive address-order locality can hurt.
- Decision:
  - The timing signal is useful but not large enough by itself to justify a production-level combine buffer layout rewrite touching K1 row allocation, K3 store, external reduce, tail ASM, graph, and uneven tokens.
  - Keep this as an upper-bound data point. Prioritize lower-blast-radius work first, especially existing-layout no-tail fused reduce / ASM-pack5 tail reduce, unless future e2e budget proves `~0.08 ms` K3-only is worth the contract churn.
- Local pulled artifacts:
  - `hygon_tmp/sglang_debug/rank_bucket_pull_20260614/k3_rank_bucket_1024_20260614_171629.json`
  - `hygon_tmp/sglang_debug/rank_bucket_pull_20260614/k3_rank_bucket_4096_20260614_171724.json`
  - `hygon_tmp/sglang_debug/rank_bucket_pull_20260614/k3_rank_bucket_full_4096_20260614_171814.json`

## 2026-06-14 normal K3 compute/store wave-specialization feasibility

- User clarified the intended idea: dedicate compute waves to MMAC and store waves to remote combine stores so K3 can overlap math and remote-store drain inside one kernel.
- Follow-up interpretation: current normal K3 fused-vs-pure growth should be treated as mostly hard overhead under the current DCU staged contract and resource shape, not as an obvious remaining codegen bug. The evidence is:
  - the K3 GEMM core is already close to the pure/all-zero/local floor;
  - resource usage is near the code-object limit (`VGPR=255`, `SGPR=102`, `LDS=64KB`);
  - remote/scattered rowptr modes raise `TA_BUSY` and `TCP_TA_DATA_STALL` while instruction counts remain stable;
  - rank-bucket layout only recovered about `0.08ms / 3.2%` at 4096 K3-only.
  This does not prove an absolute hardware lower bound, but it means small local store/wait changes should be considered low-yield unless backed by new PMC/ISA/SQTT or synthetic-overlap evidence.
- DCU KB retrieval found Hygon/CK GEMM examples that use LDS double buffering to overlap current GEMM with next data movement. This supports a producer-consumer pipeline concept, but it is about structured LDS buffering and barriers, not free same-kernel concurrency.
- Follow-up resource check confirms the user's interpretation: current K3 already uses a load/compute pipeline where A/B global data is staged into LDS with double-buffer-style scheduling. The code object declares 64KB LDS, and the ASM explicitly clamps LDS at 65536 bytes while using `buffer_load_* ... lds` plus dense `ds_read` for the MMAC input path.
- The existing K3 epilogue can reuse LDS after compute to stage/store C, but compute/store overlap would require C store buffers to be live at the same time as the next tile's A/B LDS buffers. That is the practical LDS pressure issue; it is not just a matter of adding one store wave.
- Current overlap shape:
  - A/B operands use a global-to-LDS pipeline with double-buffer-style scheduling, matching the CK/Hygon GEMM pattern of prefetching future A/B tiles while computing on current LDS tiles.
  - The pipeline is cooperative/block-tile or wave-tile style. There is no evidence in the current ASM of a permanent load-wave vs compute-wave role split for K3; the same kernel waves participate in load/LDS staging and MMAC phases separated by `s_waitcnt`/`s_barrier`.
  - Inside compute, LDS reads feed VGPR operand fragments and MMAC macros run under raised priority (`s_setprio 1`), so there is LDS-read/MMAC scheduling overlap at instruction level.
  - C accumulator writeback/combine is still an epilogue phase after GEMM for the tile, not overlapped with the next tile's compute.
- Current V3 K3 ASM-pack5 resource shape is tight:
  - `.amdhsa_next_free_vgpr 255`
  - `.amdhsa_next_free_sgpr 102`
  - `.amdhsa_group_segment_fixed_size 65536`
  - wavefront size is 64.
- Current `K3_STORE_STAGED_HALF` already stages C through LDS, loads rowptrs, waits on `vmcnt/lgkmcnt`, then issues vector remote stores. It is an epilogue store schedule, not a compute-wave/store-wave overlap pipeline.
- Key constraint: on AMD/Hygon a wave specialized as a store wave inside the same kernel still reserves the kernel's maximum VGPR/SGPR/LDS footprint. A store-only role does not become cheap unless the kernel is split into a separate code object, which would violate the no-extra-runtime-kernel direction for production.
- Store waves cannot read compute-wave VGPR accumulators directly. Compute waves would need to spill/pack finished C chunks into LDS/global ring buffers, then signal store waves. Therefore the production design would need double-buffered C staging plus block-uniform ready/reuse barriers.
- A useful overlap requires independent compute work after a C chunk is ready. If the current workgroup computes one tile and then exits, there is nothing meaningful to overlap with store drain. A real implementation likely needs either a persistent/multi-tile CTA loop or an N/hidden chunked schedule where store wave drains chunk N while compute waves produce chunk N+1.
- Potential upside is larger than rank-bucket layout: current 4096 K3 staged-rowptr vs local/all-zero split leaves roughly a 0.9ms class remote-store penalty. Perfect overlap is impossible, but a successful design could plausibly recover more than the ~0.08ms rank-bucket upper-bound. This is unproven and must be measured.
- Risk is also high: dedicating one wave to store can reduce compute wave count, extra C double-buffering consumes already-full LDS, and remote stores may contend with next-tile A/B global loads. A naive rewrite can be neutral or slower.
- Recommendation:
  - Do not patch production ASM directly first.
  - Add a `hygon_tmp` synthetic overlap probe that approximates K3 resource pressure, MMAC loop length, LDS C-buffer staging, rowptr loads, and remote stores.
  - Compare serial compute->store against compute-wave/store-wave overlap at 1024/4096, ideally with the same rowptr distributions.
  - Only if 4096 K3-only shows clear >5% or >0.2ms signal should we attempt an isolated V3 ASM-pack5 persistent/multi-tile CTA rewrite.

## 2026-06-14 LL normal-tech transfer / overlap restart

- User direction: normal tail-reduce is postponed to the end; next work should focus on LL V3 optimization, especially whether normal no-tail optimization ideas can transfer to LL and whether LL K3 has enough resource headroom for DCU-friendly remote-store overlap.
- Re-read plan/progress and local code:
  - Current retained LL path already includes K1 parallel/expert-loop stage-copy, K3 rowptr register-prefetch, and K3 epilogue cleanup.
  - Rejected LL directions remain rejected: store modifier, deferred rowptr wait, CUS32, dest-sort, rowptr raw-buffer resource, storex4 lane-pair, tile skip / changed completion semantics, and blockM 16/48/64.
  - Current LL K3 raw C kernel prefetches `row_combine_ptrs` before MMAC, keeps row addresses in registers, then does rowptr vector store in epilogue; it does not overlap C-store drain with later tile compute.
- DCU KB refresh again points to DeepEP low-latency combine overlap using fine-grained `comp_signal`, `block_m`, and `threshold`; the portable lesson is chunk/tile readiness, not whole-expert waiting or extra post-pass kernels.
- Transfer assessment:
  - Normal K1 dynamic/stage-copy producer ideas are already mostly reflected in LL K1 parallel/expert-loop stage-copy; further LL K1 work should only follow new PMC or stage timing evidence.
  - Normal K3 rank-bucket/row-ordering gave too small a signal and LL dest-sort was neutral; not a first-choice LL production change.
  - Normal compute/store wave-specialization was blocked by 255 VGPR + 64KB LDS; LL K3 no-tail has recorded lower register pressure (`VGPR=153`, `private=0`), so it is the only normal-derived high-upside idea still worth a targeted LL diagnostic.
- Guardrail for next code/probe:
  - Do not change LL K3 tile completion semantics or tail-reduce semantics.
  - First refresh LL 128 no-tail pure/local/remote/staged timing on the current tree.
  - If the delta still matches prior data, use `hygon_tmp` or an env-gated LL K3 branch to test overlap-shaped store scheduling; keep production path untouched unless correctness and timing both support the branch.

## 2026-06-14 LL K3 32/128 split and rank-bucket refresh

- Rebuilt V3 LL raw K1/K3 only on remote 11 node (`DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND=ll`, `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND=ll`, V2 disabled) and refreshed `bench_k3_ll_rowptr_modes.py` through the `sglang_megamoe` container.
- 128 tokens, `ll_block_m=32`, average rows/rank `active=768`, `local=32.75`, `remote=735.25`:
  - `pure_contiguous`: `0.23696 ms`
  - `local_rowptr`: `0.24369 ms`
  - `rowptr_all_zero`: `0.24262 ms`
  - `staged_remote_only`: `0.36527 ms`
  - `staged_rowptr`: `0.36764 ms`
  - diagnostic rank-bucket modes ignoring correctness: `remote_only=0.26803 ms`, `full=0.26870 ms`, `remote_sorted=0.26442 ms`.
- 32 tokens, `ll_block_m=32`, average rows/rank `active=192`, `local=1.25`, `remote=190.75`:
  - `pure_contiguous`: `0.23682 ms`
  - `local_rowptr`: `0.24436 ms`
  - `rowptr_all_zero`: `0.24252 ms`
  - `staged_remote_only`: `0.24458 ms`
  - `staged_rowptr`: `0.24451 ms`
  - diagnostic rank-bucket modes ignoring correctness: `remote_only=0.24152 ms`, `full=0.24139 ms`, `remote_sorted=0.24161 ms`.
- Interpretation:
  - LL 32 is already dominated by the fixed K3 kernel floor; staged remote write adds almost no visible cost over local/zero rowptr, so overlap work should not target 32 first.
  - LL 128 has a real remote/scattered combine-store penalty: staged rowptr is about `0.124 ms` slower than local rowptr and about `0.131 ms` slower than pure contiguous.
  - Rank-bucket at 128 is a strong upper-bound signal (`~0.10 ms` faster than staged rowptr), much larger than the normal 4096 rank-bucket signal. This does not preserve correctness because it rewrites rowptr destinations, but it proves LL 128 is highly sensitive to remote destination layout / writeback shape.
  - Normal K1 producer improvements have already mostly transferred into LL K1; normal K3 resource-heavy compute/store wave specialization is not directly portable, but LL K3 has enough code-object headroom (`VGPR=153`, no spill in prior ISA check) to justify a targeted 128-only overlap/writeback probe.
- Next optimization focus:
  - Do not spend time on LL 32 K3 unless a change also helps 128 without regressing 32.
  - For LL 128, prioritize a correctness-preserving env-gated or `hygon_tmp` diagnostic that changes only store/writeback scheduling or producer-consumer chunking inside K3; any production promotion must keep current row-index/tile completion semantics and avoid extra runtime kernels.

## 2026-06-14 LL K3 rowaddr wave-shuffle A/B rejected

- Hypothesis:
  - In LL K3, all lanes with the same `ld_row` but different `ld_col` load the same `row_combine_ptrs[logical_row]`; reducing those duplicate rowptr loads with a wave-level shuffle might lower rowptr/TA pressure without changing row-index or store semantics.
- Change tested:
  - Added a build-time diagnostic gate that made only `ld_col==0` lanes issue `global_load_i64_glc_device`, then broadcast the 64-bit row address within the wave by shuffling low/high 32-bit halves.
  - No production runtime kernel was added, and K1/K2/rowptr/store contracts were unchanged.
- Correctness:
  - 8-card LL no-tail and LL tail correctness passed for 32 and 128 tokens, one iteration each, with `max_abs=0.000244141`.
- Performance:
  - 128 split regressed badly: `staged_rowptr 0.3676 -> 0.4065 ms`, `staged_remote_only 0.3653 -> 0.4013 ms`.
  - 32 split also regressed: `staged_rowptr 0.2445 -> 0.2746 ms`, `staged_remote_only 0.2446 -> 0.2708 ms`.
  - After reverting and rebuilding retained LL raw, 128 short sanity returned to `local_rowptr 0.2437 ms`, `staged_remote_only 0.3666 ms`, `staged_rowptr 0.3690 ms`.
- Interpretation:
  - Duplicate rowptr loads are not the limiting issue. The wave-shuffle form likely reduces useful VMEM parallelism and/or serializes the address-ready path enough that remote stores backpressure more strongly.
  - Do not repeat rowptr-load deduplication via wave shuffle. Future LL 128 work should target the remote writeback/layout/consumer contract rather than rowptr load count.

## 2026-06-14 LL K3 store-burst epilogue A/B rejected

- Hypothesis:
  - LL K3 has lower code-object pressure than normal K3, so an epilogue scheduling variant might improve remote-store overlap by converting all BF16 store fragments for a tile first and then issuing rowptr stores in a burst.
  - The variant preserved K1/K2 contracts, row-index mapping, tile completion semantics, and did not add a runtime kernel.
- Change tested:
  - Temporary build-time gate `DG_BUILD_MEGAMOE_V3_LL_K3_STORE_BURST=1`.
  - In the `row_combine_ptrs != nullptr` epilogue path, store fragments were accumulated in a register `store_cache[kMRepeats][kNRepeats]` before issuing stores.
- Correctness:
  - 32 no-tail/tail correctness passed on the branch with `max_abs=0.000244141`.
  - After reverting the branch and rebuilding retained, 128 no-tail/tail correctness also passed with `max_abs=0.000244141`.
- Performance:
  - 32 split regressed to `staged_rowptr 0.24699 ms` vs retained `~0.2445 ms`.
  - 128 split regressed to `staged_rowptr 0.37899 ms` vs retained `~0.3676-0.3690 ms`.
  - After revert, retained 128 sanity returned to `local_rowptr 0.24413 ms`, `staged_remote_only 0.36576 ms`, `staged_rowptr 0.36745 ms`.
- Interpretation:
  - This store-burst form does not create useful overlap on DCU for the current LL K3 tile shape. It likely increases live ranges/register scheduling pressure or delays store issue enough to worsen backpressure.
  - Do not repeat delayed-store/burst-issue epilogue variants unless new PMC/ISA evidence changes the diagnosis. Remaining LL 128 space is more likely in layout/contract-level chunk readiness or a true producer-consumer overlap design, not in simply reordering stores inside the same epilogue.

## 2026-06-14 LL K3 row-order / layout diagnostic refresh

- Purpose:
  - Check whether the normal K3 row-emission / destination-rank ordering idea transfers to LL before touching K1 row allocation or K3 production code.
  - The diagnostic uses K3-only timing. Sorting rowptrs alone is not correctness-preserving, but it approximates the store-address order a correctness-preserving expert-local activation+metadata permutation would expose to K3.
- 128-token LL result:
  - `pure_contiguous`: `0.236956 ms`
  - `local_rowptr`: `0.243426 ms`
  - `staged_rowptr`: `0.366257 ms`
  - `staged_rowptr_dest_sorted`: `0.367132 ms`
  - `staged_remote_dest_sorted`: `0.365495 ms`
  - `staged_rank_bucket_full`: `0.268205 ms`
  - `staged_rank_bucket_full_sorted`: `0.265346 ms`
- Interpretation:
  - Destination-order sorting by itself is effectively neutral or slightly slower. This argues against a production K1 row-emission reorder whose only goal is to make rowptr issue order rank-sorted.
  - The strong improvement remains the rank-bucket compact layout, which changes where rows land in combine memory. That is a layout/contract-level opportunity, not a K3-only store-order or overlap tweak.
  - For LL 128, the remaining high-upside path is either a real producer-consumer/chunk-readiness design or a combine-layout/reduce-contract rewrite. Small epilogue store ordering changes now have several negative data points.
- Artifact:
  - `hygon_tmp/sglang_debug/ll_row_order_pull_20260614/k3_ll_row_order_refresh_128_20260614_200504.json`.

## 2026-06-14 LL retained formal snapshot after K3 overlap diagnostics

- Current retained V3 LL no-tail after reverting rowaddr-shuffle/store-burst diagnostics:
  - 32 tokens: correct, fused median `0.830619 ms`, min `0.818580 ms`.
  - 128 tokens: correct, fused median `1.034080 ms`, min `1.018639 ms`.
- Stage interpretation:
  - 32 stable iterations show K1 around `0.346-0.350 ms` and K3 around `0.252-0.269 ms`; the fixed floor dominates and K3 remote writeback is not the first priority at this size.
  - 128 stable iterations show K1 around `0.38-0.40 ms` and K3 around `0.37-0.42 ms`; K1 and K3 are comparable. K3 split still shows the remote/scattered rowptr store delta, but this is no longer an obvious one-instruction scheduling problem.
- Optimization implication:
  - Small K3-only source-level changes now rejected or neutral include store modifier, deferred rowptr wait, CUS change, dest-sort, rowaddr wave-shuffle, and delayed store-burst.
  - The strongest remaining K3 signal is rank-bucket compact combine layout (`~0.265-0.268 ms` K3-only at 128) versus real staged rowptr (`~0.366 ms`), but that requires K1/K3/reduce contract work.
  - For near-term LL work, the practical remaining choices are: small K1 fixed-cost cleanup if a new timing clue appears, or explicitly plan a larger combine-layout/chunk-readiness experiment instead of continuing epilogue micro-tuning.

## 2026-06-14 DCU KB refresh for LL K3 overlap direction

- Query focus: Hygon DCU/gfx938 LL GEMM remote rowptr store overlap after store modifier, deferred wait, row shuffle, store burst, CUS, and dest-sort all failed or were neutral.
- Relevant retrieved patterns:
  - DeepEP low-latency combine overlap uses fixed layout and double buffering.
  - Useful overlap is guarded by fine-grained `comp_signal`, `block_m`, and `threshold`, so communication starts when a chunk/tile is ready rather than after a whole expert/path finishes.
  - Layered mode keeps cleanup/readiness counters explicit; it does not rely on implicit ordering between unrelated stores.
- Interpretation for V3 LL:
  - The next credible K3 optimization is not another local store-order tweak. It needs either a production-compatible compact/rank-bucket combine layout or a chunk-readiness protocol that exposes tile completion to a communication/writeback backend.
  - Such a direction touches K1 row allocation, K3 rowptr/combine layout, and no-tail/tail reduce contracts. It should be planned as a larger Phase 6b/7 design item rather than hidden behind a K3-only env gate.

## 2026-06-14 LL K3 layout/chunk-readiness static feasibility

- Existing no-tail contract is slot-major:
  - K1 LL emits `row_combine_ptrs[row] = peer_sections.combine + (topk_slot * num_max_tokens_per_rank + token_idx) * hidden`.
  - K3 only consumes the pointer and writes BF16 rows to that exact destination.
  - `reduce_local_combine` ignores `row_combine_ptrs/output_index`; it directly sums local `combine[topk_slot, token_idx]` for each token.
- This explains the diagnostic split:
  - Destination-order sorting alone is neutral/slower because it does not change the final remote/scattered destination layout.
  - Rank-bucket compact layout is fast because it changes where remote stores land, but it breaks the reduce contract unless a map from compact row back to `(token, topk_slot)` is added.
- LL K3 resource transfer from normal:
  - Normal K3 compute/store wave specialization is blocked by full LDS/VGPR pressure and needs persistent or multi-tile structure.
  - LL K3 has lower recorded pressure (`VGPR=153`, no private segment in prior check), but current LL kernel still computes one tile then stores its C epilogue; there is no later tile compute inside the same CTA to overlap store drain with unless the scheduler becomes persistent/chunked.
  - Therefore a same-tile epilogue store reordering is low-yield; this matches rejected store modifier, deferred wait, wave-shuffle, dest-sort, and store-burst A/B data.
- Feasible next diagnostic:
  - Keep production untouched and implement a `hygon_tmp` sidecar that remaps active rowptrs into per-destination compact combine rows and records a mapping row -> `(token_idx, topk_slot)`.
  - Run K3 against compact rowptrs, then run a replacement mapped reduce to reconstruct `y`.
  - This can estimate whether the `~0.10 ms` K3-only compact-layout gain at LL 128 survives after paying the reduce/mapping cost.
  - Production promotion would require replacing the existing no-tail reduce launch with a mapped reduce variant, not adding an extra runtime kernel. Tail-reduce, graph, and uneven tokens would need separate contract work later.

## 2026-06-14 LL K3 compact mapped-reduce sidecar result

- Added a diagnostic-only sidecar under `hygon_tmp/sglang_debug`:
  - `bench_k3_ll_compact_mapped_reduce.py` builds compact rowptrs from the real V3 LL K1 output and compares old slot-major `K3 + barrier + reduce_local_combine` with compact `K3 + barrier + mapped_reduce`.
  - `ll_compact_mapped_reduce_ext.cu` implements the temporary mapped reduce kernel for the sidecar only; this is not a production runtime path.
- First 32-token smoke initially failed because `combine_token_offset` was called from a device kernel without `__device__`; after marking the offset helpers `__host__ __device__`, the next smoke revealed a stride mismatch:
  - requested `--max-tokens 32`, but `sym_buffer.num_max_tokens_per_rank` was actually 384;
  - using requested max tokens decoded invalid rowptrs and produced `invalid_total=1432`.
  - Fix: use the actual `sym_buffer.num_max_tokens_per_rank` for combine offset decoding and mapping.
- Correctness after the fix:
  - 32 smoke: `max_abs=0`, `invalid_total=0`, `missing_active_total=0`, actual `max_tokens=384`, `compact_rows_avg_rank=192`.
  - 128 stable run: `max_abs=0`, `invalid_total=0`, `missing_active_total=0`, actual `max_tokens=384`, `compact_rows_avg_rank=768`.
- Timing:
  - 32 smoke was very short and noisy, but showed `staged_k3_barrier_reduce ~0.3837 ms` vs `compact_k3_barrier_mapped_reduce ~0.2830 ms`; treat only as a correctness/regression sentinel.
  - 128 stable run did not preserve the old rank-bucket upper-bound signal:
    - `staged_k3_only`: `0.36856 ms` average-rank median, `0.38325 ms` max-rank median;
    - `compact_k3_only`: `0.36344 ms` average-rank median, `0.37651 ms` max-rank median;
    - `staged_k3_barrier_reduce`: `0.42368 ms` average-rank median;
    - `compact_k3_barrier_mapped_reduce`: `0.42222 ms` average-rank median.
- Interpretation:
  - The correctness-preserving compact sidecar only gives about `0.005 ms` K3-only at 128, and almost no end-to-end K3+barrier+reduce gain.
  - This conflicts with the older diagnostic `staged_rank_bucket_full ~0.265-0.268 ms`, so the next task is not productionization yet; first reconcile the layouts.
  - Static comparison suggests the older rank-bucket diagnostic compacted each local rank's active rows independently inside destination combine space, while the correctness-preserving sidecar adds cross-source prefixes so each destination rank receives separate source segments. That may restore correctness but also lose the contiguous writeback pattern that made the upper-bound fast.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/ll_compact_sidecar_smoke_32_20260614_202631.json`
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/ll_compact_sidecar_128_20260614_202907.json`

## 2026-06-14 LL K3 rank-bucket upper-bound corrected

- Follow-up diagnostic:
  - Added a non-correct `compact_collision_k3_only` mode to the compact mapped-reduce sidecar to mimic no-prefix rank-bucket rowptrs.
  - Result at 128 tokens: `compact_collision_k3_only 0.36292 ms`, `compact_k3_only 0.36501 ms`, `staged_k3_only 0.36637 ms`. This did not reproduce the older `~0.265 ms` result.
- Root cause of the older upper-bound:
  - `bench_k3_ll_rowptr_modes.py` used `args.max_tokens or tokens` for combine offset decoding.
  - The actual symmetric buffer stride for requested 128 tokens is `sym_buffer.num_max_tokens_per_rank=384`.
  - After fixing the diagnostic script to use the actual sym-buffer max tokens, 128-token rank-bucket timing became:
    - `pure_contiguous 0.23728 ms`;
    - `local_rowptr 0.24401 ms`;
    - `staged_rowptr 0.36839 ms`;
    - `staged_rank_bucket_full 0.36643 ms`;
    - `staged_rank_bucket_full_sorted 0.36541 ms`;
    - `rank_bucket_invalid_total=0`, `rank_bucket_counts_avg_rank=[96]*8`.
- Interpretation:
  - The previously recorded `staged_rank_bucket_full ~0.265-0.268 ms` was a diagnostic artifact caused by max-token stride mismatch, likely reducing/warping the effective valid writeback footprint.
  - Correctness-preserving compact layout and corrected rank-bucket rowptrs both show only noise-level improvement over staged rowptr.
  - Do not use the old rank-bucket number to justify K1/K3/reduce contract rewrite.
  - Current real LL K3 gap remains `staged_rowptr ~0.368 ms` vs `local_rowptr ~0.244 ms` / `pure ~0.237 ms`, but the tested compact-layout route does not recover it.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/ll_compact_collision_128_20260614_203221.json`
  - `hygon_tmp/sglang_debug/ll_compact_sidecar_pull_20260614/k3_ll_rowptr_actualmaxtok_128_20260614_203421.json`

## 2026-06-14 LL retained delta after compact artifact closure

- After invalidating the old rank-bucket upper-bound, reran the retained V3 LL no-tail path with stage timing on the current tree.
- The first attempt produced no useful output because the remote shell quoting around `TOKENS_LIST='32 128'` was wrong; the corrected command used `TOKENS_LIST="32 128"`.
- The script variable is `RUN_K2SKIP`, not `RUN_K2_SKIP`; one rerun unintentionally included the k2skip branch. This is now treated as useful attribution rather than a clean default-only run.
- Current retained default, stable last-40 stage samples:
  - 32 tokens: total `0.7115 ms`, pre-barrier `0.038 ms`, K1 `0.347 ms`, K2 `0.028 ms`, K3 `0.255 ms`, no-tail barrier `0.028 ms`, reduce `0.012 ms`; formal fused median `0.82664 ms`, min `0.82126 ms`.
  - 128 tokens: total `0.910 ms`, pre-barrier `0.0355 ms`, K1 `0.397 ms`, K2 `0.028 ms`, K3 `0.388 ms`, no-tail barrier `0.055 ms`, reduce `0.014 ms`; formal fused median `1.02724 ms`, min `1.00964 ms`.
- K2 skip attribution:
  - 32 tokens: k2skip median `0.82520 ms`, only noise-level better than default.
  - 128 tokens: k2skip median `1.03876 ms`, slower than default.
- Interpretation:
  - K2 is not a meaningful LL optimization target (`~0.028 ms` stage).
  - 32 tokens are now mainly K1 fixed/staged cost plus fixed barriers; K3 is lower and near floor.
  - 128 tokens have K1 and K3 at similar scale. Since K3 compact/rank-bucket route was invalidated and many K3 epilogue tweaks were rejected, the next low-risk work should inspect K1 fixed cost before considering larger K3 contract changes.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_retained_delta_20260614/v3_ll_default_32_perf_20260614_203652.json`
  - `hygon_tmp/sglang_debug/ll_retained_delta_20260614/v3_ll_default_128_perf_20260614_203809.json`
  - `hygon_tmp/sglang_debug/ll_retained_delta_20260614/v3_ll_perf_summary_20260614_203923.csv`

## 2026-06-14 LL K1 partial output_index clear A/B rejected

- Context:
  - After the retained LL delta refresh, K1 fixed cost was the next low-risk candidate because 128 tokens showed K1 and K3 at similar scale and K2 was only about `0.028 ms`.
  - K3 raw availability has two module-level signals: `K3_fused/k3_fused_ext` can still report raw unavailable, while the separate V3 raw module `K3_fused/k3_v3_fused_ext` correctly reports `raw_kernels=True`, `raw_ll=True`, `raw_normal=False` after an LL-only raw rebuild.
- Change tested:
  - Temporary env-gated K1 LL branch `MEGAMOE_DCU_V3_LL_K1_PARTIAL_OUTPUT_INDEX_CLEAR=1`.
  - It only cleared `output_index` rows tied to actual routed tokens instead of the full symmetric-buffer token stride, while keeping row layout and downstream contracts unchanged.
- First A/B run:
  - default 32: fused median `0.838419 ms`, min `0.831580 ms`.
  - partial 32: fused median `0.834480 ms`, min `0.822480 ms`.
  - default 128: fused median `1.046220 ms`, min `1.033480 ms`.
  - partial 128: fused median `1.029159 ms`, min `1.014919 ms`.
  - Stage medians were not a clean K1 win: 128 K1 was `0.3965 ms` default vs `0.4005 ms` partial, while the apparent total win came from K3/barrier noise.
- Confirmation run:
  - Ran 128 in alternating order partial/default/partial/default.
  - partial medians: `1.060960 ms`, `1.058419 ms`.
  - default medians: `1.053020 ms`, `1.053540 ms`.
  - Stage medians again showed no stable K1 reduction: default K1 around `0.397-0.398 ms`, partial K1 around `0.397-0.3995 ms`.
- Conclusion:
  - The partial clear branch is rejected. It passes correctness but does not reliably reduce K1 stage time and regresses in confirmation.
  - The env, launcher signature, and source changes were fully reverted; no `partial_output_index` symbols remain in local K1 V3 sources.
  - A stale hipify cache on remote (`k1_fused_ext.hip`, `k1_v3_fused_ext.hip`) temporarily retained removed symbols; deleting only those generated files and rebuilding restored the retained source path.
- Retained sanity after revert/rebuild:
  - 32 tokens: correct, fused median `0.829800 ms`, min `0.823700 ms`.
  - 128 tokens: correct, fused median `1.032919 ms`, min `1.018599 ms`.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_k1_partial_clear_pull_20260614/ll_k1_partial_clear_20260614_205720`
  - `hygon_tmp/sglang_debug/ll_k1_partial_clear_pull_20260614/ll_k1_partial_clear_confirm_20260614_210103`
  - `hygon_tmp/sglang_debug/ll_k1_partial_clear_pull_20260614/v3_ll_default_32_perf_20260614_210740.json`
  - `hygon_tmp/sglang_debug/ll_k1_partial_clear_pull_20260614/v3_ll_default_128_perf_20260614_210817.json`
  - `hygon_tmp/sglang_debug/ll_k1_partial_clear_pull_20260614/v3_ll_perf_summary_20260614_210855.csv`

## 2026-06-14 LL residual microtuning ceiling

- Environment/status refresh:
  - 8 DCUs were idle on the 11 node before the final split sanity.
  - K1 partial-clear symbols were absent from remote `.cu/.cuh` sources after revert.
  - V3 K3 raw module availability names are backend-specific: `dcu_megamoe_v3_k3_raw_kernels_available=True`, `dcu_megamoe_v3_k3_raw_ll_available=True`, `dcu_megamoe_v3_k3_raw_normal_available=False`.
- Current retained K3 LL 128 split:
  - active rows per rank average: `768`.
  - local rows per rank average: `95.625`.
  - remote rows per rank average: `672.375`.
  - `pure_contiguous`: median average-rank `0.237410 ms`, max-rank `0.239104 ms`.
  - `local_rowptr`: median average-rank `0.243754 ms`, max-rank `0.245168 ms`.
  - `staged_remote_only`: median average-rank `0.362768 ms`, max-rank `0.374480 ms`.
  - `staged_rowptr`: median average-rank `0.368026 ms`, max-rank `0.381904 ms`.
- Evidence synthesis:
  - Existing PMC still shows the remote path as higher TA/TCP stall rather than extra GEMM work: VMEM instruction scale is close, but remote/scattered rowptr writes raise `TA_BUSY` and `TCP_TA_DATA_STALL`.
  - DCU KB guidance for useful low-latency combine overlap continues to point to chunk-level readiness (`comp_signal`, `block_m`, `threshold`) and explicit state, not more same-tile epilogue store reordering.
  - Rejected or neutral LL microtuning directions now include K1 `output_index` full/partial skip, K1 CUS=32, K1 pre-rank-barrier removal, K3 blockM variants, K3 rowptr resource load, K3 store modifier, deferred rowptr wait, K3 CUS=32, dest-sort, rowaddr wave-shuffle, delayed store-burst, and compact/mapped-reduce based on corrected max-token stride.
- Conclusion:
  - The remaining LL 128 delta is real (`staged_rowptr - local_rowptr` about `0.124 ms`), but current evidence says it is a communication/writeback contract cost on DCU, not a local source-level scheduling knob.
  - Low-risk LL microtuning is at a practical ceiling for now. Future high-upside LL work should be planned as a larger chunk-readiness or reduce-contract design, not another K3-only epilogue A/B.
- Artifact:
  - `hygon_tmp/sglang_debug/ll_residual_triage_pull_20260614/ll_residual_triage_k3_128_20260614_211534.json`

## 2026-06-14 V3 LL uneven no-tail stats mismatch

- After LL microtuning was closed, Phase 7 LL uneven correctness matrix was started with:
  - `MEGAMOE_DCU_USE_LARGE_OPT_3STAGE=1`
  - `USE_MEGAMOE_V3=1`
  - `MEGAMOE_DCU_V3_BACKEND=ll`
  - `K3_USE_ASM_TAIL_REDUCE=0`
  - `--num-max-tokens-per-rank 64`
  - `--num-tokens-per-rank-list 32,48,16,64,24,40,56,8`
  - `--dispatch-num-tokens 64`
- Result:
  - no-tail failed before tail case started.
  - Process/rank 7 raised `AssertionError: stats mismatch`.
  - Fused per-local-expert stats were consistently lower than baseline, e.g. rank 7 first entries `38,35,68,74,90...` vs baseline `44,37,74,79,97...`.
- Interpretation:
  - This is a functional parity issue, not a performance regression.
  - The mismatch likely sits in K1 LL stats accounting under uneven token counts or in how `dispatch_num_tokens=64` interacts with per-rank actual token counts; K3 no-tail did not get a chance to be evaluated as the failure is at stats check.
  - Next debugging should compare K1 metadata/stats for uneven inputs against baseline/original staged behavior before touching K3.
- Artifact:
  - `hygon_tmp/sglang_debug/ll_uneven_matrix_pull_20260614/v3_ll_uneven_notail_20260614_211817.log`

## 2026-06-14 V3 LL uneven and graph parity

- Root cause of the earlier uneven stats mismatch:
  - `tests/test_mega_moe_dcu.py` calls the staged path with each rank's local token count, while `--dispatch-num-tokens 64` only controls the eager dispatch/capacity selection.
  - Original K3/rank-barrier metadata carries per-rank `sections.num_tokens` and a `uniform_num_tokens` flag.
  - V3 K1 LL route/stats build was using the local rank's runtime token count as a uniform scan bound for every source rank. In the uneven list `32,48,16,64,24,40,56,8`, rank 7 therefore scanned only 8 tokens from every peer and undercounted stats.
- Fix:
  - K1 V3 LL now treats uniform and uneven separately:
    - uniform: use the compact local runtime token count as route stride/scan bound;
    - uneven: keep the route stride at `num_max_tokens_per_rank` and clamp each source rank with `peer_sections[source].num_tokens`.
  - This preserves the uniform fast path while matching original staged fused semantics for per-rank token counts.
- Validation:
  - Rebuilt remote LL raw K1 after deleting stale `k1_v3_fused_ext.o/.hip` and `k1_fused_ext.o/.hip`, because header-only changes were not picked up by the first build.
  - Remote source guard: `PYTHONPATH=. python3 -m pytest -q tests/test_dcu_megamoe_v3.py` -> `12 passed`.
  - V3 LL uneven eager matrix:
    - no-tail, `K3_USE_ASM_TAIL_REDUCE=0`, token list `32,48,16,64,24,40,56,8`, max/dispatch `64`, 3 iters: passed, `max_abs=0.000244141`;
    - tail, `K3_USE_ASM_TAIL_REDUCE=1`, same token list, 3 iters: passed, `max_abs=0.000244141`.
  - V3 LL graph uniform:
    - no-tail and tail, capture bucket `128`, graph replays at `32,128`: passed;
    - worst observed graph replay `max_abs=0.000488281`, below `--atol=0.0035`.
  - V3 LL graph uneven:
    - no-tail and tail, capture bucket `64`, token list `32,48,16,64,24,40,56,8`, graph replay at token `64`: passed;
    - observed graph replay `max_abs=0.000244141`.
- Scope note:
  - This completes current V3 LL uneven + graph support. V3 normal graph remains intentionally guarded in `large_opt.py` and should be brought up separately if needed.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/v3_ll_uneven_notail_20260614_214608.log`
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/v3_ll_uneven_tail_20260614_214645.log`
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/v3_ll_graph_uniform_notail_.log`
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/v3_ll_graph_uniform_tail_latest.log`
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/v3_ll_graph_uneven_notail_latest.log`
  - `hygon_tmp/sglang_debug/ll_phase7_support_20260614/v3_ll_graph_uneven_tail_latest.log`

## 2026-06-14 V3 LL post-parity performance sanity

- Purpose:
  - Confirm that the LL uneven and graph support work did not regress the normal uniform eager performance path.
  - This is a regression sentinel only; no new optimization branch was attempted.
- Method:
  - Remote 11 node, container `sglang_megamoe`, 8 DCUs idle at start.
  - Ran default V3 LL only: `RUN_DEFAULT=1`, `RUN_K2SKIP=0`, `RUN_BLOCK48=0`, `RUN_BLOCK64=0`, `RUN_STAGE_TIMING=1`, `REPEAT=20`, `WARMUP=5`, `ITERS=1`, tokens `32 128`.
  - Compared against recent retained figures:
    - no-tail retained after revert: about `0.8298/1.0329 ms`;
    - prior tail post-stagecopy reference: about `0.8425/1.0759 ms`.
- Result:
  - no-tail (`K3_USE_ASM_TAIL_REDUCE=0`):
    - 32 tokens: fused median `0.82758 ms`, min `0.81196 ms`;
    - 128 tokens: fused median `1.02778 ms`, min `1.00364 ms`.
  - tail (`K3_USE_ASM_TAIL_REDUCE=1`):
    - 32 tokens: fused median `0.83048 ms`, min `0.81678 ms`;
    - 128 tokens: fused median `1.03996 ms`, min `1.02332 ms`.
- Interpretation:
  - The K1 LL per-source token-count fix has no visible regression on the uniform eager fast path.
  - Tail and no-tail are both within or better than the retained noise band. No rollback or performance follow-up is needed for this functional parity patch.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/v3_ll_default_32_perf_20260614_215444.json`
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/v3_ll_default_128_perf_20260614_215521.json`
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/v3_ll_perf_summary_20260614_215557.csv`
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/v3_ll_default_32_perf_20260614_215607.json`
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/v3_ll_default_128_perf_20260614_215644.json`
  - `hygon_tmp/sglang_debug/ll_phase7_perf_sanity_20260614/v3_ll_perf_summary_20260614_215720.csv`

## 2026-06-14 V3 LL graph-vs-eager performance sanity

- User clarification:
  - The no-regression requirement is specifically that V3 LL cuda graph replay should not be slower than the corresponding eager path.
- Initial graph bench before the fix:
  - no-tail graph bucket 128, replay 32/128: `0.9163/1.0727 ms`;
  - tail graph bucket 128, replay 32/128: `0.9891/1.1180 ms`;
  - exact bucket 32 no-tail still measured `0.9166 ms`, so the slowdown was not just replaying a 32-token request inside a 128-token bucket.
- Root cause:
  - `k1_symm_fused_l1_v3_pack5()` treated any graph launch with `runtime_num_tokens` as capacity-sized by `sym_buffer.num_max_tokens_per_rank`.
  - In the current test harness the symmetric buffer max token stride is `384`, even when the graph capture bucket is `32`, `64`, or `128`.
  - This made graph K1 allocate/return `l1_out` rows for 384-token capacity and forced K2/K3 to process about twice the rows needed for the 32/128 LL buckets.
- Fix:
  - In `megamoe/dcu_megamoe_large_opt/K1_fused/k1_fused_ext.cu`, set `route_capacity_tokens_per_rank = num_tokens`.
  - For graph calls, `num_tokens` is the capture bucket (`graph_max_tokens`); runtime token remains device-controlled through `runtime_num_tokens` and symmetric-buffer metadata.
  - `output_index` and combine addressing still use `num_max_tokens_per_rank`, preserving the existing symmetric-buffer stride contract.
- Validation after rebuild:
  - no-tail graph bucket 128 replay 32/128: `0.6860/0.8723 ms`, correctness max_abs `0.000244141/0.000488281`;
  - tail graph bucket 128 replay 32/128: `0.7293/0.9198 ms`, correctness max_abs `0.000244141/0.000488281`;
  - uneven graph bucket 64 no-tail/tail with token list `32,48,16,64,24,40,56,8`: both passed, max_abs `0.000244141`.
- Comparison to eager sentinel:
  - eager no-tail 32/128: `0.8276/1.0278 ms`;
  - eager tail 32/128: `0.8305/1.0400 ms`;
  - after the fix, graph replay is faster than eager in all four LL points and therefore no longer violates the no-regression requirement.
- Artifacts:
  - `hygon_tmp/sglang_debug/ll_graph_vs_eager_20260614/`

## 2026-06-15 V3 normal tail ASM-pack5 signal generation fix

- Symptom:
  - V3 normal tail ASM-pack5 correctness call could pass, but the subsequent eager fused benchmark previously stalled at the first `bench_tilelang_ms(lambda: run_fused(...))` call.
  - Progress markers narrowed the old stall to after `before_fused_timing` and before any new stage timing print.
- Key observation:
  - `rank_barrier_kernel(reset_tail_signal_slots=True)` resets this rank's peer receive slots to 0 before each tail/signal round.
  - C/CUDA V3 tail helpers publish `signal_generation` to peer slots via atomic add and wait until peer recv slots are `>= signal_generation`.
  - The ASM tail macro waited against `s78` but all peer signal callsites used hardcoded value `1`; after a reset, generation 2 could never be observed.
- Fix:
  - In both original tail ASM and new PACK5 tail ASM, replace all 8 peer signal callsites from `K3_TAIL_ATOMIC_SIGNAL offset, 1` to `K3_TAIL_ATOMIC_SIGNAL offset, s78`.
  - This preserves the existing wait/reduce/store schedule and aligns ASM with the rank barrier/C helper protocol.
- Validation:
  - PACK5 tail code object rebuild passed, sha256 `e0ce9a655185744879da8ffc1749c2307970175c648ab3f0cfcde4069c40cf35`.
  - Full extension rebuild passed.
  - Stage compare with `K3_V3_TAIL_STAGE_SIGNAL_GENERATION=2`, 1024 tokens, `v3_only`, `ZERO_COMBINE=1`: `rc=0`; all send/recv signal slots reached `2`, `max_abs=0`.
  - E2E V3 normal tail ASM-pack5 1024 correctness: `max_abs=0.000488281`.
  - Fused-only bench reached `after_fused_timing` and produced performance JSON; therefore the tail fused path no longer hangs at the second call.
- Performance sanity so far:
  - 1024 repeat=1 fused-only: tail median `2.0430 ms`, no-tail median `2.0578 ms`.
  - 1024 repeat=3 fused-only: tail median/min `2.1008/2.0094 ms`, no-tail median/min `2.0437/2.0374 ms`.
  - 4096 repeat=3 fused-only: tail median/min `5.8836/5.7742 ms`, no-tail median/min `5.8762/5.7800 ms`.
  - Interpretation: tail is functional and no longer blocked. At 4096, tail and no-tail are effectively identical; 1024 repeat=3 median has small noisy overhead while min is not worse. A formal retained-vs-current sweep can still be run later, but the current 1024/4096 paired data shows no material tail regression.
- Open issue:
  - Default full bench baseline timing currently VMFaults after fused timing. Correctness baseline still passes, so this should be tracked separately from V3 normal tail ASM support.

## 2026-06-15 V3 normal uneven and graph support

- Implementation:
  - Normal staged graph now accepts `v3_backend == "normal"` instead of fail-fasting.
  - K1 normal graph uses the same ASM-pack5 dispatch-pull L1 path as eager, with `runtime_num_tokens` passed to the existing graph-capable K1 ASM launcher.
  - K3 no-tail ASM-pack5 now receives `active_tiles`; the PACK5 ASM already had an active-tile gate but the C++ wrapper previously dropped the pointer.
  - K3 tail ASM-pack5 already carried `active_tiles` and graph runtime offset; no extra tail ASM change was needed after the signal-generation fix.
- Correctness:
  - Uneven eager no-tail/tail passed for token list `1024,768,512,896,640,384,256,128`, `max_abs=0.000488281`.
  - Uniform graph capture bucket `1024`, replay `512,1024`, passed no-tail/tail. Tail replay `512` had worst `max_abs=0.00244141`, still within `--atol=0.0035`; all other checked graph points were `0.000488281`.
  - Uneven graph bucket `1024`, same token list, passed no-tail/tail with `max_abs=0.000488281`.
- Graph performance vs eager:
  - Uniform no-tail:
    - graph replay `512/1024`: `1.6157/1.9828 ms`;
    - eager `512/1024`: `1.6919/2.0746 ms`.
  - Uniform tail:
    - graph replay `512/1024`: `1.6578/1.9539 ms`;
    - eager `512/1024`: `1.7467/1.9937 ms`.
  - Uneven:
    - no-tail graph/eager: `1.6629/1.7873 ms`;
    - tail graph/eager: `1.7242/1.8254 ms`.
  - Conclusion: graph replay is faster than eager across the checked V3 normal no-tail/tail uniform and uneven points.
- Artifact:
  - `hygon_tmp/sglang_debug/normal_uneven_graph_20260615/`

## 2026-06-15 Baseline timing VMFault triage

- Trigger question:
  - 用户观察到完整 bench 进入 baseline timing 后曾 VMFault 或卡住，担心 tail/graph 支持后 baseline 被打坏；同时指出早先 no-tail 测试正常。
- What was verified now:
  - Current V3 normal no-tail 1024:
    - `warmup=1 repeat=1` passed, baseline median `3.7225 ms`;
    - `warmup=1 repeat=3` passed, baseline median `3.7041 ms`.
  - Current V3 normal tail 1024:
    - `warmup=1 repeat=1` passed, baseline median `3.7264 ms`;
    - `warmup=1 repeat=3` passed, baseline median `3.7490 ms`;
    - `warmup=5 repeat=10` passed, baseline median `3.7142 ms`.
  - Current V3 normal no-tail 4096:
    - `warmup=1 repeat=1` passed, baseline median `9.6886 ms`;
    - `warmup=5 repeat=10` passed, baseline median `9.5075 ms`.
  - Current V3 normal tail 4096:
    - `warmup=1 repeat=1` passed, baseline median `9.5457 ms`;
    - `warmup=5 repeat=10` passed, baseline median `9.4997 ms`.
  - Current V3 normal staged graph 1024 bucket:
    - no-tail graph correctness + replay bench + final baseline timing passed, baseline median `3.6651 ms`;
    - tail graph correctness + replay bench + final baseline timing passed, baseline median `3.7357 ms`.
  - Current old persistent/no-large 128 shape:
    - `warmup=1 repeat=1` and `warmup=5 repeat=10` both passed with `before_baseline_timing -> after_baseline_timing`.
- Misleading reproduction:
  - Running no-tail and tail 8-rank distributed tests in parallel reused default `MASTER_PORT=8361` and the same 8 devices.
  - Resulting errors were `EADDRINUSE`, TCPStore recv failures, and NCCL socket aborts. These are not baseline VMFault evidence.
  - For distributed reproductions, run sequentially or set unique `MASTER_PORT` and non-overlapping devices.
- Old VMFault logs are not current baseline regressions:
  - `base_no_large_128_bench_20260613_192144.log`:
    - showed `MegaMoE HIP local barrier timeout: rank=5 block=77 sense=64`;
    - belonged to old `persistent_fused` benchmark behavior, not V3 normal ASM-pack5 baseline timing;
    - current same shape no longer reproduces.
  - `v3_normal_k3_normal_raw_1024_20260614_100912.log`:
    - was the early normal K3 raw pure wrapper passing `row_combine_ptrs=nullptr`;
    - root cause was raw kernel still consuming rowptr/mask contract while `(void)out` discarded pure output;
    - later fixed by isolated `kPureContiguous`, and not related to DeepEP/DeepGEMM baseline.
  - LL/local-barrier VMFault notes in planning include an environment-pollution case where an external `sglang serve` occupied all 8 cards.
- Current conclusion:
  - In a clean 8-HCU environment, the baseline oracle and baseline timing are currently healthy for the checked V3 normal no-tail/tail eager, staged graph, and old persistent control shapes.
  - The earlier `MEGAMOE_DCU_TEST_SKIP_BASELINE_TIMING=1` diagnostic should be treated as a fused-path isolation tool, not as proof that baseline timing is inherently broken.
  - If the issue reappears, the next useful artifact is a full log with `[BENCH_PROGRESS]`, exact env/args, unique `MASTER_PORT`, `hy-smi`/process state, and whether the last marker is `before_fused_timing`, `after_fused_timing`, or `before_baseline_timing`.
- Artifact:
  - `hygon_tmp/sglang_debug/baseline_vmfault_20260615/`

## 2026-06-15 V3 normal C/raw retirement cleanup

- Production boundary after cleanup:
  - V3 normal K1/K3 production path is ASM-pack5 only for no-tail, tail-reduce, eager, graph, uniform and uneven cases.
  - V3 raw extension build is LL-only; setup rejects normal raw backend values instead of silently rebuilding stale C/aicc paths.
  - V3 normal C/aicc raw code is not a fallback path. Historical A/B and bug records remain as evidence, but new work should not continue that route.
- Removed/retired env surface:
  - `DG_BUILD_MEGAMOE_V3_NORMAL_AICC`
  - `MEGAMOE_DCU_V3_K1_ASM_PACK5`
  - `MEGAMOE_DCU_V3_K3_ASM_PACK5`
  - `MEGAMOE_DCU_V3_NO_TAIL_SIGNAL`
  - `MEGAMOE_DCU_V3_NO_TAIL_SYNC`
  - `MEGAMOE_DCU_V3_K2_SYSTEM_FENCE`
  - `MEGAMOE_DCU_V3_REDUCE_ACQUIRE`
  - `MEGAMOE_DCU_V3_BARRIER_ACQUIRE`
- Naming cleanup:
  - LL production kernel names should use `LowLatency...`, not `Pure_LowLatency...`, because the LL kernel now carries fused/staged semantics.
  - Diagnostic pybind/function names should say `reference` rather than `pure` when they are used as a local contiguous reference path, not a production pure denominator.
- Debug script policy:
  - Scripts that actively support current LL/normal ASM-pack5 work should be kept runnable and aligned with current pybind/env names.
  - Scripts whose only purpose was normal C/aicc raw bring-up should become explicit retired stubs instead of half-working historical commands.

## 2026-06-15 K1/K3 pack5 header naming

- Current naming rule:
  - K1 V3 pack5 groupgemm header: `k1_v3_pack5_groupgemm_impl.cuh`.
  - K3 V3 pack5 groupgemm header: `k3_v3_pack5_groupgemm_impl.cuh`.
- Rationale:
  - Both files carry V3 pack5-layout kernel bodies, so only K3 having `pack5` in the filename was misleading.
  - The old `k1_v3_groupgemm_impl.cuh` name should be treated as historical only.

## 2026-06-15 V3 raw naming/build gate retirement

- Current meaning decision:
  - `raw` used to mean low-level bring-up/diagnostic C pack5 entry points that bypassed the normal staged wrapper or ASM code object.
  - After V3 normal moved to ASM-pack5 and V3 LL became the retained C pack5 path, that name became misleading.
- Current build rule:
  - V3 normal ASM-pack5 code objects are always part of the normal large-opt build.
  - V3 K1 LL pack5 source is always compiled into the K1 large-opt extension.
  - V3 K3 LL pack5 source is always compiled as `K3_fused.k3_v3_fused_ext`.
  - `DG_BUILD_MEGAMOE_V3_K1_RAW_KERNELS`, `DG_BUILD_MEGAMOE_V3_K1_RAW_BACKEND`, `DG_BUILD_MEGAMOE_V3_K3_RAW_KERNELS`, `DG_BUILD_MEGAMOE_V3_K3_RAW_BACKEND`, and `DCU_MEGAMOE_V3_ENABLE_*RAW*` are retired.
- Source boundary:
  - `k1_v3_stub_ext.cu` and `k3_v3_stub_ext.cu` are removed.
  - K1 internal launcher name is now `dcu_megamoe_v3_launch_k1_ll_symm_stage_pack5`.
  - K3 internal launcher name is now `dcu_megamoe_v3_launch_k3_ll_combine_pack5`.
  - K3 diagnostic pybind names are now `k3_v3_ll_combine`, `k3_v3_ll_reference`, and `k3_v3_ll_combine_tail`.
- Note:
  - AMD `raw_buffer` intrinsic names in `.cuh` files are ISA/builtin terminology and should not be renamed as part of this cleanup.

## 2026-06-15 Pending: V3 K1 ASM independence

- User design feedback:
  - Current V3 K1 normal ASM-pack5 path reuses the original K1 host launcher path and is selected through a `backend == "normal"` branch.
  - This is semantically confusing because the original K1 ASM path is also a normal-scenario ASM path.
  - V3 K1 should be isolated from the original K1 implementation more explicitly instead of sharing the same backend-parameter branch.
- Pending action:
  - Do not modify code until the user provides the remaining design changes.
  - When implementing, consider a V3-specific K1 normal ASM entry/host wrapper or clearer function/module boundary so original K1 ASM and V3 K1 ASM-pack5 are not distinguished only by a backend string and code-object parameter.

## 2026-06-15 Pending: V2 package/build retirement

- User design feedback:
  - `DG_BUILD_MEGAMOE_V2_EXT` is no longer needed.
  - The whole `megamoe/dcu_megamoe_v2/` package and related V2 build/package/test/debug references are no longer needed.
- Pending action:
  - Do not delete V2 code in isolation while the user is still listing design changes.
  - When implementing the combined cleanup, remove the V2 extension switch from `setup.py`, drop V2 package data/build entries, delete `megamoe/dcu_megamoe_v2/`, and clean any stale imports, source guards, scripts, or docs that still treat V2 as an available comparison path.
  - Keep historical V2 observations only in `.planning/dcu_megamoe_v3/` as task memory.

## 2026-06-15 Pending: setup.py minimal V3 build surface

- User design feedback:
  - `setup.py` should not carry broad historical scaffolding just to support V3.
  - The intended setup delta is small: add V3-related compile files/code objects that must be built by default, not preserve old raw/V2/stub build gates or extra temporary source-list variables.
  - Current `large_opt_k1_sources`, `large_opt_k3_sources`, `large_opt_k3_v3_ext`, V2 build gate/package entries, and source-guard tests should be reviewed as possible historical residue.
- Pending action:
  - Do not modify code yet; user explicitly asked to record first.
  - When implementing later, keep `setup.py` close to the original direct `modules.extend([...])` style: inline the required V3 K1 source, add the separate V3 K3 LL extension only if still needed, keep PACK5 ASM code object entries, keep `*.cuh` package data, and remove V2/raw/stub-era build switches and package entries.

## 2026-06-15 Pending: K1 normal ASM should mirror K3 pack5 isolation

- User design feedback:
  - K3 V3 normal already has an independent pack5 implementation boundary through dedicated ASM-pack5 wrapper/entry points.
  - K1 V3 normal should borrow that structure instead of sharing the original K1 ASM host path and selecting V3 only through a backend/code-object parameter.
- Pending action:
  - When implementing the K1 cleanup, introduce a V3 K1 normal ASM-pack5-specific function/module boundary analogous to K3 normal pack5.
  - Keep original K1 ASM and V3 K1 ASM-pack5 semantically separated in naming and wrapper call sites, even if the lower-level launch helper can share small common internals.

## 2026-06-15 Pending: record-only until explicit start

- User workflow direction:
  - Collect all cleanup/design feedback first.
  - Do not modify source code until the user explicitly says to start changing code.
  - Planning files may continue to capture decisions and pending work.
- Pending cleanup notes collected in this review pass:
  - `k1_v3_pack5_groupgemm_impl.cuh` still contains unused normal-C/fixed-route kernel body and helpers even though K1 V3 normal is ASM-pack5; later cleanup should remove unused normal C body and update source guards.
  - `k3_v3_pack5_groupgemm_impl.cuh` also still contains redundant or no-longer-used logic from earlier bring-up/diagnostic paths; later cleanup should prune unused K3 V3 pack5 helper/kernel variants and keep only production LL combine/tail paths plus genuinely shared helpers.
  - `MEGAMOE_DCU_K3_DEBUG_LAUNCH` is a one-off stderr launch-parameter diagnostic in `k3_fused_ext.cu`; later cleanup can remove it and any now-unused include.
  - `reduce_local_combine_vec_kernel(invalidate_before_read)` is a dormant no-tail cache-visibility diagnostic path; current large-opt path no longer passes it. Treat as cleanup candidate, but verify no external debug script still depends on it before removal.
  - `k3_l2_fused_v3_to_combine()` still has a defensive `elif sym_buffer is not None: raise NotImplementedError("V3 K3 LL no-tail signal path is not wired yet")` branch. Production LL no-tail is combine-only and does not pass `sym_buffer`; LL tail passes `sym_buffer` together with `asm_reduce_y` and signal tensors. Later cleanup should remove this unused no-tail signal guard and its source-test assertion.
  - `k3_v3_ll_reference` is a contiguous-output diagnostic pybind for old rowptr-vs-reference timing. User says it does not need to be retained; later cleanup should remove the function, pybind, source guards, and scripts that call it.
  - `tests/test_dcu_megamoe_v3.py` is mostly a development-time source guard file now. It still has useful lightweight checks for V3 env gating and pack5 layout helpers, but many assertions preserve historical implementation details, raw/V2/stub cleanup state, setup temporary variables, unused normal C bodies, and diagnostic pybinds. Later cleanup should retire or drastically shrink this file instead of carrying it as a long-term regression suite.
  - `large_opt.py` still contains development diagnostics: `_v3_debug_stage_sync`, `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC`, `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC_LABELS`, `_v3_stage_timing_enabled`, and `MEGAMOE_DCU_V3_STAGE_TIMING`. These were useful during tail/graph bring-up but should be removed or moved out of the production path once final validation is done.
  - `MEGAMOE_DCU_V3_LL_BLOCK_M` in `large_opt.py` is still an A/B tuning knob. Current retained default is 32, and previous 48/64 checks regressed; later cleanup can consider fixing LL block M to the retained value and removing the env knob if no longer needed.

## 2026-06-15 Confirmed cleanup scope from user review

- User confirmed these groups are safe to clean once implementation starts:
  - V2 as a whole: `DG_BUILD_MEGAMOE_V2_EXT`, `megamoe/dcu_megamoe_v2/`, and V2 build/package/test/script references.
  - V3 normal C/raw remnants: unused K1/K3 normal C kernel bodies, fixed-route bring-up helpers, raw/stub/availability assertions and scripts.
  - Diagnostic pybind: `k3_v3_ll_reference` plus debug scripts/source guards that still call it.
  - Debug env: `MEGAMOE_DCU_K3_DEBUG_LAUNCH`, `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC*`, `MEGAMOE_DCU_V3_STAGE_TIMING`; `MEGAMOE_DCU_V3_LL_BLOCK_M` only after the final retained block M decision is fixed.
  - Dormant API parameters: K3 `invalidate_before_read`, K3 `acquire_after_wait`, and K2 `system_fence_after_write`.
- User confirmed these areas require boundary cleanup rather than simple deletion:
  - K1 V3 normal ASM should mirror K3 normal pack5 isolation with a V3-specific entry/wrapper boundary instead of selecting through the original K1 ASM host path plus backend/code-object parameter.
  - `setup.py` should be reduced to the minimum V3 delta: required V3 sources, pack5 ASM code objects, and `*.cuh` package data only.
  - `large_opt.py` should retain only clear V3 backend dispatch after removing development sync/timing logic: normal ASM-pack5 and LL C pack5.
  - `megamoe/__init__.py` should be checked for leftover V3 normal early defense/debug logic in warmup or symmetric-buffer allocation.
- User confirmed test/script direction:
  - `tests/test_dcu_megamoe_v3.py` should not remain as the current broad source-guard suite; delete it or shrink it to V3 gate/backend, pack5 layout helper, and default-env safety checks.
  - Functional/performance regression should remain centered on `tests/test_mega_moe_dcu.py` and the remote 8-card matrix.
  - `hygon_tmp/sglang_debug` should keep only scripts that run current production paths; normal C/raw, reference/pure, and stale A/B scripts should become retired stubs or be deleted.
- Execution constraint still stands: do not modify source code until the user explicitly says to start.

## 2026-06-15 Cleanup implementation findings

- Production boundary after cleanup:
  - V3 normal is ASM-pack5 only: K1 dispatch-pull ASM-pack5, K3 combine ASM-pack5, K3 tail-reduce ASM-pack5; eager/graph and uniform/uneven support remain the intended production surface.
  - V3 LL is C pack5 only: K1/K3 stage-owned `LowLatencyMaskedGroupGemmKernel` paths remain default compiled and are the only retained C pack5 production kernels.
  - V3 normal C/aicc/raw is deleted from active kernel bodies and build gates, not retained as fallback.
- Naming decisions:
  - K1/K3 LL kernel names should use `LowLatency...`; `Pure...` is misleading because the retained kernels carry staged communication semantics.
  - `raw` should not appear in V3 build/env/pybind names except AMD `raw_buffer` intrinsic terminology, which is ISA/builtin naming and should not be renamed.
  - K1 V3 normal ASM now has a V3-specific Python wrapper and pybind entry: `k1_symm_fused_l1_v3_asm_pack5(...)`. Like K3 normal pack5, the public entry is independent while the low-level launch mechanics are shared through an internal helper.
- `setup.py` cleanup:
  - V2 extension/package data removal is safe because V2 is no longer a production or comparison target.
  - V3 build surface should stay direct and minimal: original large-opt K1/K2/K3, V3 K1 LL source, V3 K3 LL extension, pack5 ASM code objects, and `*.cuh` package data.
  - Removed source-list temporary variables were historical scaffolding, not required for current build semantics.
- Debug/env cleanup:
  - `MEGAMOE_DCU_V3_DEBUG_STAGE_SYNC*` and `MEGAMOE_DCU_V3_STAGE_TIMING` were bring-up diagnostics and should not remain on the production hot path.
  - `MEGAMOE_DCU_V3_LL_BLOCK_M` is removed after fixing retained LL block M to 32; previous block48/block64 A/B runs regressed.
  - K2/K3 dormant cache/fence/acquire parameters were not part of current V3 production paths and only enlarged API ambiguity.
- `megamoe/__init__.py`:
  - `_v3_normal_symm_warmup_enabled` was inspected and retained because it is shape/backend/env gated allocation warmup, not a debug switch; removing it could reintroduce first-run allocation or perf variance.
- Verification caveats:
  - Local Python still lacks pytest, so local source tests cannot be run through pytest until the environment is installed.
  - GPU build/perf confirmation still requires remote SSH/container access; do not claim remote build or 8-card smoke passed until those commands complete.

## 2026-06-15 Cleanup validation notes

- Local validation that passed after cleanup:
  - Python syntax/import bytecode: `compileall megamoe tests\test_mega_moe_dcu.py tests\test_dcu_megamoe_v3.py`;
  - `setup.py` and active edited debug scripts: `py_compile`;
  - whitespace/conflict hygiene: `git diff --check`;
  - lightweight V3 contract tests via manual pytest shim: gate/backend, pack5 layout, setup surface, runtime boundary, retired debug/API absence.
- Retired-symbol scan result:
  - Production sources and active scripts no longer contain V2 build/package references, normal C kernel names, `k3_v3_ll_reference`, debug stage/timing env, LL block-M env, or dormant K2/K3 API parameters.
  - Remaining matches are intentionally in `tests/test_dcu_megamoe_v3.py` as negative assertions.
- Remote validation is currently blocked by infrastructure, not by a compile/test failure:
  - Node `10.17.176.11` is reachable by ICMP ping.
  - TCP 22 refuses connection; SSH reports `Connection refused`.
  - No remote sync/build/GPU smoke has been performed for this cleanup snapshot.

## 2026-06-15 Remote cleanup validation findings

- Remote validation is no longer blocked for the cleanup snapshot:
  - SSH to `hg@10.17.176.11:22` and container `sglang_megamoe` recovered;
  - the cleanup snapshot was synced to `/workspace/DeepGEMM`;
  - remote `compileall`, source pytest, and clean `build_ext --inplace` passed.
- Current production boundary after remote validation:
  - V3 normal production path is isolated ASM-pack5 for K1 and K3, including tail-reduce;
  - V3 LL production path is C pack5 `LowLatency...` for K1/K3;
  - V2, normal C/raw, diagnostic reference pybind, debug stage/timing envs, raw/stub gates, and dormant K2/K3 API args are not active production surfaces.
- The remote 8-card matrix supports these conclusions:
  - normal ASM-pack5 tail/no-tail 32-token smoke passes and remains faster than DeepEP/DeepGEMM baseline;
  - normal ASM-pack5 uneven tail passes;
  - normal ASM-pack5 graph replay on uneven bucket is faster than eager replay at the checked 32/96/128 token buckets;
  - LL C-pack5 graph replay on the same uneven bucket is also faster than eager replay for no-tail and tail at the checked 32/96/128 token buckets;
  - normal ASM-pack5 4096-token no-tail/tail passes and remains about `1.63x` faster than the DeepEP/DeepGEMM baseline in this cleanup validation run;
  - LL tail/no-tail 32-token smoke passes with stronger speedup than normal at the small-token point;
  - legacy non-V3 large-opt tail smoke still passes, so cleanup did not break the old gated path in the checked shape.
- Residual risk / follow-up:
  - This validation is a cleanup gate, not a full Phase 6 final performance sweep over every historical 1024/4096 normal and 32/128 LL bucket;
  - additional 4096 graph/perf buckets can be run later as confidence expansion, but they are no longer blocking the V2/raw/debug cleanup;
  - remote command wrappers should avoid PowerShell here-strings with CRLF when the final bash `exit $status` matters; pipe Python/batch commands through stdin or force LF.

## 2026-06-15 Plan convergence findings

- The remaining in-progress markers in `task_plan.md` were not current implementation blockers; they were mostly historical V3 normal C/aicc/raw optimization threads that predated the retained ASM-pack5 normal path.
- Current production boundary is now explicit:
  - V3 normal: isolated ASM-pack5 K1/K3 for no-tail and tail-reduce, including eager/graph and uneven coverage;
  - V3 LL: retained C pack5 K1/K3 for no-tail and tail-reduce, including eager/graph and uneven coverage.
- Items marked abandoned are routes that either failed A/B, were replaced by ASM-pack5, or would reintroduce retired surfaces: normal C/aicc K3 scalar rowptr/store tuning, normal C/raw tail-reduce, fixed-route/raw/stub availability gates, staged-vector-store production attempts, rowptr4/rowptr8 batch-load, no-tail internal signal, and repeated fence/load-family toggles.
- Items left as optional backlog are deliberately non-blocking and require fresh profiling evidence before implementation: no-tail fused-reduce ASM-pack5 diagnostic, compute/store overlap microbench, row-emission/rank-bucket contract exploration, and LL K3 128 remote/scattered residual work.

## 2026-06-15 Symm buffer / route_scratch memory audit

- `SymmBuffer.buffer` is already relatively lean on DCU:
  - `csrc/apis/mega_dcu.hpp::get_symm_buffer_size_for_mega_moe()` allocates only peer pointer header, runtime token, input token/scale/topk metadata, and slot-major BF16 combine storage;
  - DCU `l1_acts/l1_acts_sf/l2_acts/l2_acts_sf` returned to Python are empty views at `combine_offset`, so there is no hidden CUDA-style L1/L2 activation payload in symmetric memory.
- The symmetric combine region is still a real contract, not obvious redundancy:
  - big fused and staged fused both derive combine addresses from `combine_token_offset(...)`;
  - V3 K1 normal/LL writes `row_combine_ptrs` pointing into `combine[topk_slot, token, hidden]`;
  - K3 no-tail writes combine partials and later `reduce_local_combine` reads them;
  - K3 tail-reduce paths also write/use the same combine region before reducing into `y`.
- `route_scratch` is the main memory pressure point:
  - allocation uses `dcu_route_scratch_bytes(...)`, i.e. the full big-fused route tile layout;
  - staged large_opt reuses its `x_fp8/act_bf16/act_fp8/act_scale` regions as `staged_x/l1_out/act_fp8/act_scale`, avoiding separate Python allocations;
  - this is efficient for compatibility, but conservative for V3-only or LL-only use.
- Approximate per-rank allocation using the real 384-token alignment:
  - requested 32/128 -> allocated max 384: symm about `0.019 GiB`, route_scratch about `0.192 GiB`;
  - requested 1024 -> allocated max 1152: symm about `0.057 GiB`, route_scratch about `0.556 GiB`;
  - requested 4096 -> allocated max 4224: symm about `0.210 GiB`, route_scratch about `2.012 GiB`.
- The largest route_scratch chunks at allocated max 4224 are staged_x/big-fused x_fp8 about `796 MiB`, staged l1_out view about `796 MiB`, and act_fp8 about `398 MiB`; the task workspace is only about `33 MiB`.
- V3 LL small-token path is the clearest over-allocation:
  - for requested 32/128, V3 LL K1 launches only about `2048` rows, needing roughly `8 MiB` staged_x, `16 MiB` l1_out, and `4 MiB` act_fp8;
  - the shared big-fused scratch layout provides about `76 MiB` staged_x, `76 MiB` l1_out capacity, and `38 MiB` act_fp8 for the aligned 384-token buffer.
- V3 normal ASM-pack5 also uses fewer rows than the route-tile worst case:
  - for allocated max 4224, K1 normal capacity is about `32256` rows, roughly `126 MiB` staged_x, `252 MiB` l1_out, and `63 MiB` act_fp8;
  - the shared scratch view provides about `796/796/398 MiB` for those three regions.
- Low-risk cleanup is mostly naming/documentation: avoid implying that empty DCU `l1_acts/l2_acts` views consume memory.
- Real memory reduction requires a new allocation boundary:
  - either a staged-only / V3-only `SymmBuffer` mode with smaller route_scratch and explicit big-fused disallowance;
  - or a separate staged workspace tensor sized by backend and graph bucket, while keeping `symm_buffer.buffer` unchanged.

## 2026-06-15 Big fused retirement impact on route_scratch

- User direction:
  - big fused is likely to be deleted later because this branch no longer has a clear advantage;
  - route_scratch memory optimization should therefore be planned around the retained staged/V3 paths instead of continuing to reserve for full big-fused compatibility forever.
- Updated interpretation:
  - once big fused is retired, `route_scratch` no longer needs to preserve the full `dcu_route_tile_scratch_layout` contract;
  - the retained allocation contract can be reduced to staged/V3 needs: K1 metadata/task workspace, staged input/output buffers, K2 activation buffers, K3 probability/tail/graph metadata, and graph reset flags;
  - big-fused-only persistent route tile queues, L2 pull/done counters, tile rowptr arrays, and similar scratch should become deletion candidates.
- Optimization direction:
  - introduce a staged-only route_scratch size/layout formula after the big fused deletion boundary is clear;
  - size V3 LL scratch by backend/token bucket rows instead of by the 384-token big-fused route layout;
  - size V3 normal ASM-pack5 scratch by actual K1 capacity rows instead of route-tile worst-case rows.
- Required validation before changing allocation:
  - cover V3 normal/LL, tail/no-tail, eager/graph, uniform/uneven, and 32/128/1024/4096 token buckets;
  - make the deleted big fused entry points either disappear cleanly or fail fast, so stale callers cannot silently interpret the smaller scratch with the old layout.

## 2026-06-15 Staged fused long-term V3-only boundary

- User direction:
  - staged fused is also expected to keep only the V3 implementation later;
  - V3 LL should own the small-token path, and V3 normal should own the large-token path.
- Planning impact:
  - legacy staged fused non-V3 should be treated as a transition/compatibility path, not a long-term optimization target;
  - after big fused and legacy staged fused retire, `route_scratch` can be planned as V3-only staged workspace rather than a shared compatibility layout;
  - the scratch allocator should eventually key off the retained V3 backend and token bucket: LL small-token row capacity, normal ASM-pack5 large-token row capacity.
- Validation boundary:
  - current off path behavior still must stay stable until the deletion step is explicitly implemented;
  - the future V3-only staged transition needs clear delete/fail-fast semantics for stale legacy staged callers;
  - the retained V3 matrix remains LL small token and normal large token across tail/no-tail, eager/graph, uniform/uneven.

## 2026-06-15 Required V3 sweep matrix

- User marked this as required, not optional:
  - V3 LL must be swept with graph enabled at uniform tokens per rank `8, 32, 64, 128, 256, 512, 1024`;
  - LL graph capture bucket should be `1024`;
  - V3 normal must be swept in eager mode at uniform tokens per rank `256, 512, 1024, 1025, 2048, 2050, 4096, 4097, 8192`;
  - both LL and normal must cover no-tail and tail.
- Planning implication:
  - existing 32/128/4096 smoke results do not satisfy this required sweep;
  - the matrix should be run as a fresh remote data collection pass with JSON outputs and a summary table;
  - boundary token pairs `1024/1025`, `2048/2050`, and `4096/4097` should be watched for capacity, alignment, graph bucket, or hidden layout-step regressions.
- Suggested record fields:
  - backend, token count, graph/eager mode, capture bucket when graph is used, tail mode, correctness, fused time, baseline oracle time, speedup, graph replay time, and graph-vs-eager delta when applicable.

## 2026-06-15 Phase 10 sweep results

- Run artifact:
  - local and remote result directory: `hygon_tmp/sglang_debug/phase10_v3_sweep_20260615_196S/`;
  - machine-readable summary: `summary.csv`;
  - case status table: `case_status.tsv`;
  - sweep launcher: `hygon_tmp/sglang_debug/run_v3_phase10_sweep.sh`.
- Preflight result:
  - remote cards were idle before launch and no KFD process remained after launch;
  - `compileall`, `tests/test_dcu_megamoe_v3.py`, and `build_ext --inplace` passed before benchmark cases.
- Main discovery:
  - V3 LL graph with capture bucket `1024` is not currently clean.
  - no-tail fails correctness before metric output: `max_abs=0.071502685546875` exceeds `--atol=0.0035`;
  - tail fails correctness before metric output with `stats mismatch`;
  - because the 1024 capture case fails during correctness, the required LL replay sweep for `8,32,64,128,256,512,1024` remains incomplete.
- V3 normal eager result:
  - all no-tail and tail cases passed correctness for `256,512,1024,1025,2048,2050,4096,4097,8192`;
  - no-tail fused time range: `1.593059 ms` at 256 to `12.549034 ms` at 8192;
  - tail fused time range: `1.623479 ms` at 256 to `12.009055 ms` at 8192;
  - 4096 result is consistent with the earlier manual check: no-tail `5.863377 ms`, tail `5.822198 ms`;
  - boundary pairs `1024/1025`, `2048/2050`, and `4096/4097` did not show a correctness issue or a clear performance cliff.
- Planning impact:
  - Initial Phase 10 could not be marked complete because LL graph capture bucket `1024` failed before replay metrics;
  - this was superseded by the LL route-capacity fix and the follow-up full LL graph replay sweep recorded below.

## 2026-06-15 V3 LL 1024 route capacity finding

- The Phase 10 LL 1024 failure was a K1 LL staging capacity issue, not a graph-only K3 combine problem.
- The old host formula sized each local expert bucket by the mean routed row count:
  - `ceil(tokens_per_rank * topk / local_experts)`, aligned to 64 rows;
  - for 1024 tokens/rank, `1024 * 6 / 32 = 192`, and aligned capacity was still 192.
- Random routing is not guaranteed to keep every local expert at or below the mean. Once one expert bucket exceeded 192 rows, K1 LL clipped staged rows by `m_per_expert`; `cumulative_local_expert_recv_stats` then disagreed with the baseline expert counts, and no-tail could also show large output diff.
- The initial fix added host-side headroom only for larger LL buckets:
  - 512 keeps rows/expert at 128;
  - 768 keeps rows/expert at 192;
  - 896/960/1024 move from 192 to 256.
- That initial fix preserved the already tuned 32/128/512 small-token behavior while making the 1024 capture bucket safe for correctness testing; this was later superseded by the exact 256/512 eager fix below.
- Follow-up A/B initially narrowed the threshold to `expected rows/expert >= 160`, but that conclusion is superseded:
  - exact eager bucket 256 later reproduced the same capacity overflow class that graph capture1024/replay256 had hidden with a larger captured capacity;
  - production now uses `ll_expected_rows_per_expert >= 48 ? 64 : 0`, so exact 256 and 512 get one 64-row tile of headroom while 32/128 tiny buckets keep their original capacity;
  - this makes LL actual execution safe up to at least 512 tokens/rank even when graph capture tokens are much larger.
- Performance interpretation:
  - old small-token guards are unchanged or slightly better within noise: LL tail 32 `0.711300 ms`, 128 uneven graph replay 32/96/128 `0.712760/0.786380/0.801960 ms`;
  - refined 513 no-tail/tail is `2.193979/2.279979 ms`, matching the no-headroom performance class while preserving correctness;
  - LL graph capture1024 is now correct, but 1024 LL replay/eager sits around `4.7-4.9 ms`, so it should remain correctness/capture coverage rather than the production large-token route;
  - normal backend remains the intended production path for >=512 tokens per rank.
- Follow-up:
  - if future 512/768 or uneven workloads still show rare route overflow, consider making LL rows/expert headroom a direct capacity-percentile policy instead of the current fixed threshold rule.

## 2026-06-16 V3 LL exact 256/512 row-capacity fix

- Symptom:
  - after the graph runtime-row fix, `capture=1024` graph replay 256/512 stayed correct, but exact eager bucket 256 could still exceed tolerance;
  - both `block_m=32` and `block_m=64` failed exact eager 256 before the capacity fix, so this was not a block_m tuning issue.
- Root cause:
  - graph replay captured with bucket 1024 had enough K1 LL row capacity and masked the issue;
  - exact eager bucket 256 used `ceil(256*6/32)=48` expected rows/expert, aligned to 64 rows/expert with no slack;
  - random routing can put more than 64 local rows into one expert, so K1 clips staged rows and K3 consumes incomplete metadata.
- Fix:
  - lower the LL headroom threshold to `kLlHeadroomExpectedRowsThreshold = 48`;
  - add one `kLlHeadroomRows = 64` tile for expected rows/expert >= 48;
  - preserve exact 32/128 tiny buckets unchanged, while exact 256/512 now have one tile of row headroom.
- Verification:
  - source guard added in `tests/test_dcu_megamoe_v3.py` so the old `>=160 ? 64 : 0` rule cannot silently return;
  - remote `tests/test_dcu_megamoe_v3.py` passed 7/7 and K1 object was rebuilt;
  - exact eager 32/128/256/512 no-tail and tail all passed;
  - graph capture1024 replay 256/512 no-tail and tail also passed.

## 2026-06-15 Phase 10 LL graph replay completion

- Follow-up artifact:
  - local and remote result directory: `hygon_tmp/sglang_debug/phase10_ll_graph_20260615_232857/`;
  - machine-readable summary: `summary.csv`;
  - no-tail JSON: `graph_ll_tail0_1024.json`;
  - tail JSON: `graph_ll_tail1_1024.json`.
- Scope:
  - V3 LL backend;
  - graph capture bucket fixed at `1024` tokens/rank;
  - replay tokens/rank `8,32,64,128,256,512,1024`;
  - both no-tail (`K3_USE_ASM_TAIL_REDUCE=0`) and tail (`K3_USE_ASM_TAIL_REDUCE=1`).
- Correctness:
  - no-tail and tail both passed;
  - both modes reported `max_abs=0.000488281` and `mean_abs=9.52169e-06`;
  - the previous `stats mismatch` / large-diff failure did not reproduce after the refined K1 LL row-capacity fix.
- Replay performance:

| replay tokens/rank | no-tail graph ms | tail graph ms |
| ---: | ---: | ---: |
| 8 | 1.326279 | 1.540799 |
| 32 | 1.453019 | 1.699679 |
| 64 | 1.522400 | 1.746679 |
| 128 | 1.612180 | 1.845019 |
| 256 | 1.907599 | 2.091899 |
| 512 | 2.464999 | 2.780359 |
| 1024 | 4.631458 | 4.824998 |

- Interpretation:
  - Phase 10 required LL graph sweep is now complete;
  - LL graph capture `1024` is now correctness-clean and gives a stable replay table;
  - LL 1024 remains a capture/correctness stress case rather than the recommended large-token production backend, because retained production routing is still LL for small tokens and normal ASM-pack5 for large tokens.

## 2026-06-16 V3 LL graph replay fixed-row performance bug

- Symptom:
  - after the K1 LL row-capacity fix, `capture=1024` graph correctness passed but replay small-token latency was still too high;
  - examples from the superseded intermediate table: no-tail replay 32 was `1.453019 ms`, tail replay 32 was `1.699679 ms`;
  - this contradicted the graph contract: capture bucket should bound allocation/launch args, while replay runtime tokens should determine actual useful work.
- Root cause:
  - `k1_symm_fused_l1_v3_graph()` captures with `graph_max_tokens`, so the K1 LL output tensor `l1_out` is shaped by capture capacity;
  - `large_opt.py` and K3 wrappers derive K3 rows from that fixed-capacity tensor shape;
  - V3 K3 LL host launched `V3_K3_LowLatencyMaskedGroupGemmKernel` with `kUseFixedRows=true` and `actual_m=nullptr`, so K3 did GEMM work for capture rows instead of runtime per-expert rows.
- Fix:
  - K1 V3 LL now returns a `ll_actual_m` CUDA int32 view over the per-local-expert counts written in `route_scratch_i32[0:local_experts]`;
  - K3 V3 LL combine/tail graph passes this tensor as `actual_m` and launches with `kUseFixedRows=false`;
  - the kernel clamps `actual_m[lane]` to `m_per_expert` before computing `local_tokens`.
- Final verification:
  - output directory: `hygon_tmp/sglang_debug/ll_graph_dynamic_verify_20260615_235731/`;
  - uniform `capture=1024`, replay tokens `8,32,64,128,256,512,1024`, both no-tail and tail passed;
  - no-tail replay ms: `0.552560,0.647959,0.702340,0.826720,1.244020,2.208100,4.740119`;
  - tail replay ms: `0.755800,0.851500,0.898840,1.026680,1.429820,2.340499,4.855359`;
  - uneven capture 128 replay 32/96/128 also passed: no-tail `0.598960/0.673880/0.688180 ms`, tail `0.632180/0.702880/0.719960 ms`.
- Normal graph comparison:
  - V3 normal graph uses ASM-pack5 wrappers, `runtime_num_tokens`, `active_tiles`, and tail graph runtime offsets;
  - it does not use the V3 K3 LL fixed-row C kernel path and no analogous replay-small-token-over-capture-rows issue was found by static inspection;
  - spot checks passed after cleanup: no-tail replay 512 `max_abs=0.000488281`, tail replay 1024 `max_abs=0.000488281`.
- Planning impact:
  - `phase10_ll_graph_20260615_232857` is retained as a useful intermediate artifact but its replay performance table is superseded by `ll_graph_dynamic_verify_20260615_235731`;
  - Phase 10 remains complete, now with both LL correctness and runtime-row graph performance fixed.

## 2026-06-16 V3 normal tail graph intermittent tolerance failure

- Scope:
  - V3 normal backend;
  - ASM-pack5 tail-reduce graph path;
  - hidden 4096, intermediate 2048, experts 256, topk 6, graph capture bucket 1024.
- Evidence:
  - `TAIL=1 GRAPH_TOKENS=512,1024`, repeated 4 independent processes:
    - result dir `hygon_tmp/sglang_debug/normal_graph_tail1024_repeat_20260616_001327/`;
    - 2/4 passed;
    - iter 2 failed at replay 512 with `max_abs=0.0059814453125`;
    - iter 4 failed at replay 512 with `max_abs=0.011760711669921875`.
  - `TAIL=1 GRAPH_TOKENS=512`, repeated 4 independent processes:
    - result dir `hygon_tmp/sglang_debug/normal_graph_tail512only_repeat_20260616_001740/`;
    - 4/4 passed, each `max_abs=0.000488281`.
  - `TAIL=1 GRAPH_TOKENS=1024`, repeated 4 independent processes:
    - result dir `hygon_tmp/sglang_debug/normal_graph_tail1024only_repeat_20260616_002032/`;
    - 3/4 passed;
    - iter 2 failed at replay 1024 with `max_abs=0.01580810546875`.
- Interpretation:
  - the earlier normal tail graph 1024 over-threshold is not a single environmental blip;
  - it is intermittent and graph-replay-specific, because eager correctness in the same runs still reports `max_abs=0.000488281`;
  - 512-only passing while 512 in a multi-token check can fail suggests sequence/state sensitivity, but 1024-only failure proves the problem is not solely caused by `512 -> 1024` replay ordering.
- Current hypothesis candidates:
  - tail signal/reset state not fully graph-safe across repeated captured replays;
  - runtime active tile or tail runtime offset argument/state visibility race;
  - missing graph-safe synchronization around tail-reduce completion visibility;
  - captured launch parameter storage reuse in the normal tail ASM path.
- Status:
  - do not mark normal tail graph correctness as fully closed;
  - future fix should use this repeat script as the minimal recurrence harness before changing ASM/host logic.

## 2026-06-16 V3 normal graph multi-token replay fix

- The intermittent normal graph replay failure was caused by K1, not by K3 tail math:
  - V3 normal graph uses the isolated K1 ASM-pack5 wrapper;
  - K1 ASM graph relies on clearing `staged_flags` and `meta_flags` before each captured replay;
  - `large_opt.py` only passed `k1_graph_reset_layout` for `v3_backend is None`, so V3 normal replay reused stale ready flags from earlier replays.
- The captured `flag_generation` is fixed inside the graph. Without clearing the flag buffers, a later replay can observe a previous replay's generation-ready state and enter staging/GEMM early, producing sequence-dependent large diff.
- Fix:
  - apply K1 graph flag/meta reset to `v3_backend in (None, "normal")`;
  - keep V3 LL out of this reset path because LL C K1 initializes its own route counts and metadata inside the kernel;
  - remove the temporary no-tail K3 done-counter publish/probe experiment and the `MEGAMOE_DCU_DEBUG_CLEAR_GRAPH_COMBINE` test diagnostic.
- Verification:
  - normal no-tail graph `512,1024` replay=5 passed, both tokens `max_abs=0.000488281`;
  - normal tail graph repeat harness `512,1024` passed 4/4 after the fix, all 512/1024 checks `max_abs=0.000488281`;
  - cleanup replay times stayed in the expected range: normal no-tail `512/1024 = 1.672859/2.006139 ms`, normal tail `512/1024 = 1.684199/1.994919 ms`;
  - V3 LL graph dynamic/uneven regression passed for uniform capture1024 replay `8..1024` and uneven128 replay `32/96/128`, no-tail and tail.
- Planning impact:
  - the previously reopened normal tail graph intermittent correctness bug is now closed by the K1 reset fix;
  - future normal graph regressions should first check whether any new path bypasses the K1 graph flag/meta reset contract.

## 2026-06-16 K1 >4GB and K3 remote-combine locality findings

- K1 `symm_x_span > UINT32_MAX`:
  - 4096 can benefit from `_v3_normal_symm_warmup_enabled` because warmup can move the first production symm buffer below the 4GB span limit;
  - 8192 remains above 4GB even with warmup, so it exercises the K1 ASM bit2 `{rank-local offset, source_rank}` path;
  - 8192 bit2 MUBUF path is much faster than absolute pointer/global-load (`~5.33 ms` vs `~6.40 ms` in the K1-only A/B), so switching >4GB to absolute pointers is not a valid optimization.
- K1 ASM hot path:
  - route emits row metadata through atomics, so rows inside a 256-row expert tile are not source-rank grouped;
  - stage loop reads `{offset, source_rank}` from LDS and uses `source_rank` to load the peer base before MUBUF source loads;
  - the pack5 code object already uses `sgpr_count=102` and `vgpr_count=255`, so caching all peer SRDs in spare SGPRs is not obviously safe.
- K3 remote combine locality:
  - dest-sorting rowptrs is harmful: 2048 remote-only `1.325 ms` vs dest-sorted `1.989 ms`; 8192 remote-only `5.981 ms` vs dest-sorted `8.074 ms`;
  - rank-bucket compact rowptrs are not a stable win: 2048/4096 are noise-level positive, while 8192 is inconsistent across repeated probes;
  - current ASM uses half-tile LDS staging and vectorized `global_store_dwordx4` combine stores, so preserving tile/store cadence matters more than simple destination-rank locality.
- Design implication:
  - do not implement a K3-only rowptr reorder/rank-bucket production change;
  - any future locality optimization must be a K1/K2/K3 contract change that preserves row/data/expert alignment, likely via source-rank or destination-rank grouped row emission, and must prove 2048/4096/8192 no-tail/tail correctness and performance.

## 2026-06-16 K1 compact prebuild should use bit2, not absolute pointers

- The previous compact prebuild fallback for `symm_x_span > UINT32_MAX` encoded `row_x_ptrs` as absolute 64-bit pointers and set `reserved_c0` bit1.
- A/B evidence showed this is the wrong production fallback:
  - asm-route bit2 `{rank-local offset, source_rank}` at 8192 was about `5.33 ms`;
  - absolute pointer source-load path at the same rows was about `6.40 ms`;
  - compact absolute prebuild was about `6.50 ms` even though it had fewer rows.
- Production fix:
  - compact prebuild now writes the same bit2-compatible packed metadata as asm-route for >4GB spans;
  - host no longer sets absolute-pointer bit1, including graph-forced compact capture;
  - source guard rejects `use_absolute_x_ptrs` returning to `k1_fused_ext.cu`.
- Verification after the fix:
  - K1-only compact 8192 with `v3_symm_span_gt_u32=true` is `5.156 ms` median, rows `54272`;
  - compact 8192 no-tail e2e correctness passes with `max_abs=0.000488281`, fused `10.667 ms`;
  - normal graph no-tail/tail replay 512/1024 remains correct and in the expected `~1.62-1.95 ms` replay range.
- Design implication:
  - absolute pointer should not be considered a performance fallback for V3 normal K1;
  - if the residual ASM bit1 branch is removed later, it should be treated as code-object dead-branch cleanup after source-level host guards have been stable, not as a separate performance optimization.

## 2026-06-16 8192 auto compact improves e2e mostly through K3 row pressure

- After compact prebuild switched from absolute pointers to bit2 `{rank-local offset, source_rank}`, the old auto heuristic was stale:
  - old auto selected asm-route at 8192 because compact absolute had been slower;
  - with bit2 compact, K1-only auto/compact both use rows `54272` and run at about `5.16-5.19 ms`.
- Full e2e no-tail evidence:
  - current `K1_PREBUILD_MODE=auto`: correctness passes, fused `10.792 ms`;
  - forced `K1_PREBUILD_MODE=compact`: correctness passes, fused `10.859 ms`;
  - forced old `K1_PREBUILD_MODE=asm`: correctness passes, fused `12.712 ms`.
- K3 split explains why the e2e improvement is larger than the K1-only delta:
  - asm rows `57344`: K3 ASM-pack5 `staged_rowptr` about `5.943 ms`;
  - compact rows `54272`: K3 ASM-pack5 `staged_rowptr` about `4.671 ms`;
  - K3 alone accounts for about `1.27 ms` of the e2e improvement, while K1-only accounts for about `0.25 ms`.
- Interpretation:
  - the 8192 improvement is a row-contract/downstream effect, not only a K1 source-load optimization;
  - compact rows reduce K3 GEMM/rowptr/remote-combine work enough to justify making compact/bit2 the 8192 auto choice;
  - future 8192 work should always report K1-only, K3 split, and full e2e together, because K1 route capacity changes can shift K3 cost substantially.

## 2026-06-16 K1 auto should require both relative and absolute row savings

- The old auto rule was too one-dimensional because it only looked at fractional tile saving.
- Current production heuristic should encode the two observed regimes:
  - use asm-route when there is no meaningful row reduction, because it avoids compact prebuild overhead;
  - use compact when it materially reduces local rows, because the e2e benefit can be dominated by downstream K3 rowptr/remote-combine work rather than K1 alone.
- The refined rule therefore requires:
  - estimated per-expert tile saving ratio `>= 5%`;
  - estimated local tile saving `>= 8 tiles`.
- This keeps obvious no-gain cases on asm (`256/512/1024/2048/2050`) while retaining compact for row-pressure cases (`1025/4096/4097/8192`).
- The ASM absolute pointer bit1 path is no longer a valid fallback:
  - host no longer sets bit1;
  - source guard rejects host/ASM absolute branch reintroduction;
  - ASM route emit and stage load now only distinguish default uint32-offset MUBUF and bit2 `{rank-local offset, source_rank}` MUBUF.
- Perf verification is still required after card contention clears, but the static/codegen direction is now simpler: bit1 has no production semantics.

## 2026-06-16 V3 LL graph to-256 refresh and K2 fusion feasibility

- Refreshed V3 LL graph replay with capture bucket `1024` and replay tokens `8,32,64,128,256`:
  - no-tail replay: `0.571/0.670/0.721/0.853/1.267 ms`;
  - tail replay: `0.770/0.869/0.917/1.046/1.442 ms`;
  - both modes were correctness-clean, and no remote KFD test process remained after the run.
- Current staged LL boundary:
  - K1 returns `l1_out`, `route_weights`, `m_indices`, `output_index`, and `row_combine_ptrs`;
  - K2 consumes `l1_out` plus row metadata, applies SwiGLU, route weight, clamp, row-wise amax reduction, and FP8 quantization;
  - K3 consumes `act_fp8` and `act_scale`.
- Fusing K2 into K1 is possible only as a K1/K2 contract-level kernel redesign, not as a small wrapper cleanup:
  - SwiGLU needs both halves of K1 output (`gate` and `up`) for the same row/channel;
  - FP8 scale needs a reduction across the full hidden row;
  - current K1 GEMM is tiled over N, so a K1 epilogue would need cross-tile scale aggregation or an equivalent staging pass before quantization.
- Expected benefit is limited unless profiling proves otherwise:
  - graph replay already removes most CPU launch overhead;
  - historical LL stage timing puts K2 around `0.028 ms` for 32/128 tokens, while K1 and K3 dominate;
  - likely savings are the K2 graph node, `l1_out` BF16 write/read traffic, and the small K2 device time, but a heavier K1 epilogue could reduce occupancy and erase the gain.
- Recommendation:
  - keep K2 separate for the current production path;
  - only revisit K2-in-K1 after profiler evidence shows K2 plus `l1_out` traffic is a meaningful fraction of LL replay time at 32/64/128/256;
  - any prototype must prove no regression for LL no-tail/tail, eager/graph, uniform/uneven, and the current small-token performance envelope.

## 2026-06-16 V3 LL tail reduce should consume runtime tokens in graph replay

- The earlier LL graph tail slowdown was not a fundamental tail-reduce limit; K3 LL tail reduce was still looping over capture `num_tokens` during replay.
- Production fix threads graph runtime token tensor through `large_opt.py -> k3_fused.py -> k3_v3_fused_ext.cu -> V3_K3_LowLatencyMaskedGroupGemmKernel`.
- The device reducer now clamps `effective_num_tokens` from the runtime tensor and uses it for `total_reduce_vecs`, so capture1024/replay256 reduces 256 tokens, not 1024.
- Refreshed capture1024 replay results after the fix:
  - no-tail `8/32/64/128/256/512 = 0.570/0.666/0.714/0.839/1.267/2.200 ms`;
  - tail `8/32/64/128/256/512 = 0.553/0.649/0.706/0.849/1.283/2.241 ms`;
  - all graph buckets were correctness-clean.
- K2 remains graph-shape fixed, but graph path passes `row_combine_ptrs` and K2 kernels early-return on inactive rows. Fully shrinking K2 grid is a larger graph-update/compact-row-list design item, not a safe local fix.

## 2026-06-17 Normal eager true-uneven K1 capacity contract

- Normal eager uniform path can safely size K1 route capacity from local `num_tokens`, because every peer rank contributes the same number of routed rows.
- True uneven path cannot use local `num_tokens` as the owner-rank capacity bound:
  - each owner rank receives routes for its local experts from all source ranks;
  - a small-token owner rank can still receive rows from an 8192-token source rank;
  - therefore local `y.size(0)` can under-estimate `capacity_total_tasks` and truncate K1 rows.
- Graph normal avoids this class because graph K1 is captured with a fixed
  bucket capacity and already uses graph-safe compact prebuild; replay changes
  the runtime token scalar, not the host-side route capacity.
- The production eager contract is a host-side capacity/global-max token scalar:
  - callers may pass `capacity_num_tokens=max(tokens_per_rank)` for the current
    EP group request;
  - K1 uses that value only to size host-side route capacity and compact
    metadata, while the kernels still read each source rank's actual token count
    from symmetric-buffer metadata;
  - it is not a backend selector and is not passed to graph replay;
  - this avoids adding a new kernel or D2H sync.  Falling back to
    `num_max_tokens_per_rank` is correctness-safe but can over-expand eager K1
    work versus the current request's real global max.
- Large uneven regression harness:
  - `--num-max-tokens-per-rank 8192 --num-tokens-per-rank-list 8192,4097,4096,3072,2050,2048,1025,256 --megamoe-backend normal`;
  - both `K3_USE_ASM_TAIL_REDUCE=0` and `1` must remain correctness-clean.
# 2026-06-17 graph replay token 与测试侧 setup token

- graph 生产语义：capture capacity 来自 `num_max_tokens_per_rank` / `cuda_graph_max_tokens_per_rank`；replay 的实际 token 前缀来自 device scalar `sym_buffer.cuda_graph_num_tokens`。在测试脚本中，这个 scalar 由 `--cuda-graph-test-tokens` 的每个 bucket 写入。
- `--num-tokens` 在 graph bucket sweep 中不应被理解为 replay token；它主要影响测试脚本的普通 eager correctness 输入大小、auto selector 展示，以及未提供 `--cuda-graph-test-tokens` 时的默认 bucket。
- 当前测到 LL graph capture8192 replay8：当 graph 性能命令使用 `--num-tokens 8192` 时稳定约 `0.556 ms`；使用 `--num-tokens 513` 时可出现约 `0.607 ms`。两者 replay token 都是 8，差异来自测试脚本辅助 tensor/权重分配顺序对小 token 计时的地址/allocator 扰动，不是 graph runtime-token 语义差异。
- README 的 graph 性能示例使用 `--num-tokens 8192`，并明确实际 replay work 由 `--cuda-graph-test-tokens` 控制。
- K2 graph capture8192 小 token 固定开销通过现有 K2 reg kernel 的 grid-stride row loop 与内部 `max_row_blocks=2048` 收敛；未新增 kernel，未新增 D2H sync。

## 2026-06-18 V3 normal/LL weight layout 分叉不是生产 ABI

- 现象：normal 路径使用 `flatten_pack5_weight_asm_normal()`，LL 路径使用 `flatten_pack5_weight()`；框架侧如果按此接入，需要同时持有两套 pack5 weight，显存成本不可接受。
- 根因：V3 normal ASM 初版为了复用原 ASM 的 pack5 lane 地址/accumulator store schedule，使用 plain `ni` order；LL/C pack5 为了匹配 B load 与 MMAC lane 映射，使用 transposed physical-`ni` order。两者都是同一 5pack tile nesting，但 `ni` 子维顺序不同。
- 约束：统一 layout 不允许通过 runtime weight transpose kernel 解决；不能新增 kernel，不能引入 D2H sync，性能不能劣化。
- 已证伪的简单方案：把统一 ABI 改成 normal/plain layout，并在 LL C kernel 直接按 mapped row 读 B，功能正确但 LL 小 token 性能明显回退；这是 B load coalescing/访存形态问题，不是单点 token 抖动。
- 已证伪的补救方案：plain layout 下保留连续 B load，并在寄存器中对 B 做 lane shuffle，正确性通过但 LL `32/128/256/512` 仍比 transposed layout 慢约 `7-9%`。额外 B shuffle 是稳定热循环成本，不适合作为生产路径。
- 当前方向：统一 ABI 回到 LL transposed layout；normal ASM 只需在 pack5 offset 初始化处把 logical `ni` 映射为 physical `ni=((ni&3)<<2)+(ni>>2)`，该映射不在 GEMM 主循环内，风险和性能成本都低于让 LL 热循环承担 B shuffle。
- 当前源码收敛：`v3_layout.py` 只保留 `pack5_weight()` / `flatten_pack5_weight()` 这一套 public helper；`pack5_weight_asm_normal()` / `flatten_pack5_weight_asm_normal()` 已删除，避免框架接入时再次误解为 normal/LL 两套权重 layout。
- 2026-06-18 复核：LL transposed single ABI + normal ASM global-offset remap 后，LL eager 32/512 为 `0.637/2.275 ms`，normal 512 no-tail/tail 为 `1.746/1.775 ms`，normal 4096 tail 为 `6.260 ms`；功能均正确。
- local-read/LDS fragment remap 在 normal 上正确但慢，512 tail `2.224 ms`、4096 tail `7.674 ms`、4096 no-tail `7.589 ms`。该方向会破坏 ASM 原有 MMAC/LDS fragment 性能形态，不再作为生产候选。
- 2026-06-18 额外反证：只把离线 pack5 helper 临时改成 plain `ni`，LL C kernel 保持连续 B load 和原 store 合同不动，LL 32 correctness 失败，`max_abs=0.089599609375`。因此 plain ABI 不能通过“不改 LL 热循环、只复用当前 coalesced load”直接成立。
- plain ABI 的输出列 remap 补救会把 K1/K3 当前 `bf16x4` 连续写退化为 permutation 下的非连续标量/散写；K3 remote combine 当前依赖向量 store cadence，历史上 lane-pair/vector-store/shuffle 类改动也多次显示额外 shuffle/pack/store 调度成本会吃掉收益。除非后续有新的 MMAC C-fragment layout 证据，否则不作为生产候选继续推进。
- normal ASM 恢复 transposed layout 下全局连续读的低风险点暂未找到：当前 K1/K3 ASM 的 pack5 权重路径一部分是 `buffer_load_dwordx4 ... lds` 通过 `m0` 固定步进直接落 LDS，另一部分是 `vgprG2LA` 打包后 `ds_write_b128`；global offset、LDS 写入位置、local-read/MMAC fragment 顺序共同构成合同。若把 global read 改回连续 physical `ni`，仍需在 LDS 写入或 fragment 读出阶段恢复 logical 顺序，这会落入已证伪的 local-read/LDS fragment remap 风险域。
- graph 性能命令若用 `--num-tokens 8192` 同时要求 DeepGEMM baseline，会在测试侧分配 baseline 中间张量并可能 OOM；这不是 graph replay 或 weight ABI 的 correctness 问题。graph replay correctness 应用较小 setup token 覆盖，性能用 skip-baseline 模式或 README 约定脚本单独跑。
- ASM offset patch 寄存器安全复核：K1/K3 三个 PACK5 ASM 里新增映射使用 `v10` 做 scratch；该位置后续原本就以 `GLOBAL_OFFSET_A ..., 10` 形式把 `v10` 作为地址计算临时寄存器使用，因此当前 patch 没有引入新的长期 live-range，也没有触碰主循环 accumulator / epilogue store schedule。

## 2026-06-18 Phase 14 single ABI full matrix conclusion

- LL transposed single ABI + normal ASM global-offset `ni` remap 的完整功能矩阵已通过：
  - LL eager/graph 覆盖 `8,32,33,64,128,129,256,257,512,513`；
  - normal eager/graph 覆盖 `256,512,1024,1025,2048,2050,3072,4096,4097,8192`；
  - LL/normal eager uneven、LL/normal graph uneven、README script/graph smoke 均通过。
- LL 性能基本守住历史基线：
  - eager `8/32/128/256/512 = 0.600/0.643/0.846/1.260/2.294 ms`；
  - graph capture8192 replay `8/32/128/256/512 = 0.562/0.652/0.844/1.298/2.275 ms`。
- normal 性能仍是当前 single ABI 的主要未达标点：
  - eager `4096/8192 = 6.283/11.776 ms`；
  - graph capture8192 `4096/8192 = 6.107/11.355 ms`；
  - 对比 Phase 13/latest matrix 旧双-layout normal 约 `5.83/10.88 ms`，说明 single ABI 功能已通但 normal 吞吐尚未完全恢复。
- 当前不能把 normal gap 归因于 graph 特有问题：
  - eager 和 graph 都比旧双-layout normal 慢；
  - graph 相对当前 eager 在 3072+ 反而略好；
  - 因此主要问题是 normal ASM 消费 LL-transposed weight layout 后的 global/LDS/fragment 访存合同，而不是 K2 graph 或 replay runtime-token。
- 后续优化边界：
  - 不再重复 plain layout + LL direct mapped load、plain layout + LL B-side shuffle、plain layout + 不改 LL 合同、transposed layout + normal local-read/LDS fragment remap；
  - 若继续优化，必须围绕 normal ASM 的 global-load coalescing、LDS write remap 或共同 layout 设计展开，并用 512/4096/8192 + LL 32/512 双侧性能门槛验收；
  - 任何方案不得新增 kernel、不得引入 D2H sync，且不能让 LL 热循环承担稳定 shuffle 成本。

## 2026-06-18 Phase 14 code-object metadata and normal gap attribution

- 远端 code object metadata 复核：
  - K1 dispatch-pull pack5：`SGPR=102`、`VGPR=255`、`spill=0`、`LDS=65536`、`.text=78028 bytes`；
  - K3 combine pack5：`SGPR=102`、`VGPR=255`、`spill=0`、`LDS=65536`、`.text=84820 bytes`；
  - K3 combine tail-reduce pack5：`SGPR=102`、`VGPR=255`、`spill=0`、`LDS=65536`、`.text=89856 bytes`。
- 这说明当前 normal single ABI gap 不是由新增 `logical ni -> physical ni` 映射导致的 register pressure、spill 或 LDS 占用变化造成；三份 ASM 仍维持原有 255 VGPR / 0 spill / 64 KiB LDS 的资源边界。
- 更可信的性能来源是 weight global-load locality：normal ASM 原 plain pack5 对 `ni` 子维连续取数；单 ABI 使用 LL-transposed layout 后，normal lane 必须按 physical `ni=((ni&3)<<2)+(ni>>2)` 取数，访问顺序从连续 `0..15` 变成跨组 `0,4,8,12,1,5,9,13,...`，影响 global memory coalescing/预取形态。
- 进一步优化如果要恢复 normal 吞吐，需要在不动 LL hot loop 的前提下重新设计 normal 的 global-load/LDS-write/fragment 合同；这比继续试局部 local-read shuffle 更接近根因，但风险也更高，必须用 LL+normal 双侧矩阵验收。

## 2026-06-18 Phase 14 store-side remap 反证

- 尝试过“normal ASM 连续 physical `ni` global load + epilogue/store physical->logical remap”的候选，目标是恢复 normal 旧 plain-layout 下的 global-load locality。
- 该候选不是性能不佳，而是 correctness 直接失败：normal tail 4096 `max_abs=0.114990234375`，远超现有容忍；回退到当前 global-offset remap 后同 case `max_abs=0.000488281`。
- 这说明当前 normal ASM 的 pack5 权重顺序不能只在输出 store 端修正；B operand 的 global load 顺序、LDS 落点、local-read/MMAC fragment 以及 scale/epilogue 共同构成合同。
- 当前 `buffer_load_dwordx4 ... lds` 路径的 LDS destination 由 scalar `m0` 控制，不能像 VGPR offset 那样廉价地做 per-lane LDS write permutation。要恢复连续 global load，需要更完整的 LDS/fragment 合同设计，而不是单点 offset/store patch。
- 因此本方向与此前 local-read/LDS fragment remap、plain ABI + LL shuffle 一样标为反证；后续优化 normal 只能基于 profiler/ISA 或明确的 source-backed MMAC fragment 证据继续。

## 2026-06-18 CUDA MegaMoE 单 ABI 参考边界

- 本工程 CUDA MegaMoE 对外只使用一套 weight tensor-map/descriptor 合同：L1/L2 根据 phase 选择对应 descriptor，内部 scheduler 和 epilogue 用同一套 workspace/source metadata 做 dispatch/combine。
- CUDA epilogue 会先按 tensorcore/STSM 友好的 shared-memory layout 写入，再由远端 combine writer 按另一套 shared-memory 读法取出并写 peer buffer；这证明“内部 fragment layout 可与最终 store layout 不同”，但它依赖 SM100 TMA/TMEM/STSM 合同，不能直接作为 DCU `buffer_load ... lds` + MMAC lane permutation 的证据。
- 对 DCU 的可借鉴点仅限接口原则：框架只应持有一套 transformed weight ABI，normal/LL 内部各自适配该 ABI；不能把 CUDA 的 shared-memory reorder 机制直接移植为 DCU ASM patch。
- 当前 DCU single ABI 选择 LL-transposed layout 是因为它守住 LL 性能；normal 的剩余性能 gap 应通过 DCU-specific global-to-LDS/MMAC fragment 合同继续找证据，而不是回到双 layout 或让 LL 热循环承担稳定 shuffle。

## 2026-06-18 Phase 14 accepted single ABI finding

- 当前接受版本是 LL-transposed pack5 作为唯一 weight ABI，normal ASM 在 pack5 global offset 初始化处做 `logical ni -> physical ni` 映射；该映射不进入 GEMM 主循环，且不改变 LL hot loop。
- 该版本功能矩阵已覆盖 LL/normal eager/graph、uniform/uneven 与 README smoke；normal 4096 约 `6.2 ms`，相比旧双-layout normal 最优有小幅性能代价，但换来框架侧不再持有两套 L1/L2 weight。
- 后续若继续追 normal 性能，必须建立在 profiler/ISA/source-backed MMAC fragment 合同证据上；不得回到双 weight ABI，也不得重复已反证的 LL 热循环 shuffle、normal local-read/LDS remap 或 store-side remap。

## 2026-06-18 extended-size normal finding

- 扩展 normal eager sweep 显示，当前 single ABI 在已对齐历史基线的 size 上劣化大致随 token 变大而加重：256/512/1024 约 `+1-2%`，2048/3072 约 `+5-6%`，4096/8192 约 `+8%`。
- 5120 uniform normal 曾暴露 correctness blocker：cap5120/cap8192、tail/no-tail、K1 `asm/compact` prebuild 都失败，而 4096、6144、7168、8192 通过。该问题最终定位为 K1 route capacity 固定 slack 在 5120 点恰好卡到 1024 rows/expert 边界，随机路由可越界掉行。
- 已修复：K1 执行侧和 route_scratch 估算侧统一使用基于 cap/capacity tokens 的动态 rows/expert headroom。5120 tail/no-tail correctness 通过，5120 tail perf 为 `8.182 ms`，LL 512/513 smoke 未受影响。
- 当前风险边界：single ABI arbitrary normal token size 的主要 correctness blocker 已清除；后续仍需在完整 release matrix 中继续覆盖非 2k/4k 边界 token（如 5120、6144、7168）以防新的容量或 layout 边界回归。

## 2026-06-19 dual layout becomes default performance policy

- 用户重新放宽“框架只保存一套 weight”的约束：PD 分离或显存允许时可以加载两份 weight，因此默认策略应改回 LL/normal 双 layout，并以各自最佳性能为目标。
- 设计结论：
  - 默认双 layout 是性能路径：LL 用 transposed pack5，normal ASM 用 plain pack5；
  - unified/single layout 仅作为兼容/显存路径，通过 `MEGAMOE_DCU_UNIFIED_WEIGHT_LAYOUT=1` opt-in；
  - 这比继续在单 ABI 下补 normal global-load/LDS 合同更稳，因为现有实测已经证明 default dual layout 直接恢复历史最佳性能档位。
- 代表数据：
  - default dual eager：LL 32/512 `0.641/2.297 ms`，normal 4096/8192 `5.798/10.886 ms`，normal 5120 correctness/perf 正常 `7.403 ms`；
  - default dual graph capture8192：LL replay 32/512 `0.661/2.258 ms`，normal replay 4096/8192 `5.687/10.485 ms`；
  - unified opt-in：normal 4096 eager/graph replay `6.274/6.180 ms`，功能正确但仍慢于 default dual。
- 后续 layout 挖掘边界：
  - 可以把 LL/normal kernel/code object 完全隔离，不再为了复用牺牲性能；
  - 仍可以重新设计各自 layout 或共同 layout，但任何新候选必须不低于当前 default dual 代表性能；
  - 已反证的 plain-layout LL shuffle、normal store-side remap、normal local-read/LDS fragment remap 不再重复；
  - 不新增 runtime weight transform kernel，不引入 D2H sync；layout 变化仍应发生在离线/fixture/框架权重准备阶段。

## 2026-06-19 dual layout re-exploration finding

- 第一轮重新挖掘没有发现比当前默认双 layout 更安全的立即替换方案。当前 dual layout 不是随意分叉，而是分别贴合两条热路径：
  - LL C pack5 使用 transposed physical-`ni`，匹配现有 B operand load、MMAC lane 映射和向量 store 合同；
  - normal ASM 使用 plain `ni`，匹配原 ASM 对 pack5 weight 的连续 global load / LDS feed 形态。
- DCU KB 与 Hygon optimizer 证据都指向同一条原则：matrix-load / LDS / MMAC fragment 是一个整体合同。单独改 weight global offset、local-read 顺序或 epilogue store 顺序，很容易出现 correctness fail 或吞吐回退；这与此前 4096 store-side remap fail、LL plain layout direct-load fail、LL register shuffle 慢的实验一致。
- 本工程 CUDA MegaMoE 的可借鉴点是接口与 workspace 组织，不是具体 layout remap。CUDA 侧 tensor-map / TMA / STSM 可以把内部 fragment layout 和最终 store layout 分开；DCU 当前 ASM 使用 `buffer_load_dwordx4 ... lds`、`ds_read_b128` 与手写 MMAC fragment 约束，不能把 CUDA reorder 机制直接移植。
- 因此后续 layout 优化的进入条件应提高：
  - LL 新 layout 必须先证明不破坏 B operand load/MMAC/store 合同，且 LL `32/512` 不低于当前 `~0.64/~2.29 ms`；
  - normal 新 layout 必须先证明 contiguous global load 与 LDS/MMAC fragment 同时闭合，且 normal `512/4096/5120/8192` 不低于当前 `~1.75/~5.8/~7.4/~10.9 ms`；
  - 共同 layout 只有在 profiler/ISA/source-backed 证据足够强时才恢复，不再靠单点 permutation 试错。
- source-backed 候选边界：`ds_read_m32x16_b16_normalxalt.cpp`、`altxalt.cpp`、`swizzle.cpp` 说明 DCU 上 normal/alt/sizzle 的 LDS matrix-read 合同是可表达的，但需要同时设计 LDS write layout、read offset、operand fragment 解释和 C store。它可以作为后续“normal ASM 新 code object”方向的参考，不适合作为当前默认 PACK5 ASM 的局部热补丁。
