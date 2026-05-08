import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_prefix_kernel(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset = tl.arange(0, BLOCK_EXPERT_NUM)
    counts = tl.load(num_recv_tokens_per_expert + offset, mask=offset < num_experts, other=0)
    cumsum = tl.cumsum(counts) - counts
    tl.store(expert_start_loc + offset, cumsum, mask=offset < num_experts)

    cur_start = tl.load(expert_start_loc + cur_expert)
    cur_count = tl.load(num_recv_tokens_per_expert + cur_expert)
    offs = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_count, BLOCK_E, num_stages=4):
        tl.store(m_indices + cur_start + start_m + offs, cur_expert)


@triton.jit
def _scatter_copy_kernel(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    recv_weights,
    recv_weights_stride0,
    recv_weights_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    route_weights,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offs_h = tl.arange(0, HIDDEN_SIZE_PAD)
    mask_h = offs_h < HIDDEN_SIZE
    offs_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offs_s < SCALE_HIDDEN_SIZE

    for token_i32 in range(start_token, total_token_num, grid_num):
        token = token_i32.to(tl.int64)
        x = tl.load(recv_x + token * recv_x_stride0 + offs_h, mask=mask_h, other=0.0)
        x_s = tl.load(
            recv_x_scale + token * recv_x_scale_stride0 + offs_s * recv_x_scale_stride1,
            mask=mask_s,
            other=1.0,
        )

        for slot_i32 in tl.range(0, topk_num, 1, num_stages=4):
            slot = slot_i32.to(tl.int64)
            expert = tl.load(recv_topk + token * recv_topk_stride0 + slot)
            if expert >= 0:
                dst_i32 = tl.atomic_add(expert_start_loc + expert, 1)
                dst = dst_i32.to(tl.int64)
                tl.store(output_index + token * output_index_stride0 + slot, dst_i32)
                tl.store(route_weights + dst, tl.load(recv_weights + token * recv_weights_stride0 + slot))
                tl.store(output_tensor + dst * output_tensor_stride0 + offs_h, x, mask=mask_h)
                tl.store(
                    output_tensor_scale
                    + dst * output_tensor_scale_stride0
                    + offs_s * output_tensor_scale_stride1,
                    x_s,
                    mask=mask_s,
                )


@triton.jit
def _gather_kernel(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    APPLY_WEIGHTS: tl.constexpr,
):
    cur_block = tl.program_id(0).to(tl.int64)
    start_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    offs = tl.arange(0, BLOCK_D)

    for token_i32 in range(start_token, total_token_num, grid_num):
        token = token_i32.to(tl.int64)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for slot_i32 in range(0, topk_num):
            slot = slot_i32.to(tl.int64)
            expert = tl.load(recv_topk_ids + token * recv_topk_ids_stride0 + slot)
            src_i32 = tl.load(input_index + token * input_index_stride0 + slot)
            if expert >= 0 and src_i32 >= 0:
                src = src_i32.to(tl.int64)
                v = tl.load(input_tensor + src * input_tensor_stride0 + cur_block * BLOCK_D + offs)
                if APPLY_WEIGHTS:
                    w = tl.load(recv_topk_weight + token * recv_topk_weight_stride0 + slot)
                    v *= w
                acc += v.to(tl.float32)
        tl.store(output_tensor + token * output_tensor_stride0 + cur_block * BLOCK_D + offs, acc)


@torch.no_grad()
def ep_scatter_channelwise(
    recv_x,
    recv_x_scale,
    recv_topk,
    recv_topk_weights,
    num_recv_tokens_per_expert,
    total_rows,
):
    block_e = 128
    hidden = recv_x.shape[1]
    topk = recv_topk.shape[1]
    num_experts = num_recv_tokens_per_expert.shape[0]
    scale_hidden = max(hidden // 128, 1)

    total_rows = max(((int(total_rows) + block_e - 1) // block_e) * block_e, block_e)
    grouped_x = torch.empty((total_rows, hidden), device=recv_x.device, dtype=recv_x.dtype)
    grouped_x_scale_2d = torch.empty((total_rows, scale_hidden), device=recv_x.device, dtype=torch.float32)
    route_weights = torch.zeros((total_rows,), device=recv_x.device, dtype=torch.float32)
    m_indices = torch.full((total_rows,), -1, device=recv_x.device, dtype=torch.int32)
    output_index = torch.full_like(recv_topk, -1, dtype=torch.int32)
    expert_start_loc = torch.empty((num_experts,), device=recv_x.device, dtype=torch.int32)

    if recv_x_scale.dim() == 1:
        recv_x_scale_2d = recv_x_scale.view(-1, 1)
        if scale_hidden > 1:
            recv_x_scale_2d = recv_x_scale_2d.expand(-1, scale_hidden).contiguous()
    else:
        recv_x_scale_2d = recv_x_scale.contiguous()

    _scatter_prefix_kernel[(num_experts,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        num_warps=8,
    )
    _scatter_copy_kernel[(min(recv_topk.shape[0], 1024 * 8),)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x_scale_2d,
        recv_x_scale_2d.stride(0),
        recv_x_scale_2d.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        recv_topk_weights,
        recv_topk_weights.stride(0),
        recv_topk_weights.stride(1),
        grouped_x,
        grouped_x.stride(0),
        grouped_x_scale_2d,
        grouped_x_scale_2d.stride(0),
        grouped_x_scale_2d.stride(1),
        route_weights,
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=topk,
        HIDDEN_SIZE=hidden,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden),
        SCALE_HIDDEN_SIZE=scale_hidden,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden),
        num_warps=8,
    )
    return grouped_x, grouped_x_scale_2d[:, 0].contiguous(), route_weights, m_indices, output_index


@torch.no_grad()
def ep_gather_channelwise(
    l2_out,
    recv_topk_ids,
    recv_topk_weights,
    output_index,
    recv_y,
    apply_topk_weights=False,
):
    hidden = recv_y.shape[1]
    block_d = 128 if hidden % 1024 != 0 else 1024
    assert hidden % block_d == 0
    _gather_kernel[(triton.cdiv(hidden, block_d), min(recv_y.shape[0], 1024))](
        recv_y.shape[0],
        l2_out,
        l2_out.stride(0),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weights,
        recv_topk_weights.stride(0),
        recv_topk_weights.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        recv_y,
        recv_y.stride(0),
        topk_num=recv_topk_ids.shape[1],
        BLOCK_D=block_d,
        APPLY_WEIGHTS=apply_topk_weights,
        num_warps=2,
    )
