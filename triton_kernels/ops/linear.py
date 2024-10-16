import torch
import triton
import triton.language as tl

@triton.jit
def triton_linear(a_ptr, b_ptr, c_ptr, out_ptr,
                    M, N, K,
                    stride_am, stride_ak, 
                    stride_bk, stride_bn,
                    GROUP_M : tl.constexpr,
                    EVEN_K : tl.constexpr,
                    ALLOW_TF32 : tl.constexpr,
                    ACC_TYPE : tl.constexpr,
                    B_PROLOGUE_CAST_TYPE : tl.constexpr,
                    BLOCK_M : tl.constexpr,
                    BLOCK_N : tl.constexpr,
                    BLOCK_K : tl.constexpr,
                    ):

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        offset_am = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    else:
        offset_am = offset_m % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        offset_bn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    else:
        offset_bn = offset_n % N
    offset_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offset_k[None, :] < k, other=0.)
            b = tl.load(b_ptrs, mask=offset_k[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # rematerialize offset_m and offset_n to save registers
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = offset_m[:, None]
    idx_n = offset_n[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (N*idx_m)
    c = tl.load(c_ptr + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    out = acc + c
    tl.store(out_ptr + (tl.broadcast_to(xindex, mask.shape)), out, mask)