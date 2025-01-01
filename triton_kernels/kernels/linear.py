# https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh


def _linear_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit
def gelu(x):
    c = 0.7978845608028654  # sqrt(2 / pi)
    x_cubed = x * x * x
    tanh_arg = c * (x + 0.044715 * x_cubed)
    tanh_result = tanh(tanh_arg)
    return 0.5 * x * (1 + tanh_result)


@triton.jit(launch_metadata=_linear_launch_metadata)
def _linear_fwd(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_bn[None, :]
        bias = tl.load(bias_ptrs).to(tl.float32)
        bias = tl.broadcast_to(bias, [BLOCK_SIZE_M, BLOCK_SIZE_N])
        accumulator += bias

    c = accumulator.to(tl.bfloat16)

    if ACTIVATION == "GELU":
        c = gelu(c)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def linear(x, w, b, activation=""):
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        }
    }
    # Check constraints.
    assert x.shape[-1] == w.shape[-1], "Incompatible dimensions"
    assert x.dtype == w.dtype, "Incompatible dtypes"
    N, K = w.shape
    input_shape = x.shape
    x = x.reshape(-1, K)
    M, K = x.shape
    dtype = x.dtype

    y = torch.empty((M, N), device=x.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    _linear_fwd[grid](
        x,
        w,
        y,
        b,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(1),
        w.stride(0),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=False if b is None else True,
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],
        num_stages=configs[dtype]["num_stages"],
        num_warps=configs[dtype]["num_warps"],
        ACTIVATION=activation,
    )
    y = y.reshape(*input_shape[:-1], -1)
    return y
