import torch
import triton
import triton.language as tl

from triton_kernels.kernels.utils import calculate_settings


@triton.jit
def _layer_norm_modulation_fwd(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    seq_len,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    batch_idx = row // seq_len
    Y += row * stride
    X += row * stride
    W += batch_idx * stride
    B += batch_idx * stride
    # Compute mean
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    # Compute variance
    var = tl.sum(x * x, axis=0) / N - mean * mean
    rstd = tl.rsqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    y = (x - mean) * rstd * (1 + w) + b
    tl.store(Y + cols, y, mask=mask)


class _layer_norm_modulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps=1e-5) -> torch.Tensor:
        assert x.shape[0] == weight.shape[0] == bias.shape[0]
        assert x.shape[-1] == weight.shape[-1] == bias.shape[-1]
        # TODO: handle non-contiguous tensors
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        batch_size = x.shape[0]
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        seq_len = M // batch_size
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_SIZE, num_warps = calculate_settings(N)
        _layer_norm_modulation_fwd[(M,)](
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,
            x_arg.stride(0),
            seq_len,
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_ctas=1,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        # TODO: implement backward pass
        x, w, b, m, v = ctx.saved_tensors
        return x, None, w, b, None


layer_norm_modulation = _layer_norm_modulation.apply


@triton.jit
def _rms_norm_fwd(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Rstd,  # pointer to the 1/std
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_offset = tl.program_id(0) * BLOCK_M
    row_index = row_offset + tl.arange(0, BLOCK_M)[:, None]
    col_index = tl.arange(0, BLOCK_N)[None, :]
    col_mask = col_index < N

    x = tl.load(X + N * row_index + col_index, col_mask, other=0.0)
    w = tl.load(W + col_index, col_mask, eviction_policy="evict_last", other=0.0)

    xx = x * x
    xx = tl.broadcast_to(xx, [BLOCK_M, BLOCK_N])
    mean = tl.sum(xx, axis=1)[:, None] / N
    rstd = tl.rsqrt(mean + eps)
    y = x * rstd * w

    tl.store(Rstd + row_index, rstd)  # for backward
    tl.store(Y + N * row_index + col_index, y, col_mask)


@triton.jit
def _rms_norm_bwd(
    dY,
    dX,
    dW,
    X,
    W,
    Rstd,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    dY += row * stride
    dX += row * stride
    dW += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    dy = tl.load(dY + cols, mask=mask, other=0.0)
    x = tl.load(X + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    rstd = tl.load(Rstd + row)

    m = dy * w
    dx = rstd * m
    dx += rstd * -(1 / N) * rstd * rstd * tl.sum(m * x, axis=0) * x
    dw = dy * (x * rstd)

    tl.store(dX + cols, dx, mask=mask)
    tl.store(dW + cols, dw, mask=mask)


class _rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.contiguous()
        scale = scale.contiguous()
        assert x.shape[-1] == scale.shape[-1]
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        BLOCK_M = 8
        BLOCK_N, num_warps = calculate_settings(N)
        _rms_norm_fwd[(M // BLOCK_M,)](
            x_arg,
            y,
            scale,
            rstd,
            N,
            eps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_ctas=1,
        )
        ctx.save_for_backward(x_arg, scale, rstd)
        ctx.BLOCK_SIZE = BLOCK_N
        ctx.num_warps = num_warps
        return y

    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        dx = torch.empty_like(dy)
        dy_arg = dy.reshape(-1, dy.shape[-1])
        M, N = dy_arg.shape
        x, w, r = ctx.saved_tensors
        dw = torch.empty_like(x)
        _rms_norm_bwd[(M,)](
            dy_arg,
            dx,
            dw,
            x,
            w,
            r,
            x.stride(0),
            N,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        dw = torch.sum(dw, dim=0)
        return dx, dw, None


rms_norm = _rms_norm.apply
