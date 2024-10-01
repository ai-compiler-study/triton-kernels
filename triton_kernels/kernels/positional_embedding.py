import torch
import triton
import triton.language as tl
from torch import Tensor

from triton_kernels.kernels.utils import calculate_settings


@triton.jit
def _rope_fwd(
    q_ptr,
    k_ptr,
    f_ptr,
    oq_ptr,
    ok_ptr,
    stride,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    q_start_ptr = q_ptr + bh_idx * stride
    k_start_ptr = k_ptr + bh_idx * stride
    oq_start_ptr = oq_ptr + bh_idx * stride
    ok_start_ptr = ok_ptr + bh_idx * stride

    d_half = d // 2
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets2 = tl.arange(0, BLOCK_SIZE * 2)

    f0_ptrs = f_ptr + s_idx * d * 2 + col_offsets2 * 2
    f1_ptrs = f_ptr + s_idx * d * 2 + col_offsets2 * 2 + 1
    f0 = tl.load(f0_ptrs, mask=col_offsets2 < d, other=0.0).reshape(BLOCK_SIZE, 2)
    f1 = tl.load(f1_ptrs, mask=col_offsets2 < d, other=0.0).reshape(BLOCK_SIZE, 2)

    q0_ptrs = q_start_ptr + s_idx * d + col_offsets * 2
    q1_ptrs = q_start_ptr + s_idx * d + col_offsets * 2 + 1
    q0 = tl.load(q0_ptrs, mask=col_offsets < d_half, other=0.0).reshape(BLOCK_SIZE, 1)
    q1 = tl.load(q1_ptrs, mask=col_offsets < d_half, other=0.0).reshape(BLOCK_SIZE, 1)

    k0_ptrs = k_start_ptr + s_idx * d + col_offsets * 2
    k1_ptrs = k_start_ptr + s_idx * d + col_offsets * 2 + 1
    k0 = tl.load(k0_ptrs, mask=col_offsets < d_half, other=0.0).reshape(BLOCK_SIZE, 1)
    k1 = tl.load(k1_ptrs, mask=col_offsets < d_half, other=0.0).reshape(BLOCK_SIZE, 1)

    oq = f0 * q0 + f1 * q1
    ok = f0 * k0 + f1 * k1

    oq_ptrs = oq_start_ptr + s_idx * d + col_offsets2
    ok_ptrs = ok_start_ptr + s_idx * d + col_offsets2
    tl.store(oq_ptrs, oq.reshape(BLOCK_SIZE * 2), mask=col_offsets2 < d)
    tl.store(ok_ptrs, ok.reshape(BLOCK_SIZE * 2), mask=col_offsets2 < d)


class _rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
        xq, xk, freqs_cis = xq.contiguous(), xk.contiguous(), freqs_cis.contiguous()

        b, h, s, d = xq.shape
        bh = b * h

        xq_arg = xq.reshape(-1, s, d)
        xk_arg = xk.reshape(-1, s, d)
        f_arg = freqs_cis.reshape(-1, s, d // 2, 2, 2)

        xq_out = torch.empty_like(xq)
        xk_out = torch.empty_like(xk)

        BLOCK_SIZE, num_warps = calculate_settings(d // 2)

        _rope_fwd[(bh, s)](
            xq_arg,
            xk_arg,
            f_arg,
            xq_out,
            xk_out,
            xq_arg.stride(0),
            d,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        return xq_out, xk_out

    def backward(
        ctx,
        dxq: Tensor,
        dxk: Tensor,
    ) -> Tensor:
        # TODO: implement backward pass
        (freqs_cis,) = ctx.saved_tensors
        return dxq, dxk, None


apply_rope = _rope.apply
