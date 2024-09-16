import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _rope_fwd(
    q_ptr,
    k_ptr,
    f_ptr,
    oq_ptr,
    ok_ptr,
    q_bh_stride,
    k_bh_stride,
    f_bh_stride,
    oq_bh_stride,
    ok_bh_stride,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    s_idx = tl.program_id(1)
    q_start_ptr = q_ptr + bh_idx * q_bh_stride
    k_start_ptr = k_ptr + bh_idx * k_bh_stride
    f_start_ptr = f_ptr
    oq_start_ptr = oq_ptr + bh_idx * oq_bh_stride
    ok_start_ptr = ok_ptr + bh_idx * ok_bh_stride

    d_half = d // 2
    col_offsets = tl.arange(0, BLOCK_SIZE // 2)
    col_offsets2 = tl.arange(0, BLOCK_SIZE)

    f0_ptrs = f_start_ptr + s_idx * d * 2 + col_offsets2 * 2
    f1_ptrs = f_start_ptr + s_idx * d * 2 + col_offsets2 * 2 + 1
    f0 = tl.load(f0_ptrs, mask=col_offsets2 < d, other=0.0)
    f1 = tl.load(f1_ptrs, mask=col_offsets2 < d, other=0.0)
    f0 = tl.reshape(f0, [BLOCK_SIZE // 2, 2])
    f1 = tl.reshape(f1, [BLOCK_SIZE // 2, 2])

    q0_ptrs = q_start_ptr + s_idx * d + col_offsets * 2
    q1_ptrs = q_start_ptr + s_idx * d + col_offsets * 2 + 1
    q0 = tl.load(q0_ptrs, mask=col_offsets < d_half, other=0.0)
    q1 = tl.load(q1_ptrs, mask=col_offsets < d_half, other=0.0)
    q0 = tl.reshape(q0, [BLOCK_SIZE // 2, 1])
    q1 = tl.reshape(q1, [BLOCK_SIZE // 2, 1])

    k0_ptrs = k_start_ptr + s_idx * d + col_offsets * 2
    k1_ptrs = k_start_ptr + s_idx * d + col_offsets * 2 + 1
    k0 = tl.load(k0_ptrs, mask=col_offsets < d_half, other=0.0)
    k1 = tl.load(k1_ptrs, mask=col_offsets < d_half, other=0.0)
    k0 = tl.reshape(k0, [BLOCK_SIZE // 2, 1])
    k1 = tl.reshape(k1, [BLOCK_SIZE // 2, 1])

    oq = f0 * q0 + f1 * q1
    ok = f0 * k0 + f1 * k1
    oq = oq.reshape(BLOCK_SIZE)
    ok = ok.reshape(BLOCK_SIZE)

    oq_ptrs = oq_start_ptr + s_idx * d + col_offsets2
    ok_ptrs = ok_start_ptr + s_idx * d + col_offsets2
    tl.store(oq_ptrs, oq, mask=col_offsets2 < d)
    tl.store(ok_ptrs, ok, mask=col_offsets2 < d)


class _rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
        b, h, s, d = xq.shape
        bh = b * h

        xq_arg = xq.reshape(-1, s, d)
        xk_arg = xk.reshape(-1, s, d)
        f_arg = freqs_cis.reshape(-1, s, d // 2, 2, 2)

        xq_out = torch.empty_like(xq_arg)
        xk_out = torch.empty_like(xk_arg)

        BLOCK_SIZE = triton.next_power_of_2(d)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 4096:
            num_warps = 16

        _rope_fwd[(bh, s)](
            xq_arg,
            xk_arg,
            f_arg,
            xq_out,
            xk_out,
            xq_arg.stride(0),
            xk_arg.stride(0),
            f_arg.stride(0),
            xq_out.stride(0),
            xk_out.stride(0),
            d,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

        xq_out = xq_out.reshape(*xq.shape)
        xk_out = xk_out.reshape(*xk.shape)

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
