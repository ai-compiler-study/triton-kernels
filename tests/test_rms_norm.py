import pytest
import torch

from triton_kernels import rms_norm, rms_norm_torch


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_heads", [2, 4, 8, 24])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("head_dim", [128, 256, 512])
@pytest.mark.parametrize("device", ["cuda"])
def test_rms_norm(batch_size, num_heads, seq_len, head_dim, device):

    # create data
    x = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    w = torch.randn([head_dim], device=device)
    dy = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    x.requires_grad_(True)
    w.requires_grad_(True)

    # forward pass
    y_tri = rms_norm(x, w)
    y_ref = rms_norm_torch(x, w)

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri = x.grad.clone()
    dw_tri = w.grad.clone()
    x.grad = None
    w.grad = None

    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref = x.grad.clone()
    dw_ref = w.grad.clone()

    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(dx_tri, dx_ref, atol=1e-3, rtol=0)
    torch.testing.assert_close(dw_tri, dw_ref, atol=1e-3, rtol=0)
