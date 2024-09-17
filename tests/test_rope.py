import pytest
import torch

from triton_kernels import apply_rope, apply_rope_torch


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim, device",
    (
        [16, 24, 256, 128, "cuda"],
        [16, 24, 512, 128, "cuda"],
        [16, 24, 1024, 128, "cuda"],
        [16, 24, 1024, 256, "cuda"],
        [16, 24, 1024, 512, "cuda"],
    ),
)
def test_apply_rope(batch_size, num_heads, seq_len, head_dim, device):
    # create data
    xq = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    xk = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    freqs_cis = torch.randn(1, 1, seq_len, head_dim // 2, 2, 2).to(device)
    # forward pass
    q_tri, k_tri = apply_rope(xq, xk, freqs_cis)
    q_ref, k_ref = apply_rope_torch(xq, xk, freqs_cis)
    # compare
    torch.testing.assert_close(q_tri, q_ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(k_tri, k_ref, atol=1e-5, rtol=0)
