import pytest
import torch

from triton_kernels import apply_rope, apply_rope_torch


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_heads", [2, 4, 8, 24])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("head_dim", [128, 256, 512])
@pytest.mark.parametrize("device", ["cuda"])
def test_apply_rope(batch_size, num_heads, seq_len, head_dim, device):
    # create data
    q = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    k = torch.randn_like(q)
    pe = torch.randn([1, 1, seq_len, head_dim // 2, 2, 2], device=device)
    # forward pass
    q_tri, k_tri = apply_rope(q, k, pe)
    q_ref, k_ref = apply_rope_torch(q, k, pe)
    # compare
    torch.testing.assert_close(q_tri, q_ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(k_tri, k_ref, atol=1e-5, rtol=0)
