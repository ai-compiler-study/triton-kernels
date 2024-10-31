import pytest
import torch

from triton_kernels.kernels.attention import scaled_dot_product_attention


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_heads", [2, 4, 8, 24])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("device", ["cuda"])
def test_scaled_dot_product_attention(batch_size, num_heads, seq_len, head_dim, device):
    # create data
    q = torch.randn([batch_size, num_heads, seq_len, head_dim], dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # forward pass
    y_tri = scaled_dot_product_attention(q, k, v)
    y_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0)
