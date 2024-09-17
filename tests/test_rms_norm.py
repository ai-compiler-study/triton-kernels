import pytest
import torch

from triton_kernels import rms_norm, rms_norm_torch


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
def test_rms_norm(batch_size, num_heads, seq_len, head_dim, device):
    # create data
    x = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    scale = torch.randn(head_dim).to(device)
    # forward pass
    y_tri = rms_norm(x, scale)
    y_ref = rms_norm_torch(x, scale)
    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)
