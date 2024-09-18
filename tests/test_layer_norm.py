import pytest
import torch

from triton_kernels import layer_norm_modulation, layer_norm_modulation_torch


@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, device",
    (
        [16, 256, 1024, "cuda"],
        [16, 512, 1024, "cuda"],
        [16, 1024, 1024, "cuda"],
        [16, 1024, 2048, "cuda"],
        [16, 1024, 3072, "cuda"],
    ),
)
def test_layer_norm_modulation(batch_size, seq_len, embed_dim, device):
    # create data
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    scale = torch.randn(batch_size, 1, embed_dim).to(device)
    shift = torch.randn(batch_size, 1, embed_dim).to(device)
    # forward pass
    y_tri = layer_norm_modulation(x, scale, shift)
    y_ref = layer_norm_modulation_torch(x, scale, shift)
    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)
