import pytest
import torch

from triton_kernels import layer_norm_modulation, layer_norm_modulation_torch


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("embed_dim", [1024, 2048, 3072])
@pytest.mark.parametrize("device", ["cuda"])
def test_layer_norm_modulation(batch_size, seq_len, embed_dim, device):
    # create data
    x = torch.randn([batch_size, seq_len, embed_dim], device=device)
    scale = torch.randn([batch_size, 1, embed_dim], device=device)
    shift = torch.randn_like(scale)
    # forward pass
    y_tri = layer_norm_modulation(x, scale, shift)
    y_ref = layer_norm_modulation_torch(x, scale, shift)
    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)
