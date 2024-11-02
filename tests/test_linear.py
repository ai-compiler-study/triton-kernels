import pytest
import torch

from triton_kernels import linear


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("in_channel", [256, 512, 1024])
@pytest.mark.parametrize("out_channel", [1024, 2048, 3072])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_layer_norm_modulation(batch_size, seq_len, in_channel, out_channel, dtype, device):
    # create data
    x = torch.randn([batch_size, seq_len, in_channel], dtype=dtype, device=device)
    w = torch.randn([out_channel, in_channel], dtype=dtype, device=device)
    b = torch.randn([out_channel], dtype=dtype, device=device)
    # forward pass
    y_tri = linear(x, w, b)
    y_ref = torch.nn.functional.linear(x, w, b)
    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)
