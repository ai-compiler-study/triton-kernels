import pytest
import torch

import triton_kernels as tk
from triton_kernels.flux import DoubleStreamBlock, SingleStreamBlock


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("device", ["cuda"])
def test_single_stream_block(batch_size, seq_len, device):
    num_heads = 24
    head_dim = 128
    hidden_size = num_heads * head_dim
    mlp_ratio = 4.0

    block = SingleStreamBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    block_triton = tk.SingleStreamBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    block_triton.load_state_dict(block.state_dict())
    block = block.to(device)
    block_triton = block_triton.to(device)

    # create data
    x = torch.randn([batch_size, seq_len, hidden_size], device=device)
    vec = torch.randn([batch_size, hidden_size], device=device)
    pe = torch.randn([1, 1, seq_len, head_dim // 2, 2, 2], device=device)

    # forward pass
    y_ref = block_triton(x=x, vec=vec, pe=pe)
    y_tri = block(x=x, vec=vec, pe=pe)

    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("img_seq_len", [256, 512, 1024])
@pytest.mark.parametrize("device", ["cuda"])
def test_double_stream_block(batch_size, img_seq_len, device):
    num_heads = 24
    head_dim = 128
    hidden_size = num_heads * head_dim
    mlp_ratio = 4.0
    txt_seq_len = 256
    seq_len = img_seq_len + txt_seq_len

    block = DoubleStreamBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    block_triton = tk.DoubleStreamBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    block_triton.load_state_dict(block.state_dict())
    block = block.to(device)
    block_triton = block_triton.to(device)

    # create data
    img = torch.randn([batch_size, img_seq_len, hidden_size], device=device)
    txt = torch.randn([batch_size, txt_seq_len, hidden_size], device=device)
    vec = torch.randn([batch_size, hidden_size], device=device)
    pe = torch.randn([1, 1, seq_len, head_dim // 2, 2, 2], device=device)

    # forward pass
    y_ref = block_triton(img=img, txt=txt, vec=vec, pe=pe)
    y_tri = block(img=img, txt=txt, vec=vec, pe=pe)

    # compare
    torch.testing.assert_close(y_tri, y_ref, atol=1e-5, rtol=0)
