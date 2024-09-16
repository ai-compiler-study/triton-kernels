import os

import torch
import triton
from torch import Tensor

from positional_embedding import apply_rope


def apply_rope_torch(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


@torch.compile
def apply_rope_torch_compile(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def test_apply_rope(batch_size, num_heads, seq_len, head_dim, dtype, device="cuda"):
    # create data
    xq = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    xk = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    freqs_cis = torch.randn(1, 1, seq_len, head_dim // 2, 2, 2).to(device)

    # forward pass
    q_tri, k_tri = apply_rope(xq, xk, freqs_cis)
    q_ref, k_ref = apply_rope_torch(xq, xk, freqs_cis)

    # compare
    assert torch.allclose(q_tri, q_ref, atol=1e-3, rtol=0)
    assert torch.allclose(k_tri, k_ref, atol=1e-3, rtol=0)
    print("TEST PASS")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["head_dim"],
        x_vals=[32 * i for i in range(1, 32)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch"],
        line_names=["triton", "torch_compile", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="apply-rope",
        args={"batch_size": 16, "num_heads": 24, "seq_len": 2048, "dtype": torch.float32},
    )
)
def bench_apply_rope(batch_size, num_heads, seq_len, head_dim, dtype, provider, device="cuda"):
    # create data
    xq = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    xk = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    freqs_cis = torch.randn(1, 1, seq_len, head_dim // 2, 2, 2).to(device)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return apply_rope(xq, xk, freqs_cis)

        if provider == "torch_compile":
            return apply_rope_torch_compile(xq, xk, freqs_cis)

        if provider == "torch":
            return apply_rope_torch(xq, xk, freqs_cis)

    gbps = lambda ms: 2 * xq.numel() * xq.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Test
test_apply_rope(batch_size=16, num_heads=24, seq_len=256, head_dim=128, dtype=torch.float32)
test_apply_rope(batch_size=16, num_heads=24, seq_len=512, head_dim=128, dtype=torch.float32)
test_apply_rope(batch_size=16, num_heads=24, seq_len=1024, head_dim=128, dtype=torch.float32)
test_apply_rope(batch_size=16, num_heads=24, seq_len=1024, head_dim=256, dtype=torch.float32)
test_apply_rope(batch_size=16, num_heads=24, seq_len=1024, head_dim=512, dtype=torch.float32)

# Benchmark
result_dir = "./results"
os.makedirs(result_dir, exist_ok=True)
bench_apply_rope.run(save_path=result_dir, print_data=True)
