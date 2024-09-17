import os

import torch
import triton
from torch import Tensor

from normalization import rms_norm


def rms_norm_torch(x: Tensor, scale: Tensor) -> Tensor:
    x_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    return (x * rrms).to(dtype=x_dtype) * scale


@torch.compile
def rms_norm_torch_compile(x: Tensor, scale: Tensor) -> Tensor:
    x_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    return (x * rrms).to(dtype=x_dtype) * scale


def test_rms_norm(batch_size, num_heads, seq_len, head_dim, dtype, device="cuda"):
    # create data
    x = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    scale = torch.randn(head_dim).to(device)

    # forward pass
    y_tri = rms_norm(x, scale)
    y_ref = rms_norm_torch(x, scale).to(dtype)
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch"],
        line_names=["triton", "torch_compile", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="rms-norm",
        args={"batch_size": 4, "num_heads": 24, "head_dim": 128, "dtype": torch.float32},
    )
)
def bench_rms_norm(batch_size, num_heads, seq_len, head_dim, dtype, provider, device="cuda"):
    # create data
    x = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    scale = torch.randn(head_dim).to(device)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return rms_norm(x, scale)

        if provider == "torch_compile":
            return rms_norm_torch_compile(x, scale)

        if provider == "torch":
            return rms_norm_torch(x, scale)

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Test
test_rms_norm(batch_size=16, num_heads=24, seq_len=256, head_dim=128, dtype=torch.float32)
test_rms_norm(batch_size=16, num_heads=24, seq_len=512, head_dim=128, dtype=torch.float32)
test_rms_norm(batch_size=16, num_heads=24, seq_len=1024, head_dim=128, dtype=torch.float32)
test_rms_norm(batch_size=16, num_heads=24, seq_len=1024, head_dim=256, dtype=torch.float32)
test_rms_norm(batch_size=16, num_heads=24, seq_len=1024, head_dim=512, dtype=torch.float32)

# Benchmark
result_dir = "./results"
os.makedirs(result_dir, exist_ok=True)
bench_rms_norm.run(save_path=result_dir, print_data=True)
