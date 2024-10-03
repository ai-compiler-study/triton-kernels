import os

import torch
import triton

from triton_kernels import rms_norm, rms_norm_torch, rms_norm_torch_compile


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
        args={"batch_size": 4, "num_heads": 24, "head_dim": 128},
    )
)
def bench_rms_norm(batch_size, num_heads, seq_len, head_dim, provider, device="cuda", mode="forward"):
    # create data
    x = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    scale = torch.randn([head_dim], device=device)
    dy = torch.randn_like(x)

    if provider == "triton":
        fwd = lambda: rms_norm(x, scale)
    elif provider == "torch_compile":
        fwd = lambda: rms_norm_torch_compile(x, scale)
    elif provider == "torch":
        fwd = lambda: rms_norm_torch(x, scale)
    else:
        raise Exception("invalid provider")

    x.requires_grad_(True)
    scale.requires_grad_(True)
    if mode == "fwd":
        func = fwd
    elif mode == "bwd":
        y = fwd()
        bwd = lambda: y.backward(dy, retain_graph=True)
        func = bwd
    else:
        raise Exception("invalid mode")

    ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=[0.5, 0.2, 0.8])

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
fwd_dir = "./results/fwd"
bwd_dir = "./results/bwd"
os.makedirs(fwd_dir, exist_ok=True)
os.makedirs(bwd_dir, exist_ok=True)
bench_rms_norm.run(print_data=True, save_path=fwd_dir, mode="fwd")
bench_rms_norm.run(print_data=True, save_path=bwd_dir, mode="bwd")
