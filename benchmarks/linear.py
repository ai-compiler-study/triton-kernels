import os

import torch
import triton

from triton_kernels import linear


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["triton", "torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="linear",
        args={"batch_size": 4, "embed_dim": 3072},
    )
)
def bench_linear(batch_size, seq_len, embed_dim, provider, device="cuda"):
    # create data
    dtype = torch.bfloat16
    x = torch.randn([batch_size, seq_len, embed_dim], dtype=dtype, device=device)
    w = torch.randn([embed_dim, embed_dim], dtype=dtype, device=device)
    b = torch.randn([embed_dim], dtype=dtype, device=device)

    if provider == "triton":
        fwd = lambda: linear(x, w, b)
    elif provider == "torch":
        fwd = lambda: torch.nn.functional.linear(x, w, b)
    else:
        raise Exception("invalid provider")

    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=[0.5, 0.2, 0.8])

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
fwd_dir = "./results/fwd"
os.makedirs(fwd_dir, exist_ok=True)
bench_linear.run(print_data=True, save_path=fwd_dir)
