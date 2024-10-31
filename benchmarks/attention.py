import os

import torch
import triton

from triton_kernels.kernels.attention import scaled_dot_product_attention


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["triton", "torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="attention",
        args={"batch_size": 4, "num_heads": 24, "head_dim": 128},
    )
)
def bench_scaled_dot_product_attention(batch_size, num_heads, seq_len, head_dim, provider, device="cuda"):
    # create data
    q = torch.randn([batch_size, num_heads, seq_len, head_dim], dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if provider == "triton":
        fwd = lambda: scaled_dot_product_attention(q, k, v)
    elif provider == "torch":
        fwd = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        raise Exception("invalid provider")

    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=[0.5, 0.2, 0.8])

    gbps = lambda ms: 2 * q.numel() * q.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
fwd_dir = "./results/fwd"
os.makedirs(fwd_dir, exist_ok=True)
bench_scaled_dot_product_attention.run(print_data=True, save_path=fwd_dir)
