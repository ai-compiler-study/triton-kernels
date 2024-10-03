import os

import torch
import triton

from triton_kernels import apply_rope, apply_rope_torch, apply_rope_torch_compile


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch"],
        line_names=["triton", "torch_compile", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="apply-rope",
        args={"batch_size": 4, "num_heads": 24, "head_dim": 128},
    )
)
def bench_apply_rope(batch_size, num_heads, seq_len, head_dim, provider, device="cuda"):
    # create data
    q = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    k = torch.randn([batch_size, num_heads, seq_len, head_dim], device=device)
    pe = torch.randn([1, 1, seq_len, head_dim // 2, 2, 2], device=device)

    if provider == "triton":
        fwd = lambda: apply_rope(q, k, pe)
    elif provider == "torch_compile":
        fwd = lambda: apply_rope_torch_compile(q, k, pe)
    elif provider == "torch":
        fwd = lambda: apply_rope_torch(q, k, pe)
    else:
        raise Exception("invalid provider")

    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=[0.5, 0.2, 0.8])

    gbps = lambda ms: 2 * q.numel() * q.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
fwd_dir = "./results/fwd"
os.makedirs(fwd_dir, exist_ok=True)
bench_apply_rope.run(print_data=True, save_path=fwd_dir)
