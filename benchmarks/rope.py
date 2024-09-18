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
    xq = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    xk = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    freqs_cis = torch.randn(1, 1, seq_len, head_dim // 2, 2, 2).to(device)

    def y_fwd():
        if provider == "triton":
            return apply_rope(xq, xk, freqs_cis)
        if provider == "torch_compile":
            return apply_rope_torch_compile(xq, xk, freqs_cis)
        if provider == "torch":
            return apply_rope_torch(xq, xk, freqs_cis)

    gbps = lambda ms: 2 * xq.numel() * xq.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=[0.5, 0.2, 0.8], rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
result_dir = "./results"
os.makedirs(result_dir, exist_ok=True)
bench_apply_rope.run(save_path=result_dir, print_data=True)
