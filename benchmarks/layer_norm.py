import os

import torch
import triton

from triton_kernels import layer_norm_modulation, layer_norm_modulation_torch, layer_norm_modulation_torch_compile


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch"],
        line_names=["triton", "torch_compile", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="layer-norm-modulation",
        args={"batch_size": 4, "embed_dim": 3072},
    )
)
def bench_layer_norm_modulation(batch_size, seq_len, embed_dim, provider, device="cuda"):
    # create data
    x = torch.randn([batch_size, seq_len, embed_dim], device=device)
    scale = torch.randn([batch_size, 1, embed_dim], device=device)
    shift = torch.randn([batch_size, 1, embed_dim], device=device)

    if provider == "triton":
        fwd = lambda: layer_norm_modulation(x, scale, shift)
    elif provider == "torch_compile":
        fwd = lambda: layer_norm_modulation_torch_compile(x, scale, shift)
    elif provider == "torch":
        fwd = lambda: layer_norm_modulation_torch(x, scale, shift)
    else:
        raise Exception("invalid provider")

    ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=[0.5, 0.2, 0.8])

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Benchmark
fwd_dir = "./results/fwd"
os.makedirs(fwd_dir, exist_ok=True)
bench_layer_norm_modulation.run(print_data=True, save_path=fwd_dir)
