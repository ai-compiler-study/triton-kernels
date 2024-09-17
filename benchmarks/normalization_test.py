import os

import torch
import torch.nn.functional as F
import triton

from triton_kernels import layer_norm_modulation, layer_norm_modulation_torch, layer_norm_modulation_torch_compile


def test_layer_norm_modulation(batch_size, seq_len, embed_dim, dtype, device="cuda"):
    # create data
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    scale = torch.randn(batch_size, embed_dim).to(device)
    shift = torch.randn(batch_size, embed_dim).to(device)
    # forward pass
    y_tri = layer_norm_modulation(x, scale, shift)
    y_ref = layer_norm_modulation_torch(x, scale, shift)
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=0)
    print("TEST PASS")


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
        args={"batch_size": 4, "embed_dim": 3072, "dtype": torch.float32},
    )
)
def bench_layer_norm_modulation(batch_size, seq_len, embed_dim, dtype, provider, device="cuda"):
    # create data
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    scale = torch.randn(batch_size, embed_dim).to(device)
    shift = torch.randn(batch_size, embed_dim).to(device)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return layer_norm_modulation(x, scale, shift)

        if provider == "torch_compile":
            return layer_norm_modulation_torch_compile(x, scale, shift)

        if provider == "torch":
            return layer_norm_modulation_torch(x, scale, shift)

    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


# Test
test_layer_norm_modulation(batch_size=16, seq_len=256, embed_dim=1024, dtype=torch.float32)
test_layer_norm_modulation(batch_size=16, seq_len=512, embed_dim=1024, dtype=torch.float32)
test_layer_norm_modulation(batch_size=16, seq_len=1024, embed_dim=1024, dtype=torch.float32)
test_layer_norm_modulation(batch_size=16, seq_len=1024, embed_dim=2048, dtype=torch.float32)
test_layer_norm_modulation(batch_size=16, seq_len=1024, embed_dim=3072, dtype=torch.float32)

# Benchmark
result_dir = "./results"
os.makedirs(result_dir, exist_ok=True)
bench_layer_norm_modulation.run(save_path=result_dir, print_data=True)
