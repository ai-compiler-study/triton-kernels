import os

import torch
import torch.nn.functional as F
import triton

from normalization import layer_norm_modulation


def modulate(x, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def layer_norm_modulation_torch(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
    return modulate(x, scale=scale, shift=shift)


@torch.compile
def layer_norm_modulation_torch_compile(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
    return modulate(x, scale=scale, shift=shift)


def test_layer_norm_modulation(batch_size, seq_len, embed_dim, dtype, device="cuda"):
    # create data
    x = torch.randn(batch_size, seq_len, embed_dim).to(device)
    scale = torch.randn(batch_size, embed_dim).to(device)
    shift = torch.randn(batch_size, embed_dim).to(device)
    # forward pass
    y_tri = layer_norm_modulation(x, scale, shift)
    y_ref = layer_norm_modulation_torch(x, scale, shift).to(dtype)
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-5, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["embed_dim"],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg="provider",
        line_vals=["triton", "torch_compile", "torch"],
        line_names=["triton", "torch_compile", "torch"],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="layer-norm-modulation",
        args={"batch_size": 16, "seq_len": 1024, "dtype": torch.float32},
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
