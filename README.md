# Triton Kernels
Triton kernels for [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) and [Flux](https://github.com/black-forest-labs/flux)

### Installation
```bash
pip install -e .
```

### Test
- [LayerNorm + Modulation Kernel](./normalization.py)
  - `python ./benchmarks/normalization_test.py`
- [RMSNorm Kernel](./normalization.py)
  - `python ./benchmarks/rms_norm_test.py`
- [RoPE Kernel](./positional_embedding.py)
  - `python ./benchmakrs/rope_test.py`
