# Triton Kernels
Triton kernels for [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) and [Flux](https://github.com/black-forest-labs/flux)

### Installation
```bash
pip install -e .

# for tests
pip install -e .[testing]

# for benchmarks
pip install matplotlib pandas
```

### Benchmarks
- LayerNorm + Modulation Kernel
  - `python ./benchmarks/layer_norm.py`
- RMSNorm Kernel
  - `python ./benchmarks/rms_norm.py`
- RoPE Kernel
  - `python ./benchmakrs/rope_test.py`

### Tests
```bash
python -m pytest
```
