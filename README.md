# flux.triton
Triton kernels for [Flux](https://github.com/black-forest-labs/flux)

### Installation
```bash
pip install -e .

# for tests
pip install -e .[testing]

# for benchmarks
pip install matplotlib pandas
```

### Tests
```bash
python -m pytest
```

### Benchmarks
- Linear Kernel
  - `python ./benchmarks/linear.py`
- Attention Kernel
  - `python ./benchmarks/attention.py`
- LayerNorm + Modulation Kernel
  - `python ./benchmarks/layer_norm.py`
- RMSNorm Kernel
  - `python ./benchmarks/rms_norm.py`
- RoPE Kernel
  - `python ./benchmarks/rope.py`
