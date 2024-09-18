import torch
import torch.nn.functional as F
from torch import Tensor


def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def layer_norm_modulation_torch(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
    return modulate(x, scale=scale, shift=shift)


@torch.compile
def layer_norm_modulation_torch_compile(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
    return modulate(x, scale=scale, shift=shift)


def rms_norm_torch(x: Tensor, scale: Tensor) -> Tensor:
    x_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    return (x * rrms).to(dtype=x_dtype) * scale


@torch.compile
def rms_norm_torch_compile(x: Tensor, scale: Tensor) -> Tensor:
    x_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    return (x * rrms).to(dtype=x_dtype) * scale
