import torch
import torch.nn as nn
from torch import Tensor

import triton_kernels as tk


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return tk.rms_norm(x, self.scale)


class QKNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)
