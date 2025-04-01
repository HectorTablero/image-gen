from .vp import VariancePreserving
import torch
from torch import Tensor
from typing import Tuple


class SubVariancePreserving(VariancePreserving):
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.schedule(t)
        beta_int = self.beta_integral[t.long()]
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t * (1 - torch.exp(-2 * beta_int)))
        return drift, diffusion
