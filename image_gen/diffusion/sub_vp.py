from .vp import VariancePreserving
import torch
from torch import Tensor
from typing import Tuple


class SubVariancePreserving(VariancePreserving):
    def forward_sde(self, x: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t, *args, **kwargs).view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x

        beta_integral = self.schedule.integral_beta(
            t, *args, **kwargs).view(-1, 1, 1, 1)
        diffusion = torch.sqrt(beta_t * (1 - torch.exp(-2 * beta_integral)))

        return drift, diffusion
