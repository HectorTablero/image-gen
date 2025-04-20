from .vp import VariancePreserving
import torch
from torch import Tensor
from typing import Tuple


class SubVariancePreserving(VariancePreserving):
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = (self.schedule.beta_min +
                  (self.schedule.beta_max - self.schedule.beta_min) * t)
        beta_t = beta_t.view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x

        beta_integral = (self.schedule.beta_min * t +
                         0.5 * (self.schedule.beta_max - self.schedule.beta_min) * (t ** 2))
        beta_integral = beta_integral.view(-1, 1, 1, 1)
        diffusion = torch.sqrt(beta_t * (1 - torch.exp(-2 * beta_integral)))

        return drift, diffusion
