from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple


class VariancePreserving(BaseDiffusion):
    def __init__(self, schedule: BaseNoiseSchedule):
        super().__init__(schedule)
        self._precompute_beta_integral()

    def _precompute_beta_integral(self):
        t_values = torch.arange(self.schedule.max_t + 1)
        beta_values = self.schedule(t_values)
        self.beta_integral: Tensor = torch.cumsum(
            beta_values * (1/self.schedule.max_t), dim=0)

    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t)
        return -0.5 * beta_t * x, torch.sqrt(beta_t)

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_int = self.beta_integral[t.long()]
        mean = x0 * torch.exp(-0.5 * beta_int)
        std = torch.sqrt(1 - torch.exp(-beta_int))
        return mean + std * torch.randn_like(x0), std
