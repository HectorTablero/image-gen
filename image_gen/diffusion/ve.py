from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple, Optional
import numpy as np


class VESchedule(BaseNoiseSchedule):
    def __init__(self, max_t: int, sigma_min: float, sigma_max: float):
        super().__init__(max_t)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, t: Tensor) -> Tensor:
        return self.sigma_min ** (1 - t/self.max_t) * self.sigma_max ** (t/self.max_t)


class VarianceExploding(BaseDiffusion):
    def __init__(self, schedule: BaseNoiseSchedule, sigma_min: Optional[float] = 0.01, sigma_max: Optional[float] = 25.0):
        super().__init__(schedule)
        self.schedule = VESchedule(
            max_t=schedule.max_t, sigma_min=sigma_min, sigma_max=sigma_max)

    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        g_t = self.schedule(
            t) * np.sqrt(2 * np.log(self.schedule.sigma_max/self.schedule.sigma_min))
        return torch.zeros_like(x), g_t

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        sigma_t = self.schedule(t).view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        return x0 + sigma_t * noise, noise
