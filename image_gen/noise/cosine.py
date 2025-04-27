from .base import BaseNoiseSchedule
import torch
from torch import Tensor
import numpy as np


class CosineNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, *args, s: float = 0.008, beta_min: float = 0.0001, beta_max: float = 50.0, **kwargs):
        self.s = s
        self.beta_min = beta_min
        self.beta_max = beta_max

    def _alpha_t(self, t: Tensor, *args, **kwargs) -> Tensor:
        angle1 = np.pi / 2 * (t + self.s) / (1.0 + self.s)
        cos1 = torch.cos(angle1) ** 2

        angle2 = np.pi / 2 * self.s / (1.0 + self.s)
        cos2 = np.cos(angle2) ** 2

        return cos1 / cos2

    def __call__(self, t: Tensor, max_t: float = 500.0, *args, **kwargs) -> Tensor:
        beta = 1.0 - self._alpha_t(t) / self._alpha_t(t - 1.0 / max_t)
        return self.beta_min + (self.beta_max - self.beta_min) * beta

    def integral_beta(self, t: Tensor, *args, **kwargs) -> Tensor:
        return -torch.log(self._alpha_t(t))

    def config(self) -> dict:
        return {
            "s": self.s,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max
        }
