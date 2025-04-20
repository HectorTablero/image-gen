from .base import BaseNoiseSchedule
import torch
from torch import Tensor
import math


class CosineNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, max_t: float = 1000.0, s: float = 0.008, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__(max_t)
        self.s = s
        self.beta_min = beta_min  # Currently unused, but needed for compatibility
        self.beta_max = beta_max

    def __call__(self, t: Tensor) -> Tensor:
        t_norm = (t / self.max_t).clamp(0.0, 1.0)

        angle_0 = math.atan(1.0 / (math.pi * (1.0 + self.s)))
        angle_t = torch.atan(1.0 / (math.pi * ((1.0 - t_norm) + self.s)))

        numerator = angle_t - angle_0
        denominator = 0.5 * math.pi - angle_0
        fractional = numerator / denominator

        fractional = fractional.clamp(0.0, 1.0)

        beta_t = fractional * self.beta_max

        beta_t = torch.nan_to_num(
            beta_t, nan=self.beta_max * 0.5, posinf=self.beta_max)
        beta_t = beta_t.clamp(1e-5, self.beta_max)

        return beta_t

    def config(self) -> dict:
        return {
            "max_t": self.max_t,
            "s": self.s,
            "beta_max": self.beta_max,
        }
