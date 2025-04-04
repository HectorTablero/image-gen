from .base import BaseNoiseSchedule
from torch import Tensor
from typing import Optional


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, max_t: Optional[float] = 1000.0, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__(max_t)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: Tensor) -> Tensor:
        t_norm = (t / self.max_t).clamp(0.0, 1.0)
        return self.beta_min + t_norm * (self.beta_max - self.beta_min)

    def config(self) -> dict:
        return {
            "max_t": self.max_t,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }
