from .base import BaseNoiseSchedule
from torch import Tensor


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, *args, beta_min: float = 0.0001, beta_max: float = 20.0, **kwargs):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: Tensor, *args, **kwargs) -> Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: Tensor, *args, **kwargs) -> Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

    def config(self) -> dict:
        return {
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }
