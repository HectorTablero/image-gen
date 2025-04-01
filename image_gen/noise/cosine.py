from .base import BaseNoiseSchedule
from typing import Optional
import torch
from torch import Tensor


class CosineNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, max_t: Optional[int] = None, s: Optional[float] = 0.008):
        super().__init__(max_t)
        self.s = s

    def __call__(self, t: Tensor) -> Tensor:
        return torch.pi * torch.tan(0.5 * torch.pi * (t/self.max_t + self.s) / (1 + self.s)) / (1 + self.s) / self.max_t
