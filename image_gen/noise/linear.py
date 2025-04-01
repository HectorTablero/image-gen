from .base import BaseNoiseSchedule
from torch import Tensor


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __call__(self, t: Tensor) -> Tensor:
        return 1.0 - (t / self.max_t).clamp(min=0.0, max=1.0)
