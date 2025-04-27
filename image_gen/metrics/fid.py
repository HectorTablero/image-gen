from .base import BaseMetric
from torch import Tensor


class FrechetInceptionDistance(BaseMetric):
    def __call__(self, real: Tensor, generated: Tensor, *args, **kwargs) -> float:
        pass

    @property
    def name(self) -> str:
        return "Fréchet Inception Distance (FID)"
