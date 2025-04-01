from .base import BaseMetric
from torch import Tensor


class FrechetInceptionDistance(BaseMetric):
    def __call__(self, real: Tensor, generated: Tensor) -> float:
        pass

    @property
    def name(self) -> str:
        return "Fréchet Inception Distance (FID)"
