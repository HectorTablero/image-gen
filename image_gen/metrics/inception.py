from .base import BaseMetric
from torch import Tensor


class InceptionScore(BaseMetric):
    def __call__(self, real: Tensor, generated: Tensor) -> float:
        pass

    @property
    def name(self) -> str:
        return "Inception Score (IS)"
