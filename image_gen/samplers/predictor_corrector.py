from .base import BaseSampler
from torch import Tensor


class PredictorCorrector(BaseSampler):
    def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        pass
