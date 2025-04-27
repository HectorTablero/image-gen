from .base import BaseMetric
from torch import Tensor


class BitsPerDim(BaseMetric):
    def __call__(self, real: Tensor, generated: Tensor, *args, **kwargs) -> float:
        pass

    @property
    def name(self) -> str:
        return "Bits per Dimension (BPD)"
