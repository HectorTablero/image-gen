from abc import ABC, abstractmethod
from torch import Tensor


class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, real: Tensor, generated: Tensor) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
