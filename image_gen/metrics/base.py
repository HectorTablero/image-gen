from abc import ABC, abstractmethod
from torch import Tensor


class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, real: Tensor, generated: Tensor, *args, **kwargs) -> float:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
