from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional


class BaseNoiseSchedule(ABC):
    def __init__(self, max_t: Optional[float] = 1000.0):
        self.max_t = max_t

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        ...
