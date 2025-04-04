from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional


class BaseNoiseSchedule(ABC):
    def __init__(self, max_t: Optional[float] = 1000.0):
        self.max_t = max_t

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def config(self) -> dict:
        pass

    def __str__(self) -> str:
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self.__class__.__name__}({params})"
