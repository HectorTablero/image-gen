from abc import ABC, abstractmethod
from torch import Tensor
from typing import Optional


class BaseNoiseSchedule(ABC):
    @abstractmethod
    def __call__(self, t: Tensor, *args, **kwargs) -> Tensor:
        ...

    @abstractmethod
    def config(self) -> dict:
        pass

    @abstractmethod
    def integral_beta(self, t: Tensor, *args, **kwargs) -> Tensor:
        ...

    def config(self) -> dict:
        return {}

    def __str__(self) -> str:
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self.__class__.__name__}({params})"
