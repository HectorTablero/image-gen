from abc import ABC, abstractmethod
from ..noise import BaseNoiseSchedule
from torch import Tensor
from typing import Tuple


class BaseDiffusion(ABC):
    NEEDS_NOISE_SCHEDULE = True

    def __init__(self, schedule: BaseNoiseSchedule, *args, **kwargs):
        self.schedule = schedule

    @abstractmethod
    def forward_sde(self, x: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def forward_process(self, x0: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def compute_loss(self, score: Tensor, noise: Tensor, t: Tensor, *args, **kwargs) -> Tensor:
        ...

    def backward_sde(self, x: Tensor, t: Tensor, score: Tensor, *args, **kwargs) -> Tensor:
        f, g = self.forward_sde(x, t)
        g_squared = g**2
        if g_squared.shape != score.shape:
            g_squared = g_squared.expand_as(score)

        return f - g_squared * score, g

    @property
    def schedule(self) -> BaseNoiseSchedule:
        return self._schedule

    @schedule.setter
    def schedule(self, value: BaseNoiseSchedule):
        # Schedule shouldn't be allowed to change once the class is initialized
        if not hasattr(self, '_schedule'):
            self._schedule = value
            return
        raise AttributeError("Attribute 'schedule' is not settable")

    def config(self) -> dict:
        return {}

    def __str__(self) -> str:
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self.__class__.__name__}({params})"
