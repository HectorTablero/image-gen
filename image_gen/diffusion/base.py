from abc import ABC, abstractmethod
from ..noise import BaseNoiseSchedule
from torch import Tensor
from typing import Tuple


class BaseDiffusion(ABC):
    NEEDS_NOISE_SCHEDULE = True

    def __init__(self, schedule: BaseNoiseSchedule):
        self.schedule = schedule
        self.max_t: int = schedule.max_t

    @abstractmethod
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    def backward_sde(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        f, g = self.forward_sde(x, t)
        g_squared = g**2
        if g_squared.shape != score.shape:
            g_squared = g_squared.expand_as(score)

        return f - g_squared * score, g

    def __str__(self) -> str:
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self.__class__.__name__}({params})"
