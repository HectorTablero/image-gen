from abc import ABC, abstractmethod
from ..noise import BaseNoiseSchedule
from torch import Tensor
from typing import Tuple


class BaseDiffusion(ABC):
    def __init__(self, schedule: BaseNoiseSchedule):
        self.schedule = schedule
        self.max_t: int = schedule.max_t

    @abstractmethod
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        ...
