from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
import torch
from torch import Tensor


class BaseSampler(ABC):
    def __init__(self, diffusion, num_steps: Optional[int] = 1000):
        self.diffusion = diffusion
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

    @abstractmethod
    def __call__(self,
                 x_T: Tensor,
                 score_model: Callable,
                 n_steps: int = 500,
                 seed: Optional[int] = None,
                 ) -> Tuple[Tensor, Tensor]:
        ...

    def timesteps(self, reverse: bool = True):
        ts = torch.linspace(0, self.diffusion.schedule.max_t, self.num_steps+1)
        return ts.flip(0) if reverse else ts
