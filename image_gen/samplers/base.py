from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
import torch
from torch import Tensor


class BaseSampler(ABC):
    def __init__(self, diffusion, verbose: bool = True):
        self.diffusion = diffusion
        self.verbose = verbose

    @abstractmethod
    def __call__(self,
                 x_T: Tensor,
                 score_model: Callable,
                 n_steps: int = 500,
                 seed: Optional[int] = None,
                 callback: Optional[Callable[[Tensor, int], None]] = None,
                 callback_frequency: int = 50,
                 ) -> Tuple[Tensor, Tensor]:
        ...

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
