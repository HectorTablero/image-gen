from .base import BaseNoiseSchedule
from torch import Tensor
import torch
import math


class CosineNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, max_t: float = 1.0, s: float = 0.008, beta_min: float = 0.0001, beta_max: float = 0.999):
        """
        Implements the cosine noise schedule as described in the paper.
        
        Args:
            max_t: Maximum timestep value (corresponds to T in the equations)
            s: Offset parameter for the cosine schedule
            beta_min: Minimum value for β(t)
            beta_max: Maximum value for β(t)
        """
        super().__init__(max_t)
        self.s = s  # offset parameter
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def alpha_bar(self, t: Tensor) -> Tensor:
        """Compute ᾱ(t) according to the cosine schedule."""
        # Normalize t to [0, 1]
        t_norm = (t / self.max_t).clamp(0.0, 1.0)
        
        # Compute angle
        angle_num = math.pi/2 * (t_norm + self.s)/(1 + self.s)
        angle_den = math.pi/2 * self.s/(1 + self.s)
        
        # Compute ᾱ(t)
        numerator = torch.cos(angle_num) ** 2
        denominator = math.cos(angle_den) ** 2
        
        return numerator / denominator

    
    def __call__(self, t: Tensor) -> Tensor:
        return torch.clamp(1. - self.alpha_bar(t), max=0.999)
    
    def config(self) -> dict:
        return {
            "max_t": self.max_t,
            "s": self.s,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }

