from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple, Optional

class VariancePreservingSchedule(BaseNoiseSchedule):
    def __init__(self, max_t: int, beta_min: float, beta_max: float):
        super().__init__(max_t)
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def __call__(self, t: Tensor) -> Tensor:
        # Calculate linear beta schedule
        beta_t = self.beta_min + (self.beta_max - self.beta_min) * t
        
        # Calculate alpha_bar_t for continuous time diffusion
        beta_integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)
        alpha_bar_t = torch.exp(-beta_integral)
        
        # Return sqrt(1-alpha_bar_t) - this is the coefficient for the noise term
        return torch.sqrt(1.0 - alpha_bar_t)
    
    def config(self) -> dict:
        return {
            "max_t": self.max_t,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max
        }

class VariancePreserving(BaseDiffusion):
    NEEDS_NOISE_SCHEDULE = False
    
    def __init__(self, max_t: Optional[int] = 1000, beta_min: Optional[float] = 0.1, beta_max: Optional[float] = 20.0):
        self.schedule = VariancePreservingSchedule(max_t=max_t, beta_min=beta_min, beta_max=beta_max)
        self.max_t = max_t
        
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # Calculate beta_t for each timestep
        beta_t = (self.schedule.beta_min + 
                 (self.schedule.beta_max - self.schedule.beta_min) * t)
        beta_t = beta_t.view(-1, 1, 1, 1)
        
        # VP SDE: dx = -0.5 * beta_t * x * dt + sqrt(beta_t) * dw
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        
        return drift, diffusion
    
    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # Get the noise coefficient (sqrt(1-alpha_bar_t))
        noise_coef = self.schedule(t)
        noise_coef = noise_coef.view(x0.shape[0], *([1] * (x0.dim() - 1)))
        
        # Calculate the signal coefficient (sqrt(alpha_bar_t))
        signal_coef = torch.sqrt(1.0 - noise_coef**2)
        
        # Sample x_t directly
        noise = torch.randn_like(x0)
        x_t = signal_coef * x0 + noise_coef * noise
        
        return x_t, noise
    
    def config(self) -> dict:
        return {
            "max_t": self.max_t,
            "beta_min": self.schedule.beta_min,
            "beta_max": self.schedule.beta_max,
        }