from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple

class VariancePreserving(BaseDiffusion):
    NEEDS_NOISE_SCHEDULE = True  # Requiere un noise schedule (beta_t)
    
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t).view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t)
        
        # Cálculo de alpha_bar_t (integral de beta)
        integral = self.schedule.beta_min * t + (self.schedule.beta_max - self.schedule.beta_min) * (t**2) / (2 * self.max_t)
        alpha_bar_t = torch.exp(-integral)
        
        # Coeficientes para el ruido
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        
        return xt, noise

    def backward_sde(self, x: Tensor, t: Tensor, score: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t).view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x - beta_t * score
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def config(self) -> dict:
        return {
            "type": "VP",
            "schedule": self.schedule.schedule.config(),
            "beta_min": self.schedule.beta_min,
            "beta_max": self.schedule.beta_max
        }
