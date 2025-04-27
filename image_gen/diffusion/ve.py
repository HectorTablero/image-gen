from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple
import numpy as np


class VarianceExplodingSchedule(BaseNoiseSchedule):
    def __init__(self, sigma: float, *args, **kwargs):
        self.sigma = sigma

    def __call__(self, t: Tensor, *args, **kwargs) -> Tensor:
        log_sigma = torch.log(torch.tensor(
            self.sigma, dtype=torch.float32, device=t.device))
        return torch.sqrt(0.5 * (self.sigma ** (2 * t) - 1.0) / log_sigma)

    def integral_beta(self, t: Tensor, *args, **kwargs) -> Tensor:
        return 0.5 * (self.sigma ** (2 * t) - 1) / np.log(self.sigma)

    def config(self) -> dict:
        return {
            "sigma": self.sigma
        }


class VarianceExploding(BaseDiffusion):
    NEEDS_NOISE_SCHEDULE = False

    def __init__(self, *args, sigma: float = 25.0, **kwargs):
        super().__init__(VarianceExplodingSchedule(sigma))

    def forward_sde(self, x: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        drift = torch.zeros_like(x)
        diffusion = (self.schedule.sigma ** t).view(-1, 1, 1, 1)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        sigma_t = self.schedule(t, *args, **kwargs)
        sigma = sigma_t.view(x0.shape[0], *([1] * (x0.dim() - 1)))
        noise = torch.randn_like(x0)
        return x0 + sigma * noise, noise

    def compute_loss(self, score: Tensor, noise: Tensor, t: Tensor, *args, **kwargs) -> Tensor:
        sigma_t = self.schedule(t, *args, **kwargs)
        sigma_t = sigma_t.view(score.shape[0], *([1] * (score.dim() - 1)))
        loss = (sigma_t * score + noise) ** 2
        return loss.sum(dim=tuple(range(1, loss.dim())))

    def config(self) -> dict:
        return self.schedule.config()
