from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple


class VariancePreserving(BaseDiffusion):
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
        beta_integral = self.schedule.beta_min * self.schedule(t) + 0.5 * \
            (self.schedule.beta_max - self.schedule.beta_min) * \
            (self.schedule(t) ** 2)
        alpha_bar_t = torch.exp(-beta_integral)

        noise_coef = torch.sqrt(1.0 - alpha_bar_t)
        noise_coef = noise_coef.view(x0.shape[0], *([1] * (x0.dim() - 1)))

        signal_coef = torch.sqrt(1.0 - noise_coef**2)

        noise = torch.randn_like(x0)
        x_t = signal_coef * x0 + noise_coef * noise

        return x_t, noise
