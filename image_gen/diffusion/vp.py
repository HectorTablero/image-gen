from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple


class VariancePreserving(BaseDiffusion):
    def forward_sde(self, x: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t, *args, **kwargs).view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        integral = self.schedule.integral_beta(t, *args, **kwargs)
        alpha_bar_t = torch.exp(-integral).view(-1, 1, 1, 1)

        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar_t) * x0 + \
            torch.sqrt(1.0 - alpha_bar_t) * noise

        return xt, noise

    def compute_loss(self, score: Tensor, noise: Tensor, t: Tensor, *args, **kwargs) -> Tensor:
        integral_beta = self.schedule.integral_beta(t, *args, **kwargs)
        alpha_bar_t = torch.exp(-integral_beta)
        sigma_t = torch.sqrt(1 - alpha_bar_t)
        sigma_t = sigma_t.view(score.shape[0], *([1] * (score.dim() - 1)))
        loss = (sigma_t * score + noise) ** 2
        return loss.sum(dim=tuple(range(1, loss.dim())))
