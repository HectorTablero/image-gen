# from .vp import VariancePreserving
# import torch
# from torch import Tensor
# from typing import Tuple


# class SubVariancePreserving(VariancePreserving):
#     def forward_sde(self, x: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
#         beta_t = self.schedule(t, *args, **kwargs).view(-1, 1, 1, 1)
#         drift = -0.5 * beta_t * x

#         beta_integral = self.schedule.integral_beta(
#             t, *args, **kwargs).view(-1, 1, 1, 1)
#         diffusion = torch.sqrt(beta_t * (1 - torch.exp(-2 * beta_integral)))

#         return drift, diffusion

from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple

class SubVariancePreserving(BaseDiffusion):
    def forward_sde(self, x: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t).view(-1, 1, 1, 1)
        integral_beta_t = self.schedule.integral_beta(t).view(-1, 1, 1, 1)
        exponential_term = torch.exp(-2 * integral_beta_t)
        g_squared = beta_t * (1 - exponential_term)
        diffusion = torch.sqrt(g_squared)
        drift = -0.5 * beta_t * x
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        integral_beta = self.schedule.integral_beta(t)
        alpha_bar_t = torch.exp(-integral_beta).view(-1, 1, 1, 1)
        mu_x0 = torch.sqrt(alpha_bar_t) * x0
        sigma_t = 1 - alpha_bar_t
        noise = torch.randn_like(x0)
        xt = mu_x0 + sigma_t * noise
        return xt, noise

    def compute_loss(self, score: Tensor, noise: Tensor, t: Tensor, *args, **kwargs) -> Tensor:
        integral_beta = self.schedule.integral_beta(t)
        alpha_bar_t = torch.exp(-integral_beta)
        sigma_t = 1 - alpha_bar_t
        sigma_t = sigma_t.view(score.shape[0], *([1] * (score.dim() - 1)))
        loss = (sigma_t * score + noise) ** 2
        return loss.sum(dim=tuple(range(1, loss.dim())))
