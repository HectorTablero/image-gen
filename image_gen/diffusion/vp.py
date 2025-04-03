from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple


class VariancePreserving(BaseDiffusion):
    def __init__(self, schedule: BaseNoiseSchedule):
        super().__init__(schedule)
        self._precompute_beta_integral()

    def _precompute_beta_integral(self):
        t_values = torch.linspace(0, self.schedule.max_t, int(self.schedule.max_t) + 1,
                                  device=self.schedule(torch.tensor(0.0)).device)
        beta_values = self.schedule(t_values)

        self.beta_integral = torch.cumsum(
            beta_values * (1.0 / len(t_values) * self.schedule.max_t), dim=0)

    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t)
        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
        diffusion = torch.sqrt(beta_t).view(-1, 1, 1, 1)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        t_scaled = (t / self.schedule.max_t * len(self.beta_integral) -
                    1).clamp(0, len(self.beta_integral) - 1)
        t_indices = t_scaled.long()

        beta_int = self.beta_integral.to(
            x0.device)[t_indices].view(-1, 1, 1, 1)
        mean = x0 * torch.exp(-0.5 * beta_int)
        sigma_t = torch.sqrt(1 - torch.exp(-beta_int))
        sigma = sigma_t.view(x0.shape[0], *([1] * (x0.dim() - 1)))
        noise = torch.randn_like(x0)
        xt = mean + sigma * noise
        return xt, noise
