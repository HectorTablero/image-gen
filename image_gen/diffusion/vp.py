from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple


class VariancePreserving(BaseDiffusion):
    def _init_(self, schedule: BaseNoiseSchedule):
        super()._init_(schedule)
        # Precompute beta_t and alpha_bars for all timesteps
        self.max_t = self.schedule.max_t

    def _precompute_beta_and_alpha(self):
        t_values = torch.arange(
            self.max_t + 1, device='cpu', dtype=torch.float32)
        beta = self.schedule(t_values)
        cumsum_beta = torch.cumsum(beta, dim=0)

        self.beta = beta
        self.cumsum_beta = cumsum_beta
        self.alpha_bars = torch.exp(-cumsum_beta)

    @property
    def max_t(self) -> int:
        return self._max_t

    @max_t.setter
    def max_t(self, value: int):
        self._max_t = value
        self._precompute_beta_and_alpha()

    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_t = self.schedule(t).view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        alpha_bars = self.alpha_bars.to(x0.device)

        t = t.clamp(0, self.max_t).long()
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + \
            torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise
