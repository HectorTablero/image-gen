from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule
import torch
from torch import Tensor
from typing import Tuple, Optional


class VESchedule(BaseNoiseSchedule):
    def __init__(self, max_t: int, sigma: float):
        super().__init__(max_t)
        self.sigma = sigma

    def __call__(self, t: Tensor) -> Tensor:
        log_sigma = torch.log(torch.tensor(
            self.sigma, dtype=torch.float32, device=t.device))
        return torch.sqrt(0.5 * (self.sigma ** (2 * t) - 1.0) / log_sigma)


class VarianceExploding(BaseDiffusion):
    NEEDS_NOISE_SCHEDULE = False

    def __init__(self, max_t: Optional[int] = 1000.0, sigma: Optional[float] = 25.0):
        self.schedule = VESchedule(max_t=max_t, sigma=sigma)
        self.max_t = max_t

    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        drift = torch.zeros_like(x)
        diffusion = (self.schedule.sigma ** t).view(-1, 1, 1, 1)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        sigma_t = self.schedule(t)
        sigma = sigma_t.view(x0.shape[0], *([1] * (x0.dim() - 1)))
        noise = torch.randn_like(x0)
        return x0 + sigma * noise, noise


if __name__ == "__main__":
    v = VESchedule
