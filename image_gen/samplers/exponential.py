from .base import BaseSampler
from typing import Optional
from torch import Tensor
import torch


class ExponentialIntegrator(BaseSampler):
    def __init__(self, diffusion, num_steps: Optional[int] = None):
        super().__init__(diffusion, num_steps)
        self._precompute_coefficients()

    def _precompute_coefficients(self):
        ts = self.timesteps(reverse=False)
        beta_integrals = torch.stack(
            [self.diffusion.beta_integral(t) for t in ts])
        self.exp_terms = torch.exp(-0.5 * beta_integrals * self.dt)

    def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        idx = (t * self.num_steps).long()
        drift, diffusion = self.diffusion.forward_sde(x, t)
        noise_term = diffusion * \
            torch.sqrt(1 - self.exp_terms[idx]**2) * torch.randn_like(x)
        return self.exp_terms[idx] * (x + diffusion**2 * score * self.dt) + noise_term
