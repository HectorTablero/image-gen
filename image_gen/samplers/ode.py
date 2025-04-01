from .base import BaseSampler
from torch import Tensor


class ODEProbabilityFlow(BaseSampler):
    def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        drift, diffusion = self.diffusion.forward_sde(x, t)
        adj_drift = drift - 0.5 * diffusion**2 * score
        return x + adj_drift * self.dt
