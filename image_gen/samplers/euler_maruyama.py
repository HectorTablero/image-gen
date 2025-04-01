from .base import BaseSampler
import torch
from torch import Tensor
import numpy as np


class EulerMaruyama(BaseSampler):
    def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
        drift, diffusion = self.diffusion.forward_sde(x, t)
        adj_drift = drift - diffusion**2 * score
        noise = torch.randn_like(x)
        return x + adj_drift * self.dt + diffusion * np.sqrt(self.dt) * noise
