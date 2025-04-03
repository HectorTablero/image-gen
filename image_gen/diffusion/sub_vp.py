from .vp import VariancePreserving
import torch
from torch import Tensor
from typing import Tuple


class SubVariancePreserving(VariancePreserving):
    def forward_sde(self, x: Tensor, t: Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_t = self.schedule(t)
        beta_int = self.beta_integral[t.long().to(
            self.beta_integral.device)].to(x.device)
        drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
        diffusion = torch.sqrt(
            beta_t * (1 - torch.exp(-2 * beta_int))).view(-1, 1, 1, 1)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        beta_int = self.beta_integral.to(x0.device)[t.long()].view(-1, 1, 1, 1)
        mean = x0 * torch.exp(-0.5 * beta_int)
        temp = torch.sqrt(1 - torch.exp(-beta_int))
        sigma_t = torch.sqrt(temp * (1 - torch.exp(-2 * beta_int)))
        sigma = sigma_t.view(x0.shape[0], *([1] * (x0.dim() - 1)))
        noise = torch.randn_like(x0)
        xt = mean + sigma * noise
        return xt, noise
