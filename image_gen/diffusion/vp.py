"""Variance Preserving diffusion model implementation.

This module implements the Variance Preserving diffusion model which 
is commonly used in diffusion-based generative models. It maintains a certain
level of variance throughout the diffusion process.
"""

from typing import Tuple, Any

import torch
from torch import Tensor

from .base import BaseDiffusion


class VariancePreserving(BaseDiffusion):
    """Variance Preserving diffusion model implementation.

    This class implements a diffusion model that preserves variance throughout
    the noise addition process. This approach is commonly used in various
    diffusion-based generative models.
    """

    def forward_sde(self, x: Tensor, t: Tensor, *args: Any, **kwargs: Any) -> Tuple[
            Tensor, Tensor]:
        """Calculate drift and diffusion coefficients for forward SDE.

        Args:
            x: The input tensor representing current state.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (drift, diffusion) tensors.
        """
        beta_t = self.schedule(t, *args, **kwargs).view(-1, 1, 1, 1)
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor, *args: Any, **kwargs: Any) -> Tuple[
            Tensor, Tensor]:
        """Apply the forward diffusion process.

        Adds noise to the input according to the variance preserving schedule.

        Args:
            x0: The input tensor representing initial state.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (noisy_sample, noise) tensors.
        """
        integral = self.schedule.integral_beta(t, *args, **kwargs)
        alpha_bar_t = torch.exp(-integral).view(-1, 1, 1, 1)

        noise = torch.randn_like(x0)
        xt = (torch.sqrt(alpha_bar_t) * x0 +
              torch.sqrt(1.0 - alpha_bar_t) * noise)

        return xt, noise

    def compute_loss(self, score: Tensor, noise: Tensor, t: Tensor,
                     *args: Any, **kwargs: Any) -> Tensor:
        """Compute loss between predicted score and actual noise.

        Args:
            score: The predicted noise tensor.
            noise: The actual noise tensor.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor representing the computed loss.
        """
        integral_beta = self.schedule.integral_beta(t, *args, **kwargs)
        alpha_bar_t = torch.exp(-integral_beta)
        sigma_t = torch.sqrt(1 - alpha_bar_t)
        sigma_t = sigma_t.view(score.shape[0], *([1] * (score.dim() - 1)))
        loss = (sigma_t * score + noise) ** 2
        return loss.sum(dim=tuple(range(1, loss.dim())))

    def config(self) -> dict:
        """Get configuration parameters for the diffusion model.

        Returns:
            A dictionary containing configuration parameters.
        """
        return self.schedule.config() if hasattr(self.schedule, "config") else {}
