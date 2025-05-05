"""Variance Exploding diffusion model implementation.

This module implements the Variance Exploding diffusion model and its 
corresponding noise schedule, which is particularly effective for image 
generation tasks.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Any

from .base import BaseDiffusion
from ..noise import BaseNoiseSchedule


class VarianceExplodingSchedule(BaseNoiseSchedule):
    """Variance Exploding noise schedule.

    This schedule models noise that increases exponentially over time,
    creating a "variance exploding" effect.

    Attributes:
        sigma: Base sigma value that controls the rate of variance explosion.
    """

    def __init__(self, sigma: float, *_, **__):
        """Initialize the variance exploding noise schedule.

        Args:
            sigma: Base sigma value for the schedule.
        """
        self.sigma = sigma

    def __call__(self, t: Tensor, *_, **__) -> Tensor:
        """Calculate the noise magnitude at time t.

        Args:
            t: Time step tensor.

        Returns:
            Tensor containing noise magnitudes at time t.
        """
        log_sigma = torch.log(torch.tensor(
            self.sigma, dtype=torch.float32, device=t.device))
        return torch.sqrt(0.5 * (self.sigma ** (2 * t) - 1.0) / log_sigma)

    def integral_beta(self, t: Tensor, *_, **__) -> Tensor:
        """Calculate the integrated noise intensity up to time t.

        Args:
            t: Time step tensor.

        Returns:
            Tensor containing integrated noise values.
        """
        return 0.5 * (self.sigma ** (2 * t) - 1) / np.log(self.sigma)

    def config(self) -> dict:
        """Get configuration parameters for the schedule.

        Returns:
            Dictionary containing configuration parameters.
        """
        return {
            "sigma": self.sigma
        }


class VarianceExploding(BaseDiffusion):
    """Variance Exploding diffusion model implementation.

    This model implements diffusion using a variance exploding process,
    where the noise increases exponentially with time.

    Attributes:
        NEEDS_NOISE_SCHEDULE: Class constant indicating if a custom noise
            schedule is required.
    """

    NEEDS_NOISE_SCHEDULE = False

    def __init__(self, *_, sigma: float = 25.0, **__):
        """Initialize the variance exploding diffusion model.

        Args:
            sigma: Base sigma value for variance control. Defaults to 25.0.
        """
        super().__init__(VarianceExplodingSchedule(sigma))

    def forward_sde(self, x: Tensor, t: Tensor, *_, **__) -> Tuple[
            Tensor, Tensor]:
        """Calculate drift and diffusion for the forward SDE.

        Args:
            x: Input tensor representing the current state.
            t: Time steps tensor.

        Returns:
            Tuple of (drift, diffusion) tensors.
        """
        drift = torch.zeros_like(x)
        diffusion = (self.schedule.sigma ** t).view(-1, 1, 1, 1)
        return drift, diffusion

    def forward_process(self, x0: Tensor, t: Tensor, *args: Any, **kwargs: Any) -> Tuple[
            Tensor, Tensor]:
        """Apply the forward diffusion process.

        Args:
            x0: Input tensor representing initial state.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (noisy_sample, noise) tensors.
        """
        sigma_t = self.schedule(t, *args, **kwargs)
        sigma = sigma_t.view(x0.shape[0], *([1] * (x0.dim() - 1)))
        noise = torch.randn_like(x0)
        return x0 + sigma * noise, noise

    def compute_loss(self, score: Tensor, noise: Tensor, t: Tensor,
                     *args: Any, **kwargs: Any) -> Tensor:
        """Compute loss between predicted score and actual noise.

        Args:
            score: Predicted score tensor.
            noise: Actual noise tensor.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Loss tensor.
        """
        sigma_t = self.schedule(t, *args, **kwargs)
        sigma_t = sigma_t.view(score.shape[0], *([1] * (score.dim() - 1)))
        loss = (sigma_t * score + noise) ** 2
        return loss.sum(dim=tuple(range(1, loss.dim())))

    def config(self) -> dict:
        """Get configuration parameters for the diffusion model.

        Returns:
            Dictionary containing configuration parameters.
        """
        return self.schedule.config()
