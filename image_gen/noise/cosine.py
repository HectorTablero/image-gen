"""Implementation of a cosine noise schedule for diffusion models.

This module provides a cosine-based noise scheduling as proposed in
"Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
The schedule offers smoother transitions between noise levels.
"""

import math

import torch
from torch import Tensor

from .base import BaseNoiseSchedule


class CosineNoiseSchedule(BaseNoiseSchedule):
    """Cosine noise schedule implementation for diffusion models.

    This noise schedule uses a cosine function to define the noise levels over time,
    which typically results in better sample quality compared to linear schedules.
    """

    def __init__(
        self,
        *_,
        s: float = 0.008,
        beta_min: float = 1e-4,
        beta_max: float = 20.0,
        **__
    ):
        """Initialize the cosine noise schedule.

        Args:
            s: Small offset to prevent alpha_bar(t) from being too small near t=1.
                Defaults to 0.008.
            beta_min: Minimum noise level for numerical stability.
                Defaults to 0.0001.
            beta_max: Maximum noise level for numerical stability.
                Defaults to 20.0.
        """
        self.s = s
        self.min_beta = beta_min
        self.max_beta = beta_max

    def alpha_bar(self, t: Tensor) -> Tensor:
        """Compute the cumulative product of (1-beta) up to time t.

        Uses the cosine formula: alpha_bar(t) = cos^2((t/T + s)/(1 + s) * π/2)

        Args:
            t: Tensor of timesteps in [0, 1] range.

        Returns:
            Tensor: alpha_bar values at the specified timesteps.
        """
        return torch.cos((t + self.s) / (1.0 + self.s) * math.pi * 0.5) ** 2

    def __call__(self, t: Tensor, *_, **__) -> Tensor:
        """Compute beta(t) at timestep t.

        For cosine schedule, beta(t) is derived from the derivative of alpha_bar(t):
        beta(t) = -d(log(alpha_bar))/dt = -d(alpha_bar)/dt / alpha_bar

        Args:
            t: Tensor of timesteps in [0, 1] range.

        Returns:
            Tensor: Beta values at specified timesteps.
        """
        # Compute f(t) = (t + s)/(1 + s) * π/2
        f_t = (t + self.s) / (1.0 + self.s) * math.pi * 0.5

        # Compute d(alpha_bar)/dt = d(cos^2(f(t)))/dt
        # = 2 * cos(f(t)) * (-sin(f(t))) * d(f(t))/dt
        # = -π * sin(f(t)) * cos(f(t)) / (1 + s)
        dalpha_bar_dt = -math.pi * \
            torch.sin(f_t) * torch.cos(f_t) / (1.0 + self.s)

        # Compute alpha_bar(t)
        alpha_bar_t = self.alpha_bar(t)

        # Compute beta(t) = -d(log(alpha_bar))/dt = -d(alpha_bar)/dt / alpha_bar
        beta_t = -dalpha_bar_dt / torch.clamp(alpha_bar_t, min=1e-8)

        # Ensure numerical stability
        beta_t = torch.clamp(beta_t, min=self.min_beta, max=self.max_beta)

        return beta_t

    def integral_beta(self, t: Tensor, *_, **__) -> Tensor:
        """Compute the integral of beta from 0 to t.

        For cosine schedule, this equals -log(alpha_bar(t)) which represents
        the total amount of noise added up to time t.

        Args:
            t: Tensor of timesteps in [0, 1] range.

        Returns:
            Tensor: Integrated beta values from 0 to t.
        """
        alpha_bar_t = self.alpha_bar(t)
        return -torch.log(torch.clamp(alpha_bar_t, min=1e-8))

    def config(self) -> dict:
        """Get the configuration parameters of the noise schedule.

        Returns:
            dict: Dictionary containing the configuration parameters.
        """
        return {
            "s": self.s,
            "min_beta": self.min_beta,
            "max_beta": self.max_beta
        }
