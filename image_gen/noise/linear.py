"""Implementation of a linear noise schedule for diffusion models."""

from torch import Tensor

from .base import BaseNoiseSchedule


class LinearNoiseSchedule(BaseNoiseSchedule):
    """Linear noise schedule that increases linearly over time.

    This class implements a simple linear noise schedule where the noise
    increases linearly from beta_min to beta_max over the diffusion process.
    """

    def __init__(
        self,
        *_,
        beta_min: float = 0.0001,
        beta_max: float = 20.0,
        **__
    ):
        """Initialize the linear noise schedule.

        Args:
            beta_min: Minimum noise value at t=0. Defaults to 0.0001.
            beta_max: Maximum noise value at t=1. Defaults to 20.0.
        """
        self.beta_min = beta_min
        self.beta_max = beta_max

    def __call__(self, t: Tensor, *_, **__) -> Tensor:
        """Calculate noise at specific timesteps.

        Args:
            t: Tensor containing timestep values in range [0, 1].

        Returns:
            Tensor: Noise values corresponding to the input timesteps.
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: Tensor, *_, **__) -> Tensor:
        """Calculate the integral of the noise function up to timestep t.

        The analytical solution for the integral of a linear function
        from 0 to t is: beta_min * t + 0.5 * (beta_max - beta_min) * t^2.

        Args:
            t: Tensor containing timestep values in range [0, 1].

        Returns:
            Tensor: Integrated noise values corresponding to the input timesteps.
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

    def config(self) -> dict:
        """Get the configuration parameters of the noise schedule.

        Returns:
            dict: Dictionary containing the configuration parameters.
        """
        return {
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }
