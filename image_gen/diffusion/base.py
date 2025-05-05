"""Abstract base class for diffusion models.

This module defines the interface for diffusion models used in image generation.
All diffusion implementations should inherit from this base class and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any

from torch import Tensor

from ..noise import BaseNoiseSchedule


class BaseDiffusion(ABC):
    """Abstract base class for diffusion models.

    This class defines the interface for diffusion models and provides common
    functionality for forward and backward processes.

    Attributes:
        NEEDS_NOISE_SCHEDULE: Class constant indicating if a noise schedule is required.
    """

    NEEDS_NOISE_SCHEDULE = True

    def __init__(self, schedule: BaseNoiseSchedule, *_, **__):
        """Initialize the diffusion model.

        Args:
            schedule: A noise schedule that controls noise addition over time.
        """
        self.schedule = schedule

    @abstractmethod
    def forward_sde(self, x: Tensor, t: Tensor, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
        """Calculate drift and diffusion coefficients for forward SDE.

        Args:
            x: The input tensor representing current state.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (drift, diffusion) tensors.
        """
        pass

    @abstractmethod
    def forward_process(
        self, x0: Tensor, t: Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        """Apply the forward diffusion process.

        Args:
            x0: The input tensor representing initial state.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple of (noisy_sample, noise) tensors.
        """
        pass

    @abstractmethod
    def compute_loss(
        self, score: Tensor, noise: Tensor, t: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Compute loss between predicted and actual noise.

        Args:
            score: The predicted noise tensor.
            noise: The actual noise tensor.
            t: Time steps tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor representing the computed loss.
        """
        pass

    def backward_sde(
        self, x: Tensor, t: Tensor, score: Tensor, *_, **__
    ) -> Tuple[Tensor, Tensor]:
        """Compute the backward SDE coefficients.

        Args:
            x: The input tensor representing current state.
            t: Time steps tensor.
            score: The score function output.

        Returns:
            A tuple of (drift, diffusion) tensors for the backward process.
        """
        f, g = self.forward_sde(x, t)
        g_squared = g**2
        if g_squared.shape != score.shape:
            g_squared = g_squared.expand_as(score)

        return f - g_squared * score, g

    @property
    def schedule(self) -> BaseNoiseSchedule:
        """Get the noise schedule.

        Returns:
            The noise schedule object.
        """
        return self._schedule

    @schedule.setter
    def schedule(self, value: BaseNoiseSchedule):
        """Set the noise schedule.

        The schedule can only be set once during initialization.

        Args:
            value: The noise schedule object to set.

        Raises:
            AttributeError: If trying to change the schedule after initialization.
        """
        # Schedule shouldn't be allowed to change once the class is initialized
        if not hasattr(self, '_schedule'):
            self._schedule = value
            return
        raise AttributeError("Attribute 'schedule' is not settable")

    def config(self) -> dict:
        """Get configuration parameters for the diffusion model.

        Returns:
            A dictionary containing configuration parameters.
        """
        return {}

    def __str__(self) -> str:
        """Get string representation of the diffusion model.

        Returns:
            A string describing the model with its configuration parameters.
        """
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self._class_name}({params})"

    @property
    def _class_name(self) -> str:
        """Get the class name.

        This will be automatically overridden in custom classes made by users.

        Returns:
            The name of the class.
        """
        return self.__class__.__name__
