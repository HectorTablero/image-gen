"""Base class for noise schedules in diffusion models."""

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Any


class BaseNoiseSchedule(ABC):
    """Abstract base class defining the interface for noise schedules.

    All noise schedule implementations should inherit from this class
    and implement the required abstract methods.
    """

    @abstractmethod
    def __call__(self, t: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Calculate noise at specific timesteps.

        Args:
            t: Tensor containing timestep values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Noise values corresponding to the input timesteps.
        """
        ...

    @abstractmethod
    def integral_beta(self, t: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Calculate the integral of the noise function up to timestep t.

        Args:
            t: Tensor containing timestep values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: Integrated noise values corresponding to the input timesteps.
        """
        ...

    def config(self) -> dict:
        """Get the configuration parameters of the noise schedule.

        Returns:
            dict: Configuration parameters of the noise schedule.
        """
        return {}

    def __str__(self) -> str:
        """Generate a string representation of the noise schedule.

        Returns:
            str: String representation including class name and parameters.
        """
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self._class_name}({params})"

    @property
    def _class_name(self) -> str:
        """Get the class name of the noise schedule.

        This property will be automatically overridden in custom classes
        made by users.

        Returns:
            str: Name of the class.
        """
        # This will be automatically overridden in custom classes made by users
        return self.__class__.__name__
