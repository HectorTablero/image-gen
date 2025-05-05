"""Base sampler class for diffusion models.

This module provides a base abstract class for all samplers used in diffusion
models. It defines the common interface that all samplers should implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

from torch import Tensor

from ..diffusion import BaseDiffusion


class BaseSampler(ABC):
    """Abstract base class for all diffusion model samplers.

    All samplers inherit from this class and must implement the call method
    which performs the actual sampling process.

    Attributes:
        diffusion: The diffusion model to sample from.
        verbose: Whether to print progress information during sampling.
    """

    def __init__(self, diffusion: BaseDiffusion, *_, verbose: bool = True, **__):
        """Initialize the sampler.

        Args:
            diffusion: The diffusion model to sample from.
            verbose: Whether to print progress information during sampling.
                Defaults to True.
        """
        self.diffusion = diffusion
        self.verbose = verbose

    @abstractmethod
    def __call__(
            self,
            x_T: Tensor,
            score_model: Callable,
            *args: Any,
            n_steps: int = 500,
            seed: Optional[int] = None,
            callback: Optional[Callable[[Tensor, int], None]] = None,
            callback_frequency: int = 50,
            guidance: Optional[Callable[[
                Tensor, Tensor, Tensor], Tensor]] = None,
            **kwargs: Any
    ) -> Tensor:
        """Perform the sampling process.

        Args:
            x_T: The initial noise tensor to start sampling from.
            score_model: The score model function that predicts the score.
            *args: Additional positional arguments.
            n_steps: Number of sampling steps. Defaults to 500.
            seed: Random seed for reproducibility. Defaults to None.
            callback: Optional function called during sampling to monitor progress.
                It takes the current sample and step number as inputs.
                Defaults to None.
            callback_frequency: How often to call the callback function.
                Defaults to 50.
            guidance: Optional guidance function for conditional sampling.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the final sample and the sequence of all samples.
        """
        ...

    def config(self) -> dict:
        """Return the configuration of the sampler.

        Returns:
            A dictionary with the sampler's configuration parameters.
        """
        return {}

    def __str__(self) -> str:
        """Return a string representation of the sampler.

        Returns:
            A string with the sampler's class name and its configuration.
        """
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self._class_name}({params})"

    @property
    def _class_name(self) -> str:
        """Get the class name of the sampler.

        This property is automatically overridden in custom classes made by users.

        Returns:
            The name of the sampler class.
        """
        return self.__class__.__name__
