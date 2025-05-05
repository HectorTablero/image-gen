"""Base metric class for image generation evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from torch import Tensor

if TYPE_CHECKING:
    from ..base import GenerativeModel


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    All metrics must inherit from this class and implement the required methods.

    Attributes:
        model: The generative model being evaluated.
    """

    def __init__(self, model: GenerativeModel):
        """Initialize the metric with a generative model.

        Args:
            model: The generative model to be evaluated with this metric.
        """
        self.model = model

    @abstractmethod
    def __call__(self, real: Tensor, generated: Tensor, *args: Any, **kwargs: Any) -> float:
        """Compute the metric value between real and generated samples.

        Args:
            real: Tensor containing real samples.
            generated: Tensor containing generated samples.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The computed metric value.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: The name of the metric.
        """
        pass

    @property
    @abstractmethod
    def is_lower_better(self) -> bool:
        """Indicates whether a lower metric value is better.

        Returns:
            bool: True if lower values indicate better performance, False otherwise.
        """
        pass

    def config(self) -> dict:
        """Get the configuration parameters for this metric.

        Returns:
            dict: Dictionary containing configuration parameters.
        """
        return {}

    def __str__(self) -> str:
        """Get a string representation of the metric.

        Returns:
            str: String representation including class name and parameters.
        """
        config = self.config()
        params = ", ".join(f"{k}: {v}" for k, v in config.items())
        return f"{self._class_name}({params})"

    @property
    def _class_name(self) -> str:
        """Get the class name.

        This property will be automatically overridden in custom classes made by users.

        Returns:
            str: The class name.
        """
        return self.__class__.__name__
