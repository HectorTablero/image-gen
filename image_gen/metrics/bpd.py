"""Module for Bits Per Dimension metric implementation."""

import numpy as np
from typing import Union, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseMetric


class BitsPerDimension(BaseMetric):
    """Bits per dimension (BPD) metric for evaluating density models.

    This metric evaluates probabilistic generative models based on their
    log-likelihood. Lower values indicate better models.

    Attributes:
        model: The generative model being evaluated.
    """

    def __call__(
        self,
        real: Union[Tensor, torch.utils.data.Dataset],
        _generated: Any,
        *_,
        **__
    ) -> float:
        """Computes bits per dimension for the real data.

        Args:
            real: Tensor or Dataset-like object (Dataset, Subset, etc.)
            _generated: Not used for BPD, included for API compatibility

        Returns:
            float: The computed BPD value (lower is better).
        """
        # If input is not a Tensor, assume it's a Dataset-like and load it
        if not isinstance(real, Tensor):
            dataloader = DataLoader(real, batch_size=64, shuffle=False)
            real = next(iter(dataloader))[0]  # Get first batch only

        real = real.to(self.model.device)

        # Scale images to [-1, 1] range if they're in [0, 1]
        if real.min() >= 0 and real.max() <= 1:
            real = real * 2 - 1

        # We use the model's loss function as a proxy for NLL
        with torch.no_grad():
            # Sample multiple random times for more accurate estimate
            losses = []
            # Average over multiple time samples
            for _ in range(10):
                loss = self.model.loss_function(real)
                losses.append(loss.detach().cpu())

            # Take the mean loss
            mean_loss = torch.stack(losses).mean()

        # Convert to bits per dimension
        batch_size, channels, height, width = real.shape
        num_dims = channels * height * width
        bpd = mean_loss / np.log(2) / num_dims

        return bpd.item()

    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return "Bits Per Dimension"

    @property
    def is_lower_better(self) -> bool:
        """Indicates whether a lower metric value is better.

        Returns:
            bool: True if lower values indicate better performance.
        """
        return True
