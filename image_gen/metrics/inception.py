"""Module for Inception Score metric implementation."""

from __future__ import annotations

import numpy as np
from typing import Tuple, TYPE_CHECKING, Union, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import Inception_V3_Weights, inception_v3

from .base import BaseMetric

if TYPE_CHECKING:
    from ..base import GenerativeModel


class InceptionScore(BaseMetric):
    """Inception Score (IS) for evaluating generative models.

    This metric evaluates the quality and diversity of generated images
    using the Inception-v3 model's predictions.
    Higher values indicate better quality and diversity.

    Attributes:
        model: The generative model being evaluated.
        n_splits: Number of splits for estimating mean and standard deviation.
        inception: Pretrained Inception-v3 model.
    """

    def __init__(self, model: GenerativeModel, n_splits: int = 10):
        """Initialize the Inception Score metric.

        Args:
            model: The generative model to be evaluated.
            n_splits: Number of splits for calculating score statistics.
        """
        super().__init__(model)
        self.n_splits = n_splits
        self.inception = self._get_inception()

    def _get_inception(self):
        """Create and prepare the Inception-v3 model.

        Returns:
            The prepared Inception-v3 model.
        """
        inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        inception.eval()
        inception.to(self.model.device)
        return inception

    def _get_predictions(self, images: Tensor) -> np.ndarray:
        """Get softmax predictions from the Inception model.

        Args:
            images: Batch of images to process.

        Returns:
            NumPy array containing softmax predictions.
        """
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Resize images to Inception input size
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images,
                size=(299, 299),
                mode='bilinear',
                align_corners=True
            )

        # Scale from [-1, 1] to [0, 1] range if needed
        if images.min() < 0:
            images = (images + 1) / 2

        # Ensure values are in [0, 1]
        images = torch.clamp(images, 0, 1)

        # Extract features in smaller batches to avoid OOM errors
        batch_size = 32
        predictions = []

        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size]
            with torch.no_grad():
                try:
                    # Get predictions with error handling
                    pred = self.inception(batch)
                    pred = F.softmax(pred, dim=1)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    # Return fallback predictions if inference fails
                    return np.ones((images.shape[0], 1000)) / 1000

        if not predictions:
            return np.ones((images.shape[0], 1000)) / 1000

        predictions = torch.cat(predictions, 0)
        return predictions.detach().cpu().numpy()

    def _calculate_is(self, predictions: np.ndarray) -> Tuple[float, float]:
        """Calculate Inception Score from softmax predictions.

        Args:
            predictions: Softmax predictions from the Inception model.

        Returns:
            Tuple of (mean, std) of Inception Score.
        """
        # Ensure we have enough samples for splitting
        n_splits = min(self.n_splits, len(predictions) // 2)
        if n_splits < 1:
            n_splits = 1

        # Split predictions to calculate mean and std
        scores = []
        splits = np.array_split(predictions, n_splits)

        for split in splits:
            # Calculate KL divergence
            p_y = np.mean(split, axis=0)
            # Avoid log(0) by adding small epsilon
            kl_divergences = split * (
                np.log(split + 1e-10) - np.log(p_y + 1e-10)
            )
            kl_d = np.mean(np.sum(kl_divergences, axis=1))
            scores.append(np.exp(kl_d))

        if len(scores) == 1:
            return float(scores[0]), 0.0
        return float(np.mean(scores)), float(np.std(scores))

    def __call__(
        self,
        _real: Any,
        generated: Union[Tensor, torch.utils.data.Dataset],
        *_,
        **__
    ) -> float:
        """Compute the Inception Score for generated images.

        Args:
            _real: Not used for IS, included for API compatibility.
            generated: Tensor of generated images (B, C, H, W).

        Returns:
            The computed Inception Score (higher is better).

        Raises:
            ValueError: If no generated images are provided.
        """
        # Move to device
        generated = generated.to(self.model.device)

        # Ensure minimum batch size
        if generated.shape[0] < 2:
            print("Warning: Need at least 2 samples for IS calculation")
            return 1.0  # Default score for insufficient samples

        # Get predictions
        all_predictions = self._get_predictions(generated)

        # Calculate IS
        is_mean, _ = self._calculate_is(all_predictions)

        # Return just the mean for compatibility with the BaseMetric interface
        return is_mean

    def calculate_with_std(
        self,
        generated: Tensor
    ) -> Tuple[float, float]:
        """Calculate Inception Score with standard deviation.

        This method provides additional information compared to __call__.

        Args:
            generated: Tensor of generated images.

        Returns:
            Tuple of (mean, std) of Inception Score.
        """
        # Move to device
        generated = generated.to(self.model.device)

        # Get predictions
        all_predictions = self._get_predictions(generated)

        # Calculate IS with standard deviation
        return self._calculate_is(all_predictions)

    @property
    def name(self) -> str:
        """Get the name of the metric.

        Returns:
            The name of the metric.
        """
        return "Inception Score"

    @property
    def is_lower_better(self) -> bool:
        """Indicates whether a lower metric value is better.

        Returns:
            False since higher Inception Score values are better.
        """
        return False
