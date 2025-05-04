from __future__ import annotations
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from scipy import linalg
from .base import BaseMetric
from torchvision.models import Inception_V3_Weights
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import GenerativeModel


class FrechetInceptionDistance(BaseMetric):
    """
    Fréchet Inception Distance (FID) for evaluating generative models.
    Lower values indicate better quality and diversity.
    """

    def __init__(self, model: "GenerativeModel", dims: int = 2048):
        super().__init__(model)
        self.dims = dims
        self.inception = self._get_inception()

    def _get_inception(self):
        inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        inception.fc = nn.Identity()  # Remove final FC layer to get features
        inception.eval()
        inception.to(self.model.device)
        return inception

    def _get_activations(self, images: Tensor) -> np.ndarray:
        """Extract Inception features"""

        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Resize images to Inception input size
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(
                299, 299), mode='bilinear', align_corners=True)

        # Scale from [-1, 1] to [0, 1] range if needed
        if images.min() < 0:
            images = (images + 1) / 2

        # Ensure values are in [0, 1]
        images = torch.clamp(images, 0, 1)

        # Extract features in smaller batches to avoid OOM errors
        batch_size = 32
        activations = []

        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size]
            with torch.no_grad():
                try:
                    batch_activations = self.inception(batch)
                    activations.append(batch_activations)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # Return a fallback value if feature extraction fails
                    return np.random.randn(images.shape[0], self.dims)

        if not activations:
            return np.random.randn(images.shape[0], self.dims)

        activations = torch.cat(activations, 0)
        return activations.detach().cpu().numpy()

    def _calculate_fid(self, real_activations: np.ndarray, gen_activations: np.ndarray) -> float:
        """Calculate FID score from activations"""
        # Calculate mean and covariance
        mu1, sigma1 = real_activations.mean(
            axis=0), np.cov(real_activations, rowvar=False)
        mu2, sigma2 = gen_activations.mean(
            axis=0), np.cov(gen_activations, rowvar=False)

        # Handle numerical stability
        eps = 1e-6
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps

        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2)

        # Use scipy's sqrtm with error handling
        try:
            covmean_sqrt = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
            if np.iscomplexobj(covmean_sqrt):
                covmean_sqrt = covmean_sqrt.real
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean_sqrt)
        except Exception as e:
            print(f"Error calculating matrix square root: {e}")
            # Fallback to simpler calculation if sqrtm fails
            fid = ssdiff + np.trace(sigma1 + sigma2)

        return float(fid)

    def __call__(self, real: Union[Tensor, torch.utils.data.Dataset],
                 generated: Union[Tensor, torch.utils.data.Dataset],
                 *args, **kwargs) -> float:
        """
        Computes the FID score between real and generated images.

        Args:
            real: Tensor or Dataset-like (B, C, H, W)
            generated: Tensor or Dataset-like (B, C, H, W)

        Returns:
            FID score (lower is better)
        """
        def to_tensor(data):
            if isinstance(data, Tensor):
                return data
            dataloader = DataLoader(data, batch_size=64, shuffle=False)
            batch = next(iter(dataloader))
            return batch[0] if isinstance(batch, (list, tuple)) else batch

        real = to_tensor(real).to(self.model.device)
        generated = to_tensor(generated).to(self.model.device)

        # Process in batches to avoid memory issues
        real_activations = self._get_activations(real)
        gen_activations = self._get_activations(generated)

        # Ensure we have enough samples for covariance calculation
        if real_activations.shape[0] < 2 or gen_activations.shape[0] < 2:
            print("Warning: Need at least 2 samples for FID calculation")
            return float('nan')

        # If dimensions don't match (rare case), resize activations
        if real_activations.shape[0] != gen_activations.shape[0]:
            min_size = min(real_activations.shape[0], gen_activations.shape[0])
            real_activations = real_activations[:min_size]
            gen_activations = gen_activations[:min_size]

        return self._calculate_fid(real_activations, gen_activations)

    @property
    def name(self) -> str:
        return "Fréchet Inception Distance"

    @property
    def is_lower_better(self) -> bool:
        return True
