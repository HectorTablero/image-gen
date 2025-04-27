from .base import BaseMetric
from torch import Tensor
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from typing import Tuple

class InceptionScore(BaseMetric):
    """
    Inception Score (IS) for evaluating generative models.
    Higher values indicate better quality and diversity.
    """
    def __init__(self, device='cuda', n_splits=10):
        """
        Args:
            device: Device to run the model on
            n_splits: Number of splits for calculating standard deviation
        """
        self.device = device
        self.n_splits = n_splits
        self.model = None
        
    def _get_model(self):
        if self.model is None:
            self.model = inception_v3(pretrained=True, transform_input=False)
            self.model.eval()
            self.model.to(self.device)
        return self.model
    
    def _get_predictions(self, images: Tensor) -> np.ndarray:
        """Get softmax predictions from the Inception model"""
        model = self._get_model()
        
        # Resize images to Inception input size if needed
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
        
        # Scale from [-1, 1] to [0, 1] range if needed
        if images.min() < 0:
            images = (images + 1) / 2
            
        # Ensure values are in [0, 1]
        images = torch.clamp(images, 0, 1)
        
        # Get predictions
        with torch.no_grad():
            pred = F.softmax(model(images), dim=1)
            
        return pred.detach().cpu().numpy()
    
    def _calculate_is(self, predictions: np.ndarray) -> Tuple[float, float]:
        """Calculate Inception Score from softmax predictions"""
        # Split predictions to calculate mean and std
        scores = []
        splits = np.array_split(predictions, self.n_splits)
        
        for split in splits:
            # Calculate KL divergence
            p_y = np.mean(split, axis=0)
            kl_divergences = split * (np.log(split + 1e-10) - np.log(p_y + 1e-10))
            kl_d = np.mean(np.sum(kl_divergences, axis=1))
            scores.append(np.exp(kl_d))
            
        return float(np.mean(scores)), float(np.std(scores))
    
    def __call__(self, real: Tensor = None, generated: Tensor = None, batch_size: int = 32, *args, **kwargs) -> float:
        """
        Computes the Inception Score for generated images.
        
        Args:
            real: Not used for IS, included for API compatibility
            generated: Tensor of generated images (B, C, H, W)
            batch_size: Batch size for feature extraction
            
        Returns:
            Inception Score (higher is better)
        """
        # Only generated images are used for IS
        if generated is None:
            raise ValueError("Generated images must be provided for Inception Score")
            
        # Move to device
        generated = generated.to(self.device)
        
        # Process in batches to avoid memory issues
        all_predictions = []
        
        for i in range(0, generated.shape[0], batch_size):
            gen_batch = generated[i:i + batch_size]
            all_predictions.append(self._get_predictions(gen_batch))
            
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate IS
        is_mean, is_std = self._calculate_is(all_predictions)
        
        # Return just the mean for compatibility with the BaseMetric interface
        return is_mean
    
    def calculate_with_std(self, generated: Tensor, batch_size: int = 32) -> Tuple[float, float]:
        """
        Calculate Inception Score with standard deviation.
        
        Args:
            generated: Tensor of generated images
            batch_size: Batch size for feature extraction
            
        Returns:
            Tuple of (mean, std) of Inception Score
        """
        # Move to device
        generated = generated.to(self.device)
        
        # Process in batches
        all_predictions = []
        
        for i in range(0, generated.shape[0], batch_size):
            gen_batch = generated[i:i + batch_size]
            all_predictions.append(self._get_predictions(gen_batch))
            
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # Calculate IS with standard deviation
        return self._calculate_is(all_predictions)
    
    @property
    def name(self) -> str:
        return "InceptionScore"