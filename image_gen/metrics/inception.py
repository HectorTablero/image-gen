from .base import BaseMetric
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from typing import Tuple, Optional
from torchvision.models import Inception_V3_Weights

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
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.n_splits = n_splits
        self.model = None
        
    def _get_model(self):
        if self.model is None:
            try:
                self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
                self.model.eval()
                self.model.to(self.device)
            except Exception as e:
                print(f"Error loading Inception model: {e}")
                raise
        return self.model
    
    def _get_predictions(self, images: Tensor) -> np.ndarray:
        """Get softmax predictions from the Inception model"""
        model = self._get_model()
        
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Resize images to Inception input size
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
        
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
                    pred = model(batch)
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
        """Calculate Inception Score from softmax predictions"""
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
            kl_divergences = split * (np.log(split + 1e-10) - np.log(p_y + 1e-10))
            kl_d = np.mean(np.sum(kl_divergences, axis=1))
            scores.append(np.exp(kl_d))
            
        if len(scores) == 1:
            return float(scores[0]), 0.0
        return float(np.mean(scores)), float(np.std(scores))
    
    def __call__(self, real: Optional[Tensor] = None, generated: Optional[Tensor] = None, batch_size: int = 32, *args, **kwargs) -> float:
        """
        Computes the Inception Score for generated images.
        
        Args:
            real: Not used for IS, included for API compatibility
            generated: Tensor of generated images (B, C, H, W)
            batch_size: Batch size for feature extraction
            
        Returns:
            Inception Score (higher is better)
        """
        # Handle the case where generated is passed as a keyword argument
        if generated is None and 'generated' in kwargs:
            generated = kwargs['generated']
            
        # Only generated images are used for IS
        if generated is None:
            raise ValueError("Generated images must be provided for Inception Score")
            
        # Move to device
        generated = generated.to(self.device)
        
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
        
        # Get predictions
        all_predictions = self._get_predictions(generated)
        
        # Calculate IS with standard deviation
        return self._calculate_is(all_predictions)
    
    @property
    def name(self) -> str:
        return "InceptionScore"