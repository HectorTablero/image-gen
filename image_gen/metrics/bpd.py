from .base import BaseMetric
from torch import Tensor
import torch
import numpy as np


class BitsPerDimension(BaseMetric):
    """
    Bits per dimension (BPD) metric for evaluating density models.
    Lower values indicate better models.
    """
    def __init__(self, model=None, diffusion_model=None):
        """
        Args:
            diffusion_model: Optional diffusion model with a method to compute negative log-likelihood
        """
        self.model = model
        self.diffusion_model = diffusion_model
        
    def __call__(self, real: Tensor, generated: Tensor = None, *args, **kwargs) -> float:
        """
        Computes bits per dimension for the real data.
        
        Args:
            real: Tensor of real data (B, C, H, W)
            generated: Not used for BPD, included for API compatibility
            
        Returns:
            BPD value (lower is better)
        """
        if self.diffusion_model is None:
            raise ValueError("Diffusion model must be provided for BPD calculation")
        
        # Determine the device - handle both GenerativeModel and direct diffusion models
        if hasattr(self.diffusion_model, 'device'):
            # GenerativeModel case
            device = self.diffusion_model.device
        else:
            # Fallback to CUDA if available, otherwise CPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Move data to the appropriate device
        real = real.to(device)
        
        # Scale images to [-1, 1] range if they're in [0, 1]
        if real.min() >= 0 and real.max() <= 1:
            real = real * 2 - 1
        
        # We use the model's loss function as a proxy for NLL
        with torch.no_grad():
            # Sample multiple random times for more accurate estimate
            losses = []
            for _ in range(10):  # Average over multiple time samples
                loss = self.model.loss_function(real)
                losses.append(loss.detach().cpu())
            
            # Take the mean loss
            mean_loss = torch.stack(losses).mean()
            
        # Convert to bits per dimension
        # BPD = loss (in nats) * (1/log(2)) / num_dimensions
        batch_size, channels, height, width = real.shape
        num_dims = channels * height * width
        bpd = mean_loss / np.log(2) / num_dims
        
        return bpd.item()
    
    @property
    def name(self) -> str:
        return "BitsPerDimension"