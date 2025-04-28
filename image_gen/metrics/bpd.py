from .base import BaseMetric
from torch import Tensor
import numpy as np


class BitsPerDimension(BaseMetric):
    """
    Bits per dimension (BPD) metric for evaluating density models.
    Lower values indicate better models.
    """
    def __init__(self, diffusion_model=None):
        """
        Args:
            diffusion_model: Optional diffusion model with a method to compute negative log-likelihood
        """
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
            
        # Assuming diffusion_model has a method to compute negative log-likelihood
        # This needs to be adapted to your specific diffusion implementation
        nll = self.diffusion_model.compute_nll(real)
        
        # Convert to bits per dimension
        # BPD = NLL (in nats) * (1/log(2)) / num_dimensions
        batch_size, channels, height, width = real.shape
        num_dims = channels * height * width
        bpd = nll / np.log(2) / num_dims
        
        return bpd.mean().item()
    
    @property
    def name(self) -> str:
        return "BitsPerDimension"