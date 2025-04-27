from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg
from .base import BaseMetric


class FrechetInceptionDistance(BaseMetric):
    """
    Fréchet Inception Distance (FID) for evaluating generative models.
    Lower values indicate better quality and diversity.
    """
    def __init__(self, device='cuda', dims=2048):
        """
        Args:
            device: Device to run the model on
            dims: Dimensionality of Inception features to use
        """
        self.device = device
        self.dims = dims
        self.model = None
        
    def _get_model(self):
        if self.model is None:
            self.model = inception_v3(pretrained=True, transform_input=False)
            self.model.fc = nn.Identity()  # Remove final FC layer to get features
            self.model.eval()
            self.model.to(self.device)
        return self.model
    
    def _get_activations(self, images: Tensor) -> np.ndarray:
        """Extract Inception features"""
        model = self._get_model()
        
        # Resize images to Inception input size if needed
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
        
        # Scale from [-1, 1] to [0, 1] range if needed
        if images.min() < 0:
            images = (images + 1) / 2
            
        # Ensure values are in [0, 1]
        images = torch.clamp(images, 0, 1)
        
        # Extract features
        with torch.no_grad():
            pred = model(images)
            
        return pred.detach().cpu().numpy()
    
    def _calculate_fid(self, real_activations: np.ndarray, gen_activations: np.ndarray) -> float:
        """Calculate FID score from activations"""
        # Calculate mean and covariance
        mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
        mu2, sigma2 = gen_activations.mean(axis=0), np.cov(gen_activations, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)
    
    def __call__(self, real: Tensor, generated: Tensor, batch_size: int = 32, *args, **kwargs) -> float:
        """
        Computes the FID score between real and generated images.
        
        Args:
            real: Tensor of real images (B, C, H, W)
            generated: Tensor of generated images (B, C, H, W)
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score (lower is better)
        """
        # Move to device
        real = real.to(self.device)
        generated = generated.to(self.device)
        
        # Process in batches to avoid memory issues
        real_activations = []
        gen_activations = []
        
        for i in range(0, real.shape[0], batch_size):
            real_batch = real[i:i + batch_size]
            real_activations.append(self._get_activations(real_batch))
            
        for i in range(0, generated.shape[0], batch_size):
            gen_batch = generated[i:i + batch_size]
            gen_activations.append(self._get_activations(gen_batch))
            
        real_activations = np.concatenate(real_activations, axis=0)
        gen_activations = np.concatenate(gen_activations, axis=0)
        
        return self._calculate_fid(real_activations, gen_activations)
    
    @property
    def name(self) -> str:
        return "FrechetInceptionDistance"