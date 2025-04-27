# import torch
# import math
# from abc import ABC, abstractmethod
# from torch import Tensor
# from typing import Optional
# from .base import BaseNoiseSchedule


# class CosineNoiseSchedule(BaseNoiseSchedule):
#     def __init__(self, beta_max: float = 100.0, s: float = 0.008):
#         """
#         Initialize the cosine noise schedule.
        
#         Args:
#             beta_max: Maximum value for beta at t=1 (default: 100.0)
#             s: Offset parameter to prevent singularity (default: 0.008)
#         """
#         self.beta_max = beta_max
#         self.s = s
        
#         # Precompute some values used for scaling
#         # Calculate the value at t=1 for proper scaling
#         self.cos_term_0 = math.cos(math.pi * self.s / (2 * (1 + self.s)))
#         self.cos_term_1 = math.cos(math.pi * (1 + self.s) / (2 * (1 + self.s)))
        
#         # Compute the scale factor to ensure we hit beta_max at t=1
#         # This is based on the alpha_bar ratio at t=1
#         alpha_bar_0 = (self.cos_term_0 / self.cos_term_0) ** 2  # = 1.0
#         alpha_bar_1 = (self.cos_term_1 / self.cos_term_0) ** 2
#         log_alpha_bar_ratio = math.log(alpha_bar_1 / alpha_bar_0)
#         self.scale_factor = beta_max / (-log_alpha_bar_ratio)
    
#     def alpha_bar(self, t: Tensor) -> Tensor:
#         """
#         Calculate alpha_bar(t) according to the cosine schedule formula.
        
#         Args:
#             t: Time steps in [0, 1]
            
#         Returns:
#             Alpha_bar values at each time step
#         """
#         if isinstance(t, (int, float)):
#             t = torch.tensor(float(t))
            
#         # Implementation of the alpha_bar equation from the paper
#         # We're careful to use the cos_term_0 value we precomputed
#         cos_term_t = torch.cos(math.pi * (t + self.s) / (2 * (1 + self.s)))
#         alpha_bar_t = (cos_term_t / self.cos_term_0) ** 2
        
#         # Ensure numerical stability
#         alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-10, max=1.0)
        
#         return alpha_bar_t
    
#     def __call__(self, t: Tensor, *args, **kwargs) -> Tensor:
#         """
#         Calculate beta(t) at time t.
        
#         Instead of directly computing the derivative (which can be unstable),
#         we use a discrete approximation by looking at the ratio of alpha_bar values.
        
#         Args:
#             t: Time steps in [0, 1]
            
#         Returns:
#             Beta values at each time step
#         """
#         if isinstance(t, (int, float)):
#             t = torch.tensor(float(t))
            
#         # Handle edge cases for stability
#         if torch.all(t <= 0):
#             return torch.zeros_like(t)
            
#         # Get step size (assuming uniform grid)
#         # We'll use a small step size for numerical differentiation
#         h = 0.001
            
#         # Calculate alpha_bar at t and t-h
#         alpha_bar_t = self.alpha_bar(t)
#         alpha_bar_t_prev = self.alpha_bar(torch.clamp(t - h, min=0.0))
        
#         # Calculate discrete approximation of -d/dt log(alpha_bar(t))
#         # This is equivalent to: -log(alpha_bar(t)/alpha_bar(t-h)) / h
#         log_ratio = torch.log(alpha_bar_t / alpha_bar_t_prev)
#         beta_t = -log_ratio / h
        
#         # Scale to ensure we reach beta_max at t=1
#         # We use the scaling factor computed during initialization
#         beta_t = beta_t * self.scale_factor
        
#         # Clamp to prevent any extremely large values or NaNs
#         beta_t = torch.clamp(beta_t, min=0.0, max=self.beta_max)
        
#         return beta_t
    
#     def integral_beta(self, t: Tensor, *args, **kwargs) -> Tensor:
#         """
#         Calculate the integral of beta from 0 to t.
        
#         In the cosine schedule, this is proportional to -log(alpha_bar(t))
        
#         Args:
#             t: Time steps in [0, 1]
            
#         Returns:
#             Integral of beta from 0 to t
#         """
#         alpha_bar_t = self.alpha_bar(t)
        
#         # The integral is proportional to -log(alpha_bar(t))
#         integral = -torch.log(alpha_bar_t)
        
#         # Scale by the same factor to be consistent with the beta function
#         integral = integral * self.scale_factor
        
#         return integral
    
#     def config(self) -> dict:
#         """
#         Return the configuration of the noise schedule.
        
#         Returns:
#             Dictionary containing the configuration parameters
#         """
#         return {
#             "beta_max": self.beta_max,
#             "s": self.s,
#             "scale_factor": self.scale_factor
#         }

from .base import BaseNoiseSchedule
import torch
from torch import Tensor
import math

class CosineNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, *args, s: float = 0.008, beta_min: float = 1e-4, beta_max: float = 20., **kwargs):
        """
        Cosine noise schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'
        
        Args:
            s: Small offset to prevent alpha_bar_t from being too small near t=1
            min_beta: Minimum noise level for numerical stability
            max_beta: Maximum noise level for numerical stability
        """
        self.s = s
        self.min_beta = beta_min
        self.max_beta = beta_max
        
    def alpha_bar(self, t: Tensor) -> Tensor:
        """Compute alpha_bar(t) = cos^2((t/T + s)/(1 + s) * π/2)"""
        return torch.cos((t + self.s) / (1.0 + self.s) * math.pi * 0.5) ** 2
    
    def __call__(self, t: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute beta(t) at timestep t
        
        For cosine schedule, we can compute beta(t) from the derivative of alpha_bar(t):
        beta(t) = -d(log(alpha_bar))/dt = -d(alpha_bar)/dt / alpha_bar
        """
        # Compute f(t) = (t + s)/(1 + s) * π/2
        f_t = (t + self.s) / (1.0 + self.s) * math.pi * 0.5
        
        # Compute d(alpha_bar)/dt = d(cos^2(f(t)))/dt
        # = 2 * cos(f(t)) * (-sin(f(t))) * d(f(t))/dt
        # = -π * sin(f(t)) * cos(f(t)) / (1 + s)
        dalpha_bar_dt = -math.pi * torch.sin(f_t) * torch.cos(f_t) / (1.0 + self.s)
        
        # Compute alpha_bar(t)
        alpha_bar_t = self.alpha_bar(t)
        
        # Compute beta(t) = -d(log(alpha_bar))/dt = -d(alpha_bar)/dt / alpha_bar
        beta_t = -dalpha_bar_dt / torch.clamp(alpha_bar_t, min=1e-8)
        
        # Ensure numerical stability
        beta_t = torch.clamp(beta_t, min=self.min_beta, max=self.max_beta)
        
        return beta_t
    
    def integral_beta(self, t: Tensor, *args, **kwargs) -> Tensor:
        """
        Compute the integral of beta from 0 to t.
        
        For cosine schedule, this is -log(alpha_bar(t))
        """
        alpha_bar_t = self.alpha_bar(t)
        return -torch.log(torch.clamp(alpha_bar_t, min=1e-8))
    
    def config(self) -> dict:
        return {
            "s": self.s,
            "min_beta": self.min_beta,
            "max_beta": self.max_beta
        }