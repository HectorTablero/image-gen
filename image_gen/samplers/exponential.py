"""Exponential integrator sampler for diffusion models.

This module provides an implementation of the exponential integrator method
for sampling from diffusion models, which can be more stable than simpler
numerical integration schemes.
"""

import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, Any

from tqdm.autonotebook import tqdm

from .base import BaseSampler
from ..diffusion import BaseDiffusion


class ExponentialIntegrator(BaseSampler):
    """Exponential integrator for diffusion process sampling.

    This sampler implements an exponential integration scheme for solving
    the stochastic differential equation associated with the reverse
    diffusion process. It can provide better stability properties than simpler
    methods like Euler-Maruyama.

    Attributes:
        diffusion: The diffusion model to sample from.
        verbose: Whether to print progress information during sampling.
        lambda_param: The stabilization parameter for the exponential scheme.
    """

    def __init__(
        self,
        diffusion: BaseDiffusion,
        *args: Any,
        lambda_param: float = 1.0,
        verbose: bool = True,
        **kwargs: Any
    ):
        """Initialize the exponential integrator sampler.

        Args:
            diffusion: The diffusion model to sample from.
            *args: Additional positional arguments.
            lambda_param: The lambda parameter for the exponential integration.
                Defaults to 1.0.
            verbose: Whether to print progress information during sampling.
                Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(diffusion, *args, verbose=verbose, **kwargs)
        self.lambda_param = lambda_param

    def __call__(
        self,
        x_T: Tensor,
        score_model: Callable,
        *_,
        n_steps: int = 500,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Tensor, int], None]] = None,
        callback_frequency: int = 50,
        guidance: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        **__
    ) -> Tensor:
        """Perform sampling using the exponential integrator method.

        Args:
            x_T: The initial noise tensor to start sampling from.
            score_model: The score model function that predicts the score.
            n_steps: Number of sampling steps. Defaults to 500.
            seed: Random seed for reproducibility. Defaults to None.
            callback: Optional function called during sampling to monitor 
                progress. It takes the current sample and step number as inputs.
                Defaults to None.
            callback_frequency: How often to call the callback function.
                Defaults to 50.
            guidance: Optional guidance function for conditional sampling.
                Defaults to None.

        Returns:
            A tuple containing the final sample tensor and the final sample
            tensor again (for compatibility with the base class interface).
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = x_T.device
        x_t = x_T.clone()

        # Generate time steps from 1.0 to 1e-3
        times = torch.linspace(1.0, 1e-3, n_steps + 1, device=device)
        dt = times[0] - times[1]

        # Create progress bar if verbose mode is enabled
        iterable = (
            tqdm(range(n_steps), desc='Generating')
            if self.verbose else range(n_steps)
        )

        for i in iterable:
            t_curr = times[i]
            t_batch = torch.full((x_T.shape[0],), t_curr, device=device)

            # Handle NaN/Inf values in x_t for numerical stability
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                x_t = torch.nan_to_num(
                    x_t, nan=0.0, posinf=1.0, neginf=-1.0
                )

            try:
                # Create a fresh detached copy for gradient computation
                x_t_detached = x_t.detach().clone()
                x_t_detached.requires_grad_(True)
                score = score_model(x_t_detached, t_batch)

            except Exception as e:
                print(f"Error computing score at step {i}, t={t_curr}: {e}")
                score = torch.zeros_like(x_t)

            # Get drift and diffusion from the backward SDE
            drift, diffusion = self.diffusion.backward_sde(
                x_t, t_batch, score, n_steps=n_steps
            )
            # Diffusion coefficient for the exponential formula
            g = diffusion

            # Compute exponential term for the integrator
            exponential_term = torch.exp(self.lambda_param * dt)

            # Compute the second term in the exponential integration formula
            second_term = (
                (g**2 / (2 * self.lambda_param)) *
                (torch.exp(2 * self.lambda_param * dt) - 1) *
                score
            )

            # Add noise term (stochastic component)
            noise = torch.randn_like(x_t)
            noise_term = g * torch.sqrt(torch.abs(dt)) * noise

            # Update x_t using the exponential integrator step with noise
            x_t = x_t * exponential_term + second_term + noise_term

            # Apply guidance if provided
            if guidance is not None:
                x_t = guidance(x_t, t_curr)

            # Clamp values to prevent explosion
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # Invoke callback if needed
            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t

    def config(self) -> dict:
        """Return the configuration of the sampler.

        Returns:
            A dictionary with the sampler's configuration parameters.
        """
        config = super().config()
        config.update({
            "lambda_param": self.lambda_param
        })
        return config
