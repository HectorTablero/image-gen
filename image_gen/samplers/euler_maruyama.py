"""Euler-Maruyama sampler implementation for diffusion models.

This module provides an implementation of the Euler-Maruyama numerical method
for sampling from diffusion models by solving the associated stochastic
differential equation (SDE).
"""

import torch
from torch import Tensor
from typing import Callable, Optional, Tuple

from tqdm.autonotebook import tqdm

from .base import BaseSampler


class EulerMaruyama(BaseSampler):
    """Euler-Maruyama numerical sampler for diffusion models.

    This sampler implements the Euler-Maruyama numerical scheme for solving
    the stochastic differential equation (SDE) associated with the reverse
    diffusion process.
    """

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
        """Perform sampling using the Euler-Maruyama method.

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

        # Create linearly spaced timesteps from 1.0 to 1e-3
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
            t_for_score = t_batch

            # Handle NaN/Inf values to prevent numerical instability
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                if self.verbose:
                    print(
                        f"Warning: NaN or Inf values detected in x_t at step {i}"
                    )
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

            try:
                # Create a fresh detached copy for gradient computation
                x_t_detached = x_t.detach().clone()
                x_t_detached.requires_grad_(True)
                score = score_model(x_t_detached, t_for_score)
            except Exception as e:
                print(f"Error computing score at step {i}, t={t_curr}: {e}")
                score = torch.zeros_like(x_t)

            # Compute drift and diffusion terms for the SDE
            drift, diffusion = self.diffusion.backward_sde(
                x_t, t_batch, score, n_steps=n_steps
            )

            # Handle numerical stability for the diffusion term
            diffusion = torch.nan_to_num(diffusion, nan=1e-4)
            noise = torch.randn_like(x_t)

            # Update x_t using the Euler-Maruyama update rule
            x_t = (
                x_t +
                drift * (-dt) +
                diffusion * torch.sqrt(torch.abs(dt)) * noise
            )

            # Apply guidance if provided
            if guidance is not None:
                x_t = guidance(x_t, t_curr)

            # Clamp values to prevent extreme values
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # Call callback if provided and at the right frequency
            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t
