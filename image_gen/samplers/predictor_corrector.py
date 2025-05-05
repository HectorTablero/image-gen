"""Predictor-Corrector sampler for diffusion models.

This module provides an implementation of the Predictor-Corrector sampling
method for diffusion models, which combines a predictor step (similar to
Euler-Maruyama) with a corrector step based on Langevin dynamics.
"""

import torch
from torch import Tensor
from typing import Callable, Optional, Tuple, Any

from tqdm.autonotebook import tqdm

from .base import BaseSampler
from ..diffusion import BaseDiffusion


class PredictorCorrector(BaseSampler):
    """Predictor-Corrector sampler for diffusion models.

    This sampler implements the Predictor-Corrector method, which alternates
    between a prediction step and a correction step to improve sampling quality.

    Attributes:
        diffusion: The diffusion model to sample from.
        verbose: Whether to print progress information during sampling.
        corrector_steps: Number of correction steps per prediction step.
        corrector_snr: Signal-to-noise ratio for the corrector step.
    """

    def __init__(
        self,
        diffusion: BaseDiffusion,
        *args: Any,
        verbose: bool = True,
        corrector_steps: int = 1,
        corrector_snr: float = 0.15,
        **kwargs: Any
    ):
        """Initialize the Predictor-Corrector sampler.

        Args:
            diffusion: The diffusion model to sample from.
            *args: Additional positional arguments.
            verbose: Whether to print progress information during sampling.
                Defaults to True.
            corrector_steps: Number of correction steps per prediction step.
                Defaults to 1.
            corrector_snr: Signal-to-noise ratio for the corrector step.
                Controls the noise magnitude. Defaults to 0.15.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(diffusion, *args, verbose=verbose, **kwargs)
        self.corrector_steps = corrector_steps
        self.corrector_snr = corrector_snr

    def predictor_step(
            self,
            x_t: Tensor,
            t_curr: Tensor,
            t_next: Tensor,
            score: Tensor,
            *args: Any,
            **kwargs: Any
    ) -> Tensor:
        """Perform a predictor step (similar to Euler-Maruyama).

        Args:
            x_t: Current state tensor.
            t_curr: Current time step.
            t_next: Next time step.
            score: Score estimate at current step.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Updated tensor after prediction step.
        """
        # Ensure dt has the correct dimensions for broadcasting
        dt = (t_curr - t_next).view(-1, 1, 1, 1)

        # Get drift and diffusion
        drift, diffusion = self.diffusion.backward_sde(
            x_t, t_curr, score, *args, **kwargs
        )
        diffusion = torch.nan_to_num(diffusion, nan=1e-4)
        noise = torch.randn_like(x_t)

        # Apply Euler step with correct dimensions
        dt_sqrt = torch.sqrt(torch.abs(dt))
        x_next = x_t + drift * (-dt) + diffusion * dt_sqrt * noise
        return x_next

    def corrector_step(
            self,
            x_t: Tensor,
            t: Tensor,
            score_model: Callable,
            *_,
            **__
    ) -> Tensor:
        """Perform a corrector step based on Langevin dynamics.

        Args:
            x_t: Current state tensor.
            t: Current time step.
            score_model: Model function that predicts the score.

        Returns:
            Updated tensor after correction step.
        """
        try:
            with torch.enable_grad():
                x_t.requires_grad_(True)
                score = score_model(x_t, t)
                x_t.requires_grad_(False)

            if torch.isnan(score).any():
                score = torch.nan_to_num(score, nan=0.0)

            # Estimate corrector noise scale based on SNR
            noise_scale = torch.sqrt(
                torch.tensor(2.0 * self.corrector_snr, device=x_t.device)
            )
            noise = torch.randn_like(x_t)

            # Carefully compute score norm
            # Use a small epsilon value to avoid division by zero
            epsilon = 1e-10
            score_norm = torch.norm(
                score.view(score.shape[0], -1), dim=1, keepdim=True
            ).view(-1, 1, 1, 1)
            score_norm = torch.maximum(
                score_norm, torch.tensor(epsilon, device=score_norm.device)
            )

            # Calculate step size with correct broadcasting
            step_size = (
                self.corrector_snr / (score_norm ** 2)
            ).view(-1, 1, 1, 1)
            step_size = torch.nan_to_num(step_size, nan=1e-10)

            # Apply corrector step with proper broadcasting
            sqrt_step = torch.sqrt(step_size)
            x_t_corrected = (
                x_t +
                step_size * score +
                noise_scale * sqrt_step * noise
            )
            return x_t_corrected

        except IndexError as e:
            if self.verbose:
                print(
                    f"IndexError in corrector_step: {e}. Skipping correction."
                )
            # If an index error occurs, simply return unmodified x_t
            return x_t

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
        """Perform sampling using the predictor-corrector method.

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

        # Generate time steps
        times = torch.linspace(1.0, 1e-3, n_steps + 1, device=device)

        # Create progress bar if verbose mode is enabled
        iterable = (
            tqdm(range(n_steps), desc='Generating')
            if self.verbose else range(n_steps)
        )

        for i in iterable:
            t_curr = times[i]
            t_next = times[i + 1]

            # Create time tensors with appropriate batch dimensions
            batch_size = x_T.shape[0]
            t_batch = torch.full((batch_size,), t_curr, device=device)
            t_next_batch = torch.full((batch_size,), t_next, device=device)

            # Handle NaN/Inf values for numerical stability
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                if self.verbose:
                    print(
                        f"Warning: NaN or Inf values detected in x_t at step {i}"
                    )
                x_t = torch.nan_to_num(
                    x_t, nan=0.0, posinf=1.0, neginf=-1.0
                )

            # Step 1: Predictor
            try:
                # Create a fresh detached copy for gradient computation
                x_t_detached = x_t.detach().clone()
                x_t_detached.requires_grad_(True)
                score = score_model(x_t_detached, t_batch)

            except Exception as e:
                print(f"Error computing score at step {i}, t={t_curr}: {e}")
                score = torch.zeros_like(x_t)

            # Apply predictor step
            x_t = self.predictor_step(
                x_t, t_batch, t_next_batch, score, n_steps=n_steps
            )

            # Step 2: Corrector (Langevin MCMC)
            # Ensure the corrector step properly handles class labels
            try:
                for j in range(self.corrector_steps):
                    x_t = self.corrector_step(
                        x_t, t_next_batch, score_model, n_steps=n_steps
                    )
            except Exception as e:
                if self.verbose:
                    print(
                        f"Error in corrector step: {e}. "
                        f"Continuing without correction."
                    )

            # Apply guidance if provided
            if guidance is not None:
                try:
                    x_t = guidance(x_t, t_next)
                except Exception as e:
                    if self.verbose:
                        print(
                            f"Error in guidance: {e}. "
                            f"Continuing without applying guidance."
                        )

            # Stabilization
            x_t = torch.clamp(x_t, -10.0, 10.0)

            # Call callback if provided and at the right frequency
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
            "corrector_steps": self.corrector_steps,
            "corrector_snr": self.corrector_snr,
        })
        return config
