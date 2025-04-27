from .base import BaseSampler
import torch
from torch import Tensor
from typing import Callable, Optional
from tqdm.autonotebook import tqdm


class ExponentialIntegrator(BaseSampler):
    def __init__(self, diffusion, *args, lambda_param: float = 1.0, verbose: bool = True, **kwargs):
        self.diffusion = diffusion
        self.verbose = verbose
        self.lambda_param = lambda_param

    def __call__(
        self,
        x_T: Tensor,
        score_model: Callable,
        *args,
        n_steps: int = 500,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Tensor, int], None]] = None,
        callback_frequency: int = 50,
        guidance: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        **kwargs
    ) -> Tensor:

        if seed is not None:
            torch.manual_seed(seed)

        device = x_T.device
        x_t = x_T.clone()

        # Generate time steps from 1.0 to 1e-3
        times = torch.linspace(1.0, 1e-3, n_steps + 1, device=device)
        dt = times[0] - times[1]

        iterable = tqdm(
            range(n_steps), desc='Generating') if self.verbose else range(n_steps)

        for i in iterable:
            t_curr = times[i]
            t_batch = torch.full((x_T.shape[0],), t_curr, device=device)

            # Handle NaN/Inf values in x_t
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

            # Compute score using the score model
            try:
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    score = score_model(x_t, t_batch)
                    x_t.requires_grad_(False)

                if torch.isnan(score).any():
                    score = torch.nan_to_num(score, nan=0.0)
            except Exception as e:
                print(f"Error computing score at step {i}, t={t_curr}: {e}")
                score = torch.zeros_like(x_t)
                x_t.requires_grad_(False)

            # Get drift and diffusion from the backward SDE (g is diffusion coefficient)
            drift, diffusion = self.diffusion.backward_sde(
                x_t, t_batch, score, n_steps=n_steps)
            g = diffusion  # Assumes diffusion is the same as g in the formula

            # Compute exponential term
            exponential_term = torch.exp(self.lambda_param * dt)

            # Compute the second term in the formula
            second_term = (g**2 / (2 * self.lambda_param)) * \
                (torch.exp(2 * self.lambda_param * dt) - 1) * score

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
