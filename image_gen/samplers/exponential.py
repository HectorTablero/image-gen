# from .base import BaseSampler
# from typing import Optional
# from torch import Tensor
# import torch


# class ExponentialIntegrator(BaseSampler):
#     def __init__(self, diffusion, num_steps: Optional[int] = None):
#         super().__init__(diffusion, num_steps)
#         self._precompute_coefficients()

#     def _precompute_coefficients(self):
#         ts = self.timesteps(reverse=False)
#         beta_integrals = torch.stack(
#             [self.diffusion.beta_integral(t) for t in ts])
#         self.exp_terms = torch.exp(-0.5 * beta_integrals * self.dt)

#     def step(self, x: Tensor, t: Tensor, score: Tensor) -> Tensor:
#         idx = (t * self.num_steps).long()
#         drift, diffusion = self.diffusion.forward_sde(x, t)
#         noise_term = diffusion * \
#             torch.sqrt(1 - self.exp_terms[idx]**2) * torch.randn_like(x)
#         return self.exp_terms[idx] * (x + diffusion**2 * score * self.dt) + noise_term

###########################################################################################################

from .base import BaseSampler
import torch
from torch import Tensor
from typing import Callable, Optional
from tqdm.autonotebook import tqdm


class ExponentialIntegrator(BaseSampler):
    def __call__(
        self,
        x_T: Tensor,
        score_model: Callable,
        n_steps: int = 500,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Tensor, int], None]] = None,
        callback_frequency: int = 50,
        guidance: Optional[Callable[[Tensor, Tensor], Tensor]] = None
    ) -> Tensor:

        if seed is not None:
            torch.manual_seed(seed)

        device = x_T.device
        x_t = x_T.clone()

        times = torch.linspace(1.0, 1e-3, n_steps+1, device=device)
        dt = times[0] - times[1]

        iterable = tqdm(
            range(n_steps), desc='Generating') if self.verbose else range(n_steps)

        for i in iterable:
            t_curr = times[i]

            t_batch = torch.full((x_T.shape[0],), t_curr, device=device)

            t_for_score = t_batch

            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                if self.verbose:
                    print(
                        f"Warning: NaN or Inf values detected in x_t at step {i}")
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)

            try:
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    score = score_model(x_t, t_for_score)
                    x_t.requires_grad_(False)

                if torch.isnan(score).any():
                    if self.verbose:
                        print(
                            f"Warning: NaN values in score at step {i}, t={t_curr}")
                    score = torch.nan_to_num(score, nan=0.0)
            except Exception as e:
                print(f"Error computing score at step {i}, t={t_curr}: {e}")
                score = torch.zeros_like(x_t)
                x_t.requires_grad_(False)

            drift, diffusion = self.diffusion.backward_sde(x_t, t_batch, score)
            diffusion = torch.nan_to_num(diffusion, nan=1e-4)
            noise = torch.randn_like(x_t)

            # Exponential integration scheme instead of linear Euler-Maruyama
            # For systems with drift terms that are linear in x (like VP diffusion),
            # this provides a more accurate and stable integration
            exp_dt = drift * (-dt)
            # Prevent numerical instability
            exp_dt = torch.clamp(exp_dt, min=-10.0, max=10.0)
            exp_term = torch.exp(exp_dt)

            # Apply exponential update
            x_t = x_t * exp_term + diffusion * \
                torch.sqrt(torch.abs(dt)) * noise

            if guidance is not None:
                x_t = guidance(x_t, t_curr)

            x_t = torch.clamp(x_t, -10.0, 10.0)

            # if self.verbose and (i % callback_frequency == 0 or torch.isnan(x_t).any()):
            #     print(
            #         f"Step {i}: t={t_curr:.3f}, mean={x_t.mean().item():.3f}, std={x_t.std().item():.3f}")

            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t
