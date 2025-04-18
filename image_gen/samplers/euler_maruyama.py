from .base import BaseSampler
import torch
from torch import Tensor
from typing import Callable, Optional
from tqdm.autonotebook import tqdm


class EulerMaruyama(BaseSampler):
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

            x_t = x_t + drift * (-dt) + diffusion * \
                torch.sqrt(torch.abs(dt)) * noise

            if guidance is not None:
                x_t = guidance(x_t, t_curr)

            x_t = torch.clamp(x_t, -10.0, 10.0)

            if self.verbose and (i % callback_frequency == 0 or torch.isnan(x_t).any()):
                print(
                    f"Step {i}: t={t_curr:.3f}, mean={x_t.mean().item():.3f}, std={x_t.std().item():.3f}")

            if callback and i % callback_frequency == 0:
                callback(x_t.detach().clone(), i)

        return x_t
